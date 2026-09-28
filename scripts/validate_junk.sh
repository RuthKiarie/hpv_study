#!/usr/bin/env bash
set -e
cd /mnt/c/Users/rwk/OneDrive/Desktop/hpv_study/repo

echo "### Backing up train.py"
cp sagemaker/train.py /tmp/train.bak

echo "### leakage mutation vs the dedicated test only"
sed -i 's/"heard_of_cervical_cancer", "parity_capped",/"heard_of_cervical_cancer", "parity_capped", "high_risk_flag",/' sagemaker/train.py
pytest -q tests/test_train.py 2>&1 | grep -E "^(FAILED|[0-9]+ (passed|failed))" || true
cp /tmp/train.bak sagemaker/train.py

echo
echo "### plant junk in api/ like your real folder has"
cd api
rm -rf deployment.zip response.json response2.json test_app_local.py dashboard.py dashboard_data.py lambda-api-policy.json __pycache__
echo x > deployment.zip
echo '{}' > response.json
echo '{}' > response2.json
echo x > test_app_local.py
echo x > dashboard.py
echo x > dashboard_data.py
echo '{}' > lambda-api-policy.json
mkdir -p __pycache__
echo x > __pycache__/a.pyc

echo "package files that share the excluded patterns (must SURVIVE):"
find package -name 'test_*' -o -name 'response*.json' -o -name 'dashboard*' | head -5 || true
cd ..

./scripts/build_assets.sh >/dev/null

cd cdk && rm -rf cdk.out && cdk synth --quiet 2>&1 | grep -v "feature flags" || true
echo "synth exit=$?"

# Find the Lambda asset folder (contains requirements-lambda.txt or handler app.py inside cdk.out)
ASSET=""
for d in cdk.out/asset.*; do
    if [ -d "$d" ] && [ -f "$d/app.py" ]; then
        ASSET="$d"
        break
    fi
done

echo "API asset dir: $ASSET"
if [ -n "$ASSET" ] && [ -d "$ASSET" ]; then
    echo "--- top level of the Lambda bundle:"
    ls -A "$ASSET"
    echo "--- junk that must be ABSENT:"
    for f in deployment.zip response.json test_app_local.py dashboard.py lambda-api-policy.json __pycache__ requirements-lambda.txt; do
        if [ -e "$ASSET/$f" ]; then
            echo "  PRESENT (bad): $f"
        else
            echo "  absent (good): $f"
        fi
    done
    if [ -d "$ASSET/package" ]; then
        echo "--- dependency files that must SURVIVE:"
        find "$ASSET/package" -name 'test_*' -o -name 'response*.json' | head -3 || true
    else
        echo "--- package directory not found in bundle root ---"
    fi
    echo "bundle size: $(du -sh "$ASSET" | cut -f1)"
else
    echo "Error: Asset directory not found!"
fi
