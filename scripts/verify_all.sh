#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🚀 STARTING FULL MLOPS STACK VERIFICATION"
echo "=================================================="

echo "1. Running complete Pytest suite..."
pytest -q
echo "✅ All unit tests passed."

echo "2. Building Lambda deployment assets..."
./scripts/build_assets.sh
echo "✅ Assets built successfully."

echo "3. Running artifact junk-exclusion validation..."
if [ -f "scripts/validate_junk.sh" ]; then
    bash scripts/validate_junk.sh
else
    echo "⚠️ scripts/validate_junk.sh not found, skipping."
fi

echo "4. Synthesizing AWS CDK Stack..."
cd cdk
rm -rf cdk.out
cdk synth --quiet
echo "✅ CDK synthesis successful."
cd ..

echo "=================================================="
echo "🎉 ALL MLOPS & DEVOPS CHECKS COMPLETED SUCCESSFULLY!"
echo "=================================================="
