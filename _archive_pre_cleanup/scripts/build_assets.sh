#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
need() {
  if [ ! -f "$1" ]; then
    echo "ERROR: expected source file missing: $1" >&2
    exit 1
  fi
}
need glue/glue_etl_job.py
need sagemaker/processing_script.py
need sagemaker/train.py
mkdir -p deploy_assets/scripts deploy_assets/processing-code deploy_assets/training-code/stable
cp glue/glue_etl_job.py deploy_assets/scripts/
cp sagemaker/processing_script.py deploy_assets/processing-code/
tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner \
    -cf - -C sagemaker train.py | gzip -n > deploy_assets/training-code/stable/sourcedir.tar.gz
echo "deploy_assets built from source."
