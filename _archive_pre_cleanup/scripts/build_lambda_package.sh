#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf api/package
pip install -r api/requirements-lambda.txt -t api/package \
  --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312 \
  --only-binary=:all: --quiet
AWS_DEFAULT_REGION=us-east-1 PYTHONPATH=api/package:api python -c "import app; print('Lambda package imports OK')"
