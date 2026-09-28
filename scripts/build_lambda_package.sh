#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf api/package
mkdir -p api/package

# Install dependencies cleanly, forcing binary wheels for pydantic-core
pip install --no-cache-dir \
  -r api/requirements-lambda.txt \
  -t api/package \
  --only-binary=:all: \
  --upgrade --quiet

# Verify imports
cd api && python3 -c "import sys; sys.path.insert(0, 'package'); import pydantic_core; import fastapi; import app; print('Lambda package imports OK')"
