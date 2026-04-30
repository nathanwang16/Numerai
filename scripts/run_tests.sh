#!/usr/bin/env bash
# Run the full unit-test suite.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
pytest tests/ -v "$@"
