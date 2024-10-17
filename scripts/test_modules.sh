#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
python -m pytest -s --cov=collab2/ --cov=tests --cov-report=term-missing ${@-} --cov-report=html:tests/coverage
