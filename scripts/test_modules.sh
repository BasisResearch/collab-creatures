#!/bin/bash
set -euxo pipefail

python -m pytest -s --cov=collab/ --cov=tests --cov-report=term-missing ${@-} --cov-report=html:tests/coverage
