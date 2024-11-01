#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports collab/  tests/
isort --check --profile="black" --diff collab/ tests/
black --check collab/ tests/ docs/
flake8 collab/ tests/
nbqa --nbqa-shell autoflake --nbqa-shell --recursive --check docs/
