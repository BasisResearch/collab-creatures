#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports collab/ collab2/ tests/
isort --check --profile="black" --diff collab/ collab2/ tests/
black --check collab/ collab2/ tests/ docs/
flake8 collab/ collab2/ tests/
nbqa --nbqa-shell autoflake --nbqa-shell --recursive --check docs/
