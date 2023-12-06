#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports collab/
isort --check --profile black --diff collab/ tests/
black --check collab/ tests/
flake8 collab/ tests/
