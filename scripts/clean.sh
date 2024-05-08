#!/bin/bash
set -euxo pipefail

isort --profile black collab/ tests/
black collab/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./collab ./tests
nbqa black docs/
nbqa autoflake  --nbqa-shell --remove-all-unused-imports --recursive --in-place docs/  
nbqa isort docs/
