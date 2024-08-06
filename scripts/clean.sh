#!/bin/bash
set -euxo pipefail

isort --profile="black" collab/ collab2/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./collab ./collab2 ./tests
nbqa --nbqa-shell isort --profile="black" docs/
nbqa --nbqa-shell autoflake  --nbqa-shell --remove-all-unused-imports --recursive --in-place docs/  
black collab/ collab2/ tests/ docs/
