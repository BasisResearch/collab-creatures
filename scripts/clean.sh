#!/bin/bash
set -euxo pipefail

isort --profile="black" collab/ collab2/ tests/
black collab/ collab2/ tests/ docs/
autoflake --remove-all-unused-imports --in-place --recursive ./collab ./collab2 ./tests
nbqa --nbqa-shell isort docs/
nbqa --nbqa-shell autoflake  --nbqa-shell --remove-all-unused-imports --recursive --in-place docs/  

