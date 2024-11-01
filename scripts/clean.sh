#!/bin/bash
set -euxo pipefail

isort --profile="black" collab/ tests/
autoflake --remove-all-unused-imports --in-place --recursive ./collab  ./tests
nbqa --nbqa-shell isort --profile="black" docs/
nbqa --nbqa-shell autoflake  --nbqa-shell --remove-all-unused-imports --recursive --in-place docs/  
black collab/ tests/ docs/
