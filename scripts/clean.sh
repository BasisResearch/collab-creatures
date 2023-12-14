#!/bin/bash
set -euxo pipefail

isort --profile black collab/ tests/
black collab/ tests/
nbqa black docs/
nbqa autoflake --remove-all-unused-imports --recursive --in-place docs/ 
nbqa isort docs/
