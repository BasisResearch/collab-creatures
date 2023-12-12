#!/bin/bash
set -euxo pipefail

isort --profile black collab/ tests/
black collab/ tests/
nbqa black docs/
