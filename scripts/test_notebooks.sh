#!/bin/bash


#TODO modify when notebooks land
INCLUDED_NOTEBOOKS="docs/foraging/ docs/experimental/collab2_tests"

EXCLUDED_NOTEBOOK="docs/foraging/locust_ds_validate.ipynb"

CI=1 pytest -v --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS --ignore=$EXCLUDED_NOTEBOOK
