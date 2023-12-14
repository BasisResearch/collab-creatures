#!/bin/bash


#TODO modify when notebooks land
INCLUDED_NOTEBOOKS="docs/foraging/"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
