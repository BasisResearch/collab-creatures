

Collaborative Animal Behavior Modeling with Collab
===================================================

Collab is a package that supports the use of Bayesian and causal inference
in the study of animal collaborative behavior. The current version focuses on foraging 
animals and the probabilistic identification of foraging strategies. All the functionalities
are illustrated in jupyter notebooks (listed below). Using the package you can:

1) 




Installation
------------

**Basic Setup:**

```sh

    git clone git@github.com:BasisResearch/collaborative-intelligence.git
    cd collaborative-intelligence
    git checkout main
    pip install .
```

**Dev Setup:**

To install dev dependencies for Collab, run the following command:

```sh
pip install -e ".[test]"
```

** Contributing: **

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally

```sh
make format            # runs black and isort
make lint              # linting
make tests             # notebook and unit tests
```

## Getting started


TODO add getting started guide