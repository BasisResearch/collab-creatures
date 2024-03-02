.. image:: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test_notebooks.yml/badge.svg
   :alt: Test Notebooks Badge
   :target: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test_notebooks.yml

.. image:: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test.yml/badge.svg
   :alt: Test Badge
   :target: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test.yml

.. index-inclusion-marker

Collaborative Animal Behavior Modeling with Collab
===================================================


**Collab** is a package that supports the use of Bayesian and causal inference 
with  `Pyro <https://github.com/pyro-ppl/pyro>`_ and `ChiRho <https://github.com/BasisResearch/chirho>`_ 
in the study of animal collaborative behavior. The current version focuses on foraging 
animals and the probabilistic identification of foraging strategies. All the
functionalities are illustrated in Jupyter notebooks (listed further down).
Using the package you can:


1. **Simulate Artificial Data** of foraging animals (random walkers, foragers following only food trace, foragers following a leader, foragers communicating about the position of food).

2. **Use either the simulated or real-world data** with (ideally, empirically informed) hyperparameters to **expand the dataset** into one that at each frame assigns various predictor scores to space-time points per forager, such as the presence of visible food trace, appropriate proximity of other foragers, availability of communication, etc.

3. Use the expanded data to **profile the foraging strategy using Bayesian inference**.

4. **Compartmentalize Artificial or Real-World Animal Movement Data** in preparation for Bayesian dynamical systems inference.

5. **Build Your Own Dynamical Systems Model** of the compartmentalized data and use it within a Bayesian inferential workflow.


Installation
------------

**Basic Setup:**

.. code-block:: sh

   git clone git@github.com:BasisResearch/collaborative-intelligence.git
   cd collaborative-intelligence
   git checkout main
   pip install .


**Dev Setup:**

To install dev dependencies needed to contribute to Collab, run the following command:

.. code-block:: sh

    pip install -e ".[test]"

or 

.. code-block:: sh
  
    pip install -e .[test]


**Contributing:**

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally:

  .. code-block:: sh

     make format            # runs black and isort
     make lint              # linting
     make tests             # notebook and unit tests


Getting started and notebooks
------------------------------

All the notebooks are located in the `docs` (especially `docs/foraging`) folder. The following notebooks are available:


- `random-hungry-followers.ipynb` uses of the package to simulate data of foraging animals 
and to use it to profile the foraging strategy (random/food trace focus/followers) using Bayesian inference.

- `communicators_simulations.ipynb`  illustrates the use of the package to simulate data of foraging animals 
where the animals reveal the position of food to each other.

- `communicators_inference.ipynb` shows how to use our toolkit to profile  this foraging strategy 
using Bayesian inference.

- `central_park_birds_predictors.ipynb` illustrates how to use the package to expand a real world dataset
that includes the movement of foraging birds in Central Park, New York, into one that at each frame assigns various 
predictor scores to space-time points per forager.

- `central_park_birds_inference.ipynb` contains an example that involves using the expanded data to profile 
  the proximity to other animals preferences of ducs and sparrows using Bayesian inference.

-  `locust_approximate_pipeline.ipynb` goes through an analogous workflow with a real-world dataset of foraging locust,
related to `Information integration for decision-making in desert locusts <https://doi.org/10.1016/j.isci.2023.106388>`_ by 
Günzel, Oberhauser and Couzin-Fuchs.

- `locust_ds_data.ipynb` illustrates how to compartmentalize the locust data in preparation for 
- Bayesian dynamical systems inference.

- `locust_ds_class.ipynb` shows how to build a dynamical systems mode of the compartmentalized data and use it 
within the Bayesian inferential workflow.

- `locust_ds_inference.ipynb` shows how to build a dynamical systems model of the compartmentalized data and use
it within a Bayesian inferential workflow.

- `locust_ds_validate.ipynb` uses the class we defined to validate the dynamical systems model of the 
compartmentalized data.

- `locust_ds_interpret.ipynb` elaborates on a proper way to interpret the inference results of the dynamical 
systems model.
  

*Note*: The inference steps assume some familiarity with `Pyro <https://github.com/pyro-ppl/pyro>`_ and 
probabilistic programming. The `Pyro repository <https://github.com/pyro-ppl/pyro>`_ contains links 
to introductory Pyro tutorials.
