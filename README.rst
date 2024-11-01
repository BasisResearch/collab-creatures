.. image:: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test_notebooks.yml/badge.svg
   :alt: Test Notebooks Badge
   :target: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test_notebooks.yml

.. image:: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test_modules.yml/badge.svg
   :alt: Test Modules Badge
   :target: https://github.com/BasisResearch/collaborative-intelligence/actions/workflows/test_modules.yml

.. index-inclusion-marker

Collaborative Animal Behavior Modeling with Collab
===================================================

`Collab <https://basisresearch.github.io/collab-creatures/getting_started.html>`_ is a package that supports the use of Bayesian and causal inference 
with  `Pyro <https://github.com/pyro-ppl/pyro>`_ and `ChiRho <https://github.com/BasisResearch/chirho>`_ 
in the study of animal collaborative behavior. It is developed by the Collaborative Intelligent Systems team at `Basis <https://www.basis.ai/>`_, with the goal of reasoning about a large class of collaborative behavior, spanning a broad range of species and contexts.  

The analyses in this package focus on foraging
animals and the probabilistic identification of foraging strategies, linking ideas from neuroscience, cognitive science, and statistics. When foraging, animals must assess where to move based on internal preferences (e.g., how they value food) and external cues (e.g., food location or presence of other animals). From a cognitive standpoint, animals may use an internal value function to decide on the optimal action at any moment. From a neuroscience perspective, this process involves the brain mapping environments and potential rewards, while statistical models predict movement patterns from available data. For instance, when a bird forages, the brain might generate a predictive map to estimate which locations are more or less valuable. Our analysis framework translates this into a statistical model that can then be used to predict the bird’s movement.

For more information, please see `our paper <https://www.nature.com/articles/s41598-024-71931-0>`_ :
   Urbaniak, R., Xie, M. & Mackevicius, E. Linking cognitive strategy, neural mechanism, and movement statistics in group foraging behaviors. Nature Scientific Reports 14, 21770 (2024). https://doi.org/10.1038/s41598-024-71931-0

An archival version of the repository from the time of the paper submission can be found `here <https://github.com/BasisResearch/collab-creatures/pull/137>`_ , in the `ru-staging-foraging-archive` branch. 

*This repository is a work in progress. We are continuously working on improving it, and applying it to new types of multi-agent behaviors. Please reach out if you're interested in collaborating or contributing.* 

All the functionalities are illustrated in Jupyter notebooks (listed further down).

Using the package you can:

1. **Simulate** different types of agents (random walkers, agents that value food, agents that value proximity to other agents, foragers communicating about the position of food).

2. Expand a (real or simulated) dataset by calculating a **set of predictor scores** at each location each agent may decide to move to. Scores correspond to a relevant feature agents may value, such as presense of a food trace, proximity to other foragers, and availability of information communicated from other agents.

3. **Profile the foraging strategy using Bayesian inference**, to assess how strongly each feature drives agents' decisions of where to move. 

4. **Compare** different species, behaviors, strategies, and tasks.

5. **Compartmentalize synthetic or real-world animal movement data** in preparation for Bayesian dynamical systems inference.

6. **Build your own dynamical systems model** of the compartmentalized data and use it within a Bayesian inferential workflow.


Installation
------------

**Basic Setup:**

.. code-block:: sh

   git clone git@github.com:BasisResearch/collaborative-intelligence.git
   cd collaborative-intelligence
   git checkout main
   pip install .


**Dev Setup:**
Make sure you have GraphViz, Pandoc, and LaTeX installed on your system. Then, to install dev dependencies needed to contribute to Collab, run the following command:

.. code-block:: sh

    pip install -e ".[dev]"

or 

.. code-block:: sh
  
    pip install -e .[dev]


**Contributing:**

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally:

  .. code-block:: sh

     make format            # runs black and isort
     make lint              # linting
     make test              # notebook and unit tests

To generate local version of the html documentation, and to serve on a local server run:

  .. code-block:: sh

     make gendoc
     make docserve 

Getting started and demo notebooks
----------------------------------

All the notebooks are located in the `docs` (mostly `docs/foraging`) folder. 


- `Random, hungry, followers <https://basisresearch.github.io/collab-creatures/foraging/random-hungry-followers/index.html>`_ simulates three types of foraging agents, and profiles their foraging strategies using Bayesian inference.

- `Central park birds <https://basisresearch.github.io/collab-creatures/foraging/central-park-birds/index.html>`_ illustrates using the package to infer foraging preferences from real-world datasets of birds foraging in Central Park, New York, NY. 

- `Communicators <https://basisresearch.github.io/collab-creatures/foraging/communicators/index.html>`_ simulates groups of foraging agents, some of which communicate about food locations, and uses Bayesian inference to infer the degree of communication.

- `Locust <https://basisresearch.github.io/collab-creatures/foraging/locust/index.html>`_ analyses a real-world dataset of foraging locusts, related to `Information integration for decision-making in desert locusts <https://doi.org/10.1016/j.isci.2023.106388>`_ by  Günzel, Oberhauser and Couzin-Fuchs.
  

*Note*: The inference steps assume some familiarity with `Pyro <https://github.com/pyro-ppl/pyro>`_ and 
probabilistic programming. The `Pyro repository <https://github.com/pyro-ppl/pyro>`_ contains links 
to introductory Pyro tutorials. The dynamical systems materials assume some familarity 
with `ChiRho <https://github.com/BasisResearch/chirho>`_ (see especially 
`this tutorial <https://basisresearch.github.io/chirho/dynamical_intro.html>`_).
