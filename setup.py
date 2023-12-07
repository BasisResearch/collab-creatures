import sys

from setuptools import find_packages, setup

VERSION = "0.1.0"



# examples/tutorials/notebooks
EXTRAS_REQUIRE = [
    "jupyter",
    "graphviz",
    "matplotlib",
    "pandas",
    "seaborn",
]

setup(
    name="collab",
    version=VERSION,
    description="Tools for animal behavior foraging modeling.",
    packages=find_packages(include=["collab", "collab.*"]),
    author="Basis",
    url="https://www.basis.ai/",
    project_urls={
    #     "Documentation": "",
        "Source": "https://github.com/BasisResearch/collaborative-intelligence",
    },
    install_requires=[
        "pyro-ppl>=1.8.5",
    ],
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE
        + [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "mypy",
            "black",
            "flake8",
            "isort",
            "nbval",
        ],
    },
    python_requires=">=3.10",
    keywords="animal behavior, bayesian modeling, probabilistic programming, dynamical systems",
    license="Apache 2.0",
)