
from setuptools import find_packages, setup

VERSION = "0.1.0"



# examples/tutorials/notebooks
EXTRAS_REQUIRE = [
    "jupyter",
    "graphviz",
    "matplotlib",
    "seaborn",
    "kaleido",
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
        "chirho @ git+https://github.com/BasisResearch/chirho.git#egg=chirho",
        "pyro-ppl>=1.8.6", "pandas==2.2.3", "plotly", "plotly.express", 
        "torch", "scipy", "scikit-learn",
        "matplotlib>=3.8.2", "dill", "torchdiffeq",
        "numpy==2.1.2",
    ],
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE
        + [
            "pytest==7.4.3",
            "pytest-cov",
            "pytest-xdist",
            "mypy",
            "black==24.8.0",
            "flake8",
            "isort==5.13.2",
            "nbval",
            "nbqa==1.9.0",
            "autoflake",
        ],
    },
    python_requires=">=3.9",
    keywords="animal behavior, bayesian modeling, probabilistic programming, dynamical systems",
    license="Apache 2.0",
)