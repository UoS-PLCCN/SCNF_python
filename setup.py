from pathlib import Path
from typing import List

from setuptools import find_packages, setup

cwd = Path(__file__).parent


def read(fname: str) -> List[str]:
    with open(cwd / fname) as f:
        content = f.read()
    return content.splitlines()


setup(
    name="scnfn",
    packages=find_packages(),
    install_requires=[
        "sympy",
        "scipy",
        "numpy",
    ],
    extras_require={
        "dev": ["flake8", "isort", "pytest", "pytest-cov"],
        "demo": ["gym-PBN"],
    },
)
