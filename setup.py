from pathlib import Path

from setuptools import find_packages, setup

cwd = Path(__file__).parent


def read(fname: str) -> list[str]:
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
        "networkx",  # PBN_env
        "matplotlib",  # PBN_env
    ],
    extras_require={
        "dev": ["flake8", "isort", "pytest", "pytest-cov"]
    }
)
