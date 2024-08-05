# setup.py
from setuptools import setup, find_packages

setup(
    name="py_correlations",
    version="0.3",
    packages=find_packages(),
    install_requires=['simpy>=1.12', 'scipy>=1.9.3']
    # Automatically find all packages
    # Other metadata (author, description, etc.) can be added here
)
