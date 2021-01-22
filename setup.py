from __future__ import print_function
from setuptools import setup, find_packages

VERSION = "0.1-dev"  # NOTE: keep in sync with __version__ in mdtools.__init__.py

install_requires = [
    'MDAnalysis>=0.20.1', 
    'matplotlib>=2.0.0',
    'numpy>=1.10.4',
    'scipy>=0.17'
]

if __name__ == "__main__":

    setup(
        name = "mdtools",
        version = VERSION,
        license = "MIT",
        author = "Sean Friedowitz et al. (Qin Group, Stanford University)",
        description = "A collection of tools for running and post-processing molecular simulations.",
        long_description = open("README.md").read(),
        packages = find_packages(),
        install_requires = install_requires
    )