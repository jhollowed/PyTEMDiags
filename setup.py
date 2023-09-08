# This is the main setup config file, which allowz local installaton of the package via pip.
# From the top-level directory (location of this file), simply run `pip install .`
# For more info, see the guidelines for minimal Python package structure at
# https://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import setup

setup(name='PyTEM',
      version='0.1',
      description='Package for preforming Transformed Eulerian Mean analysis on unstructured climate model datasets in Python',
      url='https://github.com/jhollowed/PyTEM',
      author='Joe Hollowed',
      author_email='hollowed@umich.edu',
      packages=['PyTEM'],
      zip_safe=False)
