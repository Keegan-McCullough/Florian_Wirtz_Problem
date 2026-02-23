# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fast_draw.pyx"),
    include_dirs=[np.get_include()] # Required to use NumPy arrays in Cython
)