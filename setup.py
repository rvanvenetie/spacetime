# Run as `python3 setup.py build_ext --inplace`
import os
from setuptools import setup
from Cython.Build import cythonize
from glob import glob

os.environ['CFLAGS'] = '-O3'
setup(ext_modules=cythonize(
    list(glob('Python/*/*py')),
    compiler_directives={'language_level': "3"},
))
