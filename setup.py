# Run as `python3 setup.py build_ext --inplace`
import os
from glob import glob
from setuptools import setup
from Cython.Build import cythonize

os.environ['CFLAGS'] = '-O3'
setup(ext_modules=cythonize(
    [fn for fn in glob('Python/*/*py') if not '_test.py' in fn],
    compiler_directives={'language_level': "3"},
))
