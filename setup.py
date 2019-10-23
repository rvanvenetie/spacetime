# Run as `python3 setup.py build_ext --inplace`
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    [
        'Python/datastructures/*[!test].py',
        'Python/space/*[!test].py',
        'Python/spacetime/*[!test].py',
        'Python/time/*[!test].py',
        'Python/applications/*[!test].py',
    ],
    compiler_directives={'language_level': "3"},
))
