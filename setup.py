import numpy
from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
    extra = dict(cmdclass=dict(build_ext=build_ext))
except ImportError:
    # Assume .c file is generated
    extra = {}

setup(
    name='ufuncexpr',
    version='0.5',
    author='Alberto Valverde Gonzalez',
    url='http://github.com/meteogrid/ufuncexpr',
    author_email='alberto@meteogrid.com',
    description="Creates ufuncs from expressions using numba",
    license="BSD",
    packages=['ufuncexpr'],
    install_requires=['numba'],
    test_suite="ufuncexpr",
    ext_modules = [
        Extension('ufuncexpr._ufuncwrapper',
                  ['ufuncexpr/_ufuncwrapper.pyx'],
                  include_dirs=[numpy.get_include()])
        ],
    **extra
    )
    


