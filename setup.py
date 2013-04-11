import numpy
from setuptools import setup

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
    tests_require=['nose', 'unittest2'],
    test_suite="nose.collector"
    )
    


