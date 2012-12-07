try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='ufuncexpr',
    version='0.5',
    author='Alberto Valverde Gonzalez',
    url='http://github.com/meteogrid/ufuncexpr',
    author_email='alberto@meteogrid.com',
    description="Creates ufuncs from expressions using numba",
    license="BSD",
    py_modules=['ufuncexpr'],
    install_requires=['numba'],
    test_suite="ufuncexpr",
    )
    


