import sys
import os
from subprocess import check_call, Popen, PIPE
import numpy
from setuptools import setup, Extension

RUNTIME_C = os.path.join(os.path.dirname(__file__), 'ufuncexpr', 'runtime.c')
RUNTIME_LL = RUNTIME_C[:-2] + '.ll'

try:
    from Cython.Distutils import build_ext
    extra = dict(cmdclass=dict(build_ext=build_ext))
except ImportError:
    # Assume .c file is generated
    extra = {}


def compile_runtime():
    py_includes = Popen(['python-config','--includes'], stdout=PIPE)\
                  .communicate()[0].split()
    numpy_includes = [numpy.get_include()]
    check_call(['clang'] + py_includes + numpy_includes + [
        '-S', '-emit-llvm', '-o', RUNTIME_LL, RUNTIME_C])

if 'develop' in sys.argv or not os.path.exists(RUNTIME_LL):
    compile_runtime()

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
    tests_require=['nose'],
    test_suite="nose.collector",
    ext_modules = [
        Extension('ufuncexpr._ufuncwrapper',
                  ['ufuncexpr/_ufuncwrapper.pyx'],
                  include_dirs=[numpy.get_include()])
        ],
    **extra
    )
    


