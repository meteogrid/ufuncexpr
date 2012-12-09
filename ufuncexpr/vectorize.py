from ._ufuncwrapper import UFuncWrapper
from .multiiter import MultiIterFunc

from numba import jit

def vectorize(signatures=None, backend='ufunc'):
    def wrap(f):
        if backend=='multiiter':
            uf = MultiIterFunc.decorate(f)
        else:
            uf = UFuncWrapper.decorate(f)
        if signatures:
            for s in signatures:
                jitted = jit(s, nopython=True)(f).lfunc
                uf.add_specialization(jitted)
        return uf
    return wrap
