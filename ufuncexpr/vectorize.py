from numba import jit

from ._ufuncwrapper import UFuncWrapper
from .multiiter import MultiIterFunc

def vectorize(signatures=None, backend='multiiter'):
    def wrap(f):
        jitted = [jit(s)(f).lfunc for s in (signatures or [])]
        if backend=='multiiter':
            uf = MultiIterFunc.decorate(f)
        else:
            assert jitted
            uf = UFuncWrapper(f.func_name, f.__doc__ or '', len(jitted[0].args))
        for f in jitted:
            uf.add_specialization(f)
        return uf
    return wrap
