from .ufunc import UFuncVectorizer

from numba import jit

def vectorize(signatures=None, backend='ufunc'):
    def wrap(f):
        uf = UFuncVectorizer.decorate(f)
        if signatures:
            for s in signatures:
                jitted = jit(s, nopython=True)(f).lfunc
                uf.add_specialization(jitted)
        return uf
    return wrap
