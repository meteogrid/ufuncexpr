from numba import decorators

from ._ufuncwrapper import UFuncWrapper

def vectorize(signatures):
    def wrap(f):
        jitted = [decorators.jit(s)(f).lfunc for s in signatures]
        uf = UFuncWrapper(f.func_name, f.__doc__ or '', len(jitted[0].args))
        for f in jitted:
            uf.add_specialization(f, decorators.context.llvm_ee)
        return uf
    return wrap
