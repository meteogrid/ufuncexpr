from numba import jit
from numba.codegen.llvmcontext import LLVMContextManager
from .builder import make_ufunc, make_gufunc

__vectorizers__ = dict(
    cpu = make_ufunc,
    gpu = make_gufunc,
    )

def vectorize(signatures, **kw):
    backend = kw.pop('backend', 'cpu')
    builder = __vectorizers__[backend]
    kw['nopython'] = True
    engine = LLVMContextManager().execution_engine
    def wrap(f):
        llvm_functions = [jit(s, **kw)(f).lfunc for s in signatures]
        return builder(f.__name__, llvm_functions, engine=engine,
                       doc=f.__doc__ or '')
    return wrap

