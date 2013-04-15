from numba import jit
from numba.codegen.llvmcontext import LLVMContextManager
from .builder import make_ufunc

def vectorize(signatures, **kw):
    kw['nopython'] = True # Using python inside UFuncs seems to segfault, perhaps GIL related? investigate...
    def wrap(f):
        llvm_functions = [jit(s, **kw)(f).lfunc for s in signatures]
        engine = LLVMContextManager().execution_engine
        return make_ufunc(llvm_functions, engine, f.__name__, f.__doc__ or '')
    return wrap

