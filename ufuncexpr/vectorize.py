from numba import jit
from numba.codegen.llvmcontext import LLVMContextManager
from .builder import make_ufunc

def vectorize(signatures):
    def wrap(f):
        llvm_functions = [jit(s)(f).lfunc for s in signatures]
        engine = LLVMContextManager().execution_engine
        return make_ufunc(llvm_functions, engine, f.__name__, f.__doc__ or '')
    return wrap

