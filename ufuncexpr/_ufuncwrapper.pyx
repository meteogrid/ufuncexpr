import inspect
import sys
import numpy as np
import numba

from llvm import core as lc, passes as lp
from llvm.core import Constant, Type, Function, Builder

from numpy cimport PyUFuncGenericFunction, PyUFunc_FromFuncAndData, import_ufunc
from libc.stdlib cimport malloc, free, realloc


import_ufunc()

__all__ = ['UFuncWrapper']

cdef class UFuncWrapper:

    cdef readonly tuple args
    cdef readonly object func
    cdef readonly object py_func
    cdef readonly tuple llvm_functions
    cdef PyUFuncGenericFunction *functions
    cdef char *types

    cdef readonly bytes __name__
    cdef bytes doc
    cdef int arity

    cdef object _seen_types
    cdef object _jit

    property __doc__:
        def __get__(self):
            return self.doc

    def __init__(self, name, doc, arity, _jit=numba.jit, _py_func=None):
        self.__name__ = name
        self.doc = doc
        self.arity = arity
        self.llvm_functions = ()
        self.types = NULL
        self.functions = NULL
        self.func = None
        self.py_func = _py_func
        self._seen_types = set()
        self._jit = _jit

    @classmethod
    def decorate(cls, func):
        arity = len(inspect.getargspec(func).args)
        return cls(func.func_name, func.__doc__ or '', arity, _py_func=func)

    def add_specialization(self, llvm_function):
        assert len(llvm_function.args)==self.arity, "Bad arity"
        return_type = llvm_function.type.pointee.return_type
        execution_engine = llvm_function.module.owner
        lfunc = make_loop_func_from_llvm_func(llvm_function)
        # keep a ref to prevent it from being gced
        self.llvm_functions += (lfunc,)

        cdef int nfuncs = len(self.llvm_functions)

        self.functions = <PyUFuncGenericFunction*>realloc(
            self.functions, sizeof(PyUFuncGenericFunction)*nfuncs)
        cdef long ptr = execution_engine.get_pointer_to_function(lfunc)
        self.functions[nfuncs-1] = <PyUFuncGenericFunction>ptr

        self.types = <char*>realloc(
            self.types, sizeof(char)*nfuncs*(self.arity+1))
        types = [a.type for a in llvm_function.args] + [return_type]
        cdef int t_offset = (nfuncs-1)*(self.arity+1)
        cdef int i
        for i in range(self.arity+1):
            self.types[t_offset+i] = <char>_llvm_ty_to_dtype(types[i]).num

        self.func = PyUFunc_FromFuncAndData(
            self.functions,
            <void**>self.functions, # dummy data, NULL segfaults although docs say it is allowed
            self.types,
            nfuncs,  #ntypes
            self.arity,
            1,
            1 if self.arity==2 else -1,
            self.__name__,
            self.doc,   #__doc__
            0)

    def __dealloc__(self):
        if self.types is not NULL:
            free(self.types)
        if self.functions is not NULL:
            free(self.functions)

    def _maybe_create_specialization_for_args(self, args):
        if self.py_func is None:
            return
        args = tuple((np.array(a) if not isinstance(a, np.ndarray) else a)
                     for a in args)
        argtypes = tuple(a.dtype for a in args)
        self._maybe_create_specialization_for_types(argtypes)

    def _maybe_create_specialization_for_types(self, argtypes):
        if argtypes not in self._seen_types:
            self._seen_types.add(argtypes)
            argtypes = [_dtype_to_numba(a) for a in argtypes]
            jitted = self._jit(argtypes=argtypes, nopython=True)(self.py_func)
            self.add_specialization(jitted.lfunc)

    def __call__(self, *args, **kw):
        self._maybe_create_specialization_for_args(args)
        if self.func is not None:
            return self.func(*args, **kw)

    def _maybe_create_specialization_for_reducer_args(self, args):
        a = args[0]
        dtype = np.array(a).dtype if not isinstance(a, np.ndarray) else a.dtype
        self._maybe_create_specialization_for_types((dtype,dtype))

    def reduce(self, *args, **kw):
        self._maybe_create_specialization_for_reducer_args(args)
        if self.func is not None:
            return self.func.reduce(*args, **kw)

    def reduceAt(self, *args, **kw):
        self._maybe_create_specialization_for_reducer_args(args)
        if self.func is not None:
            return self.func.reduceAt(*args, **kw)

    def accumulate(self, *args, **kw):
        self._maybe_create_specialization_for_reducer_args(args)
        if self.func is not None:
            return self.func.accumulate(*args, **kw)


_numpy_to_numba = {
    np.bool_: numba.bool_,
    np.int8: numba.int8,
    np.int16: numba.int16,
    np.int32: numba.int32,
    np.int64: numba.int64,
    np.float32: numba.float32,
    np.float64: numba.double,
}
_dtype_to_numba_map = dict((np.dtype(k),v) for k,v in _numpy_to_numba.items())

def _dtype_to_numba(dtype):
    return _dtype_to_numba_map[dtype]
        

_llvm_ty_str_to_numpy = {
    'i1'     : np.bool_,
    'i8'     : np.int8,
    'i16'    : np.int16,
    'i32'    : np.int32,
    'i64'    : np.int64,
    'float'  : np.float32,
    'double' : np.float64,
}

def _llvm_ty_to_numpy(ty):
    return _llvm_ty_str_to_numpy[str(ty)]

def _llvm_ty_to_dtype(ty):
    return np.dtype(_llvm_ty_to_numpy(ty))


is64bits = sys.maxint>2**31-1 
ssize_t = Type.int(64 if is64bits else 32)

# PyUFuncGenericFunction
PyUFuncGenericFunction_t = Type.function(Type.void(), [
    Type.pointer(Type.pointer(Type.int(8))), # char **data
    Type.pointer(ssize_t),                   # ssize_t *dimensions
    Type.pointer(ssize_t),                   # ssize_t *steps
    Type.pointer(Type.void()),               # void *data
    ])

def make_loop_func_from_llvm_func(func, optimize=True):
    """
    Generates IR for a function of type PyUFuncGenericFunction which
    calls `func` to compute each element. The generated code looks something
    like the following C code. N is the number of arguments. There is only
    one output variable implemented at the moment.

    void
    wrapper(char **args, ssize_t *dimensions, ssize_t *steps, void *data)
    {
        ssize_t i, n=dimensions[0];

        for (i=0; i<n; i++) {
            *((return_type*) (args[N] + steps[N]*i)) = wrapped_func(
                *((arg0_type*) (args[0] + steps[0]*i)),
                *((arg1_tye*) (args[1] + steps[1]*i)),
                ....
                *((argN_type*) (args[N-1] + steps[N-1]*i)));
        }
    }
    """
    module = func.module
    return_type = func.type.pointee.return_type
    name = '__PyUFuncGenericFunction_'+func.name
    n_args = len(func.args)
    nin = Constant.int(ssize_t, n_args)
    zero = Constant.int(ssize_t, 0)
    one = Constant.int(ssize_t, 1)
    nums = [Constant.int(ssize_t, i) for i in xrange(n_args)]
    ufunc = Function.new(module, PyUFuncGenericFunction_t, name)

    # Build it
    block = ufunc.append_basic_block('entry')
    builder = Builder.new(block)
    _args, _dimensions, _steps = ufunc.args[:3]
    _args.name = "args"; _dimensions.name='dimensions'; _steps.name='steps'

    def deref(ptr, indices, name='tmp', cast=None):
        addr = builder.gep(ptr, indices, name+'_addr')
        if cast is not None:
            addr = builder.bitcast(addr, cast)
        return builder.load(addr, name)

    # Load constants from arguments
    num_iterations = deref(_dimensions, [zero], 'idx')
    arg_bases = [deref(_args, [nums[i]], 'base') for i in xrange(n_args)]
    result_base = deref(_args, [nin], 'rbase')
    arg_steps = [deref(_steps, [nums[i]], 'step') for i in xrange(n_args)]
    result_step = deref(_steps, [nin], 'rstep')
    loop_header_block = ufunc.append_basic_block('loop-header')
    builder.branch(loop_header_block)

    # loop-header block
    builder.position_at_end(loop_header_block)
    phi = builder.phi(ssize_t, 'loop_var')
    phi.add_incoming(zero, block)

    end_cond = builder.icmp(lc.ICMP_SLT, phi, num_iterations)
    loop_block = ufunc.append_basic_block('loop')
    after_block = ufunc.append_basic_block('afterloop')
    builder.cbranch(end_cond, loop_block, after_block)

    # loop block
    builder.position_at_end(loop_block)
    Ptr = Type.pointer
    func_args = [deref(base, [builder.mul(step, phi)], cast=Ptr(arg.type))
                 for base, step, arg in zip(arg_bases, arg_steps, func.args)]
    result = builder.call(func, func_args)
    result_addr = builder.gep(result_base, [builder.mul(result_step, phi)])
    builder.store(result,
                  builder.bitcast(result_addr, Type.pointer(return_type)))

    # calculate next loop value and branch
    next_value = builder.add(phi, one, 'next')
    phi.add_incoming(next_value, loop_block)
    builder.branch(loop_header_block)

    # after-block
    builder.position_at_end(after_block)
    builder.ret_void()

    if optimize:
        func.add_attribute(lc.ATTR_ALWAYS_INLINE)
        try:
            _optimize_func(ufunc)
        finally:
            func.remove_attribute(lc.ATTR_ALWAYS_INLINE)
    
    return ufunc

def _optimize_func(lfunc, opt_level=3):
    pmb = lp.PassManagerBuilder.new()
    pmb.opt_level = opt_level
    pmb.vectorize = True
    fpm = lp.PassManager.new()
    fpm.add(lp.PASS_ALWAYS_INLINE)
    pmb.populate(fpm)
    fpm.run(lfunc.module)
