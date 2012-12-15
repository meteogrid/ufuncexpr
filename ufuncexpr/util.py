from ctypes import sizeof, c_void_p
import numpy as np
import numba

import llvm.passes as lp

def llvm_ty_to_dtype(ty):
    return np.dtype(_llvm_ty_to_numpy(ty))

def dtype_to_numba(dtype):
    return _dtype_to_numba_map[dtype]

def determine_pointer_size():
    return sizeof(c_void_p) * 8
        
def optimize_llvm_function(func, opt_level=3, inline_threshold=15000):
    pmb = lp.PassManagerBuilder.new()
    pmb.opt_level = opt_level
    pmb.vectorize = True
    pmb.use_inliner_with_threshold(inline_threshold)
    fpm = lp.PassManager.new()
    fpm.add(func.module.owner.target_data)
    pmb.populate(fpm)
    fpm.run(func.module)




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
