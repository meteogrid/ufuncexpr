from numpy cimport PyUFuncGenericFunction, PyUFunc_FromFuncAndData, import_ufunc
from libc.stdlib cimport malloc, free

import_ufunc()

__all__ = ['UFuncWrapper']

cdef class UFuncWrapper:

    cdef readonly object name
    cdef readonly object func
    cdef readonly tuple llvm_functions

    cdef PyUFuncGenericFunction *functions
    cdef char *types

    cdef object _doc

    def __init__(self, loop_functions, types, int nin, int nout=1,
                 name='test', doc=''):
        self.name = name
        self._doc = doc
        self.llvm_functions = tuple(loop_functions)
        cdef int nfuncs = len(loop_functions)
        assert len(types)==nfuncs*(nin+nout), repr((len(types),nfuncs,nin,nout))

        self.functions = <PyUFuncGenericFunction*>malloc(
            sizeof(PyUFuncGenericFunction)*nfuncs)
        gpf = loop_functions[0].module.owner.get_pointer_to_function
        cdef long ptr=0
        cdef int i=0
        for i in range(nfuncs):
            ptr = gpf(loop_functions[i])
            self.functions[i] = <PyUFuncGenericFunction>ptr

        self.types = <char*>malloc(sizeof(char)*len(types))
        
        i=0
        for dtype in types:
            self.types[i] = <char>(dtype)
            i+=1

        self.func = PyUFunc_FromFuncAndData(
            self.functions,
            <void**>self.functions, # dummy data, NULL segfaults although docs say it is allowed
            self.types,
            nfuncs,  #ntypes
            nin,
            nout,
            1 if (nin==2 and nout==1) else -1,
            <char*>self.name,
            <char*>self._doc,   #__doc__
            0)

    def __dealloc__(self):
        if self.types is not NULL:
            free(self.types)
        if self.functions is not NULL:
            free(self.functions)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __getattr__(self, k):
        return getattr(self.func, k)
     
    def __repr__(self):
        return '<UFuncWrapper %s>'%self.name
