from itertools import chain
import inspect
import numpy as np
import numba

from .util import dtype_to_numba
from .builder import UFuncBuilder
from ._ufuncwrapper import UFuncWrapper


__all__ = ['UFuncVectorizer']

class UFuncVectorizer(object):

    def __init__(self, name, doc, arity, _jit=numba.jit, _py_func=None):
        self.__name__ = name
        self.doc = doc
        self.arity = arity
        self.llvm_functions = ()
        self.types = []
        self.func = None
        self.py_func = _py_func
        self._seen_types = set()
        self._jit = _jit

    @classmethod
    def decorate(cls, func):
        arity = len(inspect.getargspec(func).args)
        return cls(func.func_name, func.__doc__ or '', arity, _py_func=func)

    def add_specialization(self, llvm_function):
        builder = UFuncBuilder(llvm_function, optimization_level=3)
        assert builder.nin==self.arity and builder.nout==1, "Bad arity"
        lfunc = builder.ufunc
        self.llvm_functions += (lfunc,)
        self.types.append([d.num for  d in builder.dtypes])
        functions, types = self._sorted_functions_and_types()
        self.func = UFuncWrapper(functions, types,
            nin=builder.nin,
            nout=builder.nout,
            name=self.__name__,
            doc=self.doc
            )
    
    def _sorted_functions_and_types(self):
        l = zip(self.types, self.llvm_functions)
        l.sort()
        flat_types = list(chain(*(i[0] for i in l)))
        functions = [i[1] for i in l]
        return functions, flat_types


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
            argtypes = [dtype_to_numba(a) for a in argtypes]
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
