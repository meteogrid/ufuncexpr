from llvm.core import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C

from .util import llvm_ty_to_dtype, optimize_llvm_function
from ._ufuncwrapper import UFuncWrapper

UFUNC_PREFIX = '__PyUFuncGenericFunction_'

def make_ufunc(llvm_functions, engine, name=None, doc="ufuncexpr wrapped function"):
    functions = []
    tyslist = []
    nin, nout = None, None
    for llvm_function in llvm_functions:
        def_ =  MultipleReturnUFunc(llvm_function)
        if nin is None:
            nin, nout = def_.nin, def_.nout
        else:
            assert (nin, nout) == (def_.nin, def_.nout)
        tyslist.extend(d.num for  d in def_.dtypes)
        functions.append(def_(llvm_function.module))
    map(optimize_llvm_function, functions)
    ptrlist = [long(engine.get_pointer_to_function(f)) for f in functions]
    ufunc = UFuncWrapper(functions, ptrlist, tyslist, nin, nout,
                         name=name or llvm_functions[0].name, doc=doc)
    return ufunc

class MultipleReturnUFunc(CDefinition):
    '''a generic ufunc that wraps the workload
    '''
    _argtys_ = [
        ('args',       C.pointer(C.char_p), [ATTR_NO_ALIAS]),
        ('dimensions', C.pointer(C.intp), [ATTR_NO_ALIAS]),
        ('steps',      C.pointer(C.intp), [ATTR_NO_ALIAS]),
        ('data',       C.void_p, [ATTR_NO_ALIAS]),
    ]

    @property
    def dtypes(self):
        types = [a.type for a in self.in_args]
        if self.returns_value:
            types.append(self.return_type)
        else:
            types.extend(a.type.pointee for a in self.out_args)
        return map(llvm_ty_to_dtype, types)

    def body(self, args, dimensions, steps, data):
        func = self.depends(self.FuncDef)

        arg_ptrs = []
        arg_steps = []
        for i in range(self.nin + self.nout):
            arg_ptrs.append(self.var_copy(args[i]))
            const_steps = self.var_copy(steps[i])
            const_steps.invariant = True
            arg_steps.append(const_steps)

        N = self.var_copy(dimensions[0])
        N.invariant = True
        with self.for_range(N) as (loop, item):
            callargs = []
            for i, arg in enumerate(self.in_args):
                casted = arg_ptrs[i].cast(C.pointer(arg.type))
                callargs.append(casted.load())

            for i, arg in enumerate(self.out_args):
                i += self.nin
                casted = arg_ptrs[i].cast(arg.type)
                callargs.append(casted)

            if self.returns_value:
                res = func(*callargs, inline=True)
                retval_ptr = arg_ptrs[self.nin].cast(C.pointer(self.return_type))
                retval_ptr.store(res, nontemporal=True)
            else:
                func(*callargs, inline=True)

            for i in range(self.nin + self.nout):
                # increment pointers
                arg_ptrs[i].assign(arg_ptrs[i][arg_steps[i]:])

        self.ret()

    def specialize(self, lfunc):
        '''specialize to a workload
        '''
        func_def = CFuncRef(lfunc)
        self._name_ = UFUNC_PREFIX + str(func_def)
        self.FuncDef = func_def
        self.in_args = [a for a in lfunc.args
                        if not isinstance(a.type, PointerType)]
        self.out_args = [a for a in lfunc.args
                         if isinstance(a.type, PointerType)]
        self.return_type = lfunc.type.pointee.return_type
        self.returns_value = str(self.return_type) != 'void'
        if self.returns_value:
            assert not self.out_args, "not supported ATM"
        self.nin = len(self.in_args)
        self.nout = len(self.out_args) + int(self.returns_value)
