from llvm.core import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C

from .util import llvm_ty_to_dtype

class MultipleReturnUFunc(CDefinition):
    '''a generic ufunc that wraps the workload
    '''
    prefix = '__PyUFuncGenericFunction_'
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
        for i in range(self.nin + self.nout):
            arg = self.var_copy(args[i])
            arg.invariant = True
            arg_ptrs.append(arg)

        with self.for_range(dimensions[0]) as (loop, item):
            callargs = []
            for i, arg in enumerate(self.in_args):
                casted = arg_ptrs[i].cast(C.pointer(arg.type))[item:][0]
                callargs.append(casted)

            for i, arg in enumerate(self.out_args):
                i += self.nin
                casted = arg_ptrs[i].cast(arg.type)[item:]
                callargs.append(casted)

            if self.returns_value:
                res = func(*callargs, inline=True)
                P = C.pointer(self.return_type)
                retval_ptr = arg_ptrs[self.nin].cast(P)[item:].cast(P)
                retval_ptr.store(res, nontemporal=True)
            else:
                func(*callargs, inline=True)


        self.ret()

    def specialize(self, lfunc):
        '''specialize to a workload
        '''
        func_def = CFuncRef(lfunc)
        self._name_ = self.prefix + str(func_def)
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
