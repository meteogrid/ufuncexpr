from llvm import core as lc
from llvm.core import Constant, Type, Function, Builder

from .util import (optimize_loop_func, dtype_to_numba, llvm_ty_to_dtype,
                   determine_pointer_size)


ssize_t = Type.int(determine_pointer_size())

# PyUFuncGenericFunction
PyUFuncGenericFunction_t = Type.function(Type.void(), [
    Type.pointer(Type.pointer(Type.int(8))), # char **data
    Type.pointer(ssize_t),                   # ssize_t *dimensions
    Type.pointer(ssize_t),                   # ssize_t *steps
    Type.pointer(Type.void()),               # void *data
    ])

class UFuncBuilder(object):
    prefix = '__PyUFuncGenericFunction_'
    def __init__(self, func, name=None, optimize=False):
        self.func = func
        self.name = name if name is not None else self.prefix+func.name
        self.optimize = optimize
        self.module = func.module
        self.in_args = [a for a in self.func.args
                        if not isinstance(a.type, lc.PointerType)]
        self.out_args = [a for a in self.func.args
                         if isinstance(a.type, lc.PointerType)]
        self.return_type = self.func.type.pointee.return_type
        self.returns_value = str(self.return_type) != 'void'
        if self.returns_value:
            assert not self.out_args, "not supported ATM"
        self.nin = len(self.in_args)
        self.nout = len(self.out_args) + int(self.returns_value)

    @property
    def dtypes(self):
        types = [a.type for a in self.in_args]
        if self.returns_value:
            types.append(self.return_type)
        else:
            types.extend(a.type.pointee for a in self.out_args)
        return map(llvm_ty_to_dtype, types)
    
    @property
    def ufunc(self):
        if self.name in [f.name for f in self.module.functions]:
            return self.module.get_function_named(self.name)
        else:
            ufunc = self._build_ufunc()
            if self.optimize:
                optimize_loop_func(ufunc, self.func)
            return ufunc

    def _build_ufunc(self):
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
        module = self.module
        func = self.func
        return_type = self.return_type
        name = self.name
        n_args = self.nin
        nin = Constant.int(ssize_t, self.nin)
        zero = Constant.int(ssize_t, 0)
        one = Constant.int(ssize_t, 1)
        nums = [Constant.int(ssize_t, i) for i in xrange(n_args)]
        ufunc = Function.new(module, PyUFuncGenericFunction_t, name)
        for a in ufunc.args:
            a.add_attribute(lc.ATTR_NO_ALIAS)

        # Build it
        block = ufunc.append_basic_block('entry')
        builder = Builder.new(block)
        _args, _dimensions, _steps = ufunc.args[:3]
        _args.name = "args"; _dimensions.name='dimensions'; _steps.name='steps'

        def deref(ptr, indices, name='tmp', cast=None):
            addr = builder.gep(ptr, indices, name+'_addr')
            if cast is not None and addr.type!=cast:
                addr = builder.bitcast(addr, cast)
            return builder.load(addr, name, invariant=True)

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
        st = builder.store(result,
                           builder.bitcast(result_addr, Type.pointer(return_type)))
        const_one = Constant.int(Type.int(), 1)
        md = lc.MetaData.get(module, [const_one])
        st.set_metadata('nontemporal', md)
        

        # calculate next loop value and branch
        next_value = builder.add(phi, one, 'next')
        phi.add_incoming(next_value, loop_block)
        builder.branch(loop_header_block)

        # after-block
        builder.position_at_end(after_block)
        builder.ret_void()

        return ufunc
