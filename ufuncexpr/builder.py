from llvm import core as lc, passes as lp
from llvm.core import Constant, Type, Function, Builder

from .util import dtype_to_numba, llvm_ty_to_dtype, determine_pointer_size


ssize_t = Type.int(determine_pointer_size())
Ptr = Type.pointer

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
                self.optimize_loop_func()
            return ufunc

    def _context(self):
        _arg_ixs = [Constant.int(ssize_t, i) for i in xrange(self.nin)]
        _out_ixs = [Constant.int(ssize_t, i)
                       for i in xrange(self.nin, self.nin+self.nout)]
        ufunc = self._new_ufunc()
        _args, _dimensions, _steps = ufunc.args[:3]
        module = self.module
        entry_bb = ufunc.append_basic_block('entry')
        builder = Builder.new(entry_bb)
        deref = self._make_deref(builder)
        zero = Constant.int(ssize_t, 0)
        # Load constants from arguments,
        builder.position_at_end(entry_bb)
        num_iterations = deref(_dimensions, [zero], 'num_iterations')
        arg_bases = [deref(_args, [i], 'abase') for i in _arg_ixs]
        arg_steps = [deref(_steps, [i], 'astep') for i in _arg_ixs]
        result_bases = [deref(_args, [i], 'rbase') for i in _out_ixs]
        result_steps = [deref(_steps, [i], 'rstep') for i in _out_ixs]

        return type('_BContext', (object,), dict(
            zero = zero,
            func = ufunc,
            entry_bb = entry_bb,
            b = builder,
            num_iterations = num_iterations,
            arg_bases = arg_bases,
            arg_steps = arg_steps,
            result_bases = result_bases,
            result_steps = result_steps,
            store = staticmethod(self._make_store(builder)),
            deref = staticmethod(deref),
        ))

    @staticmethod
    def _make_deref(builder):
        def deref(ptr, indices, name='tmp', cast=None, invariant=True, load=True):
            addr = builder.gep(ptr, indices, name+'_addr')
            if cast is not None and addr.type!=cast:
                addr = builder.bitcast(addr, cast)
            if load:
                return builder.load(addr, name, invariant=invariant)
            else:
                return addr
        return deref

    def _make_store(self, builder):
        def store(value, address, cast=None, nontemporal=True):
            if cast is not None and address.type.pointee!=cast:
               address = builder.bitcast(address, cast)
            st = builder.store(value, address)
            if nontemporal:
                st.set_metadata('nontemporal',
                    lc.MetaData.get(self.module, [Constant.int(Type.int(), 1)]))
            return st
        return store

    def _new_ufunc(self):
        ufunc = Function.new(self.module, PyUFuncGenericFunction_t, self.name)
        for a in ufunc.args:
            a.add_attribute(lc.ATTR_NO_ALIAS)
        _args, _dimensions, _steps = ufunc.args[:3]
        _args.name = "args"
        _dimensions.name='dimensions'
        _steps.name='steps'
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
        C = self._context()
        loop_header_block = C.func.append_basic_block('loop-header')
        C.b.branch(loop_header_block)

        # loop-header block
        C.b.position_at_end(loop_header_block)
        idx = C.b.phi(ssize_t, 'loop_var')
        idx.add_incoming(C.zero, C.entry_bb)

        end_cond = C.b.icmp(lc.ICMP_SLT, idx, C.num_iterations)
        loop_block = C.func.append_basic_block('loop')
        after_block = C.func.append_basic_block('afterloop')
        C.b.cbranch(end_cond, loop_block, after_block)

        # loop body block
        C.b.position_at_end(loop_block)
        if self.returns_value:
            self._call_and_store_result(C, idx)
        else:
            self._call_and_store_result_out_args(C, idx)

        # calculate next loop value and branch
        next_value = C.b.add(idx, Constant.int(ssize_t, 1), 'next')
        idx.add_incoming(next_value, loop_block)
        C.b.branch(loop_header_block)

        # after-block
        C.b.position_at_end(after_block)
        C.b.ret_void()
        return C.func

    def _call_and_store_result(self, context, idx):
        C = context
        func_args = [
            C.deref(base, [C.b.mul(step, idx)], cast=Ptr(arg.type))
                 for base, step, arg in
                     zip(C.arg_bases, C.arg_steps, self.in_args)]
        result = C.b.call(self.func, func_args)
        result_addr = C.b.gep(C.result_bases[0], [C.b.mul(C.result_steps[0], idx)])
        C.store(result, result_addr, Type.pointer(self.return_type))

    def _call_and_store_result_out_args(self, context, idx):
        C = context
        func_args = [
            C.deref(base, [C.b.mul(step, idx)], cast=Ptr(arg.type))
                 for base, step, arg in
                     zip(C.arg_bases, C.arg_steps, self.in_args)]
        func_args.extend(
            C.deref(base, [C.b.mul(step, idx)], cast=arg.type, load=False)
                 for base, step, arg in
                     zip(C.result_bases, C.result_steps, self.out_args))
        C.b.call(self.func, func_args)

    #@syncronized FIXME
    def optimize_loop_func(self, opt_level=3):
        self.func.add_attribute(lc.ATTR_ALWAYS_INLINE)
        try:
            pmb = lp.PassManagerBuilder.new()
            pmb.opt_level = opt_level
            pmb.vectorize = True
            fpm = lp.PassManager.new()
            fpm.add(lp.PASS_ALWAYS_INLINE)
            pmb.populate(fpm)
            fpm.run(self.module)
        finally:
            self.func.remove_attribute(lc.ATTR_ALWAYS_INLINE)
