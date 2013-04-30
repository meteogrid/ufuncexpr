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

class LLVMFunctionDecorator(CDefinition):
    _prefix = '_LLVMFunctionDecorator_'

    @property
    def dtypes(self):
        types = [a.type for a in self.in_args]
        if self.returns_value:
            types.append(self.return_type)
        else:
            types.extend(a.type.pointee for a in self.out_args)
        return map(llvm_ty_to_dtype, types)

    def specialize(self, lfunc):
        '''specialize to a workload
        '''
        self.original_name = lfunc.name
        func_def = CFuncRef(lfunc)
        self._name_ = self._prefix + str(func_def)
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

class MultipleReturnUFunc(LLVMFunctionDecorator):
    '''a generic ufunc that wraps the workload
    '''
    _prefix = UFUNC_PREFIX
    _argtys_ = [
        ('args',       C.pointer(C.char_p), [ATTR_NO_ALIAS]),
        ('dimensions', C.pointer(C.intp), [ATTR_NO_ALIAS]),
        ('steps',      C.pointer(C.intp), [ATTR_NO_ALIAS]),
        ('data',       C.void_p, [ATTR_NO_ALIAS]),
    ]

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

class PTXElementwiseKernel(LLVMFunctionDecorator):
    def specialize(self, lfunc):
        super(PTXElementwiseKernel, self).specialize(lfunc)
        self._argtys_ = self._generate_argtys()

    def _generate_argtys(self):
        argtys = []
        for arg in self.in_args:
            argtys.append((arg.name, C.pointer(arg.type), [ATTR_NO_ALIAS]))
        if self.returns_value:
            argtys.append(('out', C.pointer(self.return_type), [ATTR_NO_ALIAS]))
        else:
            for arg in self.out_args:
                argtys.append((arg.name, arg.type, [ATTR_NO_ALIAS]))
        argtys.append(('n_elements', C.py_ssize_t))
        return argtys


    def body(self, *args):
        I = PTXIntrinsics(self)
        func = self.depends(self.FuncDef)
        tid = self.var(C.int, I.threadIdx_x(), 'tid')
        n_threads = self.var(C.int, I.gridDim_x()*I.blockDim_x(), 'total_threads')
        cta_start = self.var(C.int, I.blockDim_x()*I.blockIdx_x(), 'cta_start')
        N = self.var_copy(args[-1].cast(C.int), 'total_elements')

        with self.for_range(cta_start+tid, N, n_threads) as (loop, idx):
            if self.returns_value:
                callargs = [a[idx] for a in args[:self.nin]]
                args[-2][idx] = func(*callargs, inline=True)
            else:
                callargs = [a[idx] for a in args[:self.nin]]
                callargs.extend(a[idx:] for a in args[self.nin:self.nin+self.nout])
                func(*callargs, inline=True)
        self.ret()

    def define(self, module):
        mod = module.clone()
        func = super(PTXElementwiseKernel, self).define(mod, optimize=False)
        func.calling_convention = CC_PTX_KERNEL
        func.name = self.original_name
        self._postprocess_function_module(func)
        return func

    def _postprocess_function_module(self, func):
        mod = func.module
        while [f for f in mod.functions
               if f is not func and not f.is_declaration]:
            for f in mod.functions:
                if f is not func and not f.uses:
                    f.delete()


class PTXIntrinsics(object):
    def __init__(self, builder):
        for name, intrinsic_id in [
            ('threadIdx_x', INTR_PTX_READ_TID_X),
            ('threadIdx_y', INTR_PTX_READ_TID_Y),
            ('threadIdx_z', INTR_PTX_READ_TID_Z),
            ('blockIdx_x', INTR_PTX_READ_CTAID_X),
            ('blockIdx_y', INTR_PTX_READ_CTAID_Y),
            ('blockIdx_z', INTR_PTX_READ_CTAID_Z),
            ('blockDim_x', INTR_PTX_READ_NTID_X),
            ('blockDim_y', INTR_PTX_READ_NTID_Y),
            ('blockDim_z', INTR_PTX_READ_NTID_Z),
            ('gridDim_x', INTR_PTX_READ_NCTAID_X),
            ('gridDim_y', INTR_PTX_READ_NCTAID_Y),
            ('gridDim_z', INTR_PTX_READ_NCTAID_Z),
        ]:
            intr = builder.get_intrinsic(intrinsic_id, [])
            setattr(self, name, intr)


hpool = dpool = None

def make_gufunc(llvm_function, name=None, doc="ufuncexpr wrapped function",
                cuda_module_options=[]):
    import pycuda.autoinit
    from pycuda.driver import module_from_buffer
    import pycuda.driver as cuda
    import pycuda.tools
    import numpy as np
    
    def_ =  PTXElementwiseKernel(llvm_function)
    func = def_(llvm_function.module)

    capability = pycuda.autoinit.device.compute_capability()
    cpu = 'sm_%d%d' % capability
    ptxtm = le.TargetMachine.lookup(arch='nvptx64', cpu=cpu)
    pm = lp.build_pass_managers(ptxtm, opt=3, fpm=False).pm
    pm.run(func.module)
    asm = ptxtm.emit_assembly(func.module)

    #XXX: Hack. llvm 3.2 doesn't set map_f64_to_f32 for cpu < sm_13 as it should
    if capability < (1, 3):
        target_str = '.target ' + cpu
        asm = asm.replace(target_str, target_str + ', map_f64_to_f32')

    mod = module_from_buffer(asm, options=cuda_module_options)
    gfunc = mod.get_function(func.name)
    gfunc.prepare('P'*(def_.nin+def_.nout) + 'i')

    global hpool, dpool
    if None in [hpool, dpool]:
        hpool = pycuda.tools.PageLockedMemoryPool()
        dpool = pycuda.tools.DeviceMemoryPool()


    def wrapper(*args, **kw):
        block = kw.setdefault('block', (256,1,1))
        grid = kw.setdefault('grid', (1,1))
        shape = args[0].shape
        assert len(args) == def_.nin
        assert all((a.dtype==d) for (a,d) in zip(args, def_.dtypes))
        call_args = []
        for a in args:
            a_gpu = dpool.allocate(a.size * a.dtype.itemsize)
            cuda.memcpy_htod(a_gpu, a)
            call_args.append(a_gpu)
        del args

        outputs = []
        for dtype in def_.dtypes[def_.nin:]:
            o = hpool.allocate(shape, dtype=dtype)
            o_gpu = dpool.allocate(a.size * a.dtype.itemsize)
            outputs.append((o,o_gpu))
            call_args.append(o_gpu)
        N = reduce(lambda a,b:a*b, shape)
        call_args.append(N)
        gfunc.prepared_call(grid, block, *call_args)
        for o, o_gpu in outputs:
            cuda.memcpy_dtoh(o, o_gpu)
        out = tuple(o[0] for o in outputs)
        if len(out)==1:
            out = out[0]
        return out
    wrapper.func_name = name or llvm_function.name
    wrapper.__doc__ = doc
    wrapper.llvm_kernel = func
    return wrapper
