import sys
import os.path
import subprocess
from cStringIO import StringIO
from shutil import copyfileobj
from ctypes.util import find_library

import llvm.core as lc
import llvm.ee as ee

from .builder import make_ufunc, UFUNC_PREFIX

lc.load_library_permanently(find_library('stdc++'))

class CModule(object):
    BITCODE_EXT = '.bc'
    ATEXIT_NAME = '__cxa_atexit'
    INITIALIZER_NAME = '__cxx_global_var_init'

    _stdout = None
    _stderr = None

    _destructor = None


    def __init__(self, name, sources, libraries=None, optimization_level=3,
                 cflags=None, language='c'):
        self.name = __name__ + '.' + name
        self.sources = sources
        self.cflags = cflags
        self.libraries = libraries or ()
        self.optimization_level = optimization_level
        self.language = language
        self.module = self._create_module_from_source_files(sources)
        for l in self.libraries:
            lc.load_library_permanently(find_library(l))
        self.module.verify()
        eb = ee.EngineBuilder.new(self.module).opt(optimization_level)
        self._ee = eb.create()
        self._ufunc_cache = {}

        initializer = self._extract_initializer()
        if initializer:
            self._ee.run_function(initializer, [])

    def __del__(self):
        if self._destructor:
            self._ee.run_function(self._destructor, [])

    def _extract_initializer(self):
        """Extracts a C++ module initilalizer created by clang if any"""
        name = self.INITIALIZER_NAME
        if name in [f.name for f in self.module.functions]:
            initializer = self.module.get_function_named(name)
            self._intercept_calls_to_atexit(initializer)
            return initializer

    def _intercept_calls_to_atexit(self, function):
        name = self.ATEXIT_NAME
        destructors = []
        for b in function.basic_blocks:
            for i in b.instructions:
                if hasattr(i,'called_function') and i.called_function.name==name:
                    destructors.append(i.operands[:2])
                    i.erase_from_parent()
        self._destructor = self._create_destructor(destructors)

    def _create_destructor(self, destructors):
        func_t = lc.Type.function(lc.Type.void(), [])
        func = lc.Function.new(self.module, func_t, '_mod_destructor')
        entry_bb = func.append_basic_block('entry')
        builder = lc.Builder.new(entry_bb)
        builder.position_at_end(entry_bb)
        for destructor, arg in destructors:
            builder.call(destructor, [arg])
        builder.ret_void()
        return func

    def get_ufunc(self, name, doc=''):
        ufunc = self._ufunc_cache.get(name)
        if ufunc is None:
            llvm_function = self.module.get_function_named(name)
            ufunc = make_ufunc([llvm_function], self._ee,
                               doc='CModule wrapped universal function')
            self._ufunc_cache[name] = ufunc
        return ufunc

    @property
    def functions(self):
        return [f.name for f in self.module.functions
                if not f.name.startswith(UFUNC_PREFIX)
                and f.linkage==lc.LINKAGE_EXTERNAL]

    def save_bitcode_to_disk(self):
        self._create_module_from_source_files(self.sources, True)

    def _create_module_from_source_files(self, sources, save_bitcode=False):
        module = lc.Module.new(self.name)
        for source in sources:
            if hasattr(source, 'read'):
                bitcode = StringIO(self._compile_to_bitcode(source))
            else:
                assembly_file = source[:source.rfind('.')] + self.BITCODE_EXT
                if not os.path.exists(assembly_file) or\
                  os.path.getmtime(source)>os.path.getmtime(assembly_file):
                    with open(source) as f:
                        bitcode_str = self._compile_to_bitcode(f,
                            cwd=os.path.dirname(source)
                            )
                    bitcode = StringIO(bitcode_str)
                    if save_bitcode:
                        copyfileobj(bitcode, open(assembly_file, 'w'))
                        bitcode.seek(0)
                else:
                    bitcode = open(assembly_file)
            mod = lc.Module.from_bitcode(bitcode)
            module.link_in(mod)
        return module
             
    def _compile_to_bitcode(self, source, cwd=None):
        
        p = subprocess.Popen(
            ['clang', '-Wall', '-emit-llvm']
            + list(self.cflags or []) +
            ['-x', self.language,
             '-c', '-', '-o', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        if hasattr(source, 'read'):
            source = source.read()
        output, self._stderr = p.communicate(source)
        if p.returncode!=0:
            #raise subprocess.CalledProcessError(p.returncode) FIXME
            print>>sys.stderr, self._stderr
            raise RuntimeError(str(p.returncode))
        return output

    def __getattr__(self, name):
        if name not in self.functions:
            raise AttributeError(name)
        return self.get_ufunc(name)
