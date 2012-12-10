import sys
import os.path
import subprocess
from cStringIO import StringIO
from shutil import copyfileobj

import llvm.core as lc
import llvm.ee as ee

from .builder import UFuncBuilder
from ._ufuncwrapper import UFuncWrapper

class CModule(object):
    BITCODE_EXT = '.bc'
    _stdout = None
    _stderr = None


    def __init__(self, name, sources, libraries=None, optimization_level=0):
        self.name = __name__ + '.' + name
        self.sources = sources
        self.libraries = libraries or ()
        self.optimization_level = optimization_level
        self.module = self._create_module_from_source_files(sources)
        for l in self.libraries:
            self.module.add_library(l)
        self.module.verify()
        self._ee = ee.ExecutionEngine.new(self.module)

    @property
    def functions(self):
        return [f.name for f in self.module.functions
                if not f.name.startswith(UFuncBuilder.prefix)
                and f.linkage==lc.LINKAGE_EXTERNAL]

    def save_bitcode_to_disk(self):
        self._create_module_from_source_files(self.sources, True)

    def _create_module_from_source_files(self, sources, save_bitcode=False):
        module = lc.Module.new(self.name)
        for source in sources:
            assembly_file = source[:source.rfind('.')] + self.BITCODE_EXT
            if not os.path.exists(assembly_file) or\
              os.path.getmtime(source)>os.path.getmtime(assembly_file):
                bitcode = StringIO(self._compile_to_bitcode(source))
                if save_bitcode:
                    copyfileobj(bitcode, open(assembly_file, 'w'))
                    bitcode.seek(0)
            else:
                bitcode = open(assembly_file)
            module.link_in(lc.Module.from_bitcode(bitcode))
        return module
             
    def _compile_to_bitcode(self, source):
        p = subprocess.Popen(['clang', '-Wall', '-c', '-emit-llvm',
            '-c', source,
            '-o', '-'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = self._stderr = p.communicate()
        if p.returncode!=0:
            #raise subprocess.CalledProcessError(p.returncode) FIXME
            print>>sys.stderr, self._stderr
            raise RuntimeError(str(p.returncode))
        return output

    def get_ufunc(self, name, doc=''):
        llvm_function = self.module.get_function_named(name)
        builder = UFuncBuilder(llvm_function, optimize=True)
        functions = [builder.ufunc]
        types = [d.num for  d in builder.dtypes]
        return UFuncWrapper(functions, types,
            nin=builder.nin,
            nout=builder.nout,
            name=name,
            doc=doc
            )

    def __getattr__(self, name):
        if name not in self.functions:
            raise AttributeError(name)
        return self.get_ufunc(name)
