import sys
import os.path
import subprocess
from cStringIO import StringIO
from shutil import copyfileobj

import llvm.core as lc, llvm.passes as lp
import llvm.ee as ee

from .builder import UFuncBuilder
from ._ufuncwrapper import UFuncWrapper

class CModule(object):
    BITCODE_EXT = '.bc'
    _stdout = None
    _stderr = None


    def __init__(self, name, sources, libraries=None, optimization_level=3,
                 language='c'):
        self.name = __name__ + '.' + name
        self.sources = sources
        self.libraries = libraries or ()
        self.optimization_level = optimization_level
        self.language = language
        self.module = self._create_module_from_source_files(sources)
        for l in self.libraries:
            self.module.add_library(l)
        self.module.verify()
        eb = ee.EngineBuilder.new(self.module)
        eb.opt(self.optimization_level)
        #eb.mattrs("-sse1,-sse2,-sse3")
        self._ee = eb.create()
        if self.optimization_level:
            self._optimize()


    def get_ufunc(self, name, doc=''):
        llvm_function = self.module.get_function_named(name)
        builder = UFuncBuilder(llvm_function,
            optimization_level=self.optimization_level)
        functions = [builder.ufunc]
        types = [d.num for  d in builder.dtypes]
        return UFuncWrapper(functions, types,
            nin=builder.nin,
            nout=builder.nout,
            name=name,
            doc=doc
            )

    @property
    def functions(self):
        return [f.name for f in self.module.functions
                if not f.name.startswith(UFuncBuilder.prefix)
                and f.linkage==lc.LINKAGE_EXTERNAL]

    def save_bitcode_to_disk(self):
        self._create_module_from_source_files(self.sources, True)

    def _optimize(self):
        pmb = lp.PassManagerBuilder.new()
        pmb.opt_level = self.optimization_level
        pmb.vectorize = True
        fpm = lp.PassManager.new()
        fpm.add(self.module.owner.target_data)
        fpm.add(lp.PASS_ALWAYS_INLINE)
        pmb.populate(fpm)
        fpm.run(self.module)

    def _create_module_from_source_files(self, sources, save_bitcode=False):
        module = lc.Module.new(self.name)
        for source in sources:
            if isinstance(source, basestring):
                assembly_file = source[:source.rfind('.')] + self.BITCODE_EXT
                if not os.path.exists(assembly_file) or\
                  os.path.getmtime(source)>os.path.getmtime(assembly_file):
                    bitcode = StringIO(self._compile_to_bitcode(source))
                    if save_bitcode:
                        copyfileobj(bitcode, open(assembly_file, 'w'))
                        bitcode.seek(0)
                else:
                    bitcode = open(assembly_file)
            else:
                bitcode = StringIO(self._compile_to_bitcode(source))
            module.link_in(lc.Module.from_bitcode(bitcode))
        return module
             
    def _compile_to_bitcode(self, source):
        p = subprocess.Popen(
            ['clang', '-Wall', '-emit-llvm',
             '-x', self.language,
             '-c', '-', '-o', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
