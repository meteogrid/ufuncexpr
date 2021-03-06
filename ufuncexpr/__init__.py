import ast
import sys
import math
from itertools import count

import ctypes, ctypes.util

from . import vectorize


__all__ = ['evaluate', 'UFuncExpression']



def evaluate(expression, _namespace=None, **namespace):
    """
    >>> a = 4
    >>> evaluate("a+2")
    6.0
    >>> def f(a,b):
    ...     return evaluate("a+b+2")
    >>> f(1,6)
    9.0

    >>> import numpy as np
    >>> b = np.array([range(2),range(2)])
    >>> evaluate("b+2")
    array([[ 2.,  3.],
           [ 2.,  3.]])
    """
    namespace = _namespace if _namespace else namespace
    if not namespace:
        f = sys._getframe(1)
        namespace = dict(f.f_globals, **f.f_locals)
    f = UFuncExpression(expression)
    return f(**namespace)


def _delegate_to_ufunc(name):
    return property(lambda s: getattr(s.func, name))

class UFuncExpression(object):
    """
    Creates an ufunc from an expression.

    >>> f1 = UFuncExpression('a+b')
    >>> f1(1,3)
    4.0

    Variables can be bound at compile time

    >>> f2 = UFuncExpression('a+b', b=3)
    >>> f2(2)
    5.0

    Functions are cached

    >>> f3 = UFuncExpression('a+b')
    >>> f3 is f1
    True
    >>> f4 = UFuncExpression('a+b', b=4)
    >>> f4 is f1
    False

    The return value is always the value of the last expression.

    >>> UFuncExpression("c=a+b;c+1")(a=1, b=4)
    6.0

    Can call functions form libm

    >>> UFuncExpression("sin(a)")(4)
    -0.7568024953079282

    Can use 'where(cond, then_value, else_value)' macro. 

    >>> f5 = UFuncExpression("where(a>1,a,b)")
    >>> f5(a=0.5, b=1), f5(a=2, b=4)
    (1.0, 2.0)

    Can use 'cond(*cases, default)' macro. 

    >>> f6 = UFuncExpression("cond((a>1,a),b)")
    >>> f6(a=0.5, b=1), f6(a=2, b=4)
    (1.0, 2.0)

    Can use 'switch(var, *cases, default)' macro. 

    >>> f7 = UFuncExpression("switch(a, ((1,2,3),0), ((4,5,6),1), -1)")
    >>> f7(a=1), f7(a=3), f7(a=4), f7(7)
    (0.0, 0.0, 1.0, -1.0)

    Can pickle and unpickle UFuncExpression
    
    >>> import pickle
    >>> pickle.loads(pickle.dumps(UFuncExpression('a+b+1')))(1, .5)
    2.5

    Can call reduce, accumulate, etc..

    >>> f6 = UFuncExpression("(a+1)*b", _backend='ufunc')
    >>> f6.reduce(map(float, range(6)))
    325.0
    >>> f6.accumulate(map(float, range(6)))
    array([   0.,    1.,    4.,   15.,   64.,  325.])
    """

    _serial = count(0)
    _cache = {}

    def __new__(cls, expression, namespace=None, **kw):
        key = (expression, tuple(sorted(kw.items())))
        inst = cls._cache.get(key)
        if inst is None:
            inst = cls._cache[key] = cls._new()
            inst._init(expression, namespace, **kw)
        return inst

    @classmethod
    def _new(cls):
        """Subclasses should override to call parent __new__"""
        return object.__new__(cls)

    def _init(self, expression, namespace=None, **kw):
        namespace = namespace if namespace is not None else dict(kw)
        self._initargs = (expression, namespace)
        def _vectorize(signatures):
            return vectorize.vectorize(signatures)
        self.globals_ = dict(
            __vectorize__ = _vectorize
        )
        install_libmath(self.globals_)
        self.globals_.update(namespace)
        self._args, self.func = self._create_ufunc_from_expression(expression)

    def _decorate_function(self, funcdef):
        signatures = [
            _N('Str', 'double(%s)'%(','.join(['double']*len(funcdef.args.args)))),
        ]
        deco = _N('Call',
            func=_N('Name', '__vectorize__', _N('Load')),
            args=[_N('List', elts=signatures, ctx=_N('Load'))],
            kwargs=None, starargs=None, keywords=[])
        funcdef.decorator_list.append(deco)

    def _create_ufunc_from_expression(self, expression):
        func_name = '__expr_func%d'%self._serial.next()
        tree = ast.parse(expression, mode='exec')
        visitor = TransformExpressionToFunction(func_name, self.globals_,
                                                self._decorate_function)
        tree = visitor.visit(tree)
        #assert False, ast.dump(tree)
        code = compile(tree, '<UFuncExpression %s>'%func_name, 'exec')
        d = dict(self.globals_)
        exec code in d
        return visitor.argument_names, d[func_name]

    def __call__(self, *args, **kw):
        if kw:
            args_ = [kw[k] for k in self._args if k in kw]
            if args_:
                kw = {}
                args = args_

        return self.func(*args, **kw)

    def __reduce__(self, proto=None):
        return (self.__class__, self._initargs)

    __reduce_ex__ = __reduce__


    reduce = _delegate_to_ufunc('reduce')
    reduceAt = _delegate_to_ufunc('reduceAt')
    accumulate = _delegate_to_ufunc('accumulate')


        

libmath = ctypes.CDLL(ctypes.util.find_library('m'))
for name, nargs in [
    ('atan2', 2),
    ('fmax', 2),
    ('fmin', 2),
    ('pow', 2),
    ('sqrt', 1),
    ('sin', 1),
    ('sinh', 1),
    ('asin', 1),
    ('asinh', 1),
    ('cos', 1),
    ('cosh', 1),
    ('acos', 1),
    ('acosh', 1),
    ('tan', 1),
    ('tanh', 1),
    ('atan', 1),
    ('atanh', 1),
    ('log', 1),
    ('log2', 1),
    ('log10', 1),
    ('log1p', 1),
    ('exp', 1),
    ('exp2', 1),
    ('expm1', 1),
    ('round', 1),
    ('ceil', 1),
    ('floor', 1),
    ('trunc', 1),
    ('fabs', 1),
    ('nan', 0),
]:
    f = getattr(libmath, name)
    f.argtypes = [ctypes.c_double]*nargs
    f.restype = ctypes.c_double

def install_libmath(namespace=None):
    if namespace is not None:
        g = namespace
    else:
        f = sys._getframe(1)
        g = f.f_locals
    g.update((k,v) for (k,v) in vars(libmath).items() if not k.startswith('_'))
    g.update(pi=math.pi)


def _N(name, *args, **kw):
    kw.setdefault('lineno', 1)
    kw.setdefault('col_offset', 0)
    node = getattr(ast, name)(*args, **kw)
    return node

class TransformExpressionToFunction(ast.NodeTransformer):
    """Wraps a single expression in a module with a function definition"""
    def __init__(self, function_name, globals_, decorator=None):
        self.function_name = function_name
        self.globals_ = globals_
        self.free_vars = set()
        self._decorator = decorator
        self.assignments = set()
        self._macros = dict(
            where = self._where_macro,
            cond = self._cond_macro,
            switch = self._switch_macro,
            )

    @property
    def argument_names(self):
        return tuple(sorted(self.free_vars))

    def visit_Name(self, node):
        if node.id not in self.globals_ and node.id not in self.assignments:
            self.free_vars.add(node.id)
        return node

    def visit_Assign(self, node):
        self.assignments.add(node.targets[0].id)
        return _N('Assign', node.targets, self.visit(node.value))

    def visit_Module(self, node):
        if len(node.body) < 1:
            raise _syntax_error(node, "No expressions")
        body = [self.visit(n.value if isinstance(n, ast.Expr) else n)
                for n in node.body]
        assert body
        if not isinstance(body[-1], ast.Return):
            body[-1] = _N('Return', body[-1])
        args = _N('arguments', [_N('Name', n, _N('Param'))
                                for n in self.argument_names], None, None, [])
        func = _N('FunctionDef', self.function_name, args, body, [])
        if self._decorator is not None:
            self._decorator(func)
        return _N('Module', body=[func])


    def visit_FunctionDef(self, node):
        raise _syntax_error(node, "Cannot define functions inside expressions")

    def visit_Call(self, node):
        if hasattr(node.func, 'id'):
            macro = self._macros.get(node.func.id)
            if macro is not None:
                return macro(map(self.visit, node.args))
        return self.generic_visit(node)


    @staticmethod
    def _where_macro(args):
        """Implements the 'where' macro."""
        try:
            test, then, else_ = args
        except ValueError:
            raise _syntax_error(args, "'where' expects 3 arguments")
        return _N('IfExp', test, then, else_)

    @staticmethod
    def _cond_macro(args):
        """Implements the 'cond' macro."""
        if not args:
            raise _syntax_error(None, "Missing conditionals")
        for a in args[:-1]:
            if not isinstance(a, ast.Tuple):
                raise _syntax_error(a, "More than one default")
        if isinstance(args[-1], ast.Tuple):
            raise _syntax_error(a,
                "Last argument to 'cond' must be the default case")
        args = list(args)
        default = args.pop(-1)
        while args:
            cond = args.pop(-1)
            test, then = cond.elts
            default = _N('IfExp', test, then, default)
        return default

    @staticmethod
    def _switch_macro(args):
        """Implements the 'switch' macro."""
        if not args:
            raise _syntax_error(None, "Missing conditionals")

        args = list(args)
        default = args.pop(-1)
        left = args.pop(0)
        while args:
            matches, then = args.pop(-1).elts
            test = _N('BoolOp',
                op=_N('Or'),
                values=[_N('Compare',
                    left=left,
                    ops=[_N('Eq')],
                    comparators=[c]) for c in matches.elts])
            default = _N('IfExp', test, then, default)
        return default


def _syntax_error(node, msg, filename='<string>'):
    e = SyntaxError(msg)
    e.filename = filename
    e.lineno = getattr(node, "lineno", 0)
    e.offset = getattr(node, "col_offset", 0)
    return e

def additional_tests():
    import doctest
    return doctest.DocTestSuite(__name__)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
