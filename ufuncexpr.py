import ast
import sys
from itertools import count

import ctypes, ctypes.util

from numba import vectorize


__all__ = ['evaluate', 'UFuncExpression']



def evaluate(expression, **namespace):
    """
    >>> a = 4
    >>> evaluate("a+2")
    6.0
    """
    if not namespace:
        namespace = sys._getframe(1).f_globals
    f = UFuncExpression(expression)
    return f(**namespace)


def _proxy_to_func(name):
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

    Can pickle and unpickle UFuncExpression
    
    >>> import pickle
    >>> pickle.loads(pickle.dumps(UFuncExpression('a+b+1')))(1, .5)
    2.5

    Can call reduce, accumulate, etc..

    >>> f6 = UFuncExpression("(a+1)*b")
    >>> f6.reduce(range(6))
    325.0
    >>> f6.accumulate(range(6))
    array([   0.,    1.,    4.,   15.,   64.,  325.])
    """

    _serial = count(0)
    _cache = {}

    def __new__(cls, expression, namespace=None, **kw):
        key = (expression, tuple(sorted(kw.items())))
        inst = cls._cache.get(key)
        if inst is None:
            inst = cls._cache[key] = cls._new()
        return inst

    @classmethod
    def _new(cls):
        """Subclasses should override to call parent __new__"""
        return object.__new__(cls)

    def __init__(self, expression, namespace=None, **kw):
        namespace = namespace if namespace is not None else kw
        self._initargs = (expression, namespace)
        self.globals_ = dict(
            __vectorize__ = vectorize.vectorize,
        )
        install_libmath(self.globals_)
        self.globals_.update(namespace)
        self._args, self.func = self._create_ufunc_from_expression(expression)

    def _decorate_function(self, funcdef):
        signatures = [
            _N('Str', 'i(%s)'%(','.join(['i']*len(funcdef.args.args)))),
            _N('Str', 'd(%s)'%(','.join(['d']*len(funcdef.args.args)))),
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


    reduce = _proxy_to_func('reduce')
    reduceAt = _proxy_to_func('reduceAt')
    accumulate = _proxy_to_func('accumulate')


        

libmath = ctypes.CDLL(ctypes.util.find_library('m'))
for name, nargs in [
    ('atan2', 2),
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
]:
    f = getattr(libmath, name)
    f.argtypes = [ctypes.c_double]*nargs
    f.restype = ctypes.c_double

def install_libmath(globals_=None):
    g = globals_ if globals_ is not None else sys.getframe(1).f_globals
    g.update((k,v) for (k,v) in vars(libmath).items() if not k.startswith('_'))


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
