from unittest import TestCase

class TestMultiIterFunc(TestCase):
    def test_arrays(self):
        shape = 1000,1000
        a = self._make_array(shape, 20)
        b = self._make_array(shape, 10)
        func = self._make_adder()
        res = func(a, b)
        self.failUnlessAllClose(res, self._make_array(shape, 30))

    def test_array_and_scalar(self):
        shape = 1000,1000
        a = self._make_array(shape, 20)
        b = 10
        func = self._make_adder()
        res = func(a, b)
        self.failUnlessAllClose(res, self._make_array(shape, 30))

    def test_scalars(self):
        a = 20
        b = 10
        func = self._make_adder()
        res = func(a, b)
        self.failUnlessEqual(res, 30)


    #
    # helpers
    #
    @staticmethod
    def failUnlessAllClose(a, b):
        import numpy
        return numpy.allclose(a, b)


    @staticmethod
    def _make_array(shape, value, dtype='d'):
        import numpy
        return numpy.ones(shape, dtype)*value

    @staticmethod
    def _make_adder():
        from numba import jit, double
        from ..vectorize import vectorize
        @vectorize(backend='multiiter')
        def func(a,b):
            return a+b
        return func

