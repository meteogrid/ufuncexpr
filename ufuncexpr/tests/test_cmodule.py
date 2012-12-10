from unittest2 import TestCase

class TestCModule(TestCase):
    def _makeOne(self, name='test', sources=(), libraries=None):
        from ..cmodule import CModule
        return  CModule(name, sources, libraries)

    def _makeOneFromString(self, source):
        from cStringIO import StringIO
        return self._makeOne(sources=[StringIO(source)])

    def _makeArray(self, data, dtype='d'):
        from numpy import array
        return array(data, dtype)

    def assertArraysEqual(self, expected, other):
        from numpy import allclose
        self.assertTrue(allclose(expected, other), repr((expected, other)))

    def test_with_int32_return_value(self):
        mod = self._makeOneFromString("int f(int a, int b) {return a*b+1;}")
        a = self._makeArray(range(10), 'i')
        b = self._makeArray(range(10), 'i')
        result = mod.f(a,b)
        self.assertEqual('int32', str(result.dtype))
        self.assertArraysEqual(a*b+1, result)

    def test_with_double_return_value(self):
        mod = self._makeOneFromString(
            "double f(double a, double b) {return a*b+1;}")
        a = self._makeArray(range(10))
        b = self._makeArray(range(10))
        result = mod.f(a,b)
        self.assertEqual('float64', str(result.dtype))
        self.assertArraysEqual(a*b+1, result)

    def test_with_int32_out_param(self):
        mod = self._makeOneFromString(
            "void f(int a, int b, int *out) {*out = a*b+1;}")
        a = self._makeArray(range(10), 'i')
        b = self._makeArray(range(10), 'i')
        result = mod.f(a,b)
        self.assertEqual('int32', str(result.dtype))
        self.assertArraysEqual(a*b+1, result)

    def test_multiple_return_values(self):
        mod = self._makeOneFromString("""
            void f(int a, int b, int *out1, double *out2) {
                *out1 = a*b+1; *out2 = a+b/(a+1);
            }""")
        a = self._makeArray(range(10), 'i')
        b = self._makeArray(range(10), 'i')
        result = mod.f(a,b)
        self.assertIsInstance(result, tuple)
        self.assertEqual('int32', str(result[0].dtype))
        self.assertEqual('float64', str(result[1].dtype))
        self.assertArraysEqual(a*b+1, result[0])
        self.assertArraysEqual(a+b/(a+1), result[1])
