#include <Python.h>
#include <numpy/arrayobject.h>

/*
 * Wrappers around numpy C API macros so they can be inlined
 */

void
__PyArray_MultiIter_NEXT(PyObject *multi) {
    PyArray_MultiIter_NEXT(multi);
}

void *
__PyArray_MultiIter_DATA(PyObject *multi, int i) {
    return PyArray_MultiIter_DATA(multi, i);
}

int
__PyArray_MultiIter_NOTDONE(PyObject* multi) {
    return PyArray_MultiIter_NOTDONE(multi);
}
