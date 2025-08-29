#define PY_SSIZE_T_CLEAN

#include <Python.h>

static PyObject * logarithm_2(PyObject * self, PyObject * args) {
    unsigned int x;
    if (!PyArg_ParseTuple(args, "i", &x)) {
        return NULL; // Error parsing argument
    }
    
    unsigned int log2x = 0;
    while (x > 1) {
        log2x++;
        x >>= 1; // Equivalent to x = x / 2
    }
    return PyLong_FromLong(log2x);
}

static PyMethodDef MyMethods[] = {
    {"logarithm_2", (PyCFunction)(void(*)(void))logarithm_2, METH_VARARGS, "Calculate the base-2 logarithm of an integer."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule", // name of module
    NULL, // module documentation, may be NULL
    -1, // size of per-interpreter state of the module,
       // or -1 if the module keeps state in global variables.
    MyMethods
};

PyMODINIT_FUNC 
PyInit_mymodule(void) {
    return PyModule_Create(&mymodule);
}