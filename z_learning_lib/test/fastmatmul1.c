// fastmatmul1.c  — single-file CPython extension + C ABI function
// Build (Linux):
//   gcc -O3 -fPIC -shared fastmatmul1.c $(python3-config --includes) \
//       -o fastmatmul1$(python3-config --extension-suffix)
// Use (Python):
//   import fastmatmul1
//   C = fastmatmul1.matmul(A, B)   # A: m×k (list of lists), B: k×n -> C: m×n
//
// If you still want ctypes compatibility (e.g., your fastmatmul.py),
// you can also dlopen this .so and call the exported C symbol:
//   void matmul(const double* A, const double* B, double* C, int m, int k, int n)

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>

/* ---------------- Core compute (triple loop) ---------------- */
static void matmul_core(const double* A, const double* B, double* C,
                        int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            const double* Ai = A + (size_t)i * k;
            for (int t = 0; t < k; ++t) {
                acc += Ai[t] * B[(size_t)t * n + j];
            }
            C[(size_t)i * n + j] = acc;
        }
    }
}

/* --------------- C ABI (kept for compatibility) --------------- */
#ifdef _MSC_VER
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void matmul(const double* A, const double* B, double* C,
            int m, int k, int n) {
    matmul_core(A, B, C, m, k, n);
}

/* -------- Helpers: parse list-of-lists -> contiguous double* -------- */
static int as_2d_rect_matrix(PyObject* obj, double** out_buf, int* rows, int* cols) {
    // Expect a sequence of sequences (rectangular), convertible to float.
    PyObject* outer = PySequence_Fast(obj, "Expected a sequence (list/tuple) of rows");
    if (!outer) return -1;

    Py_ssize_t m = PySequence_Fast_GET_SIZE(outer);
    *rows = (int)m;
    *cols = 0;

    if (m == 0) {
        // Empty: treat as 0x0
        *out_buf = NULL;
        Py_DECREF(outer);
        return 0;
    }

    PyObject** row_objs = PySequence_Fast_ITEMS(outer);
    // Determine n from first row
    PyObject* first_row_seq = PySequence_Fast(row_objs[0], "Row 0 is not a sequence");
    if (!first_row_seq) { Py_DECREF(outer); return -1; }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(first_row_seq);
    Py_DECREF(first_row_seq);
    *cols = (int)n;

    // Allocate buffer
    size_t total = (size_t)m * (size_t)n;
    double* buf = (double*)malloc(total * sizeof(double));
    if (!buf) {
        PyErr_NoMemory();
        Py_DECREF(outer);
        return -1;
    }

    size_t idx = 0;
    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject* row_seq = PySequence_Fast(row_objs[i], "Row is not a sequence");
        if (!row_seq) { free(buf); Py_DECREF(outer); return -1; }
        Py_ssize_t rn = PySequence_Fast_GET_SIZE(row_seq);
        if (rn != n) {
            Py_DECREF(row_seq);
            free(buf);
            Py_DECREF(outer);
            PyErr_SetString(PyExc_ValueError, "Matrix is not rectangular");
            return -1;
        }
        PyObject** cells = PySequence_Fast_ITEMS(row_seq);
        for (Py_ssize_t j = 0; j < n; ++j) {
            double v = PyFloat_AsDouble(cells[j]);
            if (PyErr_Occurred()) {
                Py_DECREF(row_seq);
                free(buf);
                Py_DECREF(outer);
                return -1;
            }
            buf[idx++] = v;
        }
        Py_DECREF(row_seq);
    }

    Py_DECREF(outer);
    *out_buf = buf;
    return 0;
}

/* -------- Build Python list-of-lists from C buffer -------- */
static PyObject* to_py_2d_list(const double* buf, int m, int n) {
    PyObject* outer = PyList_New(m);
    if (!outer) return NULL;
    for (int i = 0; i < m; ++i) {
        PyObject* row = PyList_New(n);
        if (!row) { Py_DECREF(outer); return NULL; }
        for (int j = 0; j < n; ++j) {
            PyObject* f = PyFloat_FromDouble(buf[(size_t)i * n + j]);
            if (!f) { Py_DECREF(row); Py_DECREF(outer); return NULL; }
            PyList_SET_ITEM(row, j, f); // steals reference
        }
        PyList_SET_ITEM(outer, i, row); // steals reference
    }
    return outer;
}

/* ---------------- Python wrapper: matmul(A, B) ---------------- */
static PyObject* py_matmul(PyObject* self, PyObject* args) {
    PyObject *Aobj, *Bobj;
    if (!PyArg_ParseTuple(args, "OO", &Aobj, &Bobj)) {
        return NULL;
    }

    double *A = NULL, *B = NULL;
    int m = 0, k = 0, k2 = 0, n = 0;

    if (as_2d_rect_matrix(Aobj, &A, &m, &k) < 0) return NULL;
    if (as_2d_rect_matrix(Bobj, &B, &k2, &n) < 0) { free(A); return NULL; }

    if (k != k2) {
        free(A); free(B);
        PyErr_SetString(PyExc_ValueError, "Incompatible shapes: A(m×k) and B(k×n)");
        return NULL;
    }

    size_t total = (size_t)m * (size_t)n;
    double* C = (double*)malloc(total * sizeof(double));
    if (!C) { free(A); free(B); PyErr_NoMemory(); return NULL; }

    matmul_core(A, B, C, m, k, n);

    PyObject* out = to_py_2d_list(C, m, n);

    free(A); free(B); free(C);
    return out;
}

/* ---------------- Module boilerplate ---------------- */
static PyMethodDef FastMethods[] = {
    {"matmul", py_matmul, METH_VARARGS,
     "Matrix multiply two 2D Python lists: (m×k) @ (k×n) -> (m×n)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastmod = {
    PyModuleDef_HEAD_INIT,
    "fastmatmul1",            /* m_name */
    "Fast matmul (PyObjects + C ABI).", /* m_doc */
    -1,                       /* m_size */
    FastMethods,              /* m_methods */
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_fastmatmul(void) {
    return PyModule_Create(&fastmod);
}
