/* GRN discrete-tick step hot loop — CPython C API (doc/36 §4.2 / §5.2 P1).
 *
 * Native C backend for the GRN step recurrence, numerically equivalent to
 * impl_python (sigmoid-threshold path, decay blend, clip) — a pure speed
 * switch.  Compiled into helixlang/_accel/grn_step/ by the [native] build.
 *
 * step(levels, src, dst, weights, decays, thresholds, default_decay)
 *   -> (new_levels, triggered)
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static PyObject *
grn_step(PyObject *self, PyObject *args)
{
    PyObject *levels, *src, *dst, *weights, *decays, *thresholds;
    double default_decay;
    if (!PyArg_ParseTuple(args, "OOOOOOd:step",
                          &levels, &src, &dst, &weights, &decays, &thresholds,
                          &default_decay))
        return NULL;

    Py_ssize_t n = PyList_Size(levels);
    if (n < 0) return NULL;

    /* accumulate incoming weighted sums */
    double *acc = (double *)PyMem_Calloc(n > 0 ? (size_t)n : 1, sizeof(double));
    if (acc == NULL) return PyErr_NoMemory();

    Py_ssize_t e, n_edges = PyList_Size(src);
    if (n_edges < 0) { PyMem_Free(acc); return NULL; }
    for (e = 0; e < n_edges; e++) {
        long s = PyLong_AsLong(PyList_GET_ITEM(src, e));
        long d = PyLong_AsLong(PyList_GET_ITEM(dst, e));
        double w = PyFloat_AsDouble(PyList_GET_ITEM(weights, e));
        if ((s == -1 && PyErr_Occurred()) || (d == -1 && PyErr_Occurred()))
            goto err;
        double lvl = PyFloat_AsDouble(PyList_GET_ITEM(levels, s));
        if (lvl == -1.0 && PyErr_Occurred()) goto err;
        acc[d] += w * lvl;
    }

    PyObject *new_levels = PyList_New(n);
    PyObject *triggered = PyList_New(0);
    if (new_levels == NULL || triggered == NULL) {
        Py_XDECREF(new_levels); Py_XDECREF(triggered);
        PyMem_Free(acc); return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        double thr = PyFloat_AsDouble(PyList_GET_ITEM(thresholds, i));
        double x = acc[i] - thr;
        double raw;
        if (x >= 0.0) {
            raw = 1.0 / (1.0 + exp(-x));
        } else {
            double z = exp(x);
            raw = z / (1.0 + z);
        }
        PyObject *dec_obj = PyList_GET_ITEM(decays, i);
        double dec = (dec_obj == Py_None) ? default_decay
                                          : PyFloat_AsDouble(dec_obj);
        if (dec == -1.0 && PyErr_Occurred()) goto err_levels;
        double lvl = PyFloat_AsDouble(PyList_GET_ITEM(levels, i));
        double blended = dec * lvl + (1.0 - dec) * raw;
        double v = blended > 0.0 ? blended : 0.0;
        if (v > 1.0) v = 1.0;
        PyObject *item = PyFloat_FromDouble(v);
        PyList_SET_ITEM(new_levels, i, item);
        if (v > 0.5) {
            PyObject *idx = PyLong_FromSsize_t(i);
            PyList_Append(triggered, idx);
            Py_DECREF(idx);
        }
    }

    PyMem_Free(acc);
    PyObject *result = Py_BuildValue("(NN)", new_levels, triggered);
    return result;

err_levels:
    Py_DECREF(new_levels); Py_DECREF(triggered);
    PyMem_Free(acc);
    return NULL;
err:
    PyMem_Free(acc);
    return NULL;
}

static PyMethodDef methods[] = {
    {"step", grn_step, METH_VARARGS,
     "step(levels, src, dst, weights, decays, thresholds, default_decay)\n"
     "Advance one GRN tick (native).  Returns (new_levels, triggered)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "impl_cext",
    "Native C GRN step hot loop (doc/36 §5.2 P1).",
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_impl_cext(void)
{
    return PyModule_Create(&moduledef);
}
