/* VM opcode dispatch hot loop — CPython C API (doc/36 §4.2 / §5.5 P0).
 *
 * Native C backend for the tiny bytecode-interpreter dispatch kernel, a pure
 * speed switch equivalent to impl_python: same opcode subset, same operand
 * stack, same IEEE-754 float arithmetic, same quota accounting (HALT consumes
 * 0 ops).  Also provides population dispatch (run_many) that executes the same
 * bytecode across N independent per-cell stacks in one C call (doc/36 §5.1.3).
 *
 * Compiled into helixlang/_accel/dispatch/ by the [native] build.
 *
 * run_quota(code, constants, *, quota=4096, gene_table=None)
 *   -> (ops_consumed, stack, halted)
 * run_many(code, constants, *, quota=4096, n_cells=1, gene_table=None)
 *   -> list of (ops_consumed, stack, halted)
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* Match helixlang._accel.dispatch.impl_python opcode subset. */
#define OP_HALT       0x11
#define OP_PUSH_CONST 0x20
#define OP_POP        0x21
#define OP_ADD        0x90
#define OP_SUB        0x91
#define OP_MUL        0x92

typedef struct {
    double *data;
    Py_ssize_t top;   /* number of elements */
    Py_ssize_t cap;
} FloatStack;

static int
floatstack_push(FloatStack *s, double v)
{
    if (s->top == s->cap) {
        Py_ssize_t ncap = s->cap ? s->cap * 2 : 64;
        double *nd = (double *)PyMem_Realloc(s->data, (size_t)ncap * sizeof(double));
        if (nd == NULL) return -1;
        s->data = nd;
        s->cap = ncap;
    }
    s->data[s->top++] = v;
    return 0;
}

static int
floatstack_pop(FloatStack *s, double *out)
{
    if (s->top == 0) {
        PyErr_SetString(PyExc_IndexError, "pop from empty dispatch stack");
        return -1;
    }
    *out = s->data[--s->top];
    return 0;
}

/* Executes one cell's program (shared by run_quota and run_many).
 * On success returns a (ops, stack_list, halted) tuple; NULL on error. */
static PyObject *
run_one(PyObject **code_items, Py_ssize_t n,
        PyObject **const_items, Py_ssize_t nconst,
        long quota)
{
    FloatStack st;
    st.data = NULL; st.top = 0; st.cap = 0;

    Py_ssize_t ip = 0;
    long ops = 0;
    int halted = 0;

    while (ip < n && ops < quota) {
        long op = PyLong_AsLong(code_items[ip]);
        if (op == -1 && PyErr_Occurred()) goto fail;
        ip++;
        if (op != OP_HALT && op != OP_PUSH_CONST && op != OP_POP &&
            op != OP_ADD && op != OP_SUB && op != OP_MUL) {
            PyErr_Format(PyExc_NotImplementedError,
                         "dispatch kernel: unhandled op 0x%02lx", op);
            goto fail;
        }
        if (op == OP_HALT) {
            halted = 1;
            break;
        } else if (op == OP_PUSH_CONST) {
            if (ip >= n) {
                /* Truncated operand: a lone trailing OP_PUSH_CONST.  Match the
                 * python reference (IndexError) rather than reading past the
                 * end of the code buffer (doc/36 §11 malicious-bytecode gate). */
                PyErr_SetString(PyExc_IndexError,
                                "dispatch kernel: truncated PUSH_CONST operand");
                goto fail;
            }
            long idx = PyLong_AsLong(code_items[ip]);
            if (idx == -1 && PyErr_Occurred()) goto fail;
            ip++;
            if (idx < 0 || idx >= nconst) {
                PyErr_SetString(PyExc_IndexError, "dispatch constant index out of range");
                goto fail;
            }
            double cv = PyFloat_AsDouble(const_items[idx]);
            if (cv == -1.0 && PyErr_Occurred()) goto fail;
            if (floatstack_push(&st, cv) != 0) { PyErr_NoMemory(); goto fail; }
        } else if (op == OP_POP) {
            double dummy;
            if (floatstack_pop(&st, &dummy) != 0) goto fail;
        } else {
            double b, a;
            if (floatstack_pop(&st, &b) != 0) goto fail;
            if (floatstack_pop(&st, &a) != 0) goto fail;
            double r;
            if (op == OP_ADD) r = a + b;
            else if (op == OP_SUB) r = a - b;
            else r = a * b;
            if (floatstack_push(&st, r) != 0) { PyErr_NoMemory(); goto fail; }
        }
        ops++;
    }

    PyObject *stack_list = PyList_New(st.top);
    if (stack_list == NULL) goto fail;
    for (Py_ssize_t i = 0; i < st.top; i++) {
        PyObject *f = PyFloat_FromDouble(st.data[i]);
        if (f == NULL) { Py_DECREF(stack_list); PyMem_Free(st.data); return NULL; }
        PyList_SET_ITEM(stack_list, i, f);
    }
    PyObject *ops_obj = PyLong_FromLong(ops);
    if (ops_obj == NULL) { Py_DECREF(stack_list); PyMem_Free(st.data); return NULL; }
    PyObject *halted_obj = halted ? Py_True : Py_False;
    Py_INCREF(halted_obj);
    PyObject *result = Py_BuildValue("(NNN)", ops_obj, stack_list, halted_obj);
    PyMem_Free(st.data);
    return result;

fail:
    PyMem_Free(st.data);
    return NULL;
}

static int
parse_common(PyObject *args, PyObject *kwargs,
             char **kwlist, long quota_default,
             PyObject **code_obj, PyObject **consts_obj,
             long *quota, PyObject **gene_table)
{
    *gene_table = Py_None;
    *quota = quota_default;
    return PyArg_ParseTupleAndKeywords(
        args, kwargs, "OO|lO", kwlist,
        code_obj, consts_obj, quota, gene_table);
}

static PyObject *
dispatch_run_quota(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *code_obj, *consts_obj, *gene_table;
    long quota;
    static char *kwlist[] = {(char *)"code", (char *)"constants",
                             (char *)"quota", (char *)"gene_table", NULL};
    if (!parse_common(args, kwargs, kwlist, 4096, &code_obj, &consts_obj,
                      &quota, &gene_table))
        return NULL;

    PyObject *code_seq = PySequence_Fast(code_obj, "code must be a sequence");
    if (code_seq == NULL) return NULL;
    PyObject *const_seq = PySequence_Fast(consts_obj, "constants must be a sequence");
    if (const_seq == NULL) { Py_DECREF(code_seq); return NULL; }

    PyObject *res = run_one(PySequence_Fast_ITEMS(code_seq),
                            PySequence_Fast_GET_SIZE(code_seq),
                            PySequence_Fast_ITEMS(const_seq),
                            PySequence_Fast_GET_SIZE(const_seq),
                            quota);
    Py_DECREF(const_seq);
    Py_DECREF(code_seq);
    return res;
}

static PyObject *
dispatch_run_many(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *code_obj, *consts_obj, *gene_table;
    long quota;
    long n_cells;
    static char *kwlist[] = {(char *)"code", (char *)"constants",
                             (char *)"quota", (char *)"n_cells",
                             (char *)"gene_table", NULL};
    gene_table = Py_None;
    quota = 4096;
    n_cells = 1;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO|llO", kwlist,
            &code_obj, &consts_obj, &quota, &n_cells, &gene_table))
        return NULL;
    if (n_cells < 0) { n_cells = 0; }

    PyObject *code_seq = PySequence_Fast(code_obj, "code must be a sequence");
    if (code_seq == NULL) return NULL;
    PyObject *const_seq = PySequence_Fast(consts_obj, "constants must be a sequence");
    if (const_seq == NULL) { Py_DECREF(code_seq); return NULL; }

    PyObject **ci = PySequence_Fast_ITEMS(code_seq);
    Py_ssize_t cn = PySequence_Fast_GET_SIZE(code_seq);
    PyObject **kitems = PySequence_Fast_ITEMS(const_seq);
    Py_ssize_t nconst = PySequence_Fast_GET_SIZE(const_seq);

    PyObject *list = PyList_New(n_cells);
    if (list == NULL) { Py_DECREF(const_seq); Py_DECREF(code_seq); return NULL; }
    for (long c = 0; c < n_cells; c++) {
        PyObject *res = run_one(ci, cn, kitems, nconst, quota);
        if (res == NULL) {
            Py_DECREF(list);
            Py_DECREF(const_seq);
            Py_DECREF(code_seq);
            return NULL;
        }
        PyList_SET_ITEM(list, c, res);
    }
    Py_DECREF(const_seq);
    Py_DECREF(code_seq);
    return list;
}

static PyMethodDef methods[] = {
    {"run_quota", (PyCFunction)dispatch_run_quota, METH_VARARGS | METH_KEYWORDS,
     "run_quota(code, constants, *, quota=4096, gene_table=None)\n"
     "Execute up to ``quota`` ops of ``code`` (native).  Returns "
     "(ops_consumed, stack, halted)."},
    {"run_many", (PyCFunction)dispatch_run_many, METH_VARARGS | METH_KEYWORDS,
     "run_many(code, constants, *, quota=4096, n_cells=1, gene_table=None)\n"
     "Population dispatch of ``code`` across ``n_cells`` independent stacks "
     "(native).  Returns a list of (ops_consumed, stack, halted)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "impl_cext",
    "Native C VM dispatch + population dispatch hot loop (doc/36 §5.5 P0).",
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_impl_cext(void)
{
    return PyModule_Create(&moduledef);
}
