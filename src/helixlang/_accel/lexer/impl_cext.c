/* HelixLang lexer/tokenizer — hand-written CPython C API (doc/03 §6.5, doc/06 §19).
 *
 * Native C backend for the compiler+VM core's front-end stage.  A faithful
 * scalar port of helixlang._accel.lexer.impl_python (itself the lexer that
 * used to live in helixlang.core.lexer): dual-mode DNA scanner producing
 * CODON / GENE_ID / ANNOT_START / ANNOT_END / FIELD / ARROW /
 * USERDIRECTIVE / NEWLINE / EOF tokens with identical line/col bookkeeping,
 * Python-style backslash line continuations and identical LexError sites.
 *
 * Compiled into helixlang/_accel/lexer/ by the [native] build.
 *
 * tokenize(source, is_annotation_keyword=None) -> list[Token]
 *   Returns the Token dataclass instances exactly as the pure reference does.
 *   ``is_annotation_keyword`` is the core keyword/grammar-registry predicate
 *   (owned by Python); when omitted every bare ``#ident`` is a GENE_ID marker.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

static PyObject *LexerTokenCls = NULL;   /* helixlang.core.lexer.Token  */
static PyObject *LexErrorCls = NULL;     /* helixlang.core.errors.LexError */

typedef struct {
    Py_UCS4 *buf;         /* UCS4 copy of the source */
    Py_ssize_t len;
    Py_ssize_t pos;
    int line;
    int col;
    int codon_counter;
    PyObject *keywords_f; /* callable is_annotation_keyword or NULL */
    PyObject *out;        /* result list of Token */
} lexer_t;

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static int
is_base(Py_UCS4 c)
{
    return c == 'A' || c == 'C' || c == 'G' || c == 'T'
        || c == 'a' || c == 'c' || c == 'g' || c == 't';
}

/* _read_ident / DNA-mode gene peek share alnum-or-`_-+`.` semantics, except the
 * `#` marker peek does NOT admit '-' (matches impl_python). */
static int
is_ident_char(Py_UCS4 c)
{
    return Py_UNICODE_ISALNUM(c) || c == '_' || c == '-' || c == '.';
}

static int
is_peek_ident_char(Py_UCS4 c)
{
    return Py_UNICODE_ISALNUM(c) || c == '_' || c == '.';
}

static void
adv(lexer_t *lx, int is_newline)
{
    Py_UCS4 c = (lx->pos < lx->len) ? lx->buf[lx->pos] : 0;
    lx->pos++;
    if (is_newline || c == '\n') {
        lx->line++;
        lx->col = 1;
    }
    else {
        lx->col++;
    }
}

static int
at_line_continuation(lexer_t *lx)
{
    if (lx->pos >= lx->len || lx->buf[lx->pos] != '\\') {
        return 0;
    }
    Py_ssize_t nxt = lx->pos + 1;
    if (nxt < lx->len && lx->buf[nxt] == '\n') {
        return 1;
    }
    if (nxt + 1 < lx->len && lx->buf[nxt] == '\r'
        && lx->buf[nxt + 1] == '\n') {
        return 1;
    }
    return 0;
}

static void
skip_line_continuation(lexer_t *lx)
{
    adv(lx, 0);                                  /* backslash */
    if (lx->pos < lx->len && lx->buf[lx->pos] == '\r') {
        adv(lx, 0);                              /* CR (CRLF) */
    }
    adv(lx, 1);                                  /* newline */
    while (lx->pos < lx->len
           && (lx->buf[lx->pos] == ' ' || lx->buf[lx->pos] == '\t')) {
        adv(lx, 0);
    }
}

static void
skip_spaces(lexer_t *lx)
{
    while (lx->pos < lx->len
           && (lx->buf[lx->pos] == ' ' || lx->buf[lx->pos] == '\t')) {
        adv(lx, 0);
    }
}

static void
skip_to_newline(lexer_t *lx)
{
    while (lx->pos < lx->len && lx->buf[lx->pos] != '\n') {
        adv(lx, 0);
    }
    if (lx->pos < lx->len) {
        adv(lx, 1);
    }
}

static void
skip_dna_gap(lexer_t *lx)
{
    for (;;) {
        if (lx->pos >= lx->len) {
            return;
        }
        Py_UCS4 c = lx->buf[lx->pos];
        if (c == ' ' || c == '\t' || c == '\r') {
            adv(lx, 0);
        }
        else if (c == '\n') {
            adv(lx, 1);
        }
        else if (c == '\\' && at_line_continuation(lx)) {
            skip_line_continuation(lx);
        }
        else {
            return;
        }
    }
}

static PyObject *
slice_unicode(const Py_UCS4 *buf, Py_ssize_t start, Py_ssize_t end)
{
    if (end <= start) {
        return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf, 0);
    }
    return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start,
                                     end - start);
}

static PyObject *
read_ident(lexer_t *lx)
{
    Py_ssize_t start = lx->pos;
    while (lx->pos < lx->len && is_ident_char(lx->buf[lx->pos])) {
        adv(lx, 0);
    }
    return slice_unicode(lx->buf, start, lx->pos);
}

/* ------------------------------------------------------------------ */
/* token emission                                                     */
/* ------------------------------------------------------------------ */

static int
emit_token(lexer_t *lx, const char *kind, PyObject *value,
           int line, int col, int codon_index)
{
    PyObject *kind_str = PyUnicode_FromString(kind);
    PyObject *line_o = PyLong_FromLong(line);
    PyObject *col_o = PyLong_FromLong(col);
    PyObject *ci_o = PyLong_FromLong(codon_index);
    if (kind_str == NULL || line_o == NULL || col_o == NULL || ci_o == NULL) {
        Py_XDECREF(kind_str);
        Py_XDECREF(line_o);
        Py_XDECREF(col_o);
        Py_XDECREF(ci_o);
        return -1;
    }
    PyObject *args = PyTuple_New(5);
    if (args == NULL) {
        Py_DECREF(kind_str);
        Py_DECREF(line_o);
        Py_DECREF(col_o);
        Py_DECREF(ci_o);
        return -1;
    }
    Py_INCREF(value);               /* tuple steals this new ref; the caller
                                       keeps its own and DECREFs it */
    PyTuple_SET_ITEM(args, 0, kind_str);
    PyTuple_SET_ITEM(args, 1, value);
    PyTuple_SET_ITEM(args, 2, line_o);
    PyTuple_SET_ITEM(args, 3, col_o);
    PyTuple_SET_ITEM(args, 4, ci_o);
    PyObject *tok = PyObject_Call((PyObject *)LexerTokenCls, args, NULL);
    Py_DECREF(args);
    if (tok == NULL) {
        return -1;
    }
    if (PyList_Append(lx->out, tok) != 0) {
        Py_DECREF(tok);
        return -1;
    }
    Py_DECREF(tok);
    return 0;
}

static int
emit_str(lexer_t *lx, const char *kind, PyObject *value,
         int line, int col)
{
    return emit_token(lx, kind, value, line, col, -1);
}

/* Raise LexError(msg, line=..., col=...) — identical site semantics.  ``msg``
 * is a Unicode object: the "unexpected char" message must reproduce Python's
 * ``{c!r}`` repr exactly. */
static int
raise_lexerror(lexer_t *lx, PyObject *msg, int line, int col)
{
    (void)lx;                  /* error path only; info lives in msg/line/col */
    PyObject *args = PyTuple_Pack(1, msg);
    PyObject *kwargs = PyDict_New();
    if (args == NULL || kwargs == NULL) {
        Py_XDECREF(args);
        Py_XDECREF(kwargs);
        return -1;
    }
    if (PyDict_SetItemString(kwargs, "line", PyLong_FromLong(line)) != 0
        || PyDict_SetItemString(kwargs, "col", PyLong_FromLong(col)) != 0) {
        Py_DECREF(args);
        Py_DECREF(kwargs);
        return -1;
    }
    PyObject *e = PyObject_Call((PyObject *)LexErrorCls, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (e == NULL) {
        return -1;
    }
    PyErr_SetObject((PyObject *)LexErrorCls, e);
    Py_DECREF(e);
    return -1;
}

/* ------------------------------------------------------------------ */
/* sub-scanners                                                       */
/* ------------------------------------------------------------------ */

static PyObject *
read_value(lexer_t *lx)
{
    if (lx->pos < lx->len && lx->buf[lx->pos] == '"') {
        adv(lx, 0);
        Py_ssize_t start = lx->pos;
        while (lx->pos < lx->len && lx->buf[lx->pos] != '"') {
            adv(lx, 0);
        }
        PyObject *s = slice_unicode(lx->buf, start, lx->pos);
        if (s == NULL) {
            return NULL;
        }
        if (lx->pos < lx->len) {
            adv(lx, 0);                        /* closing " */
        }
        PyObject *res = PyUnicode_FromFormat("\"%U\"", s);
        Py_DECREF(s);
        return res;
    }
    PyObject *parts = PyList_New(0);
    if (parts == NULL) {
        return NULL;
    }
    Py_ssize_t start = lx->pos;
    for (;;) {
        if (lx->pos >= lx->len) {
            break;
        }
        Py_UCS4 c = lx->buf[lx->pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '#') {
            break;
        }
        if (c == '\\' && at_line_continuation(lx)) {
            PyObject *piece = slice_unicode(lx->buf, start, lx->pos);
            if (piece == NULL || PyList_Append(parts, piece) != 0) {
                Py_XDECREF(piece);
                Py_DECREF(parts);
                return NULL;
            }
            Py_DECREF(piece);
            skip_line_continuation(lx);
            start = lx->pos;
            continue;
        }
        adv(lx, 0);
    }
    PyObject *piece = slice_unicode(lx->buf, start, lx->pos);
    if (piece == NULL || PyList_Append(parts, piece) != 0) {
        Py_XDECREF(piece);
        Py_DECREF(parts);
        return NULL;
    }
    Py_DECREF(piece);
    PyObject *sep = PyUnicode_FromString("");
    if (sep == NULL) {
        Py_DECREF(parts);
        return NULL;
    }
    PyObject *join = PyUnicode_Join(sep, parts);
    Py_DECREF(sep);
    Py_DECREF(parts);
    return join;
}

static int
scan_fields_on_line(lexer_t *lx)
{
    for (;;) {
        if (lx->pos >= lx->len) {
            return 0;
        }
        Py_UCS4 c = lx->buf[lx->pos];
        if (c == '\\' && at_line_continuation(lx)) {
            skip_line_continuation(lx);
            continue;
        }
        if (c == '\n') {
            adv(lx, 1);
            return 0;
        }
        if (c == ' ' || c == '\t' || c == '\r') {
            adv(lx, 0);
            continue;
        }
        if (c == '#') {
            return 0;
        }
        int line0 = lx->line;
        int col0 = lx->col;
        PyObject *ident = read_ident(lx);
        if (ident == NULL) {
            return -1;
        }
        Py_ssize_t ilen;
        const char *ibuf = PyUnicode_AsUTF8AndSize(ident, (Py_ssize_t *)&ilen);
        if (ibuf == NULL) {
            Py_DECREF(ident);
            return -1;
        }
        if (ilen == 0) {
            Py_DECREF(ident);
            adv(lx, 0);                        /* skip unknown characters */
            continue;
        }
        skip_spaces(lx);
        if (lx->pos < lx->len && lx->buf[lx->pos] == '=') {
            adv(lx, 0);
            skip_spaces(lx);
            PyObject *value = read_value(lx);
            if (value == NULL) {
                Py_DECREF(ident);
                return -1;
            }
            PyObject *val_str = PyUnicode_FromFormat("%U=%U", ident, value);
            Py_DECREF(value);
            Py_DECREF(ident);
            if (val_str == NULL) {
                return -1;
            }
            int rc = emit_str(lx, "FIELD", val_str, line0, col0);
            Py_DECREF(val_str);
            if (rc != 0) {
                return -1;
            }
        }
        else if (lx->pos + 1 < lx->len && lx->buf[lx->pos] == '-'
                 && lx->buf[lx->pos + 1] == '>') {
            adv(lx, 0);
            adv(lx, 0);
            skip_spaces(lx);
            if (at_line_continuation(lx)) {
                skip_line_continuation(lx);
            }
            PyObject *target = read_ident(lx);
            if (target == NULL) {
                Py_DECREF(ident);
                return -1;
            }
            PyObject *arrow_str = PyUnicode_FromFormat("%U->%U", ident,
                                                       target);
            Py_DECREF(target);
            Py_DECREF(ident);
            if (arrow_str == NULL) {
                return -1;
            }
            int rc = emit_str(lx, "ARROW", arrow_str, line0, col0);
            Py_DECREF(arrow_str);
            if (rc != 0) {
                return -1;
            }
        }
        else {
            PyObject *field_str = PyUnicode_FromFormat("%U=", ident);
            Py_DECREF(ident);
            if (field_str == NULL) {
                return -1;
            }
            int rc = emit_str(lx, "FIELD", field_str, line0, col0);
            Py_DECREF(field_str);
            if (rc != 0) {
                return -1;
            }
        }
    }
}

static int
scan_dna(lexer_t *lx)
{
    Py_UCS4 *bases = NULL;
    Py_ssize_t n = 0;
    int line0 = lx->line;
    int col0 = lx->col;
    skip_dna_gap(lx);
    for (;;) {
        if (lx->pos >= lx->len || !is_base(lx->buf[lx->pos])) {
            break;
        }
        Py_UCS4 *nb = (Py_UCS4 *)PyMem_Realloc(bases, (size_t)(n + 1) * sizeof(Py_UCS4));
        if (nb == NULL) {
            PyMem_Free(bases);
            PyErr_SetString(PyExc_MemoryError, "realloc in DNA scan");
            return -1;
        }
        bases = nb;
        bases[n++] = (Py_UCS4)Py_UNICODE_TOUPPER(lx->buf[lx->pos]);
        adv(lx, 0);
        skip_dna_gap(lx);
    }
    if (n % 3 != 0) {
        PyMem_Free(bases);
        PyObject *msg = PyUnicode_FromFormat(
            "DNA length %zd not multiple of 3", n);
        if (msg == NULL) {
            return -1;
        }
        int rc = raise_lexerror(lx, msg, line0, col0);
        Py_DECREF(msg);
        return rc;
    }
    for (Py_ssize_t i = 0; i < n; i += 3) {
        PyObject *codon = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
                                                    bases + i, 3);
        if (codon == NULL) {
            PyMem_Free(bases);
            return -1;
        }
        int rc = emit_token(lx, "CODON", codon, line0, col0,
                            lx->codon_counter++);
        Py_DECREF(codon);
        if (rc != 0) {
            PyMem_Free(bases);
            return -1;
        }
    }
    PyMem_Free(bases);
    return 0;
}

static int
scan_userdirective(lexer_t *lx, int line0, int col0)
{
    PyObject *parts = PyList_New(0);
    if (parts == NULL) {
        return -1;
    }
    Py_ssize_t rest_start = lx->pos;
    for (;;) {
        while (lx->pos < lx->len && lx->buf[lx->pos] != '\n') {
            adv(lx, 0);
        }
        /* piece = src[rest_start:pos].rstrip('\r') */
        Py_ssize_t end = lx->pos;
        while (end > rest_start && lx->buf[end - 1] == '\r') {
            end--;
        }
        PyObject *piece = slice_unicode(lx->buf, rest_start, end);
        if (piece == NULL) {
            Py_DECREF(parts);
            return -1;
        }
        Py_ssize_t plen = PyUnicode_GET_LENGTH(piece);
        if (plen > 0 && PyUnicode_ReadChar(piece, plen - 1) == '\\') {
            PyObject *joiner = PyUnicode_Substring(piece, 0, plen - 1);
            Py_DECREF(piece);
            if (joiner == NULL) {
                Py_DECREF(parts);
                return -1;
            }
            if (PyList_Append(parts, joiner) != 0) {
                Py_DECREF(joiner);
                Py_DECREF(parts);
                return -1;
            }
            Py_DECREF(joiner);
            if (lx->pos < lx->len) {
                adv(lx, 1);
            }
            while (lx->pos < lx->len
                   && (lx->buf[lx->pos] == ' ' || lx->buf[lx->pos] == '\t')) {
                adv(lx, 0);
            }
            rest_start = lx->pos;
            continue;
        }
        if (PyList_Append(parts, piece) != 0) {
            Py_DECREF(piece);
            Py_DECREF(parts);
            return -1;
        }
        Py_DECREF(piece);
        break;
    }
    PyObject *sep = PyUnicode_FromString("");
    if (sep == NULL) {
        Py_DECREF(parts);
        return -1;
    }
    PyObject *rest = PyUnicode_Join(sep, parts);
    Py_DECREF(sep);
    Py_DECREF(parts);
    if (rest == NULL) {
        return -1;
    }
    PyObject *stripped = PyObject_CallMethod(rest, "strip", NULL);
    Py_DECREF(rest);
    if (stripped == NULL) {
        return -1;
    }
    int rc = emit_str(lx, "USERDIRECTIVE", stripped, line0, col0);
    Py_DECREF(stripped);
    if (rc != 0) {
        return -1;
    }
    if (lx->pos < lx->len && lx->buf[lx->pos] == '\n') {
        adv(lx, 1);
    }
    return 0;
}

static int
scan_annotation(lexer_t *lx)
{
    int line0 = lx->line;
    int col0 = lx->col;
    adv(lx, 0);                                 /* skip '#' */
    if (lx->pos >= lx->len || lx->buf[lx->pos] == ' '
        || lx->buf[lx->pos] == '\t' || lx->buf[lx->pos] == '#'
        || lx->buf[lx->pos] == '\r' || lx->buf[lx->pos] == '\n') {
        skip_to_newline(lx);                    /* line comment */
        return 0;
    }
    PyObject *name = read_ident(lx);
    if (name == NULL) {
        return -1;
    }
    Py_ssize_t nlen;
    const char *nchars = PyUnicode_AsUTF8AndSize(name, (Py_ssize_t *)&nlen);
    if (nchars == NULL) {
        Py_DECREF(name);
        return -1;
    }
    if (nlen == 0) {
        Py_DECREF(name);
        PyObject *msg = PyUnicode_FromString(
            "missing annotation name after '#'");
        if (msg == NULL) {
            return -1;
        }
        int rc = raise_lexerror(lx, msg, line0, col0);
        Py_DECREF(msg);
        return rc;
    }
    PyObject *upper = PyObject_CallMethod(name, "upper", NULL);
    if (upper == NULL) {
        Py_DECREF(name);
        return -1;
    }
    int is_end = PyUnicode_CompareWithASCIIString(upper, "END") == 0;
    Py_DECREF(upper);
    if (is_end) {
        Py_DECREF(name);
        PyObject *val = PyUnicode_FromString("#end");
        if (val == NULL) {
            return -1;
        }
        int rc = emit_str(lx, "ANNOT_END", val, line0, col0);
        Py_DECREF(val);
        if (rc != 0) {
            return -1;
        }
        skip_to_newline(lx);
        return 0;
    }
    PyObject *lower = PyObject_CallMethod(name, "lower", NULL);
    if (lower == NULL) {
        Py_DECREF(name);
        return -1;
    }
    int is_use = PyUnicode_CompareWithASCIIString(lower, "use") == 0;
    if (is_use) {
        Py_DECREF(lower);
        Py_DECREF(name);
        return scan_userdirective(lx, line0, col0);
    }
    int rc = emit_str(lx, "ANNOT_START", lower, line0, col0);
    Py_DECREF(lower);
    Py_DECREF(name);
    if (rc != 0) {
        return -1;
    }
    return scan_fields_on_line(lx);
}

static int
call_is_keyword(lexer_t *lx, PyObject *gene_id, int *out)
{
    if (lx->keywords_f == NULL) {
        *out = 0;
        return 0;
    }
    PyObject *res = PyObject_CallFunctionObjArgs(lx->keywords_f, gene_id,
                                                 NULL);
    if (res == NULL) {
        return -1;
    }
    *out = PyObject_IsTrue(res);
    Py_DECREF(res);
    return (*out < 0) ? -1 : 0;
}

static int
run(lexer_t *lx)
{
    for (;;) {
        if (lx->pos >= lx->len) {
            break;
        }
        Py_UCS4 c = lx->buf[lx->pos];
        if (c == '#') {
            /* Gene-ID marker vs annotation (impl_python tokens()). */
            Py_ssize_t peek_start = lx->pos + 1;
            Py_ssize_t m_end = 0;
            while (m_end < 30 && peek_start + m_end < lx->len
                   && is_peek_ident_char(lx->buf[peek_start + m_end])) {
                m_end++;
            }
            if (m_end > 0) {
                PyObject *gene_id = slice_unicode(lx->buf, peek_start,
                                                  peek_start + m_end);
                if (gene_id == NULL) {
                    return -1;
                }
                /* after_id = peek[m_end:m_end+5].lstrip() -> starts with '='? */
                int has_eq = 0;
                Py_ssize_t j = m_end;
                while (j < m_end + 5 && peek_start + j < lx->len) {
                    Py_UCS4 wc = lx->buf[peek_start + j];
                    if (Py_UNICODE_ISSPACE(wc)) {
                        j++;
                        continue;
                    }
                    if (wc == '=') {
                        has_eq = 1;
                    }
                    break;
                }
                int is_kw = 0;
                if (!has_eq && call_is_keyword(lx, gene_id, &is_kw) != 0) {
                    Py_DECREF(gene_id);
                    return -1;
                }
                if (!has_eq && !is_kw) {
                    int line0 = lx->line;
                    int col0 = lx->col;
                    Py_ssize_t k;
                    for (k = 0; k < m_end + 1; k++) {
                        adv(lx, 0);
                    }
                    int rc = emit_str(lx, "GENE_ID", gene_id, line0, col0);
                    Py_DECREF(gene_id);
                    if (rc != 0) {
                        return -1;
                    }
                }
                else {
                    Py_DECREF(gene_id);
                    if (scan_annotation(lx) != 0) {
                        return -1;
                    }
                }
            }
            else if (scan_annotation(lx) != 0) {
                return -1;
            }
        }
        else if (is_base(c)) {
            if (scan_dna(lx) != 0) {
                return -1;
            }
        }
        else if (c == ' ' || c == '\t' || c == '\r') {
            adv(lx, 0);
        }
        else if (c == '\\' && at_line_continuation(lx)) {
            skip_line_continuation(lx);
        }
        else if (c == '\n') {
            PyObject *nl = PyUnicode_FromString("\\n");
            if (nl == NULL) {
                return -1;
            }
            int rc = emit_str(lx, "NEWLINE", nl, lx->line, lx->col);
            Py_DECREF(nl);
            if (rc != 0) {
                return -1;
            }
            adv(lx, 1);
        }
        else {
            PyObject *ch = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
                                                     &c, 1);
            if (ch == NULL) {
                return -1;
            }
            PyObject *ch_repr = PyObject_Repr(ch);
            Py_DECREF(ch);
            if (ch_repr == NULL) {
                return -1;
            }
            PyObject *msg = PyUnicode_FromFormat("unexpected char %U",
                                                 ch_repr);
            Py_DECREF(ch_repr);
            if (msg == NULL) {
                return -1;
            }
            int rc = raise_lexerror(lx, msg, lx->line, lx->col);
            Py_DECREF(msg);
            return rc;
        }
    }
    PyObject *eof = PyUnicode_FromString("");
    if (eof == NULL) {
        return -1;
    }
    int rc = emit_str(lx, "EOF", eof, lx->line, lx->col);
    Py_DECREF(eof);
    return rc;
}

/* ------------------------------------------------------------------ */
/* module entry                                                        */
/* ------------------------------------------------------------------ */

static int
ensure_classes(void)
{
    if (LexerTokenCls != NULL && LexErrorCls != NULL) {
        return 0;
    }
    PyObject *lexer_mod = PyImport_ImportModule("helixlang.core.lexer");
    PyObject *errors_mod = PyImport_ImportModule("helixlang.core.errors");
    if (lexer_mod == NULL || errors_mod == NULL) {
        Py_XDECREF(lexer_mod);
        Py_XDECREF(errors_mod);
        return -1;
    }
    PyObject *tok = PyObject_GetAttrString(lexer_mod, "Token");
    PyObject *lexerr = PyObject_GetAttrString(errors_mod, "LexError");
    Py_DECREF(lexer_mod);
    Py_DECREF(errors_mod);
    if (tok == NULL || lexerr == NULL) {
        Py_XDECREF(tok);
        Py_XDECREF(lexerr);
        return -1;
    }
    if (LexerTokenCls == NULL) {
        LexerTokenCls = tok;
    }
    else {
        Py_DECREF(tok);
    }
    if (LexErrorCls == NULL) {
        LexErrorCls = lexerr;
    }
    else {
        Py_DECREF(lexerr);
    }
    return 0;
}

static PyObject *
tokenize(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;                /* module-level function */
    static char *kwlist[] = {"source", "is_annotation_keyword", NULL};
    PyObject *source;
    PyObject *keywords_f = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "U|O:tokenize", kwlist,
                                     &source, &keywords_f)) {
        return NULL;
    }
    if (ensure_classes() != 0) {
        return NULL;
    }
    lexer_t lx;
    memset(&lx, 0, sizeof lx);
    lx.len = PyUnicode_GET_LENGTH(source);
    lx.buf = PyUnicode_AsUCS4Copy(source);
    if (lx.buf == NULL) {
        return NULL;
    }
    lx.line = 1;
    lx.col = 1;
    lx.keywords_f = (keywords_f == Py_None) ? NULL : keywords_f;
    lx.out = PyList_New(0);
    if (lx.out == NULL) {
        PyMem_Free(lx.buf);
        return NULL;
    }
    int rc = run(&lx);
    PyMem_Free(lx.buf);
    if (rc != 0) {
        Py_DECREF(lx.out);
        return NULL;
    }
    return lx.out;
}

static PyMethodDef methods[] = {
    {"tokenize", (PyCFunction)tokenize, METH_VARARGS | METH_KEYWORDS,
     "tokenize(source, is_annotation_keyword=None) -> list[Token]"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef lexer_module = {
    PyModuleDef_HEAD_INIT,
    "helixlang._accel.lexer.impl_cext",
    "HelixLang lexer/tokenizer — hand-written C backend (doc/06 §19).",
    -1,
    methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC
PyInit_impl_cext(void)
{
    return PyModule_Create(&lexer_module);
}