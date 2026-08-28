"""Pure-Python reference dispatch kernel (doc/36 §5.5, P0).

A tiny interpreter over a contiguous bytecode buffer that shares the module
contract the native backends will implement.  It is intentionally minimal and
only covers the arithmetic/stack subset so it can serve as the reference that
C/Cython/Rust must match; the full language VM remains :mod:`helixlang.core.vm`.
"""
from __future__ import annotations

# Match helixlang.core.codon_table.Op values for the subset we dispatch.
_OP_HALT = 0x11
_OP_PUSH_CONST = 0x20
_OP_POP = 0x21
_OP_ADD = 0x90
_OP_SUB = 0x91
_OP_MUL = 0x92

_HANDLED = {
    _OP_HALT, _OP_PUSH_CONST, _OP_POP, _OP_ADD, _OP_SUB, _OP_MUL,
}


def run_quota(code, constants, *, quota: int = 4096, gene_table=None):
    """Run up to ``quota`` ops; returns ``(ops_consumed, stack, halted)``."""
    stack: list[float] = []
    ip = 0
    ops = 0
    halted = False
    n = len(code)
    while ip < n and ops < quota:
        op = code[ip]
        ip += 1
        if op not in _HANDLED:
            raise NotImplementedError(f"dispatch kernel: unhandled op 0x{op:02x}")
        if op == _OP_HALT:
            halted = True
            break
        elif op == _OP_PUSH_CONST:
            idx = code[ip]
            ip += 1
            stack.append(constants[idx])
        elif op == _OP_POP:
            stack.pop()
        elif op in (_OP_ADD, _OP_SUB, _OP_MUL):
            b = stack.pop()
            a = stack.pop()
            if op == _OP_ADD:
                stack.append(a + b)
            elif op == _OP_SUB:
                stack.append(a - b)
            else:
                stack.append(a * b)
        ops += 1
    return ops, stack, halted


def run_many(code, constants, *, quota: int = 4096, n_cells: int = 1,
             gene_table=None):
    """Population dispatch: run ``code`` independently for ``n_cells`` cells.

    Every cell executes the same bytecode on its own operand stack (doc/36
    §5.1.3 `dispatch duplicated per cell`).  Returns a list of ``n_cells``
    ``(ops_consumed, stack, halted)`` tuples, byte-identical to calling
    :func:`run_quota` once per cell.
    """
    return [
        run_quota(code, constants, quota=quota, gene_table=gene_table)
        for _ in range(n_cells)
    ]
