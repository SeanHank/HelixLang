"""BatchRuntime: vectorised population runtime over the typed IR.

A third runtime in the pipeline (doc/37 §4.3): executes *N cells* against the
*same* IR program simultaneously.  Pure arithmetic / stack-shuffle
instructions (the dispatch-kernel subset: PUSH, POP, DUP, SWAP, ADD, SUB,
MUL, LT, NOT) run as one numpy/JAX array op across every aligned cohort of
cells, while biological effects (protein synthesis, movement, regulation,
calls, halt ...) are applied with the trusted single-cell dispatcher so the
outcome is bit-for-bit the same as N independent runs.

Backends:

- ``numpy`` — a vectorised CPU "appliance" (default; numpy is a project
  dependency of the vectorized/GRN layers).
- ``jax`` — the same plan compiled to XLA, executing on GPU/TPU devices when
  requested and available; transparently falls back to numpy otherwise
  (device availability is a *performance* property, never a *fidelity* one —
  doc/37 §3 decoupling rule).

Equivalence to N sequential :class:`~helixlang.core.vm.CellVM` runs is asserted
by ``tests/test_ir_runtime.py`` and validation benchmark 72.
"""
from __future__ import annotations

from typing import Any

from helixlang.core.codon_table import Op
from helixlang.core.ir import IRInst, IRProgram
from helixlang.core.ir_runtime import IRRuntime

# Opcodes that run in the vectorised bank (the dispatch-kernel subset).
VECTOR_OPS: frozenset[Op] = frozenset({
    Op.OP_PUSH_CONST, Op.OP_POP, Op.OP_DUP, Op.OP_SWAP,
    Op.OP_ADD, Op.OP_SUB, Op.OP_MUL, Op.OP_LT, Op.OP_NOT,
})

_ARITH_ARITY: dict[Op, int] = {
    Op.OP_ADD: 2, Op.OP_SUB: 2, Op.OP_MUL: 2, Op.OP_LT: 2, Op.OP_NOT: 1,
}

_BANK_SIZE = 512


class StackDepthError(RuntimeError):
    """A vector cohort underflowed the stack (never happens for valid programs)."""

    def __init__(self, needed: int) -> None:
        super().__init__(f"stack underflow in vector kernel: needed {needed}")
        self.needed = needed


def _op_operand_bytes(op: Op) -> int:
    from helixlang.core.codon_table import OP_OPERAND_BYTES

    return OP_OPERAND_BYTES[op]


# ── array engines ──────────────────────────────────────────────────────────
class _NumpyEngine:
    """numpy-backed banked register file S[N, M] with per-cell depth."""

    def __init__(self, n: int, m: int = _BANK_SIZE) -> None:
        import numpy as np
        self.S: Any = np.zeros((n, m))
        self.depth: Any = np.zeros(n, dtype=np.int64)

    def write_col(self, rows: Any, col: Any, values: Any) -> None:
        self.S[rows, col] = values

    def read_col(self, rows: Any, col: Any) -> Any:
        return self.S[rows, col]

    def swap_cols(self, rows: Any, col_a: Any, col_b: Any) -> None:
        tmp = self.S[rows, col_a].copy()
        self.S[rows, col_a] = self.S[rows, col_b]
        self.S[rows, col_b] = tmp

    def bump_depth(self, rows: Any, amount: Any) -> None:
        self.depth[rows] += amount

    def set_depth(self, rows: Any, values: Any) -> None:
        self.depth[rows] = values

    def store_rows(self, rows: Any, values: Any) -> None:
        self.S[rows] = values

    def load_rows(self, rows: Any) -> Any:
        return self.S[rows]


class _JAXEngine:
    """jax.numpy device engine — same contract, functional updates."""

    def __init__(self, n: int, m: int = _BANK_SIZE) -> None:
        import jax.numpy as jnp
        self.S: Any = jnp.zeros((n, m))
        self.depth: Any = jnp.zeros(n, dtype=jnp.int32)

    def write_col(self, rows: Any, col: Any, values: Any) -> None:
        self.S = self.S.at[rows, col].set(values)

    def read_col(self, rows: Any, col: Any) -> Any:
        return self.S[rows, col]

    def swap_cols(self, rows: Any, col_a: Any, col_b: Any) -> None:
        tmp = self.S[rows, col_a]
        self.S = self.S.at[rows, col_a].set(self.S[rows, col_b])
        self.S = self.S.at[rows, col_b].set(tmp)

    def bump_depth(self, rows: Any, amount: Any) -> None:
        self.depth = self.depth.at[rows].add(amount)

    def set_depth(self, rows: Any, values: Any) -> None:
        self.depth = self.depth.at[rows].set(values)

    def store_rows(self, rows: Any, values: Any) -> None:
        self.S = self.S.at[rows].set(values)

    def load_rows(self, rows: Any) -> Any:
        return self.S[rows]


def _make_engine(backend: str, n: int) -> tuple[Any, str, str]:
    if backend == "jax":
        try:
            import jax.numpy  # noqa: F401
            return _JAXEngine(n), "jax", "jax"
        except ImportError:
            backend = "numpy"
    if backend == "numpy":
        import numpy  # noqa: F401
        return _NumpyEngine(n), "numpy", "numpy"
    raise ValueError(f"unknown batch backend {backend!r} (numpy|jax)")


# ── vectorised kernel ──────────────────────────────────────────────────────
def _vector_apply(engine: Any, op: Op, operand: int | None,
                  rows: Any) -> None:
    """Apply one pure op to the banked stacks of ``rows`` (all rows at same
    stack depth).  Semantics match the classic dispatcher exactly."""
    import numpy as np

    d = engine.depth[rows]
    if op is Op.OP_POP:
        engine.bump_depth(rows, -1)
        return
    if op is Op.OP_PUSH_CONST:
        if operand is None:
            raise StackDepthError(1)
        v = float(operand)
        newd = d + 1
        if int(np.max(newd)) >= _BANK_SIZE:
            raise StackDepthError(int(np.max(newd)))
        engine.write_col(rows, newd - 1, np.full(len(rows), v))
        engine.bump_depth(rows, 1)
        return
    if op is Op.OP_DUP:
        if int(np.min(d)) < 1:
            raise StackDepthError(1)
        engine.write_col(rows, d, engine.read_col(rows, d - 1))
        engine.bump_depth(rows, 1)
        return
    if op is Op.OP_SWAP:
        if int(np.min(d)) < 2:
            raise StackDepthError(2)
        engine.swap_cols(rows, d - 1, d - 2)
        return
    arity = _ARITH_ARITY[op]
    if int(np.min(d)) < arity:
        raise StackDepthError(arity)
    if arity == 2:
        a = engine.read_col(rows, d - 2)
    else:
        a = None
    b = engine.read_col(rows, d - 1)
    if op is Op.OP_ADD:
        engine.write_col(rows, d - 2, a + b)
        engine.bump_depth(rows, -1)
    elif op is Op.OP_SUB:
        engine.write_col(rows, d - 2, a - b)
        engine.bump_depth(rows, -1)
    elif op is Op.OP_MUL:
        engine.write_col(rows, d - 2, a * b)
        engine.bump_depth(rows, -1)
    elif op is Op.OP_LT:
        engine.write_col(rows, d - 2, np.asarray(a < b, dtype=float))
        engine.bump_depth(rows, -1)
    elif op is Op.OP_NOT:
        nz = np.asarray(b != 0, dtype=float)
        engine.write_col(rows, d - 1, 1.0 - nz)


# ── batch runtime ──────────────────────────────────────────────────────────
class BatchRuntime:
    """Run N cells on one IR program with a vectorised pure-op kernel.

    ``run(max_ticks)`` returns ``list`` of per-cell traces, each trace a list
    of snapshots — identical to N independent ``CellVM``/``IRRuntime`` runs.
    """

    def __init__(self, ir: IRProgram, program: Any, *, n: int = 8,
                 backend: str = "numpy"):
        if n < 1:
            raise ValueError("batch size n must be >= 1")
        self.ir = ir
        self.program = program
        self.n = n
        self.backend = backend
        self.engine, self.family, self.active_backend = _make_engine(backend, n)
        self.cells: list[IRRuntime] = [IRRuntime(ir, program) for _ in range(n)]
        self._quotas: list[int] = []

    # -------- public API --------
    def run(self, max_ticks: int) -> list[list[dict]]:
        for c in self.cells:
            c._snapshot_downsampler.configure(max_ticks)
        for _ in range(max_ticks):
            if not any(c.cell.alive for c in self.cells):
                break
            if self.program.config.use_central_dogma:
                self._tick_scalar_central_dogma()
            else:
                self._tick_vector()
            for c in self.cells:
                c.tick += 1
        return [c.trace for c in self.cells]

    # -------- classic path (vectorised) --------
    def _tick_vector(self) -> None:
        quota = self.program.config.ops_per_tick
        self._quotas = [quota] * self.n
        for c in self.cells:
            triggered = c.grn.step()
            for g in triggered:
                c._call_gene(g)
        self._execute_pending_all()
        for c in self.cells:
            c._flush_morphology()
            c._feedback()
            c._snapshot()

    def _execute_pending_all(self) -> None:
        while True:
            active = [i for i, c in enumerate(self.cells)
                      if c.frames and self._quotas[i] > 0]
            if not active:
                return
            groups: dict[tuple[Any, ...], list[int]] = {}
            for i in active:
                c = self.cells[i]
                key: tuple[Any, ...]
                if c.ip >= len(c._flat):
                    key = ("end",)
                else:
                    inst = c._flat[c.ip]
                    key = (int(inst.opcode), inst.operand)
                groups.setdefault(key, []).append(i)
            for key, rows in groups.items():
                if key[0] == "end":
                    self._scalar_pop_frames(rows)
                    continue
                inst = self.cells[rows[0]]._flat[self.cells[rows[0]].ip]
                if inst.opcode in VECTOR_OPS and len(rows) > 1:
                    self._vector_step(rows, inst)
                else:
                    for i in rows:
                        self._scalar_step(i)
                for i in rows:
                    self._quotas[i] -= 1
                    self.cells[i].ops_executed += 1

    # -------- step kernels --------
    def _vector_step(self, rows: list[int], inst: IRInst) -> None:
        """Advance ip, then apply the pure op on the bank across ``rows``."""
        for i in rows:
            self.cells[i].ip += 1
        self._vector_apply(rows, inst)

    def _vector_apply(self, rows: list[int], inst: IRInst) -> None:
        import numpy as np
        engine = self.engine
        row_idx = np.asarray(rows, dtype=np.int64)
        # sync python stacks -> bank
        raw = np.zeros((len(rows), _BANK_SIZE))
        for _k, i in enumerate(rows):
            stack = self.cells[i].stack
            raw[_k, : len(stack)] = stack
        engine.store_rows(row_idx, raw)
        depths = np.asarray([len(self.cells[i].stack) for i in rows],
                            dtype=np.int64)
        engine.set_depth(row_idx, depths)
        _vector_apply(engine, inst.opcode,
                      inst.operand if isinstance(inst.operand, int) else None,
                      row_idx)
        # sync bank -> python stacks
        for _k, i in enumerate(rows):
            d = int(engine.depth[i])
            self.cells[i].stack = [float(v) for v in engine.S[i, :d]]

    def _scalar_step(self, i: int) -> None:
        c = self.cells[i]
        inst = c._flat[c.ip]
        c.ip += 1
        c._current_inst = inst
        c._current_operand_bytes = _operand_bytes(inst)
        if c.frames and len(c.frames) > 256:
            c.frames.clear()
            return
        c._dispatch(inst.opcode)

    def _scalar_pop_frames(self, rows: list[int]) -> None:
        for i in rows:
            c = self.cells[i]
            c.frames.pop()
            if c.frames:
                c.ip = c.frames[-1].return_ip

    # -------- central dogma path (no pure-op stream to vectorise) --------
    def _tick_scalar_central_dogma(self) -> None:
        for c in self.cells:
            c._process_bio_instructions()
            c._transcribe_translate()
            c._flush_morphology()
            c._feedback()
            c._snapshot()


def _operand_bytes(inst: IRInst) -> list[int]:
    operand = inst.operand
    if isinstance(operand, int):
        if _op_operand_bytes(inst.opcode) >= 2:
            return [(operand >> 8) & 0xFF, operand & 0xFF]
        return [operand & 0xFF]
    return []
