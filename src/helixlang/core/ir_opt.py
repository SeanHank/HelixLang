"""IROpt: optimization passes over the typed biological IR.

Passes (all semantics-preserving on the classic VM):

- **constant folding**: a run of pure instructions whose inputs are all
  ``OP_PUSH_CONST`` literals is replaced by a single literal for the
  arithmetic opcodes (``ADD/SUB/MUL/LT/NOT``).  Folding is evaluated strictly
  in runtime evaluation order (``b, a = pop(); pop(); a op b``) so the folded
  value is bit-identical to what the VM would compute — no IEEE-754 surprises.
- **dead instruction elimination**: removes ``PUSH;POP``, ``DUP;POP`` and
  ``READ_MEM;POP`` strands whose value is discarded unused, plus ``SWAP;SWAP``
  no-ops.
- **unreachable elimination**: removes instructions after a terminal
  ``HALT``/``RETURN`` inside a gene ORF (belt-and-braces; the IR carries no
  implicit halt).

Optimization simulates the stack over *pure* runs only, so it never crosses
an effect boundary (BUILD_PROTEIN, WRITE_MEM, REGULATE, ...).  Because effects
partition every gene into independent arithmetic micro-blocks, per-tick
side-effect ordering is preserved exactly; the only observable difference is a
reduced instruction count.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from helixlang.core.codon_table import Op
from helixlang.core.ir import PURE_OPS, IRInst, IRProgram, IRType

# Foldable arithmetic: opcode -> (arity, fold_fn, result_type).  Operands are
# passed in *stack order* (the left value first), matching VM pops:
# ``b, a = pop(); pop(); a op b``.
_FOLDERS: dict[Op, tuple[int, Callable[[list], object], IRType]] = {
    Op.OP_ADD: (2, lambda vs: vs[0] + vs[1], IRType.NUM),
    Op.OP_SUB: (2, lambda vs: vs[0] - vs[1], IRType.NUM),
    Op.OP_MUL: (2, lambda vs: vs[0] * vs[1], IRType.NUM),
    Op.OP_LT: (2, lambda vs: 1 if vs[0] < vs[1] else 0, IRType.BOOL),
    Op.OP_NOT: (1, lambda vs: 1 if not vs[0] else 0, IRType.BOOL),
}


class IROpt:
    """Optimizer: runs passes in dependency order.  Idempotent and per-gene."""

    def __init__(self) -> None:
        self.runs: dict[str, int] = {}

    def optimize(self, program: IRProgram, *,
                 passes: Iterable[str] | None = None) -> IRProgram:
        active = list(passes) if passes else ["fold", "dead", "unreachable"]
        for name in active:
            fn = getattr(self, f"_pass_{name}", None)
            if fn is None:
                raise ValueError(f"unknown IR pass {name!r}")
            for func in program.functions:
                func.instrs = fn(func.instrs)
            self.runs[name] = self.runs.get(name, 0) + 1
        return program

    # -------- constant folding --------
    @staticmethod
    def _pass_fold(instrs: list[IRInst]) -> list[IRInst]:
        out: list[IRInst] = []
        window: list[IRInst] = []  # current pure run
        for inst in instrs:
            if inst.opcode not in PURE_OPS:
                out.extend(window)
                window = []
                out.append(inst)
                continue
            window.append(inst)
            while True:
                folded = IROpt._try_fold_tail(window)
                if folded is None:
                    break
                new_inst, n_replaced = folded
                window = window[: len(window) - n_replaced] + [new_inst]
        out.extend(window)
        return out

    @staticmethod
    def _try_fold_tail(instrs: list[IRInst]) -> tuple[IRInst, int] | None:
        """Fold ``PUSH a ; PUSH b ; OP`` windows into ``PUSH(a OP b)``.

        Returns ``(folded_inst, num_replaced)`` or ``None``.  ``a`` and ``b``
        are the stack values left-to-right (a is the earlier push).
        """
        if len(instrs) < 2:
            return None
        tail = instrs[-1]
        spec = _FOLDERS.get(tail.opcode)
        if spec is None:
            return None
        arity, fn, result_type = spec
        if arity > len(instrs) - 1:
            return None
        values: list[object] = []
        for prev in instrs[-(arity + 1): -1]:
            if prev.opcode is not Op.OP_PUSH_CONST or not isinstance(prev.operand, (int, float)):
                return None
            values.append(prev.operand)
        try:
            raw: object = fn(values)
        except (TypeError, OverflowError, ZeroDivisionError):
            return None
        # The folded literal must remain a plain int (wobble constants are
        # ints); booleans/floats are coerced or refused.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        folded_value = int(raw)
        folded = IRInst(opcode=Op.OP_PUSH_CONST, operand=folded_value,
                        value_type=result_type, line=tail.line,
                        codon_index=tail.codon_index)
        return folded, arity + 1

    # -------- dead instruction elimination --------
    @staticmethod
    def _pass_dead(instrs: list[IRInst]) -> list[IRInst]:
        out: list[IRInst] = []
        for inst in instrs:
            if inst.opcode is Op.OP_POP and out and out[-1].opcode in (
                    Op.OP_PUSH_CONST, Op.OP_DUP, Op.OP_READ_MEM):
                out.pop()
                continue
            if inst.opcode is Op.OP_SWAP and out and out[-1].opcode is Op.OP_SWAP:
                out.pop()
                continue
            out.append(inst)
        return out

    # -------- unreachable elimination --------
    @staticmethod
    def _pass_unreachable(instrs: list[IRInst]) -> list[IRInst]:
        for i, inst in enumerate(instrs):
            if inst.opcode in (Op.OP_HALT, Op.OP_RETURN):
                return instrs[: i + 1]
        return instrs


def optimize_program(program: IRProgram, *,
                     passes: Iterable[str] | None = None) -> IRProgram:
    return IROpt().optimize(program, passes=passes)
