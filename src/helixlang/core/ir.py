"""Typed biological intermediate representation (HLIR).

First-class IR stage of the pipeline (doc/37 §4):

    helix source
      -> AST (Program)
      -> IRBuilder      : typed biological IR  (:mod:`ir_builder`)
      -> IROpt          : optimization         (:mod:`ir_opt`)
      -> IRLowerer      : bytecode Chunk       (:mod:`ir_lower`)
      -> runtimes       : CellVM (CPU), IRRuntime (portable CPU),
                          BatchRuntime (vectorised CPU / GPU)

The IR is a first-class abstraction because it is:

1. **Distinct** — a versioned, serialisable artifact (:mod:`ir_serialize`)
   independent of the bytecode ABI (``OPCODE_VERSION``).
2. **Typed** — every instruction that produces a runtime value carries the
   biological type of that value (gene expression level, mRNA count, protein
   count, metabolite/slot concentration, signal amount, numeric, boolean),
   so optimisation and backends can reason about the data.
3. **Faithful** — lowering reproduces the exact bytecode the legacy compiler
   emitted (same opcodes, operands, constant pool, back-patching), so the
   existing VM / C dispatch kernel keep working unchanged.
4. **Multi-consumer** — executed directly by a portable interpreter, lowered
   to bytecode for the classic VM, or compiled to a vectorised batch plan for
   the numpy/JAX runtime.

Instruction model: the IR keeps the stack-machine semantics of the source
codons (which is what makes lowering byte-faithful) and annotates each
instruction with the type of the value it produces (``value_type``) and the
bio effect it performs (selectively via its opcode).  Pure arithmetic and
stack shuffles carry stack effects from :data:`OP_POP_EFFECTS` /
:data:`OP_PUSH_EFFECTS` so the optimizer can simulate the stack without
executing side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from helixlang.core.codon_table import Op

# ── HLIR ABI version ─────────────────────────────────────────────────────
# Independent of the bytecode OPCODE_VERSION. Bump when the IR layout changes
# (new instruction fields, changed semantics), NOT when lowering changes.
IR_VERSION: int = 1


class IRType(Enum):
    """Biological + numeric value types carried by the typed IR."""

    NUM = "num"          # raw numeric constant (wobble value, integer 0..3)
    I64 = "i64"
    F64 = "f64"
    BOOL = "bool"
    GENE = "gene"        # gene expression level (0..1)
    mRNA = "mrna"        # transcript count
    PROTEIN = "protein"  # protein molecule count
    METAB = "metabolite"  # metabolite / memory-slot concentration
    SIGNAL = "signal"    # extracellular autoinducer / morphogen amount
    ENERGY = "energy"    # ATP pool
    VOID = "void"        # effect-only instruction

    @staticmethod
    def from_string(name: str) -> IRType:
        for t in IRType:
            if t.value == name:
                return t
        raise ValueError(f"unknown IRType {name!r}")

    def is_numeric(self) -> bool:
        return self in (IRType.NUM, IRType.I64, IRType.F64)

    def is_boolean(self) -> bool:
        return self is IRType.BOOL


def promote_numeric(a: IRType, b: IRType) -> IRType:
    """Widest numeric type of two operands (F64 beats I64 beats NUM)."""
    if a is IRType.F64 or b is IRType.F64:
        return IRType.F64
    if a is IRType.I64 or b is IRType.I64:
        return IRType.I64
    return IRType.NUM


# ── Stack effects of pure instructions (used by the optimizer's simulation) ──
# Effect ops (BUILD_PROTEIN, BUMP, MOVE, SIGNAL, DIVIDE, DIE, FEED,
# GROW_LSYSTEM, DIFFUSE, REACT, EMIT_MORPHOGEN, MODIFY_STATE, REGULATE,
# BIND, READ_MEM, WRITE_MEM, CALL_GENE, RETURN, HALT, boundaries) touch the
# cell/field and are never part of a fold window.

OP_PUSH_EFFECTS: dict[Op, int] = {
    Op.OP_PUSH_CONST: 1,
    Op.OP_READ_MEM: 1,
    Op.OP_NOP: 0,
    Op.OP_ADD: 1,
    Op.OP_SUB: 1,
    Op.OP_MUL: 1,
    Op.OP_LT: 1,
    Op.OP_NOT: 1,
    Op.OP_DUP: 1,  # net stack height +1 (pops 1 pushes 2)
    Op.OP_SWAP: 0,  # pops 2 pushes 2
}

OP_POP_EFFECTS: dict[Op, int] = {
    Op.OP_ADD: 2,
    Op.OP_SUB: 2,
    Op.OP_MUL: 2,
    Op.OP_LT: 2,
    Op.OP_NOT: 1,
    Op.OP_DUP: 1,
    Op.OP_SWAP: 2,
    Op.OP_WRITE_MEM: 1,
    Op.OP_REGULATE: 1,
    Op.OP_BIND: 1,
    Op.OP_JUMP_IF_ZERO: 1,
}

PURE_OPS: frozenset[Op] = frozenset({
    Op.OP_PUSH_CONST, Op.OP_POP, Op.OP_DUP, Op.OP_SWAP,
    Op.OP_ADD, Op.OP_SUB, Op.OP_MUL, Op.OP_LT, Op.OP_NOT,
    Op.OP_READ_MEM, Op.OP_WRITE_MEM, Op.OP_NOP,
})


@dataclass(slots=True)
class IRInst:
    """One typed IR instruction (stack-machine form, faithfully lowerable).

    Attributes:
        opcode: the bytecode operator this instruction lowers to.
        operand: the semantic operand value.  For ``OP_PUSH_CONST`` this is the
            constant *literal* (not its pool index); for ``OP_CALL_GENE`` the
            resolved target gene name; ``None`` for back-patched jumps and for
            operand-less opcodes.
        value_type: type of the value this instruction pushes onto the stack
            (``None`` for effect-only instructions).
        line / codon_index: source provenance for diagnostics and disassembly.
    """

    opcode: Op
    operand: int | str | None = None
    value_type: IRType | None = None
    line: int = 0
    codon_index: int = -1

    def pop_effect(self) -> int:
        return OP_POP_EFFECTS.get(self.opcode, 0)

    def push_effect(self) -> int:
        return OP_PUSH_EFFECTS.get(self.opcode, 0)

    def net_effect(self) -> int:
        return self.push_effect() - self.pop_effect()

    def is_effect(self) -> bool:
        return self.opcode not in PURE_OPS

    def __repr__(self) -> str:  # compact one-line disassembly
        op = self.opcode.name.removeprefix("OP_")
        val = "" if self.operand is None else f" {self.operand!r}"
        typ = f" :{self.value_type.value}" if self.value_type else ""
        return f"{op}{val}{typ}"


@dataclass(slots=True)
class IRFunction:
    """A gene as a typed IR function.

    ``instrs`` is the ORF's instruction stream (no implicit HALT), matching the
    source codon order.  The lowering stage re-inserts the HALT guard and the
    inter-gene jump barrier.
    """

    name: str
    instrs: list[IRInst] = field(default_factory=list)
    line: int = 1

    def __len__(self) -> int:
        return len(self.instrs)

    def disassemble(self) -> str:
        return "\n".join(f"  {i:>4}  {inst!r}" for i, inst in enumerate(self.instrs))


@dataclass(slots=True)
class IRProgram:
    """First-class typed program IR.

    Attributes:
        name: program name ('' for anonymous sources).
        table: codon table name used for lexing/parsing (``standard`` etc.).
        functions: genes in source order (``IRFunction`` per gene).
        call_targets: ``gene -> explicit call_target`` overrides (empty if none).
        use_directives: plugin opt-ins in order ``((plugin, flags), ...)``.
        lsystems: L-system declarations ``{name: (axiom, rules, angle, step)}``.
        config: snapshot of the program config dataclass.
        version: HLIR ABI version (:data:`IR_VERSION`).
    """

    name: str = ""
    table: str = "standard"
    functions: list[IRFunction] = field(default_factory=list)
    call_targets: dict[str, str] = field(default_factory=dict)
    use_directives: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    lsystems: dict[str, tuple[Any, Any, float, float]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    version: int = IR_VERSION

    def __post_init__(self) -> None:
        self.version = IR_VERSION

    def gene_names(self) -> list[str]:
        return [fn.name for fn in self.functions]

    def num_registers(self) -> int:
        """Total SSA/virtual cell registers touched by READ_MEM/MODIFY_STATE."""
        return 0  # slots are runtime cell state; kept for API symmetry

    def disassemble(self) -> str:
        lines = [f"program {self.name!r} table={self.table} v{self.version}"]
        for fn in self.functions:
            lines.append(f"gene {fn.name}:")
            lines.append(fn.disassemble())
        return "\n".join(lines)
