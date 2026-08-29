"""IRBuilder: AST (Program) -> typed biological IR (IRProgram).

Faithful construction: the IR instruction stream mirrors the legacy
compiler's ORF emission exactly (same opcode per codon, same operand bytes),
so lowering (:mod:`ir_lower`) yields a byte-identical Chunk.  The only
semantic lifts applied here are:

- ``OP_PUSH_CONST`` operands are the constant *literal* (the wobble value)
  instead of a constant-pool index — the IR carries values, not pool
  positions, and the lowerer re-adds them to the pool.
- ``OP_CALL_GENE`` operands are the *resolved target gene name* (honouring
  ``#gene call_target=<name>``, falling back to wobble % n selection exactly
  as the legacy compiler did).  The lowerer back-patches final offsets.
"""
from __future__ import annotations

from collections.abc import Callable

from helixlang.core.ast_nodes import Program
from helixlang.core.codon_table import Op, wobble
from helixlang.core.errors import CompileError
from helixlang.core.ir import IRFunction, IRInst, IRProgram, IRType


def _type_of(op: Op) -> IRType | None:
    """The stack-value type produced by a codon-mapped opcode (if any)."""
    if op is Op.OP_PUSH_CONST:
        return IRType.NUM
    if op is Op.OP_READ_MEM:
        return IRType.METAB
    return None


class IRBuilder:
    """Build the typed IR from a parsed AST program."""

    def __init__(self, table: dict[str, Op] | None = None):
        self._table = table

    def use_table(self, table: dict[str, Op]) -> IRBuilder:
        """Select the codon table (mutates builder; returned for chaining)."""
        self._table = table
        return self

    def build(self, program: Program, table_name: str = "standard") -> IRProgram:
        table = self._table
        if table is None:
            from helixlang.core.codon_table import STANDARD_TABLE

            table = STANDARD_TABLE

        ir = IRProgram(
            name=getattr(program, "name", ""),
            table=table_name,
            call_targets={g.name: str(g.fields.get("call_target"))
                          for g in program.genes if g.fields.get("call_target")},
            use_directives=[(d.plugin, tuple(d.flags))
                            for d in program.use_directives],
            config=_snapshot_config(program),
        )
        for lname, decl in program.lsystems.items():
            ir.lsystems[lname] = (decl.axiom, decl.rules, decl.angle, decl.step)

        resolver = _call_target_resolver(program, table, ir.call_targets)

        for gene in program.genes:
            instrs: list[IRInst] = []
            for codon in gene.orf:
                if codon.seq not in table:
                    raise CompileError(
                        f"unknown codon {codon.seq!r} (table has "
                        f"{len(table)} entries)",
                        line=codon.line, codon_index=codon.index)
                op = table[codon.seq]
                if op is Op.OP_CALL_GENE:
                    target = resolver(gene.name, codon.seq)
                    instrs.append(RawIR.inst(
                        op, target, None, codon.line, codon.index))
                else:
                    instrs.append(RawIR.inst(
                        op, wobble(codon.seq) if _has_operand(op) else None,
                        _type_of(op), codon.line, codon.index))
            ir.functions.append(IRFunction(name=gene.name, instrs=instrs))
        return ir


class RawIR:
    """Internal helper to construct IRInst (kept private-ish for clarity)."""

    @staticmethod
    def inst(op: Op, operand: int | str | None,
             value_type: IRType | None, line: int, codon_index: int) -> IRInst:
        return IRInst(opcode=op, operand=operand, value_type=value_type,
                      line=line, codon_index=codon_index)


def _snapshot_config(program: Program) -> dict:
    """Copy the program config into a plain dict (IR is config-agnostic)."""
    from dataclasses import asdict, is_dataclass

    cfg = program.config
    if is_dataclass(cfg):
        return {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    return dict(cfg)


def _has_operand(op: Op) -> bool:
    from helixlang.core.codon_table import OP_OPERAND_BYTES

    return OP_OPERAND_BYTES[op] > 0


def _call_target_resolver(
    program: Program,
    table: dict[str, Op],
    call_targets: dict[str, str],
) -> Callable[[str, str], str]:
    """Return a function resolving (owner_gene, codon) -> target gene name.

    Mirrors the legacy Compiler._patch_calls logic exactly: prefer the
    ``#gene call_target=<name>`` field of the owner, else select by
    ``wobble % n``.  Raises CompileError when the target is missing.
    """
    names = [g.name for g in program.genes]
    gene_by_name = {g.name: g for g in program.genes}
    by_name: dict[str, str] = call_targets

    def resolve(owner: str, codon_seq: str) -> str:
        explicit = by_name.get(owner)
        target = explicit if explicit is not None else names[wobble(codon_seq) % len(names)] if names else ""
        if target not in gene_by_name:
            raise CompileError(
                f"CALL_GENE target {target!r} not defined; "
                f"use #gene ... call_target=<name> to specify a target")
        return target

    return resolve
