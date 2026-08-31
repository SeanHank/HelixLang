"""Static physical-unit inference for the semantic layer (doc/41 Item 5).

Ring 1 of item 5: a program-level ``#quantity`` statement composes two
unit-carrying atoms — ``#quantity total=g+v`` where ``#type g=Float<µM>`` —
and the inferencer rejects incompatible dimensions at **compile time** with a
:class:`DimensionError` (distinct from the runtime
:class:`~helixlang.core.dimensions.UnitError`).

Symbol dimensions come from ``#type`` unit annotations (``Float<µM>`` ->
concentration), exactly the unit-carrying types the parser + semantic
``_check_units`` already resolve; a bare numeric literal is dimensionless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from helixlang.core.dimensions import DIMENSIONLESS, Dimension, dim_of_unit
from helixlang.core.errors import DimensionError, SemanticError
from helixlang.core.type_system import UnitType, parse_type_annotation

#: ``A+B`` / ``A-B`` / ``A+3`` — identifiers or (optionally signed) numbers.
#: A leading sign on the left operand is out of scope (v1 keeps the compact
#: two-atom binop form the generic field collector can express).
_QUANTITY_EXPR_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*"
    r"([A-Za-z_][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class QuantityExpr:
    """One parsed ``#quantity name=… expr=A+B`` statement."""

    name: str
    lhs: str
    op: str
    rhs: str
    line: int = 0


_NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def parse_quantity_expr(expr: str) -> tuple[str, str, str]:
    """Split ``expr`` into ``(lhs, op, rhs)`` or raise :class:`SemanticError`."""
    m = _QUANTITY_EXPR_RE.match(expr)
    if m is None:
        raise SemanticError(
            f"quantity expression must be `A+B` (symbol or number), "
            f"got {expr!r}")
    lhs, op, rhs = m.groups()
    return lhs, op, rhs


class DimInferencer:
    """Infer physical dimensions and reject incompatible compositions."""

    def __init__(self, program: object) -> None:
        from helixlang.core.ast_nodes import Program

        if not isinstance(program, Program):
            raise TypeError(f"DimInferencer expects a Program, got "
                            f"{type(program).__name__}")
        self.program = program
        self.symbol_dims: dict[str, Dimension] = {}

    def build_symbol_table(self) -> None:
        """Resolve every ``#type`` annotation to a :class:`Dimension`.

        Unknown units were already rejected at parse time (parser +
        semantic ``_check_units``), so this is a bounds check; bare numeric
        symbols and non-unit annotations are dimensionless.
        """
        for name, spec in self.program.type_annotations.items():
            parsed = parse_type_annotation(spec)
            if isinstance(parsed, UnitType):
                self.symbol_dims[name] = dim_of_unit(parsed.unit)
            else:
                self.symbol_dims[name] = DIMENSIONLESS

    def collect(self) -> list[QuantityExpr]:
        """Read the ``#quantity`` statements stored by the grammar hook."""
        exprs: list[QuantityExpr] = []
        for entry in self.program.sim_extensions.get("quantity", []):
            if not (isinstance(entry, dict) and "name" in entry
                    and "expr" in entry):
                continue
            name = str(entry["name"])
            lhs, op, rhs = parse_quantity_expr(str(entry["expr"]))
            exprs.append(QuantityExpr(name=name, lhs=lhs, op=op, rhs=rhs))
        return exprs

    def dim_of_atom(self, atom: str) -> Dimension:
        """Dimension of an operand: symbol (via ``#type``) or numeric."""
        if _NUMBER_RE.match(atom):
            return DIMENSIONLESS
        dim = self.symbol_dims.get(atom)
        if dim is None:
            raise SemanticError(
                f"#quantity references symbol {atom!r} with no #type "
                f"annotation (declare `#type {atom}=Float<unit>`)")
        return dim

    def infer(self) -> list[QuantityExpr]:
        """Run the dimension check over every ``#quantity`` statement."""
        self.build_symbol_table()
        exprs = self.collect()
        for expr in exprs:
            lhs_dim = self.dim_of_atom(expr.lhs)
            rhs_dim = self.dim_of_atom(expr.rhs)
            if lhs_dim != rhs_dim:
                raise DimensionError(
                    f"#quantity {expr.name}: {expr.op}: incompatible "
                    f"dimensions {expr.lhs!r} is "
                    f"{lhs_dim.tree()} vs {expr.rhs!r} is "
                    f"{rhs_dim.tree()} ({expr.lhs} {expr.op} {expr.rhs})")
        return exprs


def infer_dimensions(program: object) -> list[QuantityExpr]:
    """Compile-time dimension check; raises on the first mismatch."""
    return DimInferencer(program).infer()


__all__ = [
    "DimInferencer", "QuantityExpr", "infer_dimensions", "parse_quantity_expr",
]
