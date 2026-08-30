"""Semantic analysis: symbol table, reference integrity, ORF validity, config."""
from __future__ import annotations

from typing import Any

from helixlang.core.ast_nodes import Gene, Program, Promoter
from helixlang.core.errors import RegulationError, SemanticError


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class SemanticAnalyzer:
    """Run static checks on the AST."""

    def __init__(self, program: Program, registry: Any = None):
        self.prog = program
        self.symbols: dict[str, Promoter | Gene] = {}
        self.warnings: list[str] = []
        self.registry = registry

    def check(self) -> None:
        self._collect_symbols()
        self._check_references()
        self._check_orfs()
        self._check_regulation_cycles()
        self._check_config()
        self._check_units()
        self._check_effects()
        self._check_use_directives()
        self._check_grammar_validators()

    def _check_grammar_validators(self) -> None:
        """Run every registered grammar validator (doc/38 §5).

        Plugin grammars declare a ``validate`` hook that runs here, alongside
        the core semantic checks, so domain constraints (e.g. a gene-therapy
        vector grammar requiring ``gene=``) are enforced without touching the
        core analyzer.
        """
        from helixlang.core.grammar_registry import (
            ensure_core_grammars,
            grammar_registry,
        )

        ensure_core_grammars()
        for g in grammar_registry.grammars():
            if g.validate is not None:
                g.validate(self, self.prog)

    def _collect_symbols(self) -> None:
        for p in self.prog.promoters:
            if p.name in self.symbols:
                raise SemanticError(f"duplicate symbol {p.name!r}")
            self.symbols[p.name] = p
        for g in self.prog.genes:
            if g.name in self.symbols:
                raise SemanticError(f"duplicate symbol {g.name!r}")
            self.symbols[g.name] = g

    def _check_references(self) -> None:
        for r in self.prog.regulations:
            if r.source not in self.symbols:
                raise RegulationError(
                    f"#regulate source {r.source!r} not defined")
            if r.target not in self.symbols:
                raise RegulationError(
                    f"#regulate target {r.target!r} not defined")
        for g in self.prog.genes:
            if g.promoter and g.promoter not in self.symbols:
                raise SemanticError(
                    f"#gene {g.name!r} references unknown promoter {g.promoter!r}")

    def _check_orfs(self) -> None:
        for g in self.prog.genes:
            if not g.orf:
                raise SemanticError(f"#gene {g.name!r} has empty ORF")
            if g.orf[0].seq != "ATG":
                raise SemanticError(
                    f"#gene {g.name!r} ORF must start with ATG, got {g.orf[0].seq}")
            if g.orf[-1].seq not in ("TAA", "TAG", "TGA"):
                raise SemanticError(
                    f"#gene {g.name!r} ORF must end with STOP codon, got {g.orf[-1].seq}")

    def _check_regulation_cycles(self) -> None:
        """Detect regulation cycles and issue a warning (not an error)."""
        # Simple DFS to find cycles
        graph: dict[str, list[str]] = {}
        for r in self.prog.regulations:
            graph.setdefault(r.source, []).append(r.target)

        def dfs(node: str, seen: set[str]) -> bool:
            if node in seen:
                return True
            seen.add(node)
            for nxt in graph.get(node, []):
                if dfs(nxt, seen):
                    return True
            seen.discard(node)
            return False

        for n in graph:
            if dfs(n, set()):
                self.warnings.append(f"regulation cycle detected at {n!r}")
                break

    def _check_config(self) -> None:
        c = self.prog.config
        if c.ticks <= 0:
            raise SemanticError(f"#config ticks must be > 0, got {c.ticks}")
        if c.ops_per_tick <= 0:
            raise SemanticError("#config ops_per_tick must be > 0")
        if c.react_steps <= 0:
            raise SemanticError("#config react_steps must be > 0")
        if not _is_number(c.ticks) or not float(c.ticks).is_integer():
            raise SemanticError(
                f"#config ticks must be an integer number of ticks, "
                f"got {c.ticks!r}")
        if not _is_number(c.ops_per_tick) or \
                not float(c.ops_per_tick).is_integer():
            raise SemanticError(
                f"#config ops_per_tick is dimensionless; "
                f"got {c.ops_per_tick!r}")

    def _check_units(self) -> None:
        """Dimensional typing of ``#type`` annotations (doc/38 §8.3).

        Every annotation must parse to a known type and — when it carries a
        unit (``Float<µM>``) — to a known dimension; the error names the
        offending symbol and the resolved dimension tree.
        """
        from helixlang.core.dimensions import UnitError, dim_of_unit
        from helixlang.core.type_system import parse_type_annotation

        for name, spec in self.prog.type_annotations.items():
            try:
                parse_type_annotation(spec)
            except (UnitError, SemanticError) as exc:
                raise SemanticError(
                    f"#type {name}={spec!r}: {exc}") from exc
            if "<" not in spec:
                continue
            unit = spec.split("<", 1)[1].rstrip(">").strip()
            try:
                dim = dim_of_unit(unit)
            except UnitError as exc:
                raise SemanticError(
                    f"#type {name}={spec!r}: {exc}") from exc
            if dim.dimensionless:
                raise SemanticError(
                    f"#type {name}={spec!r}: unit {unit!r} is dimensionless; "
                    f"drop the unit annotation")

    def _check_effects(self) -> None:
        """Reject side-effecting ops in declared-pure gene regions (§7.3).

        A gene region is declared pure with ``#gene ... pure=1``; effect
        typing then rejects any side-effecting opcode compiled from it (a
        mutation, signal, movement or a side-effecting read).
        """
        if not any(g.fields.get("pure") == "1" for g in self.prog.genes):
            return
        from helixlang.core.language import LanguageConfig
        from helixlang.core.type_system import TypeChecker

        errors = TypeChecker().check_effects(
            self.prog, LanguageConfig.for_table("standard").codon_to_op)
        if errors:
            raise SemanticError("; ".join(str(e) for e in errors))

    def _check_use_directives(self) -> None:
        """Validate every ``#use`` plugin name + capability flags (doc/36 §3.2).

        An unknown plugin name is a hard ``SemanticError`` (never a silent
        no-op, F10).  Each declared capability flag is recorded on the registry
        so plugin activation honours explicit opt-ins (doc/36 §3ξ.3).
        """
        if not self.prog.use_directives:
            return
        from helixlang.core.plugin_registry import get_registry

        registry = self.registry if self.registry is not None else get_registry()
        for d in self.prog.use_directives:
            if not registry.is_registered(d.plugin):
                raise SemanticError(
                    f"#use references unknown plugin {d.plugin!r} (F10); "
                    "register it or install the matching extra "
                    "(doc/36 §3ξ.2)", line=d.line, col=d.col)
            for flag in d.flags:
                registry.declare_capability(flag)
