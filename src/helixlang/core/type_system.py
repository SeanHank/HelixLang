"""HelixLang type system and modularization.

Features:
- Primitive types: Protein, Signal, Float, Int, Bool
- Record types (Record)
- Module import/export
- Type checking

Based on:
- A simplified Hindley-Milner type inference
- HelixLang DSL requirements (not over-engineered)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helixlang.core.errors import SemanticError

if TYPE_CHECKING:
    from helixlang.core.ast_nodes import Program


class HelixType(enum.Enum):
    """Enum of HelixLang primitive types."""
    PROTEIN = "protein"
    SIGNAL = "signal"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    GENE = "gene"
    RECORD = "record"
    ANY = "any"


@dataclass
class TypeSignature:
    """Type signature: generic parameters + record fields."""
    name: str
    params: list[HelixType] = field(default_factory=list)
    fields: dict[str, HelixType] = field(default_factory=dict)


@dataclass
class TypedValue:
    """A value with its type."""
    value: Any
    type: HelixType


# Type annotation name -> HelixType (case-insensitive)
_TYPE_BY_NAME = {t.value.lower(): t for t in HelixType}


def parse_type_annotation(text: str) -> HelixType:
    """Parse type annotation text into a HelixType."""
    key = text.strip().lower()
    if key in _TYPE_BY_NAME:
        return _TYPE_BY_NAME[key]
    raise SemanticError(f"unknown type annotation: {text!r}")


class SymbolTable:
    """Symbol table: name -> type signature."""

    def __init__(self) -> None:
        self._symbols: dict[str, TypeSignature] = {}
        self._values: dict[str, Any] = {}

    def define(self, name: str, type: HelixType | TypeSignature,
               value: Any = None) -> None:
        """Define a symbol. type can be a HelixType or a TypeSignature (record type)."""
        if isinstance(type, TypeSignature):
            sig = type
            if sig.name != name:
                sig = TypeSignature(name=name, params=list(sig.params),
                                    fields=dict(sig.fields))
        else:
            sig = TypeSignature(name=name, params=[type])
        self._symbols[name] = sig
        if value is not None:
            self._values[name] = value

    def lookup(self, name: str) -> TypeSignature | None:
        return self._symbols.get(name)

    def get_type(self, name: str) -> HelixType | None:
        sig = self._symbols.get(name)
        if sig is None:
            return None
        if sig.fields:
            return HelixType.RECORD
        if sig.params:
            return sig.params[0]
        return HelixType.ANY

    def get_value(self, name: str) -> Any:
        return self._values.get(name)

    def get_all(self) -> dict[str, TypeSignature]:
        return dict(self._symbols)


@dataclass
class Module:
    """Module: exported symbols + imported modules."""
    name: str
    exports: set[str] = field(default_factory=set)
    symbols: SymbolTable = field(default_factory=SymbolTable)
    imported: dict[str, Module] = field(default_factory=dict)

    def import_module(self, name: str, other: Module) -> None:
        """Import another module."""
        self.imported[name] = other
        for sym in other.exports:
            sig = other.symbols.lookup(sym)
            if sig is not None:
                self.symbols.define(sym, sig, other.symbols.get_value(sym))


class TypeChecker:
    """Type checker: performs static type checking and inference on the AST."""

    def __init__(self) -> None:
        self.symbols: SymbolTable | None = None
        self.errors: list[SemanticError] = []

    def check(self, ast: object, symbols: SymbolTable) -> list[SemanticError]:
        """Type-check the AST, returning a list of errors (does not raise)."""
        self.symbols = symbols
        self.errors = []
        # Deferred import to avoid circular dependencies
        from helixlang.core.ast_nodes import Program
        if isinstance(ast, Program):
            self._check_program(ast, symbols)
        return self.errors

    def _check_program(self, prog: Program, symbols: SymbolTable) -> None:
        # Register/validate promoters
        for prom in prog.promoters:
            existing = symbols.lookup(prom.name)
            if existing is None:
                symbols.define(prom.name, HelixType.PROTEIN)
            elif symbols.get_type(prom.name) != HelixType.PROTEIN:
                self.errors.append(SemanticError(
                    f"symbol {prom.name!r} expected Protein, got "
                    f"{symbols.get_type(prom.name)}"))
        # Register/validate genes
        for g in prog.genes:
            if symbols.lookup(g.name) is None:
                symbols.define(g.name, HelixType.GENE)
            if g.promoter:
                if symbols.lookup(g.promoter) is None:
                    self.errors.append(SemanticError(
                        f"gene {g.name!r} references undefined promoter "
                        f"{g.promoter!r}"))
                elif symbols.get_type(g.promoter) != HelixType.PROTEIN:
                    self.errors.append(SemanticError(
                        f"gene {g.name!r} promoter {g.promoter!r} is not a "
                        f"Protein"))
        # Validate regulations
        for r in prog.regulations:
            if symbols.lookup(r.source) is None:
                self.errors.append(SemanticError(
                    f"regulation source {r.source!r} not defined"))
            if symbols.lookup(r.target) is None:
                self.errors.append(SemanticError(
                    f"regulation target {r.target!r} not defined"))

    def infer(self, expr: object) -> HelixType:
        """Type inference: literal/AST node -> HelixType."""
        from helixlang.core.ast_nodes import Codon
        # bool must be checked before int (bool is a subclass of int)
        if isinstance(expr, bool):
            return HelixType.BOOL
        if isinstance(expr, int):
            return HelixType.INT
        if isinstance(expr, float):
            return HelixType.FLOAT
        if isinstance(expr, str):
            return HelixType.STRING
        if isinstance(expr, TypedValue):
            return expr.type
        if isinstance(expr, Codon):
            return HelixType.GENE
        return HelixType.ANY


class ModuleLoader:
    """Module loader: loads .helix files as Modules."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._cache: dict[str, Module] = {}

    def load(self, path: str) -> Module:
        """Load a .helix file as a module."""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        if not p.suffix:
            p = p.with_suffix(".helix")
        p = p.resolve()
        key = str(p)
        if key in self._cache:
            return self._cache[key]
        text = p.read_text()
        module = self._build_module(text, p.stem)
        self._cache[key] = module
        return module

    def resolve_import(self, import_path: str) -> Module:
        """Resolve a module import."""
        return self.load(import_path)

    def _build_module(self, text: str, name: str) -> Module:
        from helixlang.core.lexer import Lexer
        from helixlang.core.parser import Parser
        tokens = list(Lexer(text).tokens())
        prog = Parser(tokens).parse()
        module = Module(name=name)
        for prom in prog.promoters:
            module.symbols.define(prom.name, HelixType.PROTEIN)
            module.exports.add(prom.name)
        for g in prog.genes:
            module.symbols.define(g.name, HelixType.GENE)
            module.exports.add(g.name)
        return module
