"""``LanguageConfig`` — one object shared by Parser, Compiler, and VMs (doc/38 §4).

Kills the grammatical-truth fragmentation: stop codons were hardcoded in
``Parser`` default, start codons never existed as a set, and the amino-acid
translation tables were duplicated across two plugins.  Everything here is
*derived* from ``codon_table``; nothing is reimplemented.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from helixlang.core.codon_table import (
    Op,
    get_table,
    start_codons_from_table,
    stop_codons_from_table,
    translation_table_from_ncbi,
)


@dataclass(frozen=True)
class LanguageConfig:
    """The grammatical truth for one translation-table family.

    Attributes:
        table_name: wire-format table id ("standard" | "mito_vertebrate" |
            "ciliate"); also the string stored in ``Program.config.table``.
        codon_to_op: canonical codon -> opcode map (``get_table(table_name)``).
        stop_codons: derived ``{codon | codon_to_op[codon] == OP_HALT}``.
        start_codons: derived ``{codon | codon_to_op[codon] == OP_START}``.
        translation: derived codon -> amino-acid map (NCBI standard, single
            source — never duplicated in plugins).
    """

    table_name: str
    codon_to_op: Mapping[str, Op]
    stop_codons: frozenset[str]
    start_codons: frozenset[str]
    translation: Mapping[str, str]

    @classmethod
    def for_table(cls, table_name: str = "standard") -> LanguageConfig:
        """Build the config for a registered table, deriving every property.

        Raises:
            helixlang.core.errors.HelixError: unknown table name.
        """
        table = get_table(table_name)
        return cls(
            table_name=table_name,
            codon_to_op=table,
            stop_codons=frozenset(stop_codons_from_table(table)),
            start_codons=frozenset(start_codons_from_table(table)),
            translation=translation_table_from_ncbi(table),
        )

    @classmethod
    def standard(cls) -> LanguageConfig:
        """The default config (``LanguageConfig.for_table("standard")``)."""
        return cls.for_table("standard")

    def __repr__(self) -> str:
        return (f"LanguageConfig(table_name={self.table_name!r}, "
                f"stops={len(self.stop_codons)}, starts={len(self.start_codons)})")
