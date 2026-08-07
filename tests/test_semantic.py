"""SemanticAnalyzer unit tests.

Covers src/helixlang/semantic.py:
- Valid programs pass all checks
- Duplicate symbols (promoter / gene) -> SemanticError
- Regulation source/target undefined -> RegulationError
- Gene referencing an unknown promoter -> SemanticError
- ORF validity (empty ORF, start not ATG, stop not STOP)
- Regulation cycle detection (produces only a warning, does not raise)
- Config validation (ticks/ops_per_tick/react_steps <= 0)
"""
from __future__ import annotations

import pytest

from helixlang.ast_nodes import (
    Codon,
    Config,
    Gene,
    Program,
    Promoter,
    Regulation,
)
from helixlang.errors import RegulationError, SemanticError
from helixlang.semantic import SemanticAnalyzer

# ============================================================================
# Helper functions
# ============================================================================

def _codon(seq: str, idx: int = 0, line: int = 1) -> Codon:
    return Codon(seq=seq, index=idx, line=line)


def _gene(name: str, orf_seqs: list[str], promoter: str | None = None) -> Gene:
    codons = [_codon(s, i) for i, s in enumerate(orf_seqs)]
    return Gene(name=name, promoter=promoter, codons=codons,
                orf=list(codons))


def _promoter(name: str, strength: float = 0.5) -> Promoter:
    return Promoter(name=name, strength=strength)


def _program(genes=None, promoters=None, regulations=None,
             config=None, lsystems=None, field_decl=None) -> Program:
    return Program(
        genes=genes or [],
        promoters=promoters or [],
        regulations=regulations or [],
        lsystems=lsystems or {},
        field_decl=field_decl,
        config=config if config is not None else Config(),
    )


def _check_ok(program: Program) -> list[str]:
    """Run the check and return warnings (should not raise)."""
    sa = SemanticAnalyzer(program)
    sa.check()
    return sa.warnings


def _check_raises(program: Program, exc=SemanticError):
    with pytest.raises(exc):
        SemanticAnalyzer(program).check()


# ============================================================================
# Valid programs pass
# ============================================================================

class TestValidPrograms:
    """Valid programs should pass all checks."""

    def test_minimal_valid_program(self):
        prog = _program(genes=[_gene("g", ["ATG", "GCT", "TAA"])])
        _check_ok(prog)

    def test_empty_program_passes(self):
        """An empty Program (no genes/promoters/regulations) should pass."""
        _check_ok(_program())

    def test_valid_program_with_promoter_and_regulation(self):
        prog = _program(
            promoters=[_promoter("p", -0.5)],
            genes=[_gene("g", ["ATG", "GCT", "TAA"], promoter="p")],
            regulations=[Regulation(source="p", target="g", strength=0.5)],
        )
        warnings = _check_ok(prog)
        # p->g does not form a cycle
        assert warnings == []

    def test_all_three_stop_codons_accepted(self):
        for stop in ("TAA", "TAG", "TGA"):
            prog = _program(genes=[_gene("g", ["ATG", "GCT", stop])])
            _check_ok(prog)

    def test_symbols_populated_after_check(self):
        prog = _program(
            promoters=[_promoter("p1")],
            genes=[_gene("g1", ["ATG", "TAA"])],
        )
        sa = SemanticAnalyzer(prog)
        sa.check()
        assert "p1" in sa.symbols
        assert "g1" in sa.symbols

    def test_multiple_genes_with_distinct_names(self):
        prog = _program(genes=[
            _gene("g1", ["ATG", "TAA"]),
            _gene("g2", ["ATG", "TAG"]),
            _gene("g3", ["ATG", "TGA"]),
        ])
        _check_ok(prog)


# ============================================================================
# Duplicate symbols
# ============================================================================

class TestDuplicateSymbols:
    """Duplicate symbols should raise SemanticError."""

    def test_duplicate_promoter_name(self):
        prog = _program(promoters=[
            _promoter("p", 0.5),
            _promoter("p", 0.6),
        ])
        _check_raises(prog, SemanticError)

    def test_duplicate_gene_name(self):
        prog = _program(genes=[
            _gene("g", ["ATG", "TAA"]),
            _gene("g", ["ATG", "TAG"]),
        ])
        _check_raises(prog, SemanticError)

    def test_gene_name_collides_with_promoter_name(self):
        """Promoters and genes share the same symbol table -> duplicate names should be reported."""
        prog = _program(
            promoters=[_promoter("x")],
            genes=[_gene("x", ["ATG", "TAA"])],
        )
        _check_raises(prog, SemanticError)

    def test_duplicate_error_message_mentions_symbol(self):
        prog = _program(promoters=[
            _promoter("dup"),
            _promoter("dup"),
        ])
        with pytest.raises(SemanticError) as ei:
            SemanticAnalyzer(prog).check()
        assert "dup" in str(ei.value) or "dup" in ei.value.msg


# ============================================================================
# Regulation reference integrity
# ============================================================================

class TestRegulationReferences:
    """Regulation source/target must be defined."""

    def test_unknown_source_raises_regulation_error(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            regulations=[Regulation(source="ghost", target="g", strength=0.5)],
        )
        _check_raises(prog, RegulationError)

    def test_unknown_target_raises_regulation_error(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            regulations=[Regulation(source="g", target="ghost", strength=0.5)],
        )
        _check_raises(prog, RegulationError)

    def test_both_unknown_source_and_target(self):
        prog = _program(
            regulations=[Regulation(source="a", target="b", strength=0.5)],
        )
        _check_raises(prog, RegulationError)

    def test_regulation_between_two_genes_ok(self):
        prog = _program(genes=[
            _gene("g1", ["ATG", "TAA"]),
            _gene("g2", ["ATG", "TAA"]),
        ], regulations=[Regulation(source="g1", target="g2", strength=0.5)])
        _check_ok(prog)

    def test_regulation_with_promoter_source_ok(self):
        prog = _program(
            promoters=[_promoter("p")],
            genes=[_gene("g", ["ATG", "TAA"])],
            regulations=[Regulation(source="p", target="g", strength=0.5)],
        )
        _check_ok(prog)


# ============================================================================
# Gene referencing an unknown promoter
# ============================================================================

class TestGenePromoterReference:
    """A gene's referenced promoter must be defined."""

    def test_gene_unknown_promoter_raises_semantic_error(self):
        prog = _program(genes=[
            _gene("g", ["ATG", "TAA"], promoter="missing_prom"),
        ])
        _check_raises(prog, SemanticError)

    def test_gene_known_promoter_ok(self):
        prog = _program(
            promoters=[_promoter("p")],
            genes=[_gene("g", ["ATG", "TAA"], promoter="p")],
        )
        _check_ok(prog)

    def test_gene_with_no_promoter_ok(self):
        """promoter=None means constitutive expression, so it should pass."""
        prog = _program(genes=[_gene("g", ["ATG", "TAA"], promoter=None)])
        _check_ok(prog)

    def test_unknown_promoter_error_message(self):
        prog = _program(genes=[
            _gene("g", ["ATG", "TAA"], promoter="ghost"),
        ])
        with pytest.raises(SemanticError) as ei:
            SemanticAnalyzer(prog).check()
        assert "ghost" in str(ei.value) or "ghost" in ei.value.msg


# ============================================================================
# ORF validity
# ============================================================================

class TestORFValidation:
    """An ORF must start with ATG, end with a STOP, and be non-empty."""

    def test_empty_orf_raises(self):
        g = Gene(name="g", promoter=None, codons=[], orf=[],
                 fields={})
        prog = _program(genes=[g])
        _check_raises(prog, SemanticError)

    def test_orf_not_starting_with_atg_raises(self):
        prog = _program(genes=[_gene("g", ["GCT", "TAA"])])
        _check_raises(prog, SemanticError)

    def test_orf_starting_with_gtg_raises(self):
        """GTG can serve as a start in some contexts, but this implementation requires ATG."""
        prog = _program(genes=[_gene("g", ["GTG", "GCT", "TAA"])])
        _check_raises(prog, SemanticError)

    def test_orf_not_ending_with_stop_raises(self):
        prog = _program(genes=[_gene("g", ["ATG", "GCT"])])
        _check_raises(prog, SemanticError)

    def test_orf_ending_with_gct_raises(self):
        prog = _program(genes=[_gene("g", ["ATG", "GCT", "GCT"])])
        _check_raises(prog, SemanticError)

    def test_single_atg_orf_invalid(self):
        """A single ATG codon is not a STOP -> should raise an error."""
        prog = _program(genes=[_gene("g", ["ATG"])])
        _check_raises(prog, SemanticError)

    def test_single_stop_codon_invalid(self):
        """A single TAA codon is not ATG -> should raise an error."""
        prog = _program(genes=[_gene("g", ["TAA"])])
        _check_raises(prog, SemanticError)

    def test_empty_orf_error_message_mentions_gene(self):
        g = Gene(name="mygene", promoter=None, codons=[], orf=[], fields={})
        prog = _program(genes=[g])
        with pytest.raises(SemanticError) as ei:
            SemanticAnalyzer(prog).check()
        assert "mygene" in str(ei.value) or "mygene" in ei.value.msg

    def test_bad_start_error_message(self):
        prog = _program(genes=[_gene("gg", ["GCT", "TAA"])])
        with pytest.raises(SemanticError) as ei:
            SemanticAnalyzer(prog).check()
        assert "ATG" in str(ei.value) or "ATG" in ei.value.msg


# ============================================================================
# Regulation cycle detection
# ============================================================================

class TestRegulationCycle:
    """A regulation cycle should produce a warning (without raising)."""

    def test_self_loop_produces_warning(self):
        """A self-loop g->g is a cycle."""
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            regulations=[Regulation(source="g", target="g", strength=0.5)],
        )
        sa = SemanticAnalyzer(prog)
        sa.check()
        assert any("cycle" in w.lower() for w in sa.warnings), \
            f"expected cycle warning, got {sa.warnings}"

    def test_two_node_cycle_produces_warning(self):
        """g1->g2->g1 forms a cycle."""
        prog = _program(genes=[
            _gene("g1", ["ATG", "TAA"]),
            _gene("g2", ["ATG", "TAA"]),
        ], regulations=[
            Regulation(source="g1", target="g2", strength=0.5),
            Regulation(source="g2", target="g1", strength=0.5),
        ])
        sa = SemanticAnalyzer(prog)
        sa.check()
        assert any("cycle" in w.lower() for w in sa.warnings)

    def test_no_cycle_no_warning(self):
        prog = _program(genes=[
            _gene("g1", ["ATG", "TAA"]),
            _gene("g2", ["ATG", "TAA"]),
        ], regulations=[
            Regulation(source="g1", target="g2", strength=0.5),
        ])
        sa = SemanticAnalyzer(prog)
        sa.check()
        assert sa.warnings == []

    def test_long_chain_no_cycle(self):
        """g1->g2->g3->g4 has no cycle."""
        genes = [_gene(f"g{i}", ["ATG", "TAA"]) for i in range(1, 5)]
        regs = [Regulation(source=f"g{i}", target=f"g{i+1}", strength=0.5)
                for i in range(1, 4)]
        prog = _program(genes=genes, regulations=regs)
        sa = SemanticAnalyzer(prog)
        sa.check()
        assert sa.warnings == []

    def test_cycle_does_not_raise(self):
        """A cycle is only a warning; check should not raise."""
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            regulations=[Regulation(source="g", target="g", strength=0.5)],
        )
        # Should not raise
        SemanticAnalyzer(prog).check()


# ============================================================================
# Config validation
# ============================================================================

class TestConfigValidation:
    """ticks/ops_per_tick/react_steps in config must be > 0."""

    def test_default_config_ok(self):
        prog = _program(genes=[_gene("g", ["ATG", "TAA"])],
                        config=Config())
        _check_ok(prog)

    def test_zero_ticks_raises(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(ticks=0),
        )
        _check_raises(prog, SemanticError)

    def test_negative_ticks_raises(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(ticks=-5),
        )
        _check_raises(prog, SemanticError)

    def test_zero_ops_per_tick_raises(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(ops_per_tick=0),
        )
        _check_raises(prog, SemanticError)

    def test_negative_ops_per_tick_raises(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(ops_per_tick=-1),
        )
        _check_raises(prog, SemanticError)

    def test_zero_react_steps_raises(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(react_steps=0),
        )
        _check_raises(prog, SemanticError)

    def test_negative_react_steps_raises(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(react_steps=-3),
        )
        _check_raises(prog, SemanticError)

    def test_positive_config_values_ok(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(ticks=1, ops_per_tick=1, react_steps=1),
        )
        _check_ok(prog)

    def test_zero_ticks_error_message(self):
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            config=Config(ticks=0),
        )
        with pytest.raises(SemanticError) as ei:
            SemanticAnalyzer(prog).check()
        assert "ticks" in ei.value.msg.lower()


# ============================================================================
# Check ordering / multiple errors
# ============================================================================

class TestCheckOrdering:
    """Verify that check's sub-steps run in the expected order."""

    def test_duplicate_symbol_checked_before_references(self):
        """Duplicate symbols are reported first to avoid later steps handling a conflicting symbol table."""
        prog = _program(genes=[
            _gene("g", ["ATG", "TAA"]),
            _gene("g", ["ATG", "TAA"]),
        ], regulations=[
            Regulation(source="ghost", target="g", strength=0.5),
        ])
        # Should raise a duplicate-symbol error first
        _check_raises(prog, SemanticError)

    def test_references_checked_before_orfs(self):
        """Regulation references are checked before ORFs."""
        # Setup: unknown regulation reference + invalid ORF (start not ATG)
        # Expected: the regulation error is reported first (RegulationError)
        prog = _program(
            genes=[_gene("g", ["GCT", "TAA"])],
            regulations=[Regulation(source="ghost", target="g", strength=0.5)],
        )
        with pytest.raises(RegulationError):
            SemanticAnalyzer(prog).check()

    def test_warnings_collected_after_checks(self):
        """Cycle detection runs after reference checking and does not block config checking."""
        prog = _program(
            genes=[_gene("g", ["ATG", "TAA"])],
            regulations=[Regulation(source="g", target="g", strength=0.5)],
            config=Config(ticks=10, ops_per_tick=64, react_steps=1),
        )
        sa = SemanticAnalyzer(prog)
        sa.check()
        assert len(sa.warnings) == 1


# ============================================================================
# Via parsing real source code (end-to-end)
# ============================================================================

class TestSemanticFromSource:
    """Verify SemanticAnalyzer through the full Lexer/Parser pipeline."""

    def _parse(self, src):
        from helixlang.codon_table import STANDARD_TABLE, Op
        from helixlang.lexer import Lexer
        from helixlang.parser import Parser
        stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
        toks = list(Lexer(src).tokens())
        return Parser(toks, stop_codons=stop).parse()

    def test_valid_source_passes(self):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1"
        prog = self._parse(src)
        _check_ok(prog)

    def test_duplicate_gene_from_source(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#gene name=g\nATG GCT TAA\n#end\n")
        prog = self._parse(src)
        _check_raises(prog, SemanticError)

    def test_unknown_promoter_from_source(self):
        src = "#gene name=g promoter=nope\nATG GCT TAA\n#end\n"
        prog = self._parse(src)
        _check_raises(prog, SemanticError)

    def test_regulation_unknown_target_from_source(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#regulate g -> ghost strength=+0.5\n")
        prog = self._parse(src)
        _check_raises(prog, RegulationError)
