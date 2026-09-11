"""Tests for genome-scale model validation (doc/20 §6.1, plugins/gem/validation.py).

Covers equation parsing, model building from consensus, mass-balance checking,
gene essentiality, and the top-level ``validate_model`` pipeline. FBA outcomes are
either real (small solvable models) or controlled via a patched
``FluxBalanceAnalysis``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from helixlang.plugins.gem.bottom_up import GPRRule
from helixlang.plugins.gem.consensus import ConsensusReaction, ConsensusResult
from helixlang.plugins.gem.validation import (
    _build_model_from_consensus,
    _is_numeric,
    _parse_equation_to_stoich,
    check_mass_balance,
    gene_essentiality_test,
    validate_model,
)
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction


def _rfba_model():
    # minimal solvable model: exchange produces glc, biomass consumes it
    model = MetabolicModel()
    model.add_reaction(Reaction(
        id="EX_glc", name="EX_glc", stoichiometry={"glc": 1.0},
        lower_bound=-10.0, upper_bound=0.0, subsystem="exchange"))
    model.add_reaction(Reaction(
        id="BIOMASS", name="BIOMASS", stoichiometry={"glc": -1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="biomass"))
    model.set_biomass("BIOMASS")
    return model


def _consensus(reactions):
    return ConsensusResult(reactions=reactions)


def _rxn(rid, equation, confidence=0.7, gene_ids=()):
    gpr = GPRRule(reaction_id=rid, gene_ids=list(gene_ids)) if gene_ids else None
    return ConsensusReaction(reaction_id=rid, equation=equation,
                             confidence=confidence, gpr=gpr)


# ── equation parsing ───────────────────────────────────────────────────────────

def test_parse_equation_bidirectional():
    s = _parse_equation_to_stoich("A + 2 B <=> 3 C")
    assert s["A"] == -1.0
    assert s["B"] == -2.0
    assert s["C"] == 3.0


def test_parse_equation_irreversible():
    s = _parse_equation_to_stoich("2 A -> B")
    assert s["A"] == -2.0
    assert s["B"] == 1.0


def test_parse_equation_no_arrow_returns_empty():
    assert _parse_equation_to_stoich("weird text") == {}


def test_parse_equation_handles_blank_terms():
    s = _parse_equation_to_stoich("A +   -> B  ")
    assert s["A"] == -1.0
    assert s["B"] == 1.0


def test_parse_equation_two_part_non_numeric_term():
    # "glc transport" -> 2 parts, first not numeric -> takes the fallback
    s = _parse_equation_to_stoich("-> glc transport")
    assert s == {"transport": 1.0}


def test_parse_equation_three_part_term():
    # >2 parts -> met is the last token, coefficient defaults to 1.0
    s = _parse_equation_to_stoich("-> 2 x y z")
    assert s == {"z": 1.0}


def test_is_numeric():
    assert _is_numeric("2.5") is True
    assert _is_numeric("abc") is False


# ── build model from consensus ─────────────────────────────────────────────────

def test_build_model_from_consensus():
    consensus = _consensus([
        _rxn("R1", "A -> B", confidence=0.9),
        _rxn("R2", "x y z", confidence=0.3),  # unparseable -> skipped
    ])
    model = _build_model_from_consensus(consensus)
    assert set(model.reactions) == {"R1"}
    assert model.reactions["R1"].stoichiometry == {"A": -1.0, "B": 1.0}
    # high confidence -> reversible (-1000 lower)
    assert model.reactions["R1"].lower_bound == -1000.0
    assert model.biomass_reaction == "R1"


def test_build_model_from_consensus_low_confidence_irreversible():
    consensus = _consensus([_rxn("R9", "A -> B", confidence=0.5)])
    model = _build_model_from_consensus(consensus)
    assert model.reactions["R9"].lower_bound == 0.0


def test_build_model_from_consensus_empty():
    model = _build_model_from_consensus(_consensus([]))
    assert model.biomass_reaction is None


# ── mass balance ───────────────────────────────────────────────────────────────

def test_check_mass_balance_no_violation():
    model = MetabolicModel()
    model.add_reaction(Reaction(id="R1", name="R1",
                                stoichiometry={"A_c": -1.0, "B_c": 1.0}))
    model.add_reaction(Reaction(id="R2", name="R2",
                                stoichiometry={"A_c": 1.0, "B_c": -1.0}))
    # R1 and R2 exactly cancel -> balanced
    assert check_mass_balance(model, {"R1": 5.0, "R2": 5.0}) == []


def test_check_mass_balance_reports_violations():
    model = MetabolicModel()
    model.add_reaction(Reaction(id="R1", name="R1",
                                stoichiometry={"A_c": -1.0, "B_c": 1.0}))
    # only R1 present so both its metabolites are produced/consumed
    # unbalanced -> both A_c and B_c report as violations
    viol = check_mass_balance(model, {"R1": 5.0})
    assert "A_c" in viol and "B_c" in viol


# ── gene essentiality ──────────────────────────────────────────────────────────

def test_gene_essentiality_ko_reduces_growth(monkeypatch):
    model = _rfba_model()
    consensus = _consensus([_rxn("BIOMASS", "glc -> bio", gene_ids=["geneA"])])
    state = {"n": 0}

    def factory(_m):
        state["n"] += 1
        if state["n"] == 1:  # wild-type solve
            return SimpleNamespace(solve=lambda objective=None, **k:
                                   {"BIOMASS": 1.0})
        return SimpleNamespace(solve=lambda objective=None, **k:
                               {"BIOMASS": 0.0})  # knockout solve

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        factory)
    res = gene_essentiality_test(model, consensus)
    assert res["geneA"] is True  # 0.0 / 1.0 < 0.01 -> essential


def test_gene_essentiality_non_essential(monkeypatch):
    model = _rfba_model()
    consensus = _consensus([_rxn("BIOMASS", "glc -> bio", gene_ids=["geneA"])])
    state = {"n": 0}

    def factory(_m):
        state["n"] += 1
        if state["n"] == 1:
            return SimpleNamespace(solve=lambda objective=None, **k:
                                   {"BIOMASS": 1.0})
        return SimpleNamespace(solve=lambda objective=None, **k:
                               {"BIOMASS": 0.5})

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        factory)
    res = gene_essentiality_test(model, consensus)
    assert res["geneA"] is False


def test_gene_essentiality_no_biomass_returns_empty():
    model = MetabolicModel()  # no biomass set
    assert gene_essentiality_test(model, _consensus([])) == {}


def test_gene_essentiality_fba_error_returns_empty(monkeypatch):
    model = _rfba_model()
    consensus = _consensus([_rxn("BIOMASS", "glc -> bio", gene_ids=["g"])])

    class _Boom:
        def solve(self, *a, **k):
            raise RuntimeError("solver failed")

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        lambda m: _Boom())
    assert gene_essentiality_test(model, consensus) == {}


def test_gene_essentiality_zero_wt_growth_returns_empty(monkeypatch):
    model = _rfba_model()
    consensus = _consensus([_rxn("BIOMASS", "glc -> bio", gene_ids=["g"])])

    class _Zero:
        def solve(self, *a, **k):
            return {"BIOMASS": 0.0}

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        lambda m: _Zero())
    assert gene_essentiality_test(model, consensus) == {}


def test_gene_essentiality_ko_fba_error_counts_zero(monkeypatch):
    model = _rfba_model()
    consensus = _consensus([_rxn("BIOMASS", "glc -> bio", gene_ids=["g"])])
    state = {"n": 0}

    def factory(_m):
        state["n"] += 1
        if state["n"] == 1:
            return SimpleNamespace(solve=lambda objective=None, **k:
                                   {"BIOMASS": 1.0})

        def _raise(**k):
            raise RuntimeError("ko failed")
        return SimpleNamespace(solve=_raise)

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        factory)
    res = gene_essentiality_test(model, consensus)
    assert res["g"] is True  # ko_growth treated as 0.0


def test_gene_essentiality_skips_reactions_without_gpr(monkeypatch):
    model = _rfba_model()
    # second reaction has no gpr -> excluded from gene_rxns
    consensus = _consensus([
        _rxn("BIOMASS", "glc -> bio", gene_ids=["geneA"]),
        _rxn("EX_glc", "-> glc", gene_ids=[]),
    ])
    state = {"n": 0}

    def factory(_m):
        state["n"] += 1
        if state["n"] == 1:
            return SimpleNamespace(solve=lambda objective=None, **k:
                                   {"BIOMASS": 1.0})
        return SimpleNamespace(solve=lambda objective=None, **k:
                               {"BIOMASS": 0.0})

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        factory)
    res = gene_essentiality_test(model, consensus)
    assert res == {"geneA": True}


def test_gene_essentiality_handles_gene_missing_from_model(monkeypatch):
    model = _rfba_model()
    # gene g2 gates "NOPE_REACTION" which is not in the model
    consensus = _consensus([
        _rxn("BIOMASS", "glc -> bio", gene_ids=["g1"]),
        _rxn("NOPE_REACTION", "x -> y", gene_ids=["g2"]),
    ])
    state = {"n": 0}

    def factory(_m):
        state["n"] += 1
        if state["n"] == 1:
            return SimpleNamespace(solve=lambda objective=None, **k:
                                   {"BIOMASS": 1.0})
        return SimpleNamespace(solve=lambda objective=None, **k:
                               {"BIOMASS": 0.0})

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        factory)
    res = gene_essentiality_test(model, consensus)
    assert "g1" in res and "g2" in res


# ── validate_model ─────────────────────────────────────────────────────────────

def test_validate_model_none_returns_empty():
    res = validate_model()
    assert res.validation_passed is False
    assert res.predicted_growth_rate == 0.0


def test_validate_model_builds_from_consensus():
    consensus = _consensus([_rxn("EX_glc", "-> glc", confidence=0.9),
                            _rxn("BIOMASS", "glc -> bio", confidence=0.9)])
    res = validate_model(consensus=consensus, expected_growth_rate=1.0)
    assert res.experimental_growth_rate == 1.0
    assert res.growth_rate_error is not None
    assert isinstance(res.predicted_growth_rate, float)


def test_validate_model_none_biomass_returns_early():
    model = MetabolicModel()
    model.add_reaction(Reaction(id="R1", name="R1", stoichiometry={"A": 1.0}))
    res = validate_model(model=model)
    assert res.predicted_growth_rate == 0.0


def test_validate_model_success(monkeypatch):
    model = _rfba_model()
    # balanced optimal fluxes -> positive growth, no mass-balance violations
    class _Ok:
        def solve(self, objective=None, **k):
            return {"BIOMASS": 0.5, "EX_glc": 0.5}

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        lambda m: _Ok())
    res = validate_model(model=model, expected_growth_rate=1.0)
    assert res.predicted_growth_rate == 0.5
    assert res.validation_passed is True
    assert res.mass_balance_violations == []
    assert res.blocked_reactions == []
    assert res.growth_rate_error == pytest.approx(0.5)


def test_validate_model_applies_medium_and_fba_error(monkeypatch):
    model = _rfba_model()

    class _Boom:
        def solve(self, *a, **k):
            raise RuntimeError("no solution")

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        lambda m: _Boom())
    res = validate_model(model=model, medium={"glc": -5.0})
    assert res.predicted_growth_rate == 0.0


def test_validate_model_medium_updates_exchange_bound(monkeypatch):
    model = _rfba_model()

    class _Ok:
        def solve(self, objective=None, **k):
            return {"BIOMASS": 0.5, "EX_glc": 0.5}

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        lambda m: _Ok())
    validate_model(model=model, medium={"glc": -4.0})
    assert model.reactions["EX_glc"].upper_bound == -4.0


def test_validate_model_medium_ignores_missing_exchange(monkeypatch):
    model = _rfba_model()

    class _Ok:
        def solve(self, objective=None, **k):
            return {"BIOMASS": 0.5, "EX_glc": 0.5}

    monkeypatch.setattr("helixlang.plugins.gem.validation.FluxBalanceAnalysis",
                        lambda m: _Ok())
    # 'xyz' has no EX_xyz reaction in the model -> no crash, no bound change
    res = validate_model(model=model, medium={"xyz": -2.0})
    assert res.validation_passed is True
    assert "EX_xyz" not in model.reactions
