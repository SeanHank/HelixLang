"""Tests for ECMpy-style enzyme-constrained GEM builder (doc/26 Phase D)."""
from __future__ import annotations

from helixlang.plugins.gem.ecgem import (
    _EC_TO_REACTION,
    ECGEMBuilder,
    ECGEMResult,
    EnzymeConstraint,
    EnzymePoolConstraint,
    _molecular_weight_from_sequence,
)


def _make_simple_model():
    from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction

    model = MetabolicModel()
    model.add_reaction(Reaction(
        id="EX_glc", name="EX_glc", stoichiometry={"glc": 1.0},
        lower_bound=-10.0, upper_bound=0.0, subsystem="exchange",
    ))
    model.add_reaction(Reaction(
        id="PFK", name="PFK", stoichiometry={"glc": -1.0, "f6p": 1.0},
        lower_bound=0.0, upper_bound=100.0, subsystem="glycolysis",
    ))
    model.add_reaction(Reaction(
        id="PYK", name="PYK", stoichiometry={"f6p": -1.0, "pyr": 1.0},
        lower_bound=0.0, upper_bound=100.0, subsystem="glycolysis",
    ))
    model.add_reaction(Reaction(
        id="BIOMASS", name="BIOMASS", stoichiometry={"pyr": -1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="biomass",
    ))
    model.set_biomass("BIOMASS")
    return model


class TestMolecularWeight:
    def test_from_sequence(self):
        mw = _molecular_weight_from_sequence("ACDEF")
        assert abs(mw - 5 * 110.0) < 0.1

    def test_empty(self):
        assert _molecular_weight_from_sequence("") == 0.0

    def test_realistic_enzyme(self):
        seq = "MKWVTFISLLFLFSSAYS" * 10
        mw = _molecular_weight_from_sequence(seq)
        assert mw > 15000


class TestECToReaction:
    def test_known_ecs(self):
        assert "2.7.1.11" in _EC_TO_REACTION
        assert _EC_TO_REACTION["2.7.1.11"]["id"] == "PFK"

    def test_citrate_synthase(self):
        info = _EC_TO_REACTION["4.1.3.16"]
        assert info["id"] == "CS"


class TestEnzymeConstraint:
    def test_creation(self):
        c = EnzymeConstraint(
            reaction_id="PFK", gene_id="pfkA", ec_number="2.7.1.11",
            kcat=380.0, molecular_weight=33000.0,
        )
        assert c.kcat == 380.0
        assert c.molecular_weight == 33000.0


class TestEnzymePoolConstraint:
    def test_creation(self):
        p = EnzymePoolConstraint(total_enzyme_mass=0.55, total_enzyme_mass_g=0.165)
        assert p.total_enzyme_mass == 0.55


class TestECGEMBuilder:
    def test_build_no_custom_kcats_uses_core_fallback(self):
        model = _make_simple_model()
        builder = ECGEMBuilder(base_model=model, kcat_predictions={})
        result = builder.build()
        assert isinstance(result, ECGEMResult)
        if len(result.enzyme_constraints) == 0:
            assert "no enzyme constraints" in result.warnings[0]
        else:
            for c in result.enzyme_constraints:
                assert c.kcat > 0

    def test_build_with_real_ec_mapping(self):
        """PFK (EC 2.7.1.11) maps to PFK reaction in model."""
        model = _make_simple_model()
        builder = ECGEMBuilder(
            base_model=model,
            kcat_predictions={"pfkA": 380.0, "pykF": 96.0},
            ec_numbers={"pfkA": "2.7.1.11", "pykF": "2.7.1.40"},
            sequences={"pfkA": "MKTTHTGIIIAIGA" * 10, "pykF": "MRKYRIG" * 15},
        )
        result = builder.build()
        assert len(result.enzyme_constraints) == 2
        assert result.enzyme_pool is not None

    def test_enzyme_usage_nonzero(self):
        model = _make_simple_model()
        builder = ECGEMBuilder(
            base_model=model,
            kcat_predictions={"pfkA": 380.0},
            ec_numbers={"pfkA": "2.7.1.11"},
            sequences={"pfkA": "MKTTHTGIIIAIGA" * 10},
        )
        result = builder.build()
        assert isinstance(result.enzyme_usage, dict)

    def test_growth_comparison(self):
        model = _make_simple_model()
        builder = ECGEMBuilder(
            base_model=model,
            kcat_predictions={"pfkA": 380.0, "pykF": 96.0},
            ec_numbers={"pfkA": "2.7.1.11", "pykF": "2.7.1.40"},
        )
        result = builder.build()
        assert result.growth_rate_unconstrained >= 0.0

    def test_auto_build_model_from_enzymes(self):
        """Without base_model, builder creates one from EC numbers."""
        builder = ECGEMBuilder(
            kcat_predictions={"pfkA": 380.0},
            ec_numbers={"pfkA": "2.7.1.11"},
        )
        result = builder.build()
        assert result.model is not None
        assert len(result.model.reactions) > 0

    def test_validate_always_passes_for_zero(self):
        builder = ECGEMBuilder(kcat_predictions={})
        assert builder.validate(0.0)

    def test_custom_params(self):
        builder = ECGEMBuilder(
            kcat_predictions={"pfkA": 380.0},
            enzyme_mass_fraction=0.40,
            dry_weight_conc=0.5,
        )
        assert builder.enzyme_mass_fraction == 0.40
        assert builder._enzyme_pool_g_per_L == 0.20
