"""Tests for doc/41 Item 5 — physical units in the static-semantic layer.

Ring 1 (expression-level dimension inference): ``#quantity name=T expr=A+B``
is a compile-time composition; incompatible dimensions fail
``SemanticAnalyzer.check()`` with a :class:`DimensionError` naming both
dimension trees. Ring 2 (named-unit round-trip): ``units.Q`` and
``units.declare_unit`` make ``#config``-style quantities convert exactly.
Ring 3 (runtime guard wiring): the PBPK hot path hoists its dimension checks
out of the tight loop into :meth:`_DrugPBPK.verify_units` / ``check_dimension``,
raising :class:`UnitError` on a mis-labelled parameter, bit-identical otherwise.
"""
from __future__ import annotations

import pytest

from helixlang.core.dim_inferencer import (
    DimInferencer,
    parse_quantity_expr,
)
from helixlang.core.dimensions import (
    DIM_MASS,
    DIM_TIME,
    DIM_VOLUME,
    Quantity,
    UnitError,
)
from helixlang.core.errors import DimensionError, SemanticError
from helixlang.core.parser import parse_source
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.units import Q, declare_unit
from helixlang.plugins.human.drug import get_predefined_drug
from helixlang.plugins.human.physiology import create_default_physiology
from helixlang.plugins.human.virtual_patient import _DrugPBPK


def _compile(src: str) -> None:
    SemanticAnalyzer(parse_source(src)).check()


BASIC = ("#config ticks=100 ops_per_tick=100\n"
         "#gene name=g\nATG TAA\n#end\n")


class TestDimInferencer:
    def test_cross_unit_composition_rejected(self):
        bad = (BASIC +
               "#type g=Float<µM>\n#type v=Float<L>\n"
               "#quantity c_total=g+v\n")
        with pytest.raises(DimensionError) as exc:
            _compile(bad)
        msg = str(exc.value)
        assert "incompatible dimensions" in msg
        assert "length" in msg and "amount" in msg

    def test_same_dim_composition_passes(self):
        good = (BASIC +
                "#type g=Float<µM>\n#type h=Float<µM>\n"
                "#quantity total=g+h\n"
                "#quantity delta=g-h\n")
        _compile(good)

    def test_concentration_volume_subtraction_rejected(self):
        bad = (BASIC +
               "#type g=Float<µM>\n#type v=Float<µm3>\n"
               "#quantity d=g-v\n")
        with pytest.raises(DimensionError):
            _compile(bad)

    def test_unknown_symbol_names_the_symbol(self):
        bad = (BASIC + "#quantity x=foo+bar\n")
        with pytest.raises(SemanticError, match="foo"):
            _compile(bad)

    def test_malformed_expr_rejected_at_parse(self):
        from helixlang.core.errors import ParseError

        with pytest.raises(ParseError):
            parse_source(BASIC + "#quantity x=g+v+w\n")

    def test_inferencer_requires_program(self):
        with pytest.raises(TypeError):
            DimInferencer({"not": "a program"}).infer()

    def test_quantity_expr_parser(self):
        assert parse_quantity_expr("g+v") == ("g", "+", "v")
        assert parse_quantity_expr("a - 3") == ("a", "-", "3")
        assert parse_quantity_expr("x+42.5") == ("x", "+", "42.5")


class TestQuantityRing2:
    def test_q_factory_minutes_seconds(self):
        five_min = Q("min", 5)
        assert five_min.convert_to("s") == Quantity(300, "s")
        assert five_min.convert_to("s").value == 300.0

    def test_declare_unit_round_trip(self):
        declare_unit("mo", DIM_TIME, 30.0 * 24 * 3600)
        assert Quantity(1, "mo").convert_to("min").value == 43200.0

    def test_declare_unit_conflict_is_hard(self):
        from helixlang.core.dimensions import UnitError

        with pytest.raises(UnitError):
            declare_unit("min", DIM_TIME, 1.0)

    def test_litre_unit_registered(self):
        from helixlang.core.dimensions import dim_of_unit

        assert dim_of_unit("L").tree() == dim_of_unit("L").tree()
        assert Quantity(1, "L").convert_to("µm3").value == 1e15


class TestConfigQuantityRing2:
    """doc/41 Item 5 Ring 2: ``#config`` values carry a Quantity that rounds
    trip and converts to a fixed SI basis (5 min -> 300 s)."""

    CONFIG_BASIC = ("#config ticks=100\n"
                    "#gene name=g\nATG TAA\n#end\n")

    def test_unit_tagged_config_becomes_quantity(self):
        from helixlang.core.parser import parse_source

        prog = parse_source(self.CONFIG_BASIC +
                            "#config sim dt=5min\n")
        assert "dt" in prog.config.quantities
        assert prog.config.quantity("dt", "s").value == 300.0

    def test_hour_conversion(self):
        from helixlang.core.parser import parse_source

        prog = parse_source(self.CONFIG_BASIC +
                            "#config sim hold=2h\n")
        assert prog.config.quantity("hold", "min").value == 120.0

    def test_plain_values_not_treated_as_quantity(self):
        from helixlang.core.parser import parse_source

        prog = parse_source(self.CONFIG_BASIC +
                            "#config sim backend=gem dose=40\n")
        assert prog.config.quantities == {}

    def test_unknown_unit_skipped_without_error(self):
        from helixlang.core.parser import parse_source

        # ``q=5parsec`` is not a known unit -> left as a plain sim string.
        prog = parse_source(self.CONFIG_BASIC + "#config sim q=5parsec\n")
        assert prog.config.sim["q"] == "5parsec"
        assert "q" not in prog.config.quantities

    def test_config_quantity_round_trips(self):
        from helixlang.core.hxbc import decompile
        from helixlang.core.parser import parse_source
        from helixlang.core.semantic import SemanticAnalyzer

        src = (self.CONFIG_BASIC + "#config sim dt=5min hold=2h\n")
        out = decompile(parse_source(src))
        assert "dt=5min" in out and "hold=2h" in out
        prog2 = parse_source(out)
        SemanticAnalyzer(prog2).check()
        assert prog2.config.quantity("dt", "s").value == 300.0

    def test_new_units_registered(self):
        from helixlang.core.dimensions import parse_quantity

        assert parse_quantity("1wk").convert_to("d").value == pytest.approx(7.0)
        assert parse_quantity("1mg").convert_to("µg").value == pytest.approx(1000.0)
        assert parse_quantity("500ng").convert_to("pg").value == pytest.approx(5e5)


class TestQuantityGrammarRoundTrip:
    def test_hxbc_round_trip_preserves_quantity(self):
        from helixlang.core.hxbc import decompile

        good = (BASIC +
                "#type g=Float<µM>\n#type h=Float<µM>\n"
                "#quantity total=g+h\n")
        prog = parse_source(good)
        out = decompile(prog)
        assert "#quantity name=total expr=g+h" in out
        _compile(out)

    def test_compact_and_explicit_forms(self):
        compact = parse_source(BASIC + "#quantity t=a+b\n")
        explicit = parse_source(BASIC + "#quantity name=t expr=a+b\n")
        assert compact.sim_extensions["quantity"] == \
            explicit.sim_extensions["quantity"] == \
            [{"name": "t", "expr": "a+b"}]


class TestRing3RuntimeGuard:
    """doc/41 Item 5 Ring 3 — hoisted runtime Quantity guard on the PBPK path.

    The hot stepping loop keeps plain floats (doc/39 bit-identical determinism);
    dimensional checks are lifted into ``_DrugPBPK.verify_units`` (once, at
    construction) and the public ``check_dimension`` helper, raising
    ``UnitError`` on a mis-labelled parameter.
    """

    def test_valid_config_passes_verify_units(self):
        drug = get_predefined_drug("IBUPROFEN")
        phys = create_default_physiology()
        model = _DrugPBPK(drug, phys)
        model.verify_units()  # valid units -> no exception, bit-identical state
        assert model.conc_um["central"] == 0.0

    def test_matching_unit_passes_check_dimension(self):
        _DrugPBPK.check_dimension("plasma_volume_l", 3.0, "L", DIM_VOLUME)
        _DrugPBPK.check_dimension("dose_mg", 400.0, "mg", DIM_MASS)

    def test_wrong_unit_rejected(self):
        with pytest.raises(UnitError) as exc:
            _DrugPBPK.check_dimension("plasma_volume_l", 3.0, "mg", DIM_VOLUME)
        msg = str(exc.value)
        assert "plasma_volume_l" in msg and "dimension" in msg

    def test_bioavailability_dimensionless(self):
        with pytest.raises(UnitError):
            _DrugPBPK.check_dimension(
                "bioavailability", 0.95, "L",
                Quantity(1.0, None).dim)

    def test_runtime_guard_is_readonly(self):
        drug = get_predefined_drug("IBUPROFEN")
        phys = create_default_physiology()
        a = _DrugPBPK(drug, phys)
        state_before = dict(a.conc_um), dict(a.organ_volumes_l)
        a.verify_units()
        assert (dict(a.conc_um), dict(a.organ_volumes_l)) == state_before
