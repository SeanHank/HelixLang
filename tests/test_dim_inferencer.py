"""Tests for doc/41 Item 5 — physical units in the static-semantic layer.

Ring 1 (expression-level dimension inference): ``#quantity name=T expr=A+B``
is a compile-time composition; incompatible dimensions fail
``SemanticAnalyzer.check()`` with a :class:`DimensionError` naming both
dimension trees. Ring 2 (named-unit round-trip): ``units.Q`` and
``units.declare_unit`` make ``#config``-style quantities convert exactly.
"""
from __future__ import annotations

import pytest

from helixlang.core.dim_inferencer import (
    DimInferencer,
    parse_quantity_expr,
)
from helixlang.core.dimensions import DIM_TIME, Quantity
from helixlang.core.errors import DimensionError, SemanticError
from helixlang.core.parser import parse_source
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.units import Q, declare_unit


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
