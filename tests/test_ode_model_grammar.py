"""doc/42 Phase D (RT-1): the ``#model`` / ``#ode_species`` / ``#ode_reaction``
plugin grammars live entirely in the plugin layer (doc/38 §6).

They are declared by ``helixlang.plugins.ode_model`` via ``GrammarDescriptor``
and activated by ``#use ode_model`` — with zero edits to the frozen core
keyword table.  These tests pin the four properties:
:class:`~helixlang.plugins.ode_model.__init__` provides descriptors, the
registry resolves them, ``#use`` gating works, and the annotations survive a
``.helixc`` round-trip so the ``ode_model`` backend can consume them.
"""
from __future__ import annotations

import pytest

from helixlang.core.errors import ParseError, UnknownKeywordError
from helixlang.core.grammar_registry import grammar_registry
from helixlang.core.hxbc import dumps_program, loads_program
from helixlang.core.parser import parse_source
from helixlang.core.plugin_registry import Registry


def _discover_ode() -> None:
    Registry().discover("ode_model")


ODE_SRC = """\
#use ode_model
#model name="lotka" k1=0.1 k2=0.05 t_end=10 steps=100
#ode_species name="A" initial=100 units="mol"
#ode_species name="B" initial=50 units="mol"
#ode_reaction species="A" expr="-k1*A"
#ode_reaction species="B" expr="k1*A-k2*B"
#gene name=reporter
ATG GCT TAA
#end
#config ticks=2 output=stdout
"""


def test_discovery_registers_all_three_keywords():
    _discover_ode()
    for kw in ("model", "ode_species", "ode_reaction"):
        assert grammar_registry.get(kw) is not None


def test_registry_and_provider_agree():
    reg = Registry()
    reg.discover("ode_model")
    for kw in ("model", "ode_species", "ode_reaction"):
        assert reg.provider_for_keyword(kw) is not None
        assert reg.provider_for_keyword(kw).name == "ode_model"


def test_grammar_inert_without_use():
    with pytest.raises(UnknownKeywordError, match="use ode_model"):
        parse_source("#ode_species name=A initial=1\n")


def test_use_activates_grammar_and_coerces_types():
    _discover_ode()
    prog = parse_source(ODE_SRC)
    ext = prog.sim_extensions
    assert ext["ode_model"][0] == {
        "name": "lotka", "k1": "0.1", "k2": "0.05",
        "t_end": "10", "steps": "100",
    }
    assert ext["ode_species"] == [
        {"name": "A", "initial": "100", "units": "mol"},
        {"name": "B", "initial": "50", "units": "mol"},
    ]
    assert ext["ode_reaction"] == [
        {"species": "A", "expr": "-k1*A"},
        {"species": "B", "expr": "k1*A-k2*B"},
    ]


def test_model_requires_name_and_k_params():
    _discover_ode()
    with pytest.raises(ParseError, match="requires name= field"):
        parse_source("#use ode_model\n#model k1=0.1 k2=0.05\n")
    with pytest.raises(ParseError, match="requires k1= field"):
        parse_source("#use ode_model\n#model name=x k2=0.05\n")


def test_species_requires_name_and_initial():
    _discover_ode()
    with pytest.raises(ParseError, match="requires initial= field"):
        parse_source("#use ode_model\n#ode_species name=A\n")
    with pytest.raises(ParseError, match="requires name= field"):
        parse_source("#use ode_model\n#ode_species initial=1\n")


def test_float_type_enforced():
    _discover_ode()
    with pytest.raises(ParseError, match="expects a float"):
        parse_source("#use ode_model\n#ode_species name=A initial=abc\n")


def test_reaction_requires_species_and_expr():
    _discover_ode()
    with pytest.raises(ParseError, match="requires expr= field"):
        parse_source("#use ode_model\n#ode_reaction species=A\n")
    with pytest.raises(ParseError, match="requires species= field"):
        parse_source("#use ode_model\n#ode_reaction expr=-k1*A\n")


def test_helixc_round_trip_preserves_ode_annotations():
    _discover_ode()
    prog = parse_source(ODE_SRC)
    blob = dumps_program(prog)
    loaded = loads_program(blob)
    ext = loaded.program.sim_extensions
    assert ext["ode_model"][0]["name"] == "lotka"
    assert len(ext["ode_species"]) == 2
    assert ext["ode_species"][0]["name"] == "A"
    assert ext["ode_species"][0]["initial"] == "100"
    assert ext["ode_reaction"][1]["expr"] == "k1*A-k2*B"


def test_authorable_ode_model_runs_end_to_end():
    """The backend can consume what the plugin grammars author (doc/42 RT-1).

    Uses the real `ode_model` backend: everything the user wrote in Helix —
    parameters, species, rate laws — reaches the RK4 integrator unchanged.
    """
    from helixlang.sim_runtime.backends.pipelines import _run_ode_model
    _discover_ode()
    prog = parse_source(ODE_SRC)
    out = _run_ode_model(prog)
    assert out.backend == "ode_model"
    assert {r["species"] for r in out.rows} == {"A", "B"}
    rows = {r["species"]: r for r in out.rows}
    assert rows["A"]["final"] > 0
    assert out.meta["model"] == "lotka"
    assert out.meta["rates"]["B"] == "k1*A-k2*B"
