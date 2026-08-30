"""Phase E3: backend dispatch through the BackendRegistry (doc/38 §6.5).

The ``elif backend == ...`` chain and the private ``_SIM_BACKENDS`` are gone;
``_engine.run`` resolves every backend (``#config backend=`` **and**
``#sim kind=``) through ``sim_runtime.backends``.  Behavior is preserved
except: ``#sim kind=...`` resolves uniformly even when it names a
first-class backend, and unknown names keep raising ``SimConfigError``.
"""
from __future__ import annotations

import pytest

from helixlang.core.errors import SimConfigError
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.sim_runtime import _engine, run
from helixlang.sim_runtime.backends import CORE_BACKENDS, get_backend_registry
from helixlang.sim_runtime.backends.core import CORE_IMPL_ATTRS


def parse(src: str):
    return Parser(list(Lexer(src).tokens()),
                  config=LanguageConfig.for_table("standard")).parse()


def test_sim_backends_table_is_gone():
    assert not hasattr(_engine, "_SIM_BACKENDS")
    import helixlang.sim_runtime as pkg
    assert "_SIM_BACKENDS" not in pkg.__all__


def test_classic_and_default_return_none():
    assert run(parse("#config backend=classic")) is None
    assert run(parse("#config ticks=4")) is None


def test_unknown_backend_still_raises_sim_config_error():
    with pytest.raises(SimConfigError, match="unknown backend"):
        run(parse("#config backend=transcriptomics"))


def test_registry_mirrors_legacy_dispatch_names():
    reg = get_backend_registry()
    assert set(reg.ids()) == set(CORE_IMPL_ATTRS)
    # first-class backends are ids; long-tail kinds are ids + kind aliases
    assert reg.has(backend="population")
    assert reg.has(backend="fba")
    assert reg.has(kind="stochastic")
    assert reg.has(kind="ecosystem")
    assert reg.has(backend="ecosystem")
    assert set(reg.resolve(kind="ecosystem").kinds) == {"ecosystem"}


def test_kind_dispatch_runs_without_config_backend():
    result = run(parse("#config ticks=4\n#sim kind=stochastic\n"))
    assert result is not None and result.backend == "stochastic"


def test_kind_wins_over_config_backend():
    result = run(parse("#config backend=fba ticks=4\n#sim kind=stochastic\n"))
    assert result is not None and result.backend == "stochastic"


def test_kind_wins_over_explicit_backend_argument():
    prog = parse("#config ticks=4\n#sim kind=stochastic\n")
    result = run(prog, backend="human")
    assert result is not None and result.backend == "stochastic"


def test_backend_name_argument_beats_unset_kind():
    prog = parse("#config ticks=4\n")
    result = run(prog, backend="stochastic")
    assert result is not None and result.backend == "stochastic"


def test_all_backends_are_backend_instances_and_unique_ids():
    reg = get_backend_registry()
    assert len({b.id for b in CORE_BACKENDS}) == len(CORE_BACKENDS)
    assert reg.ids() == sorted(b.id for b in CORE_BACKENDS)
    for b in CORE_BACKENDS:
        assert isinstance(b.kinds, tuple) and b.id == b.kinds[0] \
            if b.kinds else True


def test_provenance_uses_resolved_backend_id():
    result = run(parse("#config ticks=4\n#sim kind=stochastic\n"))
    assert result is not None and result.provenance["backend"] == "stochastic"
