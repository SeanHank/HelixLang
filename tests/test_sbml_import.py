"""Tests for SBML/BiGG model import (doc/24 Phase A, plugins/gem/sbml_import.py).

Covers the pure helpers (exchange/compartment detection, model info, vendored
candidate enumeration, offline gating) and the COBRA loader paths with a fake
``cobra`` module, so no network/real-model download is required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import helixlang.plugins.gem.sbml_import as si
from helixlang.core.errors import BioError
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction


class FakeIO:
    def load_json_model(self, path):
        return f"json:{path}"

    def read_sbml_model(self, path):
        return f"sbml:{path}"

    def load_model(self, model_id):
        return f"net:{model_id}"

    def write_sbml_model(self, model, path):
        self.written = (model, path)


def _fake_cobra(monkeypatch):
    io = FakeIO()
    cobra = SimpleNamespace(io=io)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._require_cobra",
                        lambda: cobra)
    monkeypatch.setattr(
        "helixlang.plugins.runtime.metabolism._from_cobra_model",
        lambda mod, preserve_gpr=True: ("MM", preserve_gpr),
    )
    return cobra, io


def _make_model():
    m = MetabolicModel()
    m.add_reaction(Reaction(id="EX_glc", name="x", stoichiometry={"GLC_e": 1.0},
                            subsystem="exchange"))
    m.add_reaction(Reaction(id="glyc", name="y",
                            stoichiometry={"G6P_c": -1.0, "F6P_c": 1.0},
                            subsystem="glycolysis"))
    m.add_reaction(Reaction(id="EX_extra", name="z",
                            stoichiometry={"MET_x": 1.0}, lower_bound=-1.0))
    m.add_reaction(Reaction(id="normal", name="w",
                            stoichiometry={"A_c": -1.0, "B_c": 1.0}))
    m.add_reaction(Reaction(id="SINGLE", name="v",
                            stoichiometry={"ONE_c": 1.0}, lower_bound=-1.0))
    m.metabolites = {"G6P_c", "F6P_c", "GLC_e", "A_c", "B_c", "bare",
                     "foo_cd", "met_abcd", "ONE_c"}
    m.set_biomass("glyc")
    m.genes["g1"] = SimpleNamespace(id="g1")
    return m


# ── _require_cobra / offline gating ────────────────────────────────────────────

def test_require_cobra_success_returns_module(monkeypatch):
    # Force the success branch deterministically (independent of suite state).
    sentinel = SimpleNamespace(io="fake-io")
    monkeypatch.setitem(__import__("sys").modules, "cobra", sentinel)

    def fake_import(name, *args, **kwargs):
        if name == "cobra":
            return sentinel
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert si._require_cobra() is sentinel


def test_require_cobra_raises_bioerror_when_missing(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "cobra":
            raise ImportError("no cobra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(BioError):
        si._require_cobra()


def test_offline_enabled(monkeypatch):
    monkeypatch.delenv("HELIX_BENCHMARK_OFFLINE", raising=False)
    assert si._offline_enabled() is False
    monkeypatch.setenv("HELIX_BENCHMARK_OFFLINE", "1")
    assert si._offline_enabled() is True


def test_vendored_candidates():
    from pathlib import Path
    cands = si._vendored_candidates("e_coli_core", Path("m"))
    exts = [c.suffix for c in cands]
    # tries .xml/.sbml/.json per normalized name
    assert ".xml" in exts and ".json" in exts
    names = {c.name for c in cands}
    assert "e_coli_core.xml" in names
    assert "ecolicore.xml" in names


# ── load_bigg_cobra_model (vendored / network) ─────────────────────────────────

def test_load_bigg_cobra_model_vendored_json(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    model_file = tmp_path / "e_coli_core.json"
    model_file.write_text("{}")
    out = si.load_bigg_cobra_model("e_coli_core", model_dir=str(tmp_path))
    assert out == f"json:{model_file}"


def test_load_bigg_cobra_model_vendored_xml(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    model_file = tmp_path / "e_coli_core.xml"
    model_file.write_text("<sbml/>")
    out = si.load_bigg_cobra_model("e_coli_core", model_dir=str(tmp_path))
    assert out == f"sbml:{model_file}"


def test_load_bigg_cobra_model_skips_missing_then_network(monkeypatch, tmp_path):
    cobra, io = _fake_cobra(monkeypatch)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._offline_enabled",
                        lambda: False)
    # no vendored files -> network load_model
    out = si.load_bigg_cobra_model("iML1515", model_dir=str(tmp_path))
    assert out == "net:iML1515"


def test_load_bigg_cobra_model_offline_raises(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    with pytest.raises(BioError):
        si.load_bigg_cobra_model("iML1515", model_dir=str(tmp_path), offline=True)


def test_load_bigg_cobra_model_offline_no_dir_raises(monkeypatch):
    _fake_cobra(monkeypatch)
    with pytest.raises(BioError):
        si.load_bigg_cobra_model("iML1515", model_dir=None, offline=True)


def test_load_bigg_cobra_model_vendored_parse_error_falls_to_network(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._offline_enabled",
                        lambda: False)
    # vendored json exists but fails to parse -> continue to network
    (tmp_path / "e_coli_core.json").write_text("{}")
    io = si._require_cobra().io
    def bad_json(_p):
        raise ValueError("bad json")
    io.load_json_model = bad_json
    out = si.load_bigg_cobra_model("e_coli_core", model_dir=str(tmp_path))
    assert out == "net:e_coli_core"


def test_load_bigg_cobra_model_network_error(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._offline_enabled",
                        lambda: False)
    _, io = _fake_cobra(monkeypatch)
    def boom(*_a, **_k):
        raise RuntimeError("network down")
    io.load_model = boom
    with pytest.raises(BioError):
        si.load_bigg_cobra_model("iML1515", model_dir=str(tmp_path))


# ── load_bigg_benchmark_model ──────────────────────────────────────────────────

def test_benchmark_exact_vendored(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    f = tmp_path / "e_coli_core.xml"
    f.write_text("<sbml/>")
    mod, src = si.load_bigg_benchmark_model("e_coli_core", model_dir=str(tmp_path))
    assert src == "exact"
    assert mod == f"sbml:{f}"


def test_benchmark_network_exact(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._offline_enabled",
                        lambda: False)
    mod, src = si.load_bigg_benchmark_model("iML1515", model_dir=str(tmp_path),
                                            fallback_id="nope")
    assert src == "exact"
    assert mod == "net:iML1515"


def test_benchmark_falls_back_to_vendored(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    io = si._require_cobra().io
    calls = {"n": 0}
    def flaky_load(mid):
        calls["n"] += 1
        if mid == "iML1515":
            raise RuntimeError("unavailable")
        return f"net:{mid}"
    io.load_model = flaky_load
    f = tmp_path / "e_coli_core.xml"
    f.write_text("<sbml/>")
    mod, src = si.load_bigg_benchmark_model(
        "iML1515", model_dir=str(tmp_path), fallback_id="e_coli_core")
    assert src == "fallback"
    assert "sbml:" in mod or "net:" in mod


def test_benchmark_no_model_dir_and_network_down_raises(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._offline_enabled",
                        lambda: False)
    io = si._require_cobra().io
    def flaky_load(mid):
        raise RuntimeError("unavailable")
    io.load_model = flaky_load
    with pytest.raises(BioError):
        si.load_bigg_benchmark_model("iML1515", model_dir=None)


def test_benchmark_allow_network_false_falls_back(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    io = si._require_cobra().io
    def flaky_load(mid):
        raise RuntimeError("unavailable")
    io.load_model = flaky_load
    f = tmp_path / "e_coli_core.xml"
    f.write_text("<sbml/>")
    mod, src = si.load_bigg_benchmark_model(
        "iML1515", model_dir=str(tmp_path), allow_network=False,
        fallback_id="e_coli_core")
    assert src == "fallback"
    assert "sbml:" in mod


def test_benchmark_no_fallback_raises(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    monkeypatch.setattr("helixlang.plugins.gem.sbml_import._offline_enabled",
                        lambda: False)
    io = si._require_cobra().io
    def flaky_load(mid):
        raise RuntimeError("unavailable")
    io.load_model = flaky_load
    with pytest.raises(BioError):
        si.load_bigg_benchmark_model("iML1515", model_dir=str(tmp_path))


# ── load_sbml_model / load_bigg_model / download ───────────────────────────────

def test_load_sbml_model_success(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    f = tmp_path / "m.sbml"
    f.write_text("<sbml/>")
    mm, preserve = si.load_sbml_model(str(f), preserve_gpr=False)
    assert mm == "MM" and preserve is False


def test_load_sbml_model_missing_file(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    with pytest.raises(BioError):
        si.load_sbml_model(str(tmp_path / "nope.xml"))


def test_load_sbml_model_parse_error(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    f = tmp_path / "m.xml"
    f.write_text("x")
    io = si._require_cobra().io
    def bad_read(_p):
        raise ValueError("bad xml")
    io.read_sbml_model = bad_read
    with pytest.raises(BioError):
        si.load_sbml_model(str(f))


def test_load_bigg_model(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    f = tmp_path / "e_coli_core.json"
    f.write_text("{}")
    mm, preserve = si.load_bigg_model("e_coli_core", model_dir=str(tmp_path))
    assert mm == "MM" and preserve is True


def test_download_bigg_model_success(monkeypatch, tmp_path):
    cobra, io = _fake_cobra(monkeypatch)
    out = tmp_path / "out.xml"
    res = si.download_bigg_model("iML1515", str(out))
    assert res == out
    assert io.written == ("net:iML1515", str(out))


def test_download_bigg_model_error(monkeypatch, tmp_path):
    _fake_cobra(monkeypatch)
    io = si._require_cobra().io
    def boom(*_a, **_k):
        raise RuntimeError("net down")
    io.load_model = boom
    with pytest.raises(BioError):
        si.download_bigg_model("iML1515", str(tmp_path / "x.xml"))


# ── pure detection helpers ─────────────────────────────────────────────────────

def test_detect_exchange_reactions():
    m = _make_model()
    ex = si.detect_exchange_reactions(m)
    assert "EX_glc" in ex      # subsystem contains exchange
    assert "EX_extra" in ex    # EX_ prefixed single-metabolite
    assert "SINGLE" in ex      # single-met with nonzero bounds (no EX_ prefix)
    assert "glyc" not in ex
    assert "normal" not in ex


def test_detect_compartments():
    m = _make_model()
    comps = si.detect_compartments(m)
    # sorted keys: bare, c, cd, e, unknown
    assert "c" in comps and "G6P_c" in comps["c"]
    assert "e" in comps and "GLC_e" in comps["e"]
    assert comps["bare"] == ["bare"]
    assert comps["unknown"] == ["met_abcd"]  # >3-char alnum suffix


def test_get_model_info():
    m = _make_model()
    info = si.get_model_info(m)
    assert info["n_reactions"] == 5
    assert info["n_metabolites"] == len(m.metabolites)
    assert info["n_genes"] == 1
    assert info["n_exchange"] == 3
    assert info["biomass_reaction"] == "glyc"
    assert "EX_glc" in info["exchange_reactions"]
