"""Targeted tests to close remaining coverage gaps in 13 source files.

Covers:
  - plugins/runtime/lsystem.py:55   (else branch of elif chain)
  - _accel/dispatch/backend.py:22-23 (run_many body)
  - _accel/grn_step/backend.py:62    (fallback when mod lacks step_mixed)
  - plugins/gem/organism_registry.py:48,150,155,160
  - plugins/grn/__init__.py:27,28,33,35
  - api/gem.py:15-16,21-22,27-28
  - plugins/cardiology/__init__.py:26-32 (all branches)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ============================================================================
# lsystem.py — branch [55, 41]: unknown char falls through all elifs
# ============================================================================

class TestLSystemUnknownChar:
    def test_unknown_character_silently_ignored(self):
        from helixlang.plugins.runtime.lsystem import LSystem
        ls = LSystem(axiom="F", rules={"F": "FX+F"}, angle=25, step=1.0)
        pts = ls.iterate()
        # 'X' is not in the turtle alphabet — falls through all elifs
        assert "X" in ls.state
        assert len(pts) > 0

    def test_only_unknown_chars(self):
        from helixlang.plugins.runtime.lsystem import LSystem
        ls = LSystem(axiom="XY", rules={}, angle=25, step=1.0)
        pts = ls.iterate()
        # All chars are unknown — every iteration falls through
        assert pts == []


# ============================================================================
# dispatch/backend.py — run_many body (lines 22-23)
# ============================================================================

class TestDispatchBackendRunMany:
    def test_run_many_via_backend_module(self):
        from helixlang._accel.dispatch.backend import run_many
        code = [0x20, 0, 0x20, 1, 0x92, 0x20, 2, 0x20, 3, 0x91,
                0x90, 0x21, 0x20, 4, 0x11]
        consts = [2.0, 3.0, 10.0, 4.0, 5.0]
        result = run_many(code, consts, n_cells=3)
        assert len(result) == 3
        assert all(isinstance(row, tuple) for row in result)


# ============================================================================
# grn_step/backend.py — line 62, branch [59, 62]: mod lacks step_mixed
# ============================================================================

class TestGrnStepBackendFallback:
    def test_step_mixed_fallback_when_mod_lacks_it(self, monkeypatch):
        from helixlang._accel.grn_step import backend as gs_backend

        fake_mod = MagicMock(spec=["step"])
        monkeypatch.setattr(
            gs_backend, "load_hot",
            lambda pkg, prefer=None: fake_mod,
        )
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.5]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, None]
        kds = [None, None]
        new, trig = gs_backend.step_mixed(
            levels, src, dst, weights, decays, thresholds, 0.99,
            hill_ns, kds,
        )
        # Should fall through to impl_python.step_mixed
        assert len(new) == 2
        assert isinstance(trig, list)


# ============================================================================
# organism_registry.py — lines 48, 150, 155, 160
# ============================================================================

class TestOrganismRegistry:
    def test_to_dict(self):
        from helixlang.plugins.gem.organism_registry import OrganismConfig
        cfg = OrganismConfig(
            organism_id="test_org",
            bigg_id="iTEST",
            name="Test Org",
            model_type="test",
            biomass_rxn="BIOMASS",
        )
        d = cfg.to_dict()
        assert d["organism_id"] == "test_org"
        assert d["bigg_id"] == "iTEST"
        assert d["name"] == "Test Org"
        assert d["model_type"] == "test"
        assert d["biomass_rxn"] == "BIOMASS"
        assert d["default_medium"] == "glucose_minimal"

    def test_get_organism_config_found(self):
        from helixlang.plugins.gem.organism_registry import get_organism_config
        cfg = get_organism_config("e_coli_k12")
        assert cfg is not None
        assert cfg.bigg_id == "iML1515"

    def test_get_organism_config_case_insensitive(self):
        from helixlang.plugins.gem.organism_registry import get_organism_config
        cfg = get_organism_config("  E_COLI_K12  ")
        assert cfg is not None

    def test_get_organism_config_missing(self):
        from helixlang.plugins.gem.organism_registry import get_organism_config
        assert get_organism_config("nonexistent") is None

    def test_list_supported_organisms(self):
        from helixlang.plugins.gem.organism_registry import list_supported_organisms
        orgs = list_supported_organisms()
        assert isinstance(orgs, list)
        assert "e_coli_k12" in orgs
        assert orgs == sorted(orgs)

    def test_has_full_model_true(self):
        from helixlang.plugins.gem.organism_registry import has_full_model
        assert has_full_model("e_coli_k12") is True

    def test_has_full_model_case_insensitive(self):
        from helixlang.plugins.gem.organism_registry import has_full_model
        assert has_full_model("  E_COLI_K12  ") is True

    def test_has_full_model_false(self):
        from helixlang.plugins.gem.organism_registry import has_full_model
        assert has_full_model("nonexistent") is False


# ============================================================================
# plugins/grn/__init__.py — lines 27,28,33,35
# ============================================================================

class TestGrnPluginInit:
    def test_check_numpy_import_error_returns_false(self, monkeypatch):
        import sys

        import helixlang.plugins.grn as grn_mod

        # Temporarily make numpy unimportable
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("no numpy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        # Clear cached numpy module
        saved = sys.modules.pop("numpy", None)
        try:
            assert grn_mod._check_numpy() is False
        finally:
            if saved is not None:
                sys.modules["numpy"] = saved

    def test_make_backend_returns_grn_class(self):
        from helixlang.plugins.grn import _make_backend
        from helixlang.plugins.runtime.grn import GRN
        assert _make_backend() is GRN

    def test_make_backend_with_config(self):
        from helixlang.plugins.grn import _make_backend
        from helixlang.plugins.runtime.grn import GRN
        assert _make_backend({"some": "config"}) is GRN

    def test_load_returns_make_backend(self):
        from helixlang.plugins.grn import _load
        backend = _load()
        assert callable(backend)


# ============================================================================
# api/gem.py — lines 15-16, 21-22, 27-28
# ============================================================================

class TestApiGem:
    def test_add_gem_core_reactions(self, monkeypatch):
        import helixlang.sim_runtime as sr
        sentinel = MagicMock(return_value="core_result")
        monkeypatch.setattr(sr, "_add_gem_core_reactions", sentinel)
        from helixlang.api.gem import add_gem_core_reactions
        result = add_gem_core_reactions("a", x=1)
        sentinel.assert_called_once_with("a", x=1)
        assert result == "core_result"

    def test_add_gem_transport_reactions(self, monkeypatch):
        import helixlang.sim_runtime as sr
        sentinel = MagicMock(return_value="transport_result")
        monkeypatch.setattr(sr, "_add_gem_transport_reactions", sentinel)
        from helixlang.api.gem import add_gem_transport_reactions
        result = add_gem_transport_reactions("b", y=2)
        sentinel.assert_called_once_with("b", y=2)
        assert result == "transport_result"

    def test_set_gem_medium(self, monkeypatch):
        import helixlang.sim_runtime as sr
        sentinel = MagicMock(return_value="medium_result")
        monkeypatch.setattr(sr, "_set_gem_medium", sentinel)
        from helixlang.api.gem import set_gem_medium
        result = set_gem_medium("c", z=3)
        sentinel.assert_called_once_with("c", z=3)
        assert result == "medium_result"


# ============================================================================
# cardiology/__init__.py — lines 26-32, branches
# ============================================================================

class TestCardiologyValidation:
    def _make_program(self, entries):
        prog = MagicMock()
        prog.sim_extensions = {"cardiac_cycle": entries} if entries else None
        return prog

    def test_validate_non_list_entries(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program("not_a_list")
        _validate_cardiac_cycle(None, prog)

    def test_validate_none_extensions(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = MagicMock()
        prog.sim_extensions = None
        _validate_cardiac_cycle(None, prog)

    def test_validate_missing_extensions(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = MagicMock()
        prog.sim_extensions = {}
        _validate_cardiac_cycle(None, prog)

    def test_validate_valid_period(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"period": "0.8", "conduction": "normal"}])
        _validate_cardiac_cycle(None, prog)

    def test_validate_period_at_boundary(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"period": "5.0"}])
        _validate_cardiac_cycle(None, prog)

    def test_validate_period_too_high_raises(self):
        from helixlang.core.errors import SemanticError
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"period": "10.0"}])
        with pytest.raises(SemanticError, match="outside the physiological range"):
            _validate_cardiac_cycle(None, prog)

    def test_validate_period_zero_raises(self):
        from helixlang.core.errors import SemanticError
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"period": "0.0"}])
        with pytest.raises(SemanticError, match="outside the physiological range"):
            _validate_cardiac_cycle(None, prog)

    def test_validate_period_negative_raises(self):
        from helixlang.core.errors import SemanticError
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"period": "-1.0"}])
        with pytest.raises(SemanticError, match="outside the physiological range"):
            _validate_cardiac_cycle(None, prog)

    def test_validate_missing_period_key(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"conduction": "normal"}])
        _validate_cardiac_cycle(None, prog)

    def test_validate_bad_period_value(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([{"period": "not_a_number"}])
        _validate_cardiac_cycle(None, prog)

    def test_validate_none_entry(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([None])
        _validate_cardiac_cycle(None, prog)

    def test_validate_empty_entries_list(self):
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        prog = self._make_program([])
        _validate_cardiac_cycle(None, prog)

    def test_validate_multiple_entries_mixed(self):
        from helixlang.core.errors import SemanticError
        from helixlang.plugins.cardiology import _validate_cardiac_cycle
        entries = [
            {"period": "0.5"},
            {"period": "bad"},
            {"period": "10.0"},
        ]
        prog = self._make_program(entries)
        with pytest.raises(SemanticError):
            _validate_cardiac_cycle(None, prog)
