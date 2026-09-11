"""Closure tests for full_pipeline.py (offline-first; no ESM3/network)."""
from __future__ import annotations

import json

import pytest

from helixlang.plugins.apps.full_pipeline import (
    PipelineConfig,
    PipelineResult,
    _infer_ec_number,
    _load_ec_map,
    _run_dfba_simulation,
    _stage_a_fasta,
    _stage_b_structure,
    _stage_d_ecgem,
    _stage_e_community,
    _stage_f_simulate,
)


class TestInferEcNumber:
    def test_space_normalized_match(self):
        assert _infer_ec_number("Citrate Synthase") == "4.1.3.16"

    def test_hyphen_normalized_match(self):
        assert _infer_ec_number("pyruvate-kinase") == "2.7.1.40"

    def test_unknown_gene(self):
        assert _infer_ec_number("zzz_not_a_gene") == ""


class TestLoadEcMap:
    def test_none_path(self, tmp_path):
        assert _load_ec_map(None) == {}

    def test_missing_path(self, tmp_path):
        assert _load_ec_map(str(tmp_path / "nope.json")) == {}

    def test_valid_dict(self, tmp_path):
        p = tmp_path / "ec.json"
        p.write_text(json.dumps({"glucokinase": "2.7.1.2"}))
        assert _load_ec_map(str(p)) == {"glucokinase": "2.7.1.2"}

    def test_valid_dict_coerces_values(self, tmp_path):
        p = tmp_path / "ec.json"
        p.write_text(json.dumps({"a": 123}))
        assert _load_ec_map(str(p)) == {"a": "123"}

    def test_non_dict(self, tmp_path):
        p = tmp_path / "ec.json"
        p.write_text("[]")
        assert _load_ec_map(str(p)) == {}

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "ec.json"
        p.write_text("{broken")
        assert _load_ec_map(str(p)) == {}


class TestStageAFasta:
    def test_empty_translation_skipped(self, monkeypatch, tmp_path):
        p = tmp_path / "x.fasta"
        p.write_text(">stop\nTAA\n>ok\nMKWV\n")
        result = _stage_a_fasta(str(p))
        assert [r.gene_id for r in result] == ["ok"]

    def test_extraction_exception(self, monkeypatch, tmp_path):
        def _boom(fasta_path):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "helixlang.plugins.annotation.sequences.extract_protein_sequences",
            _boom,
        )
        assert _stage_a_fasta(str(tmp_path / "x.fasta")) == []


class TestStageBStructure:
    def test_not_available(self, monkeypatch):
        monkeypatch.setattr(
            "helixlang.plugins.runtime.protein_structure_predictor.is_available",
            lambda: False,
        )
        assert _stage_b_structure([object()], PipelineConfig()) == {}

    def test_offline_gate(self, monkeypatch):
        monkeypatch.setenv("HELIX_BENCHMARK_OFFLINE", "1")
        monkeypatch.setattr(
            "helixlang.plugins.runtime.protein_structure_predictor.is_available",
            lambda: True,
        )
        assert _stage_b_structure([object()], PipelineConfig()) == {}

    def test_loop_success_and_raise(self, monkeypatch):
        monkeypatch.delenv("HELIX_BENCHMARK_OFFLINE", raising=False)
        monkeypatch.setattr(
            "helixlang.plugins.runtime.protein_structure_predictor.is_available",
            lambda: True,
        )

        class _Protein:
            def __init__(self, gene_id, sequence):
                self.gene_id = gene_id
                self.sequence = sequence

        def _fake_predict(seq, **kwargs):
            import types
            return types.SimpleNamespace(fake=kwargs)

        def _fake_raise(seq, **kwargs):
            raise ValueError("no structure")

        def _fake(seq, **kwargs):
            return _fake_predict(seq, **kwargs) if seq != "bad" else _fake_raise(
                seq, **kwargs)

        monkeypatch.setattr(
            "helixlang.plugins.runtime.protein_structure_predictor.predict_structure_esm",
            _fake,
        )
        proteins = [
            _Protein("good", "MKWV"),
            _Protein("bad", "bad"),
            "rawstring",
        ]
        structures = _stage_b_structure(proteins, PipelineConfig(esm_model="m"))
        assert "good" in structures
        assert "bad" not in structures
        assert "protein_1" in structures


class TestStageDEcgem:
    def _pred(self, value):
        class _P:
            pass
        _P.kcat_value = value
        return _P

    def test_build_with_optional_fields(self, monkeypatch):
        captured = {}

        class _Builder:
            def __init__(self, **kw):
                captured.update(kw)

            def build(self):
                return "BUILT"

        monkeypatch.setattr("helixlang.plugins.gem.ecgem.ECGEMBuilder", _Builder)

        class _Protein:
            def __init__(self, gene_id, sequence):
                self.gene_id = gene_id
                self.sequence = sequence

        proteins = [
            _Protein("a", "MKWV"),
            _Protein("b", ""),
            _Protein("c", "ACD"),
        ]
        kcat_preds = {
            "a": self._pred(1.0),
            "b": self._pred(2.0),
            "c": None,
        }
        out = _stage_d_ecgem(
            proteins, kcat_preds, {"a": "1.2.3.4"}, PipelineConfig()
        )
        assert out == "BUILT"
        assert captured["kcat_predictions"] == {"a": 1.0, "b": 2.0}
        assert captured["sequences"] == {"a": "MKWV", "c": "ACD"}

    def test_builder_exception(self, monkeypatch):
        class _Builder:
            def build(self):
                raise RuntimeError("boom")

        monkeypatch.setattr("helixlang.plugins.gem.ecgem.ECGEMBuilder", _Builder)
        assert _stage_d_ecgem([], {}, {}, PipelineConfig()) is None


class TestStageECommunity:
    def _stubs(self, monkeypatch, solve_raises=False, reactive=True):
        class _CommunityResult:
            total_biomass = 1.5
            converged = True

        class _OrgModel:
            instances = []

            def __init__(self, **kw):
                self.args = kw
                _OrgModel.instances.append(self)

        class _CommunityExt:
            def __init__(self, organisms=None):
                self.organisms = organisms or []

            def solve(self):
                if solve_raises:
                    raise RuntimeError("community failed")
                return _CommunityResult()

        class _FBA:
            def __init__(self, model):
                self.model = model

            def solve(self, **kw):
                if reactive:
                    return {
                        "BIOMASS": 1.0,
                        "EX_glc": 0.5,
                        "EX_co2": -0.2,
                        "EX_empty": 0.1,
                        "EX_zero": 0.0,
                    }
                raise RuntimeError("solve failed")

        monkeypatch.setattr(
            "helixlang.plugins.runtime.metabolism.FluxBalanceAnalysis", _FBA)
        monkeypatch.setattr(
            "helixlang.plugins.gem.community.OrganismModel", _OrgModel)
        monkeypatch.setattr(
            "helixlang.plugins.gem.community.CommunityFBAExtended",
            _CommunityExt,
        )
        return _OrgModel

    class _Rxn:
        def __init__(self, subsystem, stoich):
            self.subsystem = subsystem
            self.stoichiometry = stoich

    def _ecgem(self, biomass_reaction="BIOMASS"):
        class _Model:
            reactions = {
                "EX_glc": self._Rxn("exchange", {"GLC": -1.0}),
                "EX_co2": self._Rxn("", {"CO2": -1.0}),
                "EX_empty": self._Rxn("exchange", {}),
                "EX_zero": self._Rxn("exchange", {"Z": -1.0}),
                "PYK": self._Rxn("glycolysis", {"ATP": 2.0}),
            }

        _Model.biomass_reaction = biomass_reaction

        class _Ecgem:
            model = _Model()
            growth_rate = 0.4
            growth_rate_unconstrained = 0.9
            enzyme_constraints = ["a"]
            warnings = []

        return _Ecgem()

    def test_summary_and_organism(self, monkeypatch):
        Org = self._stubs(monkeypatch)
        out = _stage_e_community(self._ecgem(), PipelineConfig())
        assert out.total_biomass == 1.5
        assert Org.instances[-1].args["exchange_reactions"] == [
            "EX_glc", "EX_co2", "EX_empty", "EX_zero"]
        assert Org.instances[-1].args["production"] == {"GLC": 0.5}
        assert Org.instances[-1].args["consumption"] == {"CO2": 0.2}

    def test_no_biomass_reaction(self, monkeypatch):
        Org = self._stubs(monkeypatch)
        out = _stage_e_community(self._ecgem(biomass_reaction=None),
                                 PipelineConfig())
        assert out.converged
        assert Org.instances[-1].args["exchange_reactions"] == []

    def test_solve_exception(self, monkeypatch):
        Org = self._stubs(monkeypatch, reactive=False)
        out = _stage_e_community(self._ecgem(), PipelineConfig())
        assert out.total_biomass == 1.5
        assert Org.instances[-1].args["exchange_reactions"] == []

    def test_outer_exception(self, monkeypatch):
        self._stubs(monkeypatch, solve_raises=True)
        assert _stage_e_community(self._ecgem(), PipelineConfig()) is None


class TestStageFSimulate:
    def test_ticks_zero_skips_dfba(self):
        result = PipelineResult()
        info = _stage_f_simulate(result, PipelineConfig(ticks=0))
        assert "dfba" not in info
        assert info["ec_numbers_resolved"] == 0

    def test_ecgem_and_community_fields(self):
        class _Ecgem:
            growth_rate = 0.3
            growth_rate_unconstrained = 0.8
            enzyme_constraints = ["E1"]
            warnings = ["w"]
            model = None

        class _Community:
            total_biomass = 2.0
            converged = True

        result = PipelineResult()
        result.ecgem = _Ecgem()
        result.community = _Community()
        result.stages_completed = ["A_fasta"]
        info = _stage_f_simulate(result, PipelineConfig())
        assert info["ecgem_growth_rate"] == 0.3
        assert info["community_total_biomass"] == 2.0
        assert info["dfba"] == {"status": "skipped_no_model", "ticks": 4320}


class TestRunDFBASimulation:
    def test_no_history(self, monkeypatch):
        class _DF:
            def __init__(self, **kw):
                self.kw = kw

            def run(self, **kw):
                return []

        monkeypatch.setattr(
            "helixlang.plugins.runtime.metabolism.DynamicFluxBalance", _DF)
        out = _run_dfba_simulation(object(), PipelineConfig(ticks=100))
        assert out == {"status": "no_history", "ticks": 100}

    def test_completed_path(self, monkeypatch):
        class _DF:
            def __init__(self, **kw):
                self.kw = kw

            def run(self, **kw):
                return [
                    {"time": 1.0, "biomass": 0.1, "glucose": 9.0,
                     "growth_rate": 0.2},
                    {"time": 2.0, "biomass": 0.12, "glucose": 8.5,
                     "growth_rate": 0.25},
                ]

        monkeypatch.setattr(
            "helixlang.plugins.runtime.metabolism.DynamicFluxBalance", _DF)
        out = _run_dfba_simulation(object(), PipelineConfig(ticks=100))
        assert out["status"] == "completed"
        assert out["peak_growth_rate"] == pytest.approx(0.25)
        assert out["final_biomass"] == pytest.approx(0.12)

    def test_exception(self, monkeypatch):
        class _DF:
            def __init__(self, **kw):
                raise RuntimeError("no solver")

        monkeypatch.setattr(
            "helixlang.plugins.runtime.metabolism.DynamicFluxBalance", _DF)
        out = _run_dfba_simulation(object(), PipelineConfig(ticks=50))
        assert out["status"] == "error"
        assert "no solver" in out["error"]
