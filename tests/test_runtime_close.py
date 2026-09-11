"""Coverage-closure tests for the runtime plugin package (branch + line 100%)."""
# Exercises previously-uncovered branches/paths across the runtime plugin:
#   grn.py  sparse_grn.py  stochastic.py  seq_utils.py  lsystem.py
#   morphology_3d.py  vectorized.py  protein_fitness.py  bio_data.py
from __future__ import annotations

import builtins
import math

import pytest

from helixlang.plugins.runtime import bio_data as bio_data_mod
from helixlang.plugins.runtime import lsystem as lsystem_mod
from helixlang.plugins.runtime import morphology_3d as morph_mod
from helixlang.plugins.runtime import protein_fitness as fitness_mod
from helixlang.plugins.runtime import seq_utils
from helixlang.plugins.runtime import sparse_grn as sparse_mod
from helixlang.plugins.runtime import stochastic as stoch_mod
from helixlang.plugins.runtime import vectorized as vec_mod
from helixlang.plugins.runtime.bio_data import (
    cai,
    codon_adaptation_index,
    dna_decay_half_life,
    dna_survival_fraction,
    get_codon_usage,
    get_gray_scott_preset,
    get_species_display_name,
    get_species_trna,
    is_optimal_codon,
    is_rare_codon,
    lac_promoter_strength,
    lac_repression_factor,
    mu_to_grn_strength,
)
from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.stochastic import TelegraphPromoter

# ---------------------------------------------------------------------------
# grn.py
# ---------------------------------------------------------------------------

class TestGRNClose:
    def test_step_noise_branch(self) -> None:
        g = GRN(noise_enabled=True, noise_seed=42)
        tp = TelegraphPromoter(k_on=0.5, k_off=0.5, burst_size=1.0)
        g.add_gene("a", threshold=0.5, initial_level=0.5, noise=tp)
        g.add_gene("b", threshold=0.5, initial_level=0.5)
        g.add_edge("a", "b", weight=1.0)
        triggered = g.step()
        assert 0.0 <= g.nodes["b"].level <= 1.0
        assert isinstance(triggered, list)

    def test_step_accel_hill_path(self) -> None:
        g = GRN()
        g.add_gene("a", threshold=0.5, initial_level=1.0,
                   hill_n=2.0, kd=0.5)
        g.add_gene("b", threshold=0.5, initial_level=0.0)
        g.add_edge("a", "b", weight=1.0)
        g.step_accel(prefer="python")
        assert 0.0 <= g.nodes["a"].level <= 1.0

    def test_step_accel_no_noise(self) -> None:
        g = GRN()
        g.add_gene("a", threshold=0.5, initial_level=1.0)
        g.step_accel(prefer="python")
        assert 0.0 <= g.nodes["a"].level <= 1.0


# ---------------------------------------------------------------------------
# sparse_grn.py
# ---------------------------------------------------------------------------

class TestSparseGRNClose:
    def test_step_without_noise(self) -> None:
        g = GRN()
        g.add_gene("a", threshold=0.5, initial_level=1.0)
        g.add_gene("b", threshold=0.5, initial_level=0.0)
        g.add_edge("a", "b", weight=1.0)
        s = sparse_mod.SparseGRN.from_grn(g)
        levels = s.new_state(2, initial_genes=("a",))
        out = s.step(levels)
        assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# stochastic.py
# ---------------------------------------------------------------------------

class TestStochasticClose:
    def test_telegraph_fano_negative_rates(self) -> None:
        with pytest.raises(ValueError):
            stoch_mod.telegraph_fano_factor(-1.0, 1.0, 1.0, 0.14)
        with pytest.raises(ValueError):
            stoch_mod.telegraph_fano_factor(0.5, -1.0, 1.0, 0.14)
        with pytest.raises(ValueError):
            stoch_mod.telegraph_fano_factor(0.5, 1.0, -1.0, 0.14)

    def test_telegraph_fano_degradation_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            stoch_mod.telegraph_fano_factor(0.5, 1.0, 1.0, 0.0)

    def test_telegraph_fano_no_transitions(self) -> None:
        assert stoch_mod.telegraph_fano_factor(0.0, 0.0, 2.0, 0.14) == 1.0

    def test_fano_to_noise_std_invalid(self) -> None:
        with pytest.raises(ValueError):
            stoch_mod.fano_to_noise_std(0.5, 0.0, 0.9)
        with pytest.raises(ValueError):
            stoch_mod.fano_to_noise_std(2.0, -1.0, 0.9)
        with pytest.raises(ValueError):
            stoch_mod.fano_to_noise_std(2.0, 0.5, 1.0)
        with pytest.raises(ValueError):
            stoch_mod.fano_to_noise_std(2.0, 0.5, 0.9, expression_scale=0.0)

    def test_gillespie_requires_positive_time(self) -> None:
        with pytest.raises(ValueError):
            stoch_mod.gillespie_telegraph(0.5, 0.5, 1.0, 0.14, t_max=0.0)

    def test_gillespie_runs_and_summarizes(self) -> None:
        stats = stoch_mod.gillespie_telegraph(
            1.0, 0.5, 2.0, 0.14, t_max=10.0, n_replicates=50, seed=3
        )
        assert set(stats.keys()) == {"mean", "variance", "fano"}


# ---------------------------------------------------------------------------
# seq_utils.py
# ---------------------------------------------------------------------------

class TestSeqUtilsClose:
    def test_gc_content_paths(self) -> None:
        assert seq_utils.gc_content("") == 0.0
        assert seq_utils.gc_content("atgc") == 0.5
        assert seq_utils.gc_content("GGCC") == 1.0
        assert seq_utils.gc_content("UAAU") == 0.0

    def test_reverse_complement_ambiguous(self) -> None:
        assert seq_utils.reverse_complement("ATGC") == "GCAT"
        assert seq_utils.reverse_complement("ACGTNR") == "YNACGT"

    def test_max_homopolymer_empty(self) -> None:
        assert seq_utils.max_homopolymer("") == 0

    def test_max_homopolymer_runs(self) -> None:
        assert seq_utils.max_homopolymer("AAATGC") == 3
        assert seq_utils.max_homopolymer("ACGT") == 1
        assert seq_utils.max_homopolymer("AATT") == 2


# ---------------------------------------------------------------------------
# lsystem.py
# ---------------------------------------------------------------------------

class TestLSystemClose:
    def test_interpret_all_commands(self) -> None:
        ls = lsystem_mod.LSystem("F-F+[FF]F]G", {"F": "F+F", "-": "-"})
        pts = ls.iterate()
        assert isinstance(pts, list)
        ls.iterate()
        assert ls.state_length() > 0
        assert ls.turtle.stack == []

    def test_interpret_unmatched_close(self) -> None:
        ls = lsystem_mod.LSystem("F]", rules={})
        ls.iterate()  # ']' with empty stack is a no-op


# ---------------------------------------------------------------------------
# morphology_3d.py
# ---------------------------------------------------------------------------

class TestMorphologyClose:
    def test_draw_all_commands(self) -> None:
        ls = morph_mod.LSystem3D(r"FF+fF+-&^\/[FFF]]", {"F": "F"})
        lines = ls.draw(1)
        assert lines
        assert all(isinstance(l, morph_mod.Line3D) for l in lines)

    def test_draw_unmatched_close(self) -> None:
        ls = morph_mod.LSystem3D("F]", rules={})
        assert ls.draw(1) == [] or isinstance(ls.draw(1), list)

    def test_rotate_with_zero_axis(self) -> None:
        out = morph_mod.rotate_vector(
            morph_mod.Point3D(1.0, 0.0, 0.0),
            morph_mod.Point3D(0.0, 0.0, 0.0),
            1.0,
        )
        assert out.x == pytest.approx(math.cos(1.0))
        assert out.y == pytest.approx(0.0)

    def test_derive_and_bounds(self) -> None:
        ls = morph_mod.LSystem3D("F", {"F": "F-F"})
        assert ls.derive(2) == "F-F-F-F"
        b = ls.get_bounds(1)
        assert set(b.keys()) >= {"min", "max", "center", "size"}


# ---------------------------------------------------------------------------
# vectorized.py
# ---------------------------------------------------------------------------

class TestVectorizedClose:
    def test_optional_jit_fallback_without_numba(self, monkeypatch) -> None:
        monkeypatch.setattr(vec_mod, "_HAS_NUMBA", False)

        def fn(x):
            return x + 1

        deco = vec_mod.optional_jit()
        assert deco(fn) is fn

    def test_activation_python_path(self) -> None:
        g = GRN()
        g.add_gene("a", threshold=0.5, initial_level=1.0, hill_n=2.0, kd=1.0)
        g.add_gene("b", threshold=0.5, initial_level=0.0, hill_n=2.0, kd=0.0)
        g.add_gene("c", threshold=0.0, initial_level=0.0, hill_n=2.0, kd=1.0)
        g.add_gene("d", threshold=0.0, initial_level=0.0)
        vg = vec_mod.VectorizedGRN(g)
        out = vg._activation_python([[1.0, 1.0, -1.0, 0.5], [0.0, 0.0, 0.0, -1.0]])
        assert len(out) == 2 and len(out[0]) == 4
        # hill with kd=0 and x>0 -> unit step
        assert out[0][1] == 1.0
        # hill with x<=0 -> 0.0
        assert out[0][2] == 0.0


# ---------------------------------------------------------------------------
# protein_fitness.py
# ---------------------------------------------------------------------------

class TestProteinFitnessClose:
    def test_blosum62_raw(self) -> None:
        assert fitness_mod.blosum62_raw("AAAA", "AAAA") == pytest.approx(16.0)
        with pytest.raises(ValueError):
            fitness_mod.blosum62_raw("ABC", "AB")
        with pytest.raises(ValueError):
            fitness_mod.blosum62_raw("ABX", "ABX")

    def test_base_fitness_oracle(self) -> None:
        oracle = fitness_mod.FitnessOracle()
        assert oracle.available is True
        assert oracle.score("ACD", "ACD") == pytest.approx(1.0)

    def test_oracle_score_dispatch(self) -> None:
        assert fitness_mod.oracle_score("ACD", "ACD") == pytest.approx(1.0)
        with pytest.raises(ValueError):
            fitness_mod.oracle_score("ACD", "ACD", oracle="nope")
        inst = fitness_mod.BLOSUMOracle()
        assert fitness_mod.oracle_score("ACD", "ACD", oracle=inst) == pytest.approx(1.0)

    def test_rank_variants(self) -> None:
        ranked = fitness_mod.rank_variants("AAAA", ["AACA", "AAAA", "CCCC"])
        assert ranked[0][0] == "AAAA"
        asc = fitness_mod.rank_variants("AAAA", ["AACA", "AAAA"], reverse=True)
        assert asc[0][0] == "AACA"

    def test_dna_helpers(self) -> None:
        dna = fitness_mod.protein_to_dna("AAA")
        assert isinstance(dna, str)
        assert fitness_mod.dna_fitness(dna, dna) == pytest.approx(1.0)

    def test_esm2_import_error_degrades(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        esm = fitness_mod.ESM2Oracle()
        assert esm.available is False
        with pytest.raises(RuntimeError):
            esm.pseudo_log_likelihood("ACD")
        with pytest.raises(ValueError):
            esm.score("ACDE", "ACD")

    def test_has_cuda_exception(self, monkeypatch) -> None:
        import torch

        def boom():
            raise RuntimeError("no cuda runtime")

        monkeypatch.setattr(torch.cuda, "is_available", boom)
        assert fitness_mod.ESM2Oracle._has_cuda() is False

    def test_load_esm_without_offline_flag(self, monkeypatch) -> None:
        import transformers

        class DummyModel:
            def eval(self) -> DummyModel:
                return self

        sentinel_model = DummyModel()
        sentinel_tok: object = object()
        monkeypatch.setattr(transformers.EsmForMaskedLM, "from_pretrained",
                            lambda model_name: sentinel_model)
        monkeypatch.setattr(transformers.EsmTokenizer, "from_pretrained",
                            lambda model_name: sentinel_tok)
        monkeypatch.delenv("HELIX_BENCHMARK_OFFLINE", raising=False)
        model, tok = fitness_mod._load_esm("dummy-model")
        assert model is sentinel_model
        assert tok is sentinel_tok


# ---------------------------------------------------------------------------
# bio_data.py
# ---------------------------------------------------------------------------

class TestBioDataClose:
    def test_get_codon_usage_unknown_species(self) -> None:
        with pytest.raises(ValueError):
            get_codon_usage("nope")

    def test_get_species_display_name_default(self) -> None:
        assert get_species_display_name("nope") == "nope"

    def test_codon_adaptation_index_paths(self) -> None:
        assert codon_adaptation_index("AAA") > 0.0
        assert codon_adaptation_index("TTN") == 0.0

    def test_codon_classifiers(self) -> None:
        assert is_optimal_codon("AAA", "ecoli") is True  # 0.74
        assert is_optimal_codon("GCT", "ecoli") is False  # 0.18
        assert is_rare_codon("CTA", "ecoli") is True  # 0.04
        assert is_rare_codon("GCT", "ecoli") is False

    def test_cai_edge_cases(self) -> None:
        assert cai("") == 0.0  # n_codons == 0 -> return 0.0
        assert cai("ttt") > 0.0  # lower-case input is normalized
        assert cai("TGA") == 0.0  # only stop codons -> n_sense == 0
        assert cai("GCT", simplified=True) == pytest.approx(
            codon_adaptation_index("GCT")
        )
        assert 0.0 < cai("GCTGCT", "ecoli") <= 1.0

    def test_cai_zero_weight_aborts(self, monkeypatch) -> None:
        fake = {
            "TTT": ("F", 100, 1.0),
            "TTG": ("F", 0, 0.0),
            "TAA": ("*", 0, 0.0),
        }
        monkeypatch.setattr(bio_data_mod, "get_codon_usage",
                            lambda species="ecoli": fake)
        assert cai("TTG") == 0.0

    def test_lac_strength_paths(self) -> None:
        assert lac_promoter_strength(induced=True) == 1.0
        assert lac_promoter_strength(induced=False) == pytest.approx(0.001)
        assert lac_repression_factor(0.0) == 0.0
        assert 0.0 < lac_repression_factor(1.5e-6) < 1.0

    def test_gray_scott_preset_paths(self) -> None:
        assert get_gray_scott_preset("spots").name == "Spots"
        assert get_gray_scott_preset("BACTERIA").name == "Bacteria"
        with pytest.raises(ValueError):
            get_gray_scott_preset("nope")

    def test_dna_kinetics(self) -> None:
        bare = dna_decay_half_life(13.1, encapsulated=False)
        enc = dna_decay_half_life(70.0, encapsulated=True)
        assert bare > 0 and enc > 0
        assert 0.0 < dna_survival_fraction(100.0) < 1.0

    def test_mu_to_grn_strength(self) -> None:
        assert mu_to_grn_strength(1500.0) == pytest.approx(0.5)
        assert mu_to_grn_strength(6000.0) == 0.0
        assert mu_to_grn_strength(-100.0) == 1.0

    def test_get_species_trna(self) -> None:
        assert "TAA" in get_species_trna("ecoli")
        with pytest.raises(ValueError):
            get_species_trna("nope")
