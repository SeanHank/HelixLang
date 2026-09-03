"""Tests for doc/40 Phase C — complement (G5) + NK/mast/eosinophil/basophil (G6).

Covers the reduced complement cascade (:mod:`helixlang.plugins.human.complement`)
and the additive G6 pools + histamine released through the innate immune model.
"""
from __future__ import annotations

import copy

import pytest

from helixlang.plugins.human.complement import (
    N_L7_PARAMS,
    ComplementCascade,
    FullL7Complement,
    cohort_complement_step,
    complement_knockout,
)
from helixlang.plugins.human.immune import InnateImmuneModel, create_immune_model
from helixlang.plugins.human.tissue_blood import (
    TissueBloodModel,
    cohort_tissue_blood_step,
)


class TestComplementBaseline:
    def test_inert_at_no_signal(self):
        c = ComplementCascade()
        for _ in range(48):
            c.step(1.0, 0.0)
        assert c.c3 == pytest.approx(1.0)
        assert c.c5 == pytest.approx(1.0)
        assert c.get_c3a() == 0.0
        assert c.get_mac() == 0.0

    def test_inert_at_baseline_in_model(self):
        m, _crp = create_immune_model()
        for _ in range(48):
            m.step(1.0)
        assert m.get_c3a() == 0.0
        assert m.get_mac() == 0.0


class TestComplementActivation:
    def test_signal_drives_opsonization_and_mac(self):
        c = ComplementCascade()
        for _ in range(24):
            c.step(1.0, 0.8)
        assert c.get_opsonization() > 0.0
        assert c.get_c3a() > 0.0
        assert c.get_c5a() > 0.0
        assert c.get_mac() > 0.0
        # C3/C5 get consumed below their 1.0 baseline
        assert c.c3 < 1.0

    def test_mac_suppressed_by_anti_c5_but_opsonization_spared(self):
        placebo = ComplementCascade()
        for _ in range(24):
            placebo.step(1.0, 0.8)
        treated = ComplementCascade()
        treated.anti_c5_dose = 1.0
        for _ in range(24):
            treated.step(1.0, 0.8)
        assert treated.get_mac() < placebo.get_mac() * 0.1
        assert treated.get_opsonization() > 0.0

    def test_model_level_anti_c5(self):
        m, _crp = create_immune_model()
        m.complement.anti_c5_dose = 1.0
        m.infection_severity = 0.8
        for _ in range(24):
            m.step(1.0)
        assert m.get_mac() > 0.0


class TestG6Pools:
    def test_baseline_inert(self):
        m, _crp = create_immune_model()
        for _ in range(48):
            m.step(1.0)
        assert m.get_histamine() == 0.0
        assert m.get_nk_cells() > 0.0
        assert m.get_mast_cells() > 0.0

    def test_nk_rises_with_innate_signal(self):
        m, _crp = create_immune_model()
        m.infection_severity = 0.8
        for _ in range(24):
            m.step(1.0)
        assert m.get_nk_cells() > 0.25

    def test_ige_drive_releases_histamine(self):
        m = InnateImmuneModel()
        m.cells.igE_signal = 1.0
        peak = 0.0
        for _ in range(24):
            m.step(1.0)
            peak = max(peak, m.get_histamine())
        assert peak > 5.0

    def test_histamine_clears_after_signal_removed(self):
        m = InnateImmuneModel()
        m.cells.igE_signal = 1.0
        for _ in range(6):
            m.step(1.0)
        m.cells.igE_signal = 0.0
        for _ in range(24):
            m.step(1.0)
        assert m.get_histamine() < 5.0


class TestTissueBloodG10:
    def test_baseline_divergence_zero(self):
        m, _crp = create_immune_model()
        for _ in range(96):
            m.step(1.0)
        assert abs(m.get_tissue_blood_divergence()) < 0.05
        assert m.get_tissue_neutrophils() == pytest.approx(
            m.get_tissue_neutrophils())

    def test_infection_drives_tissue_neutrophilia(self):
        m, _crp = create_immune_model()
        m.infection_severity = 0.8
        for _ in range(72):
            m.step(1.0)
        assert m.get_tissue_blood_divergence() > 0.05
        assert m.get_tissue_neutrophils() > m.get_blood_neutrophils()

    def test_resolves_after_clearance(self):
        m, _crp = create_immune_model()
        m.infection_severity = 0.8
        for _ in range(72):
            m.step(1.0)
        m.infection_severity = 0.0
        for _ in range(168):
            m.step(1.0)
        assert m.get_tissue_blood_divergence() < 0.05

    def test_tissue_il6_rises_on_signal(self):
        m, _crp = create_immune_model()
        m.infection_severity = 0.8
        first = None
        for _ in range(72):
            m.step(1.0)
            if first is None:
                first = m.get_tissue_il6()
        assert m.get_tissue_il6() > first


class TestO10VectorizedComplement:
    def test_complement_vectorized_matches_scalar(self):
        cs = [ComplementCascade() for _ in range(4)]
        cs[1].anti_c5_dose = 1.0
        cs[2].anti_c5_dose = 0.5
        refs = copy.deepcopy(cs)
        sigs = [0.8, 0.8, 0.5, 0.0]
        for _ in range(48):
            cohort_complement_step(cs, 1.0, sigs, use_numpy=True)
            for r, m in enumerate(refs):
                m.step(1.0, sigs[r])
        for r in range(4):
            assert cs[r].get_mac() == pytest.approx(
                refs[r].get_mac(), abs=1e-9)

    def test_complement_fallback_matches_scalar(self):
        cs = [ComplementCascade() for _ in range(3)]
        refs = copy.deepcopy(cs)
        sigs = [0.9, 0.3, 0.0]
        for _ in range(24):
            cohort_complement_step(cs, 1.0, sigs, use_numpy=False)
            for r, m in enumerate(refs):
                m.step(1.0, sigs[r])
        for r in range(3):
            assert cs[r].get_total_activation() == pytest.approx(
                refs[r].get_total_activation(), abs=1e-9)


class TestO10VectorizedTissueBlood:
    def test_tissue_blood_vectorized_matches_scalar(self):
        ms = [TissueBloodModel() for _ in range(4)]
        refs = copy.deepcopy(ms)
        sigs = [0.8, 0.5, 0.3, 0.0]
        bil6 = [1.0, 2.0, 3.0, 0.5]
        bneut = [4.0, 3.5, 3.0, 4.2]
        bmono = [0.4, 0.35, 0.3, 0.42]
        for _ in range(72):
            cohort_tissue_blood_step(
                ms, 1.0, sigs, bil6, bneut, bmono, use_numpy=True)
            for r, m in enumerate(refs):
                m.blood_il6 = bil6[r]
                m.blood_neutrophils = bneut[r]
                m.blood_monocytes = bmono[r]
                m.step(1.0, sigs[r])
        for r in range(4):
            assert ms[r].get_tissue_blood_divergence() == pytest.approx(
                refs[r].get_tissue_blood_divergence(), abs=1e-9)



class TestFullL7Complement:
    """doc/40 Phase G(a) — full L7 complement network (61 dynamics refs)."""

    def test_all_params_referenced_not_padded(self):
        m = FullL7Complement()
        assert m.n_params() == N_L7_PARAMS
        # No inert placeholder keys remain (the old 142 padding rate_000..080
        # had zero effect on the ODEs and inflated the count dishonestly).
        assert all(not k.startswith("rate_") for k in m.p)
        assert N_L7_PARAMS == 61

    def test_inert_at_baseline(self):
        m = FullL7Complement()
        for _ in range(96):
            m.step(0.25, 0.0)
        assert m.get_c3a() == pytest.approx(0.0)
        assert m.get_c5a() == pytest.approx(0.0)
        assert m.get_mac() == pytest.approx(0.0)
        assert m.c3 == pytest.approx(1.0, abs=0.02)

    def test_signal_drives_effectors(self):
        m = FullL7Complement()
        for _ in range(24 * 48):
            m.step(0.25, 1.0)
        assert m.get_mac() > 0.0
        assert m.get_opsonization() > 0.0
        assert m.get_anaphylatoxin_signal() > 0.0

    def test_anti_c5_suppresses_mac(self):
        wt = FullL7Complement()
        drug = FullL7Complement()
        for _ in range(24 * 48):
            wt.step(0.25, 1.0)
            drug.step(0.25, 1.0, anti_c5_dose=1.0)
        # MAC arm is blocked, C3 opsonization spared (anti-C5 phenotype).
        assert drug.get_mac() < 0.2 * wt.get_mac()
        assert drug.get_opsonization() > 0.0

    def test_factor_d_knockout(self):
        # Compare a fresh WT vs a fresh Factor-D KO, both from baseline, so the
        # alternative-arm ablation is isolated (Zewde & Morikis 2018 KO gate).
        # Factor D knockout ablates the *alternative* convertase (C3bBb) while
        # leaving the classical (C4b2a) arm intact — a mechanistic gate.
        wt = FullL7Complement()
        ko = complement_knockout(FullL7Complement(), "factor_d", factor=0.0)
        for _ in range(24 * 48):
            wt.step(0.25, 1.0)
            ko.step(0.25, 1.0)
        assert ko.pathway_balance()["alternative"] < 0.2 * max(
            wt.pathway_balance()["alternative"], 1e-12)
        # Classical arm is preserved under Factor D KO.
        assert ko.pathway_balance()["classical"] >= 0.5 * wt.pathway_balance()["classical"]
        # Net MAC is reduced but not ablated (classical arm still forms some).
        assert ko.get_mac() < wt.get_mac()

    def test_c1_knockout(self):
        wt = FullL7Complement()
        ko = complement_knockout(FullL7Complement(), "c1_activation", factor=0.0)
        for _ in range(24 * 48):
            wt.step(0.25, 1.0)
            ko.step(0.25, 1.0)
        assert ko.get_mac() < 0.9 * wt.get_mac()

    def test_deterministic(self):
        a = FullL7Complement()
        b = FullL7Complement()
        for _ in range(48):
            a.step(0.25, 0.7)
            b.step(0.25, 0.7)
        assert round(a.get_mac(), 12) == round(b.get_mac(), 12)

    def test_parameter_set_and_override(self):
        m = FullL7Complement({"c1_activation": 0.9})
        assert m.get("c1_activation") == pytest.approx(0.9)
        assert m.n_params() == N_L7_PARAMS
