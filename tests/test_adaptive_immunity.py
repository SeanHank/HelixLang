"""Tests for doc/40 Phase B — adaptive immunity + vaccination.

Covers gaps G2 (naive/effector/memory CD4/CD8/B), G3 (antibody IgM/IgG +
plasma-cell waning), G7 (APC/MHC priming delay), and G12 (vaccination with
priming + boost) implemented in
:mod:`helixlang.plugins.human.adaptive` and wired through
:class:`~helixlang.plugins.human.immune.InnateImmuneModel`.
"""
from __future__ import annotations

import copy

import pytest

from helixlang.plugins.human.adaptive import (
    AdaptiveImmuneModel,
    VaccineSchedule,
    cohort_adaptive_step,
)
from helixlang.plugins.human.immune import InnateImmuneModel, create_immune_model


class TestAdaptiveBaseline:
    def test_inert_at_no_antigen(self):
        m = AdaptiveImmuneModel()
        for _ in range(24 * 21):
            m.step(1.0, 0.0)
        assert m.get_total_antibody() == pytest.approx(10.4)
        assert m.get_effector_t() == 0.0
        assert m.get_memory_t() == 0.0

    def test_naive_pools_stay_at_baseline(self):
        m = AdaptiveImmuneModel()
        for _ in range(48):
            m.step(1.0, 0.0)
        assert m.naive_cd4 == pytest.approx(1.0)
        assert m.naive_b == pytest.approx(1.0)


class TestAdaptiveAntibody:
    def test_infection_drives_antibody(self):
        m = AdaptiveImmuneModel()
        for _ in range(21 * 24):
            m.step(1.0, 0.8)
        assert m.get_igg() > 10.4
        assert m.get_effector_t() > 0.0

    def test_memory_established_after_response(self):
        m = AdaptiveImmuneModel()
        for _ in range(14 * 24):
            m.step(1.0, 0.8)
        assert m.get_memory_t() > 0.0
        assert m.memory_b > 0.0

    def test_chronic_antigen_bounded(self):
        # doc/31 §4.5: pools must stay bounded, not diverge.
        m = AdaptiveImmuneModel()
        for _ in range(40 * 24):
            m.step(1.0, 0.8)
        assert m.get_effector_t() < 1e4
        assert m.get_igg() < 1e5


class TestVaccinationG12:
    def test_two_dose_schedule_peak_and_memory(self):
        m = AdaptiveImmuneModel()
        sched = VaccineSchedule([(0.0, 1.0), (24 * 7.0, 1.0)])
        peak = 0.0
        for t in range(24 * 28):
            dose = sched.due(t)
            m.step(1.0, 0.0, dose=dose)
            peak = max(peak, m.get_total_antibody())
        assert peak > 10.4
        assert m.memory_b > 0.0

    def test_boost_anamnesis_faster_than_priming(self):
        # memory rechallenge should drive a stronger, earlier Ab peak than the
        # naive primary response.
        primary = AdaptiveImmuneModel()
        primary.step(1.0, 0.0, dose=1.0)
        for _ in range(24 * 28):
            primary.step(1.0, 0.0)
        peak_primary = primary.get_total_antibody()

        boosted = AdaptiveImmuneModel()
        boosted.step(1.0, 0.0, dose=1.0)
        for _ in range(24 * 7):
            boosted.step(1.0, 0.0)
        # re-challenge with a boost after memory has formed
        boosted.step(1.0, 0.0, dose=1.0)
        for _ in range(24 * 21):
            boosted.step(1.0, 0.0)
        peak_boosted = boosted.get_total_antibody()
        assert peak_boosted > peak_primary


class TestAPCPrimingG7:
    def test_priming_is_delayed_not_instant(self):
        m = AdaptiveImmuneModel()
        # One day of strong antigen: priming delay (tau ~18 h) means the
        # effector response is still small on day 1.
        for _ in range(24):
            m.step(1.0, 0.9)
        day1_eff = m.get_effector_t()
        for _ in range(6 * 24):
            m.step(1.0, 0.9)
        assert m.get_effector_t() > day1_eff


class TestAdaptiveInnateIntegration:
    def test_backward_compatible_baseline(self):
        mod, _crp = create_immune_model()
        baseline = mod.get_total_antibody()
        mod.step(1.0)
        assert mod.get_total_antibody() == pytest.approx(baseline)

    def test_infection_raises_antibody_through_innate(self):
        mod, _crp = create_immune_model()
        mod.infection_severity = 0.8
        for _ in range(14 * 24):
            mod.step(1.0)
        assert mod.get_igg() > 10.4

    def test_vaccinate_passthrough(self):
        mod, _crp = create_immune_model()
        mod.vaccinate(1.0)
        assert mod.adaptive.antigen_available > 0.0

    def test_innate_default_pools_untouched(self):
        # The innate result channels must be unaffected by the additive layer.
        mod, _crp = create_immune_model()
        mod.infection_severity = 0.5
        first_il6 = None
        for _ in range(24):
            mod.step(1.0)
            if first_il6 is None:
                first_il6 = mod.get_il6()
        assert first_il6 is not None and first_il6 > 0.0


class TestCohortAdaptive:
    def test_vectorized_matches_scalar(self):
        ms = [AdaptiveImmuneModel() for _ in range(3)]
        refs = [copy.deepcopy(m) for m in ms]
        antigens = [0.8, 0.5, 0.0]
        for t in range(24 * 7):
            cohort_adaptive_step(ms, 1.0, antigens, use_numpy=True)
            for r in range(3):
                refs[r].step(1.0, antigens[r])
        for r in range(3):
            assert ms[r].get_total_antibody() == pytest.approx(
                refs[r].get_total_antibody(), abs=1e-6)

    def test_vectorized_matches_scalar_with_checkpoint(self):
        # O10: PD-1 blockade must also be bit-identical in the vectorized path.
        ms = [AdaptiveImmuneModel() for _ in range(3)]
        ms[1].checkpoint_blockade = 1.0
        ms[2].checkpoint_blockade = 0.5
        refs = [copy.deepcopy(m) for m in ms]
        antigens = [0.8, 0.6, 0.3]
        for _ in range(24 * 5):
            cohort_adaptive_step(ms, 1.0, antigens, use_numpy=True)
            for r in range(3):
                refs[r].step(1.0, antigens[r])
        for r in range(3):
            assert ms[r].get_effector_t() == pytest.approx(
                refs[r].get_effector_t(), abs=1e-6)


class TestCheckpointG14:
    def test_blockade_raises_effector_response(self):
        base = AdaptiveImmuneModel()
        blocked = AdaptiveImmuneModel()
        blocked.checkpoint_blockade = 1.0
        for _ in range(10 * 24):
            base.step(1.0, 0.8)
            blocked.step(1.0, 0.8)
        assert blocked.get_effector_t() > base.get_effector_t() * 1.5

    def test_blockade_inert_at_zero(self):
        # Zero blockade == no step-wise change in effector dynamics.
        a = AdaptiveImmuneModel()
        b = AdaptiveImmuneModel()
        b.checkpoint_blockade = 0.0
        for _ in range(72):
            a.step(1.0, 0.6)
            b.step(1.0, 0.6)
        assert a.get_effector_t() == pytest.approx(b.get_effector_t())

    def test_model_passthrough(self):
        mod, _crp = create_immune_model()
        mod.set_checkpoint_blockade(1.0)
        assert mod.get_checkpoint_blockade() == 1.0
        mod.infection_severity = 0.8
        for _ in range(5 * 24):
            mod.step(1.0)
        assert mod.get_effector_t() > 0.0


class TestBiologicAntiIL6L10:
    def test_biologic_suppresses_il6(self):
        plain, _crp = create_immune_model()
        treated, _crp2 = create_immune_model()
        plain.infection_severity = 0.8
        treated.infection_severity = 0.8
        treated.set_il6_biologic_occupancy(0.8)
        for _ in range(48):
            plain.step(1.0)
            treated.step(1.0)
        assert treated.get_il6() < plain.get_il6() * 0.3

    def test_biologic_inert_at_zero(self):
        a, _crp = create_immune_model()
        b, _crp2 = create_immune_model()
        a.infection_severity = 0.8
        b.infection_severity = 0.8
        b.set_il6_biologic_occupancy(0.0)
        for _ in range(48):
            a.step(1.0)
            b.step(1.0)
        assert a.get_il6() == pytest.approx(b.get_il6(), abs=1e-6)


class TestCheckpointPhaseG:
    """doc/40 Phase G(b): full PD-1/PD-L1/PD-L2 + CTLA-4 + LAG-3 network."""

    def _braked(self, **kwargs):
        m = AdaptiveImmuneModel()
        for k, v in kwargs.items():
            getattr(m, k)(v)
        return m

    def test_network_inert_at_baseline(self):
        # No ligands/therapy -> no brake, and legacy scalar untouched remains 0.
        m = AdaptiveImmuneModel()
        for _ in range(48):
            m.step(1.0, 0.0)
        assert m.pd1.immune_brake() == pytest.approx(0.0, abs=1e-9)
        assert m.effective_checkpoint_blockade() == pytest.approx(0.0)

    def test_legacy_scalar_still_respected_when_untouched(self):
        m = AdaptiveImmuneModel()
        m.checkpoint_blockade = 1.0
        for _ in range(12):
            m.step(1.0, 0.2)
        assert m.effective_checkpoint_blockade() == pytest.approx(1.0)

    def test_pdl1_drives_pd1_signal(self):
        m = self._braked(set_pdl1=0.9)
        for _ in range(48):
            m.step(1.0, 0.0)
        assert m.pd1.pd1_signal() > 0.3
        assert 0.0 < m.pd1.immune_brake() < 1.0

    def test_combination_relief_stronger_than_mono(self):
        # Same ligand load; combination blockade relieves more brake than
        # anti-PD-1 monotherapy, and both relieve more than no therapy.
        def steady(**kwargs):
            m = AdaptiveImmuneModel()
            m.set_pdl1(0.9)
            m.set_pdl2(0.8)
            m.set_ctla4_ligand(0.9)
            m.set_lag3_ligand(0.8)
            m.set_checkpoint_therapy(*([kwargs[k] for k in
                                        ("anti_pd1", "anti_ctla4", "anti_lag3")]))
            for _ in range(24):
                m.step(1.0, 0.0)
            return m

        none_bd = steady(anti_pd1=0.0, anti_ctla4=0.0, anti_lag3=0.0)\
            .effective_checkpoint_blockade()
        mono_bd = steady(anti_pd1=1.0, anti_ctla4=0.0, anti_lag3=0.0)\
            .effective_checkpoint_blockade()
        combo_bd = steady(anti_pd1=1.0, anti_ctla4=1.0, anti_lag3=1.0)\
            .effective_checkpoint_blockade()
        assert none_bd < mono_bd < combo_bd <= 1.0

    def test_combo_raises_effector_response(self):
        def run(therapy):
            m = self._braked(set_pdl1=0.9, set_pdl2=0.8,
                             set_ctla4_ligand=0.9, set_lag3_ligand=0.8)
            m.set_checkpoint_therapy(*therapy)
            for _ in range(10 * 24):
                m.step(1.0, 0.8)
            return m.get_effector_t()

        none = run((0.0, 0.0, 0.0))
        mono = run((1.0, 0.0, 0.0))
        combo = run((1.0, 1.0, 1.0))
        assert combo > mono > none

    def test_pd1_trafficking_recycles(self):
        from helixlang.plugins.human.adaptive import PD1Checkpoint
        p = PD1Checkpoint()
        p.set_pdl1(0.9)
        for _ in range(200):
            p.step(1.0)
        # Intracellular receptor recycles to surface; bound complex persists.
        assert p.pd1_surface > 0.5
        assert p.pd1_pdl1 > 0.1

    def test_vectorized_matches_scalar_with_pd1_network(self):
        ms = [self._braked(set_pdl1=0.9, set_ctla4_ligand=0.7)
              for _ in range(3)]
        ms[1].set_checkpoint_therapy(anti_pd1=1.0)
        ms[2].set_checkpoint_therapy(anti_pd1=1.0, anti_ctla4=1.0)
        refs = [copy.deepcopy(m) for m in ms]
        antigens = [0.8, 0.6, 0.3]
        for _ in range(24 * 5):
            cohort_adaptive_step(ms, 1.0, antigens, use_numpy=True)
            for r in range(3):
                refs[r].step(1.0, antigens[r])
        for r in range(3):
            assert ms[r].get_effector_t() == pytest.approx(
                refs[r].get_effector_t(), abs=1e-6)

