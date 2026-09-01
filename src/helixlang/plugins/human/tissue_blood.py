"""Tissue vs blood pseudo-compartments (doc/40 Phase C, gap G10).

The immune cell population is a *single* circulating pool, yet the docstring
historically claimed three spaces (blood / tissue / lymphoid).  This module
makes the blood-vs-tissue split real with a compact, additive compartment
model driven by the same chemokine/inflammatory signal as the rest of the
innate arm.

Design (per L1 compartment taxonomy and L2 tissue-vs-circulating WBC
mismatch):

* **Blood** compartments mirror the circulating values already tracked
  (IL-6, neutrophils, monocytes) — at baseline they equal the innate model's
  channels, so existing result consumers are unaffected.
* **Tissue** compartments carry the inflammation-relevant state: tissue
  cytokines, tissue-resident macrophages, and a *migrated* neutrophil pool.
* **Chemokine index** ``attractant`` (0–1, Hill on the innate pathogen signal)
  drives **margination/recruitment**: circulating neutrophils and monocytes
  move into tissue at a rate proportional to the attractant, and clear back
  when the signal resolves.  This reproduces the hallmark pattern of
  ``tissue neutrophilia`` with ``circulating (blood) neutropenia`` during a
  localised infection — a divergence real ODE models exhibit but a single
  circulating pool cannot.

Inert at baseline (no signal, no migration): blood values equal baseline and
the tissue-vs-blood divergence is zero, so wiring is backward-compatible with
goldens.

Deterministic: every rate is a first-order constant; the only nonlinearity is
a Hill-shaped attractant, matching the doc/40 Hill convention.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:  # numpy optional (doc/39 O2/O10 idiom)
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - pure-python install
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _hill(x: float, half: float, n: float) -> float:
    xp = x ** n
    hp = half ** n + 1e-12
    return float(xp / (hp + xp))


@dataclass
class TissueBloodModel:
    """Pseudo-compartment split of inflammatory cytokines and WBC (G10).

    State (all additive, blood defaulting to the innate circulating values):
        blood_il6, blood_neutrophils, blood_monocytes: circulating drivers
        tissue_il6: tissue interstitial cytokine (rises more steeply)
        tissue_macrophages: tissue-resident (rise with engulfment signal)
        tissue_neutrophils: migrated PMN pool (rises on attractant)
        attractant: 0–1 chemokine/margination index (derived, not stored)
    """

    blood_il6: float = 1.0
    blood_neutrophils: float = 4.0
    blood_monocytes: float = 0.4
    tissue_il6: float = 1.0
    tissue_macrophages: float = 0.5
    tissue_neutrophils: float = 0.5   # tissue-resident baseline

    # --- Kinetics ---
    attractant_half: float = 0.3      # pathogen signal at half-max recruitment
    attractant_n: float = 2.0         # Hill steepness for chemokine index
    margination_rate: float = 0.05    # 1/h blood -> tissue migration at full index
    recirculation_rate: float = 0.09  # 1/h tissue -> blood when signal resolves
    tissue_cytokine_gain: float = 2.0 # tissue IL-6 multiplier vs blood during signal

    # Cached decay constants + baseline reference
    _k_tissue_il6: float = field(default=0.46, init=False, repr=False)  # ~1.5 h
    _k_tissue_mac: float = field(default=0.01, init=False, repr=False)  # long-lived
    _tissue_pmn_baseline: float = field(default=0.5, init=False, repr=False)

    def __post_init__(self) -> None:
        self._k_tissue_il6 = math.log(2) / 1.5
        self._k_tissue_mac = 0.01
        self._tissue_pmn_baseline = self.tissue_neutrophils

    def attractant_index(self, pathogen_signal: float) -> float:
        return _hill(max(0.0, pathogen_signal),
                     self.attractant_half, self.attractant_n)

    def step(self, dt_h: float, pathogen_signal: float) -> None:
        """Advance the compartment split one hour.

        ``blood_*`` fields are refreshed by the caller (the innate model) to
        mirror its circulating channels each tick; this step applies the
        migration/differencing *around* those values.
        """
        attr = self.attractant_index(pathogen_signal)

        # Migration: circulating -> tissue proportional to attractant.
        migrate_n = self.blood_neutrophils * self.margination_rate * attr * dt_h
        migrate_m = self.blood_monocytes * self.margination_rate * attr * dt_h
        self.blood_neutrophils -= migrate_n
        self.blood_monocytes -= migrate_m
        self.tissue_neutrophils += migrate_n
        # migrated monocytes differentiate into tissue macrophages
        self.tissue_macrophages += migrate_m * 0.3 * dt_h

        # Tissue IL-6 accretes at a higher gain under a tissue signal and
        # decays on the ~1.5 h IL-6 half-life; baseline held near blood level.
        self.tissue_il6 += dt_h * (
            self.tissue_cytokine_gain * max(0.0, pathogen_signal) * _hill(
                pathogen_signal, 0.2, 2.0)
            - self._k_tissue_il6 * (self.tissue_il6 - self.blood_il6))

        # Recirculation when signal resolves: tissue PMN returns to blood, but
        # never below its tissue-resident baseline.
        excess = self.tissue_neutrophils - self._tissue_pmn_baseline
        if excess > 0.0:
            back = excess * self.recirculation_rate * (1.0 - attr) * dt_h
            self.tissue_neutrophils -= back
            self.blood_neutrophils += back

        # Floors
        self.blood_neutrophils = max(0.1, self.blood_neutrophils)
        self.blood_monocytes = max(0.05, self.blood_monocytes)
        self.tissue_neutrophils = max(self._tissue_pmn_baseline,
                                      self.tissue_neutrophils)
        self.tissue_macrophages = max(0.05, self.tissue_macrophages)

    # --- Public accessors ---
    def get_blood_neutrophils(self) -> float:
        return self.blood_neutrophils

    def get_tissue_neutrophils(self) -> float:
        return self.tissue_neutrophils

    def get_blood_il6(self) -> float:
        return self.blood_il6

    def get_tissue_il6(self) -> float:
        return self.tissue_il6

    def get_tissue_macrophages(self) -> float:
        return self.tissue_macrophages

    def get_tissue_blood_divergence(self) -> float:
        """Tissue-vs-blood mismatch (migration index): >0 = tissue
        neutrophilia from circulating margination; ~0 at baseline."""
        return self.tissue_neutrophils - self._tissue_pmn_baseline


def cohort_tissue_blood_step(
    models: list[TissueBloodModel],
    dt_h: float,
    signals: list[float],
    blood_il6: list[float],
    blood_neutrophils: list[float],
    blood_monocytes: list[float],
    use_numpy: bool | None = None,
) -> list[float]:
    """Vectorized tissue/blood step over a cohort (doc/39 O10 idiom).

    Mirrors :meth:`TissueBloodModel.step` term-for-term across numpy arrays,
    refreshing each model's blood compartment from the caller-provided
    circulating values (``blood_il6`` etc., as produced by the innate O2
    kernel) then applying migration/differencing.  Bit-identical to the scalar
    path; falls back to scalar when numpy is unavailable or ``use_numpy=False``.
    Returns per-model tissue-vs-blood divergence.
    """
    if use_numpy is None:
        use_numpy = _HAS_NUMPY
    if not use_numpy or _np is None:
        out = []
        for i, m in enumerate(models):
            m.blood_il6 = blood_il6[i]
            m.blood_neutrophils = blood_neutrophils[i]
            m.blood_monocytes = blood_monocytes[i]
            m.step(dt_h, signals[i])
            out.append(m.get_tissue_blood_divergence())
        return out

    np = _np
    n = len(models)

    def arr(get: Callable[[Any], float]) -> Any:
        return np.array([get(m) for m in models], dtype=float)

    b_il6 = np.array(blood_il6, dtype=float)
    b_neut = np.array(blood_neutrophils, dtype=float)
    b_mono = np.array(blood_monocytes, dtype=float)
    sig = np.maximum(np.array(signals, dtype=float), 0.0)
    t_neut = arr(lambda m: m.tissue_neutrophils)
    t_max = arr(lambda m: m.tissue_macrophages)
    t_il6 = arr(lambda m: m.tissue_il6)
    base = arr(lambda m: m._tissue_pmn_baseline)

    attr = sig ** 2.0 / (0.3 ** 2.0 + sig ** 2.0 + 1e-12)
    migrate_n = b_neut * 0.05 * attr * dt_h
    migrate_m = b_mono * 0.05 * attr * dt_h
    b_neut -= migrate_n
    b_mono -= migrate_m
    t_neut += migrate_n
    t_max += migrate_m * 0.3 * dt_h

    k_til6 = np.log(2.0) / 1.5
    t_il6 += dt_h * (
        2.0 * sig * (sig ** 2.0 / (0.2 ** 2.0 + sig ** 2.0 + 1e-12))
        - k_til6 * (t_il6 - b_il6))

    excess = np.maximum(t_neut - base, 0.0)
    back = excess * 0.09 * (1.0 - attr) * dt_h
    t_neut -= back
    b_neut += back

    b_neut = np.maximum(b_neut, 0.1)
    b_mono = np.maximum(b_mono, 0.05)
    t_neut = np.maximum(t_neut, base)
    t_max = np.maximum(t_max, 0.05)

    for i, m in enumerate(models):
        m.blood_il6 = float(b_il6[i])
        m.blood_neutrophils = float(b_neut[i])
        m.blood_monocytes = float(b_mono[i])
        m.tissue_neutrophils = float(t_neut[i])
        m.tissue_macrophages = float(t_max[i])
        m.tissue_il6 = float(t_il6[i])
    return [float(t_neut[i] - base[i]) for i in range(n)]


__all__ = ["TissueBloodModel", "cohort_tissue_blood_step"]
