"""Stochastic gene expression: the two-state (telegraph) promoter model
and an optional Gillespie stochastic simulation algorithm (SSA).

Biological grounding (all rates are per minute, one tick = 1 min,
:mod:`helixlang.core.units`):

- A gene alternates between an active ("ON") promoter state and an
  inactive ("OFF") state.  Transitions occur at rates ``k_off``
  (ON -> OFF) and ``k_on`` (OFF -> ON).  This *telegraph* or
  *two-state* model is the standard minimal description of promoter
  stochasticity and explains the bursty, super-Poissonian mRNA
  distributions observed in single cells (Jones et al. 2014, PLoS
  Comput Biol; Paulsson 2004; Peccoud & Ycart 1995).
- While ON, transcripts are produced at rate ``r``; mRNA is degraded
  at rate ``gamma``.  The mean number of transcripts produced during a
  single ON interval is the **burst size** ``b = r / k_off``.
- The steady-state Fano factor (variance/mean) of the mRNA distribution
  has the exact closed form (Peccoud & Ycart 1995, Theor Popul Biol
  48:222; verified here by SSA to within sampling error)::

      Fano = 1 + (r/gamma) * (k_off/(k_on + k_off))
                  * (gamma/(gamma + k_on + k_off))

  with ``r = b * k_off``.  In the constitutive limit (k_on >> k_off,
  gamma) the distribution is Poisson and ``Fano -> 1``; for a switching
  promoter ``Fano > 1`` and grows with burst size ``b``.

The discrete-time GRN recurrence (:mod:`helixlang.plugins.runtime.grn`) uses
:class:`TelegraphPromoter` to inject zero-mean Gaussian noise whose
steady-state variance reproduces this Fano factor, so the *mean*
deterministic trajectory is preserved while the *variance* matches the
two-state analytic result.  :func:`gillespie_telegraph` provides an
exact continuous-time SSA reference used by the test suite.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def telegraph_fano_factor(
    k_on: float,
    k_off: float,
    burst_size: float,
    degradation_rate: float,
) -> float:
    """Steady-state Fano factor (variance/mean) of the two-state promoter.

    Args:
        k_on:  OFF -> ON promoter transition rate (1/min)
        k_off: ON -> OFF promoter transition rate (1/min)
        burst_size: mean transcripts produced per ON interval (b = r/k_off)
        degradation_rate: mRNA degradation rate gamma (1/min)

    Returns:
        Fano >= 1; == 1 in the constitutive/Poisson limit.
    """
    if k_on < 0.0 or k_off < 0.0 or burst_size < 0.0:
        raise ValueError("k_on, k_off, burst_size must be >= 0")
    if degradation_rate <= 0.0:
        raise ValueError("degradation_rate must be > 0")
    if k_on + k_off <= 0.0:
        return 1.0
    r = burst_size * k_off
    p_on = k_off / (k_on + k_off)
    damping = degradation_rate / (degradation_rate + k_on + k_off)
    return 1.0 + (r / degradation_rate) * p_on * damping


@dataclass(frozen=True, slots=True)
class TelegraphPromoter:
    """Two-state (telegraph) promoter noise parameters for a gene.

    The GRN level is a *normalized* gene-product concentration in
    [0, 1]; the physical mRNA/protein count is ``level * expression_scale``.
    Telegraph noise is applied at that copy-number scale so the
    stationary *variance* of the level equals ``Fano * mean /
    expression_scale`` (the variance of ``Fano * mean`` copy numbers
    divided by the scale squared).

    Args:
        k_on: OFF -> ON promoter transition rate (1/min)
        k_off: ON -> OFF promoter transition rate (1/min)
        burst_size: mean transcripts per ON interval (b = r/k_off)
        degradation_rate: mRNA degradation rate gamma (1/min); defaults
            to the E. coli median mRNA half-life-derived rate when not
            set (5 min half-life -> gamma = ln 2 / 5 ~= 0.14/min,
            Bernstein et al. 2002 PNAS).
        expression_scale: gene-product copy number at full (level = 1.0)
            expression; ~10^2 mRNA/cell for a strongly induced bacterial
            promoter (Bernstein et al. 2002 PNAS; So et al. 2011),
            defaults to 100.
    """

    k_on: float
    k_off: float
    burst_size: float
    degradation_rate: float = 0.14
    expression_scale: float = 100.0

    @property
    def transcription_rate(self) -> float:
        """Transcription rate r = b * k_off (transcripts/min while ON)."""
        return self.burst_size * self.k_off

    @property
    def on_fraction(self) -> float:
        """Fraction of time the promoter is ON, k_on/(k_on + k_off)."""
        return self.k_on / (self.k_on + self.k_off)

    def fano_factor(self) -> float:
        """Steady-state Fano factor (see :func:`telegraph_fano_factor`)."""
        return telegraph_fano_factor(
            self.k_on, self.k_off, self.burst_size, self.degradation_rate)


def fano_to_noise_std(
    fano: float,
    mean: float,
    decay: float,
    expression_scale: float = 100.0,
) -> float:
    """Zero-mean noise std reproducing steady-state Fano ``fano`` at
    normalized mean ``mean`` in the discrete-time AR(1) GRN update.

    For ``level' = decay*level + (1-decay)*raw + eta`` the stationary
    variance is ``var_eta/(1-decay^2)``.  The physical copy-number
    variance is ``fano * mean * expression_scale`` (Fano x mean counts),
    so the *normalized* level variance is ``fano * mean /
    expression_scale``.  Hence ``var_eta = (1-decay^2) * fano * mean /
    expression_scale``.  The deterministic mean trajectory is preserved.
    """
    if fano < 1.0:
        raise ValueError("fano must be >= 1")
    if mean < 0.0:
        raise ValueError("mean must be >= 0")
    if not (0.0 <= decay < 1.0):
        raise ValueError("decay must be in [0, 1)")
    if expression_scale <= 0.0:
        raise ValueError("expression_scale must be > 0")
    return math.sqrt(
        (1.0 - decay * decay) * fano * mean / expression_scale)


def gillespie_telegraph(
    k_on: float,
    k_off: float,
    burst_size: float,
    degradation_rate: float,
    t_max: float,
    n_replicates: int = 2000,
    seed: int | None = None,
) -> dict[str, float]:
    """Exact continuous-time Gillespie SSA for the telegraph model.

    Simulates mRNA copy-number trajectories and returns the stationary
    summary statistics:

    Returns:
        ``{"mean": .., "variance": .., "fano": ..}`` of the mRNA count
        at time ``t_max`` across ``n_replicates`` independent runs.

    The promoter is OFF initially; degradation fires in both promoter
    states, transcription only in the ON state.
    """
    if t_max <= 0.0:
        raise ValueError("t_max must be > 0")
    rng = random.Random(seed)
    r = burst_size * k_off
    final: list[float] = []
    for _ in range(n_replicates):
        m = 0.0
        on = False
        t = 0.0
        while t < t_max:
            if on:
                k_total = k_off + r + degradation_rate * m
                dt = rng.expovariate(k_total)
                u = rng.random()
                if u < k_off / k_total:
                    on = False
                elif u < (k_off + r) / k_total:
                    m += 1.0
                else:
                    m -= 1.0
            else:
                k_total = k_on + degradation_rate * m
                dt = rng.expovariate(k_total)
                u = rng.random()
                if u < k_on / k_total:
                    on = True
                else:
                    m -= 1.0
            t += dt
        final.append(m)
    n = len(final)
    mean = sum(final) / n
    var = sum((x - mean) ** 2 for x in final) / (n - 1)
    return {"mean": mean, "variance": var, "fano": var / mean if mean > 0 else 1.0}


__all__ = [
    "telegraph_fano_factor",
    "TelegraphPromoter",
    "fano_to_noise_std",
    "gillespie_telegraph",
]
