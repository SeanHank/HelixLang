"""Morphogen gradient tests: French-flag positional information (S2).

Verification goals:
- A source at one end + diffusion + decay yields a monotonically
  decreasing concentration profile with an approximately exponential
  decay length scale sqrt(D_lattice/decay) (continuum theory).
- Threshold-based genes with cross-repression form three contiguous,
  disjoint, correctly ordered domains (the French flag; Wolpert 1969;
  Driever & Nüsslein-Volhard 1988; synthetic realization in E. coli by
  Basu et al. 2005 Nature 434:1130).
- Boundary positions move outward when the morphogen amplitude rises
  (positional information encodes amplitude).
- Stronger decay shortens the gradient and pulls boundaries inward.
- The template GRN records each gene's morphogen threshold and the
  cross-repression edges.
"""
from __future__ import annotations

import math

from helixlang.apps.morphogen_gradient import (
    MorphogenGene,
    MorphogenGradient,
    MorphogenGradientConfig,
    make_template_grn,
)


def _default_steady(gradient: MorphogenGradient) -> MorphogenGradient:
    gradient.run(800)
    return gradient


def test_gradient_is_monotone_decreasing() -> None:
    grad = _default_steady(MorphogenGradient())
    assert grad.is_monotone_decreasing()
    assert grad.concentration[0] > grad.concentration[-1]


def test_gradient_decay_length_scale_matches_theory() -> None:
    cfg = MorphogenGradientConfig(diffusion_um2_s=10.0, decay_per_tick=0.05)
    grad = _default_steady(MorphogenGradient(cfg))
    # D_lattice = 10 * 60 / 100 = 6.0; lambda = sqrt(6.0/0.05) ~ 10.95
    lambda_theory = math.sqrt(6.0 / 0.05)
    # mid-strip successive ratio approximates exp(-1/lambda)
    mid = grad.config.length // 2
    c = grad.concentration
    ratios = [c[i + 1] / c[i] for i in range(mid, mid + 8)]
    r = sum(ratios) / len(ratios)
    assert abs(r - math.exp(-1.0 / lambda_theory)) < 0.1


def test_french_flag_three_ordered_domains() -> None:
    grad = _default_steady(MorphogenGradient())
    dom = grad.domains()
    near_s, near_e = dom["near"]
    mid_s, mid_e = dom["mid"]
    far_s, far_e = dom["far"]
    # all three bands are non-empty
    assert near_s >= 0 and mid_s >= 0 and far_s >= 0
    # contiguous, disjoint, ordered from source outward
    assert near_e < mid_s
    assert mid_e < far_s
    assert near_s == 0
    # the farthest band can extend to the strip end
    assert far_e >= mid_e + 1


def test_higher_amplitude_moves_boundaries_outward() -> None:
    base = MorphogenGradientConfig(source_strength_um=20.0)
    strong = MorphogenGradientConfig(source_strength_um=40.0)
    g_base = _default_steady(MorphogenGradient(base))
    g_strong = _default_steady(MorphogenGradient(strong))
    b_base = g_base.boundary_positions()
    b_strong = g_strong.boundary_positions()
    for name in ("near", "mid", "far"):
        assert b_strong[name] > b_base[name], name


def test_stronger_decay_pulls_boundaries_inward() -> None:
    weak = MorphogenGradientConfig(decay_per_tick=0.05)
    strong = MorphogenGradientConfig(decay_per_tick=0.15)
    g_weak = _default_steady(MorphogenGradient(weak))
    g_strong = _default_steady(MorphogenGradient(strong))
    b_weak = g_weak.boundary_positions()
    b_strong = g_strong.boundary_positions()
    for name in ("near", "mid", "far"):
        assert b_strong[name] < b_weak[name], name


def test_template_grn_records_thresholds_and_repression() -> None:
    genes = (MorphogenGene("near", 12.0),
             MorphogenGene("mid", 6.0),
             MorphogenGene("far", 2.0))
    grn = make_template_grn(genes, (("near", "mid"), ("mid", "far")))
    assert grn.nodes["near"].threshold == 12.0
    assert grn.nodes["mid"].threshold == 6.0
    assert grn.nodes["far"].threshold == 2.0
    edges = {(e.source, e.target): e.weight for e in grn.edges}
    assert edges[("near", "mid")] == -1.0
    assert edges[("mid", "far")] == -1.0
