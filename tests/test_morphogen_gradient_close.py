"""Coverage-closure tests for :mod:`helixlang.plugins.apps.morphogen_gradient`.

Exercises the construction guard, the ``active_genes`` property, the
inactive-gene domains/boundaries, the ``gradient_length_scale`` edge cases and
the diffusion/decay skip branches.
"""
import math

import pytest

from helixlang.plugins.apps.morphogen_gradient import (
    MorphogenGene,
    MorphogenGradient,
    MorphogenGradientCell,
    MorphogenGradientConfig,
)


class TestMorphogenGradientClosure:
    def test_length_must_be_positive(self):
        with pytest.raises(ValueError):
            MorphogenGradient(MorphogenGradientConfig(length=0))

    def test_active_genes_above_half(self):
        cell = MorphogenGradientCell(x=3, expression={
            "a": 0.9, "b": 0.4, "c": 1.0,
        })
        assert cell.active_genes == ["a", "c"]

    def test_inactive_gene_gives_unbound_domain_and_boundary(self):
        cfg = MorphogenGradientConfig(
            genes=(MorphogenGene("too_high", 1e6),),
            repression=(),
        )
        grad = MorphogenGradient(cfg)
        grad.run(3)
        assert grad.domains()["too_high"] == (-1, -1)
        assert grad.boundary_positions()["too_high"] == float(cfg.length)

    def test_gradient_length_scale_edge_cases(self):
        assert math.isfinite(MorphogenGradient(
            MorphogenGradientConfig()).gradient_length_scale())
        no_decay = MorphogenGradient(
            MorphogenGradientConfig(decay_per_tick=0.0))
        assert no_decay.gradient_length_scale() == float("inf")
        no_diffusion = MorphogenGradient(
            MorphogenGradientConfig(diffusion_um2_s=0.0))
        assert no_diffusion.gradient_length_scale() == float("inf")

    def test_no_diffusion_skips_diffuse(self):
        grad = MorphogenGradient(
            MorphogenGradientConfig(diffusion_um2_s=0.0, decay_per_tick=0.1))
        before = list(grad.concentration)
        grad.step()
        assert grad.concentration[0] > before[0]

    def test_no_decay_skips_decay(self):
        grad = MorphogenGradient(
            MorphogenGradientConfig(decay_per_tick=0.0, source_strength_um=5.0))
        grad.run(2)
        assert all(v >= 0.0 for v in grad.concentration)
