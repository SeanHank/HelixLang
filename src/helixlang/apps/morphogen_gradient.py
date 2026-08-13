"""Morphogen gradient patterning: French-flag positional information (S2).

A single morphogen is emitted at one end of a 1-D cell strip, diffuses
and decays, establishing a monotonically decreasing concentration
gradient.  Each cell hosts a gene set whose members respond to different
concentration thresholds, so the strip self-organizes into contiguous
gene-expression domains ordered by distance from the source -- the
classic *French flag* model of positional information (Wolpert 1969;
Driever & Nüsslein-Volhard 1988 for the bicoid/Hunchback system in
*Drosophila*), realized synthetically with a graded autoinducer in
*E. coli* by Basu et al. 2005 (Nature 434:1130-1134).

Model
-----
- One lattice site per cell (site edge ``LATTICE_SPACING_UM`` µm),
  one tick = ``DIFFUSION_DT_S`` s.
- Source strength ``S`` (µM per tick) injected at site 0.
- Diffusion ``D`` (physical µm^2/s, converted on-lattice via
  :func:`helixlang.units.diffusion_to_lattice`) + first-order decay
  ``d`` (fraction per tick) set the gradient length scale
  ``lambda = dx / sqrt(d/D_lattice)``.
- Gene ``g`` is expressed with level ``sigmoid(k * (c(x) - thr_g))``;
  a gene is *on* when its level exceeds 0.5, i.e. at sites where
  ``c(x) >= thr_g``.  Cross-repression edges then silence each gene
  inside its repressor's domain, so the strip self-organizes into the
  mutually exclusive, contiguously ordered *bands* of the French flag
  (higher thresholds closer to the source).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from helixlang.grn import GRN, sigmoid
from helixlang.units import (
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    diffusion_to_lattice,
)

# ============================================================================
# Config + cells
# ============================================================================

@dataclass(slots=True)
class MorphogenGene:
    """A morphogen-responsive gene and its concentration threshold."""

    name: str
    threshold_um: float


@dataclass(slots=True)
class MorphogenGradientConfig:
    """Configuration of the 1-D morphogen patterning strip.

    Args:
        length: number of cells along the strip.
        diffusion_um2_s: morphogen diffusion coefficient (µm^2/s),
            converted on-lattice.
        decay_per_tick: first-order morphogen decay fraction per tick.
        source_strength_um: morphogen injected per tick at site 0 (µM).
        response_steepness: Hill-like steepness ``k`` of the per-gene
            sigmoid response.
        genes: morphogen-responsive gene set (name + threshold).  The
            first gene with the highest threshold forms the domain
            closest to the source.
        repression: cross-repression edges ``(repressor, repressed)``
            (e.g. ``(("near", "mid"), ("mid", "far"))``); a repressed
            gene is silenced inside its repressor's domain, so the
            expression domains become the mutually exclusive *bands* of
            the French flag rather than nested responses (the cross-
            repression architecture of the *Drosophila* gap-gene system,
            Driever & Nüsslein-Volhard 1988).
        seed: RNG seed (reserved for future stochastic responses).
    """

    length: int = 64
    diffusion_um2_s: float = 10.0
    decay_per_tick: float = 0.05
    source_strength_um: float = 20.0
    response_steepness: float = 0.5
    genes: tuple[MorphogenGene, ...] = (
        MorphogenGene("near", 12.0),
        MorphogenGene("mid", 6.0),
        MorphogenGene("far", 2.0),
    )
    repression: tuple[tuple[str, str], ...] = (
        ("near", "mid"),
        ("near", "far"),
        ("mid", "far"),
    )
    seed: int | None = None


@dataclass(slots=True)
class MorphogenGradientCell:
    """One cell of the patterning strip."""

    x: int
    concentration_um: float = 0.0
    expression: dict[str, float] = field(default_factory=dict)
    grn: GRN | None = None

    @property
    def active_genes(self) -> list[str]:
        """Genes whose expression level exceeds the 0.5 on-threshold."""
        return [name for name, level in self.expression.items() if level > 0.5]


# ============================================================================
# Template GRN (threshold carriers)
# ============================================================================

def make_template_grn(genes: tuple[MorphogenGene, ...],
                      repression: tuple[tuple[str, str], ...] = ()) -> GRN:
    """Build the template GRN carrying the morphogen thresholds.

    Each gene is registered with ``add_gene(name, threshold)`` so the
    cell's regulatory program records its morphogen-responsive threshold
    (the ``threshold`` attribute of the GRN node), and cross-repression
    edges are registered as negative regulations.  Every strip cell
    deep-copies this template, mirroring the programmable-cell pattern of
    :class:`~helixlang.population.CellPopulation`.
    """
    grn = GRN()
    for gene in genes:
        grn.add_gene(gene.name, threshold=gene.threshold_um)
    for repressor, repressed in repression:
        grn.add_edge(repressor, repressed, -1.0)
    return grn


# ============================================================================
# Gradient simulator
# ============================================================================

class MorphogenGradient:
    """Simulate morphogen diffusion + threshold-based pattern formation."""

    def __init__(self, config: MorphogenGradientConfig | None = None) -> None:
        self.config = config or MorphogenGradientConfig()
        if self.config.length <= 0:
            raise ValueError("length must be >= 1")
        self.template_grn = make_template_grn(
            self.config.genes, self.config.repression)
        self.concentration: list[float] = [0.0] * self.config.length
        self.cells: list[MorphogenGradientCell] = []
        for x in range(self.config.length):
            cell = MorphogenGradientCell(
                x=x,
                expression={g.name: 0.0 for g in self.config.genes},
            )
            self.cells.append(cell)
        self.tick = 0
        self.history: list[list[float]] = []

    # -- physics ------------------------------------------------------------

    def _d_lattice(self) -> float:
        return diffusion_to_lattice(
            self.config.diffusion_um2_s, DIFFUSION_DT_S, LATTICE_SPACING_UM)

    def _diffuse_decay(self) -> None:
        """One explicit 1-D diffusion + decay step (Neumann boundaries).

        The on-lattice coefficient is realized in stable sub-steps
        (1-D explicit-scheme limit 0.25, matching the 2-D limit used by
        :mod:`helixlang.population`), so the scheme never blows up.
        """
        cfg = self.config
        d = self._d_lattice()
        if d > 0.0:
            n = max(1, math.ceil(d / 0.25))
            d_sub = d / n
            c = self.concentration
            for _ in range(n):
                new = [0.0] * len(c)
                for x in range(len(c)):
                    left = c[x - 1] if x > 0 else c[x]
                    right = c[x + 1] if x < len(c) - 1 else c[x]
                    new[x] = c[x] + d_sub * (left + right - 2.0 * c[x])
                c = new
            self.concentration = c
        if cfg.decay_per_tick > 0.0:
            self.concentration = [
                max(0.0, v * (1.0 - cfg.decay_per_tick))
                for v in self.concentration
            ]

    def _express(self) -> None:
        """Update every cell's gene expression from the local gradient.

        Raw levels are the sigmoid of ``k * (c - threshold)``; the
        cross-repression edges then silence a gene wherever its repressor
        is on, carving the gradient into exclusive bands.
        """
        cfg = self.config
        k = cfg.response_steepness
        for cell in self.cells:
            for gene in cfg.genes:
                level = sigmoid(k * (cell.concentration_um
                                     - gene.threshold_um))
                cell.expression[gene.name] = level
            for repressor, repressed in cfg.repression:
                if cell.expression.get(repressor, 0.0) > 0.5:
                    cell.expression[repressed] = 0.0

    def step(self) -> list[float]:
        """Advance one tick; returns the concentration profile."""
        cfg = self.config
        self.concentration[0] += cfg.source_strength_um
        self._diffuse_decay()
        for x, cell in enumerate(self.cells):
            cell.concentration_um = self.concentration[x]
        self._express()
        self.tick += 1
        self.history.append(list(self.concentration))
        return list(self.concentration)

    def run(self, n_ticks: int) -> list[list[float]]:
        """Run ``n_ticks`` ticks; returns the concentration history."""
        for _ in range(n_ticks):
            self.step()
        return self.history

    # -- pattern analysis ---------------------------------------------------

    def is_monotone_decreasing(self) -> bool:
        """Whether the concentration profile decreases with distance."""
        return all(
            self.concentration[i] >= self.concentration[i + 1]
            for i in range(len(self.concentration) - 1))

    def domains(self) -> dict[str, tuple[int, int]]:
        """Contiguous expression domains ``{gene: (start, end)}``.

        A gene's domain is the maximal run of sites where it is active.
        Genes with a threshold above the local concentration are
        inactive, so the domains tile the strip in threshold order
        (nested French-flag structure).
        """
        out: dict[str, tuple[int, int]] = {}
        for gene in self.config.genes:
            start: int | None = None
            last = -1
            for x, cell in enumerate(self.cells):
                if gene.name in cell.expression and cell.expression[gene.name] > 0.5:
                    if start is None:
                        start = x
                    last = x
            if start is None:
                out[gene.name] = (-1, -1)
            else:
                out[gene.name] = (start, last)
        return out

    def gradient_length_scale(self) -> float:
        """Steady-state gradient length scale in lattice sites.

        ``lambda = 1/sqrt(d/D_lattice)`` from the continuum decay-diffusion
        solution ``c(x) = c0 * exp(-x/lambda)``.
        """
        d = self.config.decay_per_tick
        dl = self._d_lattice()
        if d <= 0.0 or dl <= 0.0:
            return float("inf")
        return math.sqrt(dl / d)

    def boundary_positions(self) -> dict[str, float]:
        """Mean position of each gene-domain far edge (positional info).

        Larger source strength pushes every boundary farther from the
        source, so boundary position is a readout of morphogen amplitude
        (Wolpert's amplitude-to-position mapping).
        """
        dom = self.domains()
        out: dict[str, float] = {}
        for name, (start, end) in dom.items():
            if start < 0:
                out[name] = float(self.config.length)
            else:
                out[name] = float(end + 1)
        return out


__all__ = [
    "MorphogenGene", "MorphogenGradientConfig",
    "MorphogenGradientCell", "MorphogenGradient",
    "make_template_grn",
]
