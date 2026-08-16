"""Genome-scale GRN builder + colony closure (doc/18-programmable-cell-population-simulation.md §13 Design 5, tasks 1/3/4).

Lifts the per-cell GRN from a ~dozen-gene ``.helix`` program to E. coli
MG1655 scale (~4300 ORFs, ~10^4 sparse regulatory edges, RegulonDB
scale) and closes the loop to metabolism: expression-triggered genes
gate reaction upper bounds in the shared FBA core model, so a
"knocked-out" gene is simply a never-triggered silent node.

Layers (doc/18-programmable-cell-population-simulation.md §693-782):

1. ``#genome`` language wiring lives in ``parser.py``/``sim_runtime.py``;
   this module is the builder it dispatches to.
2. The sparse core is :class:`~helixlang.sparse_grn.SparseGRN` (CSR
   matrix; per-cell state is a row of the ``(N, G)`` level matrix, the
   active-gene budget only touches materially-driven targets).
3. :class:`GenomeColony` runs a whole population of genome-scale cells in
   one vectorized call per tick, reports per-cell triggered genes, and
   exposes an FBA closure for the E. coli core subset
   (:data:`~helixlang.metabolism.ECOLI_CORE_GENE_REACTIONS`).

References:
- Martínez-Antonio & Collado-Vides 2003, Curr Opin Microbiol 6:482
  (E. coli TF network is hierarchical; a handful of global regulators
  on top — crp/fis/lrp/hns are the canonical master hubs).
- Martínez-Antonio, Janga & Thieffry 2008, J Mol Biol 381:238
  (sparse ~10^4-edge functional network; ~10% of genes expressed at
  any instant).
- Barabási & Albert 1999 (preferential-attachment scale-free degree
  distribution for the random ``tf_map`` background).
- Orth et al. 2010, Mol Syst Biol 6:390 / Feist et al. 2007 (core-model
  gene→reaction AND gating for the essentiality closure).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from helixlang.apps.whole_cell_scale import ESSENTIALITY_FLUX_TOL
from helixlang.grn import GRN
from helixlang.metabolism import (
    ECOLI_CORE_GENE_REACTIONS,
    ECOLI_CORE_MODEL,
    FluxBalanceAnalysis,
)
from helixlang.sparse_grn import SparseGRN

#: default synthetic genome size (E. coli MG1655 ORF count, ~4300)
DEFAULT_GENES = 4300
#: default per-cell per-tick active-gene budget (doc/18-programmable-cell-population-simulation.md Design 5)
DEFAULT_ACTIVE_BUDGET = 512

#: canonical E. coli master regulators (global regulators on the top of
#: the regulatory hierarchy, Martínez-Antonio et al. 2008).
MASTER_REGULATORS = ("crp", "fis", "lrp", "hns")
#: secondary global regulators feeding the middle tier.
GLOBAL_REGULATORS = (
    "crp", "fis", "lrp", "hns",
    "ihf", "fnr", "arcA", "rpoD", "rpoS", "flhDC",
)

#: per-gene noise fraction for the synthetic genome (a minority of genes
#: carry intrinsic telegraph noise, Peccoud & Ycart 1995).
NOISE_FRACTION = 0.05

#: Representative RegulonDB-derived regulatory interactions among the
#: E. coli master/global regulators and the FBA-gated core genes (doc/19
#: §5.5 C1, ``tf_map="regulondb"``).  A curated subset of RegulonDB/
#: EcoCyc-documented TF -> target pairs: CRP-cAMP catabolite activation,
#: ArcA/FNR anaerobic TCA repression, FIS silencing and Lrp leucine
#: regulation.  The full dump path is ``parse_regulondb`` (a RegulonDB
#: network table exported as ``regulator<TAB>target<TAB>effect``).
REGULONDB_DEMO_EDGES: tuple[tuple[str, str, float], ...] = (
    # CRP-cAMP global catabolite activator: central-carbon activation
    ("crp", "gltA", 1.0), ("crp", "icdA", 1.0), ("crp", "zwf", 1.0),
    ("crp", "ptsG", 1.0), ("crp", "aceE", 1.0), ("crp", "eno", 1.0),
    ("crp", "fba", 1.0), ("crp", "ppc", 1.0),
    # ArcA two-component anaerobiosis repressor: TCA-cycle shutdown
    ("arcA", "gltA", -1.0), ("arcA", "icdA", -1.0), ("arcA", "sdhA", -1.0),
    ("arcA", "sucAB", -1.0), ("arcA", "fumA", -1.0), ("arcA", "mdh", -1.0),
    # FNR anaerobiosis regulator: ldhA activation, TCA repression
    ("fnr", "ldhA", 1.0), ("fnr", "fumA", -1.0), ("fnr", "sdhA", -1.0),
    ("fnr", "sucAB", -1.0),
    # FIS nucleoid protein: silences metabolic genes
    ("fis", "gltA", -1.0), ("fis", "icdA", -1.0), ("fis", "aceE", -1.0),
    # Lrp leucine-responsive regulator
    ("lrp", "gltA", -1.0), ("lrp", "ppc", -1.0),
)


def parse_regulondb(text: str) -> list[tuple[str, str, float]]:
    """Parse a RegulonDB-style network dump into +/- weighted edges.

    Expected TSV columns ``regulator<TAB>target<TAB>effect`` where
    ``effect`` is ``+``/``-`` (optionally with a magnitude, e.g. ``+0.8``).
    ``#`` comment lines and a ``regulator`` header row are skipped.
    """
    edges: list[tuple[str, str, float]] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        reg, target, effect = cols[0].strip(), cols[1].strip(), cols[2].strip()
        if reg.lower() == "regulator" and target.lower() == "target":
            continue
        sign = -1.0 if effect.startswith("-") else 1.0
        magnitude = effect.lstrip("+-")
        try:
            weight = sign * float(magnitude) if magnitude else sign
        except ValueError:
            continue
        edges.append((reg, target, weight))
    return edges


def b_number(i: int) -> str:
    """E. coli b-number gene name (``b0001``, ``b0002``, ...)."""
    return f"b{i:04d}"


def _core_genes() -> list[str]:
    """Core-model metabolic genes (their reactions are FBA-gated)."""
    return sorted(ECOLI_CORE_GENE_REACTIONS)


@dataclass(slots=True)
class GenomeSpec:
    """Deterministic synthetic genome-scale GRN (shared template).

    ``grn`` is the immutable sparse template; ``names`` is its column
    order; ``gene_to_reactions`` gates the E. coli core subset for the
    FBA closure.  Built once per run and shared by every cell.
    """

    grn: SparseGRN
    names: list[str]
    gene_to_reactions: dict[str, tuple[str, ...]]
    source: str
    tf_map: str
    grn_mode: str
    seed: int
    n_edges: int
    active_gene_budget: int = DEFAULT_ACTIVE_BUDGET
    rng_seed: int = field(default=0, repr=False)

    @property
    def n_genes(self) -> int:
        return len(self.names)

    @property
    def index(self) -> dict[str, int]:
        return dict(self.grn._idx)  # noqa: SLF001


def _scale_free_edges(
    names: list[str],
    m: int = 1,
    seed: int = 0,
    rng: random.Random | None = None,
) -> list[tuple[str, str, float]]:
    """Barabási-Albert preferential-attachment edges over ``names``.

    Each new node attaches to ``m`` earlier nodes with probability
    proportional to their current degree, so the degree distribution is
    scale-free (power-law tail ~ -3).  Deterministic for a fixed seed.
    """
    rng = rng or random.Random(seed)
    edges: list[tuple[str, str, float]] = []
    deg = [0] * len(names)

    def add_edge(s: int, t: int, w: float) -> None:
        deg[s] += 1
        deg[t] += 1
        edges.append((names[s], names[t], w))

    add_edge(0, 1, 1.0)
    add_edge(1, 0, 1.0)
    active = [0, 1]
    for i in range(2, len(names)):
        targets: set[int] = set()
        tries = 0
        while len(targets) < m and tries < 64:
            tries += 1
            r = rng.choices(active, weights=[deg[j] for j in active], k=1)[0]
            if r not in targets:
                targets.add(r)
        for t in targets:
            w = rng.uniform(0.4, 1.5)
            if rng.random() < 0.4:
                w = -w
            add_edge(t, i, w)
        active.append(i)
    return edges


def _regulon_layer_edges(
    names: list[str],
    regulators: tuple[str, ...],
    fraction: float,
    weight_hi: float,
    weight_lo: float,
    rng: random.Random,
) -> list[tuple[str, str, float]]:
    """Regulatory layer: each ``regulator`` targets ``fraction`` of targets."""
    edges: list[tuple[str, str, float]] = []
    targets = [n for n in names if n not in regulators]
    for r in regulators:
        n_hit = max(1, int(fraction * len(targets)))
        chosen = rng.sample(targets, n_hit)
        for t in chosen:
            w = rng.uniform(weight_lo, weight_hi)
            edges.append((r, t, w))
    return edges


def build_genome(
    source: str = "synth-4300",
    tf_map: str = "regulon",
    grn_mode: str = "sparse",
    seed: int = 7,
    n_genes: int = DEFAULT_GENES,
    active_gene_budget: int = DEFAULT_ACTIVE_BUDGET,
    noise_seed: int | None = None,
    regulondb: str | None = None,
) -> GenomeSpec:
    """Build a deterministic synthetic genome-scale GRN template.

    Args:
        source: ``synth-4300`` (b-number synthetic genome) or
            ``ecoli-mg1655`` (same builder with the canonical master
            regulators wired on top).
        tf_map: ``regulon`` (hierarchical layers seeded by the literature
            master regulators — crp/fis/lrp/hns on top), ``random``
            (pure scale-free preferential-attachment background, no
            literatur-ordered hubs), or ``regulondb`` (real RegulonDB-
            derived edges replace the synthetic attachment; see
            ``regulondb``).  ``off`` disables the regulatory map.
        grn_mode: ``sparse`` (CSR template; the only mode implemented).
        seed: reproducibility seed for topology and initial state.
        n_genes: number of synthetic b-number genes (default 4300).
        active_gene_budget: per-cell per-tick active-gene budget.
        noise_seed: RNG seed for intrinsic (telegraph) noise on a small
            minority of genes (None = noise off).
        regulondb: RegulonDB network dump (TSV, see
            :func:`parse_regulondb`) used when ``tf_map="regulondb"``;
            when None the curated :data:`REGULONDB_DEMO_EDGES` subset is
            used.

    The engineered ``.helix`` layer can overlay the same gene names
    afterwards; overlapping nodes read/write the background matrix.
    """
    if grn_mode not in ("sparse", "full"):
        raise ValueError(f"grn_mode: expected 'sparse' or 'full', got {grn_mode!r}")
    if tf_map not in ("regulon", "random", "regulondb", "off"):
        raise ValueError(
            "tf_map: expected 'regulon'|'random'|'regulondb'|'off', "
            f"got {tf_map!r}")
    if regulondb is not None and tf_map != "regulondb":
        raise ValueError(
            "regulondb= requires tf_map='regulondb' "
            f"(got tf_map={tf_map!r})")
    rng = random.Random(seed)
    names = [b_number(i + 1) for i in range(n_genes)]
    core = _core_genes()
    # ensure the FBA-gated core genes and the literature master regulators
    # are present as nodes (the latter are the regulon-layer hubs)
    for c in core + list(GLOBAL_REGULATORS):
        if c not in names:
            names.append(c)

    edges: list[tuple[str, str, float]] = []
    if tf_map in ("regulon", "random"):
        # scale-free background (preferential attachment)
        bg_rng = random.Random(seed + 101)
        edges += _scale_free_edges(names, m=1, rng=bg_rng)
    if tf_map == "regulon":
        # hierarchical regulatory layers seeded by the literature hubs
        # (crp/fis/lrp/hns dominate the out-degree ranking)
        edges += _regulon_layer_edges(names, MASTER_REGULATORS,
                                      fraction=0.2, weight_hi=1.5,
                                      weight_lo=0.8, rng=rng)
        edges += _regulon_layer_edges(names, GLOBAL_REGULATORS[4:],
                                      fraction=0.1, weight_hi=1.2,
                                      weight_lo=0.4, rng=rng)
    if tf_map == "regulondb":
        # real-map arm (doc/19 §5.5 C1): RegulonDB-derived edges REPLACE
        # the synthetic attachment; the sparse CSR machinery is kept.
        # Edges referencing genes outside the node set are dropped.
        dumped = (parse_regulondb(regulondb) if regulondb is not None
                  else list(REGULONDB_DEMO_EDGES))
        edges += [(s, t, w) for s, t, w in dumped
                  if s in names and t in names]
    # deterministically deduplicate + drop self loops
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str, float]] = []
    for s, t, w in edges:
        if s == t:
            continue
        if (s, t) in seen:
            continue
        seen.add((s, t))
        unique.append((s, t, w))

    # few hundred core-model genes get a strong driving input so their
    # expression rises above the 0.5 trigger (FBA-gated reactions open)
    for c in core:
        w = rng.uniform(1.2, 2.0)
        if (c, c) not in seen:
            unique.append((c, c, w))
            seen.add((c, c))

    grn = GRN()
    for _i, name in enumerate(names):
        grn.add_gene(name, threshold=0.5, decay=GRN.DECAY)
    for s, t, w in unique:
        grn.add_edge(s, t, w)

    if noise_seed is not None:
        from helixlang.stochastic import TelegraphPromoter
        for _i in range(max(1, int(NOISE_FRACTION * len(names)))):
            n = names[rng.randrange(len(names))]
            if grn.nodes[n].noise is None:
                grn.nodes[n].noise = TelegraphPromoter(
                    k_on=0.5, k_off=2.0, burst_size=4.0)
        grn.noise_enabled = True

    sparse = SparseGRN.from_grn(grn, noise_seed=noise_seed)
    return GenomeSpec(
        grn=sparse,
        names=names,
        gene_to_reactions=dict(ECOLI_CORE_GENE_REACTIONS),
        source=source,
        tf_map=tf_map,
        grn_mode=grn_mode,
        seed=seed,
        n_edges=sparse.n_edges,
        active_gene_budget=active_gene_budget,
        rng_seed=seed,
    )


def outdegrees(spec: GenomeSpec) -> dict[str, int]:
    """Per-gene out-degree of the sparse template (RegulonDB-hub check)."""
    deg: dict[str, int] = {}
    ptr = spec.grn.row_ptr
    for t in range(spec.grn.n_genes):
        for k in range(int(ptr[t]), int(ptr[t + 1])):
            s = spec.names[spec.grn.col_indices[k]]
            deg[s] = deg.get(s, 0) + 1
    return deg


def powerlaw_fit(degrees: list[float]) -> dict:
    """Power-law (log-log) fit of a degree distribution tail.

    Returns ``slope`` (alpha ~ -3 for Barabási-Albert) and ``r2`` of the
    log(degree) vs log(frequency) regression over the support where the
    empirical frequency is positive.
    """
    if not degrees:
        return {"slope": 0.0, "r2": 0.0}
    vals = sorted(degrees)
    unique = sorted(set(vals))
    freq = [sum(1 for v in vals if v == u) for u in unique]
    support = [(u, f) for u, f in zip(unique, freq, strict=True)
               if u > 0 and f > 0]
    if len(support) < 3:
        return {"slope": 0.0, "r2": 0.0}
    xs = np.log([u for u, _ in support])
    ys = np.log([f for _, f in support])
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    r2 = 1.0 - np.sum((ys - pred) ** 2) / max(
        np.sum((ys - np.mean(ys)) ** 2), 1e-12)
    return {"slope": float(slope), "r2": float(r2), "n_unique": len(support)}


class GenomeColony:
    """Whole population of genome-scale cells on one shared template.

    Per-cell expression state is a row of ``levels`` ``(N, G)``; one
    ``step()`` advances every cell (optionally under the active-gene
    budget).  ``triggered`` reports per-cell genes above 0.5 so the
    engineered ``.helix`` layer and the FBA closure can be driven from
    the same matrix.
    """

    def __init__(self, spec: GenomeSpec, n_cells: int,
                 initial_genes: tuple[str, ...] = (),
                 capacity: int | None = None) -> None:
        self.spec = spec
        self.n_cells = n_cells
        # ``capacity`` preallocates rows so a growing population can
        # claim rows for daughters without reallocating the matrix; rows
        # beyond ``n_cells`` start silent.  Without it the matrix grows
        # on demand.
        self._capacity = capacity
        n_rows = capacity if capacity is not None else n_cells
        self.levels = spec.grn.new_state(n_rows, initial_genes=initial_genes)
        self._next_row = n_cells
        self._free_rows: list[int] = []
        self._tick = 0

    def alloc_row(self, copy_of: int | None = None) -> int:
        """Claim a row for a newborn cell; ``copy_of`` inherits a parent
        row's expression state (division), otherwise the row is silent."""
        if self._free_rows:
            row = self._free_rows.pop()
        elif self._next_row < self.levels.shape[0]:
            row = self._next_row
            self._next_row += 1
        else:
            if self._capacity is not None:
                raise RuntimeError(
                    "GenomeColony row capacity exhausted (alive cells "
                    "exceed the preallocated capacity)")
            grow = np.zeros_like(self.levels)
            self.levels = np.concatenate([self.levels, grow], axis=0)
            row = self._next_row
            self._next_row += 1
        if copy_of is not None and copy_of != row:
            self.levels[row] = self.levels[copy_of]
        else:
            self.levels[row] = 0.0
        return row

    def free_row(self, row: int) -> None:
        """Return a dead cell's row to the free pool (zeroed)."""
        if row >= 0:
            self.levels[row] = 0.0
            self._free_rows.append(row)

    def step(self) -> dict:
        """Advance the population one tick; return per-tick statistics."""
        budget = self.spec.active_gene_budget
        self.levels = self.spec.grn.step_budgeted(self.levels, budget=budget)
        self._tick += 1
        trig = self.triggered()
        return {
            "tick": self._tick,
            "n_triggered": int(trig.sum()),
            "active_mean": float(self.spec.grn.n_active(self.levels).mean()),
            "triggered_genes": sorted(
                {self.spec.names[i] for i in range(self.spec.grn.n_genes)
                 if trig[:, i].any()}),
        }

    def triggered(self, threshold: float = 0.5) -> np.ndarray:
        """Boolean ``(N, G)`` mask of genes above ``threshold``."""
        return self.spec.grn.triggered(self.levels, threshold)

    def knock_out(self, genes: list[str]) -> None:
        """Force ``genes`` to silent (level 0) — never-triggered nodes."""
        for g in genes:
            i = self.spec.grn._idx.get(g)  # noqa: SLF001
            if i is not None:
                self.levels[:, i] = 0.0

    def core_gene_levels(self, cell: int) -> dict[str, float]:
        """Expression of the FBA-gated core-model genes in ``cell``."""
        return {g: float(self.levels[cell, self.spec.grn._idx[g]])  # noqa: SLF001
                for g in self.spec.gene_to_reactions}

    def fba_biomass(self, cell: int, uptake_glc: float = 10.0) -> float:
        """FBA biomass for ``cell`` with expression-gated reaction bounds.

        A reaction's upper bound is opened only when *every* gene that
        gates it is expressed above 0.5 (AND logic, the ``ko_model``
        semantics); otherwise its flux is clamped to the lower bound.
        This is the "expression trigger -> reaction on/off" closure.
        """
        levels = self.core_gene_levels(cell)
        gated: set[str] = set()
        for g, rxns in self.spec.gene_to_reactions.items():
            if levels.get(g, 0.0) <= 0.5:
                gated.update(rxns)
        return _gated_biomass(tuple(gated), uptake_glc)


def _gated_biomass(off_reactions: tuple[str, ...],
                   uptake_glc: float) -> float:
    """Solve FBA biomass with ``off_reactions`` clamped to lower bound."""
    from helixlang.apps.whole_cell_scale import ko_model

    if not off_reactions:
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    else:
        fba = FluxBalanceAnalysis(ko_model(ECOLI_CORE_MODEL, off_reactions))
    fba.set_uptake("GLC", uptake_glc)
    return fba.solve().get(ECOLI_CORE_MODEL.biomass_reaction or "BIOMASS", 0.0)


def expression_gated_biomass(
    gene_levels: dict[str, float],
    uptake_glc: float = 10.0,
    gene_to_reactions: dict[str, tuple[str, ...]] | None = None,
) -> float:
    """FBA biomass from a ``{gene: level}`` expression map.

    Deterministic (no RNG); the closure used by the essentiality tests.
    """
    g2r = gene_to_reactions or ECOLI_CORE_GENE_REACTIONS
    off: set[str] = set()
    for g, rxns in g2r.items():
        if gene_levels.get(g, 0.0) <= 0.5:
            off.update(rxns)
    return _gated_biomass(tuple(off), uptake_glc)


__all__ = [
    "DEFAULT_GENES",
    "DEFAULT_ACTIVE_BUDGET",
    "MASTER_REGULATORS",
    "GLOBAL_REGULATORS",
    "REGULONDB_DEMO_EDGES",
    "NOISE_FRACTION",
    "b_number",
    "GenomeSpec",
    "build_genome",
    "parse_regulondb",
    "outdegrees",
    "powerlaw_fit",
    "GenomeColony",
    "expression_gated_biomass",
    "ESSENTIALITY_FLUX_TOL",
]
