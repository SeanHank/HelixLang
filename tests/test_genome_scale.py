"""Genome-scale GRN + FBA closure tests (doc/18-programmable-cell-population-simulation.md §13 Design 5, acceptance gates).

Gates verified (doc/18-programmable-cell-population-simulation.md §771-782):

1. **Bitwise consistency** — ``SparseGRN.step`` reproduces scalar ``GRN.step``
   output levels exactly on a ~10^4-edge network; the active-gene budget
   path matches up to the dropped near-silent contributions.
2. **Essentiality closure** — a never-triggered (silent) essential gene
   gates its reactions shut in the FBA core model: biomass -> 0; a
   non-essential gene knockout still grows (EcoCyc glucose-minimal
   labels, Gerdes 2003; the ``whole_cell_scale`` reference).
3. **Performance gate** — 4300 genes / ~10^4 sparse edges / 10^4 cells,
   budgeted step under 1 s per tick (vs the ~148 MB dense matrix/cell).
4. **Network structure** — the regulon template puts crp/fis/lrp/hns at
   the top of the out-degree ranking; the random template's degree
   distribution is scale-free (Barabási-Albert slope ~ -3).
5. **Noise fidelity** — telegraph (Fano) noise on a gene minority still
   reproduces the scalar ``GRN.step`` statistics at genome scale.
"""
from __future__ import annotations

import random
import time

import pytest

from helixlang.apps.genome_scale import (
    MASTER_REGULATORS,
    REGULONDB_DEMO_EDGES,
    GenomeColony,
    _core_genes,
    build_genome,
    expression_gated_biomass,
    outdegrees,
    parse_regulondb,
    powerlaw_fit,
)
from helixlang.apps.whole_cell_scale import ESSENTIALITY_FLUX_TOL
from helixlang.grn import GRN
from helixlang.sparse_grn import SparseGRN

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

NP_REQUIRED = pytest.mark.skipif(np is None, reason="numpy required")


def _random_network(n_genes: int, n_edges: int, seed: int
                    ) -> tuple[list[str], list[tuple[str, str, float]]]:
    rng = random.Random(seed)
    names = [f"g{i:04d}" for i in range(n_genes)]
    unique: set[tuple[str, str]] = set()
    while len(unique) < n_edges:
        s, t = rng.choice(names), rng.choice(names)
        if s != t:
            unique.add((s, t))
    edges = [(s, t, rng.uniform(0.4, 1.5) * (-1.0 if rng.random() < 0.4 else 1.0))
             for s, t in unique]
    return names, edges


def _scalar_trajectory(grn: GRN, init, n_steps: int,
                       noise_enabled: bool = False,
                       noise_seed: int = 0,
                       copy_noise: bool = False) -> list[list[float]]:
    rows: list[list[float]] = []
    names = list(grn.nodes)
    for ci, row in enumerate(init):
        gg = GRN(noise_enabled=noise_enabled,
                 noise_seed=noise_seed + ci)
        for name, node in grn.nodes.items():
            gg.add_gene(name, node.threshold, decay=node.decay,
                        hill_n=node.hill_n, kd=node.kd,
                        noise=node.noise if copy_noise else None)
        for e in grn.edges:
            gg.add_edge(e.source, e.target, e.weight)
        for name, lv in zip(names, row, strict=True):
            gg.nodes[name].level = lv
        for _ in range(n_steps):
            gg.step()
        rows.append([gg.nodes[n].level for n in names])
    return rows


# ---------------------------------------------------------------------------
# gate 1: bitwise consistency vs scalar GRN.step
# ---------------------------------------------------------------------------
@NP_REQUIRED
class TestSparseBitwise:
    def test_exact_matches_scalar(self):
        names, edges = _random_network(150, 8000, seed=3)
        grn = GRN()
        for n in names:
            grn.add_gene(n, 0.5)
        for s, t, w in edges:
            grn.add_edge(s, t, w)
        sg = SparseGRN.from_grn(grn, noise_seed=1)

        rng = random.Random(9)
        init = np.array(
            [[rng.random() if rng.random() < 0.3 else 0.0 for _ in names]
             for _ in range(8)], dtype=float)
        levels = init.copy()
        for _ in range(25):
            levels = sg.step(levels)
        scalar = np.array(_scalar_trajectory(grn, init, 25))
        assert np.allclose(levels, scalar, atol=1e-12)
        assert np.abs(levels - scalar).max() <= 2e-15

    def test_budgeted_approximates_exact(self):
        names, edges = _random_network(150, 8000, seed=5)
        grn = GRN()
        for n in names:
            grn.add_gene(n, 0.5)
        for s, t, w in edges:
            grn.add_edge(s, t, w)
        sg = SparseGRN.from_grn(grn, noise_seed=1)
        rng = random.Random(2)
        init = np.array(
            [[rng.random() if rng.random() < 0.3 else 0.0 for _ in names]
             for _ in range(8)], dtype=float)
        exact = init.copy()
        budgeted = init.copy()
        for _ in range(25):
            exact = sg.step(exact)
            budgeted = sg.step_budgeted(budgeted, budget=64)
        # dropped near-silent contributions stay small (doc/18-programmable-cell-population-simulation.md §728)
        assert np.abs(budgeted - exact).max() < 0.2


# ---------------------------------------------------------------------------
# gate 2: essentiality closure (expression -> reaction on/off -> FBA)
# ---------------------------------------------------------------------------
@NP_REQUIRED
class TestEssentialityClosure:
    def test_expression_map_closure(self):
        on = {g: 1.0 for g in _core_genes()}
        wt = expression_gated_biomass(on)
        assert wt > 0.1
        ko_pgi = expression_gated_biomass({**on, "pgi": 0.0})
        assert ko_pgi < ESSENTIALITY_FLUX_TOL
        ko_ldh = expression_gated_biomass({**on, "ldhA": 0.0})
        assert ko_ldh > 0.1
        # default map: genes absent from the map are not expressed -> shut
        assert expression_gated_biomass({}) < ESSENTIALITY_FLUX_TOL

    def test_colony_knockout_silent_node(self):
        spec = build_genome(n_genes=300, tf_map="regulon", seed=7,
                            noise_seed=11)
        colony = GenomeColony(spec, n_cells=32,
                              initial_genes=tuple(_core_genes()))
        for _ in range(12):
            colony.step()
        assert float(colony.levels[0, spec.grn._idx["pgi"]]) > 0.5
        assert colony.fba_biomass(0) > 0.1
        colony.knock_out(["pgi"])  # never-triggered silent node
        for _ in range(4):
            colony.step()
        assert float(colony.levels[0, spec.grn._idx["pgi"]]) < 0.5
        assert colony.fba_biomass(0) < ESSENTIALITY_FLUX_TOL


@NP_REQUIRED
class TestColonyRowManagement:
    """Division row allocation (task 3): daughters claim rows within the
    preallocated capacity and inherit the parent's expression state."""

    def test_alloc_row_within_capacity(self):
        spec = build_genome(n_genes=120, tf_map="off", seed=7)
        colony = GenomeColony(spec, n_cells=4, capacity=8)
        assert colony.levels.shape == (8, spec.n_genes)
        # fresh rows are silent
        r = colony.alloc_row()
        assert r == 4
        assert not colony.levels[r].any()
        # copy_of inherits the parent row's expression
        colony.levels[0] = 0.75
        r2 = colony.alloc_row(copy_of=0)
        assert float(colony.levels[r2].mean()) == pytest.approx(0.75)

    def test_free_row_recycled_and_zeroed(self):
        spec = build_genome(n_genes=120, tf_map="off", seed=7)
        colony = GenomeColony(spec, n_cells=4, capacity=8)
        colony.levels[1] = 0.9
        colony.free_row(1)
        # the recycled row is zeroed and reused by the next alloc
        r = colony.alloc_row()
        assert r == 1
        assert not colony.levels[r].any()

    def test_capacity_exhaustion_raises(self):
        spec = build_genome(n_genes=120, tf_map="off", seed=7)
        colony = GenomeColony(spec, n_cells=4, capacity=4)
        with pytest.raises(RuntimeError, match="capacity"):
            colony.alloc_row()


# ---------------------------------------------------------------------------
# gate 4: network structure (regulon hubs + scale-free background)
# ---------------------------------------------------------------------------
@NP_REQUIRED
class TestNetworkStructure:
    def test_regulon_master_regulators_top4(self):
        spec = build_genome(n_genes=4300, tf_map="regulon", seed=7)
        deg = outdegrees(spec)
        top4 = [g for g, _ in sorted(deg.items(), key=lambda kv: -kv[1])[:4]]
        assert sorted(top4) == sorted(MASTER_REGULATORS)
        for m in MASTER_REGULATORS:
            assert deg[m] > 0.1 * len(spec.names)

    def test_random_background_scale_free(self):
        spec = build_genome(n_genes=4300, tf_map="random", seed=7)
        assert 3000 < spec.n_edges < 20000
        # weighted in+out sum per gene, from the pure-numpy CSR arrays so
        # the gate runs without scipy (SparseGRN._csr is only populated
        # when scipy is installed)
        data, cols, rp = spec.grn.data, spec.grn.col_indices, spec.grn.row_ptr
        deg = (np.add.reduceat(data, rp[:-1])
               + np.bincount(cols, weights=data, minlength=spec.n_genes))
        fit = powerlaw_fit([int(d) for d in deg])
        assert -4.0 < fit["slope"] < -2.0
        assert fit["r2"] > 0.9

    def test_regulondb_map_replaces_synthetic_attachment(self):
        """doc/19 §5.5 C1: tf_map='regulondb' uses only RegulonDB edges.

        The curated map lands on the sparse CSR template (crp at the top
        of the out-degree ranking, both + and - weights), and there is no
        scale-free background: every edge is a RegulonDB interaction or a
        core self-drive loop.
        """
        spec = build_genome(n_genes=4300, tf_map="regulondb", seed=7)
        assert spec.tf_map == "regulondb"
        assert spec.grn_mode == "sparse"
        # every curated edge made it into the template
        deg = outdegrees(spec)
        assert deg["crp"] >= sum(1 for s, _, _ in REGULONDB_DEMO_EDGES
                                 if s == "crp")
        assert deg["arcA"] >= 1 and deg["fnr"] >= 1
        # both activation and repression survive the import
        assert spec.grn.data.min() < 0 < spec.grn.data.max()
        # no Barabási-Albert background: n_edges == curated (unique, in
        # node set) + one self-drive per core gene
        curated = {(s, t) for s, t, _ in REGULONDB_DEMO_EDGES}
        core = set(_core_genes())
        expected = len(curated) + len(core)
        assert spec.n_edges == expected
        # crp is the top hub of the real-map template
        top = sorted(deg.items(), key=lambda kv: -kv[1])[0][0]
        assert top == "crp"

    def test_parse_regulondb_dump_and_import(self):
        dump = (
            "#RegulonDB network export\n"
            "regulator\ttarget\teffect\n"
            "crp\tgltA\t+\n"
            "crp\tzwf\t+0.8\n"
            "arcA\ticdA\t-\n"
            "fnr\tldhA\t+1.2\n"
            "crp\tnoSuchGene\t+\n"  # dropped: not in the node set
        )
        edges = parse_regulondb(dump)
        assert edges == [("crp", "gltA", 1.0), ("crp", "zwf", 0.8),
                         ("arcA", "icdA", -1.0), ("fnr", "ldhA", 1.2),
                         ("crp", "noSuchGene", 1.0)]
        spec = build_genome(n_genes=4300, tf_map="regulondb",
                            regulondb=dump, seed=7)
        assert spec.tf_map == "regulondb"
        # the four in-node-set edges are present; the unknown target is not
        assert (spec.grn.data.shape[0] == 4 + len(_core_genes()))

    def test_regulondb_requires_tf_map(self):
        with pytest.raises(ValueError, match="requires tf_map='regulondb'"):
            build_genome(n_genes=4300, tf_map="regulon", regulondb="crp\tgltA\t+\n")


# ---------------------------------------------------------------------------
# gate 5: noise fidelity at genome scale
# ---------------------------------------------------------------------------
@NP_REQUIRED
class TestNoiseFidelity:
    def test_noise_matches_scalar_statistics(self):
        names, edges = _random_network(120, 6000, seed=3)
        grn = GRN()
        for n in names:
            grn.add_gene(n, 0.5)
        for s, t, w in edges:
            grn.add_edge(s, t, w)
        from helixlang.stochastic import TelegraphPromoter

        for n in names[:24]:
            grn.nodes[n].noise = TelegraphPromoter(0.5, 2.0, 3.0)
        grn.noise_enabled = True
        sg = SparseGRN.from_grn(grn, noise_seed=42)

        n_cells = 4000
        # a small seed set ON drives the network off zero (otherwise the
        # all-silent attractor keeps every variance ~ 0)
        seed_genes = tuple(names[48:56])
        init = sg.new_state(n_cells, initial_genes=seed_genes)
        levels = init.copy()
        for _ in range(15):
            levels = sg.step(levels)

        rng = random.Random(1)
        sub = sorted(rng.sample(range(n_cells), 120))
        scalar = np.array(_scalar_trajectory(
            grn, np.asarray(init)[sub], 15,
            noise_enabled=True, noise_seed=123, copy_noise=True))

        noisy = [names.index(n) for n in names[:24]]
        quiet = [names.index(n) for n in names[24:48]]
        # quiet genes track the same deterministic mean in both paths
        assert abs(levels[sub][:, quiet].mean()
                   - scalar[:, quiet].mean()) < 0.01
        # telegraph noise: the vectorized path reproduces the scalar
        # variance (both draw zero-mean noise with the same per-gene std)
        var_vec = levels[sub][:, noisy].var()
        var_sc = scalar[:, noisy].var()
        assert var_vec > 0.0 and var_sc > 0.0
        assert abs(var_vec - var_sc) < 0.5 * var_sc + 1e-4


# ---------------------------------------------------------------------------
# gate 3: performance (4300 genes, ~10^4 edges, 10^4 cells, < 1 s/tick)
# ---------------------------------------------------------------------------
@NP_REQUIRED
class TestPerformance:
    def test_budgeted_tick_10k_cells_under_second(self):
        spec = build_genome(n_genes=4300, tf_map="regulon", seed=7)
        n_cells = 10_000
        rng = np.random.default_rng(0)
        # doc premise: only ~10% of genes significantly transcribed at any
        # instant (Martínez-Antonio et al. 2008)
        active_cols = rng.choice(spec.n_genes, 430, replace=False)
        levels = np.zeros((n_cells, spec.n_genes))
        levels[:, active_cols] = rng.random((n_cells, active_cols.size))

        t0 = time.perf_counter()
        spec.grn.step_budgeted(levels, budget=spec.active_gene_budget)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"budgeted tick took {elapsed:.3f}s"

    def test_exact_csr_tick_10k_cells_under_second(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")
        spec = build_genome(n_genes=4300, tf_map="regulon", seed=7)
        n_cells = 10_000
        rng = np.random.default_rng(1)
        active_cols = rng.choice(spec.n_genes, 430, replace=False)
        levels = np.zeros((n_cells, spec.n_genes))
        levels[:, active_cols] = rng.random((n_cells, active_cols.size))
        t0 = time.perf_counter()
        spec.grn.step(levels)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"exact CSR tick took {elapsed:.3f}s"


@NP_REQUIRED
class TestPopulationIntegration:
    """Design 5 tasks 3/4: population wiring + expression-gated FBA closure.

    Every cell's expression is a row of one shared matrix (one vectorized
    step per tick); the per-cell dFBA batch gates reaction upper bounds by
    that row, so a knocked-out gene (a never-triggered silent node)
    collapses growth to zero while a non-essential knockout keeps growing.
    """

    @pytest.fixture()
    def pop(self):
        from helixlang.codon_table import STANDARD_TABLE
        from helixlang.compiler import Compiler
        from helixlang.environment import Environment, EnvironmentConfig
        from helixlang.lexer import Lexer
        from helixlang.parser import Parser
        from helixlang.population import (
            CellPopulation3D,
            PopulationCell,
            PopulationConfig,
        )
        from helixlang.seq_utils import stop_codons_from_table

        src = """#promoter name=p_housekeeping strength=-0.4
#gene name=adhE promoter=p_housekeeping
ATG GCT GGT GCT TAA
#end
"""
        prog = Parser(
            list(Lexer(src).tokens()),
            stop_codons=stop_codons_from_table(STANDARD_TABLE)).parse()
        chunk = Compiler(STANDARD_TABLE).compile(prog)
        spec = build_genome(n_genes=300, tf_map="regulon", seed=7)
        cfg = PopulationConfig(
            max_size=32, grid_width=8, grid_height=8, grid_depth=1,
            division_threshold=1.8e9, death_threshold=1e6,
            signaling_enabled=False, dfba_enabled=True, dfba_dt_h=0.1,
            program=prog, chunk=chunk, ops_per_tick=prog.config.ops_per_tick,
            genome=spec)
        cfg.environment = Environment(EnvironmentConfig(width=8, height=8))
        cells = [PopulationCell(id=i, x=3, y=3, energy=2.0e9) for i in range(12)]
        return CellPopulation3D(cells, cfg)

    @staticmethod
    def _mean_growth(pop, ticks: int = 15) -> float:
        rates: list[float] = []
        for _ in range(ticks):
            pop.step()
            for c in pop.cells:
                if c.alive and c.dfba is not None and c.dfba.history:
                    rates.append(c.dfba.growth_rate)
        return sum(rates) / len(rates) if rates else 0.0

    def test_shared_matrix_rows_assigned(self, pop):
        assert pop._genome_colony is not None  # noqa: SLF001
        colony = pop._genome_colony
        assert colony.levels.shape[0] == 32
        assert colony.levels.shape[1] == colony.spec.n_genes
        assert [c.genome_row for c in pop.cells] == list(range(12))
        stats = pop.get_statistics()
        assert stats["genome_genes"] == colony.spec.n_genes
        assert stats["triggered_genes"] > 0

    def test_division_claims_rows(self, pop):
        # high-energy cells divide; rows stay within the preallocated capacity
        for _ in range(3):
            pop.step()
        assert sum(1 for c in pop.cells if c.alive) > 12
        assert all(0 <= c.genome_row < 32 for c in pop.cells if c.alive)

    def test_wt_grows_pgi_knockout_collapses(self, pop):
        wt = self._mean_growth(pop)
        assert wt > 0.05
        pop._genome_colony.knock_out(["pgi"])  # noqa: SLF001
        ko = self._mean_growth(pop)
        assert ko < 1e-6

    def test_nonessential_knockout_still_grows(self, pop):
        pop._genome_colony.knock_out(["ldhA"])  # noqa: SLF001
        assert self._mean_growth(pop) > 0.05
