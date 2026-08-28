"""Population DBTL loop tests (doc/19-whole-organism-lifecycle-simulation.md §5.6 D3).

The hard gate: ``apps/population_dbtl.py`` + a test proving a designed
strain is improved by the loop -- the round-0 ``synbio_designer`` strain's
growth strictly rises across design→build→test→learn rounds.
"""
from __future__ import annotations

from itertools import pairwise

from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.plugins.apps.population_dbtl import (
    DbtlConfig,
    PopulationDbtl,
    build_species,
    designed_strain,
    learn_surrogate,
    run_dbtl,
)
from helixlang.plugins.apps.population_dbtl import test_strain as strain_growth
from helixlang.sim_runtime import run as sim_run


def parse(src: str):
    return Parser(list(Lexer(src).tokens())).parse()


def test_designed_strain_is_improved_by_the_loop():
    """D3 gate: the designed strain's growth improves across rounds."""
    result = PopulationDbtl(DbtlConfig(n_rounds=4)).run()
    assert result["improved"] is True
    assert result["final_growth"] > result["round0_growth"]
    assert result["fold_improvement"] > 1.0
    bests = [r["best_growth"] for r in result["rounds"]]
    # elitism -> best_growth never decreases round over round
    for b, b2 in pairwise(bests):
        assert b <= b2


def test_design_round0_is_a_synbio_designer_construct():
    """The round-0 design is literally synbio_designer DNA (D3 Design)."""
    cfg = DbtlConfig()
    genome = designed_strain(cfg)
    assert len(genome) == cfg.genome_length_nt
    assert set(genome) <= set("ACGT")
    # a buildable species decodes real traits from the designed DNA
    sp = build_species(genome, cfg)
    assert sp.traits.uptake_gain >= 0.5
    assert sp.consumption == {"glucose": (cfg.vmax, cfg.ks)}
    # the loop's round-0 population is seeded from the designed strain
    loop = PopulationDbtl(cfg)
    assert designed_strain(cfg) in loop.design(0)


def test_growth_landscape_monotonic_in_traits():
    """Test-stage growth is monotonic in the growth-trait windows."""
    cfg = DbtlConfig()
    hi = strain_growth("T" * cfg.genome_length_nt, cfg)
    mid = strain_growth("GACTCGATAGCTAGCATCGATGCTAGCAT", cfg)
    lo = strain_growth("A" * cfg.genome_length_nt, cfg)
    assert hi > mid > lo


def test_surrogate_recovers_positive_trait_slopes():
    """Learn-stage surrogate fits positive slopes on growth traits."""
    base = "A" * 30
    windows = {
        "uptake_gain": (0, 5),
        "growth_rate_gain": (5, 10),
        "yield_c": (10, 15),
    }
    genomes = [base]
    for _t, (lo, hi) in windows.items():
        gnm = "A" * lo + "T" * (hi - lo) + "A" * (30 - hi)
        genomes.append(gnm)
    growths = [strain_growth(g, DbtlConfig()) for g in genomes]
    # each single-window-high genome grows faster than the baseline
    assert growths[1] > growths[0] and growths[2] > growths[0]
    assert growths[3] > growths[0]
    model = learn_surrogate(genomes, growths, n_samples=300, seed=1)
    assert model["best_trait"] in model["growth_traits"]
    for t in ("uptake_gain", "growth_rate_gain", "yield_c"):
        assert model["params"][f"p_{t}"] > 0.0
    pred = model["surrogate"](**model["params"])
    for i in range(1, 4):
        assert pred[i] > pred[0]

def test_run_dbtl_deterministic_with_seed():
    a = run_dbtl(seed=3, n_rounds=3)
    b = run_dbtl(seed=3, n_rounds=3)
    assert a["rounds"] == b["rounds"]
    assert a["designed_strain"] == b["designed_strain"]


def test_sim_runtime_dispatch_population_dbtl():
    """``#sim kind=population_dbtl`` dispatches and reports ``improved``."""
    src = """
#config backend=ecosystem
#sim kind=population_dbtl n_rounds=3 population_size=4 seed=0
#sim output=round,best_growth,mean_growth
"""
    result = sim_run(parse(src))
    assert result is not None
    assert result.backend == "population_dbtl"
    assert len(result.rows) == 3
    assert result.meta["improved"] is True
    assert result.meta["round0_growth"] < result.meta["final_growth"]
