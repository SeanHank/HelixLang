"""Synthetic consortium tests: quorum consensus + composition control (S1).

Verification goals:
- Density-dependent consensus: a colony below the critical density stays
  below the signal threshold (no decision), a colony above it flips
  (You et al. 2004 Nature 428:868; di Bernardo 2026 arXiv:2602.19666).
- The quorum threshold separates decisions deterministically on a
  point-source colony with first-order signal decay.
- Actuators produce output only after the consensus is reached.
- Proportional ratio feedback drives the composition to the configured
  target (Mee & Wang 2012; McCarty & Ledesma-Amaro 2019).
- The rendered .helix consensus circuit parses/compiles, and the real
  CellPopulation pipeline reproduces the sparse-OFF / dense-ON switch.
"""
from __future__ import annotations

from helixlang.apps.consortium import (
    ROLE_ACTUATOR,
    ROLE_PRODUCER,
    ROLE_SENSOR,
    ROLES,
    ConsortiumConfig,
    ConsortiumSimulator,
    make_consortium_helix,
    run_consortium_quorum,
)

CENTER = 15


def _density_sim(producers: int, threshold: float = 10.0,
                 grid: int = 30) -> ConsortiumSimulator:
    """Point-source colony: producers+sensors stacked at the centre.

    Steady-state centre signal with e = 1.0 µM/tick, decay 0.1/tick and
    D_lattice ~ 0.09 is c = e*N / (decay + 4*D_lattice) = N / 0.46, so
    N = 3 (~6.5 µM) sits below a 10 µM threshold while N = 10 (~22 µM)
    crosses it.
    """
    sim = ConsortiumSimulator(ConsortiumConfig(
        grid_width=grid,
        grid_height=grid,
        signal_threshold_um=threshold,
        emission_um_per_tick=1.0,
        signal_decay_per_tick=0.1,
        signal_diffusion_um2_s=0.15,   # D_lattice ~ 0.09
        metabolic_cost=0.0,
        energy_intake={"producer": 0.0, "sensor": 0.0, "actuator": 0.0},
        division_threshold=1e15,       # freeze growth: probe consensus only
        max_size=1_000_000,
    ))
    sim.add_cells(producers, ROLE_PRODUCER, CENTER, CENTER, stack=True)
    sim.add_cells(10, ROLE_SENSOR, CENTER, CENTER, stack=True)
    sim.add_cells(5, ROLE_ACTUATOR, CENTER, CENTER, stack=True)
    return sim


def test_dense_colony_reaches_consensus() -> None:
    sim = _density_sim(producers=10)
    last = sim.run(200)[-1]
    assert last["consensus_fraction"] == 1.0
    assert last["consensus_reached"] == 1.0


def test_sparse_colony_stays_below_consensus() -> None:
    sim = _density_sim(producers=3)
    last = sim.run(200)[-1]
    assert last["consensus_fraction"] == 0.0
    assert last["consensus_reached"] == 0.0
    assert last["max_signal"] < 10.0


def test_higher_threshold_needs_higher_density() -> None:
    # 10 producers (c ~ 22 µM) cross a 10 µM threshold but not 25 µM.
    sim_lo = _density_sim(producers=10, threshold=10.0)
    sim_hi = _density_sim(producers=10, threshold=25.0)
    assert sim_lo.run(200)[-1]["consensus_reached"] == 1.0
    assert sim_hi.run(200)[-1]["consensus_reached"] == 0.0


def test_actuator_output_only_after_consensus() -> None:
    dense = _density_sim(producers=10)
    last_dense = dense.run(200)[-1]
    assert last_dense["output_rate"] > 0.0
    assert last_dense["cumulative_output"] > 0.0
    sparse = _density_sim(producers=3)
    last_sparse = sparse.run(200)[-1]
    assert last_sparse["cumulative_output"] == 0.0


def test_ratio_control_converges_to_target() -> None:
    target = {ROLE_PRODUCER: 0.5, ROLE_SENSOR: 0.25, ROLE_ACTUATOR: 0.25}
    sim = ConsortiumSimulator(ConsortiumConfig(
        grid_width=40,
        grid_height=40,
        signal_decay_per_tick=0.1,
        metabolic_cost=1.0e7,
        max_size=20000,
        ratio_control_gain=2.0,
        target_ratios=target,
        seed=7,
    ))
    for role in ROLES:
        sim.add_cells(20, role)
    last = sim.run(500)[-1]
    for role, target_frac in target.items():
        assert abs(last[f"{role}_fraction"] - target_frac) < 0.05, (
            f"{role} fraction {last[f'{role}_fraction']:.3f} "
            f"target {target_frac}")


def test_make_consortium_helix_compiles() -> None:
    from helixlang.codon_table import STANDARD_TABLE
    from helixlang.compiler import Compiler
    from helixlang.lexer import Lexer
    from helixlang.parser import Parser
    from helixlang.semantic import SemanticAnalyzer

    src = make_consortium_helix()
    assert "ATG TCA TAA" in src
    program = Parser(list(Lexer(src).tokens())).parse()
    SemanticAnalyzer(program).check()
    Compiler(STANDARD_TABLE).compile(program)


def test_run_consortium_quorum_density_switch() -> None:
    # Mirrors examples/21_quorum_circuit.helix: 1-4 cells stay OFF,
    # a 9x9 (81-cell) colony crosses the threshold.
    assert run_consortium_quorum(1) is False
    assert run_consortium_quorum(2) is False
    assert run_consortium_quorum(9) is True
