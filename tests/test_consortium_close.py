"""Coverage-closure tests for :mod:`helixlang.plugins.apps.consortium`.

Exercises the ``add_cells`` validation guards, out-of-bounds emission/sensing
skips, the zero-diffusion and zero-decay branches, sensor-fraction and
survivor edge cases, the observation queries, and the ``run_consortium``
report convenience wrapper.
"""
import pytest

from helixlang.core.errors import BioError
from helixlang.plugins.apps.consortium import (
    ROLE_ACTUATOR,
    ROLE_PRODUCER,
    ROLE_SENSOR,
    ConsortiumConfig,
    ConsortiumReport,
    ConsortiumSimulator,
    run_consortium,
)


class TestConsortiumClosure:
    def _sim(self, **kw) -> ConsortiumSimulator:
        return ConsortiumSimulator(ConsortiumConfig(**kw))

    def test_unknown_role_rejected(self):
        with pytest.raises(BioError, match="unknown consortium role"):
            self._sim().add_cells(2, "borg")

    def test_stack_requires_coordinates(self):
        with pytest.raises(BioError, match="stack=True requires"):
            self._sim().add_cells(2, ROLE_PRODUCER, stack=True)

    def test_explicit_position_not_stack(self):
        sim = self._sim(grid_width=10, grid_height=10)
        sim.add_cells(1, ROLE_PRODUCER, x=3, y=4)
        assert (sim.cells[0].x, sim.cells[0].y) == (3, 4)

    def test_out_of_bounds_producer_emission_skipped(self):
        sim = self._sim(grid_width=40, grid_height=40)
        sim.add_cells(1, ROLE_PRODUCER, 5, 5, stack=True)
        sim.add_cells(1, ROLE_PRODUCER, 999, 999, stack=True)
        sim.step()
        assert sim.cells[0].signal_emitted > 0.0
        assert sim.cells[1].signal_emitted == 0.0

    def test_no_decay_copies_field(self):
        sim = self._sim(signal_decay_per_tick=0.0,
                        signal_diffusion_um2_s=0.0)
        sim.add_cells(1, ROLE_PRODUCER, 0, 0, stack=True)
        sim.step()
        assert sim.signal_field[0][0] > 0.0

    def test_out_of_bounds_sensor_skipped(self):
        sim = self._sim(signal_threshold_um=0.0)
        sim.add_cells(1, ROLE_SENSOR, -5, -5, stack=True)
        sim.step()
        assert not sim.cells[0].decided

    def test_sensor_fraction_with_no_sensors(self):
        sim = self._sim()
        sim.add_cells(1, ROLE_PRODUCER)
        sim.step()
        assert sim.alive_count(ROLE_SENSOR) == 0
        assert sim.run(1)[-1]["consensus_fraction"] == 0.0

    def test_cell_death_removes_survivor(self):
        sim = self._sim(metabolic_cost=0.0,
                        energy_intake={ROLE_PRODUCER: 0.0})
        sim.add_cells(1, ROLE_PRODUCER, 0, 0, stack=True)
        sim.cells[0].energy = -5.0
        sim.step()
        assert sim.cells == []

    def test_role_fractions_empty_sim(self):
        sim = self._sim()
        assert sim.role_fractions() == {r: 0.0 for r in (
            ROLE_PRODUCER, ROLE_SENSOR, ROLE_ACTUATOR)}

    def test_alive_count_by_role(self):
        sim = self._sim()
        sim.add_cells(3, ROLE_PRODUCER, stack=False)
        sim.add_cells(2, ROLE_SENSOR, x=1, y=1)
        assert sim.alive_count(ROLE_PRODUCER) == 3
        assert sim.alive_count(ROLE_SENSOR) == 2
        assert sim.alive_count(ROLE_ACTUATOR) == 0

    def test_mean_signal_at_filters_bounds(self):
        sim = self._sim(grid_width=10, grid_height=10,
                        signal_decay_per_tick=0.0)
        sim.add_cells(1, ROLE_PRODUCER, 2, 2, stack=True)
        for _ in range(3):
            sim.step()
        assert sim.mean_signal_at(sim.cells) > 0.0
        assert sim.mean_signal_at([]) == 0.0
        from helixlang.plugins.apps import consortium as _mod
        stray = _mod.ConsortiumCell(role=ROLE_SENSOR, x=12345, y=12345)
        assert sim.mean_signal_at([stray]) == 0.0

    def test_run_consortium_report(self):
        report = run_consortium(
            config=ConsortiumConfig(grid_width=12, grid_height=12,
                                    signal_decay_per_tick=0.1),
            initial={ROLE_PRODUCER: 4, ROLE_SENSOR: 4, ROLE_ACTUATOR: 4},
            n_ticks=6)
        assert isinstance(report, ConsortiumReport)
        assert report.ticks == 6
        assert report.alive > 0
        assert set(report.composition) == {
            ROLE_PRODUCER, ROLE_SENSOR, ROLE_ACTUATOR}
