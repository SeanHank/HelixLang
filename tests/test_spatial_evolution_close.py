"""Coverage-closure tests for :mod:`helixlang.plugins.apps.spatial_evolution`.

Exercises the ``mean_fitness`` / ``max_fitness`` / ``best_genome`` query
methods and the empty-population fallback of ``best_genome``.
"""
from helixlang.plugins.apps.spatial_evolution import (
    SpatialEvolution,
    SpatialEvolutionConfig,
)


class TestSpatialEvolutionClosure:
    def test_mean_and_max_fitness_records_row(self):
        ev = SpatialEvolution(SpatialEvolutionConfig(
            population_size=8, generations=1, colonization_ticks=10,
            seed=7))
        row = ev.step()
        assert ev.mean_fitness() == row["mean_fitness"]
        assert ev.max_fitness() == row["max_fitness"]

    def test_best_genome_returns_fittest_of_population(self):
        ev = SpatialEvolution(SpatialEvolutionConfig(
            population_size=8, generations=1, colonization_ticks=10,
            seed=7))
        ev.step()
        best = ev.best_genome()
        assert best in ev.population
        assert len(best) == ev.config.genome_length_nt

    def test_best_genome_empty_population(self):
        ev = SpatialEvolution(SpatialEvolutionConfig(
            population_size=0, generations=0))
        assert ev.best_genome() == ""
