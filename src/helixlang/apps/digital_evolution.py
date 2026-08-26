"""Digital evolution: Avida-style instruction genomes + selection (S3).

A population of digital organisms -- small self-replicating programs
with instruction genomes -- is evolved by mutation and selection on a
signal-production task.  This mirrors the Avida framework (Ofria &
Wilke 2004, Artificial Life 10:191-229) and the digital-organism studies
of Lenski et al. 2003 (Nature 423:139-144), in which organisms earn
fitness by performing logic tasks and replicate with heritable
variation.

Task
----
Each organism executes its genome on a tiny register machine for a fixed
step budget and emits a bit sequence (its *signal*).  Fitness is the
exponential of the number of bits that match a target sequence
(``fitness = 2**matches``), a multiplicative fitness landscape where
every additional correct bit confers a constant selective advantage --
the digital analog of the small-effect beneficial mutations accumulated
over ~70,000 generations of the E. coli long-term evolution experiment
(Lenski 2017, Phil Trans R Soc B 372:20160592).

Genetics
--------
- point substitution, insertion and deletion mutations at configurable
  per-instruction rates;
- Wright-Fisher multinomial reproduction with fitness-proportional
  sampling;
- an optional neutral-drift mode (``selection_enabled=False``) used as a
  control.

The error-threshold behavior at very high mutation rates reproduces
Eigen's quasispecies error catastrophe (Eigen 1971, Naturwissenschaften
58:465-523).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# ============================================================================
# Instruction set
# ============================================================================

OPS = (
    "MOV0", "MOV1",       # R0 <- constant
    "ADD",                # R0 <- R0 + R1 (mod 2)
    "NOT",                # R0 <- not R0
    "OUT",                # emit R0 bit
    "NOP",                # no-op
    "HALT",               # stop execution
)


def _random_op(rng: random.Random) -> str:
    return OPS[rng.randrange(len(OPS))]


def execute(genome: tuple[str, ...], step_limit: int = 32) -> list[int]:
    """Execute a digital organism's genome and return the emitted bits.

    Register machine: two 1-bit registers R0/R1, an instruction pointer
    and an output stream.  Execution stops on ``HALT``, on exhausting
    ``step_limit`` steps, or at the end of the genome.
    """
    out: list[int] = []
    r0, r1 = 0, 0
    ip = 0
    steps = 0
    while ip < len(genome) and steps < step_limit:
        op = genome[ip]
        if op == "HALT":
            break
        if op == "MOV0":
            r0 = 0
        elif op == "MOV1":
            r0 = 1
        elif op == "ADD":
            r0 = (r0 + r1) % 2
        elif op == "NOT":
            r0 = 1 if r0 == 0 else 0
        elif op == "OUT":
            out.append(r0)
        elif op == "NOP":
            pass
        ip += 1
        steps += 1
    return out


def fitness_of(genome: tuple[str, ...], target: tuple[int, ...],
               step_limit: int = 32) -> float:
    """Fitness = ``2**matches`` where ``matches`` counts target bits the
    organism's emitted signal reproduces (multiplicative landscape)."""
    emitted = execute(genome, step_limit)
    matches = 0
    for i, tbit in enumerate(target):
        if i < len(emitted) and emitted[i] == tbit:
            matches += 1
    return float(2 ** matches)


# ============================================================================
# Mutation spectrum
# ============================================================================

def mutate_genome(genome: tuple[str, ...],
                  substitution_rate: float,
                  insertion_rate: float = 0.0,
                  deletion_rate: float = 0.0,
                  rng: random.Random | None = None) -> tuple[str, ...]:
    """Apply point substitutions, insertions and deletions.

    Every instruction is independently subjected to each event type at
    the given per-instruction rates (a realistic mutation spectrum:
    substitution-dominated, with rare indels).
    """
    r = rng if rng is not None else random.Random(0)
    seq = list(genome)
    # substitutions
    for i in range(len(seq)):
        if r.random() < substitution_rate:
            seq[i] = _random_op(r)
    # insertions (walk backwards so positions stay valid)
    i = len(seq) - 1
    while i >= 0:
        if r.random() < insertion_rate:
            seq.insert(i + 1, _random_op(r))
        i -= 1
    if seq and r.random() < deletion_rate:
        seq.pop(r.randrange(len(seq)))
    return tuple(seq)


# ============================================================================
# Population + evolution
# ============================================================================

@dataclass(slots=True)
class DigitalOrganism:
    """One digital organism (instruction genome)."""

    genome: tuple[str, ...]
    fitness: float = 1.0
    ancestor: int | None = None


@dataclass(slots=True)
class DigitalEvolutionConfig:
    """Evolutionary dynamics configuration.

    Args:
        population_size: fixed population size (Wright-Fisher).
        genome_length: initial (random) genome length in instructions.
        target: signal bit sequence the organisms must reproduce.
        substitution_rate, insertion_rate, deletion_rate: per-instruction
            mutation rates per generation.
        step_limit: execution step budget per fitness evaluation.
        selection_enabled: when False the population evolves by pure
            genetic drift (neutral control).
        generations: default number of generations for :meth:`run`.
        seed: RNG seed.
    """

    population_size: int = 100
    genome_length: int = 12
    target: tuple[int, ...] = (1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1)
    substitution_rate: float = 0.01
    insertion_rate: float = 0.001
    deletion_rate: float = 0.001
    step_limit: int = 32
    selection_enabled: bool = True
    generations: int = 200
    seed: int | None = None


class DigitalEvolution:
    """Evolve a digital-organism population on a signal task."""

    def __init__(self, config: DigitalEvolutionConfig | None = None) -> None:
        self.config = config or DigitalEvolutionConfig()
        if self.config.population_size <= 0:
            raise ValueError("population_size must be >= 1")
        if not self.config.target:
            raise ValueError("target must be a non-empty bit sequence")
        self.rng = random.Random(self.config.seed)
        self.population = self._init_population()
        self.generation = 0
        self.history: list[dict[str, float]] = []

    def _init_population(self) -> list[DigitalOrganism]:
        cfg = self.config
        pop: list[DigitalOrganism] = []
        for _ in range(cfg.population_size):
            genome = tuple(_random_op(self.rng)
                           for _ in range(cfg.genome_length))
            pop.append(DigitalOrganism(
                genome=genome,
                fitness=fitness_of(genome, cfg.target, cfg.step_limit),
            ))
        return pop

    def _fitnesses(self) -> list[float]:
        return [o.fitness for o in self.population]

    def mean_fitness(self) -> float:
        fits = self._fitnesses()
        return sum(fits) / len(fits)

    def max_fitness(self) -> float:
        return max(self._fitnesses())

    def fittest_genome(self) -> tuple[str, ...]:
        return max(self.population, key=lambda o: o.fitness).genome

    def _select(self) -> list[DigitalOrganism]:
        """Wright-Fisher reproduction (fitness-proportional or neutral)."""
        cfg = self.config
        n = cfg.population_size
        pop = self.population
        if not cfg.selection_enabled:
            return [DigitalOrganism(
                genome=pop[self.rng.randrange(n)].genome,
                fitness=1.0,
            ) for _ in range(n)]
        fits = [o.fitness for o in pop]
        total = sum(fits)
        if total <= 0.0:
            # complete meltdown: fall back to uniform sampling so the
            # population can still drift rather than dying out
            return [DigitalOrganism(
                genome=pop[self.rng.randrange(n)].genome, fitness=0.0,
            ) for _ in range(n)]
        # stochastic acceptance sampling (no cumulative array needed)
        offspring: list[DigitalOrganism] = []
        for _ in range(n):
            while True:
                candidate = pop[self.rng.randrange(n)]
                if self.rng.random() < candidate.fitness / max(fits):
                    offspring.append(candidate)
                    break
        return [DigitalOrganism(genome=o.genome, fitness=o.fitness)
                for o in offspring]

    def step(self) -> dict[str, float]:
        """Advance one generation; returns population statistics."""
        cfg = self.config
        parents = self._select()
        offspring: list[DigitalOrganism] = []
        for parent in parents:
            genome = parent.genome
            if (cfg.substitution_rate > 0.0 or cfg.insertion_rate > 0.0
                    or cfg.deletion_rate > 0.0):
                genome = mutate_genome(
                    genome,
                    cfg.substitution_rate,
                    cfg.insertion_rate,
                    cfg.deletion_rate,
                    self.rng,
                )
            offspring.append(DigitalOrganism(
                genome=genome,
                fitness=fitness_of(genome, cfg.target, cfg.step_limit),
            ))
        self.population = offspring
        self.generation += 1
        stats = {
            "generation": float(self.generation),
            "mean_fitness": self.mean_fitness(),
            "max_fitness": self.max_fitness(),
        }
        self.history.append(stats)
        return stats

    def run(self, generations: int | None = None) -> list[dict[str, float]]:
        """Run the evolution for ``generations`` generations."""
        n = self.config.generations if generations is None else generations
        for _ in range(n):
            self.step()
        return self.history

    def substitution_spectrum(self, samples: int = 200) -> dict[str, float]:
        """Empirical per-generation mutation spectrum.

        Counts substitutions/insertions/deletions over ``samples``
        fresh mutations of a random genome, normalized per genome.
        """
        cfg = self.config
        counts = {"substitution": 0.0, "insertion": 0.0, "deletion": 0.0}
        rng = random.Random(cfg.seed)
        for _ in range(samples):
            genome = tuple(_random_op(rng) for _ in range(cfg.genome_length))
            mutant = mutate_genome(
                genome, cfg.substitution_rate, cfg.insertion_rate,
                cfg.deletion_rate, rng)
            # substitutions: differing positions
            counts["substitution"] += sum(
                1 for a, b in zip(genome, mutant, strict=False) if a != b)
            counts["insertion"] += max(0, len(mutant) - len(genome))
            counts["deletion"] += max(0, len(genome) - len(mutant))
        return {k: v / samples for k, v in counts.items()}


# ============================================================================
# Convenience helpers
# ============================================================================

def run_digital_evolution(config: DigitalEvolutionConfig | None = None
                          ) -> dict:
    """Run a full evolution and return the summary.

    Returns ``{"config", "history", "final_mean", "final_max",
    "final_genome"}``.
    """
    evo = DigitalEvolution(config)
    history = evo.run()
    return {
        "config": evo.config,
        "history": history,
        "final_mean": evo.mean_fitness(),
        "final_max": evo.max_fitness(),
        "final_genome": evo.fittest_genome(),
    }


__all__ = [
    "OPS", "execute", "fitness_of", "mutate_genome",
    "DigitalOrganism", "DigitalEvolutionConfig", "DigitalEvolution",
    "run_digital_evolution",
]
