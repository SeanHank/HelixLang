"""GEM reconstruction growth-rate benchmarks (doc/20 §18).

Measures predicted growth rates for E. coli K-12 and B. subtilis against
literature values using the full GEM pipeline (annotation → reconstruction
→ FBA).  Uses the curated E. coli core model and B. subtilis template.

Usage::

    python benchmarks/bench_gem_growth.py          # full run
    python benchmarks/bench_gem_growth.py --json    # JSON output
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Expected growth rates (h^-1) from literature
EXPECTED_GROWTH_RATES: dict[str, float] = {
    "e_coli_k12": 0.87,      # Orth et al. 2010, MMSB 6:390 (glucose minimal)
    "b_subtilis": 0.70,      # Nicolas et al. 2012, Nature 487:98 (glucose)
}


@dataclass
class BenchmarkResult:
    organism: str
    predicted_growth_rate: float
    expected_growth_rate: float
    error: float
    error_pct: float
    wall_time_s: float
    reactions: int
    genes_annotated: int
    passed: bool  # within 50% of expected


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult] = field(default_factory=list)
    total_wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "results": [
                {
                    "organism": r.organism,
                    "predicted_growth_rate": r.predicted_growth_rate,
                    "expected_growth_rate": r.expected_growth_rate,
                    "error": r.error,
                    "error_pct": r.error_pct,
                    "wall_time_s": r.wall_time_s,
                    "reactions": r.reactions,
                    "genes_annotated": r.genes_annotated,
                    "passed": r.passed,
                }
                for r in self.results
            ],
            "total_wall_time_s": self.total_wall_time_s,
        }


def _find_genome_fasta(organism: str) -> str:
    """Find the genome FASTA for an organism."""
    data_dir = Path(__file__).parent.parent / "data"
    candidates = [
        data_dir / f"{organism}_core_genome.fasta",
        data_dir / f"{organism}_genome.fasta",
        data_dir / "ecoli_core_genome.fasta",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def benchmark_organism(organism: str) -> BenchmarkResult:
    """Run the full GEM pipeline for one organism and return benchmark."""
    from helixlang.plugins.apps.gem_pipeline import run_gem_pipeline
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis

    expected = EXPECTED_GROWTH_RATES.get(organism, 0.0)
    genome = _find_genome_fasta(organism)
    if not genome:
        return BenchmarkResult(
            organism=organism,
            predicted_growth_rate=0.0,
            expected_growth_rate=expected,
            error=expected,
            error_pct=100.0,
            wall_time_s=0.0,
            reactions=0,
            genes_annotated=0,
            passed=False,
        )

    t0 = time.perf_counter()
    result = run_gem_pipeline(
        genome_fasta=genome,
        organism=organism,
        use_database_interactions=True,
        include_spontaneous=True,
        run_gapfill=True,
        target_organism="Escherichia coli" if "coli" in organism else "Bacillus subtilis",
    )
    wall_time = time.perf_counter() - t0

    growth_rate = 0.0
    n_reactions = 0
    if result.consensus is not None:
        try:
            from helixlang.plugins.gem.biomass import build_biomass_reaction
            from helixlang.plugins.gem.bridge import consensus_to_metabolic_model
            from helixlang.plugins.gem.gapfill import _parse_equation_to_stoich
            from helixlang.plugins.runtime.metabolism import Reaction

            model = consensus_to_metabolic_model(result.consensus)
            n_reactions = len(model.reactions)

            if result.gapfill:
                for rxn in result.gapfill.added_reactions:
                    stoich = _parse_equation_to_stoich(rxn.equation)
                    if stoich and rxn.reaction_id not in model.reactions:
                        model.add_reaction(Reaction(
                            id=rxn.reaction_id,
                            name=rxn.reaction_id,
                            stoichiometry=stoich,
                            lower_bound=-1000.0,
                            upper_bound=1000.0,
                            subsystem="exchange",
                        ))

            biomass = build_biomass_reaction(organism)
            biomass_stoich: dict[str, float] = {}
            for c in biomass.components:
                met = c.metabolite_id
                if met.endswith(("_c", "_e", "_p")):
                    met = met[:-2]
                if met in model.metabolites or not model.metabolites:
                    biomass_stoich[met] = (
                        biomass_stoich.get(met, 0.0) + c.coefficient
                    )
            _RECYCLED_COFACTORS = {"nad", "nadp", "coa"}
            biomass_stoich = {
                k: v for k, v in biomass_stoich.items()
                if k not in _RECYCLED_COFACTORS
            }
            if biomass_stoich:
                model.add_reaction(Reaction(
                    id="BIOMASS_reaction",
                    name="BIOMASS_reaction",
                    stoichiometry=biomass_stoich,
                    lower_bound=0.0,
                    upper_bound=1000.0,
                    subsystem="biomass",
                ))
                model.set_biomass("BIOMASS_reaction")

            fba = FluxBalanceAnalysis(model)
            fluxes = fba.solve(objective="BIOMASS_reaction")
            growth_rate = fluxes.get("BIOMASS_reaction", 0.0)
        except Exception as exc:
            print(f"  FBA failed: {exc}", file=sys.stderr)

    error = abs(growth_rate - expected)
    error_pct = (error / expected * 100) if expected > 0 else 100.0
    passed = error_pct < 50.0

    return BenchmarkResult(
        organism=organism,
        predicted_growth_rate=growth_rate,
        expected_growth_rate=expected,
        error=error,
        error_pct=error_pct,
        wall_time_s=wall_time,
        reactions=n_reactions,
        genes_annotated=result.annotated_genes,
        passed=passed,
    )


def run_benchmarks(organisms: list[str] | None = None) -> BenchmarkReport:
    """Run growth-rate benchmarks for the specified organisms."""
    if organisms is None:
        organisms = list(EXPECTED_GROWTH_RATES.keys())

    report = BenchmarkReport()
    t0 = time.perf_counter()
    for org in organisms:
        print(f"Benchmarking {org}...", file=sys.stderr)
        result = benchmark_organism(org)
        report.results.append(result)
    report.total_wall_time_s = time.perf_counter() - t0
    return report


def print_report(report: BenchmarkReport) -> None:
    """Print a human-readable benchmark report."""
    print("=" * 72)
    print("GEM Growth Rate Benchmarks")
    print("=" * 72)
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n  Organism:       {r.organism}")
        print(f"  Predicted:      {r.predicted_growth_rate:.4f} h^-1")
        print(f"  Expected:       {r.expected_growth_rate:.4f} h^-1")
        print(f"  Error:          {r.error:.4f} ({r.error_pct:.1f}%)")
        print(f"  Reactions:      {r.reactions}")
        print(f"  Genes:          {r.genes_annotated}")
        print(f"  Wall time:      {r.wall_time_s:.2f}s")
        print(f"  Status:         {status}")
    print(f"\n  Total time:     {report.total_wall_time_s:.2f}s")
    all_pass = all(r.passed for r in report.results)
    print(f"  Overall:        {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="GEM growth-rate benchmarks")
    parser.add_argument("--json", action="store_true",
                        help="output JSON instead of table")
    parser.add_argument("--organisms", nargs="*", default=None,
                        help="organisms to benchmark (default: all)")
    args = parser.parse_args()

    report = run_benchmarks(args.organisms)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
    return 0 if all(r.passed for r in report.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
