"""Performance benchmarks: protect hot spots against performance regression.

Uses pytest-benchmark to record baseline execution times of key hot spots,
**without hard timing assertions** (machines vary widely); relies only on
pytest-benchmark's built-in regression comparison
(``--benchmark-compare`` / ``--benchmark-save``).

Covered hot spots:
1. FBA solve (ECOLI_CORE_MODEL, GLC=10, maximize biomass)
2. CRISPR PAMIndex.search (1200bp genome, SpCas9, max_mismatches=3)
3. evolution.mutate_batch (100 individuals x 1000bp)
4. protein_structure.predict_secondary_gor (500aa sequence)

CI does not install pytest-benchmark by default, so the whole module is
skipped by ``importorskip``; optionally install it locally and run::

    pip install pytest-benchmark
    pytest tests/test_benchmark.py

References:
- Orth 2010 Mol Syst Biol 6:390 (E. coli core model)
- Jinek 2012 Science 337:816-821 (SpCas9 NGG PAM)
- Drake 1991 Genetics 148:1667-1686 (E. coli mutation rate)
- Garnier, Osguthorpe & Robson 1978 J Mol Biol 120:97-120 (GOR method)
"""
from __future__ import annotations

import random

import pytest

# CI does not install pytest-benchmark -> the whole module is skipped during collection
pytest.importorskip("pytest_benchmark")

from helixlang.plugins.runtime.crispr import GuideRNA, PAMIndex, find_pam_sites
from helixlang.plugins.runtime.evolution import Individual, mutate_batch
from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis
from helixlang.plugins.runtime.protein_structure import predict_secondary_gor

# ============================================================================
# FBA solve hotspot
# ============================================================================

def test_benchmark_fba(benchmark):
    """FBA solve benchmark: ECOLI_CORE_MODEL + GLC=10, maximize biomass flux.

    ECOLI_CORE_MODEL is built once from JSON at module load; every solve
    re-runs the two-phase simplex (numpy vectorized path). This is the core
    hotspot of the metabolism module.
    """
    fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    fba.set_uptake("GLC", 10.0)

    def run():
        return fba.solve(objective="biomass")

    result = benchmark.pedantic(run, iterations=10, rounds=3)
    # Lightweight sanity check (no timing assertion)
    assert "BIOMASS" in result


# ============================================================================
# CRISPR PAMIndex.search hotspot
# ============================================================================

def test_benchmark_crispr_offtarget(benchmark):
    """PAMIndex.search benchmark: 1200bp genome, SpCas9, max_mismatches=3.

    Index construction (scanning PAM sites + multi-bucket k-mer hashing) is a
    one-time cost and is kept outside the benchmark timer; only search itself
    is measured (multi-bucket seed-and-extend lookup + candidate exact
    alignment + Hsu 2013 scoring).
    """
    rng = random.Random(42)
    genome = "".join(rng.choice("ACGT") for _ in range(1200))
    index = PAMIndex(genome, cas_variant="SpCas9")

    # Build the guide from real PAM sites in the genome to ensure the index has bucket hits
    sites = find_pam_sites(genome, "SpCas9", both_strands=False)
    assert sites, "the test genome should contain NGG PAM sites"
    first = sites[0]
    guide = GuideRNA(
        spacer=first["spacer"],
        pam=first["pam"],
        pam_position="3prime",
        cas_variant="SpCas9",
        target_position=first["position"],
        strand=first["strand"],
    )

    def run():
        return index.search(guide, max_mismatches=3)

    result = benchmark.pedantic(run, iterations=10, rounds=3)
    assert isinstance(result, list)


# ============================================================================
# evolution.mutate_batch hotspot
# ============================================================================

def test_benchmark_mutate_batch(benchmark):
    """mutate_batch benchmark: 100 individuals x 1000bp.

    Signature (confirmed by reading evolution.py)::
        mutate_batch(individuals, mutation_rate, indel_rate, ratio, rng)
            -> list[tuple[str, list[str]]]

    Uses real Individual objects. mutation_rate is raised to 0.01 to fully
    exercise the substitution/indel paths (at the real E. coli rate of
    2.2e-10, mutations almost never trigger and the hotspot branches would
    not be covered). mutate_batch only reads Individual.dna and does not
    modify the input objects, so the same batch of individuals can be reused
    across benchmark rounds.
    """
    rng = random.Random(42)
    dna = "ACGT" * 250  # 1000bp
    individuals = [
        Individual(dna=dna, fitness=1.0, generation=0)
        for _ in range(100)
    ]

    def run():
        return mutate_batch(individuals, 0.01, 0.001, 2.0, rng)

    result = benchmark.pedantic(run, iterations=10, rounds=3)
    assert len(result) == 100


# ============================================================================
# predict_secondary_gor hotspot
# ============================================================================

def test_benchmark_gor(benchmark):
    """predict_secondary_gor benchmark: 500aa sequence (GOR III 17-residue window).

    GOR III accumulates singlet information (log-odds) for each residue over a
    17-residue window across the H/E/T three states, then takes the argmax.
    500 residues x 3 states x 17 windows ~= 25.5k lookups, making this the
    core hotspot of the protein_structure module.
    """
    rng = random.Random(42)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    seq = "".join(rng.choice(aas) for _ in range(500))

    def run():
        return predict_secondary_gor(seq)

    result = benchmark.pedantic(run, iterations=10, rounds=3)
    ss_string, segments = result
    assert len(ss_string) == 500
    assert len(segments) >= 1
