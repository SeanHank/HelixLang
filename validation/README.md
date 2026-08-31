# HelixLang Scientific Validation Suite

Reproducible benchmarks that validate biological correctness against
published reference models. See doc/34 §3 for design rationale.

## Running

```bash
# Run all benchmarks
cd validation && python run_all.py

# Run a single benchmark
python benchmarks/01_codon_translation/run.py

# Verify golden outputs
python goldens/verify_goldens.py

# Regenerate goldens
python goldens/generate_goldens.py
```

## Evidence Chain Schema

Every benchmark must produce an evidence chain following the pattern:
**Reference → Expected → Actual → Error → Reproducibility**

```json
{
  "id": "03_ecoli_fba",
  "status": "PASS",
  "layer": "metabolism",
  "name": "E. coli core FBA growth rate",
  "reference": {
    "source": "BiGG e_coli_core via COBRApy",
    "doi": "10.1371/journal.pcbi.1000822",
    "authors": "Orth et al.",
    "year": 2010,
    "journal": "PLoS Comput Biol 6:e1000822"
  },
  "expected": {
    "metric": "growth_rate",
    "value": 0.877,
    "tolerance": 0.05,
    "unit": "h^-1"
  },
  "actual": {
    "value": 0.872
  },
  "error": {
    "abs_error": 0.005,
    "rel_error": 0.0057,
    "passed": true
  },
  "reproducibility": {
    "deterministic": true,
    "environment": "Python 3.11.5",
    "golden_hash": "verified"
  }
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `id` | Benchmark identifier (matches directory name) |
| `status` | `PASS`, `FAIL`, or `SKIP` (external artefact unavailable — never a failure, doc/41 §2) |
| `layer` | Biological domain (32 values in use, reconciled in `benchmark.yaml` below; `run_all.py` merges it as the single source of truth) |
| `level` | **Canonical validation level `L0`–`L5` (required, doc/41 §3)** — see taxonomy below |
| `reference` | Source of truth (paper, database, analytical solution) |
| `expected` | What the reference says the result should be |
| `actual` | What HelixLang produced |
| `error` | Quantified difference between expected and actual |
| `reproducibility` | Whether the result is deterministic and reproducible |

## Validation Levels (doc/41 §3)

Every benchmark declares exactly **one** level in `benchmark.yaml`. The level
**classifies the reference, not the outcome**.

| Level | Name | Definition | Min required evidence for PASS |
|---|---|---|---|
| `L0` | Functional test | API/import/smoke; no external truth | programmatic check + `status` |
| `L1` | Analytical validation | closes a closed-form solution / conservation law (mass balance, analytic ODE, FBA optimality) | analytic reference + error metric |
| `L2` | Reference-implementation validation | same input through a trusted second implementation (COBRApy, `_accel`) | reference-impl id + error metric + `golden_hash` |
| `L3` | Literature validation | parameters/ranges anchored to published literature | citation (`reference.doi`) + expected range/tolerance |
| `L4` | Experimental validation | quantitative comparison vs published *measured* data | `experimental_comparison {min,max,unit}` + citation |
| `L5` | Clinical validation | outcomes matched to patient-level trials / case series | external clinical dataset + statistical report + `DISCLAIMER` (none today; disclaimed) |

Current distribution: `L0` ×57 · `L1` ×2 · `L2` ×8 · `L3` ×2 · `L4` ×6 · `L5` ×0.

Level gates are enforced as **warnings** in `schema.py::EvidenceChain.level_gate_violations()`
and surfaced in the report (doc/41 §3.2 Rule 5), so legacy shapes still normalize.

### Legacy-scheme reconciliation

| Legacy scheme | Where | Maps to |
|---|---|---|
| doc/34 §1.3 `A/D` letters | `doc/34:56-63` | ⊂ `L1`–`L4` |
| doc/32 §6.2 `L1`–`L5` (pipeline-ready) | `doc/32:267-273` | doc/32 L1–L3 ⊂ `L3`, L4 ⊂ `L5` |
| `grn_inference.EvidenceLevel` | `grn_inference.py:10-17` | data (edge quality); left as-is, not a benchmark level |
| `DDIRule.evidence_level` | doc/28 | data; left as-is |
| pivotal / adequate / exploratory | doc/31 | exploratory = L0/L3, adequate = L3/L4, pivotal = L4/L5 |
| `bio_validity.ScopeLevel` | `bio_validity.py:31-35` | runtime scope guard, orthogonal to levels |

No new parallel scheme is introduced.

### Helper Function

Use `validation/schema.py::make_evidence_chain()` for automatic error computation
(it now also carries the canonical `level`):

```python
from validation.schema import make_evidence_chain

chain = make_evidence_chain(
    benchmark_id="03_ecoli_fba",
    reference_source="BiGG e_coli_core via COBRApy",
    expected_metric="growth_rate",
    expected_value=0.877,
    actual_value=0.872,
    tolerance=0.05,
    unit="h^-1",
    layer="metabolism",
    level="L4",
    doi="10.1371/journal.pcbi.1000822",
    authors="Orth et al.",
    year=2010,
)
result = chain.to_dict()
```

## Benchmarks

| # | Name | Layer | Level | Validates | Reference |
|---|------|-------|-------|-----------|-----------|
| 01 | Codon translation | language | L3 | 64→20 mapping | Standard genetic code |
| 02 | lac operon | runtime | L3 | GRN regulation | Jakob & Monod 1961 |
| 03 | E. coli FBA | metabolism | L4 | Growth rate | Orth et al. 2010 |
| 04 | iML1515 | metabolism | L2 | Genome-scale FBA | Monk et al. 2017 |
| 05 | iJN678 photoauto | metabolism | L2 | Photoautotrophy | Knoop 2010 |
| 06 | dFBA diauxic | metabolism | L4 | Biphasic growth | Enjalbert 2015 |
| 07 | Repressilator | kinetics | L4 | Oscillation | Elowitz 2000 |
| 08 | Population dynamics | population | L4 | Doubling time | Analytical |
| 09 | Reaction-diffusion | spatial | L2 | Pattern formation | Reference + robustness |
| 10 | Whole-cell | virtual_cell | L4 | Division time | Wanner 1996 |
| 11-45 | Various | Various | Various | Functional + performance | Multiple |

## Adding a benchmark

1. Create `benchmarks/NN_name/` directory.
2. Write `benchmark.yaml` with id, layer, **level (L0–L5)**, reference, expected values.
3. Write `run.py` that:
   - Defines `run() -> dict` function
   - Produces evidence chain with `reference`, `expected`, `actual`, `error`, `reproducibility`
   - Prints JSON to stdout
   - Exits 0 on PASS/SKIP, 1 on FAIL
   - Returns `SKIP` (with a `reason`) when an external artefact cannot be obtained
     (doc/41 §2 — never degrade to a partial PASS)
4. Run once to generate `results/NN_name.json`.
5. Run again to verify determinism.
6. Run `python goldens/generate_goldens.py` to create golden output.
