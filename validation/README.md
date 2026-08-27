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
| `status` | `PASS` or `FAIL` |
| `layer` | Biological domain: `language`, `runtime`, `metabolism`, `kinetics`, `pharmacology`, `cell_biology`, `ecosystem`, `crispr`, `determinism` |
| `reference` | Source of truth (paper, database, analytical solution) |
| `expected` | What the reference says the result should be |
| `actual` | What HelixLang produced |
| `error` | Quantified difference between expected and actual |
| `reproducibility` | Whether the result is deterministic and reproducible |

### Helper Function

Use `validation/schema.py::make_evidence_chain()` for automatic error computation:

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
    doi="10.1371/journal.pcbi.1000822",
    authors="Orth et al.",
    year=2010,
)
result = chain.to_dict()
```

## Benchmarks

| # | Name | Layer | Validates | Reference |
|---|------|-------|-----------|-----------|
| 01 | Codon translation | language | 64→20 mapping | Standard genetic code |
| 02 | lac operon | runtime | GRN regulation | Jakob & Monod 1961 |
| 03 | E. coli FBA | metabolism | Growth rate | Orth et al. 2010 |
| 04 | iML1515 | metabolism | Genome-scale FBA | Monk et al. 2017 |
| 05 | iJN678 photoauto | metabolism | Photoautotrophy | Knoop 2010 |
| 06 | dFBA diauxic | metabolism | Biphasic growth | Enjalbert 2015 |
| 07 | Repressilator | kinetics | Oscillation | Elowitz 2000 |
| 08 | Population dynamics | population | Doubling time | Analytical |
| 09 | Reaction-diffusion | spatial | Pattern formation | Reference + robustness |
| 10 | Whole-cell | cell_biology | Division time | Wanner 1996 |
| 11-45 | Various | Various | Functional + performance | Multiple |

## Adding a benchmark

1. Create `benchmarks/NN_name/` directory.
2. Write `benchmark.yaml` with id, layer, reference, expected values.
3. Write `run.py` that:
   - Defines `run() -> dict` function
   - Produces evidence chain with `reference`, `expected`, `actual`, `error`, `reproducibility`
   - Prints JSON to stdout
   - Exits 0 on PASS, 1 on FAIL
4. Run once to generate `results/NN_name.json`.
5. Run again to verify determinism.
6. Run `python goldens/generate_goldens.py` to create golden output.
