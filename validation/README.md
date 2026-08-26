# HelixLang Scientific Validation Suite

Reproducible benchmarks that validate biological correctness against
published reference models. See doc/34 §3 for design rationale.

## Running

```bash
# Run all benchmarks
cd validation && python run_all.py

# Run a single benchmark
python benchmarks/01_codon_translation/run.py
```

## Benchmarks

| # | Name | Layer | Validates | Reference |
|---|------|-------|-----------|-----------|
| 01 | Codon translation | language | 64→20 mapping | Standard genetic code |
| 02 | lac operon | runtime | GRN regulation | Jakob & Monod 1961 |
| 03 | E. coli FBA | metabolism | Growth rate | Orth et al. 2010 |
| 04 | iML1515 | metabolism | Genome-scale FBA | Monk et al. 2017 |
| 05 | iJN678 photoauto | metabolism | Photoautotrophy | Knoop 2010 |

## Adding a benchmark

1. Create `benchmarks/NN_name/` directory.
2. Write `benchmark.yaml` with id, reference, expected values.
3. Write `run.py` that executes the benchmark and prints JSON.
4. Run once to generate `results/NN_name.json`.
5. Run again to verify determinism.

## Result format

Each benchmark produces a JSON file in `results/`:

```json
{
  "id": "01_codon_translation",
  "status": "PASS",
  "helix_version": "2026.8.4",
  "expected": {...},
  "actual": {...},
  "error": null,
  "runtime_seconds": 0.01
}
```
