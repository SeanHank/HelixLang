# 21. GEM ↔ Ecosystem Bridge: Gene-to-Ecosystem End-to-End Pipeline

> **Status: ✅ IMPLEMENTED** (2026-08-20). All design items landed;
> tests pass; mypy clean.

## 1. Problem

doc/20 (GEM reconstruction) and doc/19 (ecosystem simulation) are both
complete but **completely isolated**. The GEM pipeline produces a
`MetabolicModel` + FBA fluxes, but the ecosystem's `Species` class and
`Patch._growth_rate()` use hardcoded Monod `(vmax, ks)` and simple
`yield_c` parameters. No code path connects the two.

**Current state:**

| Component | Status |
|-----------|--------|
| GEM pipeline (doc/20) | ✅ 6 stages, CLI `--gem` |
| Ecosystem (doc/19) | ✅ Monod uptake, Lotka-Volterra, CENTURY, N-cycle |
| Language `#gem` / `#species` | ✅ Parsing works |
| `#sim kind=ecosystem` | ✅ Multi-species, multi-patch |
| **Bridge: GEM → Ecosystem** | **✅ IMPLEMENTED** (doc/21 §3, `gem_to_species()`, `Species.metabolic_model`, `_growth_rate_gem()`) |

**Consequence:** A user can run `helixlang --gem` on a real genome and
get FBA results, and separately run `#sim kind=ecosystem` with manually
defined species, but there is no way to say *"simulate this ecosystem
using the metabolic models reconstructed from these three genomes"*.

## 2. Goal

A single `.helix` source can specify:

```
#species name=ecoli genome=ecoli.fasta
#species name=bsub genome=bsubtilis.fasta
#patch name=chemostat kind=chemostat initial.ecoli=100 initial.bsub=50
  substrate.glucose.initial=10
#sim kind=ecosystem ticks=4320
```

The system automatically:
1. Runs the GEM pipeline on each species' genome
2. Extracts metabolic parameters (vmax, yield, secretion, nutritional needs) from the FBA solution
3. Feeds these into the ecosystem tick loop as per-species metabolic traits
4. Optionally runs per-tick community FBA to reconcile shared resource allocation

## 3. Design: Bridge Layer

### 3.1 `gem_to_species()` — Extract metabolic parameters from GEM

```python
def gem_to_species(
    pipeline_result: GemPipelineResult,
    organism: str,
    medium: str = "glucose_minimal",
) -> dict[str, float]:
    """Extract ecosystem-compatible parameters from a GEM pipeline result.

    Returns a dict with keys:
      vmax, ks, yield_c, secretion_rate, cn_ratio, maintenance
    suitable for populating a Species via Species.from_gem_params().
    """
```

**Extraction logic:**

| Ecosystem param | GEM source |
|----------------|------------|
| `vmax` (mmol/gDW/h) | Max exchange flux from FBA (EX_glc__D_e) |
| `ks` (mM) | Km estimate from `km_estimator` |
| `yield_c` | Biomass flux / glucose uptake flux × C-ratio |
| `secretion` | Non-zero export fluxes (EX_ac_e, EX_etoh_e, etc.) |
| `cn_ratio` | Biomass reaction C:N stoichiometry |
| `maintenance` | ATP maintenance flux (ATPM) from FBA |

### 3.2 `Species.metabolic_model` — Optional GEM attachment

Add an optional `metabolic_model: MetabolicModel | None` field to the
`Species` dataclass. When present, the ecosystem tick loop uses
GEM-derived parameters instead of the hardcoded Monod defaults.

```python
@dataclass(slots=True)
class Species:
    ...
    metabolic_model: MetabolicModel | None = None
    gem_fluxes: dict[str, float] = field(default_factory=dict)
    gem_kcat: dict[str, float] = field(default_factory=dict)
    gem_km: dict[str, float] = field(default_factory=dict)
```

### 3.3 `Patch._growth_rate()` — Optional FBA-backed growth

When a species has a `metabolic_model`, the growth rate calculation
optionally uses a lightweight FBA solve (per-site, per-tick) instead of
the simple Monod formula. This is controlled by `EcosystemConfig.gem_driven_growth`:

- **Default `False`**: Monod kinetics (fast, current behavior)
- **`True`**: Per-species FBA each tick (accurate but slower)

The FBA-backed path:
1. Sets exchange bounds from local substrate concentrations
2. Solves for biomass flux
3. Returns the FBA growth rate instead of Monod-derived rate

### 3.4 `#gem` + `#sim kind=ecosystem` integration

When both `#gem` and `#sim kind=ecosystem` blocks appear in the same
source, the pipeline:

1. Runs `run_gem_pipeline()` for each `#species` with a `genome=`
2. Calls `gem_to_species()` on each result
3. Attaches the `MetabolicModel` to each `Species`
4. Builds the `Ecosystem` and runs the tick loop

This is wired in `sim_runtime._run_ecosystem()`.

### 3.5 Language syntax

```
#gem organism=e_coli_k12 genome=ecoli.fasta
#species name=ecoli genome=ecoli.fasta
#species name=bsub genome=bsubtilis.fasta substrate=glucose vmax=0.02 ks=0.1
#patch name=chemostat kind=chemostat
  initial.ecoli=100 initial.bsub=50
  substrate.glucose.initial=10
#sim kind=ecosystem ticks=4320 gem_driven=true
```

When `gem_driven=true` in `#sim`, the ecosystem backend:
1. For each `#species` with a `genome=`, runs the GEM pipeline
2. Extracts metabolic parameters from FBA
3. Uses them in the tick loop

When `gem_driven=false` (default), the ecosystem uses the manually
specified Monod parameters (backward compatible).

## 4. Files Modified (✅ IMPLEMENTED)

| File | Change |
|------|--------|
| `apps/ecosystem.py` | ✅ Added `gem_to_species()`; added `metabolic_model`/`gem_fluxes`/`gem_kcat`/`gem_km` to `Species`; added `gem_driven` to `EcosystemConfig`; added `Patch._growth_rate_gem()` for FBA-backed growth; added `Patch.__init__(gem_driven=)` parameter |
| `sim_runtime.py` | ✅ Added `_attach_gem_to_ecosystem_species()` (runs GEM pipeline per species, calls `gem_to_species()`, attaches model + params); wired into `_run_ecosystem()` when `gem_driven=true` |
| `parser.py` | No changes needed (parsing already works) |
| `apps/gem_pipeline.py` | No changes needed (pipeline already complete) |

## 5. Verification (✅ ALL PASSING)

1. ✅ mypy clean on `apps/ecosystem.py` and `sim_runtime.py`
2. ✅ `import helixlang.apps.ecosystem` and `import helixlang.sim_runtime` succeed
3. ✅ GEM integration tests pass (`test_gem_integration.py`, `test_metabolism_proxy.py`, `test_omics.py`)
4. ✅ Ecosystem/population tests pass
5. ✅ Full test suite: **all pass** (no regressions)
6. ✅ Backward compatibility: `#sim kind=ecosystem` without `gem_driven` unchanged
