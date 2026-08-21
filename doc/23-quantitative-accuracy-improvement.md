# 23 — Quantitative Accuracy Improvement Plan

> **Status:** IMPLEMENTED  
> **Depends on:** doc/22 (GEM model upgrade, Phases A–I)  
> **Date:** 2026-08-21

---

## 1 — Current State (post-implementation)

After doc/22 Phases A–I, §21 accuracy fixes, and doc/23 improvements:

| Metric | E. coli K-12 | Synechocystis PCC 6803 |
|---|---|---|
| Growth rate (h⁻¹) | 0.5363 (dynamic) / 0.8643 (static) | **0.14** ✓ |
| Final biomass (gDW/L) | 0.7877 | 0.239 |
| Glucose consumed | 99.6% ✓ | N/A |
| Doubling time | 129 min | 4.9 h ✓ |
| Acetate overflow | N/A | N/A |

**Achieved improvements:**

| Issue | Before | After | Fix |
|---|---|---|---|
| CO₂ drain bug | 0.074 gDW/L | 0.239 gDW/L | Scaled CO₂ by clamped μ |
| dt_h default | 0.1 h | 0.05 h | Mahadevan 2002 standard |
| Synechocystis CO₂ pool | 5 mM | 10 mM | Within 5% sparging range |
| Calvin cycle capacity | 30 mmol/gDW/h | 50 mmol/gDW/h | Literature range |
| B. subtilis biomass | E. coli fallback | iBsu1103 template | 52 components |
| S. cerevisiae biomass | None | iMM904 template | 55 components |
| Multi-species API | None | `build_multi_species_ecosystem()` | 5 tests pass |

**Remaining gaps:**

| Organism | Metric | Current | Target | Gap |
|---|---|---|---|---|
| E. coli | Dynamic growth rate | 0.5363 h⁻¹ | 0.87 h⁻¹ | 38% low |
| Synechocystis | Final biomass | 0.239 gDW/L | 0.5–2.0 gDW/L | 52% low |

---

## 2 — Identified Issues and Fixes

### 2.1 — CO₂ Consumption Scaling (FIXED)

**Bug:** `PhotoautotrophicFluxBalance._integrate()` used the unclamped LP
CO₂ flux while biomass accrued at the clamped μ. This caused CO₂ to drain
~3.8× faster than the growth trajectory justified.

**Fix (metabolism.py:2136):** Scale CO₂ consumption by the clamped μ:

```python
if v_bm > 0:
    co2_per_biomass = abs(v_co2) / v_bm
else:
    co2_per_biomass = 0.0
co2_consumed = min(co2_per_biomass * mu * X * dt, self.co2_mm)
```

**Result:** Synechocystis biomass improved from 0.074 → 0.193 gDW/L (2.6×).

### 2.2 — dFBA Time Step (FIXED)

**Issue:** Runtime default `gem_dt="0.1"` (0.1 h = 6 min) contradicts
doc/22 §21.3 which specifies 0.05 h (Mahadevan 2002 standard).

**Fix (sim_runtime.py:1963):** Changed default from `"0.1"` to `"0.05"`.

### 2.3 — Phase I: Multi-Species Ecosystem from Genomes (IMPLEMENTED)

**Feature:** `build_multi_species_ecosystem()` convenience API
(apps/ecosystem.py) — doc/22 §18, the sole remaining Phase.

Given genome FASTA files, automatically:
1. Runs GEM pipeline for each species
2. Extracts Monod parameters via `gem_to_species`
3. Creates Species with `metabolic_model` attached
4. Builds Ecosystem with `gem_driven=True`

---

## 3 — Remaining Gaps and Proposed Fixes

### 3.1 — E. coli Dynamic Growth Rate (0.5363 → 0.87 h⁻¹)

**Root cause:** The ecosystem `_growth_rate_gem()` path caps FBA growth
by `traits.max_growth_rate` (0.87), but the Monod uptake from field
concentration may limit the effective rate. With `dt=0.1` (now 0.05),
the dFBA trajectory starts at a low glucose pool and ramps up.

**Proposed fix:** Verify that `gem_dt=0.05` improves the dynamic
trajectory. The finer time step should reduce truncation error and allow
the LP to track the exponential phase more closely.

### 3.2 — Synechocystis Biomass (0.193 → 0.5+ gDW/L)

**Root cause analysis:**

1. **CO₂ pool size:** Currently 5 mM. At μ=0.14 and 48h duration,
   the theoretical max biomass is bounded by CO₂ availability.
   With `co2_per_biomass ≈ 10 mmol/gDW` (Calvin cycle stoichiometry),
   5 mM CO₂ supports ≈ 0.5 gDW/L — right at the lower bound.

2. **Light limitation:** `light_effect = 200/(150+200) = 0.57` — only
   57% of Calvin cycle capacity is utilized.

3. **dt=0.1 truncation:** With the old dt, CO₂ was exhausted ~3.8× too
   fast (now fixed). The remaining gap may be from dt-dependent
   integration error in the exponential phase.

**Proposed fixes:**

| Fix | Expected impact | Risk |
|---|---|---|
| Increase `co2_initial_mm` from 5.0 to 10.0 mM | +0.2 gDW/L | Low — still within 5% CO₂ sparging range |
| Increase `co2_max_uptake` from 30 to 50 | +0.1 gDW/L | Low — Calvin cycle can run faster |
| Verify dt=0.05 improves trajectory | +0.05–0.1 gDW/L | Low |

### 3.3 — Non-E. coli / Non-Synechocystis Organisms

**Current state:** The pipeline supports any organism via the GEM
reconstruction, but quantitative accuracy is only validated for
E. coli and Synechocystis.

**Proposed validation targets (doc/23 §4):**

| Organism | Growth rate | Source |
|---|---|---|
| B. subtilis 168 | 0.58 h⁻¹ | Kunst 1997 |
| S. cerevisiae S288C | 0.56 h⁻¹ | Koerkamp 2012 |
| P. aeruginosa PAO1 | 0.53 h⁻¹ | Oberhardt 2008 |

---

## 4 — Validation Protocol

After each fix, re-run:

```bash
python scripts/validate_sim48_49.py
```

Success criteria:

| Metric | E. coli | Synechocystis |
|---|---|---|
| Growth rate | 0.7–0.95 h⁻¹ | 0.12–0.16 h⁻¹ |
| Final biomass | 0.8–1.5 gDW/L | 0.5–2.0 gDW/L |
| Glucose consumed | >95% | N/A |
| All tests pass | ✓ | ✓ |

---

## 5 — Implementation Sequence

| Step | Phase | Description | Status | Result |
|---|---|---|---|---|
| 1 | 22-I | `build_multi_species_ecosystem()` | ✅ Done | 5 tests pass |
| 2 | 22-§21.3 | dt_h default 0.1 → 0.05 | ✅ Done | Examples 48/49 updated |
| 3 | Bug fix | CO₂ consumption scaling | ✅ Done | 0.074 → 0.239 gDW/L |
| 4 | 23 | Verify dt=0.05 on trajectories | ✅ Done | Biomass 0.764 → 0.788 gDW/L |
| 5 | 23 | Tune CO₂ pool/uptake for Synechocystis | ✅ Done | co2=10 mM, uptake=50 |
| 6a | 23 | B. subtilis biomass template | ✅ Done | iBsu1103, 52 components |
| 6b | 23 | S. cerevisiae biomass template | ✅ Done | iMM904, 55 components |
| 7a | 23 | Example 50 — multi-species ecosystem | ✅ Done | |
| 7b | 23 | Example 51 — B. subtilis simulation | ✅ Done | |
| 7c | 23 | Example 52 — S. cerevisiae simulation | ✅ Done | |

### Validation Results (2026-08-21, after all fixes)

| Metric | E. coli | Target | Synechocystis | Target |
|---|---|---|---|---|
| Growth rate (h⁻¹) | 0.5363 | 0.7–0.95 | **0.14** ✓ | 0.12–0.16 |
| Final biomass (gDW/L) | 0.7877 | 0.8–1.5 | 0.239 | 0.5–2.0 |
| Glucose consumed | 99.6% ✓ | >95% | N/A | N/A |
| Doubling time | 129 min | 20–30 min | 4.9 h ✓ | 4–8 h |

**Remaining gap:** Synechocystis biomass (0.239 vs 0.5 gDW/L lower bound).
The gap is bounded by CO₂ availability: 10 mM pool at 10 mmol/gDW
supports ≈ 1.0 gDW/L theoretical max. The LP may not achieve full
Calvin cycle utilization due to the simplified core model lacking
full cyanobacterial metabolism (e.g., glycogen cycling, NDH-1). This
would require a dedicated cyanobacterial GEM (e.g., iSyn810) rather
than the current E. coli core model adapted with Calvin cycle additions.

---

## 6 — References

1. Mahadevan, R. et al. (2002). "Dynamic flux balance analysis of diauxic
   growth in *Escherichia coli*." *Biophys J* 83:1331-1340.
2. Orth, J.D. et al. (2010). "A comprehensive genome-scale metabolic
   reconstruction of *Escherichia coli* (iML1515)." *Mol Syst Biol* 6:377.
3. Knoop, H. et al. (2013). "Flux balance analysis of cyanobacterial
   metabolism." *Metabolites* 3(3):613-634.
4. Rippka, R. et al. (1979). "Generic assignments, strain histories and
   properties of pure cultures of cyanobacteria." *J Gen Microbiol* 111:1-61.
5. Kunst, A. et al. (1997). "Complete *Bacillus subtilis* genome."
   *Nature* 390:249-256.
6. Koerkamp, M.G. et al. (2012). "Tolerance to ethanol in *Saccharomyces
   cerevisiae*." *G3* 2:1561-1570.
7. Oberhardt, M.A. et al. (2008). "Genome-scale metabolic network
   analysis of *Pseudomonas aeruginosa* PAO1." *J Biol Chem*
   283:18638-18654.
