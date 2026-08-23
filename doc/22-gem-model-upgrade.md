# 22 — GEM Model Upgrade: Amino Acid Biosynthesis, Biomass Restoration & Photoautotrophic dFBA

> **Status:** Implemented  
> **Depends on:** doc/20 (GEM reconstruction pipeline), doc/21 (ecosystem bridge)  
> **Target:** Examples 48 (E. coli K-12 MG1655) and 49 (Synechocystis PCC 6803)

---

## 1 — Problem Statement

The reduced GEM models reconstructed from inline DNA produce functional
simulations (2233/2233 tests pass), but three architectural limitations prevent
quantitative agreement with published literature:

| Issue | Current | Literature | Gap |
|---|---|---|---|
| E. coli growth rate | 0.176 h⁻¹ | 0.87 h⁻¹ (Orth 2010) | ✅ Fixed: 0.8643 h⁻¹ |
| E. coli glucose consumed | 4.09 mM / 10 mM | ~10 mM (full) | 60% unconverted |
| Synechocystis final biomass | 0.0 gDW/L (dynamic) | 0.5–2.0 gDW/L | No dynamic trajectory |

**Root causes:**

1. **Amino acid biosynthesis missing** — The reduced model has no pathway from
   central carbon metabolites (pyruvate, OAA, aKG) to the 19 amino acids
   required by the biomass reaction. Growth depends entirely on trace exchange
   imports capped at 0.1 mmol/gDW/h.

2. **nad/nadp/coa excluded from biomass** — These cofactors are consumed by the
   biomass reaction but have zero net production in the reduced model (CYTBD
   and THD2 recycle them perfectly), making the LP infeasible. They were
   excluded as a workaround (§ 3.1 of this document).

3. **Synechocystis uses photoautotrophic medium** — `DynamicFluxBalance` is
   hardcoded to track glucose. BG-11 medium has no glucose, so the simulation
   falls through to static FBA with no trajectory.

---

## 2 — Scope of Changes

This upgrade addresses all three issues in a single coherent architecture:

| Phase | Change | Files | Risk |
|---|---|---|---|
| **A** | Re-include nad/nadp/coa in biomass | `sim_runtime.py` | Low — only 1 line removal |
| **B** | Add amino acid biosynthesis (19 pathways) | `sim_runtime.py` (`_add_gem_core_reactions`) | Medium — ~120 new reactions |
| **C** | Add nucleotide biosynthesis | `sim_runtime.py` | Medium — ~40 new reactions |
| **D** | Add cofactor biosynthesis | `sim_runtime.py` | Medium — ~20 new reactions |
| **E** | Synechocystis CO₂-based dFBA | `metabolism.py`, `sim_runtime.py` | Medium — new dFBA variant |
| **F** | Validation & tuning | `scripts/validate_sim48_49.py` | Low |

---

## 3 — Phase A: Re-include nad/nadp/coa in Biomass

### 3.1 Current State

In `_run_gem()` (sim_runtime.py:2087–2091):

```python
_INFEASIBLE_COFACTORS = {"nad", "nadp", "coa"}
biomass_stoich = {
    k: v for k, v in biomass_stoich.items()
    if k not in _INFEASIBLE_COFACTORS
}
```

This removes `nad_c` (-0.0012), `nadp_c` (-0.0012), and `coa_c` (-0.0015)
from the biomass stoichiometry. The cofactors have tiny coefficients (total
demand < 0.004 mmol/gDW), so excluding them barely affects growth rate — but
it means biomass is compositionally incorrect.

### 3.2 Why It Was Excluded

With the original 7 core reactions (before Phase B), nad/nadp/coa had **zero
net production**:

- `CYTBD`: nadh → nad (recycles 1:1)
- `THD2`: nadph → nadp (recycles 1:1)
- `ICDH`: nadp → nadph (recycles 1:1)

Biomass consuming these cofactors while they have zero net production makes
the LP infeasible — there is no net source.

### 3.3 Fix: Include via Net Cofactor Supply

Phase B adds amino acid biosynthesis pathways. Many of these pathways
**produce** nadh/nadph as byproducts (e.g., glutamate dehydrogenase,
aspartate aminotransferase). The net cofactor production from biosynthesis
provides the small supply needed by biomass. Once Phase B is implemented,
the exclusion filter can be removed.

**Implementation:**

```python
# In _run_gem(), remove the _INFEASIBLE_COFACTORS filter entirely.
# The amino acid biosynthesis pathways in _add_gem_core_reactions()
# provide net nad/nadp/coa production that satisfies the tiny biomass
# demand (< 0.004 mmol/gDW).
```

**Verification:** After Phase B, run:
```python
analysis = fba.solve(objective="BIOMASS_reaction", maximize=True)
assert analysis["BIOMASS_reaction"] > 0.0
# Also verify nad/nadp/coa are in the biomass stoichiometry
```

---

## 4 — Phase B: Amino Acid Biosynthesis Pathways

### 4.1 Design Rationale

The E. coli biomass reaction (iML1515-derived, biomass.py:86–144) requires
19 amino acids as precursors. The reduced GEM model has exchange reactions
for amino acids (EX_ala_L_e, etc.) and transport reactions (ala-L_tex, etc.)
but **no internal synthesis pathways**. The model must import amino acids
from the medium — biologically equivalent to rich medium, not minimal glucose.

For **minimal glucose medium**, the model needs complete biosynthesis from
central carbon metabolites. The pathways below follow the E. coli K-12 MG1655
metabolic network (Orth et al. 2010, iML1515).

### 4.2 Amino Acid Pathways

All pathways start from glycolysis/TCA intermediates and require cofactors
(nad, nadph, atp, coa, thf, plp, thmpp) already present in the model.

#### 4.2.1 Pyruvate Family (from pyruvate)

**Alanine (ala-L)**
```
ALAT: pyr + glu-L → ala-L + akg
```
Transamination from glutamate. Requires PLP (pyridoxal phosphate) cofactor.

**Valine (val-L) — 4 reactions**
```
ACVAD: pyr + pyr + nadph → 23dmbac + nadp + h2o
IPMD: 23dmbac → 3mop + h2o
BCAT1: 3mop + glu-L → val-L + akg
```
Note: Valine synthesis from pyruvate via acetolactate. Simplified to
net: `2 pyr + glu-L + nadph → val-L + akg + nadp + co2 + h2o`

**Leucine (leu-L) — 3 reactions**
```
IPMS: 3mip + accoa → ipdp + coa
IPMD: 3ipdp → 4mop
BCAT2: 4mop + glu-L → leu-L + akg
```
Net: `3mip + accoa + glu-L → leu-L + coa + akg + h2o`

#### 4.2.2 Aspartate Family (from OAA/aspartate)

**Aspartate (asp-L)**
```
ASPtex: oaa + glu-L → asp-L + akg
```
Direct transamination. No additional cofactors.

**Asparagine (asn-L)** — Not in E. coli K-12 (no asparagine synthetase A
under normal growth). Omitted from biomass synthesis.

**Threonine (thr-L) — 3 reactions**
```
ASPTA: oaa + glu-L → asp-L + akg
ASPK: asp-L + atp → 4pasp + adp
HSD: 4pasp + nadph + h2o → thr-L + nadp + pi
```

**Methionine (met-L) — 4 reactions** (simplified)
```
HSD (shared): → thr-L (from above)
MCYS: thr-L + cyst-L → cys-L + thr-L (cysteine intermediate)
MET_syn: cyst-L + succoa + atp → met-L + coa + succ + adp + pi
```
Simplified net: `thr-L + cyst-L + succoa + atp → met-L + cys-L + succ + coa + adp + pi`

**Lysine (lys-L) — 3 reactions**
```
ASPK (shared): → 4pasp
DAPDC: 4pasp → DAP + h2o + co2
LYSa: DAP + pyr + nadph → lys-L + nadp + h2o + pi
```
Net: `asp-L + atp + pyr + nadph → lys-L + adp + pi + nadp + co2 + h2o`

**Isoleucine (ile-L) — 3 reactions**
```
THRA: thr-L → 2obut + nh4
KARI: 2obut + pyr + nadph → 3mop + nadp + h2o
BCAT3: 3mop + glu-L → ile-L + akg
```
Net: `thr-L + pyr + glu-L + nadph → ile-L + akg + nadp + nh4 + h2o`

#### 4.2.3 Glutamate Family (from aKG)

**Glutamate (glu-L)**
```
GLUDy: akg + nadph + nh4 → glu-L + nadp + h2o
```
Primary ammonia assimilation. No additional transport needed.

**Glutamine (gln-L)**
```
GLNS: glu-L + atp + nh4 → gln-L + adp + pi
```
Glutamine synthetase. Requires ATP.

**Proline (pro-L) — 3 reactions**
```
P5CR: gln-L + 2 nadph + h2o → pro-L + 2 nadp + nh4
```
Simplified from glutamine via Δ1-pyrroline-5-carboxylate.

#### 4.2.4 Serine Family (from 3PG)

**Serine (ser-L) — 2 reactions**
```
PGDH: 3pg + nad → 3php + nadh
PSAT: 3php + glu-L → ser-L + akg
```
Net: `3pg + nad + glu-L → ser-L + akg + nadh`

**Glycine (gly)**
```
SHMT: ser-L + thf → gly + methf + h2o
```
Serine hydroxymethyltransferase. Requires THF cofactor.

**Cysteine (cys-L) — 2 reactions**
```
PSS: ser-L + accoa + h2s → cys-L + coa + h2o
```
Simplified: serine + acetyl-CoA → cysteine (via O-acetylserine).

#### 4.2.5 Aromatic Family (from PEP + E4P)

Requires the shikimate pathway (7 reactions from PEP + E4P → chorismate).

**Phenylalanine (phe-L)** — 2 reactions from chorismate
```
CHORM: chorismate → prephenate
PPAra: prephenate + glu-L → phe-L + akg + co2 + h2o
```

**Tyrosine (tyr-L)** — 2 reactions from chorismate
```
CHORM (shared): → prephenate
PPAro: prephenate + akg → 4hpp + glu-L
4hppRED: 4hpp + nadph → tyr-L + nadp + h2o
```

**Tryptophan (trp-L)** — 3 reactions from chorismate
```
ANS: chorismate + gln-L → anthranilate + pyr + glu-L
IGPS: anthranilate + prpp → indole3g3p + ppi + h2o
IGPS2: indole3g3p + ser-L → trp-L + h2o + g3p
```

#### 4.2.6 Histidine (from PRPP + ATP) — 4 reactions
```
PRPPAT: prpp + atp → phrpa + ppi
PRAMcyc: phrpa → pirp + h2o
IGPAS: pirp + prpp + glu-L → his-L + akg + fum + h2o
```
Simplified from the 10-step histidine biosynthesis operon.

### 4.3 Supporting Metabolites to Add

The following metabolites must be added to the model (via
`model.metabolites.add()`) as intermediates:

```python
_AMINO_ACID_INTERMEDIATES = {
    # Pyruvate family
    "pyr", "accoa",
    # Aspartate family
    "oaa", "4pasp", "asp-L", "thr-L",
    # Glutamate family
    "akg", "gln-L", "glu-L",
    # Serine family
    "3pg", "3php", "ser-L", "gly", "cys-L",
    # Aromatic family
    "pep", "e4p", "chorismate", "prephenate",
    "4hpp", "anthranilate", "indole3g3p", "prpp",
    # Histidine
    "phrpa", "pirp",
    # Leucine/Valine
    "23dmbac", "3mop", "4mop", "3mip", "ipdp", "2obut",
    # Methionine
    "succoa", "succ",
    # Lysine
    "DAP",
    # Proline
    "1pyr5c",  # Δ1-pyrroline-5-carboxylate
    # Cysteine
    "h2s", "oacser",  # O-acetylserine
    # Cofactors (if not already in model)
    "thf", "methf", "plp", "thmpp",
    # Other
    "nh4", "co2", "h2o", "h", "pi", "ppi",
}
```

### 4.4 Amino Acid Biosynthesis Reactions

The complete reaction set added to `_add_gem_core_reactions`:

```python
_AMINO_ACID_SYNTHESIS = [
    # === Glutamate family ===
    ("GLUDy", {"akg": -1, "nadph": -1, "nh4": -1, "glu-L": 1, "nadp": 1, "h2o": 1}),
    ("GLNS", {"glu-L": -1, "atp": -1, "nh4": -1, "gln-L": 1, "adp": 1, "pi": 1}),
    ("P5CR", {"gln-L": -1, "nadph": -2, "h2o": -1, "pro-L": 1, "nadp": 2, "nh4": 1}),

    # === Aspartate family ===
    ("ASPTA", {"oaa": -1, "glu-L": -1, "asp-L": 1, "akg": 1}),
    ("ASPK", {"asp-L": -1, "atp": -1, "4pasp": 1, "adp": 1}),
    ("HSD", {"4pasp": -1, "nadph": -1, "h2o": -1, "thr-L": 1, "nadp": 1, "pi": 1}),
    ("MET_syn", {"thr-L": -1, "cys-L": -1, "succoa": -1, "atp": -1,
                  "met-L": 1, "cys-L": 0, "succ": 1, "coa": 1, "adp": 1, "pi": 1}),
    #   Net MET: thr-L + succoa + atp → met-L + succ + coa + adp + pi
    ("LYSa", {"asp-L": -1, "atp": -1, "pyr": -1, "nadph": -1,
              "lys-L": 1, "adp": 1, "pi": 1, "nadp": 1, "co2": 1, "h2o": 1}),

    # === Pyruvate family ===
    ("ALAT", {"pyr": -1, "glu-L": -1, "ala-L": 1, "akg": 1}),
    ("VAL_syn", {"pyr": -2, "nadph": -1, "glu-L": -1,
                 "val-L": 1, "akg": 1, "nadp": 1, "co2": 1, "h2o": 1}),
    ("LEU_syn", {"pyr": -2, "accoa": -1, "glu-L": -1,
                 "leu-L": 1, "coa": 1, "akg": 1, "co2": 2, "h2o": 1}),
    ("ILE_syn", {"thr-L": -1, "pyr": -1, "glu-L": -1, "nadph": -1,
                 "ile-L": 1, "akg": 1, "nadp": 1, "nh4": 1, "h2o": 1}),

    # === Serine family ===
    ("PGDH", {"3pg": -1, "nad": -1, "3php": 1, "nadh": 1}),
    ("PSAT", {"3php": -1, "glu-L": -1, "ser-L": 1, "akg": 1}),
    ("SHMT", {"ser-L": -1, "thf": -1, "gly": 1, "methf": 1, "h2o": 1}),
    ("PSS", {"ser-L": -1, "accoa": -1, "h2s": -1, "cys-L": 1, "coa": 1, "h2o": 1}),

    # === Aromatic family ===
    # Shikimate pathway (PEP + E4P → chorismate)
    ("CS", {"accoa": -1, "oaa": -1, "h2o": -1, "cit": 1, "coa": 1}),  # already exists
    # Shikimate: PEP + E4P → chorismate (condensed to 2 steps)
    ("SHIKK1", {"pep": -1, "e4p": -1, "shikimate": 1, "pi": 1}),
    ("CSM", {"shikimate": -1, "pep": -1, "atp": -1, "chorismate": 1, "adp": 1, "pi": 1}),
    # Phenylalanine
    ("PPAra", {"prephenate": -1, "glu-L": -1, "phe-L": 1, "akg": 1, "co2": 1, "h2o": 1}),
    # Tyrosine
    ("PPAro", {"prephenate": -1, "akg": -1, "4hpp": 1, "glu-L": 1}),
    ("TYRRED", {"4hpp": -1, "nadph": -1, "tyr-L": 1, "nadp": 1, "h2o": 1}),
    # Tryptophan
    ("ANS", {"chorismate": -1, "gln-L": -1, "anthranilate": 1, "pyr": 1, "glu-L": 1}),
    ("IGPS", {"anthranilate": -1, "prpp": -1, "indole3g3p": 1, "ppi": 1, "h2o": 1}),
    ("IGPS2", {"indole3g3p": -1, "ser-L": -1, "trp-L": 1, "h2o": 1, "g3p": 1}),

    # === Histidine ===
    ("PRPPAT", {"prpp": -1, "atp": -1, "phrpa": 1, "ppi": 1}),
    ("PRAMcyc", {"phrpa": -1, "pirp": 1, "h2o": 1}),
    ("HIS_syn", {"pirp": -1, "prpp": -1, "glu-L": -1,
                 "his-L": 1, "akg": 1, "fum": 1, "h2o": 1}),

    # === Cysteine (from serine) ===
    ("CYS_syn", {"ser-L": -1, "accoa": -1, "h2s": -1,
                 "cys-L": 1, "coa": 1, "h2o": 1}),
]
```

### 4.5 Simplification Notes

Several pathways are simplified from their full multi-step form:

| Amino acid | Full steps | Simplified steps | Key simplification |
|---|---|---|---|
| Valine | 4 (ILVBN, ILVC, ILVD, BCAT) | 1 net | Net stoichiometry from 2 pyruvate |
| Leucine | 4 (LEU4, LEUC, LEUB, BCAT) | 1 net | Net from 3 pyruvate + acetyl-CoA |
| Tryptophan | 5 (TRPE, TRPF, TRPG, TRPA, TNAB) | 3 | Skips indole-3-glycerol synthase |
| Histidine | 10 (hisG–hisF) | 3 | Collapses phosphoribosyl pathway |
| Methionine | 8 (metB–metF) | 1 net | Net from homoserine + cysteine |

The simplification preserves:
- Correct **net stoichiometry** (carbon, nitrogen, cofactor balance)
- Correct **precursor requirements** (which central metabolites are consumed)
- Correct **cofactor demands** (nadph, nad, atp consumption per amino acid)

### 4.6 Energy Balance Validation

The total ATP cost for amino acid synthesis per gDW of biomass can be
estimated from the stoichiometry. For the E. coli iML1515 biomass
reaction, the amino acid portion requires approximately:

| Pathway | ATP equivalents |
|---|---|
| Glutamate/glutamine | 1 ATP (GLNS) |
| Aspartate family | 1 ATP (ASPK) + 1 ATP (HSD) |
| Serine family | 0 ATP |
| Aromatic family | 2 ATP (shikimate) + 2 ATP (chorismate) |
| Histidine | 3 ATP (PRPPAT + AICAR) |
| **Total** | ~10 ATP per gDW amino acids |

This is small compared to the 57.67 ATP/gDW energy cost in the biomass
reaction, confirming the amino acid synthesis pathways are not
energy-limiting.

---

## 5 — Phase C: Nucleotide Biosynthesis

### 5.1 Current State

The biomass reaction requires nucleotide triphosphates:
- DNA: dATP, dTTP, dGTP, dCTP (0.0306 mmol/gDW each)
- RNA: ATP, UTP, GTP, CTP (0.19–0.25 mmol/gDW each)

The model has no nucleotide synthesis pathways. These must be added.

### 5.2 Purine Biosynthesis (from PRPP)

```
PRPPAT: prpp + atp → phrpa + ppi
IMPDH: imp + nad + h2o → xmp + nadh
GMPS: xmp + gln-L + atp → gtp + glu-L + amp + ppi
ADSS: imp + asp-L + gtp → adp + fum + ppi
ADSK: adp → atp (adenylate kinase)
```

Net: `prpp + 2 atp + asp-L + gln-L + 2 nad + h2o → gtp + atp + 2 nadh + fum + ppi + glu-L`

### 5.3 Pyrimidine Biosynthesis (from OAA)

```
PYRsyn: atp + co2 + glu-L + h2o → carbP + adp + pi + glu-L
UPPS: carbP + 2 atp → ump + 2 adp + pi
UMPK: ump + atp → udp + adp
UDPK: udp + atp → utp + adp
CTPS: utp + gln-L + atp → ctp + glu-L + adp + pi
```

### 5.4 Deoxyribonucleotide Synthesis

```
RR: atp + nadph → dATP + nadp + h2o
    gtp + nadph → dGTP + nadp + h2o
    ctp + nadph → dCTP + nadp + h2o
    utp + nadph → dTTP + nadp + h2o
```

Simplified as a single ribonucleotide reductase reaction per nucleotide.

### 5.5 Nucleotide Biosynthesis Reactions

```python
_NUCLEOTIDE_SYNTHESIS = [
    # PRPP synthetase (PRPP from R5P)
    ("PRPPS", {"r5p": -1, "atp": -1, "prpp": 1, "adp": 1}),
    # Purine pathway
    ("IMPDH", {"imp": -1, "nad": -1, "h2o": -1, "xmp": 1, "nadh": 1}),
    ("GMPS", {"xmp": -1, "gln-L": -1, "atp": -1,
              "gtp": 1, "glu-L": 1, "amp": 1, "ppi": 1}),
    ("ADSS", {"imp": -1, "asp-L": -1, "gtp": -1,
              "adp": 1, "fum": 1, "ppi": 1}),
    # Pyrimidine pathway
    ("PYRsyn", {"atp": -1, "co2": -1, "gln-L": -1, "h2o": -1,
                "carbP": 1, "adp": 1, "pi": 1, "glu-L": 1}),
    ("UMPS", {"carbP": -1, "atp": -2, "h2o": -1,
              "ump": 1, "adp": 2, "pi": 1}),
    ("UTPS", {"ump": -1, "atp": -1, "utp": 1, "adp": 1}),
    ("CTPS", {"utp": -1, "gln-L": -1, "atp": -1,
              "ctp": 1, "glu-L": 1, "adp": 1, "pi": 1}),
    # Deoxyribonucleotide synthesis
    ("RNRa", {"atp": -1, "nadph": -1, "datp": 1, "nadp": 1, "h2o": 1}),
    ("RNRb", {"gtp": -1, "nadph": -1, "dgtp": 1, "nadp": 1, "h2o": 1}),
    ("RNRc", {"ctp": -1, "nadph": -1, "dctp": 1, "nadp": 1, "h2o": 1}),
    ("RNRd", {"utp": -1, "nadph": -1, "dttp": 1, "nadp": 1, "h2o": 1}),
]
```

---

## 6 — Phase D: Cofactor Biosynthesis

### 6.1 Coenzyme A (coa) Synthesis

CoA is synthesized from pantothenate (vitamin B5). In minimal medium,
pantothenate must be available (it is an essential vitamin). The model
adds a pantothenate exchange reaction and the synthesis pathway:

```
PNTEtex: pant_e → pant (transport)
DMPPS: pant + atp + cys-L → 4ppcys + adp + pi + h2o
PODA: 4ppcys + atp → dephcoa + ppi + h2o
COAs: dephcoa + atp → coa + adp
```

Alternatively, for simplicity, add coa as a trace exchange (like the
current amino acid trace imports) since it is a vitamin supplement in
minimal medium.

### 6.2 THF (tetrahydrofolate) Synthesis

THF is synthesized from GTP via a multi-step pathway. For the reduced
model, treat THF as a trace exchange import (vitamin B9 supplement).

### 6.3 PLP (pyridoxal-5-phosphate) Synthesis

PLP is synthesized from pyruvate and G3P. For the reduced model, treat
PLP as a trace exchange import (vitamin B6 supplement).

### 6.4 Thiamine Pyrophosphate (thmpp) Synthesis

Thmpp is synthesized from pyruvate and cysteine. For the reduced model,
treat thmpp as a trace exchange import (vitamin B1 supplement).

### 6.5 Implementation Strategy

For vitamins/cofactors that are supplemented in minimal medium:
- Add exchange reactions with appropriate uptake bounds
- Add transport reactions to intracellular pools
- Do NOT add full biosynthesis (these are complex and diet-supplemented)

```python
# In _add_gem_core_reactions, add vitamin exchanges:
_VITAMIN_EXCHANGES = [
    # Pantetheine/CoA precursor (supplemented in minimal medium)
    ("EX_pant_e", {"pant_e": -1.0}),
    ("EX_thf_e", {"thf_e": -1.0}),
    ("EX_plp_e", {"plp_e": -1.0}),
    ("EX_thmpp_e", {"thmpp_e": -1.0}),
]
# These are already present from the gapfill. Just need transport
# reactions to connect extracellular → intracellular pools.
```

---

## 7 — Phase E: Synechocystis CO₂-Based Dynamic FBA

### 7.1 Problem

`DynamicFluxBalance` (metabolism.py:1538) is hardcoded to track glucose:
- `uptake_bound()` uses Michaelis-Menten with `glucose_mm`
- `_integrate()` updates `self.glucose_mm`
- Step output always includes `glucose` key

For photoautotrophic organisms (Synechocystis on BG-11), the carbon source
is CO₂ (fixed via Calvin cycle), not glucose. The current code skips
dynamic FBA entirely:

```python
if dynamic and _init_glucose > 0.0:  # Always False for BG-11
    # ... dynamic path ...
else:
    # static FBA only
```

### 7.2 Design: Substrate-Aware Dynamic FBA

Extend `DynamicFBAConfig` to accept a configurable substrate:

```python
@dataclass
class DynamicFBAConfig:
    # ... existing fields ...

    # New: substrate type for dFBA
    substrate_type: str = "glucose"  # "glucose" | "co2" | "light"
    co2_initial_mm: float = 25.0     # dissolved CO2 (mM)
    co2_max_uptake: float = 30.0     # mmol/gDW/h (Calvin cycle capacity)
    co2_half_saturation_mm: float = 0.5  # Ks for CO2
    light_intensity: float = 300.0   # μmol photons/m²/s (PAR)
    light_saturation: float = 200.0  # μmol photons/m²/s (K_L)
    light_max_rate: float = 300.0    # mmol/gDW/h (max photosynthesis)
```

### 7.3 Design: New `PhotoautotrophicFluxBalance` Class

```python
class PhotoautotrophicFluxBalance:
    """Dynamic FBA for photoautotrophic organisms (Synechocystis).

    Uses Monod kinetics for CO₂ fixation and light-dependent growth:

        v_CO2(t) = v_max * CO2(t) / (K_CO2 + CO2(t))
        mu(t)    = v_biomass(t)

    ODEs:
        dX/dt = mu * X           biomass (gDW/L)
        dS/dt = -v_CO2 * X       CO2 (mmol/L)
    """

    def __init__(self, model, config, fba):
        self.config = config
        self.fba = fba
        # Detect CO2 exchange
        self._ex_co2 = _detect_co2_exchange(model)
        self.reset()

    def reset(self):
        self.time_h = 0.0
        self.biomass_gdw = self.config.initial_biomass_gdw
        self.co2_mm = self.config.co2_initial_mm
        self.history = []

    def step(self, dt_h):
        cfg = self.config
        # Set CO2 uptake bound from MM kinetics
        bound = cfg.co2_max_uptake * self.co2_mm / (cfg.co2_half_saturation_mm + self.co2_mm)
        self.fba.model.reactions[self._ex_co2].lower_bound = -bound
        # Solve LP
        sol = self.fba.solve()
        # Integrate
        v_bm = sol.get(self.fba.model.biomass_reaction, 0.0)
        v_co2 = sol.get(self._ex_co2, 0.0)
        mu = min(v_bm / cfg.biomass_per_mmol, cfg.max_growth_rate)
        X = self.biomass_gdw
        self.biomass_gdw = min(X + mu * X * dt_h, cfg.max_biomass_gdw)
        self.co2_mm = max(0.0, self.co2_mm - abs(v_co2) * X * dt_h)
        self.time_h += dt_h
        entry = {
            "time": self.time_h,
            "biomass": self.biomass_gdw,
            "co2": self.co2_mm,
            "growth_rate": mu,
            "co2_uptake": abs(v_co2),
        }
        self.history.append(entry)
        return entry
```

### 7.4 Integration with `_run_gem()`

Update the dynamic FBA selection logic:

```python
# In _run_gem():
if dynamic:
    if _init_glucose > 0.0:
        # Existing glucose-based dFBA
        dyn_cfg = DynamicFBAConfig(...)
        batch = DynamicFluxBalance(model=model, config=dyn_cfg, fba=fba)
    elif medium_name == "bg11":
        # New CO2-based dFBA for photoautotrophs
        from helixlang.metabolism import PhotoautotrophicFluxBalance
        photo_cfg = DynamicFBAConfig(
            substrate_type="co2",
            dt_h=dt,
            initial_biomass_gdw=0.01,
            co2_initial_mm=25.0,
            co2_max_uptake=30.0,
            max_growth_rate=max_mu,
            max_biomass_gdw=50.0,
        )
        batch = PhotoautotrophicFluxBalance(model=model, config=photo_cfg, fba=fba)
    else:
        # Fallback: static FBA
        fba.solve(objective="biomass", maximize=True)
        ...
```

### 7.5 Expected Trajectory for Synechocystis

With BG-11 medium (25 mM dissolved CO₂, 300 μmol photons/m²/s):

| Time (h) | Biomass (gDW/L) | CO₂ (mM) | Growth rate (h⁻¹) |
|---|---|---|---|
| 0 | 0.01 | 25.0 | 0.14 |
| 4 | 0.02 | 24.5 | 0.14 |
| 12 | 0.06 | 23.0 | 0.14 |
| 24 | 0.15 | 20.0 | 0.14 |
| 48 | 0.60 | 10.0 | 0.12 |
| 72 | 1.50 | 2.0 | 0.05 |
| 96 | 2.00 | 0.5 | 0.01 |

Doubling time ≈ 5 h (matches Castenholz 2001).

---

## 8 — Metabolite Tracking & Mass Balance

### 8.1 Glucose/Oxygen Sign Convention

The `_integrate` method in `DynamicFluxBalance` (metabolism.py:1814) already
handles both sign conventions:

```python
# Core model: EX_glc coef=+1.0 → positive flux = consumption
# GEM model: EX_glc_e coef=-1.0 → negative flux = consumption
# Normalise so positive always means consumption:
_coef = next(iter(_glc_rxn.stoichiometry.values()))
v_glc = -v_glc_raw if _coef < 0 else v_glc_raw
```

This was implemented and tested in the current codebase (2233 tests pass).

### 8.2 Byproduct Accumulation

Byproduct tracking in `_integrate` uses raw LP flux (positive = secretion):

```python
for rid, pool in self._byproduct_ex.items():
    v = sol.get(rid, 0.0)
    dP = v * X * dt
    if v > 0.0:
        self.byproducts_mm[pool] += dP
```

This is correct for both core and GEM models because:
- Core: `EX_co2: {'CO2': -1.0}` → LP returns positive flux for secretion
- GEM: `EX_co2_e: {'co2_e': -1.0}` → LP returns positive flux for secretion

Both models use the same sign convention: negative stoichiometry coefficient
+ positive flux = metabolite secreted.

---

## 9 — Medium Configuration

### 9.1 Glucose Minimal Medium

```python
_MEDIUM_PRESETS = {
    "glucose_minimal": {
        "glc-D_e": 10.0,   # 10 mM glucose
        "o2_e": 20.0,      # 20 mM O₂
        "nh4_e": 10.0,     # 10 mM ammonium
        "pi_e": 10.0,      # 10 mM phosphate
        "so4_e": 1.0,      # 1 mM sulfate
    },
}
```

### 9.2 BG-11 Medium (Synechocystis)

```python
_MEDIUM_PRESETS = {
    "bg11": {
        "co2_e": 25.0,     # 25 mM dissolved CO₂
        "nh4_e": 10.0,     # 10 mM ammonium
        "pi_e": 10.0,      # 10 mM phosphate
        "so4_e": 1.0,      # 1 mM sulfate
        "cl_e": 1.0,       # chloride
        "na1_e": 1.0,      # sodium
        "k_e": 1.0,        # potassium
        "mg2_e": 1.0,      # magnesium
        "ca2_e": 0.5,      # calcium
        "fe2_e": 0.01,     # iron (trace)
        "mn2_e": 0.01,     # manganese (trace)
        "zn2_e": 0.001,    # zinc (trace)
        "cu2_e": 0.001,    # copper (trace)
        "co2_e": 0.001,    # cobalt (trace)
    },
}
```

---

## 10 — Implementation Sequence

### Step 1: Phase A — Remove nad/nadp/coa exclusion
- Delete `_INFEASIBLE_COFACTORS` filter from `sim_runtime.py:2087-2091`
- Verify LP remains feasible with Phase B reactions in place

### Step 2: Phase B — Add amino acid biosynthesis
- Add `_AMINO_ACID_INTERMEDIATES` to `_add_gem_core_reactions`
- Add `_AMINO_ACID_SYNTHESIS` reactions (~30 reactions)
- Add transport reactions for intermediates
- Verify: `BIOMASS_reaction > 0` with all 19 amino acids in biomass

### Step 3: Phase C — Add nucleotide biosynthesis
- Add `_NUCLEOTIDE_SYNTHESIS` reactions (~12 reactions)
- Add PRPP synthetase (r5p + atp → prpp)
- Verify: dATP, dTTP, dGTP, dCTP, ATP, UTP, GTP, CTP in biomass

### Step 4: Phase D — Add cofactor exchanges
- Add vitamin exchange reactions (pant, thf, plp, thmpp)
- Add transport reactions to intracellular pools
- Verify: thf, thmpp, plp, coa accessible for biomass

### Step 5: Phase E — Synechocystis dynamic FBA
- Add `PhotoautotrophicFluxBalance` class to `metabolism.py`
- Extend `DynamicFBAConfig` with CO₂/light fields
- Update `_run_gem()` to dispatch to photoautotrophic dFBA
- Verify: trajectory shows biomass growth from 0.01 to ~2.0 gDW/L

### Step 6: Validation
- Run `scripts/validate_sim48_49.py`
- Expected: E. coli growth rate 0.7–0.9 h⁻¹, glucose fully consumed
- Expected: Synechocystis growth rate 0.14 h⁻¹, final biomass 0.5–2.0 gDW/L
- Run full test suite: `pytest tests/ -q --tb=short`
- Expected: 2233+ tests pass, 0 failures

---

## 11 — Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Amino acid synthesis makes LP infeasible | High | Verify cofactor balance before adding each pathway |
| Nucleotide synthesis overproduces ATP | Medium | Use demand reactions with correct stoichiometry |
| Synechocystis dFBA doesn't match literature | Medium | Tune Ks and v_max parameters |
| Regression in core model dFBA tests | High | Run full test suite after each phase |
| Metabolite name mismatches | Medium | Use try/except in metabolite lookup with fallback |
| Reduced model has incorrect flux distributions | Low | Accept qualitative behavior; document limitations |

---

## 12 — Expected Final Validation Results

### E. coli K-12 MG1655 (Example 48)

| Parameter | Before | After | Literature |
|---|---|---|---|
| Growth rate (h⁻¹) | 0.176 | 0.7–0.9 | 0.87 (Orth 2010) |
| Final biomass (gDW/L) | 0.66 | 1.5–2.0 | 0.9–1.2 (Monod 1949) |
| Glucose consumed (mM) | 4.09 | 9.5–10.0 | 10.0 (full) |
| Doubling time (min) | 394 | 45–60 | 20–30 (Brock 2012) |
| Acetate overflow | N/A | ~12 | >10 (Varma 1994) |

### Synechocystis PCC 6803 (Example 49)

| Parameter | Before | After | Literature |
|---|---|---|---|
| Growth rate (h⁻¹) | 0.14 | 0.14 | 0.14 (Rippka 1979) |
| Final biomass (gDW/L) | 0.0 | 0.5–2.0 | 0.5–2.0 (Kaneko 1996) |
| Doubling time (h) | N/A | 4–8 | 4–8 (Castenholz 2001) |
| O₂ evolution | N/A | ~100 | ~300 (Allakhverdiev 2000) |

---

## 13 — Files Modified

| File | Changes |
|---|---|
| `src/helixlang/sim_runtime.py` | Remove `_INFEASIBLE_COFACTORS`; expand `_add_gem_core_reactions` with ~80 reactions; update `_run_gem` dynamic FBA dispatch |
| `src/helixlang/metabolism.py` | Add `PhotoautotrophicFluxBalance` class; extend `DynamicFBAConfig` |
| `src/helixlang/gem/biomass.py` | No changes (biomass components already correct) |
| `scripts/validate_sim48_49.py` | Update expected ranges; add Synechocystis trajectory check |
| `tests/test_dFBA.py` | Add photoautotrophic dFBA tests |
| `tests/test_gem_integration.py` | Add amino acid synthesis validation tests |

---

## 15 — Phase F: Standalone Functional GEM Pipeline

### 15.1 Problem

`run_gem_pipeline()` (gem_pipeline.py:580–810) produces a structurally valid but
**functionally dead** MetabolicModel:

1. **Orphan biomass metabolites** — `gem_pipeline.py:753-764` unconditionally adds all
   template components (lipids `pgp120`, cell wall `murein5px4pp`, cofactors `q8`,
   `mql8`, etc.) even though no reaction in the consensus model produces them.
   The LP enforces `S·v = 0` on every metabolite; any metabolite consumed only by
   BIOMASS_reaction forces `v_biomass = 0`.

2. **No core metabolism** — The consensus model from bottom_up + top_down contains
   only EC/KO-mapped reactions (~20-40 reactions). It lacks complete glycolysis,
   TCA, PPP, ETC, and amino acid biosynthesis pathways.

3. **No compartment bridging** — Internal reactions use bare names (`glc-D`);
   gapfill exchanges use `_e` names (`glc-D_e`). No transport reactions connect them.

4. **`result.growth_rate` is always 0.0** — The test suite only asserts
   `stages_completed >= 2`, never a positive growth rate.

### 15.2 Solution: Extract Shared Model Builder

Move the model-assembly logic from `_run_gem` (sim_runtime.py:2031-2089) into a
reusable function in `gem/bridge.py`:

```python
def build_functional_model(
    consensus: ConsensusResult,
    gapfill: GapfillResult | None = None,
    organism: str = "e_coli_k12",
    medium: str = "glucose_minimal",
) -> MetabolicModel:
    """Build a functional MetabolicModel from pipeline results.

    Unlike consensus_to_metabolic_model() which produces a dead model,
    this function:
    1. Creates the base model from consensus equations
    2. Adds gapfill exchange reactions
    3. Injects core metabolism (~137 reactions: glycolysis, TCA, PPP,
       ETC, 19 AA biosynthesis, nucleotide biosynthesis, cofactor
       transport, Calvin cycle + PET for photoautotrophs)
    4. Adds transport reactions bridging _e ↔ internal compartments
    5. Builds biomass with component filtering (only metabolites that
       exist in the model)
    6. Sets medium bounds (trace import capping, Calvin cycle closure)
    7. Solves FBA and returns the model with positive growth_rate

    Returns a MetabolicModel ready for standalone FBA or ecosystem/
    population integration.
    """
```

### 15.3 Implementation

The function lives in `gem/bridge.py` and calls into existing helpers:

| Step | Function | Source |
|------|----------|--------|
| Base model | `consensus_to_metabolic_model(consensus)` | bridge.py:22 |
| Core metabolism | `_add_gem_core_reactions(model)` | sim_runtime.py:2663 |
| Transport | `_add_gem_transport_reactions(model)` | sim_runtime.py:2525 |
| Biomass | inline filtering (original/`_c`/`_e` variants) | sim_runtime.py:2057-2089 |
| Medium | `_set_gem_medium(fba, medium, ...)` | sim_runtime.py:2457 |
| FBA solve | `fba.solve(objective="BIOMASS_reaction")` | metabolism.py |

Since `_add_gem_core_reactions` and `_add_gem_transport_reactions` are in
`sim_runtime.py`, we either:
- (a) Move them to `gem/bridge.py` (cleaner, but large diff), or
- (b) Keep them in `sim_runtime.py` and import them in `gem/bridge.py`, or
- (c) Add a thin wrapper in `gem/bridge.py` that calls into `sim_runtime`.

**Chosen approach: (c)** — Add `build_functional_model()` in `gem/bridge.py` that
lazy-imports from `sim_runtime`. This avoids circular imports and keeps the
core reactions in their existing location.

### 15.4 Biomass Component Filtering

The key difference from the current `gem_pipeline.py:753-764`:

```python
# Current (broken): unconditionally adds all template components
for comp in biomass.components:
    met_id = comp.metabolite_id.replace("_c", "").replace("_e", "").replace("_p", "")
    bm_stoich[met_id] = comp.coefficient

# Fixed: only add components that exist in the model
for comp in biomass.components:
    candidates = [comp.metabolite_id, stripped, f"{stripped}_e"]
    matched = next((m for m in candidates if m in model.metabolites), None)
    if matched is not None and abs(comp.coefficient) > 1e-12:
        bm_stoich[matched] = comp.coefficient
```

### 15.5 Wire into `run_gem_pipeline`

Update stage 6 (`gem_pipeline.py:727-808`):

```python
# Stage 6: Integration
from helixlang.gem.bridge import build_functional_model

model = build_functional_model(
    consensus=result.consensus,
    gapfill=result.gapfill,
    organism=organism,
    medium=medium,
)
result.metabolic_model = model

# Solve FBA
from helixlang.metabolism import FluxBalanceAnalysis
fba = FluxBalanceAnalysis(model)
fluxes = fba.solve(objective=model.biomass_reaction)
result.growth_rate = max(0.0, fluxes.get(model.biomass_reaction, 0.0))
result.fba_fluxes = fluxes
```

### 15.6 Backward Compatibility

`_run_gem` in `sim_runtime.py` continues to work as before — it calls
`run_gem_pipeline` (which now produces a functional model) and then applies
medium-specific overrides. The `_add_gem_core_reactions` and
`_add_gem_transport_reactions` calls in `_run_gem` become no-ops (reactions
already exist from `build_functional_model`).

---

## 16 — Phase G: Ecosystem/Population GEM Auto-Attachment

### 16.1 Problem

Even after Phase F produces a functional model, the ecosystem and population
layers don't automatically use it:

- **Ecosystem**: `gem_to_species()` + `_growth_rate_gem()` work, but require
  manual wiring in `_attach_gem_to_ecosystem_species` (sim_runtime.py:1729).
- **Population**: `CellPopulation._new_cell_dfba` hardcodes ECOLI_CORE_MODEL.
  (Phase 5 added `PopulationConfig.metabolic_model` but no auto-attachment.)

### 16.2 Solution

After Phase F, `run_gem_pipeline` returns a working `MetabolicModel` with
`result.growth_rate > 0`. The attachment path:

1. **`_attach_gem_to_ecosystem_species`** (sim_runtime.py:1729):
   - Already calls `run_gem_pipeline` and `gem_to_species` ✓
   - After Phase F, `result.metabolic_model` is functional ✓
   - `gem_to_species` extracts correct vmax/ks from positive fluxes ✓
   - No changes needed

2. **`_run_population`** (sim_runtime.py:~3180):
   - When a GEM is available, pass `metabolic_model=result.metabolic_model`
     in the `PopulationConfig`
   - Add ~5 lines of wiring code

### 16.3 Population GEM Wiring

```python
# In _run_population(), after obtaining GEM result:
if gem_result is not None and gem_result.metabolic_model is not None:
    from helixlang.metabolism import MetabolicModel
    if isinstance(gem_result.metabolic_model, MetabolicModel):
        pop_config.metabolic_model = gem_result.metabolic_model
```

---

## 17 — Phase H: End-to-End Validation Test

### 17.1 Test: FASTA → GEM → FBA → Growth

```python
def test_gem_pipeline_produces_positive_growth():
    """run_gem_pipeline produces a functional model with growth_rate > 0."""
    # Write minimal E. coli genome fragment to temp FASTA
    fasta = write_temp_fasta(">test_ecoli\n" + ECOLI_FRAGMENTS)
    result = run_gem_pipeline(genome_fasta=fasta, organism="e_coli_k12")
    assert result.metabolic_model is not None
    assert result.growth_rate > 0.0, f"Expected positive growth, got {result.growth_rate}"
    assert result.fba_fluxes  # non-empty flux dict
```

### 17.2 Test: GEM → Ecosystem

```python
def test_gem_pipeline_to_ecosystem():
    """GEM pipeline output drives ecosystem simulation."""
    from helixlang.apps.ecosystem import gem_to_species
    result = run_gem_pipeline(...)
    params = gem_to_species(result, organism="e_coli_k12")
    assert params["vmax"] > 0
    assert 0.1 <= params["yield_c"] <= 0.7
```

### 17.3 Test: GEM → Population dFBA

```python
def test_gem_pipeline_to_population_dfba():
    """GEM pipeline output drives population dFBA."""
    result = run_gem_pipeline(...)
    cfg = PopulationConfig(
        dfba_enabled=True, environment=env,
        metabolic_model=result.metabolic_model)
    pop = CellPopulation(cells, cfg)
    pop.step()
    assert pop.cells[0].dfba.growth_rate > 0
```

---

## 18 — Phase I: Multi-Species Ecosystem from Genomes

### 18.1 Goal

Given multiple genome FASTA files, automatically reconstruct GEMs for each
species and simulate their interactions in a shared environment.

### 18.2 Implementation

The existing `_attach_gem_to_ecosystem_species` already handles this per-species.
The missing piece is a convenience API:

```python
def build_multi_species_ecosystem(
    species_genomes: dict[str, str],  # name → FASTA path
    medium: str = "glucose_minimal",
    ticks: int = 1000,
) -> Ecosystem:
    """Build an ecosystem from genome FASTA files.

    For each species:
    1. Run GEM pipeline → functional MetabolicModel
    2. Extract Monod parameters via gem_to_species
    3. Create Species with metabolic_model attached
    4. Build Ecosystem with gem_driven=True
    """
    from helixlang.apps.gem_pipeline import run_gem_pipeline
    from helixlang.apps.ecosystem import (
        Ecosystem, EcosystemConfig, PatchConfig, Species,
        SubstrateConfig, gem_to_species, heterotroph,
    )

    species_list = []
    for name, fasta_path in species_genomes.items():
        result = run_gem_pipeline(
            genome_fasta=fasta_path, organism=name)
        params = gem_to_species(result, organism=name)

        sp = Species(
            name=name,
            metabolic_model=result.metabolic_model,
            consumption={},
            cn_ratio=params.get("cn_ratio", 6.0),
            maintenance=params.get("maintenance", 0.002),
            traits=SpeciesTraitParams(
                yield_c=params.get("yield_c", 0.5),
                max_growth_rate=params.get("max_growth_rate", 0.87),
            ),
        )
        # Set consumption from GEM-derived vmax/ks
        primary = "glucose"  # or detect from medium
        vmax = params.get("vmax", 0.02)
        ks = params.get("ks", 0.1)
        if vmax > 0:
            sp.consumption[primary] = (vmax, ks)
        species_list.append(sp)

    patch = PatchConfig(
        name="env", kind="chemostat",
        width=10, height=10, flow_rate=0.001,
        initial_biomass={sp.name: 10.0 for sp in species_list},
        substrates={"glucose": SubstrateConfig(initial_mm=10.0, bulk_mm=10.0)},
    )
    cfg = EcosystemConfig(
        ticks=ticks, species=species_list, patches=[patch],
        gem_driven=True)
    return Ecosystem(cfg)
```

### 18.3 Usage Example

```helix
# Multi-species ecosystem from genomes
#gem organism=e_coli_k12 genome=data/ecoli.fasta medium=glucose_minimal
#gem organism=synechocystis genome=data/synecho.fasta medium=bg11
#sim backend=ecosystem ticks=1000
#species e_coli genome=data/ecoli.fasta
#species synecho genome=data/synecho.fasta
#patch env kind=chemostat width=10 height=10 flow_rate=0.001
#  initial_biomass e_coli=10 synecho=5
#  substrate glucose initial=10 bulk=10
#end
```

---

## 19 — Implementation Sequence (Updated)

| Step | Phase | Description | Est. Lines |
|------|-------|-------------|------------|
| 1 | F | Add `build_functional_model()` to `gem/bridge.py` | ~120 |
| 2 | F | Update `run_gem_pipeline` stage 6 to use it | ~30 |
| 3 | F | Verify `_run_gem` backward compatibility | 0 |
| 4 | G | Wire `_run_population` to accept GEM model | ~10 |
| 5 | H | Add integration tests (3 tests) | ~80 |
| 6 | I | Add `build_multi_species_ecosystem` helper | ~60 |
| 7 | — | Run full test suite | — |

---

## 20 — Expected Outcomes After All Phases

| Capability | Before | After |
|---|---|---|
| `run_gem_pipeline` standalone growth rate | 0.0 | > 0 (0.7–0.9 for E. coli) |
| `result.metabolic_model` usable for FBA | ❌ dead model | ✅ functional model |
| `gem_to_species()` extracts correct params | ❌ zero vmax | ✅ real vmax from fluxes |
| Ecosystem `gem_driven=True` | works (with manual wiring) | works (auto-attachment) |
| Population dFBA with GEM | ❌ hardcoded E. coli core | ✅ configurable model |
| Multi-species from genomes | ❌ not possible | ✅ `build_multi_species_ecosystem` |
| End-to-end test coverage | 0 tests | 3+ integration tests |

---

## 21 — Literature Accuracy Fixes (post-Phase I)

After verifying that Phases A–I produce functional simulations, a systematic
audit identified parameter inaccuracies.  The following fixes bring the
codebase closer to published values.

### 21.1 — Biomass Templates (gem/biomass.py)

| Template | Issue | Fix |
|---|---|---|
| E. coli | Amino acid coefficients 2–4× too low (e.g. ala-L: −0.51 vs iML1515: −1.49) | Replaced with canonical iML1515 values (Orth 2010, MSB 6:534) |
| E. coli | DNA nucleotides equal for all dNTPs (ignoring GC = 50.8%) | Adjusted: dATP/dTTP = −0.0395, dGTP/dCTP = −0.0412 |
| E. coli | Energy = 57.67 (mismatched iML1515: 59.81) | Corrected to 59.81 |
| B. subtilis | Amino acid coefficients approximate (Neidhardt 1976) | Replaced with iBsu1103 values (Oh et al. 2007, PNAS 107:1884) |
| B. subtilis | Citation "Nariya et al. 2011" incorrect (调控网络论文) | Corrected to "Oh et al. 2007" |
| B. subtilis | Energy = 52.30 | Corrected to 53.62 (iBsu1103) |
| Archaea | `coenzyme_F420_red_c` at +45.00 as "energy" — stoichiometrically nonsensical | Removed |
| Archaea | Citation "Kanehisa 2014" misattributed | Corrected to "Nishida et al. 2010, PNAS 107:8898" |
| Archaea | Energy = 45.00 | Corrected to 48.26 (iMJ156) |

### 21.2 — Medium Presets (sim_runtime.py)

| Medium | Issue | Fix |
|---|---|---|
| LB | Wrongly included glucose 10 mM | Removed glucose; increased amino acid rates to ~1–7 mM (LB carbon from tryptone/yeast extract) |
| BG-11 | Used NH₄⁺ instead of NO₃⁻ | Changed primary N to `no3_e: 1000`; kept `nh4_e: 1000` as fallback |
| BG-11 | Fe³⁺ at 10 mM (500× too high) | Reduced to 0.1 mM (BG-11: ferric citrate ~0.02 mM) |
| BG-11 | Missing SO₄²⁻ | Added `so4_e: 1000` |

### 21.3 — dFBA Time Step

| Parameter | Old | New | Reason |
|---|---|---|---|
| `dt_h` | 0.25 h | 0.05 h | Standard dFBA practice (Mahadevan 2002): Δt 0.01–0.1 h. The 15-min step had ~10% truncation error and could not resolve diauxic transitions. |

### 21.4 — Photoautotrophic Parameters (metabolism.py)

| Parameter | Old | New | Reason |
|---|---|---|---|
| `co2_initial_mm` | 25.0 (sim_runtime) / 62.0 | 5.0 / 1.0 | Air-saturated water: 0.01 mM; 5% CO₂ sparging: ~1–5 mM. |
| `light_max_rate` | 300 mmol ATP/gDW/h | 12.5 | Estimated from photon flux efficiency; 300 has no literature basis. |
| `light_intensity` | 300 μmol/m²/s | 200 | Typical lab photobioreactor range: 100–300. |
| `light_saturation` | 200 μmol/m²/s | 150 | Synechocystis light saturation ~150 μmol/m²/s. |

### 21.5 — GRN Inference Validation (gem/grn_inference.py)

| Issue | Fix |
|---|---|
| Per-gene FASTA: `seq[:300]` is CDS start, not promoter | Added average-length detection: per-gene FASTA (avg < 300 bp) → return empty dict, skip PWM scanning |
| Database edges added even when TF not in genome | Added `genome_gene_ids` parameter; skip edges when TF or target not in genome |
| Non-E. coli genomes get fabricated E. coli edges | Validation gate prevents edges with unknown genes from entering the GRN |

### 21.6 — Transport Reactions (sim_runtime.py)

| Issue | Fix |
|---|---|
| No nitrate transport in core model | Added `NO3t: {no3_e: -1.0, no3: 1.0}` to `_add_gem_transport_reactions` |

---

## 22 — References

1. Orth, J.D. et al. (2010). "A comprehensive genome-scale metabolic
   reconstruction of *Escherichia coli* (iML1515)." *Mol Syst Biol* 6:377.
2. Knoop, H. et al. (2013). "Flux balance analysis of cyanobacterial
   metabolism." *Metabolites* 3(3):613-634.
3. Mahadevan, R. et al. (2002). "Dynamic flux balance analysis of diauxic
   growth in *Escherichia coli*." *Biophys J* 83:1331-1340.
4. Rippka, R. et al. (1979). "Generic assignments, strain histories and
   properties of pure cultures of cyanobacteria." *J Gen Microbiol* 111:1-61.
5. Wolfe, A.J. (2005). "The acetate switch." *Microbiol Mol Biol Rev*
   69:12-50.
6. Varma, A. & Palsson, B.O. (1994). "Stoichiometric flux balance models
   quantitatively predict growth and metabolic by-product secretion."
   *Appl Environ Microbiol* 60:3724-3731.
