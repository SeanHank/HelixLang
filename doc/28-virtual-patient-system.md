# 28 — Virtual Patient System: Whole-Person Pharmacogenomic Simulation (Genome → Traits → Disease → Drugs → Outcomes)

> **Status:** IMPLEMENTED  
> **Depends on:** doc/27 (human pathology & drug simulation)  
> **Date:** 2026-08-24

---

## 1 — Motivation

Doc/27 delivered a working **human pathology + drug simulation** stack: Recon3D/Human1 GEM loading, tissue overlays, disease perturbation (`human/disease.py`), drug specification (`human/drug.py`), PBPK ADME (`human/pharmacokinetics.py`), PD dose-response (`human/pharmacodynamics.py`), and long-term dFBA+PK/PD integration (`human/simulation.py`).

But the doc/27 patient is an **abstract composite**: a default `HumanPhysiology` (70 kg, 170 cm, 30-year-old) with manually-specified disease genes and drug PK constants. It cannot answer the question a clinician or pharmacologist would actually ask:

> *"Given THIS person — this genome, this age, this weight, these habits, these diseases — and THIS drug regimen, what happens to EVERYTHING measurable in their body, during treatment AND after they stop?"*

Specifically, doc/27 lacks:

- **Genomic input.** No way to ingest a person's variants. A CYP2D6 poor metabolizer (*4/*4) clears codeine, metoprolol, and tamoxifen at ~10% of normal rate — the doc/27 simulator treats them identically to an extensive metabolizer.
- **External trait input.** Age, sex, weight/height (organ volumes, cardiac output, GFR, CYP capacity all scale with these), smoking (CYP1A2 induction up to 2-fold), alcohol (CYP2E1), pregnancy (cardiac output +50%, renal clearance +50%), ethnicity-informed allele priors — none are representable.
- **Clinical observables.** The doc/27 result exposes drug concentrations, fluxes, and metabolic biomarkers. It produces **no ALT, no creatinine/eGFR, no CBC, no electrolytes, no lipids, no CRP** — nothing a lab report contains.
- **Vital signs.** No blood pressure, heart rate, respiratory rate, temperature, SpO₂, or body-weight trajectory.
- **Disease staging and progression.** `DiseaseState.severity` is a static scalar. There is no CKD staging by eGFR, no Child-Pugh scoring, no NYHA class, and no dynamic progression/response/relapse machinery.
- **Drug-drug interactions.** Co-administering simvastatin with clarithromycin (CYP3A4 inhibition, AUC ↑ ~10×) or the "triple whammy" nephrotoxic combination is invisible to doc/27.
- **Post-treatment recovery.** The doc/27 simulation ends at `duration_days`. Nothing models washout, biomarker normalization kinetics, liver/kidney recovery, steroid/opioid rebound, or permanent sequelae such as cisplatin ototoxicity.

The goal of doc/28 is the **Virtual Patient System**: a single façade that accepts (1) a complete genome, (2) external traits, (3) diseases with parameters, and (4) a drug specification, and predicts **all body parameter changes during and after medication**, as a continuous hourly time series.

**Baseline standard: 100% real.** Every scaling law, staging cutoff, interaction fold-change, and recovery half-life is anchored to published clinical literature. No mocks, no stubs, no placeholders.

---

## 2 — Current-State Audit (post-doc/27)

| Capability | Status | Module | Notes |
|---|---|---|---|
| Genome/variant ingestion (VCF) | ✗ Missing | — | Nothing consumes genotype data anywhere in the package |
| CYP450 metabolizer phenotype calling | ✗ Missing | — | `HumanPhysiology.cytochrome_p450_activity` exists as a dict but is never derived from genotype |
| External traits (age/sex/BMI/smoking/etc.) | ✗ Missing | — | `HumanPhysiology` has weight/height/age/sex only; no smoking, alcohol, pregnancy, exercise, comorbidities |
| Trait → physiology scaling laws | ✗ Missing | — | Organs fixed at adult-male defaults regardless of input |
| Clinical laboratory values | ✗ Missing | — | No ALT/AST/creatinine/CBC/electrolytes anywhere |
| Vital signs dynamics | ✗ Missing | — | No BP/HR/RR/temp/SpO₂ model |
| Disease staging (eGFR/Child-Pugh/NYHA) | ✗ Missing | — | `DiseaseState.severity` is static [0,1]; no staging rubrics |
| Dynamic progression / relapse | ✗ Missing | — | Severity never evolves during `HumanSimulation.run()` |
| Drug-drug interactions | ✗ Missing | — | `config.drugs` is a flat list; each PBPK engine independent |
| Post-treatment recovery & rebound | ✗ Missing | — | Simulation stops at last dose; no washout phase |
| Permanent sequelae tracking | ✗ Missing | — | Toxicity events are transient log entries |
| Unified whole-person façade | Partial | `human/simulation.py` | `HumanSimulation` integrates GEM+PK/PD but takes no genome/traits and emits no clinical outputs |
| PBPK / PD / dFBA engines | ✓ Exists | doc/27 | Directly reused as the inner engine of doc/28 |
| Disease application to GEM | ✓ Exists | `human/disease.py` | `apply_disease_state()` reused |
| Drug entity + SMILES parsing | ✓ Exists | `human/drug.py` | Reused; extended with regimen metadata |
| Time-series recording spine | ✓ Exists | `human/simulation.py` | Recording pattern extended to labs/vitals/stages |

**Key finding:** doc/27 supplies the *engine room* (PBPK, PD, dFBA). Doc/28 supplies everything around it: the *inputs* (genome, traits), the *outputs* (labs, vitals, stages), the *couplers* (DDI, progression, recovery), and the *façade* (`VirtualPatient`).

---

## 3 — Architecture

### 3.1 — Module Map

| # | Module Path | Purpose |
|---|---|---|
| 1 | `human/genotype.py` | `Variant`, `GenotypeProfile` — VCF ingestion, CYP450 star-allele calling, UM/EM/IM/PM classification, metabolism-rate modifiers |
| 2 | `human/phenotype.py` | `ExternalTraits`, `PhenotypeCalculator` — trait-driven scaling of organ volumes, flows, clearances, CYP activities |
| 3 | `human/clinical_output.py` | `ClinicalLabs`, `ClinicalLabModel` — hepatic/renal/CBC/metabolic/lipid/inflammatory panels computed from state |
| 4 | `human/vitals.py` | `VitalSigns`, `VitalsModel` — hemodynamics, baroreflex, thermoregulation, oxygenation, weight balance |
| 5 | `human/disease_progression.py` | `DiseaseStage`, `DiseaseProgressionModel` — clinical staging (eGFR/Child-Pugh/NYHA) + dynamic severity evolution |
| 6 | `human/ddi.py` | `DDIRule`, `DDIModel` — CYP/P-gp interactions, induction ramps, additive toxicity channels |
| 7 | `human/recovery.py` | `RecoveryModel` — post-washout biomarker relaxation, organ regeneration, rebound, permanent sequelae, relapse/remission |
| 8 | `human/virtual_patient.py` | `VirtualPatient` façade — hourly integration loop, `VirtualPatientResult` |

### 3.2 — Layer Diagram

```
        INPUTS                     INTERPRETATION LAYER              PHYSIOLOGY CORE
┌────────────────────┐    ┌──────────────────────────────┐    ┌─────────────────────────┐
│ Genome (VCF text)  │───▶│ human/genotype.py            │    │ HumanPhysiology (doc/27)│
│  all genes, all    │    │  Variant · GenotypeProfile   │───▶│  organ volumes, blood   │
│  variants          │    │  CYP2D6/2C19/2C9/3A4/1A2/2E1 │    │  flows, plasma volume   │
└────────────────────┘    │  → UM/EM/IM/PM + modifiers   │    │  CYP450 activity dict   │
┌────────────────────┐    └──────────────────────────────┘    │            ▲            │
│ ExternalTraits     │    ┌──────────────────────────────┐    │            │ rescaled    │
│  age sex wt ht     │───▶│ human/phenotype.py           │───▶│            └────────────│
│  ethnicity smoke   │    │  ExternalTraits · BMI        │    └───────────┬─────────────┘
│  alcohol preg ex   │    │  PhenotypeCalculator         │                │
│  comorbidities     │    └──────────────────────────────┘                │
└────────────────────┘                                                    │
┌────────────────────┐    ┌──────────────────────────────┐                │
│ Diseases + params  │───▶│ human/disease_progression.py │◀───────────────┤
│  type/severity/    │    │  DiseaseStage · staging      │    staging uses labs
│  stage/onset       │    └──────────────┬───────────────┘                │
└────────────────────┘                   │ severity(t)                    │
                                         ▼                                ▼
┌────────────────────┐    ┌──────────────────────────────────────────────────────────┐
│ Drug regimen       │───▶│                    ENGINE ROOM (doc/27)                  │
│  type SMILES dose  │    │  human/ddi.py → effective clearances per drug/enzyme     │
│  route interval    │    │        │                                                 │
│  duration          │    │        ▼                                                 │
│  observation       │    │  pharmacokinetics.PBPKModel ─▶ pharmacodynamics (Hill)   │
└────────────────────┘    │        │                          │                      │
                          │        ▼                          ▼                      │
                          │  organ C(t) ──────────▶ dFBA on disease-perturbed GEM     │
                          └──────────────────────────────┬───────────────────────────┘
                                                         │ physiology + disease +
                                                         │ drug effects + time
                                                         ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │                  OBSERVATION LAYER                       │
                          │  human/clinical_output.py  → ALT AST creatinine eGFR CBC │
                          │  human/vitals.py           → BP HR RR temp SpO2 weight   │
                          └──────────────┬───────────────────────────────────────────┘
                                         │ labs feed back into staging & progression
                                         ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │                   OUTCOME LAYER                          │
                          │  progression: stage transitions, response, relapse       │
                          │  human/recovery.py: washout → recovery → sequelae        │
                          │  clinical_events: Hy's law, AKI, QT, myopathy, …         │
                          └──────────────────────────────────────────────────────────┘
```

---

## 4 — Module 1: `human/genotype.py` — Genome → Pharmacogenomic Phenotype

### 4.1 — Design

Accepts a **complete genome** as VCF text (all genes, all variants). Only pharmacogenomically actionable loci influence simulation; all other variants are stored and ignored (forward-compatible). Allele → function assignments follow **PharmVar** (Gaedigk et al. 2018) and **CPIC** guidelines.

Two layers of output:

1. **Metabolizer status** per CYP gene: ultrarapid (UM), extensive/normal (EM/NM), intermediate (IM), poor (PM).
2. **Modifiers**: numeric multipliers applied to PBPK intrinsic clearances and to disease-risk terms.

### 4.2 — Core Data Structures

```python
Zygosity = Literal["het", "hom_ref", "hom_alt"]

@dataclass(frozen=True)
class Variant:
    """One called variant from a person's genome."""
    gene_id: str            # HGNC symbol: "CYP2D6", "VKORC1", "TPMT", "GBA1", ...
    chromosome: str         # "chr22"
    position: int           # GRCh38 1-based coordinate
    ref: str                # reference allele, e.g. "G"
    alt: str                # alternate allele, e.g. "A"
    zygosity: Zygosity      # het | hom_ref | hom_alt
    rsid: str = ""          # dbSNP id if known, e.g. "rs16947"
    star_allele: str = ""   # PharmVar assignment, e.g. "*2"

@dataclass
class MetabolizerStatus:
    """Called phenotype for one pharmacogene."""
    gene: str                     # "CYP2D6"
    category: str                 # "UM" | "EM" | "NM" | "IM" | "PM"
    activity_score: float         # CPIC-style: 0.0 .. >= 2.0
    allele_functions: list[tuple[str, str]]  # [("*4", "nonfunctional"), ("*1", "normal")]
    confidence: str               # "called" | "defaulted"  (defaulted = no covering data)

@dataclass
class GenotypeProfile:
    """Complete pharmacogenomic profile of one person."""
    variants: list[Variant]                       # ALL input variants (full genome)
    metabolizer_status: dict[str, MetabolizerStatus] = field(default_factory=dict)

    def cyp_status(self, gene: str) -> MetabolizerStatus: ...
    def metabolism_rate_modifier(self, cyp_fraction: dict[str, float]) -> float: ...
    def disease_risk_modifier(self, disease_gene: str) -> float: ...
```

### 4.3 — Star-Allele → Function Mapping (literature-anchored)

Curated subset shipped in `human/data/pgx.json`:

| Gene | Allele | Defining variant | Function | Activity | Source |
|---|---|---|---|---|---|
| CYP2D6 | *1 | — (reference) | Normal | 1.0 | PharmVar |
| CYP2D6 | *4 | 1846G>A splice defect | Nonfunctional | 0.0 | Gaedigk 2018 |
| CYP2D6 | *5 | Gene deletion | Nonfunctional | 0.0 | PharmVar |
| CYP2D6 | *10 | 100C>T P34S | Decreased | 0.5 | PharmVar |
| CYP2D6 | *41 | 2988G>A splice | Decreased | 0.5 | PharmVar |
| CYP2D6 | ×N | Duplication | Normal × copies | n | CPIC |
| CYP2C19 | *1 | — | Normal | 1.0 | PharmVar |
| CYP2C19 | *2 | 681G>A splice | Nonfunctional | 0.0 | PharmVar |
| CYP2C19 | *3 | 636G>A W212X | Nonfunctional | 0.0 | PharmVar |
| CYP2C19 | *17 | −806C>T promoter | Increased | 1.5 | Scott 2013 |
| CYP2C9 | *2 | 430C>T R144C | Decreased | 0.5 | PharmVar |
| CYP2C9 | *3 | 1075A>C I359L | Very decreased | 0.15 | Johnson 2017 |
| CYP3A4 | *1 | — | Normal | 1.0 | PharmVar |
| CYP3A5 | *3 | 6986A>G splice | Nonexpresser | 0.05 | PharmVar |
| CYP1A2 | *1F | −163C>A promoter | Inducible (with smoke) | 1.0 (+induction) | Sachse 1999 |
| CYP2E1 | — | Regulatory haplotype | Normal (ethanol-inducible) | 1.0 | Novak 2001 |

### 4.4 — Category Assignment Rules

CPIC-style activity-score summation (Caudle 2017; Scott 2013):

| Gene | UM | EM/NM | IM | PM |
|---|---|---|---|---|
| CYP2D6 | score ≥ 2.25 | 1.25 – 2.0 | 0 < score ≤ 1.0 | 0.0 |
| CYP2C19 | score > 2.0 (incl. *17) | 1.5 – 2.0 | 1.0 | 0.0 – 0.5 |
| CYP2C9 | — (no UM class) | *1/*1 | *1/*2, *2/*2 | carries *3 |
| CYP3A4/5 | expresser ×N | *1/*1 | 3A5 *3 het | 3A5 *3/*3 |

If the genome does not cover a pharmacogene's defining loci, status is `"defaulted"` to population prior (EM) with a `clinical_events` note — never silently assumed.

### 4.5 — Computed Outputs

```python
def metabolism_rate_modifier(self, cyp_fraction: dict[str, float]) -> float:
    """Overall intrinsic-clearance multiplier for a drug whose CYP partition
    is {enzyme: fraction_of_metabolism}. Enzyme-level factors:
    UM -> 2.0, EM -> 1.0, IM -> 0.5, PM -> 0.1 (residual extrahepatic).
    Returns sum_i fraction_i * factor_i."""
```

`disease_risk_modifier(gene)` folds carrier-status multipliers into the progression module, e.g. `APOE ε4` (AD risk ×2–3), `HFE` C282Y hom (hemochromatosis), `LDLR` pathogenic variants (FH), `SERPINA1` Z/Z (emphysema).

### 4.6 — VCF Parser

```python
def create_genotype_from_vcf(vcf_text: str) -> GenotypeProfile:
    """Parse VCF 4.2/4.3 text into a GenotypeProfile.
    Rules:
      - Skip header/meta lines; require CHROM POS ID REF ALT FORMAT columns.
      - Multi-allelic sites are split; star alleles matched on (gene, pos, ref, alt).
      - Sites with DP < 10 or missing GT are skipped (recorded in parse_report).
      - Zygosity from GT: 0/0 -> hom_ref, 0/1|0/2 -> het, 1/1 -> hom_alt.
      - Unrecognized variants are retained in .variants but carry star_allele="".
    """
```

Copy-number events (CYP2D6 duplications) may be supplied through an optional `##PGxCNV` meta line or the `star_allele="xN"` convention; absence of CNV data leaves the gene defaulted.

---

## 5 — Module 2: `human/phenotype.py` — External Traits → Organ Scaling

### 5.1 — Design

`PhenotypeCalculator` takes `GenotypeProfile + ExternalTraits` and returns a **rescaled `HumanPhysiology`** (doc/27 §4.4). Every multiplier is deterministic, literature-cited, and clamped to plausible physiological ranges.

### 5.2 — Core Data Structure

```python
@dataclass
class ExternalTraits:
    """Everything observable about the person that is not genomic."""
    age_years: float = 35.0
    sex: str = "male"                    # "male" | "female"
    weight_kg: float = 70.0
    height_cm: float = 170.0
    ethnicity: str = "unspecified"       # informs allele priors + renal baselines
    smoking_pack_years: float = 0.0      # packs/day × years
    currently_smoking: bool = False
    alcohol_drinks_per_week: float = 0.0
    exercise_level: str = "light"        # sedentary | light | moderate | vigorous
    pregnancy: bool = False
    pregnancy_trimester: int = 0         # 1..3 when pregnant
    comorbidities: list[str] = field(default_factory=list)  # ["hypertension","T2DM","HF"]

    @property
    def bmi(self) -> float:
        """weight_kg / (height_m ** 2)"""
```

### 5.3 — Scaling Laws (all cited)

| Target | Equation / Multiplier | Basis |
|---|---|---|
| BSA | Mosteller: `sqrt(H_cm × W_kg / 3600)` m² | Mosteller 1987, NEJM |
| Liver volume | `SLV_mL = 1021 × BSA − 222`; ×1.10 if BMI ≥ 30 (steatosis) | Vauthey 1999; d'Assignies 2013 |
| Kidney volume | `× BSA/1.73` relative to 300 mL | Nyengaard 1992 |
| Cardiac output | `5.0 L/min × (1 − 0.010 × max(0, age−30))`; +45% pregnancy T3 | Guyton 2016; Anderson 2005 |
| Plasma volume | 3.0 L; ×1.45 pregnancy; −10% BMI ≥ 35 relative expansion | Guyton 2016; Anderson 2005 |
| Baseline GFR | `125 − 0.75 × max(0, age−30)` mL/min; ×1.5 pregnancy; comorbidity hits | KDIGO 2013 |
| Muscle volume | 24 L × (W/70)^0.7 × sex factor (M 1.0, F 0.78) × exercise factor | Janssen 2000 |
| Adipose volume | from BMI: `fat_mass = W × bf(BMI, sex)` (Deurenberg 1991) | Deurenberg 1991 |
| Hepatic blood flow | `× (1 − 0.005 × max(0, age−30))`; ↓15% if BMI ≥ 30 steatosis | Zeeh & Platt 2002 |

### 5.4 — CYP450 Activity Modulation

Applied multiplicatively onto genotype-derived enzyme activity:

| Factor | Condition | Enzymes affected | Magnitude | Source |
|---|---|---|---|---|
| Age decline | age > 30 | Phase I (esp. CYP3A) | ×(1 − 0.005·(age−30)), floor 0.6 | Zeeh 2002; Schmucker 2003 |
| Smoking induction | current smoker, pack_years > 10 | CYP1A2 | ×1.8 (range 1.5–2.4) | Kroon 2007 |
| Smoking waning | quit < 4 wk | CYP1A2 | linear decay to ×1.0 | Kroon 2007 |
| Alcohol induction | > 14 drinks/wk | CYP2E1 | ×2.0 (range 1.5–3.0) | Novak 2001 |
| Exercise | vigorous | global lean-mass linked | creatinine prod ×1.10 | Janssen 2000 |
| Pregnancy T2/T3 | pregnant | CYP2D6 ×1.5; CYP3A4 ×1.3; CYP1A2 ×0.5; CYP2C19 ×0.5 | | Anderson 2005 |
| Ethnicity prior | unspecified genotype loci | population allele frequencies (CYP2C19 PM: ~3% EUR, ~15% EA; CYP2D6 *10: ~45% EA) | Bayesian default | PharmGKB |

### 5.5 — Interface

```python
class PhenotypeCalculator:
    def __init__(self, traits: ExternalTraits, genotype: GenotypeProfile | None = None): ...

    def apply(self, physiology: HumanPhysiology) -> HumanPhysiology:
        """Return a rescaled copy (never mutates input):
        organ volumes, blood flows, plasma volume, cardiac output,
        hematocrit (pregnancy-diluted), cytochrome_p450_activity dict,
        albumin (pregnancy ↓ ~0.8 g/dL), baseline GFR."""
```

---

## 6 — Module 3: `human/clinical_output.py` — Clinical Lab Values

### 6.1 — Design

Translates the internal state (physiology + disease + drug effects + metabolite pools + time) into a **standard clinical chemistry report**, updated every simulated hour.

### 6.2 — Data Structure

```python
@dataclass
class ClinicalLabs:
    # Hepatic panel
    alt_u_per_l: float = 25.0            # ULN 33 (M) / 25 (F), Prati 2002
    ast_u_per_l: float = 25.0
    alp_u_per_l: float = 90.0
    ggt_u_per_l: float = 30.0
    bilirubin_total_mg_dl: float = 0.8
    bilirubin_direct_mg_dl: float = 0.2
    albumin_g_dl: float = 4.5
    inr: float = 1.0
    # Renal panel
    creatinine_mg_dl: float = 0.9
    bun_mg_dl: float = 15.0
    egfr_ml_min_1_73m2: float = 95.0
    cystatin_c_mg_l: float = 0.85
    # Hematology (CBC)
    wbc_k_ul: float = 7.0
    rbc_m_ul: float = 4.9
    hemoglobin_g_dl: float = 14.5
    hematocrit_pct: float = 43.0
    platelets_k_ul: float = 260.0
    mcv_fl: float = 90.0
    mch_pg: float = 30.0
    # Metabolic panel
    glucose_mg_dl: float = 92.0
    hba1c_pct: float = 5.4
    sodium_mmol_l: float = 140.0
    potassium_mmol_l: float = 4.2
    chloride_mmol_l: float = 104.0
    bicarbonate_mmol_l: float = 25.0
    calcium_mmol_l: float = 2.35
    phosphate_mmol_l: float = 1.2
    # Lipid panel
    total_cholesterol_mg_dl: float = 190.0
    ldl_mg_dl: float = 115.0
    hdl_mg_dl: float = 52.0
    triglycerides_mg_dl: float = 120.0
    # Inflammatory markers
    crp_mg_l: float = 1.0
    esr_mm_h: float = 8.0
```

### 6.3 — `ClinicalLabModel`

```python
class ClinicalLabModel:
    def compute(self, physiology, disease_state, drug_effects,
                metabolite_pools, t_h) -> ClinicalLabs: ...
```

Sub-models (each first-order toward its driver, integrated hourly):

**Hepatic injury (CYP-mediated hepatotoxicity).** Reactive-metabolite load λ (per drug: concentration × bioactivation fraction, e.g. acetaminophen→NAPQI gated by GSH pool and CYP2E1 status) drives hepatocyte death rate; ALT/AST rise proportionally and decay with their plasma half-lives (ALT t½ ≈ 47 h, AST ≈ 17 h — Kim 2008). **Hy's-law event** raised when `alt > 3×ULN and bilirubin_direct > 2 mg/dL` (Temple 2006). Chronic cholestatic pattern: ALP/GGT + conjugated bilirubin.

**Renal.**
- Creatinine: production `P_cr ≈ 20 mg/kg/day (lean mass adjusted)`; steady state `Cr = P_cr / (k × GFR)`; hourly first-order approach with t½ ~ 1–3 days (inverse of GFR).
- **eGFR: CKD-EPI 2021 race-free** (Inker 2021):
  `eGFR = 142 × min(Scr/κ,1)^α × max(Scr/κ,1)^−1.200 × 0.9938^age × 1.012[female]`,
  κ = 0.7/0.9 (F/M), α = −0.241/−0.302 (F/M).
- Nephrotoxic exposure (cisplatin cumulative AUC, aminoglycoside trough) reduces functional GFR; cystatin-C follows GFR with less muscle dependence (used as cross-check).

**CBC.**
- Myelosuppression: chemotherapy kills cycling progenitors; `WBC(t)` falls to nadir at ~day 7–10 post-dose (granulocyte transit), recovers over ~2–4 weeks (Friberg & Karlsson 2003 semi-mechanistic model, simplified).
- Anemia: CRP-driven hepcidin → iron sequestration (chronic-disease pattern, MCV low-normal), cisplatin erythroid toxicity, occult bleeding when `INR > 3.5` (Hb drop with compensatory reticulosis omitted).
- Platelets mirror WBC with longer recovery.

**Metabolic.**
- Glucose: from dFBA blood-glucose pool + metformin-class effect (hepatic gluconeogenesis ↓) + insulin sensitivity of traits.
- **HbA1c: `dA1c/dt = (A1c_eq(mean_glucose) − A1c)/τ`, τ ≈ 35 d equivalent** (RBC lifespan ~120 d; Nathan 2007 ADAG study).
- Sodium/potassium: renal handling + drug effects (loop/thiazide ↓Na↓K, ACEi/ARB/spironolactone ↑K, insulin shifts K intracellular).
- Calcium/phosphate: CKD stage ≥ G4 → phosphate ↑, calcium ↓ (secondary hyperparathyroidism trend).

**Lipids.** Statin intensity → LDL fractional reduction (high ≥50%, moderate 30–49%, low <30%; Grundy 2018 ACC/AHA) with receptor-turnover t½ ≈ 4 d. Triglycerides track adipose flux.

**Inflammation.** CRP production ∝ IL-6 proxy (tumor burden, infection flag, tissue necrosis from injury models); CRP t½ ≈ 19 h (Pepys & Hirschfield 2003). ESR integrates fibrinogen over days.

---

## 7 — Module 4: `human/vitals.py` — Vital Signs Dynamics

### 7.1 — Data Structure

```python
@dataclass
class VitalSigns:
    systolic_bp_mmhg: float = 118.0
    diastolic_bp_mmhg: float = 76.0
    heart_rate_bpm: float = 70.0
    respiratory_rate_per_min: float = 14.0
    temperature_c: float = 36.8
    spo2_pct: float = 98.0
    weight_kg: float = 70.0
```

### 7.2 — `VitalsModel.step(dt_h, labs, drug_effects, disease_states)`

**Hemodynamics (Guyton 2016):**

```
MAP = CO(L/min) × SVR(dyn·s·cm⁻⁵) / 80        # conversion 80 dyne→mmHg
PP  = SV / C_arterial                          # pulse pressure
SV  = CO / HR ;  C_arterial ≈ 1.5 mL/mmHg/kg, ↓ with age (stiffening)
SBP = MAP + (2/3)·PP ;  DBP = MAP − (1/3)·PP
```

Drug-induced vasodilation/vasoconstriction act on SVR (antihypertensives −15…−30%; vasopressors +50…+200%; septic shock collapse). Hypertension comorbidity sets baseline SVR +20%.

**Baroreceptor reflex:** first-order HR controller on MAP deviation with gain ≈ −4 bpm per mmHg below baseline (saturating ±40 bpm), τ ≈ 1 min (smoothed neural+humoral); β-blockers halve the gain; hypovolemia shifts the operating point.

**Respiratory rate:** metabolic-acidosis compensation — `RR = 14 + 1.2 × max(0, 25 − bicarbonate)` (Kussmaul breathing), plus fever drive +2 breaths/°C, opioid depression ×0.5.

**Temperature:** `dT_core/dt = (T_setpoint − T_core)/τ`, τ ≈ 20 min;
`T_setpoint = 37 + min(2.5, 0.02 × max(0, CRP − 10))` (fever from inflammation) + drug hyperthermia terms (serotonergic/NMS syndromes: +1.5…+2 °C acute) − shock hypothermia.

**SpO₂:** `PaO2 = 96 − 0.30 × age − lung_impairment` (pack-years, bleomycin/amiodarone pneumonitis);
`SaO2 = PaO2^n / (PaO2^n + P50^n)`, P50 = 26.8 mmHg, n = 2.7 (Severinghaus dissociation curve).

**Weight:** `dW/dt = fluid_balance/1000 + catabolic_term`;
heart-failure retention up to +1.0 kg/day (decompensated), diuretics −1.0 kg/day; cachexia −0.15 kg/day when CRP-driven tumor burden high (Fearon 2011).

All variables clamped to survivable ranges; crossing critical thresholds (MAP < 60, SpO₂ < 88, temp ≥ 40) raises `clinical_events` entries.

---

## 8 — Module 5: `human/disease_progression.py` — Disease Staging & Dynamics

### 8.1 — Enum

```python
class DiseaseStage(Enum):
    PRECLINICAL = "preclinical"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
```

### 8.2 — Clinical Staging Rubrics

| Disease family | Rubric | Mapping to `DiseaseStage` | Source |
|---|---|---|---|
| CKD | eGFR: G1 ≥90 · G2 60–89 · G3a 45–59 · G3b 30–44 · G4 15–29 · G5 <15 | G1–G2 → MILD · G3 → MODERATE · G4 → SEVERE · G5 → CRITICAL | KDIGO 2013 |
| Cirrhosis | Child-Pugh (bili, albumin, INR, ascites, encephalopathy): A 5–6 · B 7–9 · C 10–15 | A → MODERATE · B → SEVERE · C → CRITICAL | Pugh 1973 |
| Heart failure | NYHA I–IV | I–II → MILD/MODERATE · III → SEVERE · IV → CRITICAL | NYHA 1994 |
| Cancer | tumor-burden fraction from biomarkers (LDH, lesion-proxy flux) | quartiles → four stages | RECIST-inspired |
| Diabetes | HbA1c bands (<7 / 7–9 / >9 with complications) | MILD → SEVERE | ADA Standards 2026 |

Initial stage comes from user-supplied `onset_age`, `progression_stage`, and `clinical_staging`; thereafter staging is recomputed hourly from labs.

### 8.3 — Dynamic Severity

Continuous severity `s ∈ [0,1]` behind the discrete stage:

```
ds/dt = ρ · s · (1 − s/κ)          # logistic progression (ρ = progression rate)
      − ε · drug_strength(t)       # treatment kill/response (exponential)
      + σ · relapse_pressure(t)    # rebound after discontinuation
```

- `drug_strength` = aggregate PD strength from doc/27 pharmacodynamics targeting this disease.
- Resistance: optional `ε(t) = ε0 · exp(−k_res · cumulative_exposure)`.
- **Treatment can halt or reverse** progression (ds/dt < 0 while on effective therapy); **discontinuation causes rebound** with disease-specific latency (e.g., corticosteroid flare within days; statin lipid drift back over weeks).

### 8.4 — Interface

```python
class DiseaseProgressionModel:
    def __init__(self, disease: DiseaseState, rubric: str): ...
    def stage(self, labs: ClinicalLabs, vitals: VitalSigns) -> DiseaseStage: ...
    def step(self, dt_h: float, drug_effects: dict, labs: ClinicalLabs) -> DiseaseStage:
        """Advance severity, re-stage, log transitions to clinical_events."""
    def relapse_probability(self) -> float: ...
```

---

## 9 — Module 6: `human/ddi.py` — Drug-Drug Interactions

### 9.1 — Data Structures

```python
Mechanism = Literal["cyp_inhibition", "cyp_induction",
                    "pgp_inhibition", "pgp_induction",
                    "transporter_oatp", "additive_qt", "additive_nephro",
                    "additive_serotonin", "additive_myopathy"]

@dataclass
class DDIRule:
    precipitant_drug: str        # the affecting drug ("clarithromycin")
    object_drug: str | None      # None => any substrate of the enzyme
    mechanism: Mechanism
    enzyme: str                  # "CYP3A4", "CYP2C9", "P-gp", "OATP1B1", ""
    fold_change: float           # AUC ratio for inhibition/induction
    onset_h: float               # ramp time to full effect (inducers: days)
    offset_h: float              # decay after precipitant stops
    evidence_level: str          # "strong" | "moderate" | "weak"

class DDIModel:
    def __init__(self, rules_db: list[DDIRule], genotype: GenotypeProfile | None): ...
    def effective_clearances(self, drugs: list[Drug], t_h: float,
                             physiology: HumanPhysiology) -> dict[str, dict[str, float]]:
        """{drug: {enzyme: CL_multiplier}} combining genotype status,
        inhibitor/inducer ramps (onset/offset exponentials), and
        mechanism-specific handling. Multiplicative across rules."""
    def additive_toxicity_scores(self, drugs: list[Drug], t_h: float) -> dict[str, float]:
        """QT-ms sum, nephrotoxicity product, serotonin burden, myopathy burden."""
```

### 9.2 — Built-In Rule Database (curated classics)

| Object drug | Precipitant | Mechanism | Observed effect | Source |
|---|---|---|---|---|
| Midazolam | Ketoconazole | CYP3A4 inhib | AUC ↑ ~15× | Olkkola 1994 |
| Midazolam | Erythromycin | CYP3A4 inhib | AUC ↑ 3.4–4× | Olkkola 1993 |
| Warfarin | Fluconazole | CYP2C9 inhib | INR ↑; dose ↓ ~50% | Black 1996 |
| Metoprolol | Quinidine | CYP2D6 inhib | AUC ×2–3; EM→phenocopy PM | Kirchheiner 2005 |
| Simvastatin | Clarithromycin | CYP3A4 inhib | AUC ↑ ~10× → myopathy/rhabdo risk | Jacobson 2005 |
| Digoxin | Verapamil / Amiodarone | P-gp inhib | digoxin levels ↑ 60–100% | Fenichel 2000 |
| Cyclosporine | Rifampin | CYP3A4 + P-gp induction | AUC ↓ ~70% | Hebert 1992 |
| Any CYP3A4 substrate | Rifampin | CYP3A4 induction (onset ~1 wk) | CL ↑ up to 8–10× | Niemi 2003 |
| — | NSAID + ACEi + diuretic ("triple whammy") | additive_nephro | AKI OR 1.31 | Lapi 2013, BMJ |
| QT-prolongers combo (macrolides, fluoroquinolones, ondansetron, antipsychotics) | pairwise | additive_qt | QTc thresholds 480/500 ms → torsades event | Roden 2004 |

Induction ramps use enzyme-synthesis kinetics: onset τ ≈ 5–7 days (rifampin), offset t½ ≈ 26–48 h; inhibitors act within distribution half-lives (hours).

Genotype interplay: a CYP2D6 PM already at the floor gains little from quinidine; a UM given quinidine drops to apparent-IM — handled naturally because `effective_clearances` multiplies genotype factor × interaction factor.

---

## 10 — Module 7: `human/recovery.py` — Post-Treatment Recovery

### 10.1 — Design

Activated automatically once `t > duration_days` (last scheduled dose) plus one washout interval (≥ 5 × terminal t½). Models the **post-medication window** until `observation_days`.

### 10.2 — Sub-models

**Biomarker relaxation.** Each lab relaxes to its trait/genotype-defined baseline:

```
dB/dt = −ln(2) / t½_disease · (B(t) − B_baseline)
```

Disease-specific half-lives (ordered): ALT 47 h < CRP 19 h < creatinine 1–3 d < WBC/plts 2–4 wk < LDL 4 d–3 wk (statin off-drift) < Hb 40 d (RBC lifespan) < HbA1c ~35 d lag < bone/mineral markers months.

**Organ functional recovery.**

| Organ | Injury mode | Recovery course | Source |
|---|---|---|---|
| Liver | toxic/necrotic loss | functional reserve t½ ≈ 3 wk; mass restoration ~8–12 wk after major loss | Michalopoulos 2010 |
| Kidney | ATN (cisplatin, ischemia) | weeks–months; incomplete if severe (risk factor for CKD) | KDIGO AKI 2012 |
| Bone marrow | myeloablation | counts recover 2–4 wk post-last-dose | Friberg 2003 |
| Cochlea | cisplatin ototoxicity | **permanent** — flagged irreversible | Knight 2005 |
| Cognition | "chemo brain" | partial recovery months; sequelae flag | Janelsins 2014 |

**Rebound phenomena.**

| Withdrawal | Timeline modeled | Source |
|---|---|---|
| Corticosteroid (HPA-axis suppression, >3 wk therapy) | rebound flare within days; adrenal recovery weeks–months ∝ duration | Axelrod 2003 |
| Opioid (tolerance/hyperalgesia) | withdrawal peak ~72 h, resolves 7–14 d | Kosten 2004 |
| β-blocker | rebound tachycardia/hypertension 24–72 h | Psaty 1990 |

**Relapse vs remission.** Daily hazard `h(t) = h0 · exp(−λ · maintenance_exposure)`; `p_relapse(t) = 1 − exp(−∫h)`; remission declared after severity < threshold held for the disease-specific dwell time. Outcome recorded in final `clinical_events` summary.

### 10.3 — Interface

```python
class RecoveryModel:
    def __init__(self, snapshot: "EndOfTreatmentSnapshot", traits: ExternalTraits): ...
    def step(self, dt_h: float) -> dict[str, float]:
        """Advance one hour of washout/recovery; returns current labs+vitals deltas."""
    def recovery_trajectory(self) -> dict[str, list[float]]: ...
    def permanent_sequelae(self) -> list[str]: ...
    def outcome(self) -> str:   # "remission" | "relapse" | "ongoing_recovery" | "sequelae_only"
```

---

## 11 — Module 8: `human/virtual_patient.py` — Unified Facade

### 11.1 — Inputs

```python
@dataclass
class VirtualPatientInput:
    genome_vcf_text: str | None = None       # complete genome (preferred form)
    genotype: GenotypeProfile | None = None  # or pre-built profile
    traits: ExternalTraits = field(default_factory=ExternalTraits)
    diseases: list[DiseaseState] = field(default_factory=list)   # doc/27 entities
    drugs: list[Drug] = field(default_factory=list)              # doc/27 entities
    observation_days: float = 180.0          # TOTAL window incl. post-treatment

    def validate(self) -> list[str]: ...     # duration <= observation, sex/pregnancy consistency
```

### 11.2 — Hourly Integration Loop

```python
class VirtualPatient:
    """Combines GenotypeProfile + ExternalTraits + DiseaseState(s) + Drug regimen."""

    def __init__(self, inp: VirtualPatientInput):
        # 1. genotype: VCF -> GenotypeProfile (or accept pre-built)
        # 2. physiology: PhenotypeCalculator(genotype, traits).apply(default_physiology())
        # 3. engines: PBPK per drug, PD per drug, DDIModel, dFBA batch (doc/27)
        # 4. monitors: ClinicalLabModel, VitalsModel, DiseaseProgressionModels
        ...

    def run(self, total_days: float | None = None) -> VirtualPatientResult:
        ...
```

Per-hour pipeline (order matters — DDI must modify clearance *before* PBPK):

```
for hour in range(total_hours):
  1. DOSING      if hour % interval == 0 and hour < duration_days*24: administer dose
  2. DDI         effective_clearances(drugs, t) — genotype × inhibitor/inducer ramps
  3. PBPK        60 sub-steps × 1 min per drug with DDI-modified clearances
  4. PD          target-tissue C(t) → Hill strength → flux-bound corrections
  5. dFBA        one 1-h step on disease-perturbed GEM with corrected bounds
  6. POOLS       metabolite pools integrate flux solution
  7. LABS        ClinicalLabModel.compute(physiology, disease, drug_effects, pools, t)
  8. VITALS      VitalsModel.step(1 h, labs, drug_effects, diseases)
  9. PROGRESS    each DiseaseProgressionModel.step(1 h, drug_effects, labs)
                 → stage transitions logged; relapse pressure tracked
 10. EVENTS      toxicity checks: Hy's law, AKI (eGFR ↓ ≥25%), QTc > threshold,
                 myopathy (CK-proxy), neutropenia grade (ANC < 0.5), shock
 11. RECOVERY    if past treatment end + washout: RecoveryModel.step() replaces 1–5
 12. RECORD      append snapshot (hourly resolution)
post-loop:      RecoveryModel.finalize(); attach trajectory, outcome, sequelae
```

Feedback loops preserved: labs ↔ vitals ↔ progression (e.g., worsening CKD lowers drug clearance → higher exposure → more nephrotoxicity → faster progression). Loop gains clamped; monotonic guards prevent oscillation (§21).

### 11.3 — Result

See §12 for the exact schema. Every requested observable is present at hourly resolution for the **entire observation period including post-treatment**.

---

## 12 — Output Schema: `VirtualPatientResult`

```python
@dataclass
class VirtualPatientResult:
    time_h: list[float]                                   # 0..observation_days*24, step 1 h
    vitals: dict[str, list[float]]                        # systolic_bp, diastolic_bp, heart_rate,
                                                          # respiratory_rate, temperature_c, spo2, weight_kg
    labs: dict[str, list[float]]                          # ALT, AST, ALP, GGT, bilirubin_total, bilirubin_direct,
                                                          # albumin, INR, creatinine, BUN, eGFR, cystatin_C,
                                                          # WBC, RBC, hemoglobin, hematocrit, platelets, MCV, MCH,
                                                          # glucose, HbA1c, sodium, potassium, chloride, bicarbonate,
                                                          # calcium, phosphate, total_cholesterol, LDL, HDL,
                                                          # triglycerides, CRP, ESR
    drug_concentrations: dict[str, dict[str, list[float]]]  # {drug_name: {organ: [C(t)]}}
    metabolite_pools: dict[str, list[float]]              # {pool_id: [concentration(t)]}
    flux_history: dict[str, list[float]]                  # {reaction_id: [flux(t)]}
    biomarker_history: dict[str, list[float]]             # disease-specific biomarkers (Gb3, Phe, ...)
    disease_stage_over_time: list[str]                    # per-hour stage string per primary disease
    clinical_events: list[dict]                           # {time_h, kind, severity, detail}
                                                          # toxicity, organ injury, recovery milestones
    recovery_trajectory: dict[str, list[float]]           # post-treatment key biomarkers vs baseline

    # Metadata
    patient_summary: dict                                 # genotype calls, traits, staging at t=0
    warnings: list[str]                                   # defaulted genotypes, unsupported combos
    disclaimer: str                                       # "educational simulation — not medical advice"
```

Consistency invariants enforced at construction: all lists share `len(time_h)`; `drug_concentrations[d][o]` likewise; `recovery_trajectory` covers `[washout_end_h, len(time_h))`.

---

## 13 — End-to-End Data Flow

```
 VCF text ──create_genotype_from_vcf──▶ GenotypeProfile ──┐
                                                          ├─▶ PhenotypeCalculator
 ExternalTraits ──────────────────────────────────────────┘         │
                                                                    ▼
                                                     HumanPhysiology (scaled)
                                                                    │
 DiseaseState[] ──apply_disease_state──▶ perturbed GEM ──┐          │
 Drug[] ───────────────────────────────────────────────┐ │          │
                                                       ▼ ▼          ▼
                              ┌──────────── DDIModel ◀─ hourly loop ◀┘
                              │                 │
                              │                 ▼
                              │   PBPK (CL_eff) ─▶ PD (Hill) ─▶ dFBA ─▶ pools
                              │                                        │
                              │                                        ▼
                              │                    ClinicalLabModel ◀──┘
                              │                         │
                              │                         ▼
                              │                    VitalsModel
                              │                         │
                              │                         ▼
                              └──▶ DiseaseProgressionModel(s) ──▶ clinical_events
                                                        │
                                     t > treatment_end + washout
                                                        ▼
                                                  RecoveryModel ──▶ recovery_trajectory
                                                                        │
                                                                        ▼
                                                      VirtualPatientResult (everything)
```

---

## 14 — Helix DSL Integration

### 14.1 — New Keywords

| Keyword | Purpose | Example |
|---|---|---|
| `#genome` | Inline or file-reference genotype | `#genome vcf="patient_001.vcf"` |
| `#traits` | External traits | `#traits age=68 sex=male weight=82 height=175 smoking_pack_years=30` |
| `#regimen` | Dosing schedule wrapper | `#regimen observation=180d` |
| `#sim kind=virtual_patient` | Run the whole-person simulation | `#sim kind=virtual_patient` |

Existing `#human` (doc/27) remains valid and maps into `ExternalTraits`; `#disease`, `#drug`, `#pk`, `#pd` are reused unchanged. Parser extension via the established `sim_extensions` dispatch:

```python
"genome": _parse_genome_block,
"traits": _parse_traits_block,
"regimen": _parse_regimen_block,
```

Backend registration in `sim_runtime.py::_SIM_BACKENDS`: `"virtual_patient": _run_virtual_patient_simulation`.

### 14.2 — Example Program: `examples/59_virtual_patient_ckd_warfarin_ddi.helix`

Scenario: 68-year-old man, 82 kg, 175 cm, former smoker (30 pack-years), CKD stage 3a, atrial fibrillation on warfarin; a 7-day fluconazole course is co-administered. He carries CYP2C9 *1/*3 (intermediate metabolizer). Expected behavior: genotype already halves warfarin dose requirement; fluconazole further cuts CYP2C9 clearance → INR climbs above 3.5 → bleeding-risk clinical event → recovery phase normalizes INR after washout.

```helix
#sim kind=virtual_patient observation=180d dt=1h

#genome {
  vcf: "examples/data/patient_001.vcf"   # full-genome VCF; CYP2C9 *1/*3 among calls
}

#traits {
  age: 68 years
  sex: male
  weight: 82 kg
  height: 175 cm
  ethnicity: european
  smoking_pack_years: 30
  currently_smoking: false
  alcohol_drinks_per_week: 4
  exercise_level: light
  comorbidities: ["atrial_fibrillation", "ckd_stage_3a"]
}

#disease {
  name: "chronic kidney disease stage 3a"
  category: organ_failure
  progression_stage: moderate
  onset_age: 62 years
  clinical_staging: kdigo_egfr
  severity: 0.45
}

#drug {
  name: "warfarin"
  type: small_molecule
  smiles: "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
  mw: 308.3 Da
  dose: 5 mg
  route: oral
  interval: 24h
  duration: 90d
  pk { half_life: 36 h  vd: 0.14 L/kg  protein_binding: 0.99 }
  pd { target: VKORC1  effect: inhibition  ic50: 1.8 uM  hill_coefficient: 1.0 }
}

#drug {
  name: "fluconazole"
  type: small_molecule
  smiles: "OC(Cn1cncn1)C(F)(F)c1ccc(F)cc1F"
  mw: 306.3 Da
  dose: 200 mg
  route: oral
  interval: 24h
  duration: 7d
  pk { half_life: 32 h  vd: 0.7 L/kg  protein_binding: 0.11 }
}

#output {
  time_resolution: 1h
  tracks: [INR, egfr, creatinine, warfarin_plasma, disease_stage, vitals]
  format: csv
}
```

Further shipped examples: `examples/60_virtual_patient_pregnancy_labetalol.helix` (pregnancy hemodynamics + clearance changes) and `examples/61_virtual_patient_chemo_recovery.helix` (cisplatin → nephrotoxicity/ototoxicity, marrow nadir, post-treatment recovery with permanent sequelae).

---

## 15 — Parameter Validation Benchmarks

| Scenario | Metric | Simulation must produce | Literature anchor |
|---|---|---|---|
| CYP2D6 *4/*4 vs *1/*1, metoprolol 100 mg oral | AUC ratio | ≥ 2× higher in PM | Kirchheiner 2005 |
| CYP2C9 *1/*3, warfarin | stable dose requirement | ~50–65% of *1/*1 dose | Aithal 1999; Johnson 2017 |
| Fluconazole co-administration | INR trajectory | rise ≥ 1.0 above personal baseline within 5 d | Black 1996 |
| Simvastatin + clarithromycin | simvastatin AUC ratio | ≥ 5× | Jacobson 2005 |
| Age 80 vs 30, same weight/sex | eGFR | 80-yo within ±20% of CKD-EPI cohort means (~70 vs ~105) | Inker 2021 |
| Pregnancy T3 | CO, GFR, plasma volume | +45% ± 10, +50% ± 10, +45% ± 10 | Anderson 2005 |
| Smoker vs non-smoker, theophylline/caffeine | clearance | ×1.5–2.0 in smoker | Kroon 2007 |
| Acetaminophen overdose (virtual) | ALT peak | > 1000 U/L; Hy's-law event raised | Temple 2006 |
| Cisplatin cycles | WBC nadir timing | day 7–10 post-dose; eGFR decline ∝ cumulative dose | Friberg 2003; Kelland 2007 |
| High-intensity statin 8 wk | LDL reduction | 50–55% of baseline | Grundy 2018 |
| Sepsis-like inflammatory drive | CRP doubling | ~every 8 h early phase; t½ 19 h on source control | Pepys 2003 |
| 90-day post-treatment recovery | lab ordering | normalization order matches half-life ordering (ALT before HbA1c before bone markers) | §10.2 table |

---

## 16 — Test Plan

### 16.1 — Unit Tests

| Test File | Coverage |
|---|---|
| `tests/test_human_genotype.py` | VCF parsing (multi-allelic, DP filter, GT mapping), star-allele calling, activity scores, category boundaries, defaulted-gene behavior |
| `tests/test_human_phenotype.py` | BMI derivation, organ-volume scalings, age curves, smoking/alcohol/pregnancy multipliers, immutability of input physiology |
| `tests/test_human_clinical_output.py` | Baselines by sex/age, ALT/AST injury-decay kinetics, CKD-EPI computation, CBC nadir model, HbA1c lag, electrolyte drug effects |
| `tests/test_human_vitals.py` | MAP=CO×SVR identity, baroreflex gain/saturation, fever drive from CRP, SpO₂ curve, weight balance |
| `tests/test_human_disease_progression.py` | KDIGO/Child-Pugh/NYHA staging boundaries, logistic progression, reversal on treatment, rebound on discontinuation |
| `tests/test_human_ddi.py` | Rule lookup, multiplicative combination, onset/offset ramps, genotype interplay, additive toxicity scores |
| `tests/test_human_recovery.py` | Relaxation half-lives, organ recovery courses, rebound timelines, permanence flags, relapse hazard math |
| `tests/test_human_virtual_patient.py` | Facade construction, hourly loop order, schema invariants (list lengths), event logging, determinism (seeded) |
| `tests/test_human_dsl_v28.py` | Parsing of `#genome`, `#traits`, `#regimen`, `#sim kind=virtual_patient`; `#human` back-compat |

### 16.2 — Integration Tests

| Test | Description |
|---|---|
| `test_vp_ckd_warfarin_fluconazole` | Full example 59: genotype-modulated INR + DDI spike + washout recovery |
| `test_vp_pm_vs_em_auc` | Same regimen on two genomes → expected AUC separation |
| `test_vp_pregnancy_clearance` | Renal-cleared drug needs higher dose when pregnant |
| `test_vp_chemo_full_cycle` | Nadir, infection-flag CRP rise, G-CSF-free recovery curve |
| `test_vp_triple_whammy` | NSAID+ACEi+diuretic → AKI event, eGFR drop, partial recovery |
| `test_vp_post_treatment_relapse` | Stop effective drug → staged relapse within horizon |
| `test_vp_180_day_performance` | 180-day run completes < 60 s with 3 drugs, 2 diseases |

---

## 17 — File Impact Summary

### New Files

| File | Lines (est.) | Purpose |
|---|---|---|
| `src/helixlang/human/genotype.py` | 450 | Variant, GenotypeProfile, star-allele tables, VCF parser |
| `src/helixlang/human/phenotype.py` | 400 | ExternalTraits, PhenotypeCalculator, scaling laws |
| `src/helixlang/human/clinical_output.py` | 550 | ClinicalLabs, ClinicalLabModel panels |
| `src/helixlang/human/vitals.py` | 350 | VitalSigns, VitalsModel |
| `src/helixlang/human/disease_progression.py` | 400 | DiseaseStage, staging rubrics, severity ODE |
| `src/helixlang/human/ddi.py` | 400 | DDIRule, DDIModel, rules database |
| `src/helixlang/human/recovery.py` | 350 | RecoveryModel, rebound, sequelae, relapse |
| `src/helixlang/human/virtual_patient.py` | 550 | VirtualPatient facade, VirtualPatientResult |
| `src/helixlang/human/data/pgx.json` | 250 | PharmVar allele-function table |
| `src/helixlang/human/data/ddi_rules.json` | 300 | Curated interaction rules |
| `src/helixlang/human/data/lab_reference_ranges.json` | 120 | Sex/age-specific baselines and ULNs |
| `tests/test_human_genotype.py` | 300 | Genotype tests |
| `tests/test_human_phenotype.py` | 250 | Trait-scaling tests |
| `tests/test_human_clinical_output.py` | 300 | Lab model tests |
| `tests/test_human_vitals.py` | 200 | Vitals tests |
| `tests/test_human_disease_progression.py` | 250 | Staging/progression tests |
| `tests/test_human_ddi.py` | 250 | DDI tests |
| `tests/test_human_recovery.py` | 250 | Recovery tests |
| `tests/test_human_virtual_patient.py` | 400 | Facade integration tests |
| `tests/test_human_dsl_v28.py` | 150 | DSL tests |
| `examples/59_virtual_patient_ckd_warfarin_ddi.helix` | 90 | Flagship scenario |
| `examples/60_virtual_patient_pregnancy_labetalol.helix` | 80 | Pregnancy scenario |
| `examples/61_virtual_patient_chemo_recovery.helix` | 100 | Chemo + recovery scenario |

### Modified Files

| File | Change |
|---|---|
| `src/helixlang/core/parser.py` | Add `genome`, `traits`, `regimen` to annotation dispatch |
| `src/helixlang/sim_runtime/` | Register `"virtual_patient": _run_virtual_patient_simulation` |
| `src/helixlang/human/__init__.py` | Re-export new classes |
| `src/helixlang/human/simulation.py` | Expose internals needed by facade (PBPK engines, bound override hooks) |
| `README.md` / `README_PYPI.md` | Add virtual patient highlight |

---

## 18 — Dependencies

| Dependency | Required? | Purpose | Already Installed? |
|---|---|---|---|
| `numpy` | Yes | Arrays, integration helpers | ✓ (`[fast]` extra) |
| `scipy` | Yes | ODE stiff solver for coupled loops | ✓ (`[fast]` extra) |
| `rdkit` | Optional | SMILES property refinement (doc/27) | ✓ Installed (2026.03.5) |
| `cobra` / `cobrapy` | Optional | Human GEM loading (doc/27 path) | ✓ (`[bio]` extra) |
| stdlib only | Yes | VCF parser (no pysam dependency — plain-text VCF 4.x) | ✓ |

No new hard dependencies. VCF parsing is deliberately dependency-free; binary formats (BCF/gVCF) are out of scope (see §21).

---

## 19 — Implementation Phases

### Phase 1: Genotype Foundation (Week 1)
- [ ] `genotype.py`: `Variant`, `GenotypeProfile`, `create_genotype_from_vcf`
- [ ] `data/pgx.json` star-allele/function table (CYP2D6/2C19/2C9/3A4/3A5/1A2/2E1 + TPMT, VKORC1)
- [ ] Metabolizer category assignment + `metabolism_rate_modifier`
- [ ] Unit tests (incl. malformed-VCF handling)

### Phase 2: Trait Scaling (Week 1–2)
- [ ] `phenotype.py`: `ExternalTraits`, BMI derivation, `PhenotypeCalculator.apply`
- [ ] Age/BMI/pregnancy/smoking/alcohol/exercise/ethnicity multipliers with citations
- [ ] Unit tests + golden-value tests against §15 anchors

### Phase 3: Clinical Observables (Week 2–3)
- [ ] `clinical_output.py`: `ClinicalLabs`, panel sub-models (hepatic, renal, CBC, metabolic, lipid, inflammatory)
- [ ] `vitals.py`: hemodynamics, baroreflex, thermoregulation, SpO₂, weight
- [ ] Unit tests with known kinetic anchors (ALT 47 h, CRP 19 h, HbA1c lag)

### Phase 4: Disease Dynamics + DDI (Week 3)
- [ ] `disease_progression.py`: enum, rubric stagers, severity ODE, relapse pressure
- [ ] `ddi.py`: rule schema, rules JSON, ramp math, additive toxicity channels
- [ ] Unit + interaction tests

### Phase 5: Recovery (Week 4)
- [ ] `recovery.py`: washout detection, relaxation courses, organ regeneration, rebound library, permanence flags, relapse/remission
- [ ] Unit tests

### Phase 6: Facade Integration (Week 4–5)
- [ ] `virtual_patient.py`: input validation, wiring, hourly loop, event system, result assembly
- [ ] Schema invariants + performance pass (vectorize hot loops)
- [ ] Full integration tests (§16.2)

### Phase 7: DSL + Docs (Week 5)
- [ ] Parser/runtime registration; `#genome`/`#traits`/`#regimen`
- [ ] Examples 59–61; README updates; benchmark suite wired to §15 table

---

## 20 — Acceptance Criteria

1. **Genome ingestion:** `create_genotype_from_vcf` correctly parses a VCF 4.2 sample containing CYP2D6 *1/*4, CYP2C19 *1/*17, CYP2C9 *1/*3 and reports exact activity scores/categories.
2. **Genotype → exposure:** a CYP2D6 *4/*4 genome yields ≥ 2× metoprolol AUC versus *1/*1 in identical traits/regimen (Kirchheiner 2005 anchor).
3. **Trait realism:** age-80 default traits reproduce cohort-mean eGFR within ±20% (CKD-EPI 2021) and CO within published aging range; pregnancy reproduces CO/GFR/plasma-volume increases within ±10 percentage points.
4. **Labs:** all 34 `ClinicalLabs` fields populate hourly; eGFR matches CKD-EPI 2021 given the simulated creatinine; acetaminophen-toxicity scenario crosses Hy's-law and logs the event.
5. **Vitals:** MAP equals CO×SVR/80 within numerical tolerance; baroreflex opposes MAP deviations with documented gain; fever tracks CRP drive.
6. **Progression:** eGFR-driven CKD restaging matches KDIGO bands; effective treatment reverses severity slope sign; discontinuation triggers documented rebound.
7. **DDI:** fluconazole-on-warfarin raises INR ≥ 1.0 within 5 simulated days; clarithromycin-on-simvastatin yields AUC ratio ≥ 5×; triple-whammy triggers an AKI event with eGFR drop ≥ 25%.
8. **Recovery:** after washout, biomarkers normalize in the correct half-life ordering; corticosteroid withdrawal produces rebound; cisplatin ototoxicity is flagged permanent; relapse/remission probabilities are reported.
9. **Completeness:** `VirtualPatientResult` contains every field of §12 with length-consistent series spanning treatment **and** post-treatment windows; `clinical_events` records at least one milestone in each phase (treatment, toxicity-checkable, recovery) where applicable.
10. **Performance:** 180-day, 1-h-resolution, 3-drug/2-disease run completes in < 60 s on CI hardware.
11. **Determinism:** identical inputs → bit-identical results (fixed integration order, seeded stochastic terms).
12. **DSL:** examples 59–61 parse and execute end-to-end via `helix run`.
13. **Quality gates:** `ruff` and `mypy` clean on all new modules; full test suite green.
14. **Safety framing:** result carries the standing disclaimer; no module claims clinical validity outside simulation scope.

---

## 21 — Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Star-allele calling ambiguity (CNV, structural variants, gVCF) | High | Conservative fallback to population-default EM with explicit `confidence: "defaulted"` + warning in result; CNV accepted via explicit meta syntax |
| Feedback-loop instability (labs ↔ vitals ↔ progression coupling) | High | Clamped gains, physiologic bounds on all states, monotonic-transition guards, stability stress-tests with extreme inputs |
| Performance blow-up (hourly × 180 d × submodels × LP solves) | Medium | Reuse LP basis when bounds unchanged; vectorized lab/vitals updates; optional coarse mode (6-h clinical steps, 1-min PBPK only near doses) |
| Parameter explosion / citation drift | Medium | Single-source JSON data files with per-entry `source` fields; §15 benchmark table wired into CI regression |
| Ethnicity adjustments perceived as essentialist | Medium | Framed strictly as population-frequency statistics (PharmGKB/PharmVar); race-free CKD-EPI 2021; all priors overridable by explicit genotype data |
| Overshoot into pseudo-clinical authority | High (reputational) | Persistent machine-readable disclaimer in results; docs state educational-simulation scope; no dosing recommendations emitted |
| Binary genome formats (BCF/gVCF) unsupported | Low | Documented scope limit; users convert externally (`bcftools view`) |
| DDI database coverage gaps | Medium | Rule schema extensible; unknown combinations fall back to genotype-only PK with warning; community-extensible JSON |
| Long-horizon drift (numerical integration error) | Low | Adaptive sub-stepping near discontinuities (dose events), energy/conservation spot-checks in tests |

---

## 22 — Literature References

1. Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics.* 5th ed. Wolters Kluwer; 2020.
2. Guyton AC, Hall JE. *Textbook of Medical Physiology.* 13th ed. Elsevier; 2016.
3. Zanger UM, Schwab M. Cytochrome P450 enzymes in drug metabolism: regulation of expression, activities, and impact of genetic variation. *Pharmacol Ther* 2013;138:103-141.
4. Gaedigk A et al. The Pharmacogene Variation (PharmVar) Consortium: incorporation of the Human Cytochrome P450 (CYP) Allele Nomenclature Database. *Clin Pharmacol Ther* 2018;103:399-401.
5. Caudle KE et al. Clinical Pharmacogenetics Implementation Consortium guideline for CYP2D6 genotype and codeine therapy. *Clin Pharmacol Ther* 2017;101:46-48.
6. Johnson JA et al. Clinical Pharmacogenetics Implementation Consortium guideline for pharmacogenetics-guided warfarin dosing: 2017 update. *Clin Pharmacol Ther* 2017;102:397-404.
7. Scott SA et al. Clinical Pharmacogenetics Implementation Consortium guideline for CYP2C19 genotype and clopidogrel therapy: 2013 update. *Clin Pharmacol Ther* 2013;94:317-323.
8. Aithal GP et al. Association between polymorphisms in CYP2C9 and the therapeutic dose of warfarin. *Lancet* 1999;354:717-719.
9. Kirchheiner J, Brockmöller J. Clinical consequences of CYP2C9 polymorphisms. *Clin Pharmacol Ther* 2005;77:1-16.
10. Inker LA et al. New creatinine- and cystatin C–based equations to estimate GFR without race. *N Engl J Med* 2021;385:1737-1749.
11. KDIGO. Clinical practice guideline for the evaluation and management of chronic kidney disease. *Kidney Int Suppl* 2013;3:1-150.
12. KDIGO. Clinical practice guideline for acute kidney injury. *Kidney Int Suppl* 2012;2:1-138.
13. Anderson GD. Pregnancy-induced changes in pharmacokinetics: a mechanistic-based approach. *Clin Pharmacokinet* 2005;44:989-1008.
14. Kroon LA. Drug interactions with smoking. *Am J Health Syst Pharm* 2007;64:1917-1921.
15. Novak RF, Woodcroft KJ. Alcohol-inducible cytochrome P450 2E1. *Arch Toxicol* 2001;75:109-121.
16. Zeeh J, Platt D. The aging liver: structural and functional changes and their consequences for drug treatment in old age. *Gerontology* 2002;48:121-127.
17. Mosteller RD. Simplified calculation of body-surface area. *N Engl J Med* 1987;317:1098.
18. Vauthey JN et al. Standardized liver volume in the SWEN tumor database. *J Am Coll Surg* 1999;188:441-446.
19. Deurenberg P et al. Body mass index as a measure of body fatness: age- and sex-specific prediction formulas. *Br J Nutr* 1991;65:105-114.
20. Janssen I et al. Skeletal muscle mass and distribution in 468 men and women aged 18–88 yr. *J Appl Physiol* 2000;89:81-88.
21. Prati D et al. Updated definitions of healthy ranges for serum alanine aminotransferase levels. *Ann Intern Med* 2002;137:1-10.
22. Kim WR et al. Kinetics of alanine aminotransferase in clinical practice. *Hepatology* 2008;47:1363-1369.
23. Temple R. Hy's laboratories: the FDA and the safety of drugs. *Pharmacoepidemiol Drug Saf* 2006;15:241-250.
24. Nathan DM et al. Translating the A1C assay into estimated average glucose values. *Diabetes Care* 2007;30:1473-1478.
25. Pepys MB, Hirschfield GM. C-reactive protein: a critical update. *J Clin Invest* 2003;111:1805-1812.
26. Friberg LE et al. Model of chemotherapy-induced myelosuppression. *J Clin Oncol* 2003;21:1671-1677 (semimechanistic model adapted).
27. Severinghaus JW. Simple, accurate equations for human blood O2 dissociation computations. *J Appl Physiol* 1979;46:599-602.
28. Roden DM. Drug-induced prolongation of the QT interval. *N Engl J Med* 2004;350:1013-1022.
29. Olkkola KT et al. Dose-related interaction between oral midazolam and ketoconazole. *Clin Pharmacol Ther* 1994;55:398-402.
30. Black DJ et al. Warfarin–fluconazole interaction: pharmacokinetic and pharmacodynamic aspects. *Antimicrob Agents Chemother* 1996;40:1123-1128.
31. Lapi F et al. Concurrent use of diuretics, angiotensin converting enzyme inhibitors, and nonsteroidal anti-inflammatory drugs and risk of acute kidney injury. *BMJ* 2013;346:e8525.
32. Michalopoulos GK. Liver regeneration after partial hepatectomy: critical analysis of mechanistic dilemmas. *Am J Pathol* 2010;176:2-13.
33. Knight KR et al. Ototoxicity in children receiving platinum chemotherapy. *Lancet Oncol* 2005;6:449-458.
34. Fearon K et al. Understanding and managing cancer wasting. *Nat Rev Clin Oncol* 2011;8:229-239.
35. Grundy SM et al. 2018 AHA/ACC cholesterol guideline. *Circulation* 2019;139:e1082-e1143.
36. Pugh RNH et al. Transection of the oesophagus for bleeding oesophageal varices. *Br J Surg* 1973;60:646-649.
37. Fletcher C, Peto R. The natural history of chronic airflow obstruction. *Br Med J* 1977;1:1645-1648.
