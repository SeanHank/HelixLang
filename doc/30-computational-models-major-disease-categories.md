# 30 — Computational Models for Major Disease Categories: Research for the Virtual Patient System

> **Status:** RESEARCH DOCUMENT (design input for doc/28 virtual-patient disease modules)
> **Depends on:** doc/27 (pathology & drug simulation), doc/28 (virtual patient system)
> **Date:** 2026-08-24

---

## 0 — Scope and Method

This document surveys the most-cited, clinically validated computational models for eight major
disease categories and evaluates each for implementation inside the HelixLang virtual patient
system (doc/28 `human/disease_progression.py` and downstream). For every category we report:

1. The **best-validated model** (most cited / regulatory-recognized),
2. **State variables** with normal ranges,
3. **Progression equations** where available,
4. **Drug-intervention mechanism** (how therapy enters the math),
5. **Clinical validation evidence**,
6. **Implementation complexity** (ODE count, parameter availability, runtime cost).

Sources were gathered from primary literature (JCO, Nature Sci Rep, BMC Syst Biol, Clin Transl
Sci, Brain Multiphysics, PLoS Comput Biol, Hypertension, Sci Rep, JASN/NDT, CPT:PSP, Frontiers,
MDPI Mathematics) retrieved August 2026.

**Recommended architecture principle:** implement each category as a *reduced ODE "system
model"* (fast, runs hourly inside the virtual-patient loop) with an optional *high-fidelity
variant* (PDE / agent-based / QST) kept behind an interface flag. All drug effects enter through
the same convention: plasma/organ concentration C(t) from the doc/27 PBPK engine drives an
Emax/Hill pharmacodynamic term or target-mediated drug disposition (TMDD) binding.

---

## 1 — Cardiovascular Disease

### 1.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| Long-term BP regulation | Guyton renal-body-fluid model (modernized) | Guyton, Coleman & Granger 1972; Karaaslan et al. PLoS Comput Biol 2012 |
| Whole-cardiovascular hemodynamics | Modular calibrated CV+renal model | BioUML agent-based model, PMC8632703 (Front Physiol 2021) |
| Atherosclerosis (whole-patient) | LDL-exposure ODE plaque-burden model | Cholesterol/plaque computational model PMC10452179 (2023) |
| Atherosclerosis (high fidelity) | Multiphase PDE plaque model validated on 94 human coronaries | Nature Sci Rep 2020, PMC7562914; Myerscough group Bull Math Biol 2023 |
| Plaque growth law (image-linked) | Tang-style WSS/wall-stress growth law | Tang et al. 2008; carotid CFD applications 2022 |

### 1.2 State variables and normal ranges

| Variable | Symbol | Normal range | Notes |
|---|---|---|---|
| Mean arterial pressure | MAP | 70–105 mmHg | long-term setpoint = renal-function-curve intersection |
| Cardiac output | CO | 4.5–5.5 L/min | Guyton: venous return determines CO except in failure |
| Total peripheral resistance | TPR | ~17–22 PRU | MAP = CO × TPR |
| Extracellular fluid volume / blood volume | ECFV, BV | 14 L / 5 L (70 kg) | integral-control state of Guyton loop |
| Renin–angiotensin II, aldosterone, ADH | ANGII, ALDO, ADH | ng/ml-scale, pg/ml-scale | humoral actuators |
| Plasma LDL-C | LDL | <100 mg/dL optimal | input flux to intima |
| Plasma HDL-C | HDL | >40 (M) / >50 (F) mg/dL | reverse cholesterol transport efflux term |
| Intima oxLDL / foam-cell density | c_LDL, M_f | — | lesion drivers in PDE models |
| Plaque volume / stenosis % | V_plaque | 0%; stenosis <50% subclinical | outcome variable |
| Endothelial function (NO bioavailability) | E_NO | normalized 1.0 | impaired by oxLDL, smoking, hyperglycemia |
| LVEF, LVEDV/LVESV | EF | 55–70%; 120±20 / 50±10 mL | heart-failure state |
| BNP / NT-proBNP | — | <100 pg/mL | HF staging biomarker |

### 1.3 Key equations

**Guyton long-term BP (core insight):** arterial pressure is regulated by an integral-control
renal-body-fluid loop with effectively *infinite gain*:

```
d(ECFV)/dt = Na_intake + water_intake − UrineOutput(MAP, renal_function_curve, ALDO, ADH)
MAP_longterm  = solution(MAP ⇔ renal function curve)   # kidneys dominate long-term BP
```

Modern instantiations: QCP-2005, HumMod 3.0.4, BioGears, and the BioUML modular model (which
adds explicit LV pressure-volume loops and reproduces equilibrium states for uncomplicated
hypertension, hypertensive/non-hypertensive diastolic HF, LV hypertrophy, pulmonary hypertension).
*Caveat:* Kurtz et al. (Hypertension 2018) showed all Guyton-lineage models reproduce salt-loading
responses poorly — treat salt-sensitivity predictions as qualitative.

**Atherosclerosis, whole-patient ODE form (recommended default):**

```
dV_plaque/dt = k_dep · LDL_flux(E_no) · WSS_factor(geometry) − k_reg · Efflux(HDL) · [LDL < LDL_thresh]
LDL_flux     = permeability(E_no) · C_intima(LDL_plasma)
E_no         : decreases with oxLDL, smoking, hyperglycemia; restored by statins/RAS-blockade
```

Anchors: CIMT progression ≈ 0.01–0.02 mm/yr untreated; event risk tracks LDL-exposure years;
the 2023 cholesterol-modeling study reproduced that interventions lowering LDL-C slow plaque
growth, matching imaging cohorts.

**High-fidelity PDE variant:** two-phase (cell/fluid) intima model — oxLDL diffusion-reaction,
monocyte→macrophage chemotaxis, foam-cell lipid internalization, efferocytosis, necrotic-core
formation; coupled Navier-Stokes lumen flow. Validated qualitatively against 94 reconstructed
human coronary arteries (Sci Rep 2020). Use only in research mode; too expensive for hourly
whole-person stepping.

### 1.4 Drug intervention mechanisms

| Drug class | Model hook |
|---|---|
| Statins | ↓ hepatic LDL synthesis (HMGCR inhibition): LDL_plasma −30–55%; small pleiotropic ↑E_no |
| Ezetimibe / PCSK9 mAbs | ↓ intestinal absorption / ↑ LDL-receptor recycling: additional −15–25% / −60% |
| ACEi / ARBs | ↓ANGII → TPR ↓, aldosterone ↓ → ECFV ↓; vascular protection ↑E_no |
| Calcium-channel blockers | direct TPR ↓ |
| Beta-blockers | HR ↓, contractility ↓, renin ↓ |
| Diuretics | ECFV ↓ (Guyton integral state) |
| SGLT2i (HF) | preload/afterload modulation + ECFV ↓; HF hospitalization ↓~30% |
| ARNI (HF) | natriuresis + vasodilation; NT-proBNP trajectory modifier |

### 1.5 Validation evidence

- Guyton model: foundational; validated qualitatively for salt/BP coupling, criticized
  quantitatively (Kurtz 2018). Modern modular CV models (BioUML 2021) calibrated against
  >200 studies including Pagani, Fujimoto, Melenovsky hemodynamic datasets.
- Plaque PDE model: validated against 94 human coronary artery geometries (Sci Rep 2020);
  LDL-correlated growth models predict onset regions consistent with in-vivo MRI data.

### 1.6 Implementation complexity — **MEDIUM**

Guyton-class ODE core: ~15–40 states, parameters public, runs in ms. Simplified plaque ODE: 4–6
states. Full multiphase PDE atherosclerosis: HIGH — keep optional/offline. Heart-failure staging
(NYHA) maps naturally onto EF + BNP + congestion states already produced by this module.

---

## 2 — Metabolic Disease (Type 2 Diabetes, Obesity, NAFLD)

### 2.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| Insulin resistance / β-cell indices | HOMA / HOMA2 | Matthews 1985; Wallace, Levy & Matthews, Diabetes Care 2004 |
| Glucose-insulin dynamics (meal/IV) | Bergman minimal model / Mari model | Bergman 1979; Mari Diabetes Care 2011 review PMC3714535 |
| Whole-body glucose regulation QSP | Integrated glucose-insulin-glucagon model | Schaller et al. CPT:PSP 2013 |
| Incretin axis QSP | PB-QSP of GLP-1/GIP/DPP4 | PMC7306617 (CPT:PSP 2020); linagliptin QSP 2023; 4GI-HbA1c model 2026 |
| β-cell mass progression | UKPDS-derived decline + Topp/Hale-class β-cell mass ODE | UKPDS; ~50% loss at diagnosis, ~4%/yr thereafter |
| Hepatic steatosis | Liver-TAG meal-response ODE; unified steatosis dynamics | Pratt et al. Math Biosci 2015; Clin Nutr 2024 framework |
| NAFLD natural history | Stage-transition model HS→NASH→F0–F4→cirrhosis/HCC | CGH 2022 natural-history review |

### 2.2 State variables and normal ranges

| Variable | Normal | Prediabetes/T2D thresholds |
|---|---|---|
| Fasting plasma glucose | 70–99 mg/dL | 100–125 prediabetes; ≥126 diabetes |
| HbA1c | <5.7% | 5.7–6.4% prediabetes; ≥6.5% diabetes |
| Fasting insulin | ~5–15 µU/mL | hyperinsulinemic early IR |
| HOMA-IR | <2.0 (sex medians ≈2.4–2.5 flagged high) | elevated = insulin resistance |
| HOMA-%B | ~100% | ~50% at T2D diagnosis; −4%/yr untreated |
| β-cell mass (model state) | 1.0 normalized | progressive decline; recoverable partially early |
| Body fat mass / BMI | BMI 18.5–24.9 | energy-balance state (Hall-class model) |
| Liver TAG fraction | <5% | steatosis ≥5%; NASH + inflammation/ballooning |
| Plasma TAG, NEFA, leptin, glucagon | — | coupled metabolic states (11-variable system, J Theor Biol 2020) |

### 2.3 Key equations

**HOMA (fasting steady-state algebraic indices):**

```
HOMA-IR = (Insulin[µU/mL] × Glucose[mg/dL]) / 405          # = 22.5/(mmol/L × µU/mL)
HOMA-%B = 360 × Insulin / (Glucose − 63)                    # mg/dL form; 20·I/(G−3.5) mmol/L
```
HOMA2 replaces the linearization with the full nonlinear model. Interpretation: high IR +
rising %B = compensated resistance; high IR + falling %B = decompensation ("pancreatic Starling
curve").

**β-cell progression (virtual-patient form):**

```
dBetaMass/dt = replication(BG, lipotoxicity) − apoptosis(BG, duration)      # Topp/Hale-class
BetaFunction(t) ≈ 50% · exp(−0.04 · years_since_diagnosis)                   # UKPDS anchor
HbA1c(t)     = f(G(t)) with ~3-month lag (glycation kinetics, hemoglobin RBC lifetime)
```

**Steatosis (Pratt-class liver-TAG module):** liver TAG integrates dietary FFA influx,
*de novo* lipogenesis (insulin/carbohydrate-driven), VLDL export, β-oxidation; validated
against 24-h mixed-meal datasets. Counter-intuitive verified behavior: acute high-fat meals
transiently *lower* liver TAG via insulin-driven storage shift.

### 2.4 Drug intervention mechanisms

| Drug | Model hook |
|---|---|
| Metformin | hepatic gluconeogenesis ↓ (AMPK): EGO −20–30% → FPG ↓; modest weight neutral; no β-cell rescue |
| SGLT2 inhibitors | renal glucose excretion +60–80 g/day → HbA1c −0.5–0.8%, weight −2–3 kg; also feeds renal module (§6) |
| GLP-1 receptor agonists | glucose-dependent insulin secretion ↑, glucagon ↓, gastric emptying delayed, appetite ↓ → weight −5–15%; modeled in incretin QSP (PMC7306617) |
| DPP-4 inhibitors | prolong endogenous GLP-1 half-life (DPP4 degradation term ↓) |
| Insulin | exogenous insulin input into glucose-utilization term; suppresses EGO |
| Thiazolidinediones | PPARγ: adipose insulin sensitivity ↑, hepatic fat may ↑ |
| Pioglitazone/elafibranor (NASH) | hepatic FFA influx modifiers (validated in steatosis OOC/chip studies) |

The physiologically-based QSP model of GLP-1/GIP secretion-degradation (intraduodenal glucose →
incretin secretion → DPP4/NEP clearance) is the correct substrate for tirzepatide-class dual
agonists and oral semaglutide scenarios.

### 2.5 Validation evidence

- HOMA/HOMA2: thousands of citations; standard trial endpoint; expanded version explicitly
  built for *clinical-trial outcome modeling* of antidiabetic agents (PMC3714535).
- Gompertz-analog for β-cell decline: UKPDS longitudinal cohort (50%/4% figures).
- Incretin QSP: fitted/validated against IVGTT and clamp data in healthy and T2D subjects;
  4GI-HbA1c model predicts GLP-1/glucagon-agonist HbA1c outcomes from early PK + in-vitro data.
- Steatosis model: validated vs postprandial plasma lipid datasets; NAFLD natural-history
  Markov models reproduce biopsy-cohort stage distributions.

### 2.6 Implementation complexity — **LOW to MEDIUM**

Static HOMA indices: trivial. Meal-level glucose/insulin ODEs: ~10–15 states, parameters
well-known. Full incretin QSP: HIGH but optional; the reduced form (drug → {EGO scale factor,
insulin secretion multiplier, gastric-emptying delay, appetite/weight} ) captures 90% of
gameplay-relevant behavior.

---

## 3 — Cancer (Solid Tumors, Leukemia)

### 3.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| Untreated growth | **Reduced Gompertz** (population-validated winner) | Benzekry et al.; Vaghi et al. PLoS Comput Biol 2020 (PMC7059968) |
| Tumor–immune interaction | de Pillis–Radunskaya–Wiseman class ODEs | Cancer Res 2005; Gompertz-immune bifurcation analysis MDPI Mathematics 2026 |
| Cytotoxic chemotherapy | TGI transit-compartment model (Simeoni) + Emax kill on proliferation | Simeoni 2004; Friberg-compatible kill term |
| Radio-/radiopharmaceutical | Linear-quadratic kill + Gompertz regrowth | J Nucl Med 2024 (¹⁷⁷Lu-PSMA model, PMC11705791) |
| Metastasis | Gompertz with shared carrying capacity | Br J Cancer 2026 (s41416-025-03306-9) |
| Leukemia | same ODE core on cell-count (not volume) states + marrow competition (links §8) | standard practice |

### 3.2 State variables

| Variable | Typical values / anchors |
|---|---|
| Tumor volume V (or cell count N) | 1 mm³ ≈ 10⁶ cells initial condition |
| Carrying capacity K | species/tissue dependent; shared-K metastasis model K ≈ 1.2×10⁴ mm³ (preclinical fit) |
| Intrinsic growth rate α | median r ≈ 0.085 day⁻¹ (breast/lung xenograft population fit) |
| Immune effector count E | NK + CD8 pools; s = influx (cells/day), d = death (1/day) |
| Endothelial/angiogenesis compartment C (optional) | Hahnfeldt 1999 dynamic-K formulation |
| Resistant clone fraction | carrier model for targeted-therapy escape |
| Biomarkers | LDH, CTC count, PSA doubling time, RECIST sum-of-diameters output |

### 3.3 Key equations

**Gompertz growth (empirically dominant — beat exponential and logistic in nonlinear
mixed-effects comparison of 94 animals / 833 measurements):**

```
dV/dt = α · V · ln(K/V)
```

Strong empirical linear correlation between α and β (deceleration rate) permits the
**one-parameter reduced Gompertz**, which improved predictive power especially with Bayesian
estimation — ideal for sparse virtual-patient data ("age of tumor" estimable from limited
measurements at diagnosis). Note: Gompertz implies no stable tumor-free equilibrium (residual
disease realism), unlike logistic.

**Tumor–immune (Gompertz-immune predator–prey, MDPI 2026):**

```
dE/dt = s − d·E − m·E·T + p·T·E/(g + T)        # influx, death, inactivation, antigen-driven recruitment
dT/dt = α·T·ln(K/T) − n·E·T/(g + T) − v·C_drug·T·saturation(h,T)
```

Bifurcation structure reproduces **cancer immunoediting phases**: dormant low-tumor state,
active growth, bistability; chemotherapy introduces multistability (up to 4 coexisting stable
states) and treatment-induced oscillations when drug saturation is included — important for
simulating pulsed schedules.

**Chemotherapy kill:**

```
Edrug = Emax · C(t) / (EC50 + C(t))            # or sigmoid Emax·C^γ/(EC50^γ + C^γ)
dProl/dt = k_prol·Prol·(1 − Edrug) − k_tr·Prol # Norton–Simon style: kill acts on proliferating fraction
```

Radiopharmaceutical variant couples PBPK dose-rate to linear-quadratic survival.

### 3.4 Drug intervention mechanisms

| Modality | Model hook |
|---|---|
| Cytotoxics (taxanes, platinum, antimetabolites) | Emax log-kill on cycling cells; schedule-dependent nadirs couple to §8 myelosuppression (shared PK driver) |
| Anti-PD1/PDL1 immunotherapy | restore/expand E: ↑p (recruitment), ↓m (exhaustion/inactivation) |
| CTLA-4 | ↑ naive→effector priming (↑s) |
| TKIs (e.g., EGFR) | direct kill term + resistant-clone selection dynamics |
| Anti-VEGF | stabilizes/reduces K (angiogenesis compartment) |
| Hormonal (ER/AR) | growth-rate α scaling by hormone-ligand state |
| CAR-T | transient supraphysiologic E expansion with exhaustion kinetics |

### 3.5 Validation evidence

- Gompertz superiority: Vaghi et al. population analysis (breast + lung cancer xenografts,
  833 measurements) — logistic/exponential failed fits, Gompertz excellent; code/data public
  (`github.com/cristinavaghi/plumky`).
- de Pillis 2005 model: parameterized and validated against melanoma patient data; the
  standard teaching/validation benchmark for tumor-immune ODEs.
- TGI (Simeoni) models: industry-standard for xenograft→clinical translation; FDA-facing
  pharmacometrics literature extensive.
- Shared-capacity Gompertz metastasis model: validated across multiple mouse models/cell lines
  (Br J Cancer 2026).

### 3.6 Implementation complexity — **LOW to MEDIUM**

Core (Gompertz + Emax kill + immune ODEs): 3–6 states, cheap. Adding angiogenesis, resistance
carriers, and metastatic seeding: still tractable (<15 states). Leukemia variant swaps volume
for circulating/blastic cell counts and shares bone-marrow reserve with the hematological
module — a natural cross-module coupling in the virtual patient.

---

## 4 — Autoimmune Disease (RA, IBD, MS)

### 4.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| Anti-TNF PK/PD (RA) | TMDD + cytokine-feedback model | Rheumatology 44:323 (2005) TNF-neutralization model; Kimura et al. DMPK 2014 |
| Biologic PK | Double central-peripheral TMDD (infliximab) | Pharmaceutics 2021;13:1821 |
| Anti-IL6R | PopPK with linear + Michaelis-Menten elimination; CRP/sIL-6R PD | SUMMACTA/BREVACTA analyses, PMC5363244 |
| RA tissue pathology | Synovitis–cartilage–bone ODE model | J Theor Biol 2019 "Rheumatoid arthritis – a mathematical model" |
| IBD | Dynamic QSP model (single framework CD+UC) | Clin Transl Sci 2021 (cts.12849); IL-6 multiscale module (Dwivedi) |
| Response prediction | Cytokine-panel classifier (IL-6, IL-2, CRP, DAS28) | Heliyon 2023 ROC-AUC 0.80–0.89 |

### 4.2 State variables

| Variable | Normal / disease anchors |
|---|---|
| TNF-α, IL-6, IL-1, IL-17, IL-12/23, IFNγ | cytokine network states; TNF baseline R_C0 ≈ 3.3 nM (central), 0.46 nM (peripheral, infliximab TMDD fit) |
| CRP | <5 mg/L; falls within days of effective biologic |
| ESR | <20 mm/h |
| DAS28-ESR | composite; remission ≤2.6; response = Δ1.2 + Δ0.6 rules |
| Tender/swollen joint count (TJC/SJC28) | 0–28; model output of Kimura-style PD chain |
| Cartilage thickness / erosion (Sharp–vdHeijde) | cumulative damage states |
| Calprotectin (fecal), CDEIS/Mayo endoscopic subscore | IBD activity states |
| Relapse probability / lesion load (MS) | stochastic-event layer atop continuous inflammation state |

### 4.3 Key equations

**TMDD for anti-TNF biologics (canonical structure):**

```
dL/dt  = kin − kout·L − kon·L·R + koff·Cplx − kel·L        # ligand (TNF)
dR/dt  = ksyn − kdeg·R − kon·L·R + koff·Cplx               # target/receptor
dCplx/dt = kon·L·R − koff·Cplx − kint·Cplx                 # complex internalization
Effect: free-TNF ↓ → cytokine-amplification loop ↓ → CRP/joint inflammation decay
```

Published infliximab double-TMDD parameters: R_C0 = 3.3 nM, R_P0 = 0.46 nM, K_SS,C = 15.4 nM,
k_int,C = 0.17 day⁻¹, k_int,P = 0.0079 day⁻¹ (slower peripheral turnover explains tissue
residence and trough-guided dosing).

**Indirect-response PD chain (biologic → biomarker → clinical score):**

```
dCRP/dt  = kin·(1 − INH_TNF,IL6(t)) − kout·CRP          # CRP production inhibited by cytokine blockade
dDAS28/dt = f(CRP, SJC) with 12–24 week clinical lag    # matches observed TJC-ratio trajectories (Kimura 2014)
```

**Tissue damage (RA chronicity):**

```
dCartilage/dt = −k_deg·(cytokine_state) ;  dBoneErosion/dt = k_erosion·(RANKL/O PG balance)
```

Damage accrues while inflammation persists even if symptoms improve — reproduces the clinical
observation that early biologic initiation prevents structural progression.

### 4.4 Drug intervention mechanisms

| Drug | Model hook |
|---|---|
| Infliximab / adalimumab / etanercept / golimumab | TMDD on TNF (soluble + membrane); different kon/koff/FCR → different onset speed (IFX fastest fluctuation, ADA/ETN most stable) |
| Tocilizumab / sarilumab | IL-6R blockade → near-complete CRP suppression; popPK with MM elimination at high doses |
| Rituximab | B-cell depletion state variable |
| JAK inhibitors (tofacitinib…) | intracellular cytokine signaling block: multi-cytokine efficacy scalar |
| Methotrexate | antifolate: slows proliferating lymphocyte turnover; background modifier of biologic clearance (anti-drug antibodies ↓) |
| Anti-integrin (vedolizumab) / anti-IL12/23 (ustekinumab) (IBD) | leukocyte trafficking / differentiation-pathway switches in the IBD QSP cell-state graph |
| Steroids | broad cytokine production ↓ (kin ↓) with rebound dynamics handled by doc/28 recovery module |

### 4.5 Validation evidence

- Kimura et al. (DMPK 2014): simulated serial tender-joint-count ratios under IFX/ETN/ADA in
  *good agreement with observed clinical data*, correctly ordering onset speed and fluctuation.
- Tocilizumab SC popPK/PD across SUMMACTA+BREVACTA supported label dosing (noninferiority
  weekly-SC vs q4w-IV).
- IBD QSP: single mechanistic framework reproduces biomarker behavior in both CD and UC
  (Clin Transl Sci 2021); infliximab TMDD fitted on 133 IBD patients (Pharmaceutics 2021).
- Anti-TNF response prediction: IL-6/IL-2/CRP/DAS28 panel discriminates remitters ROC-AUC
  0.80–0.89 (Heliyon 2023) — usable as covariate priors for virtual-population generation.
- MS: no comparably mature mechanistic ODE standard exists; recommend a semi-mechanistic
  relapse-rate model (annualized relapse rate + lesion-load Poisson process modulated by
  inflammation state and drug effect) rather than pretending to a full immunology QSP.

### 4.6 Implementation complexity — **MEDIUM**

Cytokine-network ODEs: 8–20 states. TMDD adds stiffness (solve with stiff integrator or
quasi-steady-state approximation, as the published IBD model did). Parameters for the major
biologics are published — this category is parameter-rich compared to others.

---

## 5 — Neurological Disease (Alzheimer's, Parkinson's, Epilepsy)

### 5.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| Alzheimer's pathophysiology | Hao–Friedman PDE/ODE model | BMC Syst Biol 2016 (164+ citations) |
| Aβ–tau synergistics | Bertsch et al. Smoluchowski + kinetic-transport model | Brain Multiphysics 2021 |
| Biomarker progression | Dynamical ATN (dATN) model | bioRxiv 2026.01.27.701320 (calibrated on ADNI + BioFINDER-2) |
| Parkinson's treatment | Integrative levodopa PK–DA-kinetics–basal-ganglia model | Véronneau-Veilleux et al. JPKPD 2020 |
| Parkinson's symptom task | Baston–Ursino BG model + Hill PD (finger tapping) | Front Hum Neurosci 2016 |
| Epilepsy | Wendling neural-mass model (+ Virtual Epileptic Patient for network mode) | Wendling 2002/2008; Jirsa TVB/VEP |

### 5.2 State variables

**Alzheimer's:**

| Variable | Anchors |
|---|---|
| Soluble Aβ oligomers Aβ_o, plaque-bound Aβ | amyloid PET centiloids; CSF Aβ42 ↓ early |
| Hyperphosphorylated tau (NFT) | Braak stage I–VI; p-tau217 plasma marker |
| Neuron density (degree of malfunctioning distribution) | hippocampal/entorhinal atrophy rates |
| Astrocytes, microglia, peripheral macrophages, MCP-1, TNF-α | neuroinflammation states (Hao–Friedman) |
| Cognitive score | MMSE / ADAS-Cog / CDR-SB trajectories |

**Parkinson's:**

| Variable | Anchors |
|---|---|
| SNpc dopaminergic neuron fraction | ~50% lost at motor diagnosis; symptoms emerge ~70–80% striatal DA-terminal loss |
| Striatal dopamine concentration | passive-stabilization kinetics: release ↓ and reuptake ↓ together |
| Levodopa plasma / effect-compartment concentration | 2-compartment PK + effect site |
| Motor output | UPDRS-III proxy (finger-tapping frequency), Hoehn–Yahr |
| Complication states | wearing-off (shortened effect duration), dyskinesia risk (pulsatile dose × denervation) |

**Epilepsy (Wendling neural mass, per node):**

| Variable | Meaning |
|---|---|
| Pyramidal-cell PSP y₀ (+ derivatives) | EEG surrogate output |
| Excitatory, slow-inhibitory, fast-inhibitory synaptic gains (A, B, G) | B (slow/dendritic GABA) is the seizure-transition parameter |
| Slow drift variable | moves region between interictal → preonset → ictal states |

### 5.3 Key equations

**Alzheimer's (Hao–Friedman skeleton):** neurons, astrocytes, microglia, macrophages, Aβ
aggregation (oligomer↔plaque), hyperphosphorylated tau, cytokines as a PDE/ODE system; used for
*in-silico trials* of failed and ongoing candidates — notably predicted combination
anti-Aβ + TNF-α inhibition outperforms monotherapy. The dATN model adds prion-like aggregation
+ network-propagation of tau + Aβ-catalyzed tau acceleration + tau-driven atrophy, and
reproduced Braak-like cortical tau spread on longitudinal PET (ADNI, BioFINDER-2).

**Parkinson's (Véronneau-Veilleux 2020):**

```
Levodopa PK (2-cmt) → striatal DA kinetics (synthesis, release, reuptake DAT, metabolism MAO/COMT)
→ D1/D2 pathway gains → basal-ganglia action-selection network → motor-output score
Denervation D(t): dD/dt = −k_neuro (years-scale); scales vesicular DA capacity
```

Validated against patient PK and finger-tapping time courses (Baston 2016, 6 patients incl.
"wearing-off" phenotypes; influential params: Hill coefficient, EC50, effect-compartment ke0).
Reproduces shortened levodopa benefit as D(t) falls — the core clinical phenomenon.

**Epilepsy (Wendling-Chauvel, reduced 8-ODE form):**

```
Second-order linear "pulse-to-wave" synapses (pyramidal, exc., slow inh., fast inh.)
+ static sigmoid wave-to-pulse conversion per population
Slow drift of B (slow-inhibitory gain) → torus-canard transitions:
interictal spikes → low-voltage-fast onset → ictal bursting
```

Four brain states (interictal/preonset/onset/ictal) match intracranial EEG clustering in TLE
patients (Sci Rep 2023 unsupervised classification study).

### 5.4 Drug intervention mechanisms

| Drug | Model hook |
|---|---|
| Anti-amyloid mAbs (lecanemab/donanemab) | plaque-clearance term on Aβ state; downstream tau-catalysis coefficient ↓ (dATN supports testing timing hypotheses) |
| Cholinesterase inhibitors | synaptic ACh ↑ (symptomatic: shifts BG/cortical transfer functions; no disease-state change) |
| Memantine | NMDA conductance modifier |
| Levodopa + carbidopa/DDCI | input into PK compartment; long-term pulsatility drives dyskinesia state |
| Dopamine agonists / MAO-B / COMT inhibitors | modify DA kinetics constants (release, metabolism) |
| Benzodiazepines / barbiturates | ↑ fast+somatic inhibitory gain G/A in Wendling model |
| Na-channel ASM (carbamazepine, phenytoin, lamotrigine) | ↓ pyramidal excitability (sigmoid gain ↓) |
| SV2A (levetiracetam), ethosuximide (T-type Ca) | neurotransmission-release and thalamic burst modifiers |

### 5.5 Validation evidence

- Hao–Friedman: simulates drug classes that failed/succeeded in trials; widely reused AD
  model. Bertsch model outputs agree qualitatively with regional disease distribution and tau
  dependence. dATN: calibrated on two independent longitudinal biomarker cohorts.
- PD integrative model: parameters measured/estimated from human + animal data; reproduces
  wearing-off and response-duration shortening (JPKPD 2020, 1763 accesses, 19 citations).
- Wendling model: reproduces iEEG state taxonomy in patients and in-vitro high-K rat slices;
  basis of the clinically deployed Virtual Epileptic Patient (EZ localization, EPINOV trial).

### 5.6 Implementation complexity — **HIGH**

Neurodegeneration requires multi-timescale integration (protein aggregation months–decades;
PK hours) — needs adaptive-step stiff solvers or operator splitting. Epilepsy neural-mass nodes
are cheap individually (8 ODEs/node) but network mode (VEP-style connectome) is expensive;
recommend a single-focus-region reduced mode for gameplay and network mode as research option.

---

## 6 — Renal Disease (CKD, AKI)

### 6.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| CKD progression | **Total/chronic eGFR slope model** (regulatory-endorsed surrogate) | Inker et al. JASN 2019 meta-analysis; KDIGO 2024 |
| GFR computation | CKD-EPI 2021 (creatinine ± cystatin C) | standard |
| Risk stratification | Kidney Failure Risk Equation (KFRE) + UACR categories | KDIGO heatmap |
| Slope estimation | linear mixed-effects / two-slope (acute+chronic) | Clin Exp Nephrol 2026 beginner's guide; Lilly estimand framework arXiv 2504.07411 |
| AKI | KDIGO staging + creatinine kinetic model | creatinine distribution-volume kinetics |
| Mechanistic option | renal QST/QSP module (transporters, tubular handling) | Nat Rev Drug Discov 2025 QST review |

### 6.2 State variables and normal ranges

| Variable | Normal | Staging anchors |
|---|---|---|
| eGFR | ≥90 mL/min/1.73m² | G1–G5: 90+/60–89/45–59/30–44/15–29/<15 |
| Serum creatinine | 0.6–1.2 (M) / 0.5–1.1 (F) mg/dL | AKI: ↑≥0.3 mg/dL/48h or ≥1.5× baseline |
| UACR | <30 mg/g | A1 <30, A2 30–300, A3 >300 |
| eGFR slope | ~−1 mL/min/yr age-related | rapid progression >−5/yr (KDIGO); ESKD projection = (eGFR−15)/slope |
| Proteinuria/albuminuria (state) | trace | drives slope acceleration |
| Tubular function markers (optional) | NGAL, KIM-1, cystatin C | early AKI detection states |

### 6.3 Key equations

```
eGFR(t) = CKD_EPI2021(age, sex, creatinine(t), [cystatin C])
Slope   = d(eGFR)/dt estimated by linear mixed-effects over repeated measures
Two-slope decomposition: eGFR(t) = eGFR0 + slope_acute·min(t,t_acute) + slope_chronic·max(0, t−t_acute)
Time-to-KRT = (eGFR_now − 15) / slope_chronic                       # linear projection
KFRE: Cox-form hazard from {age, sex, eGFR, UACR, diabetes, hypertension}
```

AKI: model serum creatinine as first-order approach to a new steady state (generation =
constant, clearance ∝ GFR) — this produces the characteristic lag between true GFR fall and
observed creatinine, which the virtual patient must respect when staging.

### 6.4 Drug intervention mechanisms

| Agent / insult | Model effect |
|---|---|
| SGLT2 inhibitors | chronic slope improvement ~50% (DAPA-CKD, EMPA-KIDNEY) + acute dip artifact |
| RAAS blockers (ACEi/ARB/MRA) | chronic slope benefit + acute hemodynamic dip (finerenone FIDELIO-DKD: acute −3.18, chronic −2.66 vs placebo −3.97 mL/min/yr; CREDENCE placebo slope −4.71) |
| NSAIDs | afferent arteriole constriction → GFR ↓; "triple whammy" with ACEi + diuretic (doc/28 DDI channel) |
| Aminoglycosides, amphotericin, contrast, cisplatin | proximal-tubule injury state → creatinine rise, Mg/K wasting, non-oliguric pattern |
| Clinically meaningful deceleration | −1.49 mL/min/yr sustained ⇒ HR 0.8 for kidney-failure risk (CKJ 2025) |

### 6.5 Validation evidence

- eGFR slope: FDA/EMA-accepted surrogate endpoint; Inker 2019 meta-analysis of RCT treatment
  effects links slope change to clinical outcomes; CKJ 2025 stage-stratified confirmation in
  Japanese cohort (n=2713, 985 KFRT events).
- Two-slope (acute/chronic) decomposition: standard in FIDELIO/DAPA-CKD reporting; the
  documented acute-dip magnitudes give exact calibration targets for the simulator.
- Natural-history heterogeneity: in advanced CKD, trajectories are linear 38%, nonlinear 24%,
  positive 15% (mean slope −3.35 ± 4.45) — support nonlinear slope options.

### 6.6 Implementation complexity — **LOW**

This is the cheapest category to make accurate: the slope formalism is literally designed for
sparse longitudinal data, and all calibration constants (trial slopes, dips, KDIGO cutoffs) are
published. Mechanistic nephron QSP optional later.

---

## 7 — Hepatic Disease (Cirrhosis, Hepatitis, DILI)

### 7.1 Recommended model stack

| Component | Model | Reference |
|---|---|---|
| DILI prediction | **DILIsym quantitative systems toxicology** (mechanistic gold standard) | Watkins Curr Opin Toxicol 2020; Howell/Woodhead/Shoda series |
| Reduced DILI engine | ALT-release ↔ hepatocyte-mass ↔ bilirubin chain | derived from DILIsym structure |
| Severity staging | Child-Pugh (score 5–15 → A/B/C) and MELD | standard clinical rubrics |
| Fibrosis/cirrhosis progression | Stage-transition (Markov/ODE hybrid) F0→F4 + HVPG | NAFLD/natural-history literature (CGH 2022) |
| Viral hepatitis | viral-dynamics ODEs (target-cell limited) | standard HIV/HBV/HCV modeling tradition |

### 7.2 State variables

| Variable | Normal | Clinical anchors |
|---|---|---|
| Functional hepatocyte mass | 100% | bilirubin rises when loss >~30% (DILIsym finding) |
| ALT / AST | <35–40 U/L | release ∝ hepatocyte death rate |
| Total bilirubin | 0.2–1.2 mg/dL | global liver function proxy |
| Albumin / INR | 3.5–5 g/dL / 0.8–1.2 | synthetic function (Child-Pugh inputs) |
| Alkaline phosphatase, GGT | <120 / <40 | cholestasis discriminator (R-value) |
| Fibrosis stage / HVPG | F0 / HVPG 3–5 mmHg | F4 cirrhosis; HVPG ≥10 clinically significant portal HTN |
| Platelets | 150–400×10³/µL | fall with portal HTN (splenic sequestration) — couples to §8 |
| Hy's Law flag | off | ALT>3×ULN AND TBL>2×ULN without ALP explanation |

### 7.3 Key equations

**DILIsym architecture (reduced form for the virtual patient):**

```
Drug/metabolite exposure (doc/27 PBPK) ──▶ { oxidative stress, mitochondrial ETC impairment,
                                              bile-acid transporter (BSEP) inhibition }
each scaled by in-vitro IC50/EC50 ──▶ hepatocyte death rate (apoptotic + necrotic channels)
dALT_serum/dt = release(death_rate) − k_clear·ALT
Functional_mass(t) integrates death − regeneration (mitochondrial biogenesis, NRF-2 adaptation,
              FXR feedback included in full model)
TBL responds once functional mass < ~70% of baseline
```

Key DILIsym finding to preserve: **three assessable mechanisms (ROS, mitochondrial respiration
interference, bile-acid homeostasis disruption) account for hepatotoxicity of >80% of drugs**
in their validation set; prospective successes include predicting telcagepant/MK-3207
hepatotoxicity and confirming ubrogepant safety before phase 3 (FDA approval without liver
warnings).

**Scoring rubrics:**

```
Child-Pugh = bili + alb + INR + ascites + encephalopathy grades       → A(5–6)/B(7–9)/C(10–15)
MELD(i) = 3.78·ln(TBL) + 11.2·ln(INR) + 9.57·ln(creatinine) + 6.43    (± Na correction)
```

### 7.4 Drug intervention mechanisms

| Insult/therapy | Model effect |
|---|---|
| Acetaminophen overdose | NAPQI-mediated GSH depletion → necrotic death channel; N-acetylcysteine restores GSH (validated DILIsym application, JPET 2012) |
| Troglitazone/tolvaptan-class | BSEP inhibition + mitochondrial dual hit; SimPops reproduce idiosyncratic susceptibility tails |
| Statins | mild ALT elevations; no functional-mass impact at licensed doses |
| Valproate | mitochondrial/urea-cycle stressors |
| Antivirals (HBV/HCV) | viral-load exponential decay (viral dynamics module) → inflammation state ↓ → fibrosis progression rate ↓ |
| Alcohol | additive oxidative-stress + steatosis coupling to §2 NAFLD module |
| Ursodiol / FXR agonists | bile-acid pool normalization (obeticholic acid QST prediction published 2026) |

### 7.5 Validation evidence

- DILIsym: developed by a 19-pharma + FDA consortium; retrospective + prospective validations
  (Entolimod phase-I interpretation; migraine-grepant program; obeticholic acid risk
  prediction CPT 2026); SimPops variability reproduces incidence *and severity* distributions.
- Child-Pugh/MELD: universally validated transplant/mortality predictors — trivial to embed
  exactly as clinical calculators fed by the module's lab outputs.

### 7.6 Implementation complexity — **MEDIUM to HIGH**

Full DILIsym-scale QST: very high (dozens of submodels) — do not reimplement wholesale. A
faithful reduced engine (exposure → 3 mechanism scalars → death rate → ALT/TBL/INR/albumin →
Child-Pugh/MELD) is ~10–15 states and captures gameplay-relevant behavior; the three-mechanism
structure is the part worth preserving exactly.

---

## 8 — Hematological Disease (Anemia, Neutropenia, Thrombocytopenia)

### 8.1 Recommended model stack

| Lineage | Model | Reference |
|---|---|---|
| Neutrophils | **Friberg semimechanistic myelosuppression model** (field gold standard) | Friberg et al. JCO 2002; "most frequent… gold-standard approach" (CPT:PSP 2017) |
| Red cells | Cell-kinetic erythropoiesis + iron model | Fuertinger/Marciniak/Czochra Sci Rep 2020 (PMC7248076) |
| Red cells (minimal) | 4-equation EPO-feedback model validated on 36 studies | Dor & Alon PLoS Comput Biol 2026 |
| EPO drugs | EPO PK/PD + absorption submodule | same Sci Rep model |
| Platelets | Friberg-transit analog with thrombopoietin feedback | standard extensions (van Hasselt etc.) |
| Modern QSP option | Avadomide-style neutrophil lifecycle QSP (maturation block vs cytotoxicity) | AAPS J 2021 (PMC8397660) |

### 8.2 State variables and normal ranges

| Variable | Normal range | Notes |
|---|---|---|
| ANC | 1.5–8.0 ×10³/µL (baseline Circ₀ ≈ 4–5) | grade 3 <1.0, grade 4 <0.5 |
| Proliferating progenitor pool (Prol) | steady state = Circ₀·(k_tr/k_prol) | chemo target |
| Transit compartments Tx1–Tx3 | maturation delay | MTT ≈ 4/k_tr ≈ 5–7 days (docetaxel MTT ≈ 134 h) |
| Hemoglobin / RBC mass | 13.5–17.5 (M) / 12–15.5 (F) g/dL | RBC lifespan ≈ 120 d (25 d in hemolysis) |
| Reticulocytes | 25–100 ×10³/µL (0.5–2.5%) | marrow pool leads circulating count by ~3–5 d |
| EPO | 4–26 U/L | inverse log-linear with Hb; disease-specific deviations are diagnostic patterns |
| Iron panel | ferritin, TSAT, hepcidin, transferrin saturation | hepcidin–ferroportin axis in Sci Rep model |
| Platelets | 150–400 ×10³/µL | nadir later than ANC (longer transit) |
| Bone-marrow reserve | aggregate progenitor state | shared with §3 leukemia module |

### 8.3 Key equations

**Friberg model (exact published form):**

```
dProl/dt = k_tr · (Circ₀/Circ)^γ · (1 − E_drug) · Prol − k_tr · Prol
dTx1/dt  = k_tr · (Prol − Tx1) ;  dTx2/dt = k_tr·(Tx1−Tx2) ;  dTx3/dt = k_tr·(Tx2−Tx3)
dCirc/dt = k_tr · (Tx3 − Circ)
E_drug   = Emax · C(t) / (EC50 + C(t))          # or sigmoid/power forms per drug
```

Design properties that made it the standard: *system parameters* (k_tr, γ ≈ 0.16–0.23, Circ₀)
are **consistent across drugs and patients**; only Emax/EC50 (and occasionally MTT) are
drug-specific — exactly the right factoring for a virtual-patient engine where many drugs share
the physiology. Nadir day 7–14 emerges from transit times, not hardcoding; recovery overshoot
emerges from the (Circ₀/Circ)^γ feedback.

**Minimal erythropoiesis (Dor–Alon):** four ODEs — erythroid progenitors H(t) with
EPO-dependent proliferation a(E) and differentiation d(E) (logistic cap H_max), marrow
reticulocytes R(t), circulating RBCs C(t) with finite lifespan, plasma EPO E(t) from
kidney hypoxia sensing. Validated across healthy, iron-deficiency, aplastic, hemolytic,
and renal-anemia datasets (36 aggregated studies). The fuller Sci Rep model adds BE/CE/PEB/MEB
stages, hepcidin/NTBI/transferrin iron compartments, chemo cytotoxicity, and EPO-derivative PK.

### 8.4 Drug intervention mechanisms

| Agent | Model hook |
|---|---|
| Any myelosuppressive cytotoxic | Emax/EC50 on Prol proliferation (drug-specific pair; system params shared) |
| Schedule effects | lower more-frequent dosing yields shallower nadirs than larger spaced doses (model-derived, taxane-consistent) — emergent, not scripted |
| G-CSF filgrastim/pegfilgrastim | ↑k_prol, shortened transit (↑k_tr) → faster recovery; pegfilgrastim = sustained exposure |
| EPO analogues / HIF-PHI | EPO compartment forcing; validated across dosing schedules in Sci Rep model |
| Iron (oral/IV) | iron-module fluxes; hepcidin feedback limits utilization |
| Transfusion | direct step addition to RBC pool |
| TPO mimetics (eltrombopag/romiplostim) | megakaryocyte-proliferation drive on platelet lineage |
| Linezolid, clozapine, rituximab late-onset | drug-specific neutropenia mechanisms better served by the QSP maturation-block variant than plain Friberg |

### 8.5 Validation evidence

- Friberg 2002: developed on docetaxel/paclitaxel/etoposide, applied successfully to DMDC,
  irinotecan, vinflunine, pemetrexed, topotecan, epirubicin regimens; Kloft log-transformed
  variant improves nadir capture; daily-monitoring studies show model-based ANC forecasting
  supports dose individualization.
- Hybrid PKPD/ML: PKPD-enrichment of ML features improves ANC prediction beyond either alone
  (CPT:PSP 2023) — useful precedent for HelixLang's numeric core.
- Erythropoiesis models: quantitative agreement with RBC/iron/EPO time series under chemo,
  EPO dosing, donation, malnutrition (Sci Rep 2020).

### 8.6 Implementation complexity — **LOW**

Five ODEs + published parameters make this the highest-value-per-effort module; it also
provides the toxicity backbone that makes oncology dosing (§3) feel real.

---

## 9 — Cross-Module Coupling Map (for the virtual-patient integrator)

These couplings are what turn eight disease modules into one person; all run through the
doc/28 hourly loop:

```
Metabolic(§2) ──glucose/LDL──▶ Cardiovascular(§1)      # T2D accelerates plaque; metformin/SGLT2i feed CV risk
Cardiovascular(§1) ──MAP/perfusion──▶ Renal(§6)        # Guyton ECFV ↔ RAAS ↔ GFR; ACEi touches both
Renal(§6) ──EPO/phosphate/VitD──▶ Hematology(§8)/Bone  # renal anemia (EPO ↓), CKD-MBD
Renal(§6) ──clearance──▶ ALL drug PK (doc/27)          # eGFR scales renally-cleared drug dosing
Hepatic(§7) ──clearance/protein──▶ ALL drug PK         # Child-Pugh scales hepatic clearance; albumin → free fraction
Hepatic(§7) ──steatosis bridge──▶ Metabolic(§2)        # NAFLD ↔ insulin resistance bidirectional
Cancer(§3) ◀──shared marrow──▶ Hematology(§8)          # tumor burden ↔ marrow reserve; chemo hits both
Autoimmune(§4) ──chronic inflammation──▶ CV(§1)/Bone   # RA raises CV risk; steroids touch everything
Neurological(§5) ──mostly autonomous──                 # slow states; drugs interact via PK only (plus ACh effects)
Hematology(§8) ──platelets/immunity──▶ Bleeding/infection events
```

Recommended integration order (by dependency): **§8 hematology and §6 renal first** (they gate
drug PK and toxicity for everything else), then §2 metabolic + §7 hepatic (PK/scoring hubs),
then §1 cardiovascular, §4 autoimmune, §3 cancer, and finally §5 neurological.

## 10 — Implementation Complexity Summary

| Category | Core-model size | Parameter availability | Runtime | Overall |
|---|---|---|---|---|
| 8 Hematological | 5–12 ODEs | Excellent (Friberg params published per drug) | µs | **LOW** |
| 6 Renal | 2–8 states | Excellent (equations are standards) | µs | **LOW** |
| 2 Metabolic | 10–15 ODEs (QSP optional +50) | Very good | ms | **LOW–MED** |
| 3 Cancer | 3–15 ODEs | Good (population fits published) | ms | **LOW–MED** |
| 1 Cardiovascular | 15–40 ODEs (plaque PDE optional) | Good; some Guyton caveats | ms | **MEDIUM** |
| 4 Autoimmune | 8–20 ODEs + TMDD stiffness | Good for major biologics | ms | **MEDIUM** |
| 7 Hepatic | reduced 10–15 states (full QST ≫) | Good (DILIsym publications; IC50 data needed per drug) | ms | **MED–HIGH** |
| 5 Neurological | 8 ODEs/node; decades-timescale states | Mixed; cohort-calibrated | ms–s | **HIGH** |

---

## 11 — Key References

**Cardiovascular:** Guyton AC et al., *Circulation: overall regulation*, Annu Rev Physiol 1972 ·
Karaaslan F et al., PLoS Comput Biol 2012 (virtual patients + Guyton SA) · Kurtz TW et al.,
Hypertension 2018 (validity assessment) · Kiselev/Kutumov modular CV model, Front Physiol 2021
(PMC8632703) · Coronary plaque model, Sci Rep 2020 (PMC7562914) · Multiphase plaque models,
Bull Math Biol 2023 · Cholesterol-plaque computational model, PMC10452179 (2023).

**Metabolic:** Matthews DR et al., Diabetologia 1985 (HOMA) · Wallace TM, Levy JC, Matthews DR,
Diabetes Care 2004 (HOMA2) · HOMA clinical-trial expansion, PMC3714535 · UKPDS β-cell decline
(Diabetes Obes Metab 2023 review) · Schaller S et al., CPT:PSP 2013 (glucose QSP) ·
PB-QSP GLP-1/GIP, CPT:PSP 2020 (PMC7306617) · Pratt AC, Wattis JAD, Salter AM, Math Biosci 2015
(hepatic lipid) · Steatosis unified framework, Clin Nutr 2024 · NAFLD natural history, CGH 2022.

**Cancer:** Vaghi C et al., PLoS Comput Biol 2020 (reduced Gompertz; PMC7059968) ·
Benzekry S et al. (Gompertz correlation) · de Pillis LG, Radunskaya AE, Wiseman CL, Cancer Res
2005 · Gompertz tumor–immune bifurcations, Mathematics 14(3):491, 2026 · Simeoni M et al. 2004
(TGI) · Shared-K metastatic Gompertz, Br J Cancer 2026 · ¹⁷⁷Lu-PSMA Gompertz+LQ, JNM 2024.

**Autoimmune:** TNF-neutralization model, Rheumatology 2005;44:323 · Kimura K et al., Drug
Metab Pharmacokinet 2014 · Double-TMDD infliximab, Pharmaceutics 2021;13:1821 · Tocilizumab
popPK/PD SUMMACTA/BREVACTA, PMC5363244 · RA mathematical model, J Theor Biol 2019 · Dynamic QSP
of IBD, Clin Transl Sci 2021 (cts.12849) · Anti-TNF response prediction, Heliyon 2023 ·
Quantitative modelling in RA review, Cells 2020;9:74.

**Neurological:** Hao W, Friedman A, BMC Syst Biol 2016 · Bertsch M et al., Brain Multiphysics
2021 · dATN model, bioRxiv 2026.01.27.701320 · Véronneau-Veilleux F et al., J Pharmacokinet
Pharmacodyn 2020 · Baston C, Ursino M et al., Front Hum Neurosci 2016;10:280 · Computational
models in PD, JNNP 2018;89:1181 · Wendling F et al., Eur J Neurosci 2002 · Neural-mass seizure
classification, Sci Rep 2023 · Köksal Ersöz E et al., PLoS Comput Biol 2020.

**Renal:** Inker LA et al., JASN 2019 (slope surrogate) · KDIGO 2024 CKD guideline ·
Clinically meaningful eGFR slope, Clin Kidney J 2025;18:sfae398 · eGFR-slope methods guide,
Clin Exp Nephrol 2026 · DAPA-CKD, NEJM 2020; EMPA-KIDNEY, NEJM 2023; FIDELIO-DKD ·
QST review, Nat Rev Drug Discov 2025.

**Hepatic:** Watkins PB, Curr Opin Toxicol 2020 · Howell BA et al., CPT:PSP 2014 · Woodhead JL
et al., JPET 2012 (NAC/APAP), Toxicol Sci 2017 (tolvaptan) · Shoda LKM et al., Biopharm Drug
Dispos 2014 · Obeticholic acid QST, CPT 2026 · Andrade RJ et al., Nat Rev Dis Primers 2019.

**Hematological:** Friberg LE et al., J Clin Oncol 2002;20:4713 · Kloft C et al., Clin Cancer
Res 2006 · Hansson EK, Friberg LE, Cancer Chemother Pharmacol 2012 · QSP myelosuppression
review, CPT:PSP 2017 (psp4.12191) · Avadomide QSP, AAPS J 2021;23:103 · Fuertinger DH et al.,
Sci Rep 2020 (erythropoiesis + iron; PMC7248076) · Dor H, Alon U, PLoS Comput Biol 2026
(minimal RBC model).

---

## 12 — Recommendation Summary

1. **Adopt** Friberg myelosuppression (§8), eGFR-slope renal (§6), HOMA/meal-response metabolic
   (§2), and reduced-Gompertz + Emax-kill oncology (§3) as the first wave — lowest complexity,
   strongest validation, immediate gameplay payoff (chemotherapy becomes genuinely dangerous
   and schedulable).
2. **Adopt in reduced form**: DILI engine (three-mechanism chain, §7), TMDD biologics for RA/IBD
   (§4), Guyton-lineage BP core (§1).
3. **Keep behind research flags**: multiphase plaque PDE (§1), incretin full QSP (§2),
   network epilepsy/VEP (§5), full DILIsym-scale QST (§7).
4. Every module must expose its states through the same `ClinicalLabs`/`DiseaseStage` interface
   defined in doc/28 so staging rubrics (CKD G-stage, Child-Pugh, NYHA, DAS28, RECIST, ANC
   grades) remain the single source of clinical truth.
