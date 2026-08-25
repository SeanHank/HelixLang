# 30 — Frontier Virtual Patient Design: Research Grounding and Extension Roadmap

> **Status:** PROPOSED (research synthesis + extension design; not yet implemented)
> **Depends on:** doc/28, doc/29 (implemented Virtual Patient System)
> **Date:** 2026-08-24

---

## 1 — Motivation

Docs 28–29 delivered a working façade: genome (VCF) → traits → diseases → drugs → hourly
labs/vitals/stages, with stateful PBPK, DDI, staging, and recovery. It covers a *narrow but
quantitatively honest* slice of whole-person simulation: pharmacogenomics-driven disposition,
toxicity channels, and coarse disease dynamics on a GEM metabolic substrate.

This document answers two questions:

1. **Where does the current implementation sit relative to the published state of the art?**
   (A survey of PBPK parameter prediction, whole-body physiology models, QSP, immune ABMs,
   endocrine feedback loops, tissue-repair models, validation frameworks, digital twins,
   and ML-for-biology.)
2. **What should we build next, in what order, and what must we explicitly NOT attempt to
   build ourselves?**

Every claim below carries a citation anchor so future implementation work can verify against
the primary literature.

---

## 2 — State-of-the-Art Survey Findings

### 2.1 — PK/PBPK parameter prediction from molecular structure

| Reference | Contribution | Reported accuracy |
|---|---|---|
| Lombardo et al., *J. Med. Chem.* 2016–2018 | Foundational IV-PK dataset (~1,350 compounds: VDss, CL, t½, fu) | — |
| PKSmart, *J. Cheminformatics* (2025), doi:10.1186/s13321-025-01066-5 | Open-source RF predictors of human VDss/CL/fu/t½ from SMILES (+ predicted animal PK) | VDss ext. R² ≈ 0.39–0.53; CL ≈ 0.24–0.46 |
| Novartis DeepCt (GitHub 2024) | Deep learning of full concentration–time curves from SMILES | profile-level, not summary stats |
| MMPK (GitHub 2025) | Multimodal end-to-end prediction of 8 oral PK parameters (graphs/substructures/SMILES) | — |
| XGBoost + ChemBERTa (PubMed 39589671, 2024) | CL GMFE 1.768 (r²=0.528); VDss GMFE 1.401 (r²=0.902) when animal/in vitro data included | ~76% CL within 3-fold; 100% VDss within 3-fold |
| Hybrid AI→PBPK pipeline (PAGE 2026) | Transformer predicts logP/solubility/permeability/**fu** → feeds mechanistic engine | intrinsic CL AFE 1.12; **fu worst endpoint (R² = 0.213)** |

**Tools:** Simcyp (commercial; used in ~80% of FDA PBPK submissions; 29 population libraries),
GastroPlus (ACAT™ GI absorption), **PK-Sim/MoBi — open source GPLv2, regulatory-accepted**
(finerenone, vericiguat labels).

**Takeaway for HelixLang:** structure-only predictions are ±2-fold at best; they are adequate
for *defaults* and *triage*, not for clinical-grade claims. Our current approach (user-supplied
PK constants with literature anchors, doc/28 §14.2) remains the correct backbone; ML predictors
should be added as an optional auto-fill with explicit uncertainty flags.

### 2.2 — Whole-body physiology models

| Model | Scale | Notes |
|---|---|---|
| Guyton/Coleman 1972 (*Annu. Rev. Physiol.*) | ~150 variables | Original CV/fluid regulation model |
| Coleman & Randall HUMAN (1983) / QCP | ~4,000 vars | C++ descendant |
| **HumMod**, Hester et al., *Front. Physiol.* 2:12 (2011), doi:10.3389/fphys.2011.00012 | **5,000+ variables** | XML-defined; parses 2,900 files <10 s; 24 h sim in ~30 s; CV/respiratory/renal/neural/endocrine/metabolic |
| SAPHIR (Thomas et al.) | modular Guyton port | Physiome Project lineage |
| Guyton 2008 CellML (models.physiomeproject.org) | full model | importable format |

**Key limitation (stated by HumMod authors themselves):** no other group has produced a
comparably stable integrative whole-body model; Physiome modules remain narrow-scope and
un-integrated. These models are empirically-fitted curvilinea functions, not first-principles
biochemistry, and carry no native immune or drug layers.

**Takeaway:** the HumMod/QCP variable graph is public and is the only credible candidate for a
"healthy-human + common pathophysiology" backbone beyond our current 6-compartment physiology.
Porting it is realistic (30–60k LOC equivalent) but is a *separate project* from the drug/disease
stack; integration coupling (organs ↔ GEM ↔ PK) does not exist anywhere in the literature and
would be original work.

### 2.3 — Disease modeling

- **T2D:** Topp et al. 2000 minimal β-cell/glucose model; Karin et al. dynamical-compensation
  extensions (*Mol. Sys. Biol.* 16:e9510, 2020); Palumbo et al. calibrated to the Diabetes
  Prevention Program (*PLoS ONE* 13(2):e0192472, 2018) — individual forecast errors ~5% for
  weight/HbA1c.
- **Atherosclerosis:** free-boundary PDE plaque growth (PMC8415469); Bhui & Hayenga hybrid
  ABM+CFD leukocyte migration; Glagov remodeling simulations (PLoS ONE 2018).
- **Cancer:** dominated by QSP platforms (§2.5).
- **Autoimmune:** RA multi-scale joint model with calibrated virtual population matched to
  PRECISESADS multiomics clusters (*npj Systems Biology* 2024); cutaneous lupus QSP virtual
  population n=968 (*Cell Reports Medicine*, 2025).

**Takeaway:** every credible disease model is bespoke, single-disease, excludes onset/progression
phases by design, and required its own calibration campaign (typically person-years). Cross-disease
interaction is unsolved even by the Dassault MEDITWIN consortium. For HelixLang the reachable goal
is a *framework + authoring surface* seeded by the ~dozen open validated models — not pre-built
coverage of "any disease."

### 2.4 — Immune system modeling

| Model | Type | Scale | Calibration source |
|---|---|---|---|
| Basic Immune Simulator, Folcik et al., *Theor. Biol. Med. Model.* 4:39 (2007) / 8:1 (2011) | agent-based (RepastJ) | 3 compartments (tissue/lymph/blood), cells-as-agents, diffusing cytokines | 146 source papers for rules |
| Innate Immune Response ABM, An et al., *Crit. Care Med.* (2004) | ABM | cytokine network | reproduced failed anti-TNF/anti-IL-1 trial outcomes qualitatively |
| IIRABM GA calibration, Cockrell et al., *Front. Physiol.* 12:662845 (2021) | ABM | **432 free continuous parameters** | burn-patient cytokine time series via genetic algorithm |
| COMMBINI, *Front. Immunol.* 14:1231329 (2023) | ABM + PDE (PhysiCell/BioFVM, 10 µm grid) | bone injury macrophages | GA-calibrated + independent validation |
| Netflux-class logic models | ODE/logic | 91 nodes, 142 reactions | literature-curated topology |

**Takeaway:** rule-based ABM granularity (BIS-class innate response skeleton) is implementable;
comprehensive adaptive immunity with clonal diversity has never reached clinical-grade validation
anywhere. Cytokine diffusion constants in living tissue are unknown — uniform abstractions are the
accepted compromise.

### 2.5 — Drug mechanism-of-action modeling (QSP)

- QSP-IO platform: Sové et al., *CPT PSP* (2020), PMC7499194 — open MATLAB/SimBiology toolbox,
  modular cancer/APC/T-cell/checkpoint/antibody-PK modules.
- Milberg et al., *Sci. Reports* 9:8169 (2019): melanoma checkpoint model capturing delayed
  responses, non-monotonic dynamics, responder diversity under CTLA-4/PD-1/PD-L1 blockade.
- Wang et al. NSCLC triple-combination QSP (*Pharmaceutics* 17:238, 2024): **216 ODEs +
  55 algebraic equations**, 10 modules, validated against the IMpower131 phase-3 trial.
- Transcriptome-informed metastatic TNBC model (*Science Advances* adg0289, 2023) with metastatic
  + draining-LN compartments; pembrolizumab biomarker discovery.
- Bispecific antibody QSP platform (*Front. Pharmacol.* 2025): >10 checkpoints, 20+ antibodies,
  60+ dose levels for calibration.

**What replaces bare Hill equations in practice:** mass-action receptor-ligand binding with measured
Kd values, competitive binding at immunological synapses, TMDD for biologics, mechanism-based DDIs
(reversible/TDI/induction-with-turnover), target-mediated clearance, logic-based signaling cascades.

**Takeaway:** compute cost of these models is trivial (<1 min/patient-run). The real currency is the
parameter evidence chain — every new MoA needs measured binding/enzymology data. Hill PD (our current
`human/pharmacodynamics.py`) is the correct v1 primitive; competitive-binding and TMDD are the natural
v2 extensions.

### 2.6 — Organ crosstalk / endocrine feedback

| Axis | Canonical model | States | Source |
|---|---|---|---|
| Insulin/glucose | Bergman minimal model (1981, *Ann. Biomed. Eng.*) → HOMA-IR, Disposition Index | ~3–6 | 2,000+ publications |
| HPA axis | Karin et al. gland-mass dynamics (*Mol. Sys. Biol.* 16:e9510, 2020) | ~6–9 | explains week-scale dysregulation (anorexia, alcoholism) |
| HPT axis | 4-ODE minimal model w/ FT3 homeostasis proof | 4 | *Front. Endocrinol.* 13:825107 (2022) |
| Unifying motifs | five circuit classes across insulin/PTH/thyroid/HPA/hypothalamic-pituitary | — | Karin et al., *Nat. Comms* s41467-025-65924-4 (2025) |
| Ultradian cortisol | Walker et al. pituitary-adrenal oscillator (experimentally confirmed) | 2 | — |

**Takeaway:** each axis is 3–10 ODEs with Hill feedbacks — trivially expressible in the Helix DSL and
microseconds to solve. This is the cheapest, highest-value realism upgrade available; the difficulty is
cross-axis parameterization consistency, not mathematics. HumMod already embeds coarser versions of all
of these (ADH, aldosterone, cortisol, insulin, EPO), which argues for the port rather than re-derivation.

### 2.7 — Tissue repair / recovery

- Rouillard/Holmes cardiac fibrosis: tissue ABM coupled to a 91-node fibroblast signaling LDE network
  (*Front. Physiol.* 10:1481, 2019) — first intracellular-network→ECM coupling; validated vs human
  cardiac fibroblasts.
- Skin wound healing multiscale hybrid model (discrete epidermis + dermal continuum PDEs;
  PMC6561509, 2019) — predicts hypertrophic vs hypotrophic scarring from clot density and geometry.
- Cellular-Potts senescence/wound model (*PLOS Comp Biol* e1012298, 2024).
- Skeletal muscle regeneration ABM (*eLife* 12:e91924, 2024) — >100 parameters/rules from >100 studies,
  predicts 13% CSA recovery improvement under combined perturbations.
- Vermolen/Boon FE wound contraction mechanics (Biomech. Model. Mechanobiol., 2020).

Standard architecture everywhere: **hybrid discrete-cells (ABM/CPM) + continuum ECM/cytokines
(reaction-diffusion PDEs) + optional mechanics (FEM)**, spanning hours-to-months phases.

**Takeaway:** our `human/recovery.py` first-order relaxation model is consistent with how these papers
handle *biomarker* kinetics (the half-life ladder ALT 47 h < CRP 19 h < creatinine days < CBC weeks is
exactly their output layer). Structural repair (fibrosis yes/no trajectories) would need an ABM+RD
layer per organ — feasible but each organ is a 5–15k LOC campaign.

### 2.8 — Clinical validation frameworks

- Zhao et al. FDA workshop summary, *Clin. Transl. Sci.* (2021), PMC8592512: best-practices consensus;
  ASME V&V credibility framework adapted to PBPK; context-of-use risk tiering.
- Zhang et al., *Pharmaceutics* 17:1413 (2025): audit of 245 FDA NDAs 2020–2024 — **26.5% submitted
  PBPK as pivotal evidence**; 81.9% DDI-focused; reviewer taxonomy pivotal/adequate/exploratory.
- EMA Guideline EMA/CHMP/458101/2016: reporting requirements; platform qualification vs drug-model
  evaluation; ≥100 virtual individuals recommended per simulation.
- NPDE method for population-PBPK adequacy (PMC7293575).
- ENRICHMENT playbook (Dassault + FDA, completed 2024): in silico trials for devices.

**Doctrine distilled (should become built-in tooling):**
1. Verify software ≠ validate model; both required, separately evidenced.
2. Context of use determines rigor tier.
3. Unbroken chain: in vitro → IVIVE → clinical calibration → prospective prediction
   (LIVDELZI exemplar vs ATTRUBY failure case).
4. Acceptance metrics: predicted/observed within **0.8–1.25×** for DDI AUC ratios, **2-fold** for
   concentration profiles; visual predictive checks + percentile bands.
5. Virtual populations need demographic justification; pediatric/hepatic-impairment PBPK informs
   but does not yet replace studies.

**Gap in current implementation:** doc/28 §15 defines benchmark scenarios, but there is no
built-in VPC/fold-error/NPDE/sensitivity reporting machinery. This is cheap to add and is what makes
outputs credible.

### 2.9 — Digital twin projects (scope reality check)

| Project | Scope | Status |
|---|---|---|
| Living Heart (Levine et al., *J. Med. Devices* 2014; Dassault) | first commercial whole-heart electro-mechanical twin; FEM, imaging-personalized | mature; ENRICHMENT (with US FDA) completed device in-silico-trial playbook 2024 |
| MEDITWIN consortium (14 orgs) | 7 diseases, multi-organ ambition | early; treats cross-disease interaction as open frontier |
| Unlearn.AI | AI-generated twin control arms (PROCOVA™); Alzheimer's trial power 80→90%; FDA Type C recognition | commercial, deployed |
| Simcyp population PBPK | virtual subjects; 11 regulator licenses; 120+ novel drugs | industry standard |
| DILIsym / Virtual Tumour / Quris-AI | organ-specific suites linked to GastroPlus | commercial niche |

No integrated multi-organ mechanistic twin with drugs+disease+genetics exists commercially. Bottlenecks
are coordination/validation/data — not algorithms.

### 2.10 — ML for biology (drug response & progression)

- GDSC/CCLE benchmarks crowded with GNN+omics fusion models (DeepCDR, GraphDRP, GPDRP, TransCDR,
  GCNPath, DBDNMF): warm-start IC50 R² ≈ 0.85–0.90, **collapses cold-start** (unseen scaffolds/
  cell-line clusters) — documented explicitly by TransCDR.
- Progression/survival: Dynamic-DeepHit, TrajSurv (neural CDEs, ICML 2025), SCOPE transformer
  (*npj Digital Medicine* 2024, myeloma, beat ISS staging p<0.001); recurrent DL beats Cox modestly.
- Pharmacogenomics: genotype effects handled best by alleles → activity scores → enzyme kinetics
  inside mechanistic models, NOT end-to-end ML (CYP2D6 GDDI network, PMC12087690: GMFE 1.38–1.56).

**Takeaway:** exactly matches our architecture choice (doc/28 genotype → activity score → clearance
multiplier). ML belongs at the edges as pretrained encoders (ChemBERTa/gin embeddings → parameter
priors), never as the core dynamics.

---

## 3 — Gap Analysis: Current System vs Survey

| Capability | Current state (docs 27–29) | SOTA benchmark | Gap size |
|---|---|---|---|
| Drug disposition | 6-compartment well-stirred PBPK, µM-stateful (doc/29 fix D1) | Simcyp-class: 15+ tissues, transporters, enzyme turnover, induction | Medium |
| Parameter sourcing | user-supplied, literature-anchored | ML-from-SMILES priors (PKSmart/DeepCt class) | Small (wrapper + uncertainty flag) |
| PD primitives | Hill/E_max per target | mass-action binding, competitive synapse, TMDD | Medium |
| DDI | curated fold-change rules with ramps (doc/28 §9) | mechanism-based (TDI, enzyme turnover) | Small-Medium |
| Physiology | static scaled organ volumes + flows | HumMod 5k-var dynamic regulation (baroreflex, RAAS, renal-fluid) | Large |
| Endocrine axes | implicit in disease modifiers | explicit 3–10-state axis models w/ cross-talk | Small-Medium |
| Disease dynamics | logistic severity + staging rubrics | bespoke QSP per disease (RA/NSCLC/DMD exemplars) | Large (per disease) |
| Immune system | CRP/WBC proxy channels | BIS-class ABM; IIRABM 432-param calibrated | Large |
| Tissue repair | half-life relaxation + organ recovery tables | hybrid ABM+PDE per organ | Large (per organ) |
| Genetics | PharmVar star alleles → activity scores | same (validated approach) | ✓ At par |
| Labs/vitals | 34 lab channels + 7 vitals, hourly | same order of magnitude | ✓ At par |
| Recovery | washout, rebound, sequelae, relapse hazard (doc/28 §10) | same classes of models | ✓ At par |
| Validation tooling | benchmark table (manual checks) | VPC, NPDE, fold-error tiers, sensitivity analysis | Small (high value) |
| Population simulation | single patient | ≥100 virtual individuals (EMA norm) | Small-Medium |

---

## 4 — Build vs Integrate Decisions

| Component | Decision | Rationale |
|---|---|---|
| Mechanistic PBPK core | **Build** (extend existing) | Already µM-stateful; Simcyp is closed; PK-Sim is GPL and Python-hostile; our scale is defensible for educational/triage use |
| SMILES → PK parameter priors | **Integrate** pretrained models (PKSmart/DeepCt weights) behind an uncertainty-flagged auto-fill API | Structure-only accuracy is ±2-fold; never silently override user-supplied constants |
| Whole-body regulation | **Port** HumMod/QCP variable graph as an optional dynamic-physiology backend behind the existing `HumanPhysiology` interface | XML definitions are public; re-derivation is wasted effort; keeps current fast path intact |
| Endocrine axes | **Build** directly in DSL/Python (Bergman/Karin/HPT 3–10-state models) | Trivial math; high realism-per-LOC ratio; needed for steroid/opioid rebound credibility (extends doc/28 §10 rebound tables into mechanisms) |
| Immune system | **Build** BIS-granularity rule-based ABM (macrophage/neutrophil/T-cell/cytokine skeleton) as a new `human/immune.py`, initially driving CRP/WBC instead of proxy formulas | Agent+rule+compartment maps naturally onto the Helix DSL; adaptive immunity out of scope until innate layer validates |
| QSP drug MoA | **Extend** PD module: mass-action binding + TMDD + competitive antagonism as new PD block types; keep Hill for simple cases | Directly increases drug-space coverage without new infrastructure |
| Tissue repair | **Defer** ABM+PDE organs; keep half-life ladder; add fibrosis end-state flags driven by injury integrals | Per-organ campaigns are person-year projects with thin validation data |
| Validation suite | **Build** first-class: VPC plots, fold-error tables vs 0.8–1.25/2-fold tiers, one-at-a-time sensitivity, virtual-population runner (n≥100) | Required by EMA/FDA norms; absent from most academic tools; differentiator |
| Disease breadth | **Author modules** per-disease in DSL, seeded from open models (Palumbo T2D, RA npj 2024, QSP-IO) | Matches field reality: frameworks generalize, content doesn't |

---

## 5 — Proposed Architecture Additions

### 5.1 — New/changed modules

| # | Module Path | Purpose | Est. LOC |
|---|---|---|---|
| 1 | `human/param_predict.py` | SMILES → {CL, Vd, fu, t½} priors via wrapped pretrained models; returns `(value, confidence)`; refuses to override user PK blocks unless `allow_ml_defaults=true` | 300–800 |
| 2 | `human/endocrine.py` | HPA/HPT/insulin-glucose axis ODEs; outputs cortisol, FT3/FT4/TSH, insulin sensitivity index; consumed by recovery rebound (replacing table lookup) and by vitals temperature/stress drives | 400–700 |
| 3 | `human/immune.py` | Rule-based innate ABM: agents {macrophage, neutrophil, DC, T-cell}, signals {TNF, IL-1, IL-6, IL-10}, 3 compartments (tissue/blood/lymph); emits IL-6 → CRP driver and WBC counts, replacing proxy channels | 1,500–3,000 |
| 4 | `human/qsp_binding.py` | Mass-action receptor-ligand + TMDD + competitive antagonist PD block types (`pd { kind: mass_action ... }`) | 500–900 |
| 5 | `human/physio_backend.py` | Optional HumMod-variable-graph backend: XML loader + solver stepping the 5k-var graph; adapts outputs onto `HumanPhysiology` fields (CO, flows, GFR, volumes) | 5,000–15,000 (port) |
| 6 | `human/validation.py` | Virtual-population runner (demographic sampling from traits distributions), VPC construction, observed-vs-predicted fold-error tier report, OAT sensitivity sweep | 800–1,500 |
| 7 | `human/data/pgx.json` (extend) | CYP2C19 *4/*5/*6/*7/*8, CYP2B6, DPYD, UGT1A1, SLCO1B1, TPMT/NUDT15 — completes CPIC level-A gene coverage | +150 rows |

### 5.2 — DSL surface

```helix
#sim kind=virtual_patient observation=180d dt=1h population=100   # virtual population mode

#traits {
  ...
}

#drug {
  name: "novel_compound_x"
  smiles: "..."
  pk { predict_from_structure: true }        # param_predict.py fills CL/Vd/fu/t½ with uncertainty flags
  pd {
    kind: mass_action                        # new QSP-style binding block
    target: PD1
    kd_nM: 12.4
    competitors: [nivolumab]
  }
}

#output {
  tracks: [...]
  validation: [vpc, fold_error, sensitivity_oat]   # validation suite hooks
}
```

### 5.3 — Integration order (dependency-safe)

```
Phase 1 (validation + genetics completion)     — no engine changes, pure observability
  validation.py, pgx.json extension
Phase 2 (parameter automation)                 — input-side only
  param_predict.py, DSL pk{predict_from_structure}
Phase 3 (mechanistic depth, cheap wins)        — extends existing solvers
  endocrine.py (axes feed recovery/vitals), qsp_binding.py PD block types
Phase 4 (immune ABM)                           — replaces CRP/WBC proxies incrementally
  immune.py behind ClinicalLabModel driver interface
Phase 5 (dynamic physiology backend)           — largest, optional, behind interface flag
  physio_backend.py (HumMod port)
```

---

## 6 — Complexity and Compute Estimates (from survey)

| Phase | Effort | Runtime impact |
|---|---|---|
| 1 | 1–2 weeks | none (report generation only) |
| 2 | 1–2 weeks | none (inference is ms-scale) |
| 3 | 2–4 weeks | negligible (≤20 extra states, RK45 handles trivially) |
| 4 | 1–2 months | ABM adds minutes per run at tissue-agent counts × population n; acceptable for n≥100 populations if agents capped (~10³–10⁴) |
| 5 | 3–6 months | HumMod-class graphs simulate months in seconds (Hester 2011); bottleneck is port fidelity + coupling, not speed |

Compute was never the constraint anywhere in the survey — every cited model runs desktop-scale.
Parameterization/calibration data and validation evidence are the true costs.

---

## 7 — Hard Limits (carry forward verbatim into any roadmap discussion)

1. **Integration itself is novel.** No system couples dynamic whole-body regulation + GEM metabolism
   + PBPK/QSP + labs. Every successful project is fit-for-purpose and narrow; expect coupling bugs
   (cf. doc/29 defect class) at every new interface.
2. **"Any drug" fails without evidence.** Novel MoAs require measured binding/enzymology; regulatory
   law enforces the evidence chain regardless of simulator quality.
3. **"Any genetic variation"** works today only through pharmacogene activity-score mappings
   (our approach, validated by the field) plus per-gene disease-risk multipliers. Disease-variant
   effects need bespoke pathway knowledge per variant class.
4. **"Any disease" is an authoring ecosystem**, not shipped content. The language surface is the
   product; modules arrive per-disease with their own calibration campaigns.
5. **Cold-start ML collapses.** Structure-only PK prediction is triage-grade; response prediction on
   unseen scaffolds fails (TransCDR et al.). Use ML for defaults and priors with visible uncertainty,
   never as silent ground truth.

---

## 8 — Key References (consolidated)

**PBPK/PK prediction:** Lombardo *JMC* 2016–18; PKSmart doi:10.1186/s13321-025-01066-5; DeepCt
(GitHub 2024); MMPK (GitHub 2025); PubMed 39589671; PAGE 2026 AI→PBPK abstract; Simcyp tutorial
PMC9286711; EMA/CHMP/458101/2016.

**Whole-body physiology:** Guyton/Coleman/Granger *ARP* 1972; Hester et al. *Front Physiol*
doi:10.3389/fphys.2011.00012; Liang *PCB* e1002571 (2012); Guyton 2008 CellML workspace.

**QSP/IO:** Sové *CPT PSP* PMC7499194 (2020); Milberg *Sci Rep* 9:8169 (2019); Wang *Pharmaceutics*
17:238 (2024); TNBC *Sci Adv* adg0289 (2023); bispecific platform *Front Pharmacol* (2025).

**Immune:** Folcik *TBiomedM* 4:39 (2007)/8:1 (2011); An *CCM* (2004); Cockrell *Front Physiol*
12:662845 (2021); COMMBINI *Front Immunol* 14:1231329 (2023); COPD hybrid *PCB* e1008413 (2021).

**Endocrine:** Bergman *ABE* 1981; Karin *MSB* 16:e9510 (2020) + *Nat Comms* s41467-025-65924-4
(2025); HPT *Front Endocrinol* 13:825107 (2022); Walker ultradian ACTH/cortisol.

**Repair:** Rouillard/Holmes *Front Physiol* 10:1481 (2019); wound hybrid PMC6561509 (2019);
CPM senescence *PCB* e1012298 (2024); muscle *eLife* 12:e91924 (2024); Vermolen/Boon *BMMB* (2020).

**Validation/regulatory:** Zhao *CTS* PMC8592512 (2021); Zhang *Pharmaceutics* 17:1413 (2025);
NPDE PMC7293575; ENRICHMENT playbook (2024).

**Digital twins:** Levine *JMD* (2014); ENRICHMENT (FDA, 2024); Unlearn.AI PROCOVA™ publications.

**ML-biology:** GDSC/CCLE fusion-model family (DeepCDR/GraphDRP/GPDRP/TransCDR/GCNPath/DBDNMF);
TrajSurv ICML 2025 (PMLR v298); SCOPE *npj Dig Med* (2024); CYP2D6 GDDI PMC12087690.

**Disease exemplars:** Palumbo *PLoS ONE* e0192472 (2018); de Gaetano *PCB* e1010914 (2023);
RA Vpop *npj Sys Biol Appl* (2024); cutaneous lupus *Cell Rep Med* (2025).
