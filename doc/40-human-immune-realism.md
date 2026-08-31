# doc/40 — Human Immune System Simulation Realism Upgrade Plan (Literature-Grounded)

> **Status:** Phase A implemented (G1/G4/G8/G9/G11) · Phase B–E plan · 2026-08-31 · baseline 2026.8.5
>
> **Depends on:** doc/31 §2.4/§5.1 (immune ABM design), doc/37 (validity framework), doc/39 (performance budget), doc/33 (validation lineage)
>
> **Goal:** Replace the current population-ODE approximation of the innate immune system
> (`plugins/human/immune.py`) with the multi-tier, literature-grounded innate + adaptive
> model specified in doc/31 — organized per the requested structure **Content /
> Method / Results / Conclusions** of each reference work, with every
> upgrade traced to a concrete paper, a concrete model formulation, an expected quantitative
> outcome, and a validation gate.

---

## 1 — Executive Summary

doc/31 §2.4 surveyed agent-based and ODE immune models (BIS, IIRABM, COMMBINI, QSP) and
§5.1 #3 mandated a "BIS-granularity rule-based ABM … initially driving CRP/WBC instead of
proxy formulas". What shipped in `immune.py` is a **population-level forward-Euler ODE**
(338 lines) that reuses the doc/31 vocabulary (compartments, agents, PAMP/DAMP in the
docstring) without implementing any of its structure. It is a faithful *skeleton* of
IL-6→CRP and neutrophil/WBC coupling, but it lacks: adaptive immunity (antibody,
CD4/CD8/Th/Tc, memory), complement, type I IFN, NK/mast/eosinophil/basophil, chemokines,
APC/MHC priming, a real acute-phase panel (CRP is the only output), bone-marrow
granulopoiesis (Friberg-style transit compartments), vaccination, circadian HPA coupling,
and inter-individual (virtual-population) variability.

This document turns the literature — papers that doc/31 lists plus the primary sources —
into a **phased, falsifiable plan** (Phases A–E). Each phase claims a specific realism gain
and a specific validation gate under doc/37's Biological-Accuracy framework, and each
respects doc/39's performance budget (adaptive/vectorized ODE backbone, cohort-level numpy).

---

## 2 — Content Audit: What `immune.py` Actually Is

### 2.1 Module structure and values (`plugins/human/immune.py`, 338 lines)

| Component | Line | Reality |
|---|---|---|
| `CytokinePool` | `:47` | 4 cytokines, forward Euler `step` `:75`; baselines TNF-α 5.0, IL-1β 2.0, IL-6 1.0, IL-10 5.0 pg/mL `:55-58` |
| Cytokine half-lives | `:61-64` | 0.5 / 1.0 / 1.5 / 2.0 h; rate constants recomputed every call (`k = ln2/T`, `:77-80`) |
| Cytokine production | `:67-70` | 10 / 8 / 12 / 6 pg/mL/h, scaled by scalar `pathogen_signal` (0–1, `:83`) |
| IL-10 coupling | `:91` | anti-stim = stim + 0.1·(TNF+IL-1)/10 — weak, static gain; no IL-10 tissue "mass", no negative feedback loop shaping |
| `ImmuneCellPopulation` | `:108` | 5 pools; linear production − clearance with floors `:138-176`; mobilisation `gcsf_sensitivity·(IL-6−1)/10` `:146`; no maturation transit, no margination, no consumption (no immune-cell killing of pathogens) |
| `InnateImmuneModel` | `:190` | couples pools; cortisol suppression `:235-239`, immunosuppression `:242-246`, anti-inflammatory (JAK/NSAID-style) `:250-253`, `infection_severity`+`autoimmune_activation` scale `pathogen_signal` `:225-227` |
| `CRPDriver` | `:283` | IL-6→CRP: production linear above threshold 0.3, clearance 0.036/h (~19 h), ceiling 200 mg/L `:292-296`. Docstring claims a sigmoid; code is piecewise-linear `:298-306` |
| `create_immune_model` | `:314` | factory; only knobs = infection, autoimmune, cortisol>20 µg/dL suppression, immunosuppression |

Key audit finding: the module docstring (`:21-25`) cites **BIS-class (Candiani et al. 2024,
432-parameter innate immune ABM)** and **IIRABM (Marina et al. 2024)** and claims three
compartments, cell agents, tissue/blood/lymphoid tracking — **none of which exist in code**.
(doc/31 `:98` attributes the 432-param GA-calibrated IIRABM to Cockrell & An, *Front. Physiol.*
12:662845, 2021; the docstring's attribution is loose.) The model is a valid ODE *core*, but
every agent/space claim is aspirational.

### 2.2 Downstream consumption (what must keep working)

| Consumer | Location |
|---|---|
| Lazy init, disease-severity feed-in | `virtual_patient.py:953-982`; base severity attrs `:981-982` |
| Per-tick cortisol suppression + `immune.step(dt_h)` | `virtual_patient.py:1309-1321` |
| CRP driver step | `virtual_patient.py:1348` |
| Bio-score/state channels (IL-6, TNF) | `virtual_patient.py:1227-1228`, `:1393-1394`, `:1538-1539` |
| Autoimmune / infection modulation | `virtual_patient.py:1698`, `:1767` |
| Result channels (il6, tnf, neutrophils) | `virtual_patient.py:1927-1929` |
| WBC numeric seam | `virtual_patient.py:840` |
| Fever coupling | `virtual_patient.py:1567-1570` (via `emergent_complexity` `fever_c`) |
| Other human modules importing immune | `disease.py`, `disease_ode_models.py`, `disease_progression.py`, `organ_crosstalk.py`, `microbiome.py`, `emergent_complexity.py` |

Any upgrade must extend `InnateImmuneModel`/`CRPDriver` **behind the same API** during
Phases A–B (drop-in), and only rewire consumers in Phase C when the new channels (fever,
APR panel, vaccination) go live.

---

## 3 — Gap Matrix: Biology Present vs Missing

| # | Biological feature | State in `immune.py` | Needed for (doc/40 target) |
|---|---|---|---|
| G1 | Type I IFN (IFN-α/β) antiviral loop | **implemented** — `IFNPool` (Hill activation + antiviral suppression) | viral kinetics realism (flu bursts, HSV) |
| G2 | Adaptive immunity: naive/effector/memory CD4/CD8, B | single undifferentiated `t_cells` pool (`:116`) | vaccination, memory, secondary response |
| G3 | Antibody dynamics, IgM→IgG class switch, waning | missing | vaccine trials, serology channels |
| G4 | Bone-marrow granulopoiesis / neutropenia | **implemented** — Friberg 4-compartment transit chain (`friberg_*` fields) | chemo myelosuppression plots (doc/32), recovery kinetics |
| G5 | Complement (C3/C5 opsonization, MAC, anaphylatoxins) | missing | bacterial opsonization, anti-C5 PD (eculizumab) |
| G6 | NK / mast / eosinophil / basophil | missing | anaphylaxis (existing disease text claims mast/IgE), tumor surveillance |
| G7 | APC/MHC antigen presentation → T-cell priming delay | missing | vaccines (G2/G3), autoimmunity onset dynamics |
| G8 | Acute-phase panel (CRP, SAA, ferritin, PCT, fibrinogen) | **implemented** — `CRPDriver` v2 APR panel fields | sepsis severity scoring (PCT), chronic disease |
| G9 | IL-6/CRP kinetics fidelity (lag, sigmoid, wide dynamic range) | **implemented** — `CRPDriver` v2 saturating Hill + ~6 h lag, ceiling 1000 mg/L | vs Sproston/Ashworth ranges (<1→1000 µg/ml) |
| G10 | Chemokines / tissue-vs-blood pseudo-compartments | none (docstring claims 3 spaces) | tissue inflammation vs circulating WBC mismatch |
| G11 | Circadian HPA/cortisol coupling (time-resolved) | **implemented** — `circadian_amplitude`/`circadian_phase_h` sine modulation (default 0) | stress/inflammation daily rhythm |
| G12 | Vaccination stimulus (antigen, dose, boosting) | no | vaccine-response simulation (L9 below) |
| G13 | Inter-individual variability (virtual population) | none — fixed init | probability-of-response cohorts |
| G14 | Immune exhaustion / checkpoints (PD-1/PD-L1) | none | PD-target pharmacology (already a PD target in the project) |
| G15 | Cross-scale agents/spaces | claimed only | see §4 L1/L2 for how far to go |

---

## 4 — Literature Evidence Base, by Content / Method / Results / Conclusions

The project's own prior surveys (doc/31 §2.4 table) plus primary literature. Each entry
records **Content** (systems modelled), **Method** (formulation used by the authors), **Results**
(quantitative outcomes), **Conclusions** (what our plan adopts or rejects).

### L1 — Basic Immune Simulator (BIS), Folcik/An/Orosz, *Theor. Biol. Med. Model.* 4:39 (2007, 8:1 2011)

- **Content:** innate (macrophage, neutrophil, DC, mast, eosinophil, NK) + adaptive
  (B, Th, Tc, Tr; IgG/IgM/IgA/IgE; complement) agents moving through blood/tissue/lymph.
- **Method:** agent-based (RepastJ), cell-as-agent, diffusing cytokines; rules from ~146 papers.
- **Results:** reproduces qualitatively correct immune timeline (innate peak → adaptive peak →
  resolution), memory anamnesis, dose-dependence of infection outcomes.
- **Conclusions:** the *rule skeleton* is implementable (doc/31 `:102`), but full spatial ABM is a
  person-year. Adopt: its **entity decomposition** (agents+cells+signals) as the taxonomy for
  our adaptive layer; reject: its spatial cost in the near term.

### L2 — Innate Immune Response ABM (IIRABM) lineage, An 2004 (*Crit. Care Med.* 32:2050); Cockrell & An 2021 (*Front. Physiol.* 12:662845)

- **Content:** macrophage/neutrophil/tissue-cytokine network for sepsis; prior-knowledge rules.
- **Method:** ABM; Cockrell 2021 adds **GA calibration of 432 continuous free parameters** to fit
  burn-patient cytokine time series.
- **Results:** An 2004 qualitatively reproduced the *failed* anti-TNF and anti-IL-1 sepsis trials
  (networks compensate); calibrated IIRABM recovers clinical cytokine trajectories.
- **Conclusions:** (a) cytokine networks are **redundant/counterbalanced** — no cure from blocking one
  cytokine; our IL-10 static gain must become a real negative-feedback loop to ever display
  this; (b) GA/ABC calibration (doc/37 `ParameterFitter`) is the right estimation tool.

### L3 — Reduced ODE models of the acute inflammatory response, Reynolds et al. 2006 (*J Theor Biol* 242:220-236); Day et al. 2006 (part II, 242:237-256); Chow et al. 2005 (*Shock* 24:74-84)

- **Content:** P/DAMP + pro-inflammatory (P) + anti-inflammatory (A) mediator balance.
- **Method:** minimal 3–4 ODEs, bistability, dimensional analysis; part II adds tolerance/potentiation.
- **Results:** two stable phenotypes (resolution vs persistence) emerge from *parameter* changes;
  explains chronic inflammation and sepsis death vs survival, and drug-effect paradoxes.
- **Conclusions:** an ODE core can exhibit realistic nonlinear behavior **without ABM**. This is the
  mathematical template for upgrading `CytokinePool` from linear+gain to saturating (Hill)
  production with a genuine A-compartment. (Also cf. 4-ODE sepsis survival predictor, *Front
  Med* 2021; separated P/A/bacteria/damage.)

### L4 — Semimechanistic (Friberg) granulopoiesis / myelosuppression, Friberg et al. 2002 (*JCO* 20:4713); 2003 (*Invest New Drugs* 21:183); QSP consolidations (e.g. Craig 2017)

- **Content:** proliferating → maturation transit compartments (with mean transit time) →
  circulating neutrophils; feedback via circulating cell count driving proliferation; drug
  kills proliferating cells; G-CSF naturally supported.
- **Method:** ODE compartment chain (4–5 transit pools), feedback Hill; fit to clinical ANC nadir
  curves across many chemo drugs with consistent parameters.
- **Results:** gold-standard reproduction of delayed onset, ANC nadir ≈7–10 days, recovery
  overshoot; parameter consistency across drugs.
- **Conclusions:** **G4 fix.** Replaces the linear production/floor in `ImmuneCellPopulation` (`:126-149`)
  with a transit-chain + ANC feedback. Directly serves existing chemo/myelosuppression content
  (doc/32) and adds neutropenia states. Cost is ~5–8 ODEs — trivial inside doc/39's O10.

### L5 — Within-host influenza immune model, Pawelek/Huynh/Quinlivan/Cullinane/Rong/Perelson 2012 (*PLoS Comput Biol* 8(6):e1002588)

- **Content:** target-cell epithelial infection + **type I IFN** + innate (NK-like) + **CD4/CD8 T**
  + **antibody** (dual effector arms).
- **Method:** ODEs, parameterized from mouse influenza; effectors act with time delays/loss.
- **Results:** early viral decline driven by IFN/innate arm, late clearance by CD8 effectors;
  antibody matters for recurrence/reinfection; predicts which arm dominates per scenario.
- **Conclusions:** a *complete but compact* immune equation set exists and is validated — use it as the
  canonical formulation for both **G1 (IFN)** and **G2/G3 (adaptive)** rather than inventing
  kinetics. (Precursor: Baccam et al. *J Virol* 80:7590, 2006 target-cell model.)

### L6 — COVID-19 immune QSP with virtual population, *npj Systems Biol & Appl.* 11 (2023) s41540-023-00269-6

- **Content:** within-host COVID: virus, IFN, CD8 activation (log-linear), innate immune;
  virtual population from parameter sampling.
- **Method:** QSP ODE + population sampling fitting real patient trajectories; log-linear T-cell
  activation avoids sharp thresholds.
- **Results:** population-level heterogeneity explains individual courses; intervention
  (interferon timing) effects differ by virtual-patient subgroup.
- **Conclusions:** the **virtual-population methodology (G13)** — sample parameters, keep log-linear
  activation over threshold switches — is directly portable to our cohort runs and ties into
  doc/39 O2/O8 vectorization.

### L7 — Complement pathway computational models, Zewde & Morikis 2018 (*PLoS ONE* 13(6):e0198644); Bansal et al. 2022 (*Front Pharmacol* 13:855743)

- **Content:** C3/C5 convertase networks; alternative + classical/lectin; in vivo plasma+surface.
- **Method:** large ODE/Gillespie networks (tens–hundreds of states, 142+ rate constants);
  inhibition with compstatin/eculizumab-type agents tested *in silico*.
- **Results:** predicts C3 fragment deposition, MAC formation thresholds, and inhibitor dose-effect
  curves consistent with assays.
- **Conclusions:** full network is overkill (state explosion); adopt a **reduced 6–10 ODE complement
  module** (G5: C3→C3b/C3a, C5→C5a/C5b-9, regulators, anti-C5 drug) with the partial-cleavage
  formalism from Zewde/Bansal — enough to drive opsonization and fever (C3a/C5a) without the
  142-parameter burden.

### L8 — Vaccine-induced antibody dynamics: consensus model, *Front. Immunol.* 16:1596518 (2025)

- **Content:** two-dose vaccine schedule → memory B + long-lived plasma cells (LLPC), IgM→IgG
  class switching, waning.
- **Method:** consensus linear ODE (chain-trick compartment model), fit across vaccines
  (incl. mRNA), Age/adjuvant terms.
- **Results:** resembles observed sigmoid rise → peak ~2–4 wks → biphasic waning (short vs long-lived
  plasma); second dose boost amplitude ruled by memory-to-PC conversion.
- **Conclusions:** **G2/G3/G7/G12 unified formulation** for the vaccination feature: an 8–10 compartment
  linear chain, safe inside existing runtime and physiologically validated. This is the smallest
  unit that makes "vaccine response" a real output of the platform.

### L9 — IL-6/CRP acute-phase kinetics and functions, Sproston & Ashworth 2018 (*Front Immunol* 9:754); Volanakis 2001; Pepys & Hirschfield 2003 (already cited in `immune.py`)

- **Content:** IL-6-driven hepatic CRP; broad dynamic range **<1 → 1000 µg/ml**; ~6 h synthesis
  onset after stimulus, ~19 h half-life; CRP is delayed/compensatory, also IL-1β/IL-17.
- **Method:** clinical measurements + genotype/kinetics analysis.
- **Results:** CRP peak lags IL-6 by ~18–48 h; very high ranges (>300 mg/L) occur in severe sepsis;
  chronic low-level elevation tracks atherosclerosis/aging.
- **Conclusions:** **G8/G9 fix.** Current `CRPDriver` linear-IL6→linear-CRP with hard cap 200 and no
  lag cannot reach severe-sepsis 400–1000 or show the lag. Replace with: saturating production,
  IL-6 delay compartment, widened cap, optional SAA/ferritin/PCT analogs (frontier: personal
  baselines per DBS longitudinal acute-phase studies).

### L10 — Anti-IL-6 pharmacology QSP, *Front. Immunol.* 13:919489 (2022)

- **Content:** tocilizumab/siltuximab PD on IL-6→CRP, with ~10% lymph-node/tissue penetration.
- **Method:** target-mediated drug disposition + IL-6/CRP circuit QSP, calibrated to sIL-6R.
- **Results:** reproduces CRP drop kinetics under anti-IL-6 and rebound on washout.
- **Conclusions:** our existing `anti_inflammatory`/`immunosuppression` pathway (`:250-253`) should
  become **biologic-aware**: when PD-IL-6 antagonists are on board, route them through a
  TMDD-style reduction rather than an arbitrary 0.9 factor (G14/biologicals scope).

### L11 — Type I IFN and innate antiviral kinetics, in vitro HSV-1 model (PMC1664656, 2006); Howat et al. spatial-stochastic IFN (J.)

- **Content:** cell-virus-IFN circuit; paracrine IFN-α/β establishes antiviral state with
  characteristic biphasic/multi-staged viral curves.
- **Method:** within-host ODE/stochastic with IFN autocrine-paracrine loop, delay, priming.
- **Results:** predicts early exponential viral rise slowed/aborted by IFN, and tissue-dependent
  spread restriction.
- **Conclusions:** supplies the **EGF-ish autocrine shape** for the G1 component with physiologically
  cited rate constants (quantitates priming; pairs with L5's IFN block).

### L12 — Systems-biology-of-inflammation reviews, Vodovotz *et al.* (2006–2017); Smith & Weaver *Curr Opin Syst Biol* (2024)

- **Content:** consensus that acute inflammation is a **dynamical system with redundancies**,
  resolvable at ODE *and* ABM granularity; influenza-control lessons on nonlinearity.
- **Method / Results / Conclusions:** hybrid ODE+ABM tiers are the community norm; nonlinear (threshold/Hill)
  interactions are the *sine qua non* for clinically relevant emergent behavior. Adopt as the
  overall design rubric for Phases A–C (ODE core ∪ {agents later}).

---

## 5 — Content / Method / Results / Conclusions Synthesis: From Literature to Implementation

Mapping each gap of §3 to its evidence and to a concrete planned intervention. The four columns
answer: **Content** — what we will build; **Method** — modeling method from the cited paper(s);
**Results** — expected quantitative behavior; **Conclusions** — the validation gate proving it.

| Gap | Content (build) | Method (from) | Results (expected) | Conclusions (gate) |
|---|---|---|---|---|
| G1+G11 | Type-I IFN block in `CytokinePool`; saturating Hill production; genuine IL-10 negative-feedback A-compartment | L3 (P/A/damage ODE), L5, L11 | LPS/endotoxin trains with resolution vs persistence bistability; IL-10 tracks with lag; IFN abortive peak for flu | Accepted if two distinct outcome branches emerge on parameter axis (doc/37 spot-check against L3 fits) |
| G8+G9 | `CRPDriver` v2: IL-6 delay compartment + saturating production + widened range (<1→1000), optional SAA/ferritin/PCT | L9 | CRP peak lag ~18–48 h behind IL-6; severe sepsis ≥400 mg/L | Golden: CRP time-course within L9 clinical bands |
| G4 | Friberg transit granulopoiesis replacing neutrophil ODE + floors | L4 | ANC nadir ≈7–10 d post-chemo, recovery overshoot | Golden: neutropenia curve vs published chemo fits |
| G2+G3+G7+G12 | Naive→effector→memory CD4/CD8 + B; antibody IgM→IgG chain; APC priming delay; vaccination stimulus | L5 (effector arms), L8 (8–10 cl chain) | Two-dose kinetics: sigmoid rise, peak 2–4 wks, biphasic waning, memory anamnesis on rechallenge | Golden: two-dose Ab curve within L8 consensus bands |
| G5 | Reduced 6–10 ODE complement module (C3/C3a/C3b, C5/C5a/C5b-9, regulators, anti-C5) | L7 | C3-fragment deposition & MAC thresholds; dose-effect of anti-C5 | Golden: inhibitor dose–response consistent w/ L7 in silico |
| G6 | Minimal NK/mast/eosinophil pools + histamine/IgE anaphylaxis coupling | L1 (entity set), doc/31 §2.4 | Anaphylaxis pattern (rapid systemic mediator release) existing text claims become simulated | Golden: anaphylaxis flag timelines |
| G10 | Tissue vs blood pseudo-compartments for cytokines/cells (rewire docstring claim, 3 spaces) | L1 (compartment taxonomy), L2 | Divergent tissue-vs-circulating WBC during infection (e.g., neutropenia w/ tissue neutrophilia) | Acceptance: separate tissue/blood series exposed in results |
| G11 | Circadian cortisol modulation of suppression term | existing HPA (`virtual_patient.py:1309`) × diurnal curve | daily phase in cytokine production | Acceptance: cortisol/IL-6 phase relationship, diurnal amplitude |
| G13 | Virtual-population immune params (baselines, rates sampled log-uniform) | L6, L13 (personal baselines) | P(response) distributions, subgroup differential drug effects | Gate: cohort-level validation via doc/37; perf via doc/39 O2/O8 |
| G14 | PD-1/PD-L1 checkpoint toggle when PD is the target | L5 (effector loss), doc/31 immune targets | Exhaustion-induced relapse pattern under sustained antigen | Gate: PD-drug curves vs expected immune-brake phenotype |
| G15 | Optional ABM later (beyond C) | L1, L2 | (stretch) | Explicitly deferred to post-D review |

---

## 6 — Phased Implementation Plan

- **Phase A — Innate fidelity, zero API break (2–3 wk).** G8/G9 (CRPDriver v2 + APR panel),
  G1/G11 (IFN + Hill + IL-10 loop), G4 (Friberg chain behind `ImmuneCellPopulation` API).
  **No consumer edits**; `virtual_patient.py:1321/1348` clocks unchanged. **Co-runs with
  doc/39 Phase 2** (adaptive/vectorized ODE backbone, O1/O2/O10). Gate: 75/75 + new LPS /
  neutropenia goldens inside clinical bands; perf unchanged or better.

  **STATUS: implemented (2026.8.5).** `IFNPool` (G1), `CRPDriver` v2 saturating Hill +
  ~6 h lag + APR panel `saa_mg_l`/`ferritin_ng_ml`/`pct_ng_ml`/`fibrinogen_g_l` (G8/G9),
  Friberg 4-compartment transit chain behind `ImmuneCellPopulation.step` (G4, K normalized to
  healthy ANC + `friberg_drug_kill` for myelosuppression), circadian cortisol sine modulation
  with `circadian_amplitude` default 0 (G11). All added behind the same `step(dt_h, il6, tnf)`
  / `step(dt_h, il6_pg_ml)` APIs with backward-compatible defaults; the O2 vectorized
  `cohort_immune_step` mirrors the new equations term-for-term and `run_cohort` O9 carries the
  new state. Zero consumer edits (`virtual_patient.py`, `simulation.py` untouched). New tests
  in `tests/test_disease_ode_modules.py` (IFNPool, CRPDriverV2, FribergTransit,
  ImmuneCircadianCortisol) all green.
- **Phase B — Adaptive immunity + vaccination (3–4 wk).** G2/G3/G7/G12. New `adaptive.py`
  module (naive/effector/memory chains + antibody + APC priming delay) plugged into
  `InnateImmuneModel` without touching existing output channels. Vaccination exposed via
  `immune_configs` ("vaccine": schedule, dose, antigenicity). Gate: two-dose Ab golden +
  rechallenge anamnesis; `sim` suite labeled benchmarks stay green.
- **Phase C — System-level wiring (3 wk).** G5 (complement module), G6, G10
  (tissue/blood spaces), G11 (circadian), fever set-point hookup to `emergent_complexity`
  `fever_c`. Consumers (Labs WBC, `:840`; bio-score `:1393-1540`) rewire to new channels.
  Gate: new result channels exposed + anaphylaxis/CRP/sepsis-feature goldens.
- **Phase D — Population & pharmacology (2–3 wk).** G13 virtual-population sampling,
  G14 PD toggle, L10 biologic-aware anti-IL-6 pathway. Gate: cohort probability-of-response
  output on virtual patient runs; PD tests extended.
- **Phase E — Validation & conditioning (continuous).** Every new golden enters
  `validation/` behind doc/37's Biological-Accuracy framework; regression suite keeps 75+;
  goldens carry the same SHA256 rigor; cherry-picked references updated (CRP ranges, ANC
  nadir days, Ab waning half-life). Determinism: seeded sampling retained (doc/39 §5).

**Budget:** ~10–13 wk part-time total; Phase A,B = ~2/3 of the realism value for ~1/2 the cost.
ABM/G15 deliberately deferred to keep runtime and doc/39's ≤3×-native target intact.

---

## 7 — Non-Goals and Risks

- **Non-goals:** full spatial ABM (G15, deferred), full 142-param complement (L7, overkill),
  432-param GA re-fit for every patient (IIRABM-style Bayesian fits are a doc/37 batch job,
  not a runtime loop), PD-1 space expansion beyond the existing PD-target toggle.
- **Risks:** (1) goldens churn — every Phase A/B change is gated to *add* goldens, not
  silently rewrite; (2) perf regression from +30–80 ODEs — mitigated by doc/39 O10
  (adaptive/vectorized backbone) and O2 (cohort numpy); (3) docstring-vs-code drift — Phase
  C rewires the compartment claims to reality and keeps them honest; (4) calibration data
  sparsity for IFN/complement — mitigated by adopting published rate constants (L5, L7, L11)
  rather than free-fitting from scratch.

---

## 8 — References

1. Folcik VA, An G, Orosz CG. The Basic Immune Simulator. *Theor Biol Med Model* 4:39 (2007); 8:1 (2011).
2. An G. In-silico experiments of existing and hypothetical cytokine-directed clinical trials. *Crit Care Med* 32:2050 (2004); Cockrell C, An G. IIRABM GA calibration. *Front Physiol* 12:662845 (2021).
3. Reynolds A, Rubin J, Clermont G, Day J, Vodovotz Y, Ermentrout GB. Reduced model of the acute inflammatory response I. *J Theor Biol* 242:220 (2006); Day J, et al. II. *J Theor Biol* 242:237 (2006); Chow CC, et al. *Shock* 24:74 (2005).
4. Friberg LE, et al. Model of chemo-induced myelosuppression with parameter consistency across drugs. *J Clin Oncol* 20:4713 (2002); Friberg LE, Karlsson MO. *Invest New Drugs* 21:183 (2003); QSP myelosuppression reviews incl. Craig (2017).
5. Pawelek KA, Huynh GT, Quinlivan M, Cullinane A, Rong L, Perelson AS. Within-host influenza with immune responses. *PLoS Comput Biol* 8(6):e1002588 (2012); Baccam P, et al. target-cell model. *J Virol* 80:7590 (2006).
6. COVID-19 immune QSP + virtual population. *npj Sys Biol Appl* s41540-023-00269-6 (2023).
7. Zewde N, Morikis D. Complement pathway computational model. *PLoS ONE* 13(6):e0198644 (2018); Bansal S, et al. *Front Pharmacol* 13:855743 (2022).
8. Consensus model of vaccine-induced antibody dynamics. *Front Immunol* 16:1596518 (2025).
9. Sproston NR, Ashworth JJ. Role of C-reactive protein at sites of inflammation and infection. *Front Immunol* 9:754 (2018); Volanakis JE. *Mol Immunol* 38:189 (2001); Pepys MB, Hirschfield GM. *J Clin Invest* 111:1805 (2003).
10. Anti-IL-6 (tocilizumab/siltuximab) QSP. *Front Immunol* 13:919489 (2022).
11. Type I IFN antiviral kinetics, in vitro HSV-1 model (PMC1664656, 2006); Howat et al. spatial-stochastic IFN modeling (2006).
12. Vodovotz Y, et al. systems biology of acute inflammation reviews (2006–2017); Smith & Weaver. *Curr Opin Syst Biol* (2024).
13. Longitudinal acute-phase protein panel with personal baselines (bioRxiv, 2019).

Category anchors to existing docs: doc/31 §2.4 (survey), §5.1 #3 (module mandate),
§4.5 (perf budget); doc/33 (validation lineage); doc/37 (Biological-Accuracy goldens);
doc/39 (performance budget this plan runs inside of); doc/32 (chemo/PK context for G4);
doc/27/28/29 (virtual patient context for G13).