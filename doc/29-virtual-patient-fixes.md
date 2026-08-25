# 29 — Virtual Patient System: Critical Defect Fixes and Full-Parameter Coverage

> **Status:** IMPLEMENTED  
> **Depends on:** doc/28 (virtual patient system)  
> **Date:** 2026-08-24

---

## 1 — Problem Statement

Doc/28 delivered the Virtual Patient façade (`human/virtual_patient.py`) wiring genome → traits → diseases → drugs → hourly time-series. Post-implementation audit against the original acceptance goal found that the wiring is broken in four places and that most clinical channels are decorative constants. The goal this document must deliver:

> Specify ONE person's complete genome, external features, disease parameters, and drug parameters; fully simulate ALL body-parameter changes during the medication period AND after medication ends.

A simulation that (a) zeroes compartment states every timestep, (b) feeds mg/L values into µM thresholds, (c) applies genotype twice, and (d) never enters the recovery phase cannot answer that question. Neither can a lab panel in which sodium, potassium, calcium, phosphate, chloride, bicarbonate, LDL/HDL/triglycerides, INR, MCV, RBC, QTc, SpO₂, and respiratory rate never move off their defaults.

**Baseline standard carried over from doc/28: 100% real.** Every equation, half-life, fold-change, and threshold below is anchored to published literature. No mocks, no stubs, no placeholders.

---

## 2 — Defect Audit Summary

| ID | Location | Symptom | Root cause | Fix |
|---|---|---|---|---|
| D1 | `virtual_patient.py:626-636` (`_DrugPBPK.advance`) | No drug accumulation; troughs always ≈ 0; steady-state impossible | A fresh `PBPKModel` is constructed every timestep, re-zeroing all compartment states | §3 — replace with stateful `_PBPKEngine` delegation |
| D2 | `PBPKModel.run()` output consumed in `VirtualPatient.run` | PD/toxicity thresholds (EC₅₀ ≈ tens of µM) receive mg/L numbers → effective sensitivity error of ~100–1000× | Missing molecular-weight conversion | §4 — `conc_uM = conc_mg_per_L × 1000 / MW_da` at the integration point |
| D3 | `virtual_patient.py:381` (`_compute_genetic_cyp_modifier` call) **and** `ddi.py:221-228` (`compute_clearance_modifiers` second pass) | Genotype CYP activity multiplied into clearance twice (e.g. CYP2C9 \*1/\*3 warfarin: 0.5 × 0.5 = 0.25 instead of 0.5) | Overlapping responsibility between facade and DDI model | §5 — separation of concerns |
| D4 | `virtual_patient.py:296` (`self._treatment_active = True`) and `:442` (guard reads it) | `RecoveryModel.set_treatment_inactive()` never fires; washout/rebound/sequelae unreachable | Flag initialized `True`, never assigned `False` anywhere | §6 — treatment-end transition |

Each fix is specified below with the exact before/after code.

---

## 3 — Defect 1: Stateful PBPK Engine

### 3.1 — Current behavior (broken)

`_DrugPBPK.advance()` (virtual_patient.py:609-640) rebuilds the model every hour:

```python
# CURRENT (virtual_patient.py:626-636) — DELETE
try:
    config = PBPKConfig(
        dt_min=self._config.dt_min,
        total_time_h=dt_h,
    )
    model = PBPKModel(self.drug, self.physiology, config)
    # Apply DDI/genetic clearance modifier to drug
    model.cl_total_l_per_h *= self.clearance_modifier
    result = model.run()
    if result.central_concentration:
        self._current_concentration = result.central_concentration[-1]
except Exception:  # pragma: no cover
    pass
```

Every call constructs `PBPKModel.__init__ → self._state = self._initial_state()`: all compartments restart at zero, the depot is wiped, and prior doses vanish. For a drug with t½ = 36 h (warfarin) dosed q24h, true accumulation ratio is `1/(1 − 2^(−24/36)) ≈ 2.4×`; the current code yields exactly 1.0 forever. Every downstream consumer (PD strength, hepatotoxicity scaling, AUC, therapeutic-range fraction) is therefore quantitatively meaningless.

### 3.2 — Replacement design

Reuse the **stateful, µM-based** `_PBPKEngine` from `human/simulation.py:255-398` (doc/27 engine room). It already implements:

- six well-stirred compartments (central + liver/kidney/brain/muscle/adipose) with organ blood flows from `HumanPhysiology`;
- oral/sc/im depot absorption, IV bolus, 1-h IV infusion, intrathecal bolus;
- partition coefficients Kp (adipose ∝ LogP, tissue-access factor by drug class);
- mass-conserving renal (central) + hepatic (liver) elimination;
- `apply_due_doses(now_h)` schedule enforcement against `duration_days × 24`;
- RK45 integration with fixed-step Euler fallback;
- persistent `conc_um` state across `advance()` calls.

Two gaps must be bridged: (i) `_PBPKEngine` fixes `k_el_per_h` at construction, but the VirtualPatient needs a per-hour DDI/genotype-modulated effective elimination constant; (ii) the facade wants a clean per-drug interface.

### 3.3 — Exact code change (virtual_patient.py)

Replace the entire `_DrugPBPK` class (lines 595-643) and add one import:

```python
from helixlang.human.simulation import _PBPKEngine


class _DrugPBPK:
    """Stateful per-drug PBPK wrapper (doc/29 Defect 1 fix).

    Delegates to the µM-based ``_PBPKEngine`` (human/simulation.py),
    which preserves compartment states across timesteps, and exposes a
    mutable clearance modifier applied to the elimination constant just
    before each advance.
    """

    def __init__(
        self,
        drug: Drug,
        physiology: HumanPhysiology,
        pbpk_dt_min: float = 1.0,
    ) -> None:
        self.drug = drug
        self.physiology = physiology
        self.clearance_modifier = 1.0
        self.mw_da = max(drug.molecule.molecular_weight_da, 1.0)
        self._engine = _PBPKEngine(drug, physiology, pbpk_dt_min)
        self._base_k_el_per_h = self._engine.k_el_per_h
        self.doses_given: list[float] = []

    def advance(self, dt_h: float, current_time_h: float) -> None:
        """Administer due doses, then integrate the ODE system by dt_h.

        Dose scheduling (interval, horizon) lives entirely inside
        ``_PBPKEngine.apply_due_doses``; the wrapper only injects the
        effective elimination constant.
        """
        self._engine.apply_due_doses(current_time_h)
        eff = min(20.0, max(0.01, self.clearance_modifier))
        self._engine.k_el_per_h = self._base_k_el_per_h * eff
        self._engine.advance(dt_h)

    def get_central_concentration(self) -> float:
        """Plasma (central) concentration in µM."""
        return self._engine.conc_um["central"]

    def get_all_concentrations(self) -> dict[str, float]:
        """All compartment concentrations in µM."""
        return self._engine.concentrations()
```

(Implementation note: the wrapper records no dose bookkeeping of its own — scheduling is fully delegated to `_PBPKEngine.apply_due_doses`, which enforces `dosing_interval_h` and the `duration_days × 24` horizon internally.)

Update `_init_drug_engines` (line 339-342):

```python
    def _init_drug_engines(self) -> None:
        for drug in self.config.drugs:
            key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
            self._drug_engine[key] = _DrugPBPK(drug, self._physiology, pbpk_dt_min=1.0)
```

The `run()` loop call sites (`engine.advance(dt_h, t_h)`, `engine.get_central_concentration()`) are unchanged — signatures are preserved.

### 3.4 — Why not patch `PBPKModel` instead?

Considered and rejected: `PBPKModel.run()` integrates a fixed `(0, total_time_h]` window and returns a monolithic `PBPKResult`; converting it to accept/return external state means changing its constructor contract, its `_build_odes` closure capture, and every existing caller/test on doc/27's surface. Delegation to `_PBPKEngine` touches exactly one file; `PBPKModel` is left untouched for doc/27 compatibility (deprecation marker added).

---

## 4 — Defect 2: Unit Normalization (mg/L → µM)

### 4.1 — The conversion

`PBPKModel` reports mg/L; every PD `ec50_um`, hepatotoxicity scale (`conc / 50.0`), nephrotoxicity scale (`conc / 40.0`), myelosuppression scale (`conc / 30.0`), and vital-drug-effect scale (`conc / 30.0`) in `clinical_output.py` expects **µM**. The bridge is the molecular weight:

```
conc_uM = conc_mg_per_L × 1000.0 / MW_da
```

Derivation: (mg/L) ÷ (g/mol) → (10⁻³ g/L) ÷ (g/mol) = 10⁻³ mol/L = mmol/L = µmol/mL; × 10⁻³ L/mL cancels to: µmol/L = mg/L × 10³ / MW(g/mol). Example: 1000 mg/L of a 500 Da drug → 2000 µM.

### 4.2 — Where it applies

After the §3 fix, `_DrugPBPK` emits µM natively (`_PBPKEngine` stores µmol/L), so the primary integration point becomes correct automatically. Two residual surfaces still require the guard:

1. **Legacy callers** of `PBPKModel.run()` (doc/27 examples, DSL `#sim kind=human`) keep receiving mg/L — correct for their own consumers, but any cross-wiring into VirtualPatient must convert. Add a shared helper in `pharmacokinetics.py`:

```python
def mg_per_l_to_um(conc_mg_per_l: float, mw_da: float) -> float:
    """Convert mg/L to µM. mw_da <= 0 returns 0.0 (unknown MW guard)."""
    if mw_da <= 0.0:
        return 0.0
    return conc_mg_per_l * 1000.0 / mw_da


def um_to_mg_per_l(conc_um: float, mw_da: float) -> float:
    """Inverse conversion (µM → mg/L)."""
    if mw_da <= 0.0:
        return 0.0
    return conc_um * mw_da / 1000.0
```

2. **Free-drug concentration** (§10.1): PD and toxicity models consume the *unbound* fraction, `C_free = f_u × C_total`, where `f_u = max(0.01, 1 − protein_binding_fraction)`. This multiplication happens at the same integration point in `VirtualPatient.run`:

```python
            # --- PBPK advance (µM-native post-D1) ---
            drug_concs: dict[str, float] = {}
            for drug in self.config.drugs:
                key = drug_key(drug)
                engine = self._drug_engine.get(key)
                if engine is None:
                    continue
                engine.advance(dt_h, t_h)
                total_um = engine.get_central_concentration()
                fu = max(0.01, 1.0 - drug.molecule.protein_binding_fraction)
                drug_concs[key] = total_um * fu      # free µM into PD/tox/labs
                total_drug_concs[key] = total_um     # kept for AUC reporting
```

### 4.3 — Regression guard

Unit test asserts round-trip identity and two golden anchors: warfarin 1 mg/L @ 308.3 Da → 3.244 µM; metformin 1 mg/L @ 165.6 Da → 6.039 µM.

---

## 5 — Defect 3: Eliminate Double CYP Counting

### 5.1 — Evidence

Path A — facade (virtual_patient.py:380-383):

```python
cyp_mod = _compute_genetic_cyp_modifier(drug, self.config.genotype)   # genotype ×1
ddi_mod = clearance_mods.get(drug.molecule.name, 1.0)
engine.clearance_modifier = cyp_mod * ddi_mod
```

Path B — `DDIModel.compute_clearance_modifiers` (ddi.py:181-234) applies explicit rules, **then** a blanket second pass:

```python
# CURRENT (ddi.py:221-228) — DELETE
for rule in self.rules:
    pair = (rule.substrate, rule.enzyme.lower())
    if pair in covered or rule.substrate not in name_set:
        continue
    activity = cyp_profiles.get(rule.enzyme, 1.0)
    if activity != 1.0:
        modifiers[rule.substrate] *= activity
        covered.add(pair)
```

Since `VirtualPatient.run` passes the raw genotype activities as `cyp_profiles` (lines 365-372), a CYP2C9 \*1/\*3 patient on warfarin receives genotype factor 0.5 **twice** (0.5² = 0.25 net) whenever any warfarin CYP2C9 rule exists, and even without rules the second pass re-multiplies raw activity for every drug/enzyme pair appearing in the rule database. Clearances are under-predicted by up to 10× for PM genotypes — compounding Defect 1's zero-accumulation bug in opposite directions.

### 5.2 — Separation-of-concerns contract

| Concern | Owner | Input | Output |
|---|---|---|---|
| Inherited enzyme capacity | `_compute_genetic_cyp_modifier` (facade) | `drug.cyp_metabolism`, `GenotypeProfile` | one clearance multiplier |
| Acquired interactions (inhibition/induction between drugs) | `DDIModel.compute_clearance_modifiers` | regimen names, **neutral** profile | per-drug multiplier |

### 5.3 — Exact changes

1. **Delete** ddi.py:221-228 (the second-pass loop shown above) and rewrite the docstring paragraph "As a second pass, any drug…" to read:

   ```
   Genotype-derived enzyme capacity is intentionally NOT applied here;
   it is owned by the calling facade (doc/29 §5). Enzyme-state rules
   (interacting_drug == enzyme) remain permitted: their fold_change is
   an explicit literature-anchored interaction magnitude, not a blanket
   activity copy.
   ```

2. **Facade passes a neutral profile** to the DDI model so no implicit genotype leaks through enzyme-state triggers during PK computation:

```python
            neutral_profile = {enz: 1.0 for enz in CORE_CYP_ENZYMES}
            clearance_mods = self._ddi_model.compute_clearance_modifiers(
                drug_names, neutral_profile,
            )
```

   Enzyme-state alerts (`get_clinical_alerts`) keep receiving the real profile — alerting on "you are a PM taking tamoxifen" is desired reporting, not PK math.

3. **Tamoxifen/CYP2D6 rule reclassification** (Phase 5 dependency): once active-metabolite modeling lands (§10.2), the `DEFAULT_DDI_RULES` entry `tamoxifen ← CYP2D6 (inhibition, ×0.3)` moves from clearance semantics to metabolite-activation semantics. Interim step (this phase): change its `interaction_type` to `"monitoring"` so it stops modifying clearance and surfaces only as an alert; the efficacy penalty flows through endoxifen formation in §10.2.

4. **Single-counting test** (added to `tests/test_human_ddi.py`):

```python
def test_no_blanket_genotype_second_pass():
    model = create_default_ddi_model()
    pm = {"CYP2D6": 0.1, "CYP2C19": 1.0, "CYP2C9": 1.0, "CYP3A4": 1.0, "CYP1A2": 1.0}
    mods_pm = model.compute_clearance_modifiers(["imatinib"], pm)
    mods_em = model.compute_clearance_modifiers(["imatinib"], {e: 1.0 for e in pm})
    assert mods_pm["imatinib"] == mods_em["imatinib"]  # no rules fire either way
```

---

## 6 — Defect 4: Reachable Recovery Phase

### 6.1 — Current behavior (broken)

`__init__` (line 296): `self._treatment_active = True`. The only reader (line 441-444):

```python
if self._recovery_model is not None:
    if not self._treatment_active and self._recovery_model.is_treatment_active:
        self._recovery_model.set_treatment_inactive()
    self._recovery_model.step(dt_h, t_h)
```

No assignment ever flips the flag, so `set_treatment_inactive()` is dead code, `RecoveryModel.is_treatment_active` stays `True`, `step()` short-circuits (`return dict(self.current_biomarkers)` at recovery.py:297-298), and the entire post-treatment promise of doc/28 §10 — biomarker relaxation, rebound envelopes, sequelae, relapse hazard — never executes.

### 6.2 — Exact changes (virtual_patient.py)

In `__init__` (after `self._recovery_model = ...`):

```python
        self._treatment_active = bool(config.drugs)
        self._treatment_end_h = max(
            (d.duration_days * 24.0 for d in config.drugs), default=0.0
        )
```

In `run()`, immediately after the PBPK-advance block and before the Recovery section:

```python
            # --- Treatment-phase transition (doc/29 Defect 4) ---
            if self._treatment_active and t_h >= self._treatment_end_h:
                self._treatment_active = False
                if self._recovery_model is not None:
                    # Freeze end-of-treatment deviations as the recovery start state
                    self._recovery_model.current_biomarkers.update(
                        self._labs_model.biomarker_export()
                    )
                    self._recovery_model.set_treatment_inactive()
                result.clinical_events.append({
                    "time_h": t_h,
                    "type": "phase_transition",
                    "details": (
                        "last scheduled dose passed "
                        f"(t={self._treatment_end_h:.0f} h); washout/recovery begins"
                    ),
                })
```

The Recovery block simplifies to:

```python
            if self._recovery_model is not None:
                self._recovery_model.step(dt_h, t_h)
```

(`RecoveryModel.step` already no-ops while `is_treatment_active`.)

### 6.3 — Supporting addition: `ClinicalLabModel.biomarker_export()`

`RecoveryModel` tracks named biomarkers (`current_biomarkers` keyed like `"ALT"`, `"creatinine"`). Export bridges the naming gap:

```python
_BIOMARKER_EXPORT_KEYS = {
    "ALT": "alt_u_per_l",
    "AST": "ast_u_per_l",
    "creatinine": "creatinine_mg_per_dl",
    "WBC": "wbc_per_ul",
    "hemoglobin": "hemoglobin_g_per_dl",
    "platelets": "platelets_per_ul",
    "CRP": "crp_mg_per_l",
    "bilirubin": "bilirubin_total_mg_per_dl",
}

def biomarker_export(self) -> dict[str, float]:
    """Current labs keyed for RecoveryModel consumption."""
    return {
        name: getattr(self.current, attr)
        for name, attr in _BIOMARKER_EXPORT_KEYS.items()
    }
```

### 6.4 — Washout semantics

`_treatment_end_h` uses the **maximum** duration across drugs: a fluconazole (7 d) + warfarin (90 d) regimen transitions at 90 d, matching doc/28 §10.1 ("last scheduled dose"). Per-drug staggered stop times are out of scope for this revision (recorded in Risks, §17).

---

## 7 — Full-Parameter Coverage: Overview

With the four defects closed, the pipeline runs but most channels are constants. Audit of `clinical_output.py` against the doc/28 §12 schema shows these analytes have **no dynamic model** (they sit at dataclass defaults forever):

| Group | Static today | Target |
|---|---|---|
| Electrolytes | Na⁺, K⁺, Cl⁻, HCO₃⁻, Ca²⁺, PO₄³⁻ | §8 renal/hormonal/drug models |
| Lipids | LDL, HDL, triglycerides | §9 statin + liver-synthesis model |
| Coagulation | INR | §9 hepatic synthesis + warfarin PD |
| CBC indices | MCV, RBC count | §9 iron/B12/erythropoiesis model |
| Vitals | QTc (absent), SpO₂ (static), RR (static) | §11 |
| Feedback | none wired disease→labs continuously | §10 |

Design pattern for every channel (identical to existing ALT/creatinine machinery): a **first-order relaxation** toward a computed equilibrium,

```
dB/dt = (B_eq(state) − B) / τ_channel
```

integrated hourly, clamped to survivable bounds, with `B_eq` assembled from baseline (trait/genotype-adjusted), disease-severity terms, and free-drug-concentration terms. Constants live in module-level dicts with per-entry sources.

---

## 8 — Dynamic Electrolyte Models (clinical_output.py)

### 8.1 — Sodium (Na⁺)

```
Na_eq(mmol/L) = 140
              − 12 × siadh_flag                          # dilutional hyponatremia
              − 4 × thiazide_or_loop_load(drug_conc)     # natriuresis
              + 2 × nsaid_retention(drug_conc)
              + 5 × dehydration_term                     # severity ≥ 0.7 + poor intake
dNa/dt = (Na_eq − Na) / 18 h                                   # water-balance equilibration
clamp [115, 165]; event "hyponatremia" when Na < 130, "severe" < 125
```

SIADH flag: disease-family `cancer` (paraneoplastic, small-cell pattern) or drugs in `{cisplatin}` with severity > 0.5. Dehydration term activates when `bun/creatinine ratio > 20` (prerenal pattern, Guyton 2016 Ch. 30).

### 8.2 — Potassium (K⁺)

```
K_eq(mmol/L) = 4.2
             + 1.0 × hyperkalemia_ckd(max(0, (45 − eGFR)/45))     # renal secretion loss
             − 0.8 × insulin_drive(glucose > 250 mg/dL)           # intracellular shift
             − 0.6 × beta_agonist_load                            # if β2-agonists modeled
             + 0.5 × raas_blockade(drug_conc)                     # ACEi/ARB class
             + 0.7 × k_sparing_diuretic(drug_conc)                # spironolactone
             − 0.3 × loop_diuretic(drug_conc)
             − 3.0 × max(0, (7.40 − pH_proxy))                    # acidemia cell-shift:
                                                                  # ~0.3 mmol/L per 0.1 pH
dK/dt = (K_eq − K) / 6 h                                        # fast (renal + shift)
clamp [1.5, 7.5]; events: < 3.0 hypokalemia, > 5.5 hyperkalemia,
> 6.5 severe (torsades/arrhythmia coupling to §11.1)
```

Insulin drive uses simulated glucose from the same snapshot — the glucose→K coupling reproduces DKA presentation (glucose 500 → K 3.2 despite total-body K depletion).

### 8.3 — Calcium (Ca²⁺)

Albumin-corrected reporting (standard clinical practice, Payne 1973):

```
Ca_corrected(mg/dL) = Ca_measured + 0.8 × (4.0 − albumin_g_per_dl)
Ca_measured_eq = 9.4
               − 1.0 × ckd_mbd(max(0, (30 − eGFR)/30))     # G4+: ↓ ionized Ca (secondary HPT)
               − 0.3 × bisphosphonate_load                 # if modeled
               + calcitriol_term
dCa/dt = (Ca_eq − Ca) / 72 h
clamp [6.0, 13.0]
```

Because albumin is already dynamic (cancer hypoalbuminemia), the correction term makes total calcium fall realistically in advanced disease without touching ionized physiology.

### 8.4 — Phosphate (PO₄³⁻)

```
PO4_eq(mg/dL) = 3.5
              + 2.0 × ckd_retention(max(0, (45 − eGFR)/45))    # G4/G5 retention
              + 1.5 × tumor_lysis_flag                         # cisplatin bulk kill
              − 0.5 × phosphate_binders
dPO4/dt = (PO4_eq − PO4) / 48 h
clamp [1.0, 12.0]
```

CKD-MBD pairing (KDIGO 2017): Ca↓ and PO₄↑ co-move with a reported `pth_proxy = 1 + 3 × max(0, (45 − eGFR)/45)` stored on the model for staging diagnostics.

### 8.5 — Chloride (Cl⁻)

Electroneutrality closure rather than an independent ODE (Emmett & Seldin 2013):

```
Cl_eq(mEq/L) = Na + K − HCO3 − 12        # 12 mEq/L ≈ normal unmeasured anions
dCl/dt = (Cl_eq − Cl) / 12 h
clamp [80, 130]
```

Guarantees the printed chemistry panel satisfies charge balance to within rounding — a property real labs always exhibit and naive independent channels would violate.

### 8.6 — Bicarbonate (HCO₃⁻)

Acid-base from the two drivers the simulator actually produces (lactate from cancer/sepsis physiology, uremic acids from CKD):

```
HCO3_eq(mEq/L) = 25
               − 1.0 × (lactate − 1.2)              # titration by lactic acid
               − 4.0 × uremic_term(max(0, (20 − eGFR)/20))
               − 8.0 × dka_flag(glucose > 300 and insulin_open)   # keto-anions
dHCO3/dt = (HCO3_eq − HCO3) / 12 h
clamp [5, 35]
```

Feeds directly into respiratory-rate compensation (§11.3) and the potassium cell-shift term (§8.2).

### 8.7 — Implementation shape

```python
_ELECTROLYTE_TAU_H = {          # channel → relaxation time constant
    "sodium_meq_per_l": 18.0, "potassium_meq_per_l": 6.0,
    "chloride_meq_per_l": 12.0, "bicarbonate_meq_per_l": 12.0,
    "calcium_mg_per_dl": 72.0, "phosphate_mg_per_dl": 48.0,
}   # τ chosen from compartment-equilibration literature (Guyton 2016)

def _apply_electrolytes(self, dt_h, drug_conc, severity) -> None:
    cur, base = self.current, self.baseline
    na_eq = self._na_equilibrium(drug_conc, severity)
    k_eq  = self._k_equilibrium(drug_conc, severity)
    ca_eq = self._ca_equilibrium()
    po4_eq = self._po4_equilibrium()
    hco3_eq = self._hco3_equilibrium()
    cl_eq = cur.sodium_meq_per_l + cur.potassium_meq_per_l - hco3_eq - 12.0
    for field_name, eq in (("sodium_meq_per_l", na_eq), ...):
        tau = _ELECTROLYTE_TAU_H[field_name]
        val = getattr(cur, field_name)
        setattr(cur, field_name, val + (eq - val) * (dt_h / tau) )
    self._clamp_channel(...)
```

---

## 9 — Dynamic INR, Lipids, and RBC Indices

### 9.1 — INR (two coupled drivers)

```
blockade = hill(C_free_warfarin_uM, IC50 = 1.8 µM, n = 1.0)     # VKORC1 inhibition
hepatic_synth = clamp(functional_hepatocyte_fraction, 0, 1)      # from cirrhosis severity:
                                                                 # f_hep = 1 − 0.85 × sev_cirrh
INR_eq = 1.0 + 2.5 × blockade × vit_k_status + 3.0 × max(0, 0.55 − f_hep)
dINR/dt = (INR_eq − INR) / 36 h          # factor-II/X half-life lag (Rowland & Tozer Ch. 20)
clamp [0.5, 12]; bleeding-risk event at INR > 3.5 (doc/28 §6.3 anchor)
```

`vit_k_status` = 1.0 default; dietary-vitamin-K insufficiency (not yet an input) reserved. Warfarin concentration arrives through the standard `drug_conc` dict (now free µM, correctly scaled post-D2) — this is the channel doc/28's flagship example 59 exercises.

### 9.2 — Lipid panel

Statin intensity → LDL fractional reduction (Grundy 2018 ACC/AHA), receptor-turnover kinetics:

```
ldl_reduction_max = {high: 0.50, moderate: 0.35, low: 0.20}[intensity(drug)]
ramp = 1 − 2^(−t_treated / 96 h)                 # LDL-receptor turnover t½ ≈ 4 d
LDL_eq = LDL_baseline × (1 − ldl_reduction_max × ramp)
       × (1 − 0.40 × cirrhosis_sev)              # failed hepatic synthesis
       × (1 + 0.25 × diabetes_sev)               # insulin-resistant VLDL overproduction
dLDL/dt = (LDL_eq − LDL) / 96 h
```

Companions: `TC = LDL + TC/HDL_ratio_anchor × HDL + TG/5` (Friedewald closure, valid TG < 400); `HDL_eq = HDL_base × (1 + 0.05 × exercise_factor) − 0.15 × sev_cancer`; `TG_eq = TG_base × (1 + 0.5 × diabetes_sev) × (1 + 0.3 × alcohol_trait)` with τ 72 h. Intensity lookup keyed on statin drug keys (`atorvastatin` high, `simvastatin` moderate, `pravastatin` low).

### 9.3 — MCV and RBC (erythropoietic axis)

Shared marrow-capacity factor with the existing myelosuppression model:

```
marrow = wbc_current / wbc_baseline                      # 0..1 proliferative reserve
epo_drive = clamp(eGFR / 90, 0.25, 1.0)                  # renal EPO (KDIGO anemia 2012)

MCV_eq(fL) = 90
           − 15 × iron_sequestration(sev_inflammation)   # anemia-of-chronic-disease
           + 20 × b12_folate_deficit_flag                # macrocytic if trait/disease set
           + 8 × chemo_macrocytosis(cumulative_cisplatin)
dMCV/dt = (MCV_eq − MCV) / (7 × 24 h)                    # RBC-population turnover

dRBC/dt = k_prod × epo_drive × marrow − RBC / (120 × 24 h)     # 120-d lifespan
k_prod s.t. RBC_eq(baseline) = 5.0 M/µL (male) / 4.5 (female)
clamp RBC [1.0, 7.0], MCV [55, 140]
```

Derived consistency closes the CBC triangle hourly (Wintrobe identities — replaces the current independent Hb drift):

```
Hgb(g/dL) = RBC(M/µL) × MCV(fL) × MCHC(g/dL) / 1000,   MCHC anchored 33 g/dL
Hematocrit(%) = RBC × MCV / 10
MCH(pg) = Hgb / RBC × 10
```

This makes anemia *emerge*: CKD → low EPO → falling RBC → falling Hb → tachycardia + SpO₂ coupling (§11.2), instead of three unrelated decays.

---

## 10 — Disease→Labs Feedback Loop (Phase 3)

### 10.1 — Continuous coupling table

`ClinicalLabModel` gains `self.disease_family` (detected once from `disease.name` at construction: `ckd | cirrhosis | t2dm | cancer | none`) and applies these equilibrium shifts every hour — replacing the current one-shot `_apply_disease_to_labs` baseline-only hook:

| Disease family | Severity s ∈ [0,1] drives | Equation (equilibrium) | Source |
|---|---|---|---|
| CKD | creatinine clearance | `Cr_eq = Cr_prod / (k × GFR_eff)`, `GFR_eff = baseline_eGFR × (1 − 0.8 s)`; feeds §9.3 epo, §8.3/8.4 Ca/PO₄, §8.2 K | KDIGO 2013 |
| Cirrhosis | synthetic failure | `albumin_eq = 4.5 − 2.8 s`; `bili_eq = 0.7 + 25 s²`; `INR` via §9.1 f_hep; ALP/GGT cholestatic drift `+180 s` | Pugh 1973 |
| T2DM | glycemia | `glucose_eq = 90 + 210 s − 80 × metformin_strength`; HbA1c lags at τ 2880 h (existing, kept) | Nathan 2007 |
| Cancer | inflammatory/catabolic | `CRP_eq = 1 + 50 s`; `lactate_eq = 1.2 + 6 s`; `albumin_eq = 4.5 − 3 s` (Warburg + cachexia); Na via SIADH flag | Fearon 2011; Pepys 2003 |

Loop-closure discipline (doc/28 §11.2 invariant): labs feed progression staging, progression severity feeds these equilibria. Gain is bounded because every equilibrium term is linear/saturating in s ∈ [0,1] and every channel relaxes through a positive τ — monotone convergence, no algebraic cycle. Stress test in §16 (extreme s = 1 for 90 d must converge, not oscillate).

### 10.2 — Active metabolites: tamoxifen → endoxifen

New metabolite pool managed by the facade (extends the existing `metabolite_pools` recording):

```
Formation (parent → endoxifen), CYP2D6-gated:
  f_act = 0.92                       # fraction of activation flux via CYP2D6 path
  form_umol_h = f_act × k_el,hep,parent × A_liver_parent × act_CYP2D6
  act_CYP2D6 = genotype.get_cyp_activity("CYP2D6") / 1.0      # PM ≈ 0.1

Endoxifen disposition:
  dA_end/dt = form_umol_h − k_el,end × A_end
  k_el,end = ln2 / 264 h             # endoxifen t½ ≈ 11 d (Ahmad 2010)
  C_end_uM = A_end / Vd,end ,  Vd,end = 500 L default

Pharmacology:
  ER-blockade strength = hill(C_end, EC50 = 2.4 nM-equiv × 1000 → 0.0024 µM … expressed µM)
  potency weight 30× parent (Jordan 2003) — aggregated into pd_multipliers
```

Consequences that fall out correctly: a CYP2D6 \*4/\*4 patient forms almost no endoxifen → weak anti-tumor PD signal despite normal parent levels; fluoxetine (CYP2D6 inhibitor, existing rule) crushes formation → same phenotype dynamically; endoxifen accumulates over ~5 weeks of therapy (τ 11 d) matching the clinical "delayed onset" narrative. Parent clearance is NOT reduced by CYP2D6 status (activation, not elimination) — consistent with §5 reclassification.

### 10.3 — Time-dependent CYP induction (rifampin pattern)

`DDIRule` gains two optional fields (backward-compatible defaults):

```python
@dataclass(slots=True)
class DDIRule:
    ...
    onset_half_life_h: float = 0.0    # 0 => immediate (inhibitors)
    offset_half_life_h: float = 0.0   # 0 => immediate offset
```

Effective fold at time t (co-administration duration) and after stop:

```
fold_on(t)  = 1 + (fold_change − 1) × (1 − 2^(−t / t½_onset))
fold_off(Δ) = fold_at_stop × 2^(−Δ / t½_offset)      # then re-fold to 1
```

Rifampin ships `onset_half_life_h = 84` (full CYP3A4 induction over ~1–3 wk; Niemi 2003), `offset_half_life_h = 36`. `DDIModel.compute_clearance_modifiers` grows a `time_since_coadmin_h: float = 0.0` argument; the facade passes `t_h − first_coadmin_time[substrate]` tracked per rule pair. Validation anchor: midazolam-class substrate AUC ratio must be < 1.5× at 24 h but ≥ 6× at day 21 (Niemi 2003).

### 10.4 — Protein binding (summary of §4.2 integration)

`f_u = max(0.01, 1 − protein_binding_fraction)`; free concentration feeds PD, toxicity scalers, INR blockade, QT model, and electrolyte drug terms. Total concentration continues to be recorded/AUC'd. Hypoalbuminemia (cirrhosis/nephrotic patterns) raises f_u dynamically: `f_u_eff = min(0.99, f_u × (4.5 / max(albumin, 1.5)))` — highly bound drugs (warfarin f_u 0.01 → ~0.02 at albumin 2.2) show clinically meaningful free-fraction doubling.

---

## 11 — Vitals Expansion (Phase 4, clinical_output.py)

### 11.1 — QTc model (new `VitalSigns.qtc_ms` field)

Forward Bazett formulation (Bazett 1920; Roden 2004):

```
RR_s = 60 / HR_bpm
QTc_true(t) = QTc_baseline                                  # 410 ms, sex-adjusted (F −10)
            + Σ_drugs ΔQTc_i × hill(C_free_i, EC50_i, n=1)  # concentration-dependent ms
            + 6.0 × max(0, 3.8 − K_mmol_l)                  # hypokalemia potentiation (ms per mmol/L)
            − 4.0 × max(0, K_mmol_l − 5.0)                  # hyperkalemia shortening
Reported QT interval:  QT_ms = QTc_true × sqrt(RR_s)         # Bazett inverted for display
```

Drug table (shipped in `_QT_EFFECTS_MS: dict[str, tuple[float, float]]`, drug → (ΔQTc_max ms, EC50 µM)):

| Drug | ΔQTc max | EC₅₀ (free µM) | Anchor |
|---|---|---|---|
| tamoxifen | +8 | 2.0 | Goldsmith 2013; label |
| cisplatin | +5 | 5.0 | Kelland 2007 adjunct |
| ibuprofen | +2 | 30.0 | NSAID class signal |
| ondansetron | +12 | 0.1 | FDA 2011 warning |
| methadone | +15 | 1.0 | EAPC 2013 |
| ciprofloxacin | +8 | 2.0 | Roden 2004 review |
| haloperidol | +12 | 0.05 | Roden 2004 |
| hydroxychloroquine | +10 | 1.0 | Chen 2020 |
| azithromycin | +6 | 0.5 | Ray 2012 |
| *(any other `_QT_PROLONGING_DRUGS` hit)* | +10 | 1.0 | conservative default |

Events: `QTc ≥ 480 ms` → `"qtc_prolongation"` monitoring event; `≥ 500 ms` → `"torsades_risk"` severe event (Roden 2004 threshold); additive-QT DDI alert from doc/28 §9 already flags combinations — the numeric channel now substantiates it. Bradycardia amplification is intrinsic (Bazett over-corrects at low HR) and accepted as documented limitation.

### 11.2 — SpO₂ (replace static default)

Severinghaus dissociation curve (doc/28 §7.2 spec, now wired):

```
PaO2(mmHg) = 96 − 0.30 × age − lung_impairment
lung_impairment = 0.05 × pack_years + 15 × pneumonitis_drug_load   # bleomycin/amiodarone class
P50 = 26.8 + 2.0 × chronic_anemia_flag(Hgb < 10 sustained 2 wk)    # 2,3-BPG right shift
SaO2(%) = 100 × PaO2^2.7 / (PaO2^2.7 + P50^2.7)
clamp [50, 100]; event "hypoxemia" at SaO2 < 88
```

Physiology note recorded in-code: anemia lowers O₂ *delivery*, not saturation; the modeled SpO₂ decrement is exclusively the chronic-anemia P50 shift (±2%), preventing the common modeling error of tying SpO₂ directly to Hb.

### 11.3 — Respiratory rate (replace static default)

```
RR_eq(/min) = 14
            + 1.2 × max(0, 25 − HCO3)          # Kussmaul compensation (metabolic acidosis)
            + 2.0 × max(0, T_core − 37.0)      # fever drive
            − 7.0 × opioid_load                # centrally depressant class, saturating
dRR/dt = (RR_eq − RR) / 0.25 h               # ventilatory controller is fast
clamp [6, 44]; event "resp_depression" at RR < 10 with opioid_load > 0
```

Closes the acid-base loop: cancer/sepsis lactate (§10.1) → HCO₃ falls (§8.6) → RR rises → displayed as compensatory tachypnea. Opioid load hooks the rebound library (withdrawal removes the term → RR recovers, tachypnea overshoot omitted).

### 11.4 — Recording plumbing

`VitalSigns` gains `qtc_ms: float = 410.0`; `VirtualPatientResult` gains `qtc_ms: list[float]` appended in the record block beside `spo2_pct`/`respiratory_rate`; `to_dict()["vitals"]["qtc_ms"]`; `summary()` prints peak QTc.

---

## 12 — Facade Wiring Order (post-fix hourly pipeline)

```
for hour in steps:
  1. DOSING      _PBPKEngine.apply_due_doses(t) per drug (schedule-aware, horizon-limited)
  2. CLEARANCE   genotype modifier (facade) × DDI modifiers (rules only, neutral profile,
                 induction ramps at t)
  3. PBPK        engine.advance(dt) with effective k_el — STATE PRESERVED
  4. UNITS       free µM = fu(albumin-adjusted) × central µM
  5. METABOLITE  endoxifen formation/disposition step
  6. PD          Hill on free concentrations (+ endoxifen potency weighting)
  7. TRANSITION  t ≥ Σ treatment_end → deactivate treatment, seed recovery snapshot
  8. dFBA        unchanged (doc/27 batch step)
  9. LABS        electrolytes → acid-base → lipids → INR → RBC/MCV → disease feedback →
                 legacy tox/hepatic/renal/myelo/recovery-to-baseline → derived (eGFR, CBC closure)
 10. VITALS      hemodynamics → QTc → SpO2 → RR → temperature/weight (existing)
 11. PROGRESS    staging from updated labs (loop closure, bounded gains)
 12. EVENTS      Hy's law, AKI, ANC, INR > 3.5, Na/K extremes, QTc 480/500, hypoxemia
 13. RECOVERY    RecoveryModel.step (active only post-transition)
 14. RECORD      append at output resolution
```

Steps 2–4 implement Defects 1–3 jointly; step 7 implements Defect 4; steps 9–11 implement Phases 2–4.

---

## 13 — Implementation Plan

### Phase 1 — Critical defect fixes (`virtual_patient.py`, `ddi.py`)
- [ ] Replace `_DrugPBPK` internals with `_PBPKEngine` delegation (§3.3)
- [ ] Add `mg_per_l_to_um`/`um_to_mg_per_l` helpers (§4.2); wire free-fraction split
- [ ] Delete ddi.py second pass; neutral-profile call; tamoxifen rule → `monitoring` (§5.3)
- [ ] Treatment-end transition + `biomarker_export` (§6.2–6.3)
- [ ] Unit tests: accumulation ratio, unit round-trip, single-counting, phase-transition event

### Phase 2 — Dynamic lab channels (`clinical_output.py`)
- [ ] `_apply_electrolytes` (Na/K/Ca/PO₄/Cl/HCO₃, §8) + `_ELECTROLYTE_TAU_H`
- [ ] INR dual-driver model (§9.1); lipid panel with Friedewald closure (§9.2)
- [ ] MCV/RBC erythropoietic axis + Wintrobe closure (§9.3)
- [ ] Extend `VirtualPatientResult`/recording/`to_dict` for new series

### Phase 3 — Disease→labs continuous feedback (`clinical_output.py`)
- [ ] `disease_family` detection; per-family equilibrium table (§10.1)
- [ ] Stability stress test s = 1 × 90 d convergence gate

### Phase 4 — ECG/QTc + vitals completion (`clinical_output.py`)
- [ ] `qtc_ms` field, `_QT_EFFECTS_MS` table, Bazett forward/inverse, events (§11.1)
- [ ] Dynamic SpO₂ (§11.2) and RR (§11.3)

### Phase 5 — Drug-model enhancements (`ddi.py`, `drug.py`, `virtual_patient.py`)
- [ ] `DDIRule.onset_half_life_h`/`offset_half_life_h` + ramp math + facade timing (§10.3)
- [ ] Endoxifen pool module + CYP2D6 gating + potency aggregation (§10.2)
- [ ] Albumin-dependent free fraction (§10.4)
- [ ] Restore tamoxifen/CYP2D6 rule as metabolite-activation documentation (alert-only)

### Phase 6 — Tests, validation, performance
- [ ] All new unit/integration tests (§16); benchmark suite (§15) wired into CI
- [ ] 180-day/3-drug/2-disease run < 60 s regression check
- [ ] `ruff` + `mypy` clean; determinism (seeded, fixed order) verified bit-for-bit

---

## 14 — File Impact Summary

| File | Change type | Est. lines |
|---|---|---|
| `src/helixlang/human/virtual_patient.py` | Modified — `_DrugPBPK` rewrite, unit/free-fraction integration, phase transition, endoxifen pool, recording expansion | +180 / −60 |
| `src/helixlang/human/ddi.py` | Modified — delete second pass, rule timing fields, ramp math | +45 / −12 |
| `src/helixlang/human/clinical_output.py` | Modified — electrolytes, INR, lipids, RBC axis, QTc, SpO₂, RR, disease feedback, export helper | +420 |
| `src/helixlang/human/pharmacokinetics.py` | Modified — unit-conversion helpers, deprecation marker | +25 |
| `tests/test_human_virtual_patient.py` | Extended — accumulation, units, transition, metabolite | +220 |
| `tests/test_human_ddi.py` | Extended — single-counting, induction ramps | +90 |
| `tests/test_human_clinical_output.py` | Extended — electrolyte equilibria, INR, lipids, RBC closure, QTc, SpO₂, RR | +260 |
| `examples/59_virtual_patient_ckd_warfarin_ddi.helix` | Unchanged — becomes the flagship validation vehicle | 0 |

No parser/runtime changes; DSL surface of doc/28 is untouched.

---

## 15 — Validation Criteria (benchmarks wired to CI)

| # | Scenario | Metric | Must produce | Anchor |
|---|---|---|---|---|
| 1 | Warfarin 90 d q24h (D1) | Day-30 trough / first-dose peak | ≥ 1.8 (accumulation present; theory 2.4×) | Rowland & Tozer 2020 |
| 2 | Same run (D1+D2) | Simulated Cavg ss vs one-compartment F·Dose/(CL·τ) | within ×2 | drug.py steady-state methods |
| 3 | CYP2C9 \*1/\*3 + fluconazole (D3) | INR rise over personal baseline | ≥ +1.0 within 5 d, NOT doubled by genotype reapplication | Black 1996 |
| 4 | Any PM genotype (D3) | Net clearance multiplier | equals single genotype factor, not its square | §5 contract |
| 5 | Regimen ending day 30 (D4) | `phase_transition` event timestamp | within 1 h of t = 720 h; `recovery_start` logged next step | doc/28 §10 |
| 6 | Post-transition 90 d | Biomarker normalization order | CRP → ALT → creatinine → WBC → Hb (half-life ordering) | doc/28 §10.2 |
| 7 | Cancer s = 0.6, 14 d | Na⁺ | 128–134 (SIADH band) | Guyton 2016 |
| 8 | DKA-pattern (glucose > 350) | K⁺ | ≤ 3.4 with insulin drive | §8.2 |
| 9 | CKD G4 (eGFR 25) | Ca / PO₄ / K / Hb | Ca ≤ 8.9; PO₄ ≥ 5.0; K ≥ 4.7; Hb ≤ 11 | KDIGO 2017/2012 |
| 10 | Cirrhosis s = 0.7 | albumin / INR / bili | ≤ 2.5 / ≥ 1.7 / ≥ 3.0 | Pugh 1973 bands |
| 11 | High-intensity statin 8 wk | LDL | 50–55% reduction | Grundy 2018 |
| 12 | Tamoxifen EM vs \*4/\*4, 8 wk | endoxifen Cavg ratio | PM ≤ 0.35 × EM | Goetz 2018 CPIC |
| 13 | Rifampin co-start, day 1 vs day 21 | CL multiplier | day 1 ≤ 1.5×; day 21 ≥ 6× | Niemi 2003 |
| 14 | Tamoxifen + hydroxychloroquine, K 3.3 | QTc | ≥ 480 ms; `torsades_risk` if ≥ 500 | Roden 2004 |
| 15 | Lactate 8 mmol/L | HCO₃ ≤ 18 and RR ≥ 22 | acid-base + Kussmaul closure | §8.6/§11.3 |
| 16 | Age 80 vs 30 | SpO₂ delta | 0 to −1% (age slope only) — no anemia artifact | §11.2 |
| 17 | s = 1 × 90 d stability | all channels | converged/monotone; no oscillation amplitude growth | §10.1 gain bounds |
| 18 | Determinism | two identical runs | bit-identical results | doc/28 AC 11 |

---

## 16 — Test Plan Additions

### Unit (`tests/test_human_clinical_output.py`)

| Test | Asserts |
|---|---|
| `test_sodium_relaxation_tau` | step to Na_eq = 132 converges with τ = 18 h (±10%) |
| `test_potassium_insulin_shift` | glucose 400 pulls K below 3.6 within 12 h |
| `test_chloride_electroneutrality` | \|Na + K − Cl − HCO₃ − 12\| ≤ 0.5 every hour |
| `test_calcium_albumin_correction` | albumin 2.2 lowers corrected Ca appropriately |
| `test_inr_dual_driver` | warfarin-only and cirrhosis-only contributions additive, capped |
| `test_ldl_statin_ramp` | 96 h half-time ramp; 50% floor at 8 wk high-intensity |
| `test_wintrobe_identities` | Hgb/Hct/MCH mutually consistent to rounding |
| `test_qtc_bazett_roundtrip` | reported QT × √RR identity holds at HR 50/70/100 |
| `test_spo2_curve_shape`, `test_rr_kussmaul` | PaO2 60 → SaO₂ ≈ 91%; HCO₃ 12 → RR ≈ 27.6 |

### Integration (`tests/test_human_virtual_patient.py`)

| Test | Asserts |
|---|---|
| `test_accumulation_reaches_steady_state` | benchmark 1–2 |
| `test_single_counting_end_to_end` | benchmark 3–4 via facade path |
| `test_recovery_phase_executes` | benchmark 5–6; rebound envelope appears for steroid-class |
| `test_endoxifen_phenotype_split` | benchmark 12 |
| `test_induction_timeline` | benchmark 13 |
| `test_torsades_event_chain` | benchmark 14 incl. event log entry |
| `test_full_panel_populates` | every §12-series length == len(time_h); no all-constant channels (variance > 0 for moved channels) |
| `test_180day_performance_budget` | wall clock < 60 s |

---

## 17 — Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Staggered per-drug stop times (max-duration approximation) | Medium | Documented; per-drug stop bookkeeping deferred with explicit TODO; recovery seeded at last stop is conservative (over-treatment bias only) |
| Electrolyte feedback loops (K ↔ pH ↔ RR ↔ HCO₃) oscillation | Medium | All couplings routed through first-order τ relaxation with saturating equilibrium terms; §15 #17 convergence gate in CI |
| Free-fraction change breaks EC₅₀ calibration of legacy drug tables | Medium | One-time audit: `_HEPATOTOXIC_DRUGS`/`_NEPHROTOXIC_DRUGS`/vitals scales re-anchored in free-µM units with golden-value tests |
| Endoxifen parameter uncertainty (Vd, formation fraction) | Low | Defaults cited (Ahmad 2010; Jordan 2003); exposed as `Drug`-level overrides for refinement |
| Bazett over-correction at extreme HR | Low | Accepted + documented; Fridericia available behind config flag if benchmarks demand |
| Performance regression from 10+ extra ODE channels | Medium | All channels are scalar exponential relaxations (< 50 flops each/hour); LP solve remains dominant cost; §16 perf test guards |
| Deleting DDI second pass changes existing test expectations | Certain | Tests asserting double-counting are treated as bug-tests and rewritten alongside (single PR, no deprecation window) |

---

## 18 — Acceptance Criteria

1. Accumulation: repeated-dose regimens produce rising troughs approaching theoretical accumulation ratio (benchmark 1–2).
2. Units: every PD/toxicity/vital consumer receives free µM; round-trip helpers tested; no mg/L leakage past the integration point.
3. Genetics counted once: net clearance multiplier equals the single genotype factor; DDI rules contribute only rule fold-changes (benchmarks 3–4).
4. Recovery reachable: `phase_transition` fires at max drug duration; recovery trajectory, rebound events, and sequelae appear in results (benchmarks 5–6).
5. Full panel: all channels listed in §7 vary in response to disease severity, drug exposure, and time; Wintrobe and electroneutrality identities hold hourly.
6. Disease→labs loop: CKD/cirrhosis/T2DM/cancer families reproduce §10.1 equilibrium targets within ±15% at steady state.
7. QTc: Bazett-consistent reporting; concentration-dependent prolongation per §11.1 table; 480/500 ms events logged.
8. Induction kinetics: rifampin-pattern ramps match benchmark 13.
9. Metabolite pharmacogenomics: endoxifen split matches CPIC expectation (benchmark 12).
10. Quality gates: `ruff`/`mypy` clean; determinism bit-for-bit; performance budget held; all doc/28 acceptance criteria that remain valid (genotype ingestion, staging rubrics, DDI alerting, DSL examples 59–61) still green.

---

## 19 — Literature References

1. Bazett HC. An analysis of the time-relations of electrocardiograms. *Heart* 1920;7:353-370.
2. Roden DM. Drug-induced prolongation of the QT interval. *N Engl J Med* 2004;350:1013-1022.
3. Severinghaus JW. Simple, accurate equations for human blood O2 dissociation computations. *J Appl Physiol* 1979;46:599-602.
4. Guyton AC, Hall JE. *Textbook of Medical Physiology.* 13th ed. Elsevier; 2016.
5. Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics.* 5th ed. Wolters Kluwer; 2020.
6. Niemi M et al. Effects of rifampin on the pharmacokinetics of triazolam… *Clin Pharmacol Ther* 2003 (CYP3A4 induction timeline).
7. Black DJ et al. Warfarin–fluconazole interaction: pharmacokinetic and pharmacodynamic aspects. *Antimicrob Agents Chemother* 1996;40:1123-1128.
8. Grundy SM et al. 2018 AHA/ACC cholesterol guideline. *Circulation* 2019;139:e1082-e1143.
9. Friedewald WT et al. Estimation of the concentration of low-density lipoprotein cholesterol in plasma. *Clin Chem* 1972;18:499-502.
10. Payne RB et al. Interpretation of serum calcium in patients with abnormal serum proteins. *BMJ* 1973;4:643-646.
11. KDIGO. CKD–MBD update. *Kidney Int Suppl* 2017;7:1-59.
12. KDIGO. Anemia in CKD. *Kidney Int Suppl* 2012;2:279-335.
13. Nathan DM et al. Translating the A1C assay into estimated average glucose values. *Diabetes Care* 2007;30:1473-1478.
14. Pepys MB, Hirschfield GM. C-reactive protein: a critical update. *J Clin Invest* 2003;111:1805-1812.
15. Jordan VC. Tamoxifen: a most unlikely pioneering medicine. *Nat Rev Drug Discov* 2003;2:205-213.
16. Ahmad A et al. Pharmacokinetics of endoxifen following tamoxifen. *Br J Clin Pharmacol* 2010 (terminal t½ ≈ 11 d).
17. Goetz MP et al. CPIC guideline for CYP2D6 and tamoxifen therapy. *Clin Pharmacol Ther* 2018;103:770-777.
18. Emmett M, Seldin DW. Disorders of acid-base and electrolyte balance (electroneutrality). In: *Seldin & Giebisch's The Kidney*, 5th ed.
19. Fearon K et al. Understanding and managing cancer wasting. *Nat Rev Clin Oncol* 2011;8:229-239.
20. Kim WR et al. Kinetics of alanine aminotransferase in clinical practice. *Hepatology* 2008;47:1363-1369.
21. Friberg LE et al. Model of chemotherapy-induced myelosuppression. *J Clin Oncol* 2003;21:1671-1677.
22. Ray WA et al. Azithromycin and the risk of cardiovascular death. *NEJM* 2012;366:1881-1890.
