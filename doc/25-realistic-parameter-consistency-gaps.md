# Doc/25: Realistic Parameter Consistency Gap Closure

## Status: Phase I-VI ✅ IMPLEMENTED | Phase VII-X ✅ IMPLEMENTED (new gaps)

## Overview

doc/24 completed full GEM import, making `use_full_model=true` produce publication-grade
results (E. coli iML1515 mu=0.877 h^-1). After auditing the genome-reconstruction path
(genome -> reconstruction -> GRN -> expression -> FBA) and DSL configurability, the
following gaps were found to block "realistic parameter consistency." This doc lists
each gap and its fix.

---

## Phase I-VI: Original Gap Fixes (all implemented)

### G1: GRN Inference Pipeline Wiring Bug ✅
**Location**: `apps/gem_pipeline.py:669-673`
**Problem**: `run_gem_pipeline()` accepts `use_database_interactions` but never passes it
to `infer_grn()`; `genome_gene_ids` is also never passed.
**Consequence**: GRN always uses a hardcoded E. coli table, ignoring genome validation;
results do not change when a different organism is used.
**Fix**: Extract `gene_ids` from `result.annotations` and pass to `infer_grn()` as
`genome_gene_ids`; pass `use_database_interactions` as `database_interactions`
(True -> None uses default table, False -> [] disables).

### G2: Expression Inference Not DSL-Configurable ✅
**Location**: `omics/expression_inference.py:111-123`, `sim_runtime.py:2240-2260`
**Problem**: `#gene` does not support `expression_level=` (enzyme concentration /
expression level). `infer_expression()` has hardcoded internal parameters; DSL cannot
override per-gene.
**Consequence**: The `expression=true` path in the gem backend produces all inferred
enzyme concentrations, which cannot be calibrated with experimental data.
**Fix**:
1. `#gene` fields dict already auto-supports `expression_level=` (no parser/AST changes needed)
2. In `sim_runtime._run_gem`, collect `program.genes[].fields["expression_level"]`
3. Build `ExpressionModel` and use DSL values to override `promoter_strength[gene]`
4. Pass to `infer_expression(model=...)`, DSL overrides are applied via `update()` to the final result

### G3: #enzyme kcat Ignored Under gem Backend ✅
**Location**: `sim_runtime.py:2265-2275`
**Problem**: `#enzyme gene=X reaction=Y kcat=N` only takes effect in the fba backend;
the gem backend ignores it.
**Consequence**: User-specified experimental kcat values in DSL are ignored.
**Fix**: After the gem backend builds `EnzymeCapacity`, iterate over `program.enzymes`
and use DSL `kcat` to override `ec.kcat[reaction]`.

### G4: #patch Missing temperature= / ph= Direct Fields ✅
**Location**: `apps/ecosystem.py:358-397`, `sim_runtime.py:1645-1726`
**Problem**: `PatchConfig` has no temperature/pH fields. Only indirectly simulated via
`scalar`.
**Consequence**: DSL cannot directly declare environment temperature/pH.
**Fix**:
1. `PatchConfig` adds `temperature_c: float = 25.0` and `ph: float = 7.0`
2. `_build_ecosystem_patches` parses `temperature=` and `ph=`
3. DSL can directly write `#patch name=env temperature=30.0 ph=7.5`

### G5: Medium Preset Not Partially Overridable ✅
**Location**: `sim_runtime.py:2665-2732`
**Problem**: Once you select `bg11`, you cannot change only Fe3+ concentration.
**Consequence**: Fine-tuning experimental conditions is impractical.
**Fix**:
1. `#gem` adds `medium_override=fe3_e:0.5,co2_e:500` (comma-separated met:value pairs)
2. `_run_gem` parses into `dict[str, float]`
3. `_set_gem_medium` accepts `medium_override` parameter, applies `update()` on top of preset
4. `_run_gem_full_model` directly modifies model exchange bounds

### G6: _ORGANISM_MAX_GROWTH_RATE Not DSL-Configurable ✅
**Location**: `sim_runtime.py:2595-2610`
**Problem**: e_coli=0.87, synechocystis=0.14, etc. are hardcoded; DSL cannot override.
**Consequence**: Even when full-model FBA yields a more accurate growth rate, it is capped.
**Fix**:
1. `#gem` adds `max_growth_rate=0.5` field
2. `_run_gem` and `_run_gem_full_model` read and override `max_mu`

---

## Phase VII: GRN-to-Metabolism Closed Loop ✅

### G7: GRN Regulatory Edges Do Not Constrain FBA Bounds

**Problem**: `RegulatoryEdge` objects have `regulation_type` ("activation" / "repression")
and `confidence`, but no `target_reaction` field. The GRN is generated but never fed back
to constrain metabolic flux bounds. A repressed gene should reduce the upper bound of
reactions catalyzed by that gene's enzyme; an activated gene should increase it.

**Fix**:

1. **Add `target_reaction` field to `RegulatoryEdge`** in `grn_inference.py`:
   ```python
   target_reaction: str | None = None  # mapped from GPR association
   ```

2. **Create `apply_regulatory_bounds()` in `gem/bridge.py`**:
   ```python
   def apply_regulatory_bounds(
       model: MetabolicModel,
       grn_edges: list[RegulatoryEdge],
       gpr_map: dict[str, list[str]],  # gene -> [reaction_ids]
       base_fraction: float = 0.1,     # min fraction of original bound when repressed
   ) -> None:
       """Apply GRN regulatory edges to FBA reaction bounds.
       
       For each regulatory edge:
       - Repression: scale upper_bound down by (1 - confidence * base_fraction)
       - Activation: scale upper_bound up towards original (1.0) by confidence
       """
   ```

3. **Wire into `_run_gem()` and `_growth_rate_gem()`**: After FBA model is built,
   call `apply_regulatory_bounds(model, grn_edges, gpr_map)` before solving.

4. **DSL syntax**: Automatic from GRN inference; no additional DSL needed. The
   `#gem expression=true` flag triggers the full pipeline including this step.

**Implementation in `gem/bridge.py`**:

```python
def apply_regulatory_bounds(
    model: MetabolicModel,
    grn_edges: list[RegulatoryEdge],
    gpr_map: dict[str, list[str]],
    base_fraction: float = 0.1,
) -> int:
    """Apply GRN regulatory edges to FBA reaction bounds.
    
    Returns the number of reactions whose bounds were modified.
    """
    n_modified = 0
    for edge in grn_edges:
        if edge.target_reaction is None:
            # Try to find reaction via GPR mapping
            reactions = gpr_map.get(edge.target_gene, [])
            if not reactions:
                continue
        else:
            reactions = [edge.target_reaction]
        
        for rxn_id in reactions:
            if rxn_id not in model.reactions:
                continue
            rxn = model.reactions[rxn_id]
            orig_ub = rxn.upper_bound
            orig_lb = rxn.lower_bound
            
            if edge.regulation_type == "repression":
                # Scale bounds toward zero by confidence * base_fraction
                scale = 1.0 - edge.confidence * base_fraction
                rxn.upper_bound = orig_ub * scale
                # Don't collapse negative bounds (exchange reactions)
                if orig_lb < 0:
                    rxn.lower_bound = orig_lb * scale
            elif edge.regulation_type == "activation":
                # Ensure bounds are at least confidence fraction of original
                min_ub = orig_ub * edge.confidence
                rxn.upper_bound = max(rxn.upper_bound, min_ub)
            
            n_modified += 1
    
    return n_modified
```

**Wiring in `sim_runtime.py` `_run_gem()`**: After Stage 3 (GRN inference) and
before Stage 6 (FBA), insert:

```python
if grn_result and grn_result.regulatory_edges:
    from helixlang.plugins.gem.bridge import apply_regulatory_bounds
    n = apply_regulatory_bounds(model, grn_result.regulatory_edges, gpr_map)
    _extra_meta["grn_bounds_applied"] = n
```

---

## Phase VIII: Temperature / pH Effects on Enzyme Rates ✅

### G8: Patch temperature_c and ph Not Used in GEM Growth Calculation

**Problem**: `PatchConfig.temperature_c` and `PatchConfig.ph` exist (doc/25 G4) but
`_growth_rate_gem()` in `ecosystem.py` does not use them to correct enzyme kcat or
FBA bounds. Temperature affects enzyme kinetics via the Arrhenius equation; pH affects
enzyme activity via protonation of active-site residues.

**Fix**:

1. **Create `enzyme_correction()` in `metabolism.py`**:
   ```python
   def enzyme_correction(
       temperature_c: float,
       ph: float,
       ea_kj_mol: float = 50.0,     # activation energy (kJ/mol, typical for enzymes)
       t_opt_c: float = 37.0,       # optimal temperature
       ph_opt: float = 7.0,         # optimal pH
       ph_width: float = 2.0,       # pH tolerance window (Gaussian sigma)
   ) -> float:
       """Calculate enzyme activity correction factor from temperature and pH.
       
       Uses Arrhenius for temperature and Gaussian for pH:
       f(T) = exp(-Ea/R * (1/T - 1/T_opt))
       f(pH) = exp(-(pH - pH_opt)^2 / (2 * ph_width^2))
       
       Returns a multiplier in [0, 1] where 1.0 = optimal conditions.
       """
       R = 8.314e-3  # gas constant in kJ/(mol*K)
       T = temperature_c + 273.15
       T_opt = t_opt_c + 273.15
       # Arrhenius (capped at 1.0 at optimal)
       arr = math.exp(-ea_kj_mol / R * (1/T - 1/T_opt))
       arr = min(arr, 1.0)
       # pH Gaussian
       ph_factor = math.exp(-(ph - ph_opt)**2 / (2 * ph_width**2))
       return arr * ph_factor
   ```

2. **Wire into `_growth_rate_gem()` in `ecosystem.py`**: After building the FBA
   object, apply temperature/pH correction:
   ```python
   # Apply temperature/pH correction to enzyme capacity
   from helixlang.metabolism import enzyme_correction
   patch_temp = self.config.temperature_c
   patch_ph = self.config.ph
   correction = enzyme_correction(patch_temp, patch_ph)
   if sp.metabolic_model and hasattr(sp.metabolic_model, '_adapter'):
       adapter = sp.metabolic_model._adapter
       if hasattr(adapter, 'kcat_map'):
           for rxn_id, kcat in adapter.kcat_map.items():
               adapter.kcat_map[rxn_id] = kcat * correction
   ```

3. **Pass `temperature_c` and `ph` from `Patch` to `_growth_rate_gem()`**: The
   `_modifiers()` method already computes `t_mod` from Q10; add `ph_mod` and pass
   both to the GEM path.

**Implementation in `metabolism.py`**:

```python
def enzyme_correction(
    temperature_c: float,
    ph: float,
    ea_kj_mol: float = 50.0,
    t_opt_c: float = 37.0,
    ph_opt: float = 7.0,
    ph_width: float = 2.0,
) -> float:
    R = 8.314e-3  # kJ/(mol*K)
    T = temperature_c + 273.15
    T_opt = t_opt_c + 273.15
    arr = math.exp(-ea_kj_mol / R * (1/T - 1/T_opt))
    arr = min(arr, 1.0)
    ph_factor = math.exp(-(ph - ph_opt)**2 / (2 * ph_width**2))
    return arr * ph_factor
```

**Wiring in `ecosystem.py` `_growth_rate_gem()`**:

```python
# After line 1292 (fba = FluxBalanceAnalysis(model)):
from helixlang.metabolism import enzyme_correction
correction = enzyme_correction(
    self.config.temperature_c, self.config.ph)
# Scale all enzyme capacities by the correction factor
for rxn_id in list(model.reactions):
    rxn = model.reactions[rxn_id]
    if rxn.subsystem not in ("exchange", "biomass", "maintenance"):
        rxn.upper_bound *= correction
        if rxn.lower_bound < 0:
            rxn.lower_bound *= correction
```

---

## Phase IX: Population Dynamics Feedback to GEM ✅

### G9: Static FBA Model Not Updated by Population Density

**Problem**: In the ecosystem tick loop, `_growth_rate_gem()` uses a static
`sp.metabolic_model` with only exchange bounds updated from local substrate
concentrations. There is no mechanism to:
- Scale enzyme capacity based on total population density (quorum effect)
- Jointly optimize substrate uptake across multiple species (community FBA)
- Update enzyme levels based on population-level protein economy

**Fix**:

1. **Add density-dependent enzyme scaling to `_growth_rate_gem()`**:
   ```python
   # Scale enzyme capacity by total biomass density (quorum effect)
   total_biomass = sum(
       self.fields[s].get(x, y) 
       for s in self.fields
   )
   density_scale = min(1.0, total_biomass / carrying_capacity)
   # Apply density scaling to non-exchange reaction bounds
   for rxn_id in model.reactions:
       if not rxn_id.startswith("EX_"):
           model.reactions[rxn_id].upper_bound *= density_scale
   ```

2. **Add community FBA mode** (`community_fba=True` in `EcosystemConfig`):
   When enabled, all species share a single FBA model with combined exchange
   constraints. This is activated by the existing `community_fba` parameter.

3. **Store FBA solution on Species for cross-tick persistence**:
   Add `last_fba_fluxes: dict[str, float]` field to `Species` dataclass;
   update it after each FBA solve in `_growth_rate_gem()`.

**Implementation in `ecosystem.py` `_growth_rate_gem()`**:

```python
# After setting exchange bounds (line 1306):
# Density-dependent enzyme scaling (quorum effect)
total_biomass = bx  # current total biomass at this site
carrying = self.config.carrying_capacity if hasattr(self.config, 'carrying_capacity') else 1e5
density_scale = min(1.0, max(0.1, total_biomass / carrying))
# Apply to internal reactions (not exchange)
for rxn_id, rxn in model.reactions.items():
    if not rxn_id.startswith("EX_"):
        rxn.upper_bound *= density_scale
        if rxn.lower_bound < 0:
            rxn.lower_bound *= density_scale

# Store FBA result on species for cross-tick persistence
sp.gem_fluxes = dict(fluxes)
```

---

## Phase X: Genome Evolution -> Re-run FBA ✅

### G10: CRISPR/Evolve Does Not Trigger GEM Update

**Problem**: `#crispr` and `#evolve` instructions modify DNA sequences and record
edits, but never trigger GEM re-reconstruction or FBA re-solve. A mutation that
changes an enzyme's kcat or knocks out a gene should immediately affect the
predicted growth rate.

**Fix**:

1. **Add `gem_dirty` flag to VM state**: Set to `True` after any CRISPR/evolve
   instruction that targets a gene in the GEM's GPR associations.

2. **In `_run_gem()` / `_run_ecosystem()` tick loop**: After processing bio
   instructions, check `gem_dirty` and re-solve FBA if set:
   ```python
   if vm._gem_dirty and vm._metabolic_model:
       from helixlang.metabolism import FluxBalanceAnalysis
       fba = FluxBalanceAnalysis(vm._metabolic_model)
       # Update enzyme levels from edited DNA
       vm._update_enzyme_levels_from_edits()
       new_fluxes = fba.solve()
       vm._growth_rate = new_fluxes.get(vm._metabolic_model.biomass_reaction, 0.0)
       vm._gem_dirty = False
   ```

3. **Add `_update_enzyme_levels_from_edits()` to VM**: For each edited gene,
   look up its GPR associations and adjust enzyme kcat:
   - Synonymous mutation: no change to kcat
   - Non-synonymous mutation: reduce kcat by 20-50% (based on Grantham distance)
   - Frameshift/nonsense: set kcat to 0 (gene knockout)

**Implementation in `vm.py`**:

```python
def _update_enzyme_levels_from_edits(self) -> None:
    """Update enzyme levels from CRISPR/evolve edits on GEM genes."""
    for edit in self._crispr_edits:
        if not edit.get("success", False):
            continue
        gene = edit.get("target", "")
        if gene not in self._gem_gpr_map:
            continue
        reactions = self._gem_gpr_map[gene]
        edit_type = edit.get("edit_type", "substitution")
        for rxn_id in reactions:
            if rxn_id in self._enzyme_kcat:
                if edit_type == "frameshift" or edit_type == "deletion":
                    self._enzyme_kcat[rxn_id] = 0.0  # knockout
                elif edit_type == "substitution":
                    self._enzyme_kcat[rxn_id] *= 0.7  # 30% reduction
    self._gem_dirty = False
```

---

## DSL Syntax Reference

```helix
# --- Gene + Enzyme with kinetic parameters (kcat, km) ---
#gene name=pgi promoter=strong ATG ... TAA #end
#enzyme gene=pgi reaction=PGI kcat=1200 km=0.3

# --- Direct reaction definition (DSL-authored metabolic network) ---
#reaction id=PGI name="Phosphoglucose isomerase" \
#  substrate=g6p product=f6p \
#  lower_bound=0 upper_bound=1000 \
#  subsystem=glycolysis reversible=true

# --- Per-gene expression level override (experimental data calibration) ---
#gene name=gltA expression_level=0.7 ATG ... TAA #end

# --- Per-reaction kcat/km override (experimental measurement) ---
#enzyme gene=gltA reaction=CS kcat=2800 km=0.04

# --- Environment temperature/pH direct declaration ---
#patch name=env temperature=30.0 ph=7.5

# --- Medium preset + partial override ---
#gem organism=e_coli_k12 medium=bg11 medium_override=fe3_e:0.5,co2_e:500

# --- Growth rate cap override ---
#gem organism=e_coli_k12 max_growth_rate=0.5

# --- GRN closed-loop (automatic when expression=true) ---
#gem organism=e_coli_k12 expression=true

# --- Genome evolution triggering GEM update ---
#crispr target=gltA position=100 new_sequence="ATG" cas=SpCas9
#evolve target=gltA generations=10 mutation_rate=0.01
```

### New DSL Extensions (added in this phase)

1. **`km=` in `#enzyme`**: Adds Michaelis constant to `EnzymeDecl` AST node and
   `EnzymeCapacity.km` dict.  Allows DSL authors to specify experimentally
   measured Km values for Monod-style uptake calculations.

2. **`#reaction` block**: New AST node `ReactionDecl` with fields `id`, `name`,
   `substrate`, `product`, `substrate_coeff`, `product_coeff`, `lower_bound`,
   `upper_bound`, `subsystem`, `reversible`.  Enables DSL-authored metabolic
   networks without a pre-built model file.  Collected in `Program.reactions`
   and built into a `MetabolicModel` by `_build_model_from_reactions()`.

## Modified Files (Phase VII-X)

| File | Changes |
|---|---|
| `ast_nodes.py` | Add `km` to `EnzymeDecl`; add `ReactionDecl`; add `reactions` to `Program` |
| `parser.py` | Parse `km=` in `#enzyme`; add `#reaction` block handler |
| `hxbc.py` | Serialize/deserialize `km` field |
| `metabolism.py` | Add `enzyme_correction()`; add `km` to `EnzymeCapacity` |
| `sim_runtime.py` | Wire `km` from DSL to `EnzymeCapacity`; add `_build_model_from_reactions()` |
| `gem/grn_inference.py` | Add `target_reaction` field to `RegulatoryEdge` |
| `gem/bridge.py` | Add `apply_regulatory_bounds()` function |
| `apps/ecosystem.py` | Wire temperature/pH + density correction into `_growth_rate_gem()` |
| `apps/ecosystem.py` | Add `last_fba_fluxes` to `Species`; deep-copy model before FBA |
| `vm.py` | Add `_update_enzyme_levels_from_edits()`, `gem_dirty` flag, re-FBA after edits |
