"""Bridge: convert GEM results into existing HelixLang data structures.

Connects gem/ reconstruction output to:
- metabolism.py: MetabolicModel, Reaction, EnzymeCapacity
- grn.py: GRN, GeneNode, Edge
- virtual_cell.py: per-organism enzyme constraints
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helixlang.gem.consensus import ConsensusResult
    from helixlang.gem.grn_inference import RegulatoryEdge
    from helixlang.kinetics.kcat_predictor import KcatPrediction


# ---------------------------------------------------------------------------
# GEM → MetabolicModel + EnzymeCapacity
# ---------------------------------------------------------------------------

def consensus_to_metabolic_model(
    consensus: ConsensusResult,
    biomass_rxn_id: str = "BIOMASS_reaction",
) -> Any:
    """Convert a ConsensusResult into a metabolism.MetabolicModel.

    Parameters
    ----------
    consensus : merged reconstruction result
    biomass_rxn_id : reaction ID for the biomass objective

    Returns
    -------
    MetabolicModel instance
    """
    from helixlang.metabolism import MetabolicModel, Reaction

    model = MetabolicModel()
    for rxn in consensus.reactions:
        stoich = _parse_equation_to_stoich(rxn.equation)
        if not stoich:
            continue
        # Exchange reactions (EX_ prefix, single metabolite) need
        # subsystem="exchange" so DynamicFluxBalance can apply uptake limits.
        # They always need lb=-1000 to allow import from the environment.
        is_exchange = rxn.reaction_id.startswith("EX_") and len(stoich) == 1
        if is_exchange:
            subsystem = "exchange"
            lb = -1000.0
        else:
            subsystem = "gem_reconstructed"
            lb = -1000.0 if rxn.confidence >= 0.6 else 0.0
        model.add_reaction(Reaction(
            id=rxn.reaction_id,
            name=rxn.reaction_id,
            stoichiometry=stoich,
            lower_bound=lb,
            upper_bound=1000.0,
            subsystem=subsystem,
        ))
    if biomass_rxn_id in {r.reaction_id for r in consensus.reactions}:
        model.set_biomass(biomass_rxn_id)
    return model


def build_enzyme_capacity(
    consensus: ConsensusResult,
    kcat_predictions: list[KcatPrediction],
    enzyme_scale: float = 1.0,
    protein_mass_fraction: float | None = None,
) -> Any:
    """Build an EnzymeCapacity from GEM GPR rules + kcat predictions.

    Parameters
    ----------
    consensus : merged reconstruction with GPR rules
    kcat_predictions : per-reaction kcat values
    enzyme_scale : global kcat rescaling factor
    protein_mass_fraction : sMOMENT global enzyme-pool budget (None = disabled)

    Returns
    -------
    EnzymeCapacity instance
    """
    from helixlang.metabolism import EnzymeCapacity

    # gene → reactions mapping from GPR rules
    gene_to_reactions: dict[str, tuple[str, ...]] = {}
    for rxn in consensus.reactions:
        if rxn.gpr and rxn.gpr.gene_ids:
            for gene in rxn.gpr.gene_ids:
                existing = gene_to_reactions.get(gene, ())
                gene_to_reactions[gene] = (*existing, rxn.reaction_id)

    # kcat mapping from predictions
    kcat: dict[str, float] = {}
    for pred in kcat_predictions:
        kcat[pred.reaction_id] = pred.kcat_value

    return EnzymeCapacity(
        gene_to_reactions=gene_to_reactions,
        kcat=kcat,
        enzyme_scale=enzyme_scale,
        protein_mass_fraction=protein_mass_fraction,
    )


def _parse_equation_to_stoich(equation: str) -> dict[str, float]:
    """Parse a simplified reaction equation into a stoichiometry dict.

    Supports formats:
      "A + 2 B -> C + D"
      "A <=> B"
      "A -> "  (exchange)
      ""       (empty → skip)
    """
    if not equation or not equation.strip():
        return {}

    stoich: dict[str, float] = {}
    if "<=>" in equation:
        parts = equation.split("<=>")
        side_mult = [-1.0, 1.0]
    elif "->" in equation:
        parts = equation.split("->")
        side_mult = [-1.0, 1.0]
    else:
        return {}

    for i, part in enumerate(parts):
        mult = side_mult[i]
        for term in part.split("+"):
            term = term.strip()
            if not term:
                continue
            tokens = term.split()
            if len(tokens) == 1:
                coeff, met = mult, tokens[0]
            elif len(tokens) == 2:
                try:
                    coeff = float(tokens[0]) * mult
                    met = tokens[1]
                except ValueError:
                    coeff, met = mult, term
            else:
                coeff, met = mult, term
            stoich[met] = stoich.get(met, 0.0) + coeff

    # Remove zero-coeff metabolites
    return {k: v for k, v in stoich.items() if abs(v) > 1e-12}


# ---------------------------------------------------------------------------
# RegulatoryEdge → GRN
# ---------------------------------------------------------------------------

def regulatory_edges_to_grn(
    edges: list[RegulatoryEdge],
    gene_names: list[str] | None = None,
    noise_enabled: bool = False,
) -> object:
    """Convert GEM regulatory edges into a grn.GRN instance.

    Parameters
    ----------
    edges : list of RegulatoryEdge from grn_inference
    gene_names : explicit gene list (if None, extracted from edges)
    noise_enabled : enable telegraph noise in GRN

    Returns
    -------
    GRN instance
    """
    from helixlang.grn import GRN

    grn = GRN(noise_enabled=noise_enabled)

    # Collect all gene names
    all_genes: set[str] = set()
    for e in edges:
        all_genes.add(e.tf_id)
        all_genes.add(e.target_gene)
    if gene_names:
        all_genes.update(gene_names)

    # Add gene nodes with default threshold
    for gene in sorted(all_genes):
        grn.add_gene(gene, threshold=0.5, initial_level=0.0)

    # Add edges
    for e in edges:
        weight = e.confidence * (1.0 if e.regulation_type == "activation" else -1.0)
        grn.add_edge(e.tf_id, e.target_gene, weight)

    return grn


# ---------------------------------------------------------------------------
# GPR rules → VirtualCell genome dict
# ---------------------------------------------------------------------------

def gpr_to_genome_dict(
    consensus: ConsensusResult,
) -> dict[str, str]:
    """Extract a gene-name → placeholder-sequence dict from GPR rules.

    For VirtualCell compatibility: each gene gets a placeholder sequence
    so the GRN can reference it. Real sequences would come from the genome FASTA.
    """
    genome: dict[str, str] = {}
    for rxn in consensus.reactions:
        if rxn.gpr and rxn.gpr.gene_ids:
            for gene in rxn.gpr.gene_ids:
                if gene not in genome:
                    genome[gene] = "ATG" + "NNN" * 10  # placeholder
    return genome


# ---------------------------------------------------------------------------
# VirtualCell factory from GEM
# ---------------------------------------------------------------------------

def build_virtual_cell_from_gem(
    consensus: ConsensusResult,
    kcat_predictions: list[KcatPrediction],
    grn_edges: list[RegulatoryEdge] | None = None,
    config: Any | None = None,
    name: str = "gem-cell",
) -> Any:
    """Create a VirtualCell from GEM reconstruction results.

    Parameters
    ----------
    consensus : merged reconstruction
    kcat_predictions : per-reaction kcat values
    grn_edges : regulatory edges (optional)
    config : VirtualCellConfig (optional, defaults created with enzyme_capacity_enabled)
    name : cell label

    Returns
    -------
    VirtualCell instance wired with GEM-derived model + kinetics
    """
    from helixlang.metabolism import FluxBalanceAnalysis
    from helixlang.virtual_cell import VirtualCell, VirtualCellConfig

    # Build MetabolicModel
    model = consensus_to_metabolic_model(consensus)
    fba = FluxBalanceAnalysis(model)

    # Build EnzymeCapacity from GPR + kcat
    enzyme_cap = build_enzyme_capacity(
        consensus, kcat_predictions, enzyme_scale=1.0
    )
    fba.set_enzyme_capacity(enzyme_cap)

    # Build GRN
    if grn_edges:
        grn_obj = regulatory_edges_to_grn(grn_edges)
    else:
        from helixlang.grn import GRN
        grn_obj = GRN()

    # Build genome dict
    genome = gpr_to_genome_dict(consensus)

    # Config with enzyme capacity enabled
    if config is None:
        cfg = VirtualCellConfig()
        cfg.enzyme_capacity_enabled = True
    else:
        cfg = config

    return VirtualCell(
        genome=genome,
        grn=grn_obj,  # type: ignore[arg-type]
        fba=fba,
        config=cfg,
        name=name,
    )


# ---------------------------------------------------------------------------
# build_functional_model — produces a working MetabolicModel from pipeline
# ---------------------------------------------------------------------------

def build_functional_model(
    consensus: ConsensusResult,
    gapfill: Any | None = None,
    organism: str = "e_coli_k12",
    medium: str = "glucose_minimal",
) -> Any:
    """Build a functional MetabolicModel from pipeline results.

    Unlike :func:`consensus_to_metabolic_model` which produces a dead
    model (biomass flux = 0 due to orphan metabolites and missing core
    metabolism), this function produces a model with positive growth rate:

    1. Creates the base model from consensus equations
    2. Adds gapfill exchange reactions
    3. Injects core metabolism (~137 reactions: glycolysis, TCA, PPP,
       ETC, 19 AA biosynthesis, nucleotide biosynthesis, cofactor
       transport, Calvin cycle + PET)
    4. Adds transport reactions bridging ``_e`` ↔ internal compartments
    5. Builds biomass with component filtering (only metabolites that
       exist in the model)
    6. Sets medium bounds (trace import capping, Calvin cycle closure)
    7. Solves FBA and returns the model with positive ``growth_rate``

    Parameters
    ----------
    consensus : merged reconstruction result
    gapfill : gapfill result (optional, adds exchange reactions)
    organism : organism identifier for biomass template selection
    medium : medium preset name

    Returns
    -------
    MetabolicModel ready for standalone FBA or ecosystem/population
    integration.  The model's ``biomass_reaction`` attribute is set.
    """
    from helixlang.metabolism import FluxBalanceAnalysis, Reaction

    # 1. Base model from consensus
    model = consensus_to_metabolic_model(consensus)

    # 2. Add gapfill exchange reactions (with lb=-1000 for import)
    if gapfill is not None and hasattr(gapfill, "added_reactions"):
        from helixlang.gem.gapfill import _parse_equation_to_stoich
        for rxn_entry in gapfill.added_reactions:
            rxn_id = rxn_entry.reaction_id
            if rxn_id in model.reactions:
                continue
            stoich = _parse_equation_to_stoich(rxn_entry.equation)
            if not stoich:
                continue
            is_exchange = rxn_id.startswith("EX_") and len(stoich) == 1
            model.add_reaction(Reaction(
                id=rxn_id, name=rxn_id, stoichiometry=stoich,
                lower_bound=-1000.0 if is_exchange else 0.0,
                upper_bound=1000.0,
                subsystem="exchange" if is_exchange else "gapfill",
            ))

    # 3. Inject core metabolism
    from helixlang.sim_runtime import (
        _add_gem_core_reactions,
        _add_gem_transport_reactions,
    )
    _add_gem_core_reactions(model)
    _add_gem_transport_reactions(model)

    # 4. Build biomass with component filtering
    from helixlang.gem.biomass import build_biomass_reaction
    biomass = build_biomass_reaction(organism)
    bm_stoich: dict[str, float] = {}
    for comp in biomass.components:
        raw = comp.metabolite_id
        stripped = raw.replace("_c", "").replace("_e", "").replace("_p", "")
        candidates = [raw, stripped, f"{stripped}_e"]
        matched = next(
            (m for m in candidates if m in model.metabolites), None)
        if matched is not None and abs(comp.coefficient) > 1e-12:
            bm_stoich[matched] = comp.coefficient

    if bm_stoich:
        model.add_reaction(Reaction(
            id="BIOMASS_reaction",
            name="BIOMASS_reaction",
            stoichiometry=bm_stoich,
            lower_bound=0.0,
            upper_bound=1000.0,
            subsystem="biomass",
        ))
        model.set_biomass("BIOMASS_reaction")

    # 5. Set medium bounds
    fba = FluxBalanceAnalysis(model)
    from helixlang.sim_runtime import _set_gem_medium
    _set_gem_medium(fba, medium, organism=organism, model=model)

    # 6. Solve FBA
    try:
        fluxes = fba.solve(objective="BIOMASS_reaction", maximize=True)
    except Exception:
        fluxes = {}

    # Attach growth rate to the model for downstream consumers
    model._growth_rate = fluxes.get("BIOMASS_reaction", 0.0)  # type: ignore[attr-defined]
    model._fba_fluxes = fluxes  # type: ignore[attr-defined]

    return model


# ---------------------------------------------------------------------------
# build_functional_model_full — full genome-scale model (doc/24 Phase D)
# ---------------------------------------------------------------------------

def build_functional_model_full(
    organism: str = "e_coli_k12",
    medium: str = "glucose_minimal",
    sbml_path: str | None = None,
) -> Any:
    """Build a functional MetabolicModel from a full genome-scale GEM.

    Unlike :func:`build_functional_model` which uses a 42-reaction core model
    with injected pathways, this function loads a complete genome-scale model
    (e.g. iML1515 with 2712 reactions) and applies organism-aware medium
    settings.

    Parameters
    ----------
    organism : organism identifier (must be in organism_registry)
    medium : medium preset name (e.g. "glucose_minimal", "bg11")
    sbml_path : optional path to a local SBML file (overrides BiGG download)

    Returns
    -------
    MetabolicModel ready for standalone FBA or ecosystem integration.
    Model has ``_growth_rate`` and ``_fba_fluxes`` attributes attached.
    """
    from helixlang.gem.full_model import FullModelAdapter

    if sbml_path:
        adapter = FullModelAdapter.from_sbml(sbml_path, organism)
    else:
        adapter = FullModelAdapter.from_bigg(organism)

    adapter.apply_medium(medium)
    fluxes = adapter.solve()
    adapter.model._growth_rate = adapter.growth_rate  # type: ignore[attr-defined]
    adapter.model._fba_fluxes = fluxes  # type: ignore[attr-defined]
    adapter.model._adapter = adapter  # type: ignore[attr-defined]
    return adapter.model
