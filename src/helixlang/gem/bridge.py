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
        model.add_reaction(Reaction(
            id=rxn.reaction_id,
            name=rxn.reaction_id,
            stoichiometry=stoich,
            lower_bound=-1000.0 if rxn.confidence >= 0.6 else 0.0,
            upper_bound=1000.0,
            subsystem="gem_reconstructed",
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
