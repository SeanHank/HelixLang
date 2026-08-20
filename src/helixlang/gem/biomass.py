"""Biomass reaction construction (doc/20 §6.2).

Taxonomy-aware biomass templates for prokaryotic GEM reconstruction.

Module structure:
    BiomassComponent         single precursor metabolite entry
    BiomassReaction          complete biomass reaction template
    ECOLI_BIOMASS_COMPONENTS E. coli K-12 biomass (gram-negative)
    BACILLUS_BIOMASS_COMPONENTS  B. subtilis biomass (gram-positive)
    SARCODINA_BIOMASS_COMPONENTS  archaeal biomass
    build_biomass_reaction   auto-detect organism and build reaction
    list_available_templates print available organism templates
    get_biomass_composition  look up components by organism
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BiomassComponent:
    """A single component of the biomass reaction."""

    metabolite_id: str
    coefficient: float
    category: str  # "amino_acid", "nucleotide", "lipid", "cofactor", "cell_wall"


@dataclass
class BiomassReaction:
    """Template biomass reaction for prokaryotic model."""

    name: str
    equation: str
    components: list[BiomassComponent] = field(default_factory=list)

    def to_string(self) -> str:
        reactants = []
        products = []
        for c in self.components:
            term = f"{abs(c.coefficient):.4f} {c.metabolite_id}"
            if c.coefficient < 0:
                reactants.append(term)
            else:
                products.append(term)
        return " + ".join(reactants) + " -> " + " + ".join(products)


# ---------------------------------------------------------------------------
# Taxonomy sets
# ---------------------------------------------------------------------------

GRAM_NEGATIVE_ORGANISMS: set[str] = {
    "e_coli",
    "e_coli_k12",
    "e_coli_mg1655",
    "p_aeruginosa",
    "s_typhimurium",
}

GRAM_POSITIVE_ORGANISMS: set[str] = {
    "b_subtilis",
    "s_aureus",
    "l_lactis",
}

ARCHAEA: set[str] = {
    "m_jannaschii",
    "s_solfataricus",
}

# All known organism keys (used for auto-detection)
_ALL_ORGANISMS: set[str] = (
    GRAM_NEGATIVE_ORGANISMS | GRAM_POSITIVE_ORGANISMS | ARCHAEA
)


# ---------------------------------------------------------------------------
# E. coli K-12 biomass composition (iML1515, simplified).
# Coefficients: mmol per gDW of cells.
# ---------------------------------------------------------------------------

ECOLI_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids
    BiomassComponent("ala-L_c", -0.5105, "amino_acid"),
    BiomassComponent("arg-L_c", -0.2775, "amino_acid"),
    BiomassComponent("asp-L_c", -0.2390, "amino_acid"),
    BiomassComponent("cys-L_c", -0.0838, "amino_acid"),
    BiomassComponent("glu-L_c", -0.2587, "amino_acid"),
    BiomassComponent("gln-L_c", -0.2513, "amino_acid"),
    BiomassComponent("gly_c", -0.5682, "amino_acid"),
    BiomassComponent("his-L_c", -0.0917, "amino_acid"),
    BiomassComponent("ile-L_c", -0.2660, "amino_acid"),
    BiomassComponent("leu-L_c", -0.4201, "amino_acid"),
    BiomassComponent("lys-L_c", -0.3140, "amino_acid"),
    BiomassComponent("met-L_c", -0.1461, "amino_acid"),
    BiomassComponent("phe-L_c", -0.1933, "amino_acid"),
    BiomassComponent("pro-L_c", -0.2107, "amino_acid"),
    BiomassComponent("ser-L_c", -0.2071, "amino_acid"),
    BiomassComponent("thr-L_c", -0.2462, "amino_acid"),
    BiomassComponent("trp-L_c", -0.0545, "amino_acid"),
    BiomassComponent("tyr-L_c", -0.1312, "amino_acid"),
    BiomassComponent("val-L_c", -0.3908, "amino_acid"),
    # Nucleotides (DNA)
    BiomassComponent("dATP_c", -0.0306, "nucleotide"),
    BiomassComponent("dTTP_c", -0.0306, "nucleotide"),
    BiomassComponent("dGTP_c", -0.0306, "nucleotide"),
    BiomassComponent("dCTP_c", -0.0306, "nucleotide"),
    # Nucleotides (RNA)
    BiomassComponent("ATP_c", -0.2331, "nucleotide"),
    BiomassComponent("UTP_c", -0.2151, "nucleotide"),
    BiomassComponent("GTP_c", -0.2509, "nucleotide"),
    BiomassComponent("CTP_c", -0.1883, "nucleotide"),
    # Lipids
    BiomassComponent("pgp120_c", -0.0080, "lipid"),
    BiomassComponent("pgp160_c", -0.0165, "lipid"),
    BiomassComponent("pgp161_c", -0.0289, "lipid"),
    BiomassComponent("pgp181_c", -0.0998, "lipid"),
    # Cofactors
    BiomassComponent("nad_c", -0.0012, "cofactor"),
    BiomassComponent("nadp_c", -0.0012, "cofactor"),
    BiomassComponent("coa_c", -0.0015, "cofactor"),
    BiomassComponent("thf_c", -0.0010, "cofactor"),
    BiomassComponent("thmpp_c", -0.0005, "cofactor"),
    BiomassComponent("pydx5p_c", -0.0005, "cofactor"),
    BiomassComponent("q8_c", -0.0062, "cofactor"),
    BiomassComponent("mql8_c", -0.0062, "cofactor"),
    BiomassComponent("2kgs_c", -0.0005, "cofactor"),
    # Cell wall (gram-negative: thin peptidoglycan + LPS/outer membrane)
    BiomassComponent("murein5px4pp_c", -0.0030, "cell_wall"),
    BiomassComponent("lipidA_c", -0.0027, "cell_wall"),
    BiomassComponent("glycogen_c", -0.0005, "cell_wall"),
    # Energy
    BiomassComponent("atp_c", -57.67, "energy"),
    BiomassComponent("h2o_c", -57.67, "energy"),
    BiomassComponent("adp_c", 57.67, "energy"),
    BiomassComponent("pi_c", 57.67, "energy"),
    BiomassComponent("h_c", 57.67, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


# ---------------------------------------------------------------------------
# B. subtilis biomass composition (gram-positive).
# Thick peptidoglycan, no outer membrane, teichoic acids present.
# Reference: iBsu1103 /GOBI (Nariya et al. 2011)
# ---------------------------------------------------------------------------

BACILLUS_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids (B. subtilis protein composition, Neidhardt 1976 adapted)
    BiomassComponent("ala-L_c", -0.5570, "amino_acid"),
    BiomassComponent("arg-L_c", -0.3210, "amino_acid"),
    BiomassComponent("asp-L_c", -0.2640, "amino_acid"),
    BiomassComponent("cys-L_c", -0.0680, "amino_acid"),
    BiomassComponent("glu-L_c", -0.3000, "amino_acid"),
    BiomassComponent("gln-L_c", -0.2900, "amino_acid"),
    BiomassComponent("gly_c", -0.6210, "amino_acid"),
    BiomassComponent("his-L_c", -0.1010, "amino_acid"),
    BiomassComponent("ile-L_c", -0.3010, "amino_acid"),
    BiomassComponent("leu-L_c", -0.4600, "amino_acid"),
    BiomassComponent("lys-L_c", -0.3630, "amino_acid"),
    BiomassComponent("met-L_c", -0.1610, "amino_acid"),
    BiomassComponent("phe-L_c", -0.2200, "amino_acid"),
    BiomassComponent("pro-L_c", -0.2410, "amino_acid"),
    BiomassComponent("ser-L_c", -0.2310, "amino_acid"),
    BiomassComponent("thr-L_c", -0.2740, "amino_acid"),
    BiomassComponent("trp-L_c", -0.0600, "amino_acid"),
    BiomassComponent("tyr-L_c", -0.1480, "amino_acid"),
    BiomassComponent("val-L_c", -0.4290, "amino_acid"),
    # Nucleotides (DNA) -- B. subtilis GC content ~43%
    BiomassComponent("dATP_c", -0.0271, "nucleotide"),
    BiomassComponent("dTTP_c", -0.0271, "nucleotide"),
    BiomassComponent("dGTP_c", -0.0345, "nucleotide"),
    BiomassComponent("dCTP_c", -0.0345, "nucleotide"),
    # Nucleotides (RNA)
    BiomassComponent("ATP_c", -0.2520, "nucleotide"),
    BiomassComponent("UTP_c", -0.1970, "nucleotide"),
    BiomassComponent("GTP_c", -0.2780, "nucleotide"),
    BiomassComponent("CTP_c", -0.1730, "nucleotide"),
    # Lipids (B. subtilis: mostly PG and PE)
    BiomassComponent("pgp160_c", -0.0120, "lipid"),
    BiomassComponent("pgp161_c", -0.0240, "lipid"),
    BiomassComponent("pgp180_c", -0.0080, "lipid"),
    BiomassComponent("pgp181_c", -0.0560, "lipid"),
    BiomassComponent("pe160_c", -0.0100, "lipid"),
    BiomassComponent("pe161_c", -0.0180, "lipid"),
    BiomassComponent("pe181_c", -0.0420, "lipid"),
    # Cofactors
    BiomassComponent("nad_c", -0.0010, "cofactor"),
    BiomassComponent("nadp_c", -0.0010, "cofactor"),
    BiomassComponent("coa_c", -0.0012, "cofactor"),
    BiomassComponent("thf_c", -0.0008, "cofactor"),
    BiomassComponent("thmpp_c", -0.0004, "cofactor"),
    BiomassComponent("pydx5p_c", -0.0004, "cofactor"),
    BiomassComponent("menaquinone_c", -0.0050, "cofactor"),
    BiomassComponent("demethylmenaquinone_c", -0.0040, "cofactor"),
    # Cell wall (gram-positive: thick peptidoglycan, teichoic/lipoteichoic acid)
    BiomassComponent("murein5px4pp_c", -0.0120, "cell_wall"),
    BiomassComponent("teichoic_acid_c", -0.0040, "cell_wall"),
    BiomassComponent("lipoteichoic_acid_c", -0.0020, "cell_wall"),
    BiomassComponent("glycogen_c", -0.0008, "cell_wall"),
    # Energy
    BiomassComponent("atp_c", -52.30, "energy"),
    BiomassComponent("h2o_c", -52.30, "energy"),
    BiomassComponent("adp_c", 52.30, "energy"),
    BiomassComponent("pi_c", 52.30, "energy"),
    BiomassComponent("h_c", 52.30, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


# ---------------------------------------------------------------------------
# Archaeal biomass composition (e.g. M. jannaschii / S. solfataricus).
# Ether-linked lipids, unique cofactors (F420, coenzyme M), no peptidoglycan
# (pseudopeptidoglycan or S-layer in most archaea).
# Reference: iMJ156 / iSul (Kanehisa 2014)
# ---------------------------------------------------------------------------

SARCODINA_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids (archaeal proteome; heavily weighted toward Glu/Asp)
    BiomassComponent("ala-L_c", -0.4200, "amino_acid"),
    BiomassComponent("arg-L_c", -0.2100, "amino_acid"),
    BiomassComponent("asp-L_c", -0.3200, "amino_acid"),
    BiomassComponent("cys-L_c", -0.0600, "amino_acid"),
    BiomassComponent("glu-L_c", -0.3800, "amino_acid"),
    BiomassComponent("gln-L_c", -0.3500, "amino_acid"),
    BiomassComponent("gly_c", -0.4800, "amino_acid"),
    BiomassComponent("his-L_c", -0.0750, "amino_acid"),
    BiomassComponent("ile-L_c", -0.2400, "amino_acid"),
    BiomassComponent("leu-L_c", -0.3700, "amino_acid"),
    BiomassComponent("lys-L_c", -0.2800, "amino_acid"),
    BiomassComponent("met-L_c", -0.1300, "amino_acid"),
    BiomassComponent("phe-L_c", -0.1750, "amino_acid"),
    BiomassComponent("pro-L_c", -0.1900, "amino_acid"),
    BiomassComponent("ser-L_c", -0.1800, "amino_acid"),
    BiomassComponent("thr-L_c", -0.2300, "amino_acid"),
    BiomassComponent("trp-L_c", -0.0480, "amino_acid"),
    BiomassComponent("tyr-L_c", -0.1200, "amino_acid"),
    BiomassComponent("val-L_c", -0.3500, "amino_acid"),
    BiomassComponent("pyrrolys_c", -0.0100, "amino_acid"),  # archaeal-specific
    BiomassComponent("selenocys_c", -0.0050, "amino_acid"),  # archaeal-specific
    # Nucleotides (DNA) -- M. jannaschii ~31% GC
    BiomassComponent("dATP_c", -0.0345, "nucleotide"),
    BiomassComponent("dTTP_c", -0.0345, "nucleotide"),
    BiomassComponent("dGTP_c", -0.0250, "nucleotide"),
    BiomassComponent("dCTP_c", -0.0250, "nucleotide"),
    # Nucleotides (RNA)
    BiomassComponent("ATP_c", -0.2100, "nucleotide"),
    BiomassComponent("UTP_c", -0.2250, "nucleotide"),
    BiomassComponent("GTP_c", -0.2400, "nucleotide"),
    BiomassComponent("CTP_c", -0.1950, "nucleotide"),
    # Lipids (archaeal: ether-linked isoprenoid chains, no ester-linked)
    BiomassComponent("archaeol_c", -0.0350, "lipid"),
    BiomassComponent("caldarchaeol_c", -0.0200, "lipid"),
    BiomassComponent("archaeal_pg_c", -0.0120, "lipid"),
    BiomassComponent("archaeal_pe_c", -0.0080, "lipid"),
    # Cofactors (archaeal-specific)
    BiomassComponent("nad_c", -0.0008, "cofactor"),
    BiomassComponent("nadp_c", -0.0008, "cofactor"),
    BiomassComponent("coa_c", -0.0010, "cofactor"),
    BiomassComponent("thf_c", -0.0007, "cofactor"),
    BiomassComponent("coenzyme_F420_c", -0.0020, "cofactor"),
    BiomassComponent("coenzyme_M_c", -0.0015, "cofactor"),
    BiomassComponent("coenzyme_B_c", -0.0015, "cofactor"),
    BiomassComponent("methanofuran_c", -0.0010, "cofactor"),
    BiomassComponent("methanopterin_c", -0.0010, "cofactor"),
    BiomassComponent("thmpp_c", -0.0003, "cofactor"),
    BiomassComponent("pydx5p_c", -0.0003, "cofactor"),
    # Cell wall (S-layer or pseudomurein, no peptidoglycan)
    BiomassComponent("pseudomurein_c", -0.0025, "cell_wall"),
    BiomassComponent("slayer_glycoprotein_c", -0.0015, "cell_wall"),
    BiomassComponent("glycogen_c", -0.0004, "cell_wall"),
    # Energy
    BiomassComponent("atp_c", -45.00, "energy"),
    BiomassComponent("h2o_c", -45.00, "energy"),
    BiomassComponent("adp_c", 45.00, "energy"),
    BiomassComponent("pi_c", 45.00, "energy"),
    BiomassComponent("h_c", 45.00, "energy"),
    BiomassComponent("coenzyme_F420_red_c", 45.00, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


# ---------------------------------------------------------------------------
# Organism-to-template mapping
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[BiomassComponent]] = {
    # Gram-negative
    "e_coli": ECOLI_BIOMASS_COMPONENTS,
    "e_coli_k12": ECOLI_BIOMASS_COMPONENTS,
    "e_coli_mg1655": ECOLI_BIOMASS_COMPONENTS,
    "p_aeruginosa": ECOLI_BIOMASS_COMPONENTS,  # closest template
    "s_typhimurium": ECOLI_BIOMASS_COMPONENTS,
    # Gram-positive
    "b_subtilis": BACILLUS_BIOMASS_COMPONENTS,
    "s_aureus": BACILLUS_BIOMASS_COMPONENTS,
    "l_lactis": BACILLUS_BIOMASS_COMPONENTS,
    # Archaea
    "m_jannaschii": SARCODINA_BIOMASS_COMPONENTS,
    "s_solfataricus": SARCODINA_BIOMASS_COMPONENTS,
}


# ---------------------------------------------------------------------------
# Organism classification helpers
# ---------------------------------------------------------------------------

def _classify_organism(organism: str) -> str:
    """Return ``'gram_negative'``, ``'gram_positive'``, or ``'archaea'``.

    Falls back to ``'gram_negative'`` for unrecognised names (conservative
    default matching E. coli-centric tooling).
    """
    key = organism.lower().replace(" ", "_").replace("-", "_")
    if key in ARCHAEA:
        return "archaea"
    if key in GRAM_POSITIVE_ORGANISMS:
        return "gram_positive"
    return "gram_negative"


def _resolve_organism_key(organism: str) -> str | None:
    """Resolve a freeform organism name to a known template key, or None."""
    key = organism.lower().replace(" ", "_").replace("-", "_")
    if key in _TEMPLATES:
        return key
    # Partial match: return first template key that contains the search term
    for tmpl_key in _TEMPLATES:
        if key in tmpl_key or tmpl_key in key:
            return tmpl_key
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_biomass_composition(
    organism: str,
) -> list[BiomassComponent]:
    """Return the biomass component list for *organism*.

    Parameters
    ----------
    organism:
        Freeform organism name, identifier, or partial match.

    Returns
    -------
    list[BiomassComponent]
        Copy of the matching component list.  Falls back to E. coli
        components if no match is found.
    """
    key = _resolve_organism_key(organism)
    if key is not None:
        return list(_TEMPLATES[key])
    return list(ECOLI_BIOMASS_COMPONENTS)


def list_available_templates() -> dict[str, str]:
    """Return a mapping of template key -> organism classification.

    Example output::

        {"e_coli": "gram_negative", "b_subtilis": "gram_positive", ...}
    """
    return {k: _classify_organism(k) for k in sorted(_TEMPLATES)}


def build_biomass_reaction(
    organism: str = "e_coli_k12",
    custom_composition: list[BiomassComponent] | None = None,
) -> BiomassReaction:
    """Build a biomass reaction for the organism.

    Auto-detects the organism type (gram-negative, gram-positive, archaea)
    and selects the most appropriate template when available.

    Parameters
    ----------
    organism:
        Target organism (e.g. ``"e_coli_k12"``, ``"b_subtilis"``).
    custom_composition:
        Override with a custom list of :class:`BiomassComponent`.

    Returns
    -------
    BiomassReaction
    """
    if custom_composition is not None:
        components = custom_composition
    else:
        components = get_biomass_composition(organism)

    eq = " -> ".join([
        " + ".join(
            f"{abs(c.coefficient):.4f} {c.metabolite_id}"
            for c in components if c.coefficient < 0
        ),
        " + ".join(
            f"{c.coefficient:.4f} {c.metabolite_id}"
            for c in components if c.coefficient > 0
        ),
    ])

    org_type = _classify_organism(organism)
    reaction_name = f"BIOMASS_reaction_{org_type}"

    return BiomassReaction(
        name=reaction_name,
        equation=eq,
        components=components,
    )
