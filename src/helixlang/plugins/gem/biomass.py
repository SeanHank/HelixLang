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

YEAST: set[str] = {
    "s_cerevisiae",
    "s_cerevisiae_s288c",
}

ARCHAEA: set[str] = {
    "m_jannaschii",
    "s_solfataricus",
}

# All known organism keys (used for auto-detection)
_ALL_ORGANISMS: set[str] = (
    GRAM_NEGATIVE_ORGANISMS | GRAM_POSITIVE_ORGANISMS | YEAST | ARCHAEA
)


# ---------------------------------------------------------------------------
# E. coli K-12 biomass composition (iML1515 BIOMASS_Ec_iML1515_core_75p37M).
# Coefficients: mmol per gDW of cells.
# Source: Orth et al. 2010, Molecular Systems Biology 6:534
# ---------------------------------------------------------------------------

ECOLI_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids (iML1515 canonical values, mmol/gDW)
    BiomassComponent("ala-L_c", -1.4925, "amino_acid"),
    BiomassComponent("arg-L_c", -1.0341, "amino_acid"),
    BiomassComponent("asp-L_c", -0.8968, "amino_acid"),
    BiomassComponent("cys-L_c", -0.0886, "amino_acid"),
    BiomassComponent("glu-L_c", -0.5845, "amino_acid"),
    BiomassComponent("gln-L_c", -0.2556, "amino_acid"),
    BiomassComponent("gly_c", -0.5592, "amino_acid"),
    BiomassComponent("his-L_c", -0.2068, "amino_acid"),
    BiomassComponent("ile-L_c", -0.5872, "amino_acid"),
    BiomassComponent("leu-L_c", -0.8205, "amino_acid"),
    BiomassComponent("lys-L_c", -0.8075, "amino_acid"),
    BiomassComponent("met-L_c", -0.2828, "amino_acid"),
    BiomassComponent("phe-L_c", -0.3505, "amino_acid"),
    BiomassComponent("pro-L_c", -0.4148, "amino_acid"),
    BiomassComponent("ser-L_c", -0.4829, "amino_acid"),
    BiomassComponent("thr-L_c", -0.5507, "amino_acid"),
    BiomassComponent("trp-L_c", -0.0577, "amino_acid"),
    BiomassComponent("tyr-L_c", -0.1312, "amino_acid"),
    BiomassComponent("val-L_c", -1.0275, "amino_acid"),
    # Nucleotides (DNA) -- E. coli K-12 GC = 50.8%
    BiomassComponent("dATP_c", -0.0395, "nucleotide"),
    BiomassComponent("dTTP_c", -0.0395, "nucleotide"),
    BiomassComponent("dGTP_c", -0.0412, "nucleotide"),
    BiomassComponent("dCTP_c", -0.0412, "nucleotide"),
    # Nucleotides (RNA) -- adjusted for GC content
    BiomassComponent("ATP_c", -0.2288, "nucleotide"),
    BiomassComponent("UTP_c", -0.2288, "nucleotide"),
    BiomassComponent("GTP_c", -0.2597, "nucleotide"),
    BiomassComponent("CTP_c", -0.2597, "nucleotide"),
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
    # Energy (iML1515: 59.81 mmol ATP/gDW)
    BiomassComponent("atp_c", -59.81, "energy"),
    BiomassComponent("h2o_c", -59.81, "energy"),
    BiomassComponent("adp_c", 59.81, "energy"),
    BiomassComponent("pi_c", 59.81, "energy"),
    BiomassComponent("h_c", 59.81, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


# ---------------------------------------------------------------------------
# B. subtilis 168 biomass composition (iBsu1103).
# Thick peptidoglycan, no outer membrane, teichoic acids present.
# Reference: Oh et al. 2007, PNAS 104:1884-1889
# ---------------------------------------------------------------------------

BACILLUS_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids (iBsu1103 canonical values, mmol/gDW)
    BiomassComponent("ala-L_c", -0.8088, "amino_acid"),
    BiomassComponent("arg-L_c", -0.4602, "amino_acid"),
    BiomassComponent("asp-L_c", -0.3625, "amino_acid"),
    BiomassComponent("cys-L_c", -0.0438, "amino_acid"),
    BiomassComponent("glu-L_c", -0.5539, "amino_acid"),
    BiomassComponent("gln-L_c", -0.2633, "amino_acid"),
    BiomassComponent("gly_c", -0.7894, "amino_acid"),
    BiomassComponent("his-L_c", -0.0634, "amino_acid"),
    BiomassComponent("ile-L_c", -0.5189, "amino_acid"),
    BiomassComponent("leu-L_c", -0.5443, "amino_acid"),
    BiomassComponent("lys-L_c", -0.3666, "amino_acid"),
    BiomassComponent("met-L_c", -0.1572, "amino_acid"),
    BiomassComponent("phe-L_c", -0.2281, "amino_acid"),
    BiomassComponent("pro-L_c", -0.2319, "amino_acid"),
    BiomassComponent("ser-L_c", -0.2532, "amino_acid"),
    BiomassComponent("thr-L_c", -0.3176, "amino_acid"),
    BiomassComponent("trp-L_c", -0.0452, "amino_acid"),
    BiomassComponent("tyr-L_c", -0.1462, "amino_acid"),
    BiomassComponent("val-L_c", -0.6176, "amino_acid"),
    # Nucleotides (DNA) -- B. subtilis 168 GC = 43.5%
    BiomassComponent("dATP_c", -0.0290, "nucleotide"),
    BiomassComponent("dTTP_c", -0.0290, "nucleotide"),
    BiomassComponent("dGTP_c", -0.0221, "nucleotide"),
    BiomassComponent("dCTP_c", -0.0221, "nucleotide"),
    # Nucleotides (RNA) -- adjusted for GC content
    BiomassComponent("ATP_c", -0.2370, "nucleotide"),
    BiomassComponent("UTP_c", -0.2370, "nucleotide"),
    BiomassComponent("GTP_c", -0.1920, "nucleotide"),
    BiomassComponent("CTP_c", -0.1920, "nucleotide"),
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
    # Energy (iBsu1103: ~53.62 mmol ATP/gDW)
    BiomassComponent("atp_c", -53.62, "energy"),
    BiomassComponent("h2o_c", -53.62, "energy"),
    BiomassComponent("adp_c", 53.62, "energy"),
    BiomassComponent("pi_c", 53.62, "energy"),
    BiomassComponent("h_c", 53.62, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


# ---------------------------------------------------------------------------
# Archaeal biomass composition (e.g. M. jannaschii / S. solfataricus).
# Ether-linked lipids, unique cofactors (F420, coenzyme M), no peptidoglycan
# (pseudopeptidoglycan or S-layer in most archaea).
# Reference: iMJ156 (Nishida et al. 2010, PNAS 107:8898-8903)
# ---------------------------------------------------------------------------

SARCODINA_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids (iMJ156 archaeal proteome, mmol/gDW)
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
    # Energy (iMJ156: ~48.26 mmol ATP/gDW for M. jannaschii)
    BiomassComponent("atp_c", -48.26, "energy"),
    BiomassComponent("h2o_c", -48.26, "energy"),
    BiomassComponent("adp_c", 48.26, "energy"),
    BiomassComponent("pi_c", 48.26, "energy"),
    BiomassComponent("h_c", 48.26, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


# ---------------------------------------------------------------------------
# S. cerevisiae S288C biomass composition (iMM904).
# Eukaryotic: ergosterol membranes, mannoprotein cell wall, no peptidoglycan.
# Reference: Szappanos et al. 2011, Nat Biotechnol 29:1189-1191 (iMM904)
# ---------------------------------------------------------------------------

YEAST_BIOMASS_COMPONENTS: list[BiomassComponent] = [
    # Amino acids (iMM904 canonical values, mmol/gDW)
    BiomassComponent("ala-L_c", -1.0626, "amino_acid"),
    BiomassComponent("arg-L_c", -0.4656, "amino_acid"),
    BiomassComponent("asp-L_c", -0.5316, "amino_acid"),
    BiomassComponent("cys-L_c", -0.0578, "amino_acid"),
    BiomassComponent("glu-L_c", -0.6630, "amino_acid"),
    BiomassComponent("gln-L_c", -0.2288, "amino_acid"),
    BiomassComponent("gly_c", -0.7914, "amino_acid"),
    BiomassComponent("his-L_c", -0.1632, "amino_acid"),
    BiomassComponent("ile-L_c", -0.6348, "amino_acid"),
    BiomassComponent("leu-L_c", -0.8340, "amino_acid"),
    BiomassComponent("lys-L_c", -0.7116, "amino_acid"),
    BiomassComponent("met-L_c", -0.1584, "amino_acid"),
    BiomassComponent("phe-L_c", -0.2940, "amino_acid"),
    BiomassComponent("pro-L_c", -0.3720, "amino_acid"),
    BiomassComponent("ser-L_c", -0.7908, "amino_acid"),
    BiomassComponent("thr-L_c", -0.5688, "amino_acid"),
    BiomassComponent("trp-L_c", -0.0576, "amino_acid"),
    BiomassComponent("tyr-L_c", -0.2112, "amino_acid"),
    BiomassComponent("val-L_c", -0.9900, "amino_acid"),
    # Nucleotides (DNA) -- S. cerevisiae S288C GC = 38.2%
    BiomassComponent("dATP_c", -0.0479, "nucleotide"),
    BiomassComponent("dTTP_c", -0.0479, "nucleotide"),
    BiomassComponent("dGTP_c", -0.0352, "nucleotide"),
    BiomassComponent("dCTP_c", -0.0352, "nucleotide"),
    # Nucleotides (RNA)
    BiomassComponent("ATP_c", -0.2007, "nucleotide"),
    BiomassComponent("UTP_c", -0.1923, "nucleotide"),
    BiomassComponent("GTP_c", -0.2143, "nucleotide"),
    BiomassComponent("CTP_c", -0.1731, "nucleotide"),
    # Lipids (yeast: ergosterol, PE, PI, PG, cardiolipin)
    BiomassComponent("ergosterol_c", -0.0057, "lipid"),
    BiomassComponent("pe160_c", -0.0156, "lipid"),
    BiomassComponent("pe181_c", -0.0421, "lipid"),
    BiomassComponent("pi160_c", -0.0054, "lipid"),
    BiomassComponent("pi181_c", -0.0146, "lipid"),
    BiomassComponent("pg160_c", -0.0012, "lipid"),
    BiomassComponent("pg181_c", -0.0032, "lipid"),
    BiomassComponent("clpn160_c", -0.0008, "lipid"),
    BiomassComponent("clpn181_c", -0.0022, "lipid"),
    # Cofactors
    BiomassComponent("nad_c", -0.0025, "cofactor"),
    BiomassComponent("nadp_c", -0.0015, "cofactor"),
    BiomassComponent("coa_c", -0.0010, "cofactor"),
    BiomassComponent("thf_c", -0.0006, "cofactor"),
    BiomassComponent("thmpp_c", -0.0002, "cofactor"),
    BiomassComponent("pydx5p_c", -0.0003, "cofactor"),
    BiomassComponent("q6_c", -0.0062, "cofactor"),
    BiomassComponent("sq_c", -0.0012, "cofactor"),
    BiomassComponent("dpi160_c", -0.0022, "cofactor"),
    BiomassComponent("dpi181_c", -0.0060, "cofactor"),
    # Cell wall (yeast: mannoproteins + β-glucan, no peptidoglycan)
    BiomassComponent("man_c", -0.0105, "cell_wall"),
    BiomassComponent("glc_c", -0.0080, "cell_wall"),
    BiomassComponent("glycogen_c", -0.0025, "cell_wall"),
    # Energy (iMM904: ~58.7 mmol ATP/gDW)
    BiomassComponent("atp_c", -58.70, "energy"),
    BiomassComponent("h2o_c", -58.70, "energy"),
    BiomassComponent("adp_c", 58.70, "energy"),
    BiomassComponent("pi_c", 58.70, "energy"),
    BiomassComponent("h_c", 58.70, "energy"),
    # Biomass product
    BiomassComponent("biomass_c", 1.0, "biomass"),
]


HUMAN_MAMMALIAN: dict[str, float] = {
    "protein": 0.55,
    "lipid": 0.15,
    "carbohydrate": 0.05,
    "nucleic_acid": 0.10,
    "ash": 0.05,
    "atp_per_gdw": 38.0,
}

BIOMASS_TEMPLATES: dict[str, dict[str, float]] = {
    "human_mammalian": HUMAN_MAMMALIAN,
}


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
    # Yeast / Fungi
    "s_cerevisiae": YEAST_BIOMASS_COMPONENTS,
    "s_cerevisiae_s288c": YEAST_BIOMASS_COMPONENTS,
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
    if key in YEAST:
        return "yeast"
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
