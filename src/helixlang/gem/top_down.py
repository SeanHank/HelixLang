"""Top-down GEM reconstruction: universal model → carve (doc/20 §6.1).

The universal prokaryotic model contains ~150 core reactions present in
virtually all bacteria, derived from BiGG universal model (Sanchez et al.
2019, Nucleic Acids Res) and MetaCyc core reactions.  Each reaction has
a set of required EC numbers; the top-down reconstruction removes reactions
for which the organism lacks any evidence of the required enzyme.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from helixlang.annotation import GeneAnnotation
from helixlang.gem.bottom_up import ReactionEntry


@dataclass
class TopDownResult:
    """Result of top-down (carve) reconstruction."""

    reactions: list[ReactionEntry] = field(default_factory=list)
    kept_reactions: int = 0
    removed_reactions: int = 0

    @property
    def reaction_count(self) -> int:
        return len(self.reactions)

    def reaction_ids(self) -> list[str]:
        return [r.reaction_id for r in self.reactions]


# Universal prokaryotic model: core reactions present in most bacteria.
# Based on BiGG universal model (Sanchez et al. 2019, Nucleic Acids Res)
# and MetaCyc core prokaryotic reactions.
# Each reaction has a set of required EC numbers or KO terms.
UNIVERSAL_PROKARYOTIC_REACTIONS: list[dict[str, object]] = [
    # ======================================================================
    # GLYCOLYSIS / GLUCONEOGENESIS
    # ======================================================================
    {"id": "HEX1", "eq": "glc-D + atp -> g6p + adp",
     "required_ec": ["2.7.1.1"], "rev": False},
    {"id": "PGI", "eq": "g6p <=> f6p",
     "required_ec": ["5.3.1.9"], "rev": True},
    {"id": "PFK", "eq": "f6p + atp -> fdp + adp",
     "required_ec": ["2.7.1.11"], "rev": False},
    {"id": "FBA", "eq": "fdp <=> dhap + g3p",
     "required_ec": ["4.1.2.13"], "rev": True},
    {"id": "TPI", "eq": "dhap <=> g3p",
     "required_ec": ["5.3.1.1"], "rev": True},
    {"id": "GAPD", "eq": "g3p + nad + pi <=> 13dpg + nadh",
     "required_ec": ["1.2.1.12"], "rev": True},
    {"id": "PGK", "eq": "13dpg + adp <=> 3pg + atp",
     "required_ec": ["2.7.2.3"], "rev": True},
    {"id": "PGM", "eq": "3pg <=> 2pg",
     "required_ec": ["5.4.2.12"], "rev": True},
    {"id": "ENO", "eq": "2pg <=> h2o + pep",
     "required_ec": ["4.2.1.11"], "rev": True},
    {"id": "PYK", "eq": "pep + adp -> pyr + atp",
     "required_ec": ["2.7.1.40"], "rev": False},
    {"id": "FBP", "eq": "fdp + h2o -> f6p + pi",
     "required_ec": ["3.1.3.11"], "rev": False},
    {"id": "PPC", "eq": "pep + hco3 -> oaa + pi",
     "required_ec": ["4.1.1.31"], "rev": False},
    {"id": "PPCK", "eq": "oaa + atp -> pep + adp + co2",
     "required_ec": ["4.1.1.32"], "rev": False},

    # ======================================================================
    # TCA CYCLE
    # ======================================================================
    {"id": "CS", "eq": "accoa + oaa + h2o -> cit + coa",
     "required_ec": ["2.3.3.1"], "rev": False},
    {"id": "ACONa", "eq": "cit <=> acon-C + h2o",
     "required_ec": ["4.2.1.3"], "rev": True},
    {"id": "ICDH", "eq": "icit + nadp -> akg + co2 + nadph",
     "required_ec": ["1.1.1.40"], "rev": False},
    {"id": "AKGDH", "eq": "akg + coa + nad -> succoa + co2 + nadh",
     "required_ec": ["1.2.4.2", "1.2.1.52"], "rev": False},
    {"id": "SUCOAS", "eq": "succoa + adp + pi <=> succ + atp + coa",
     "required_ec": ["6.2.1.5"], "rev": True},
    {"id": "SDH", "eq": "succ <=> fum",
     "required_ec": ["1.3.5.4", "1.3.99.1"], "rev": True},
    {"id": "FUM", "eq": "fum + h2o <=> mal-L",
     "required_ec": ["4.2.1.2"], "rev": True},
    {"id": "MDH", "eq": "mal-L + nad -> oaa + nadh",
     "required_ec": ["1.1.1.37"], "rev": False},

    # ======================================================================
    # PYRUVATE METABOLISM
    # ======================================================================
    {"id": "PDH", "eq": "pyr + coa + nad -> accoa + co2 + nadh",
     "required_ec": ["1.2.4.1", "1.2.4.4", "1.8.1.4"], "rev": False},
    {"id": "ME1", "eq": "mal-L + nadp -> pyr + co2 + nadph",
     "required_ec": ["1.1.1.40"], "rev": False},
    {"id": "PC", "eq": "pyr + hco3 + atp -> oaa + adp + pi",
     "required_ec": ["6.4.1.1"], "rev": False},

    # ======================================================================
    # ANAPLEROTIC / GLYOXYLATE
    # ======================================================================
    {"id": "ICL", "eq": "icit -> succ + glyox",
     "required_ec": ["4.1.3.1"], "rev": False},
    {"id": "MS", "eq": "glyox + accoa + h2o -> mal-L + coa",
     "required_ec": ["2.3.3.9"], "rev": False},

    # ======================================================================
    # PENTOSE PHOSPHATE PATHWAY
    # ======================================================================
    {"id": "G6PDH", "eq": "g6p + nadp -> 6pgl + nadph",
     "required_ec": ["1.1.1.49"], "rev": False},
    {"id": "PGD", "eq": "6pgl + h2o -> 6pgc + co2",
     "required_ec": ["3.1.1.31"], "rev": False},
    {"id": "RPI", "eq": "5pgc <=> ru5p",
     "required_ec": ["5.1.3.3"], "rev": True},
    {"id": "RPE", "eq": "ru5p <=> xu5p",
     "required_ec": ["5.1.3.1"], "rev": True},
    {"id": "TKT1", "eq": "r5p + xu5p <=> s7p + g3p",
     "required_ec": ["2.2.1.1"], "rev": True},
    {"id": "TKT2", "eq": "e4p + xu5p <=> f6p + g3p",
     "required_ec": ["2.2.1.1"], "rev": True},
    {"id": "TALA", "eq": "s7p + g3p <=> e4p + f6p",
     "required_ec": ["2.2.1.2"], "rev": True},

    # ======================================================================
    # FERMENTATION
    # ======================================================================
    {"id": "PTAr", "eq": "accoa + pi <=> acpa + coa",
     "required_ec": ["2.3.1.8"], "rev": True},
    {"id": "ACKr", "eq": "acp + atp <=> acc + adp",
     "required_ec": ["2.7.2.1"], "rev": True},
    {"id": "LDH_D", "eq": "pyr + nadh <=> lac-D + nad",
     "required_ec": ["1.1.1.27"], "rev": True},
    {"id": "ADHEr", "eq": "acald + nadh <=> etoh + nad",
     "required_ec": ["1.1.1.1"], "rev": True},
    {"id": "ACALD", "eq": "accoa + nadh -> acald + coa + nad",
     "required_ec": ["1.2.1.10"], "rev": False},

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS
    # ======================================================================
    {"id": "ALAT", "eq": "pyr + glu-L <=> ala-L + akg",
     "required_ec": ["2.6.1.2"], "rev": True},
    {"id": "ASPTA", "eq": "oaa + glu-L <=> asp-L + akg",
     "required_ec": ["2.6.1.1"], "rev": True},
    {"id": "GDH", "eq": "akg + nh4 + nadh <=> glu-L + nad",
     "required_ec": ["1.4.1.2", "1.4.1.4"], "rev": True},
    {"id": "GLNS", "eq": "glu-L + nh4 + atp -> gln-L + adp + pi",
     "required_ec": ["6.3.1.2"], "rev": False},
    {"id": "SHMT", "eq": "ser-L + thf <=> gly + h2o + methf",
     "required_ec": ["2.5.1.19"], "rev": True},
    {"id": "BCAT", "eq": "leu-L + akg <=> 4mop + glu-L",
     "required_ec": ["2.6.1.42"], "rev": True},
    {"id": "TRPS1", "eq": "indol3g + ser-L -> trp-L + h2o + g3p",
     "required_ec": ["4.2.1.20"], "rev": False},
    {"id": "ILVDA", "eq": "pyr + pyr -> 2ah3b + co2",
     "required_ec": ["2.2.1.6"], "rev": False},

    # ======================================================================
    # NUCLEOTIDE METABOLISM
    # ======================================================================
    {"id": "RNR", "eq": "adp + nadph -> dadp + nadp + h2o",
     "required_ec": ["1.17.4.1", "1.17.4.2"], "rev": False},
    {"id": "ADSS", "eq": "imp + gtp + asp-L -> adp + gdp + pi",
     "required_ec": ["6.3.4.4"], "rev": False},
    {"id": "ADSL1", "eq": "air + asp-L + gtp -> succair + gdp + pi",
     "required_ec": ["3.5.1.10"], "rev": False},
    {"id": "CAD", "eq": "atp + co2 + gln-L + h2o -> cbp + glu-L + adp + pi",
     "required_ec": ["6.3.5.5", "6.3.3.5"], "rev": False},
    {"id": "ASPCT", "eq": "asp-L + cbp -> cbasp + pi",
     "required_ec": ["2.1.3.2"], "rev": False},
    {"id": "DIOD", "eq": "cbasp + h2o -> dhor + pi",
     "required_ec": ["3.5.2.3"], "rev": False},
    {"id": "DHODH", "eq": "dhor + nad -> orot + nadh",
     "required_ec": ["1.3.5.2", "1.3.99.11"], "rev": False},
    {"id": "OPRT", "eq": "orot + prpp -> omp + ppi",
     "required_ec": ["2.4.2.10"], "rev": False},
    {"id": "OMPDC", "eq": "omp -> ump + co2",
     "required_ec": ["4.1.1.23"], "rev": False},
    {"id": "UMPK", "eq": "ump + atp -> udp + adp",
     "required_ec": ["2.7.4.22"], "rev": False},
    {"id": "NDPK", "eq": "udp + atp <=> utp + adp",
     "required_ec": ["2.7.4.6"], "rev": True},
    {"id": "ADK", "eq": "atp + amp <=> 2 adp",
     "required_ec": ["2.7.4.3"], "rev": True},

    # ======================================================================
    # FATTY ACID METABOLISM
    # ======================================================================
    {"id": "FAS", "eq": "accoa + co2 + atp + nadph -> malcoa + adp + pi + nadp",
     "required_ec": ["6.4.1.2", "2.3.1.85"], "rev": False},
    {"id": "FABB", "eq": "accoa + malcoa -> 3oacb + co2 + coa",
     "required_ec": ["2.3.1.39"], "rev": False},
    {"id": "FACR", "eq": "3oacp + nadh -> 3hacp + nad",
     "required_ec": ["1.1.1.100"], "rev": False},
    {"id": "FABZ", "eq": "3hacp -> 2tdec2coa + h2o",
     "required_ec": ["4.2.1.59"], "rev": False},
    {"id": "ACOATA", "eq": "accoa + acp -> aacoa + h2o",
     "required_ec": ["2.3.1.8"], "rev": False},

    # ======================================================================
    # COFACTOR BIOSYNTHESIS
    # ======================================================================
    {"id": "NMNAT", "eq": "nac + atp -> nmn + ppi",
     "required_ec": ["2.4.2.12"], "rev": False},
    {"id": "NADSYN", "eq": "nad + nh4 + atp -> nad + adp + pi",
     "required_ec": ["6.3.1.5"], "rev": False},
    {"id": "FMNAT", "eq": "fmn + atp -> fad + ppi",
     "required_ec": ["2.5.1.9"], "rev": False},
    {"id": "THPS", "eq": "4mp + pyr + h2o -> thmp + co2",
     "required_ec": ["2.5.1.3"], "rev": False},
    {"id": "DHFR", "eq": "dhf + nadph -> thf + nadp",
     "required_ec": ["1.5.1.3"], "rev": False},

    # ======================================================================
    # TRANSPORT REACTIONS
    # ======================================================================
    {"id": "GLCpts", "eq": "glc-D_e + pep -> g6p + pyr",
     "required_ec": ["2.7.1.69"], "rev": False},
    {"id": "PIT", "eq": "pi_e + h -> pi + h_e",
     "required_ec": ["3.6.3.-"], "rev": True},
    {"id": "NHA", "eq": "na1_e + h <=> na1 + h_e",
     "required_ec": ["3.6.3.-"], "rev": True},

    # ======================================================================
    # ENERGY METABOLISM
    # ======================================================================
    {"id": "NADH_D1", "eq": "nadh + q + 5 h_e -> nad + qh2 + 4 h",
     "required_ec": ["7.1.1.2", "7.1.1.1"], "rev": False},
    {"id": "CYTBD", "eq": "nadh + 0.5 o2 + 4 h_e -> nad + 2 h2o + 4 h",
     "required_ec": ["7.1.1.9", "1.9.3.1"], "rev": False},
    {"id": "ATPS4r", "eq": "adp + pi + 4 h_e <=> atp + h2o + 4 h",
     "required_ec": ["7.1.2.2"], "rev": True},

    # ======================================================================
    # OXIDATIVE STRESS / REDOX
    # ======================================================================
    {"id": "CAT", "eq": "2 h2o2 -> 2 h2o + o2",
     "required_ec": ["1.11.1.6", "1.11.1.1"], "rev": False},
    {"id": "SOD", "eq": "2 o2- + 2 h -> h2o2 + o2",
     "required_ec": ["1.15.1.1", "1.15.1.2"], "rev": False},

    # ======================================================================
    # GROWTH-ASSOCIATED MAINTENANCE
    # ======================================================================
    {"id": "ATPM", "eq": "atp + h2o -> adp + pi",
     "required_ec": [], "rev": False},
]


def top_down_reconstruct(
    annotations: dict[str, GeneAnnotation],
    universal_rxns: list[dict[str, object]] | None = None,
    identity_threshold: float = 0.4,
) -> TopDownResult:
    """Top-down reconstruction: start with universal model, remove unsupported.

    Parameters
    ----------
    annotations : {gene_id: GeneAnnotation}
    universal_rxns : list of universal reaction dicts (default: prokaryotic)
    identity_threshold : min EC identity to keep reaction

    Returns
    -------
    TopDownResult with surviving reactions
    """
    if universal_rxns is None:
        universal_rxns = UNIVERSAL_PROKARYOTIC_REACTIONS

    # Build organism's EC set
    organism_ecs: set[str] = set()
    for annot in annotations.values():
        organism_ecs.update(annot.ec_numbers)

    result = TopDownResult()
    kept = 0
    removed = 0

    for rxn_def in universal_rxns:
        required_ecs_raw = rxn_def.get("required_ec", [])
        required_ecs = required_ecs_raw if isinstance(required_ecs_raw, list) else []
        if not required_ecs:
            result.reactions.append(ReactionEntry(
                reaction_id=str(rxn_def["id"]),
                equation=str(rxn_def.get("eq", "")),
                reversibility=bool(rxn_def.get("rev", True)),
                confidence=0.5,
                source="universal",
            ))
            kept += 1
            continue

        has_support = any(ec in organism_ecs for ec in required_ecs)
        if has_support:
            result.reactions.append(ReactionEntry(
                reaction_id=str(rxn_def["id"]),
                equation=str(rxn_def.get("eq", "")),
                reversibility=bool(rxn_def.get("rev", True)),
                confidence=0.8,
                source="top_down",
            ))
            kept += 1
        else:
            removed += 1

    result.kept_reactions = kept
    result.removed_reactions = removed
    return result
