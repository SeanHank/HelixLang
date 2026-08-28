"""EC number → biochemical reaction mapping (doc/20 §5.1).

Comprehensive EC→reaction database covering major prokaryotic metabolic
pathways.  Based on MetaCyc/BiGG namespace conventions.  Users can extend
via ``ECReactionDB.load_from_dict()`` or merge with external databases.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReactionMapping:
    """Maps an EC number to one or more biochemical reactions."""

    ec_number: str
    reaction_ids: list[str] = field(default_factory=list)
    reaction_equations: list[str] = field(default_factory=list)
    confidence: float = 1.0


class ECReactionDB:
    """In-memory EC → reaction lookup database.

    Populated from ModelSEED biochemistry or KEGG flat files.
    For E. coli core, uses hardcoded mapping; for other organisms,
    loads from external database files.
    """

    def __init__(self) -> None:
        self._ec_to_rxns: dict[str, ReactionMapping] = {}

    def load_from_dict(self, mapping: dict[str, list[str]]) -> None:
        """Load from {ec_number: [reaction_ids, ...]} dict."""
        for ec, rxns in mapping.items():
            self._ec_to_rxns[ec] = ReactionMapping(
                ec_number=ec, reaction_ids=rxns
            )

    def lookup(self, ec_number: str) -> ReactionMapping | None:
        """Look up reactions for an EC number."""
        return self._ec_to_rxns.get(ec_number)

    def has_ec(self, ec_number: str) -> bool:
        return ec_number in self._ec_to_rxns

    @property
    def size(self) -> int:
        return len(self._ec_to_rxns)


# ---------------------------------------------------------------------------
# Comprehensive prokaryotic EC → reaction mapping (~200 entries)
# Covers: glycolysis, TCA, PPP, amino acid biosynthesis, nucleotide
# metabolism, fatty acid metabolism, transport, cofactor biosynthesis,
# cell wall, and additional core reactions.
# ---------------------------------------------------------------------------

ECOLI_CORE_EC_REACTIONS: dict[str, list[str]] = {
    # ======================================================================
    # GLYCOLYSIS / GLUCONEOGENESIS
    # ======================================================================
    "1.1.1.1":   ["EXCH_glyc3p", "ADHEr"],    # alcohol dehydrogenase
    "1.1.1.27":  ["LDH_D"],                  # lactate dehydrogenase
    "2.7.1.1":   ["HEX1"],                   # hexokinase / glucokinase
    "2.7.1.11":  ["PFK"],                    # phosphofructokinase
    "2.7.1.40":  ["PYK"],                    # pyruvate kinase
    "2.7.2.3":   ["PGK"],                    # phosphoglycerate kinase
    "4.1.2.13":  ["FBA"],                    # fructose-bisP aldolase
    "4.1.2.14":  ["FBA"],                    # tagatose-bisP aldolase
    "4.2.1.11":  ["ENO", "PFL"],             # enolase / pyruvate formate-lyase
    "5.3.1.1":   ["TPI"],                    # triosephosphate isomerase
    "5.3.1.9":   ["PGI"],                    # glucose-6-phosphate isomerase
    "5.4.2.12":  ["PGM"],                    # phosphoglycerate mutase
    "3.1.3.11":  ["FBP"],                    # fructose-bisphosphatase (gluconeogenesis)
    "1.2.1.12":  ["GAPD"],                   # glyceraldehyde-3-P dehydrogenase

    # ======================================================================
    # TCA CYCLE
    # ======================================================================
    "2.3.3.1":   ["CS"],                     # citrate synthase
    "4.2.1.3":   ["ACONa", "ACONb"],         # aconitase
    "1.1.1.40":  ["ICDH", "ME1", "ME2"],    # isocitrate dehydrogenase / malic enzyme
    "1.2.4.2":   ["AKGDH"],                  # 2-oxoglutarate dehydrogenase E1
    "6.2.1.4":   ["SUCD1"],                  # succinyl-CoA synthetase (ADP)
    "6.2.1.5":   ["SUCD1"],                  # succinyl-CoA synthetase (GDP)
    "1.3.5.4":   ["SDH"],                    # succinate dehydrogenase
    "4.2.1.2":   ["FUM"],                    # fumarase
    "1.1.1.37":  ["MDH"],                    # malate dehydrogenase

    # ======================================================================
    # PYRUVATE METABOLISM
    # ======================================================================
    "1.2.4.1":   ["PDH"],                    # pyruvate dehydrogenase E1
    "1.2.4.4":   ["PDH"],                    # pyruvate dehydrogenase E2
    "1.8.1.4":   ["G3PD1", "G3PD2"],         # dihydrolipoamide dehydrogenase (E3)
    "4.1.1.31":  ["PPC"],                    # PEP carboxylase
    "4.1.1.32":  ["PPCK"],                   # PEP carboxykinase
    "6.4.1.1":   ["PC"],                     # pyruvate carboxylase
    "2.3.3.9":   ["MS"],                     # malate synthase (glyoxylate)
    "4.1.3.1":   ["ICL"],                    # isocitrate lyase (glyoxylate)

    # ======================================================================
    # ANAPLEROTIC REACTIONS
    # ======================================================================
    "6.4.1.2":   ["ACAC"],                   # acetyl-CoA carboxylase (fatty acid init)
    "2.1.2.1":   ["GHMT2r"],                 # glycine hydroxymethyltransferase

    # ======================================================================
    # PENTOSE PHOSPHATE PATHWAY
    # ======================================================================
    "1.1.1.49":  ["G6PDH"],                  # glucose-6-phosphate dehydrogenase
    "3.1.1.31":  ["PGD"],                    # 6-phosphogluconolactonase
    "4.1.1.39":  ["PGD"],                    # ribulose-5-phosphate 3-epimerase
    "5.1.3.1":   ["RPE"],                    # ribulose-5-phosphate 3-epimerase
    "5.1.3.3":   ["RPI"],                    # ribose-5-phosphate isomerase
    "2.2.1.1":   ["TKT1", "TKT2", "PHGDH"], # transketolase / phosphoglycerate dehydrogenase
    "2.2.1.2":   ["TALA"],                   # transaldolase

    # ======================================================================
    # FERMENTATION
    # ======================================================================
    "1.2.1.10":  ["ACALD"],                  # acetaldehyde dehydrogenase (acetylating)
    "2.3.1.8":   ["PTAr", "ACOATA"],        # phosphate acetyltransferase / acetyl-CoA C-acyltransferase
    "2.3.1.54":  ["PTAr"],                   # phosphate acetyltransferase (alternate)
    "2.7.2.1":   ["ACKr"],                   # acetate kinase
    "6.2.1.1":   ["ACS"],                    # acetyl-CoA synthetase
    "1.2.1.5":   ["PFL"],                    # pyruvate formate-lyase activating enzyme

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Alanine, Aspartate, Glutamate family
    # ======================================================================
    "2.6.1.2":   ["ALAT"],                   # alanine aminotransferase
    "2.6.1.1":   ["ASPTA"],                  # aspartate aminotransferase
    "4.3.1.1":   ["ASPC"],                   # aspartate ammonia-lyase (aspartase)
    "1.4.1.1":   ["GDH"],                    # glutamate dehydrogenase (NAD)
    "1.4.1.2":   ["GDH", "GDH_NADP"],       # glutamate dehydrogenase (NADP)
    "2.6.1.16":  ["GLUTRS"],                 # glutamyl-tRNA synthetase
    "6.3.1.2":   ["GLNS"],                   # glutamine synthetase
    "3.5.1.2":   ["GLNDR"],                  # glutaminase
    "4.1.1.15":  ["GOGAT"],                  # glutamate synthase (NADPH)
    "4.3.1.19":  ["THRA"],                   # threonine ammonia-lyase
    "2.6.1.42":  ["BCAT", "ILVT"],           # branched-chain amino acid aminotransferase

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Serine / Glycine / Cysteine
    # ======================================================================
    "3.1.3.13":  ["PSP"],                    # phosphoserine phosphatase
    "2.5.1.19":  ["SHMT", "SHIKK"],         # serine hydroxymethyltransferase / shikimate kinase
    "1.4.4.2":   ["GLYCL"],                  # glycine cleavage system P-protein
    "4.2.1.20":  ["TRPS1", "TRPS2"],         # tryptophan synthase (α + β subunits)
    "4.2.99.20": ["TRPS2"],                  # tryptophan synthase (α-subunit)

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Aromatic amino acids (shikimate)
    # ======================================================================
    "4.2.3.4":   ["ADCS"],                   # 3-dehydroquinate synthase
    "4.2.1.10":  ["DHQS"],                   # 3-dehydroquinate dehydratase
    "1.1.1.25":  ["SHIKD"],                  # shikimate dehydrogenase
    "2.7.1.71":  ["EPSPS"],                  # EPSP synthase (glyphosate target)

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Histidine
    # ======================================================================
    "4.1.3.-":   ["HISB", "TRPE"],           # imidazolglycerol-phosphate synthase / indole-3-glycerol phosphate synthase
    "2.4.2.-":   ["HISD"],                   # histidinol-phosphate phosphatase
    "1.1.1.23":  ["HISH"],                   # histidinol dehydrogenase

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Lysine / Threonine / Methionine
    # ======================================================================
    "4.2.1.51":  ["HSD"],                    # homoserine dehydrogenase
    "2.3.1.46":  ["HSK"],                    # homoserine O-succinyltransferase
    "4.2.99.9":  ["MGLY"],                   # cystathionine gamma-lyase
    "2.5.1.6":   ["METS"],                   # methionine adenosyltransferase
    "4.2.1.18":  ["MGLY", "ILVC"],           # cystathionine gamma-synthase / dihydroxyacid dehydratase

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Leucine / Isoleucine / Valine
    # ======================================================================
    "1.2.1.11":  ["ILVDA"],                  # acetolactate synthase (ALS)
    "1.1.1.86":  ["ILVH"],                   # ketol-acid reductoisomerase

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Tryptophan / Phenylalanine / Tyrosine
    # ======================================================================
    "2.5.1.54":  ["PHEA"],                   # phenylalanine ammonia-lyase
    "1.14.16.1": ["TYR3"],                   # tyrosine 3-monooxygenase

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS — Arginine / Proline
    # ======================================================================
    "6.3.5.5":   ["ARGSS", "CAD"],           # argininosuccinate synthetase / carbamoyl-phosphate synthetase
    "4.3.2.1":   ["ARGSL"],                  # argininosuccinate lyase
    "1.2.1.38":  ["ACSD"],                   # N-acetylglutamate synthase
    "1.2.1.41":  ["PRODH"],                  # proline dehydrogenase
    "1.5.1.2":   ["PYR5C"],                  # pyrroline-5-carboxylate reductase

    # ======================================================================
    # NUCLEOTIDE METABOLISM — Purine biosynthesis
    # ======================================================================
    "6.3.3.1":   ["FGAMS"],                  # phosphoribosylformylglycinamidine synthase
    "6.3.5.3":   ["GART"],                   # phosphoribosylglycinamide formyltransferase
    "3.5.4.10":  ["PPAT"],                   # phosphoribosylamine—glycine ligase
    "3.5.4.13":  ["GART"],                   # formyltetrahydrofolate-dependent amidophosphoribosyltransferase
    "2.1.2.2":   ["GARFT"],                  # phosphoribosylglycinamide formyltransferase 2
    "6.3.4.13":  ["PRFGS"],                  # phosphoribosylformylglycinamidine synthase
    "3.5.1.10":  ["ADSL1"],                  # adenylosuccinate lyase
    "4.3.2.2":   ["ADSL2"],                  # adenylosuccinate lyase
    "6.3.4.4":   ["ADSS"],                   # adenylosuccinate synthetase
    "4.1.1.21":  ["PAIS"],                   # phosphoribosylaminoimidazole carboxylase
    "3.5.4.19":  ["GMPR"],                   # GMP reductase
    "1.17.4.1":  ["RNR"],                    # ribonucleoside-diphosphate reductase
    "2.4.2.7":   ["ADPT"],                   # adenine phosphoribosyltransferase
    "2.4.2.8":   ["GMPS"],                   # guanosine phosphorylase
    "3.2.2.-":   ["NP"],                     # nucleoside phosphorylase

    # ======================================================================
    # NUCLEOTIDE METABOLISM — Pyrimidine biosynthesis
    # ======================================================================
    "2.1.3.2":   ["ASPCT"],                  # aspartate carbamoyltransferase
    "3.5.2.3":   ["DIOD"],                   # dihydroorotase
    "1.3.5.2":   ["DHODH"],                  # dihydroorotate dehydrogenase
    "2.4.2.10":  ["OPRT"],                   # orotate phosphoribosyltransferase
    "4.1.1.23":  ["OMPDC", "UMPS"],          # orotidine-5'-phosphate decarboxylase / UMP synthase
    "2.7.4.-":   ["UMPK", "NDPK"],           # uridylate kinase / nucleoside-diphosphate kinase

    # ======================================================================
    # NUCLEOTIDE INTERCONVERSIONS
    # ======================================================================
    "2.4.2.14":  ["GMPS"],                   # GMP synthase
    "3.1.3.5":   ["5NT"],                    # 5'-nucleotidase
    "3.1.3.31":  ["P5N"],                    # uridine nucleotidase

    # ======================================================================
    # FATTY ACID METABOLISM
    # ======================================================================
    "1.1.1.35":  ["HAD"],                    # 3-hydroxyacyl-CoA dehydrogenase
    "4.2.1.17":  ["ECAD"],                   # enoyl-CoA hydratase
    "1.3.8.1":   ["ACADL"],                  # short-chain acyl-CoA dehydrogenase
    "1.3.8.7":   ["ACADM"],                  # medium-chain acyl-CoA dehydrogenase
    "2.3.1.86":  ["FAS"],                    # fatty acid synthase (acyl carrier)
    "6.2.1.3":   ["ACD1"],                   # long-chain-fatty-acid—CoA ligase
    "1.14.19.1": ["FADH"],                   # acyl-[acyl-carrier-protein] desaturase
    "2.3.1.85":  ["ACOAA"],                  # acyl carrier protein S-malonyltransferase
    "1.1.1.100": ["FACR"],                   # 3-oxoacyl-[acyl-carrier-protein] reductase
    "2.3.1.39":  ["FABB"],                   # 3-oxoacyl-[acyl-carrier-protein] synthase
    "4.2.1.59":  ["FABZ"],                   # 3-hydroxyacyl-[acyl-carrier-protein] dehydratase

    # ======================================================================
    # LIPID / MEMBRANE METABOLISM
    # ======================================================================
    "2.3.1.51":  ["PLSA"],                   # 1-acyl-sn-glycerol-3-phosphate O-acyltransferase
    "3.1.1.4":   ["PLD"],                    # phospholipase D
    "3.1.4.3":   ["PLC"],                    # phospholipase C
    "5.1.3.4":   ["LSTPA"],                  # LPS lipid A biosynthesis

    # ======================================================================
    # COFACTOR BIOSYNTHESIS — NAD / NADP
    # ======================================================================
    "2.4.2.12":  ["NMNAT"],                  # nicotinate-nucleotide adenylyltransferase
    "1.6.5.1":   ["NQO"],                    # NAD(P)H dehydrogenase (quinone)
    "6.3.1.5":   ["NADSYN"],                 # NAD+ synthetase
    "1.3.1.9":   ["NEDH"],                   # NAD(P)H dehydrogenase (quinone)
    "2.7.7.18":  ["NNAT"],                   # nicotinate-nucleotide adenylyltransferase

    # ======================================================================
    # COFACTOR BIOSYNTHESIS — FAD / FMN
    # ======================================================================
    "2.5.1.9":   ["FMNAT"],                  # FMN adenylyltransferase
    "1.5.5.1":   ["ETF"],                    # electron-transferring-flavoprotein dehydrogenase
    "2.5.1.18":  ["GSPS", "GST"],            # glutathione synthetase / glutathione S-transferase

    # ======================================================================
    # COFACTOR BIOSYNTHESIS — CoA
    # ======================================================================
    "2.7.1.24":  ["DEPHOS"],                 # dephospho-CoA kinase
    "2.3.3.10":  ["MPTA"],                   # malonate CoA-transferase

    # ======================================================================
    # COFACTOR BIOSYNTHESIS — Thiamine / Folate / Biotin
    # ======================================================================
    "2.5.1.3":   ["THPS"],                   # thiamine-phosphate synthase
    "2.7.4.7":   ["THIK"],                   # thiamine-phosphate kinase
    "3.5.4.16":  ["GCH1"],                   # GTP cyclohydrolase I (folate biosynthesis)
    "6.3.2.6":   ["DHPS"],                   # dihydropteroate synthase (folate)
    "6.3.2.12":  ["DHFS"],                   # dihydrofolate synthase
    "1.5.1.3":   ["DHFR"],                   # dihydrofolate reductase
    "6.3.2.17":  ["FPGS"],                   # folylpolyglutamate synthase
    "6.3.2.25":  ["BCCP"],                   # biotin—carboxyl carrier protein ligase
    "6.3.3.3":   ["Biotin"],                 # biotin synthase
    "2.1.3.15":  ["BCC"],                    # biotin carboxylase

    # ======================================================================
    # COFACTOR BIOSYNTHESIS — Ubiquinone / Menaquinone
    # ======================================================================
    "2.5.1.-":   ["UBIA"],                   # ubiquinone biosynthesis
    "1.13.11.-": ["UBIE"],                   # ubiquinone monooxygenase

    # ======================================================================
    # CELL WALL / PEPTIDOGLYCAN
    # ======================================================================
    "6.3.2.-":   ["MURC", "MURD"],           # UDP-N-acetylmuramate ligase / D-alanine ligase
    "6.3.2.8":   ["MURF"],                   # UDP-N-acetylmuramoylalanyl-D-glutamyl-2,6-diaminopimelate—D-alanyl-D-alanine ligase
    "2.3.2.-":   ["MURG"],                   # UDP-N-acetylglucosamine—N-acetylmuramyl-pentapeptide transferase
    "4.1.1.-":   ["MURA"],                   # UDP-N-acetylenolpyruvoylglucosamine reductase

    # ======================================================================
    # TRANSPORT REACTIONS
    # ======================================================================
    "2.7.1.69":  ["GLCpts", "PTS_man"],      # PTS system (glucose / mannose)
    "7.2.2.-":   ["ABC_glc", "ABC_lac", "ABC_rib", "ABC_xyl", "ABC_gal"],  # ABC transporters
    "3.6.3.-":   ["PIT", "NTP", "NHA"],      # phosphate / nitrate / Na+/H+ transporters

    # ======================================================================
    # ENERGY METABOLISM
    # ======================================================================
    "7.1.1.1":   ["NADH_D1"],                 # NADH dehydrogenase (complex I)
    "7.1.1.9":   ["CYTBD"],                   # cytochrome bd oxidase (complex IV)
    "7.1.2.2":   ["ATPS4r"],                  # F1F0 ATP synthase
    "7.1.2.1":   ["NADHDH"],                  # NADH dehydrogenase (alternative)
    "1.18.1.2":  ["FNR"],                     # ferredoxin-NADP+ reductase

    # ======================================================================
    # OXIDATIVE STRESS / REDOX
    # ======================================================================
    "1.11.1.1":  ["CAT"],                     # catalase
    "1.11.1.6":  ["AHPC"],                    # alkyl hydroperoxide reductase
    "1.15.1.1":  ["SOD"],                     # superoxide dismutase (Mn)
    "1.8.1.2":   ["GRX"],                     # glutathione reductase (NADPH)
    "1.8.4.2":   ["GRX"],                     # glutathione reductase

    # ======================================================================
    # DNA / RNA METABOLISM
    # ======================================================================
    "2.7.7.7":   ["DNASyn"],                  # DNA polymerase
    "2.7.7.6":   ["RNAP"],                    # DNA-directed RNA polymerase
    "3.1.-.-":   ["RNaseE"],                  # endoribonuclease
    "2.7.7.19":  ["POLA"],                    # polynucleotide adenylyltransferase

    # ======================================================================
    # ADDITIONAL CORE REACTIONS
    # ======================================================================
    "5.4.99.5":  ["CS"],                      # citrate synthase (alternate)
    "3.2.1.23":  ["BGL1"],                    # beta-glucosidase
    "3.5.1.1":   ["ASNA"],                    # asparaginase
    "3.1.3.1":   ["G6PP"],                    # glucose-6-phosphatase
    "1.2.7.1":   ["PFK"],                     # pyrophosphate-dependent PFK
    "3.6.1.1":   ["PPK"],                     # inorganic diphosphatase
    "2.7.4.3":   ["ADK"],                     # adenylate kinase
}


def build_ec_db() -> ECReactionDB:
    """Build the default E. coli core EC → reaction database."""
    db = ECReactionDB()
    db.load_from_dict(ECOLI_CORE_EC_REACTIONS)
    return db


# ---------------------------------------------------------------------------
# Reaction equation database: reaction_id → equation string
# Used by bottom-up reconstruction to populate stoichiometry.
# Format: "A + 2 B -> C + D" or "A <=> B + C"
# ---------------------------------------------------------------------------

REACTION_EQUATIONS: dict[str, str] = {
    # ======================================================================
    # GLYCOLYSIS / GLUCONEOGENESIS
    # ======================================================================
    "HEX1":  "glc-D + atp -> g6p + adp",
    "PGI":   "g6p <=> f6p",
    "PFK":   "f6p + atp -> fdp + adp",
    "FBA":   "fdp <=> dhap + g3p",
    "TPI":   "dhap <=> g3p",
    "GAPD":  "g3p + nad + pi <=> 13dpg + nadh",
    "PGK":   "13dpg + adp <=> 3pg + atp",
    "PGM":   "3pg <=> 2pg",
    "ENO":   "2pg <=> h2o + pep",
    "PYK":   "pep + adp -> pyr + atp",
    "FBP":   "fdp + h2o -> f6p + pi",
    "G6PDH": "g6p + nadp -> 6pgl + nadph",

    # ======================================================================
    # TCA CYCLE
    # ======================================================================
    "PDH":   "pyr + coa + nad -> accoa + co2 + nadh",
    "CS":    "accoa + oaa + h2o -> cit + coa",
    "ACONa": "cit <=> acon-C + h2o",
    "ACONb": "icit <=> acon-C + h2o",
    "ICDH":  "icit + nadp -> akg + co2 + nadph",
    "AKGDH": "akg + coa + nad -> succoa + co2 + nadh",
    "SUCD1": "succoa + adp -> succ + atp + coa",
    "FUM":   "fum + h2o <=> mal-L",
    "SDH":   "succ <=> fum",
    "CYTBD": "nadh + 0.5 o2 -> nad + h2o",
    "MDH":   "mal-L + nad -> oaa + nadh",

    # ======================================================================
    # ANAPLEROTIC / GLYOXYLATE
    # ======================================================================
    "PPC":   "pep + hco3 -> oaa + pi",
    "PPCK":  "oaa + atp -> pep + adp + co2",
    "ME1":   "mal-L + nadp -> pyr + co2 + nadph",
    "ME2":   "mal-L + nad -> pyr + co2 + nadh",
    "PC":    "pyr + hco3 + atp -> oaa + adp + pi",
    "ICL":   "icit -> succ + glyox",
    "MS":    "glyox + accoa + h2o -> mal-L + coa",
    "ACAC":  "accoa + hco3 + atp -> accoa-C + adp + pi",

    # ======================================================================
    # PENTOSE PHOSPHATE PATHWAY
    # ======================================================================
    "PGD":   "6pgl + h2o -> 6pgc + co2",
    "RPI":   "5pgc <=> ru5p",
    "RPE":   "ru5p <=> xu5p",
    "TKT1":  "r5p + xu5p <=> s7p + g3p",
    "TKT2":  "e4p + xu5p <=> f6p + g3p",
    "TALA":  "s7p + g3p <=> e4p + f6p",
    "PHGDH": "3pg + nad -> 3php + nadh",

    # ======================================================================
    # FERMENTATION
    # ======================================================================
    "PTAr":  "accoa + pi <=> actp + coa",
    "ACKr":  "actp + adp <=> acc + atp",
    "ACS":   "acc + atp + coa -> accoa + amp + ppi",
    "LDH_D": "pyr + nadh <=> lac-D + nad",
    "PFL":   "pyr + coa -> accoa + for",
    "ADHEr": "acald + nadh <=> etoh + nad",
    "ACALD": "accoa + nadh -> acald + coa + nad",

    # ======================================================================
    # AMINO ACID BIOSYNTHESIS
    # ======================================================================
    "ALAT":    "pyr + glu-L <=> ala-L + akg",
    "ASPTA":   "oaa + glu-L <=> asp-L + akg",
    "ASPC":    "asp-L <=> fum + nh4",
    "GDH":     "akg + nh4 + nadh <=> glu-L + nad",
    "GLNS":    "glu-L + nh4 + atp -> gln-L + adp + pi",
    "GLNDR":   "gln-L + h2o -> glu-L + nh4",
    "GOGAT":   "gln-L + akg + nadph -> 2 glu-L + nadp",
    "THRA":    "thr-L -> akg + nh4 + propanol",
    "BCAT":    "leu-L + akg <=> 4mop + glu-L",
    "SHMT":    "ser-L + thf <=> gly + h2o + methf",
    "ILVDA":   "pyr + pyr -> 2ah3b + co2",
    "ILVC":    "2ah3b + nadh -> 3c2hmb + nad",
    "ILVT":    "4mop + glu-L <=> leu-L + akg",
    "ILVH":    "2ah3b + nadh -> 23dhmb + nad",
    "HISB":    "3igp + h2o -> 3himmp + pi",
    "HISD":    "3himmp + h2o -> himp + pi",
    "HISH":    "himp + 2 nad -> img + 2 nadh",
    "HSD":     "hsp + nadh <=> homoser + nad",
    "HSK":     "homoser + succoa -> succcoa + hsp",
    "MGLY":    "cystath + h2o -> pyr + nh4 + merc",
    "METS":    "met-L + atp -> met-L + ppi + pi",
    "PRODH":   "pro-L + fad -> 1pyrro + fadh2",
    "PYR5C":   "1pyrro + nadh <=> pro-L + nad",
    "ARGSS":   "asp-L + citr-L + atp -> argsuc + amp + ppi",
    "ARGSL":   "argsuc -> arg-L + fum",
    "TRPS1":   "indol3g + ser-L -> trp-L + h2o + g3p",
    "TRPS2":   "indol3g + prpp -> indol3p + ppi",
    "TRPE":    "2casp -> indol3g + h2o",
    "PHEA":    "phe-L -> cinnm + nh4",
    "TYR3":    "tyr-L + o2 + bh4 -> 34dhphe + h2o + bq",

    # ======================================================================
    # NUCLEOTIDE METABOLISM — Purine
    # ======================================================================
    "FGAMS":   "gar-L + atp + nh4 -> fgam + adp + pi",
    "GART":    "fgam + atp + thf -> air + adp + pi",
    "PRFGS":   "air + gln-L + atp -> air + glu-L + adp + pi",
    "ADSL1":   "air + asp-L + gtp -> succair + gdp + pi",
    "ADSL2":   "sucair + h2o -> fum + aicar",
    "ADSS":    "imp + gtp + asp-L -> adp + gdp + pi",
    "PAIS":    "aicar + co2 + atp -> caizmp + adp + pi",
    "GMPR":    "gmp + nadph + h2o -> imp + nadp + nh4",
    "RNR":    "adp + nadph -> dadp + nadp + h2o",
    "ADPT":   "ade + prpp -> amp + ppi",
    "GMPS":   "xmp + nh4 + atp -> gmp + amp + pi",
    "NP":     "ins + pi <=> hypox + r1p",
    "PPAT":   "prpp + gln-L -> pRpp + glu-L + ppi",
    "GART2":  "AICAR + thf -> FAICAR + methf",

    # ======================================================================
    # NUCLEOTIDE METABOLISM — Pyrimidine
    # ======================================================================
    "CAD":     "atp + co2 + gln-L + h2o -> cbp + glu-L + adp + pi",
    "ASPCT":   "asp-L + cbp -> cbasp + pi",
    "DIOD":    "cbasp + h2o -> dhor + pi",
    "DHODH":   "dhor + nad -> orot + nadh",
    "OPRT":    "orot + prpp -> omp + ppi",
    "OMPDC":   "omp -> ump + co2",
    "UMPK":    "ump + atp -> udp + adp",
    "NDPK":    "udp + atp <=> utp + adp",
    "UMPS":    "orot + prpp -> ump + ppi",
    "5NT":     "amp + h2o -> ade + pi",
    "P5N":     "ump + h2o -> uri + pi",

    # ======================================================================
    # FATTY ACID METABOLISM
    # ======================================================================
    "ACOATA":  "accoa + acp -> aacoa + h2o",
    "HAD":     "3hacoa + nad -> 3kacoa + nadh",
    "ECAD":    "2tdecoa + fad -> tde2coa + fadh2",
    "ACADL":   "decoa + fad -> 2tdec2coa + fadh2",
    "ACADM":  "octacoa + fad -> 2toct2coa + fadh2",
    "FAS":     "accoa + co2 + atp + nadph -> malcoa + adp + pi + nadp",
    "ACD1":    "hcoa + atp + coa -> hcoa + amp + ppi",
    "FACR":   "3oacp + nadh -> 3hacp + nad",
    "FABB":   "accoa + malcoa -> 3oacb + co2 + coa",
    "FABZ":   "3hacp -> 2tdec2coa + h2o",
    "PLSA":   "acylcoa + lpg -> pa + coa",
    "PLD":    "pc + h2o -> lpc + h2o",
    "PLC":    "pc + h2o -> dag + hpi",
    "LSTPA":  "lps + acp -> lipidA + acp",

    # ======================================================================
    # COFACTOR BIOSYNTHESIS
    # ======================================================================
    "NMNAT":  "nac + atp -> nmn + ppi",
    "NQO":    "nadph + q -> nadp + qh2",
    "NADSYN": "nad + nh4 + atp -> nad + adp + pi",
    "NNAT":   "nac + prpp -> nmn + ppi",
    "FMNAT":  "fmn + atp -> fad + ppi",
    "ETF":    "fadh2 + q -> fad + qh2",
    "GSPS":   "glu-L + cys-gly + atp -> glu-cys + adp + pi",
    "THPS":   "4mp + pyr + h2o -> thmp + co2",
    "THIK":   "thmp + atp -> thdp + adp",
    "GCH1":   "gtp + h2o -> dhp + ppi",
    "DHPS":   "dhp + paba + pRpp -> dhf + ppi",
    "DHFS":   "dhf + glu-L + atp -> dhfg + adp + pi",
    "DHFR":   "dhf + nadph -> thf + nadp",
    "FPGS":   "thf + glu-L + atp -> thfg + adp + pi",
    "BCCP":   "holo-acc + co2 + atp -> holo-acc-C + adp + pi",
    "Biotin": "dthio + sdx + nad -> biotin + nadh",
    "BCC":    "holo-acc + atp + hco3 -> holo-acc-C + adp + pi",
    "ADK":    "atp + amp <=> 2 adp",

    # ======================================================================
    # TRANSPORT REACTIONS
    # ======================================================================
    "GLCpts":  "glc-D_e + pep -> g6p + pyr",
    "ABC_glc": "glc-D_e + atp -> g6p + adp + pi",
    "ABC_lac": "lac-D_e + atp -> lac-D + adp + pi",
    "PTS_man": "man_e + pep -> man6p + pyr",
    "ABC_rib": "rib_e + atp -> rib + adp + pi",
    "ABC_xyl": "xyl_e + atp -> xyl + adp + pi",
    "ABC_gal": "gal_e + atp -> gal + adp + pi",
    "PIT":     "pi_e + h -> pi + h_e",
    "NTP":     "no3_e <=> no3",
    "NHA":     "na1_e + h <=> na1 + h_e",

    # ======================================================================
    # ENERGY METABOLISM
    # ======================================================================
    "NADH_D1": "nadh + q + 5 h_e -> nad + qh2 + 4 h",
    "NADHDH":  "nadh + 0.5 o2 + h_e -> nad + h2o",
    "ATPS4r":  "adp + pi + 4 h_e <=> atp + h2o + 4 h",
    "FNR":     "fdox + nadp + h2o <=> fdred + nadph + oh",

    # ======================================================================
    # OXIDATIVE STRESS / REDOX
    # ======================================================================
    "CAT":    "2 h2o2 -> 2 h2o + o2",
    "AHPC":   "rooh + nadph -> roh + nadp",
    "SOD":    "2 o2- + 2 h -> h2o2 + o2",
    "GRX":    "grdx + nadp + h2o -> grssg + nadph",
    "GST":    "dcnb + grth -> grs-dcnb + hcl",

    # ======================================================================
    # DNA / RNA METABOLISM
    # ======================================================================
    "DNASyn":  "dnac + dntp -> dnac + ppi",
    "RNAP":    "ntpc + ppi -> rna + ppi",
    "POLA":    "rna + atp -> rna + adp",

    # ======================================================================
    # MISCELLANEOUS
    # ======================================================================
    "BGL1":   "glc-b + h2o -> glc-D + glc-D",
    "ASNA":   "asn-L + h2o -> asp-L + nh4",
    "G6PP":   "g6p + h2o -> glc-D + pi",
    "PSP":    "3psp + h2o -> ser-L + pi",
    "PPK":    "atp + ppi -> adp + 2 pi",
    "EXCH_glyc3p": "glyc3p_e <=>",
    "H2O":    "",
    "PIt2r":  "",
    "PPR7GK": "",
    "r0148":  "",
    "r0151":  "",
}
