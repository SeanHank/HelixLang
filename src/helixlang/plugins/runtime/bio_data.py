"""Real biological data module: replaces fictional parameters with
values measured in the literature.

Data sources:
- E. coli K-12 MG1655 codon usage frequencies: GenScript CUTG (Codon
  Usage Tabulated from GenBank)
- S. cerevisiae (yeast) codon usage frequencies: Kazusa CUTG (S.
  cerevisiae 4932)
- H. sapiens (human) codon usage frequencies: Kazusa CUTG (H. sapiens
  9606)
- lac operon promoter strength: Miller 1972 / Kennedy 1977
  beta-galactosidase MU assays
- Gray-Scott 14-parameter presets: Pearson 1993 Complex Systems
  7:331-378
- PCR error rates: Saiki 1988 Science 239:487-491, Potapov 2017 PLoS
  ONE 12:e0169774
- Sequencing platform error rates: Ceze, Nivala, Strauss Nat Rev Genet
  2019 20:456-466
- DNA synthesis error rates: Filges 2021 Clinical Chemistry 67:1384-1394
- DNA storage density: Goldman 2013, Erlich 2017, Organick 2018
- DNA decay: Allentoft 2012 Proc R Soc B 279:4724-4733, Grass 2015
  Angew Chem 54:2552-2555
- Codon adaptation index: Sharp & Li 1987 Nucleic Acids Res
  15:1281-1295
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ============================================================================
# E. coli K-12 MG1655 codon usage frequency table
# Data source: GenScript Codon Usage Frequency Table (CUTG E. coli 511145)
# Fields: codon -> (amino_acid, per_thousand, fraction)
# per_thousand: occurrences of the codon per 1000 codons
# fraction: proportion of this synonymous codon within its amino acid
# ============================================================================

ECOLI_CODON_USAGE: dict[str, tuple[str, float, float]] = {
    # Phe
    "TTT": ("F", 22.1, 0.58), "TTC": ("F", 16.0, 0.42),
    # Leu
    "TTA": ("L", 14.3, 0.14), "TTG": ("L", 13.0, 0.13),
    "CTT": ("L", 11.9, 0.12), "CTC": ("L", 10.2, 0.10),
    "CTA": ("L", 4.2, 0.04),  "CTG": ("L", 48.4, 0.47),
    # Ile
    "ATT": ("I", 29.8, 0.49), "ATC": ("I", 23.7, 0.39),
    "ATA": ("I", 6.8, 0.11),
    # Met
    "ATG": ("M", 26.4, 1.00),
    # Val
    "GTT": ("V", 19.8, 0.28), "GTC": ("V", 14.3, 0.20),
    "GTA": ("V", 11.6, 0.17), "GTG": ("V", 24.4, 0.35),
    # Ser
    "TCT": ("S", 10.4, 0.17), "TCC": ("S", 9.1, 0.15),
    "TCA": ("S", 8.9, 0.14),  "TCG": ("S", 8.5, 0.14),
    "AGT": ("S", 9.9, 0.16),  "AGC": ("S", 15.2, 0.25),
    # Pro
    "CCT": ("P", 7.5, 0.18),  "CCC": ("P", 5.4, 0.13),
    "CCA": ("P", 8.6, 0.20),  "CCG": ("P", 20.9, 0.49),
    # Thr
    "ACT": ("T", 10.3, 0.19), "ACC": ("T", 22.0, 0.40),
    "ACA": ("T", 9.3, 0.17),  "ACG": ("T", 13.7, 0.25),
    # Ala
    "GCT": ("A", 17.1, 0.18), "GCC": ("A", 24.2, 0.26),
    "GCA": ("A", 21.2, 0.23), "GCG": ("A", 30.1, 0.33),
    # Tyr
    "TAT": ("Y", 17.5, 0.59), "TAC": ("Y", 12.2, 0.41),
    # Stop
    "TAA": ("*", 2.0, 0.61),  "TAG": ("*", 0.3, 0.09), "TGA": ("*", 1.0, 0.30),
    # His
    "CAT": ("H", 12.5, 0.57), "CAC": ("H", 9.3, 0.43),
    # Gln
    "CAA": ("Q", 14.6, 0.34), "CAG": ("Q", 28.4, 0.66),
    # Asn
    "AAT": ("N", 20.6, 0.49), "AAC": ("N", 21.4, 0.51),
    # Lys
    "AAA": ("K", 35.3, 0.74), "AAG": ("K", 12.4, 0.26),
    # Asp
    "GAT": ("D", 32.7, 0.63), "GAC": ("D", 19.2, 0.37),
    # Glu
    "GAA": ("E", 39.1, 0.68), "GAG": ("E", 18.7, 0.32),
    # Cys
    "TGT": ("C", 5.2, 0.46),  "TGC": ("C", 6.1, 0.54),
    # Trp
    "TGG": ("W", 13.9, 1.00),
    # Arg
    "CGT": ("R", 20.0, 0.36), "CGC": ("R", 19.7, 0.36),
    "CGA": ("R", 3.8, 0.07),  "CGG": ("R", 5.9, 0.11),
    "AGA": ("R", 3.6, 0.07),  "AGG": ("R", 2.1, 0.04),
    # Gly
    "GGT": ("G", 25.5, 0.35), "GGC": ("G", 27.1, 0.37),
    "GGA": ("G", 9.5, 0.13),  "GGG": ("G", 11.3, 0.15),
}


# ============================================================================
# S. cerevisiae (yeast) codon usage frequency table
# Data source: Kazusa CUTG (S. cerevisiae 4932)
# Yeast prefers A/T-ending codons (similar to E. coli but a different
# preference pattern)
# Reference: Sharp & Cowe 1991 Yeast 7:657-678, Nakamura 2000 Nucleic
# Acids Res 28:292
# ============================================================================

YEAST_CODON_USAGE: dict[str, tuple[str, float, float]] = {
    # Phe
    "TTT": ("F", 26.1, 0.58), "TTC": ("F", 18.7, 0.42),
    # Leu
    "TTA": ("L", 26.2, 0.36), "TTG": ("L", 12.9, 0.17),
    "CTT": ("L", 12.5, 0.17), "CTC": ("L", 5.4, 0.07),
    "CTA": ("L", 6.4, 0.09),  "CTG": ("L", 10.5, 0.14),
    # Ile
    "ATT": ("I", 30.1, 0.46), "ATC": ("I", 17.0, 0.26),
    "ATA": ("I", 17.8, 0.28),
    # Met
    "ATG": ("M", 20.4, 1.00),
    # Val
    "GTT": ("V", 22.1, 0.35), "GTC": ("V", 11.6, 0.18),
    "GTA": ("V", 11.8, 0.19), "GTG": ("V", 17.8, 0.28),
    # Ser
    "TCT": ("S", 23.5, 0.27), "TCC": ("S", 13.9, 0.16),
    "TCA": ("S", 18.4, 0.21), "TCG": ("S", 8.7, 0.10),
    "AGT": ("S", 14.2, 0.16), "AGC": ("S", 9.8, 0.11),
    # Pro
    "CCT": ("P", 13.5, 0.31), "CCC": ("P", 6.8, 0.16),
    "CCA": ("P", 18.3, 0.42), "CCG": ("P", 5.3, 0.12),
    # Thr
    "ACT": ("T", 20.3, 0.35), "ACC": ("T", 12.7, 0.22),
    "ACA": ("T", 17.8, 0.30), "ACG": ("T", 8.0, 0.14),
    # Ala
    "GCT": ("A", 21.4, 0.34), "GCC": ("A", 12.2, 0.19),
    "GCA": ("A", 16.3, 0.26), "GCG": ("A", 12.9, 0.21),
    # Tyr
    "TAT": ("Y", 18.8, 0.56), "TAC": ("Y", 14.8, 0.44),
    # Stop
    "TAA": ("*", 1.0, 0.48),  "TAG": ("*", 0.5, 0.24), "TGA": ("*", 0.6, 0.28),
    # His
    "CAT": ("H", 13.7, 0.64), "CAC": ("H", 7.8, 0.36),
    # Gln
    "CAA": ("Q", 27.3, 0.69), "CAG": ("Q", 12.1, 0.31),
    # Asn
    "AAT": ("N", 36.4, 0.59), "AAC": ("N", 25.1, 0.41),
    # Lys
    "AAA": ("K", 41.9, 0.58), "AAG": ("K", 30.4, 0.42),
    # Asp
    "GAT": ("D", 37.5, 0.66), "GAC": ("D", 19.5, 0.34),
    # Glu
    "GAA": ("E", 49.1, 0.72), "GAG": ("E", 19.1, 0.28),
    # Cys
    "TGT": ("C", 9.9, 0.62),  "TGC": ("C", 6.1, 0.38),
    # Trp
    "TGG": ("W", 10.4, 1.00),
    # Arg
    "CGT": ("R", 6.4, 0.13),  "CGC": ("R", 2.6, 0.05),
    "CGA": ("R", 3.0, 0.06),  "CGG": ("R", 5.9, 0.12),
    "AGA": ("R", 21.3, 0.43), "AGG": ("R", 10.4, 0.21),
    # Gly
    "GGT": ("G", 23.9, 0.37), "GGC": ("G", 9.4, 0.15),
    "GGA": ("G", 22.1, 0.34), "GGG": ("G", 9.5, 0.15),
}


# ============================================================================
# H. sapiens (human) codon usage frequency table
# Data source: Kazusa CUTG (H. sapiens 9606)
# Human prefers C/G-ending codons (GC preference, opposite to yeast)
# Reference: Nakamura 2000 Nucleic Acids Res 28:292, Plotkin 2004
# Nature 428:926-930
# ============================================================================

HUMAN_CODON_USAGE: dict[str, tuple[str, float, float]] = {
    # Phe
    "TTT": ("F", 17.2, 0.46), "TTC": ("F", 20.3, 0.54),
    # Leu
    "TTA": ("L", 7.2, 0.07),  "TTG": ("L", 12.9, 0.13),
    "CTT": ("L", 13.2, 0.13), "CTC": ("L", 19.6, 0.20),
    "CTA": ("L", 7.2, 0.07),  "CTG": ("L", 39.8, 0.40),
    # Ile
    "ATT": ("I", 15.1, 0.32), "ATC": ("I", 24.0, 0.52),
    "ATA": ("I", 7.5, 0.16),
    # Met
    "ATG": ("M", 22.8, 1.00),
    # Val
    "GTT": ("V", 11.0, 0.18), "GTC": ("V", 14.5, 0.24),
    "GTA": ("V", 7.1, 0.12),  "GTG": ("V", 28.6, 0.46),
    # Ser
    "TCT": ("S", 15.2, 0.19), "TCC": ("S", 17.7, 0.22),
    "TCA": ("S", 12.2, 0.15), "TCG": ("S", 4.4, 0.05),
    "AGT": ("S", 12.1, 0.15), "AGC": ("S", 19.5, 0.24),
    # Pro
    "CCT": ("P", 17.5, 0.29), "CCC": ("P", 19.8, 0.33),
    "CCA": ("P", 16.9, 0.28), "CCG": ("P", 6.9, 0.11),
    # Thr
    "ACT": ("T", 13.1, 0.24), "ACC": ("T", 18.9, 0.35),
    "ACA": ("T", 15.1, 0.28), "ACG": ("T", 6.1, 0.11),
    # Ala
    "GCT": ("A", 18.4, 0.27), "GCC": ("A", 28.5, 0.41),
    "GCA": ("A", 16.0, 0.23), "GCG": ("A", 7.6, 0.11),
    # Tyr
    "TAT": ("Y", 12.2, 0.47), "TAC": ("Y", 15.6, 0.53),
    # Stop
    "TAA": ("*", 1.0, 0.30),  "TAG": ("*", 0.7, 0.20), "TGA": ("*", 1.6, 0.50),
    # His
    "CAT": ("H", 10.9, 0.42), "CAC": ("H", 15.1, 0.58),
    # Gln
    "CAA": ("Q", 12.3, 0.25), "CAG": ("Q", 34.2, 0.75),
    # Asn
    "AAT": ("N", 17.0, 0.47), "AAC": ("N", 19.1, 0.53),
    # Lys
    "AAA": ("K", 24.4, 0.43), "AAG": ("K", 31.9, 0.57),
    # Asp
    "GAT": ("D", 21.8, 0.46), "GAC": ("D", 25.1, 0.54),
    # Glu
    "GAA": ("E", 29.0, 0.42), "GAG": ("E", 39.6, 0.58),
    # Cys
    "TGT": ("C", 10.6, 0.45), "TGC": ("C", 12.6, 0.55),
    # Trp
    "TGG": ("W", 13.2, 1.00),
    # Arg
    "CGT": ("R", 4.5, 0.08),  "CGC": ("R", 10.4, 0.19),
    "CGA": ("R", 6.2, 0.11),  "CGG": ("R", 11.4, 0.21),
    "AGA": ("R", 12.2, 0.22), "AGG": ("R", 12.0, 0.22),
    # Gly
    "GGT": ("G", 10.8, 0.16), "GGC": ("G", 22.2, 0.34),
    "GGA": ("G", 16.5, 0.25), "GGG": ("G", 16.5, 0.25),
}


# ============================================================================
# Multi-species codon usage table summary
# ============================================================================

#: species name -> codon usage table
#: supported species: ecoli / yeast / human
SPECIES_CODON_USAGE: dict[str, dict[str, tuple[str, float, float]]] = {
    "ecoli": ECOLI_CODON_USAGE,
    "yeast": YEAST_CODON_USAGE,
    "human": HUMAN_CODON_USAGE,
}

#: species display name
SPECIES_DISPLAY_NAMES: dict[str, str] = {
    "ecoli": "Escherichia coli K-12 MG1655",
    "yeast": "Saccharomyces cerevisiae S288C",
    "human": "Homo sapiens",
}


def get_codon_usage(species: str = "ecoli") -> dict[str, tuple[str, float, float]]:
    """Get the codon usage frequency table for the given species.

    Args:
        species: species name ("ecoli" / "yeast" / "human")

    Returns:
        codon usage table {codon -> (amino_acid, per_thousand,
        fraction)}
    """
    if species not in SPECIES_CODON_USAGE:
        raise ValueError(
            f"unknown species {species!r}; "
            f"available: {list(SPECIES_CODON_USAGE.keys())}"
        )
    return SPECIES_CODON_USAGE[species]


def get_species_display_name(species: str) -> str:
    """Get the Latin display name of the species."""
    return SPECIES_DISPLAY_NAMES.get(species, species)


def codon_adaptation_index(codon: str, species: str = "ecoli") -> float:
    """Codon adaptation index (simplified CAI): the proportion of this
    synonymous codon within its amino acid.

    1.0 = optimal codon, <0.3 = rare codon.

    Args:
        codon:   codon (3 nt)
        species: species name ("ecoli" / "yeast" / "human")
    """
    table = get_codon_usage(species)
    if codon not in table:
        return 0.0
    return table[codon][2]


def is_optimal_codon(codon: str, species: str = "ecoli") -> bool:
    """Whether it is a high-frequency optimal codon for the given
    species (fraction >= 0.4).

    Args:
        codon:   codon (3 nt)
        species: species name ("ecoli" / "yeast" / "human")
    """
    return codon_adaptation_index(codon, species) >= 0.4


def is_rare_codon(codon: str, species: str = "ecoli") -> bool:
    """Whether it is a rare codon for the given species (fraction <
    0.15).

    Args:
        codon:   codon (3 nt)
        species: species name ("ecoli" / "yeast" / "human")
    """
    return codon_adaptation_index(codon, species) < 0.15


def _codon_family_max_fraction(
    table: dict[str, tuple[str, float, float]],
) -> dict[str, float]:
    """Maximal synonymous-codon fraction per amino acid.

    Each amino-acid family's reference (Sharp & Li 1987) weight is the
    fraction of its most abundant codon.
    """
    family_max: dict[str, float] = {}
    for aa, _per_thousand, fraction in table.values():
        family_max[aa] = max(family_max.get(aa, 0.0), fraction)
    return family_max


def cai(sequence: str, species: str = "ecoli", simplified: bool = False) -> float:
    """Codon adaptation index (Sharp & Li 1987 Nucleic Acids Res
    15:1281-1295).

    For every sense codon ``c`` the relative adaptiveness is the ratio
    of its synonymous-codon fraction to the maximum fraction within its
    amino-acid family::

        w(c) = f_c / max_{j in family(c)} f_j

    ``simplified=False`` (default) returns the true Sharp-Li CAI, the
    geometric mean of ``w(c)`` over all sense codons::

        CAI = exp(mean(log w(c)))

    ``simplified=True`` returns the legacy per-codon-fraction arithmetic
    mean (the "proportion of preferred codons" approximation) for
    backward compatibility.

    Stop codons and unknown codons are skipped. A sequence containing a
    codon whose family maximum is undefined (or a non-coding empty
    sequence) returns 0.0. The most abundant codon of each family always
    contributes ``w = 1.0``.

    Args:
        sequence:  coding DNA sequence (multiples of 3 nt)
        species:   species name ("ecoli" / "yeast" / "human")
        simplified: use the legacy arithmetic-mean approximation
    """
    table = get_codon_usage(species)
    family_max = _codon_family_max_fraction(table)
    cds = sequence.upper()
    n_codons = len(cds) // 3
    if n_codons == 0:
        return 0.0
    log_sum = 0.0
    frac_sum = 0.0
    n_sense = 0
    for i in range(n_codons):
        codon = cds[i * 3 : i * 3 + 3]
        entry = table.get(codon)
        if entry is None or entry[0] == "*":
            continue
        frac = entry[2]
        w = frac / family_max[entry[0]] if family_max[entry[0]] > 0 else 0.0
        if simplified:
            frac_sum += frac
        elif w > 0:
            log_sum += math.log(w)
        else:
            return 0.0
        n_sense += 1
    if n_sense == 0:
        return 0.0
    if simplified:
        return frac_sum / n_sense
    return math.exp(log_sum / n_sense)


# ============================================================================
# lac operon real promoter strength (Miller units, beta-galactosidase
# activity)
# Data source: Miller 1972 Experiments in Molecular Genetics
#              Kennedy 1977 J Mol Biol
#              Bayer 1988 J Bacteriol (lacI repression assays)
# ============================================================================

# Miller unit (MU) ~ nmol ONPG hydrolyzed/min/cell, reflects promoter
# transcription strength
# induced vs repressed ratio is ~1000x (lacI repression efficiency)
LAC_OPERON_PARAMS: dict[str, dict] = {
    "lacI": {
        "promoter": "lacIp",
        "promoter_strength_mu": 1.0,        # lacI promoter, constitutive
                                            # weak expression
        "repressor_per_cell": 10,           # ~10 tetramer/cell
                                            # (Miller 1972)
        "kd_dna": 1e-13,                    # lacI-operator Kd ~0.1 pM
        "kd_iptg": 1.5e-6,                  # lacI-IPTG Kd ~1.5 uM
    },
    "lacZ": {
        "promoter": "lacP",
        "promoter_strength_uninduced_mu": 3.0,    # repressed state ~3 MU
        "promoter_strength_induced_mu": 3000.0,   # IPTG-induced state
                                                  # ~3000 MU
        "induction_ratio": 1000,                  # 1000x induction ratio
        "km_lactose": 1.2e-3,                     # beta-gal Km for lactose
                                                  # ~1.2 mM
        "km_onpg": 9.5e-4,                        # beta-gal Km for ONPG
                                                  # ~0.95 mM
        "kcat": 600,                              # beta-gal kcat ~600/s
    },
    "lacY": {
        "promoter": "lacP",
        "promoter_strength_uninduced_mu": 3.0,
        "promoter_strength_induced_mu": 3000.0,
        "permease_per_cell_induced": 1500,        # induced state ~1500
                                                  # permease/cell
    },
    "lacA": {
        "promoter": "lacP",
        "promoter_strength_uninduced_mu": 3.0,
        "promoter_strength_induced_mu": 3000.0,
    },
}


def lac_promoter_strength(induced: bool = False) -> float:
    """lac operon promoter strength (normalized to GRN threshold range
    0..1).

    induced=False: repressed state 3 MU -> 0.003
    induced=True:  induced state 3000 MU -> 1.0
    """
    if induced:
        return 1.0
    return 3.0 / 3000.0  # 0.001


def lac_repression_factor(iptg_concentration: float) -> float:
    """Fraction of lacI derepression under IPTG induction (0=fully
    repressed, 1=fully induced).

    Hill equation: f = [IPTG]^n / (Kd^n + [IPTG]^n), n~1.5 (Bayer 1988)
    """
    kd = float(LAC_OPERON_PARAMS["lacI"]["kd_iptg"])
    n = 1.5
    return float((iptg_concentration ** n) / (kd ** n + iptg_concentration ** n))


# ============================================================================
# Gray-Scott reaction-diffusion parameter presets
# Data source: Pearson 1993 Complex Systems 7:331-378
#              https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/
# 14 measured parameter sets corresponding to different Turing patterns
# ============================================================================

@dataclass(slots=True, frozen=True)
class GrayScottPreset:
    name: str
    F: float           # feed rate
    k: float           # kill rate
    Du: float = 0.16   # U diffusion (Pearson standard)
    Dv: float = 0.08   # V diffusion (Pearson standard)
    description: str = ""


GRAY_SCOTT_PRESETS: list[GrayScottPreset] = [
    GrayScottPreset("Bacteria",     0.014, 0.045, description="bacterial-like spots"),
    GrayScottPreset("Coral",        0.016, 0.048, description="coral-like fractals"),
    GrayScottPreset("Fingerprint",  0.034, 0.057, description="fingerprint-like stripes"),
    GrayScottPreset("Solitons",     0.030, 0.062, description="soliton ripples"),
    GrayScottPreset("Mazes",        0.029, 0.057, description="maze-like networks"),
    GrayScottPreset("Holes",        0.039, 0.058, description="hole structures"),
    GrayScottPreset("Spots",        0.035, 0.065, description="spot array (default)"),
    GrayScottPreset("Worms",        0.058, 0.065, description="worm-like"),
    GrayScottPreset("Mitosis",      0.0367, 0.0649, description="mitosis-like"),
    GrayScottPreset("Pearson_alpha",0.014, 0.045, description="Pearson alpha region"),
    GrayScottPreset("Pearson_beta", 0.022, 0.059, description="Pearson beta region"),
    GrayScottPreset("Pearson_gamma",0.050, 0.065, description="Pearson gamma region"),
    GrayScottPreset("Pearson_delta",0.030, 0.060, description="Pearson delta region"),
    GrayScottPreset("Pearson_epsilon", 0.094, 0.057, description="Pearson epsilon region"),
]


def get_gray_scott_preset(name: str) -> GrayScottPreset:
    """Get a Gray-Scott preset by name."""
    for p in GRAY_SCOTT_PRESETS:
        if p.name.lower() == name.lower():
            return p
    raise ValueError(f"unknown Gray-Scott preset {name!r}; "
                     f"available: {[p.name for p in GRAY_SCOTT_PRESETS]}")


# ============================================================================
# Measured PCR error rates (Saiki 1988 / Potapov 2017)
# ============================================================================
# Data source:
# - Saiki et al. Science 1988 239:487-491 (introduced Taq for PCR, no
#   per-base rate reported)
# - Tindall & Kunkel Biochemistry 1988 27:6008-6013 (classic Taq
#   fidelity measurements)
# - Potapov & Ong PLoS ONE 2017 12(1):e0169774 (NEB comparison of
#   Taq/Pfu/Q5/Phusion)
# - Lee et al. Nucleic Acids Res 2016 44(13):e118 (single-molecule
#   sequencing maps polymerase errors)

# polymerase error rates per base per doubling (Potapov 2017)
PCR_ERROR_RATES: dict[str, float] = {
    "substitution_taq":         1.5e-4,   # Taq standard mode (Potapov 2017)
    "substitution_pfu":         5.1e-6,   # Pfu proofreading mode
    "substitution_q5":          5.3e-7,   # Q5 high-fidelity (NEB, ~280x Taq)
    "substitution_phusion":     3.9e-6,   # Phusion (~39-50x Taq)
    "indel_taq":                4.5e-6,   # indels are ~1-3% of total errors
    "indel_pfu":                1.5e-7,
    "indel_q5":                 1.6e-8,
    "indel_phusion":            1.2e-7,
}

DEFAULT_PCR_CYCLES_REAL = 30  # standard PCR, 30 cycles

# Transition / Transversion bias matrix (Potapov 2017)
# Transitions (A<->G, C<->T) account for ~86%, Transversions ~14%
# Ratio ~ 6:1, caused by keto-enol tautomerism + geometric similarity
# Note: this is P(transition | substitution) ~ 0.857 (a single
# probability), unlike TRANSITION_TRANSVERSION_RATIO in evolution.py
# (=2.0, a transition:transversion count ratio). The two mean different
# things, hence the rename to BIO_SUBSTITUTION_TRANSITION_PROB to
# remove ambiguity.
BIO_SUBSTITUTION_TRANSITION_PROB = 6.0 / 7.0  # P(transition | substitution) ~ 0.857
# backward-compatible alias (deprecated): keep the old name to preserve
# existing imports; new code should use
# BIO_SUBSTITUTION_TRANSITION_PROB.
TRANSITION_TRANSVERSION_RATIO = BIO_SUBSTITUTION_TRANSITION_PROB  # deprecated alias

# Taq mutation spectrum (Potapov 2017, consistent for Sanger/PacBio)
# A->G / T->C: 66%  | G->A / C->T: 19-21%  | A->T / T->A: 9-10%  | others:
# ~3-4%
TAQ_MUTATION_SPECTRUM: dict[str, dict[str, float]] = {
    "A": {"G": 0.66, "T": 0.10, "C": 0.04},  # A->G accounts for 66%
    "T": {"C": 0.66, "A": 0.10, "G": 0.04},  # T->C accounts for 66%
                                             # (complement)
    "G": {"A": 0.20, "C": 0.04, "T": 0.02},  # G->A accounts for 20%
    "C": {"T": 0.20, "G": 0.04, "A": 0.02},  # C->T accounts for 20%
                                             # (complement)
}

# ============================================================================
# Illumina sequencing error rates (Schirmer 2016)
# ============================================================================
ILLUMINA_ERROR_RATES: dict[str, float] = {
    "hiseq_novaseq_per_base":  1.0e-3,   # 0.1% (Q30 threshold)
    "miseq_nextseq_per_base":  5.0e-3,   # 0.5%
    "q30_threshold":           1.0e-3,   # Q30 = 99.9% accuracy
}


# ============================================================================
# All-platform sequencing error rates (Ceze, Nivala, Strauss Nat Rev
# Genet 2019 20:456-466 review)
# ============================================================================
# Covers the three major platforms: Illumina SBS, PacBio HiFi CCS, ONT
# R10.4
# Illumina: substitution-dominated, Q30=1e-3
# PacBio HiFi: random errors, CCS Q40+ ~ 1e-4
# ONT R10.4: indel-dominated, simplex ~1%, duplex <1%
SEQUENCING_PLATFORM_ERROR_RATES: dict[str, dict[str, float | str]] = {
    "illumina_hiseq_novaseq": {
        "substitution": 1.0e-3,   # Q30
        "indel":        1.0e-4,   # few indels
        "description":  "Illumina SBS, Q30, substitution-dominated",
    },
    "illumina_miseq": {
        "substitution": 5.0e-3,
        "indel":        5.0e-4,
        "description":  "Illumina MiSeq 2x300, higher per-base error",
    },
    "pacbio_hifi": {
        "substitution": 1.0e-4,   # Q40+
        "indel":        1.0e-4,
        "description":  "PacBio HiFi CCS, random errors, >=99.9% read accuracy",
    },
    "ont_r10_4_simplex": {
        "substitution": 5.0e-3,
        "indel":        1.0e-2,   # indel-dominated
        "description":  "Oxford Nanopore R10.4 simplex, indel-dominated",
    },
    "ont_r10_4_duplex": {
        "substitution": 5.0e-4,
        "indel":        1.0e-3,
        "description":  "Oxford Nanopore R10.4 duplex, <1% per-base",
    },
}


# ============================================================================
# DNA chemical synthesis error rates (Filges, Mouhanna, Stahlberg 2021)
# ============================================================================
# Data source: Filges et al. Clinical Chemistry 2021 67(10):1384-1394
#         DOI: 10.1093/clinchem/hvab136
# Measured on commercial oligo vendors: IDT/Eurofins/Sigma-Aldrich/
# BioSearch
#
# Key findings:
# - phosphoramidite coupling efficiency 98.5-99.5%/base (typically 99%)
# - 140-mer full-length fraction: 98.5% coupling -> ~10%, 99.5% coupling
#   -> ~50%
# - error spectrum: deletion-dominated (deletion:substitution ~ 7:1)
# - overall accuracy 97.2% (across all vendors/purity grades)
SYNTHESIS_ERROR_RATES: dict[str, float] = {
    # per-base coupling efficiency
    "coupling_efficiency_low":     0.985,    # 98.5% (Filges 2021 lower bound)
    "coupling_efficiency_typical": 0.99,     # 99% (median)
    "coupling_efficiency_high":    0.995,    # 99.5% (Filges 2021 upper bound)
    # per-base deletion probability ~ 1 - coupling_efficiency
    "deletion_rate_typical":       1.0e-2,   # 1% (Filges 2021)
    "deletion_rate_high":          5.0e-3,   # 0.5% (high purity)
    # per-base substitution probability (far below deletion)
    "substitution_rate_typical":   1.4e-3,   # ~deletion/7 (Filges 2021)
    "substitution_rate_high":      7.0e-4,
    # insertions are rarer
    "insertion_rate_typical":      1.0e-4,
    # full-length fraction (140-mer)
    "full_length_fraction_140mer_low":   0.10,  # 98.5% coupling
    "full_length_fraction_140mer_high":  0.50,  # 99.5% coupling
    # overall oligo accuracy (all vendors averaged)
    "overall_intact_oligo_fraction":     0.972,  # 97.2% (Filges 2021)
}

# Deletion : Substitution ratio (Filges 2021)
SYNTHESIS_DELETION_TO_SUBSTITUTION_RATIO = 7.0


# ============================================================================
# DNA storage density benchmarks (measured values, multiple papers)
# ============================================================================
# Data source:
# - Goldman et al. Nature 2013 494:77-80 (DOI: 10.1038/nature11875)
# - Erlich & Zielinski Science 2017 355:950-954 (DOI: 10.1126/science.aaj2038)
# - Organick et al. Nat Biotechnol 2018 36:242-248 (DOI: 10.1038/nbt.4079)
# - Ceze, Nivala, Strauss Nat Rev Genet 2019 20:456-466 (review)
#
# Shannon limit: log2(4) - GC/homopolymer constraint loss ~ 1.58 bit/nt
DNA_STORAGE_DENSITY_BENCHMARKS: dict[str, dict] = {
    "goldman_2013": {
        "density_bit_per_nt": 0.29,
        "archive_size_bytes": 757_000,
        "scheme": "Huffman ternary + 4x overlap redundancy",
        "theoretical_pb_per_gram": 0.83e6,  # 0.83 PB/g
        "citation": "Goldman et al. Nature 2013 494:77-80",
    },
    "erlich_2017": {
        "density_bit_per_nt": 1.57,
        "archive_size_bytes": 2_140_000,
        "scheme": "LT fountain + RSD + RS inner code",
        "theoretical_pb_per_gram": 2.14e6,  # 2.14 PB/g
        "shannon_limit_bit_per_nt": 1.58,
        "shannon_efficiency": 1.57 / 1.58,  # 99.4%
        "citation": "Erlich & Zielinski Science 2017 355:950-954",
    },
    "organick_2018": {
        "density_bit_per_nt": 0.83,
        "archive_size_bytes": 200_000,
        "scheme": "Primer-indexed random access",
        "num_files": 35,
        "citation": "Organick et al. Nat Biotechnol 2018 36:242-248",
    },
}

# Shannon limit (constrained by GC 45-55% + homopolymers <=3)
DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT = 1.58


# ============================================================================
# DNA decay/storage stability (Allentoft 2012, Grass 2015)
# ============================================================================
# Data source:
# - Allentoft et al. Proc R Soc B 2012 279:4724-4733 (DOI:
#   10.1098/rspb.2012.1745)
#   measured half-lives of 324bp mtDNA in 158 dated moa bones
# - Grass et al. Angew Chem Int Ed 2015 54:2552-2555 (DOI:
#   10.1002/anie.201411378)
#   accelerated aging experiments on silica-encapsulated DNA
#
# Key findings:
# - bone DNA half-life 521 years (13.1C), temperature-dependent via
#   Arrhenius
# - silica encapsulation: 70C -> 2000+ years, 9C -> ~2 million years
DNA_DECAY_RATES: dict[str, dict] = {
    "bone_dna": {
        "half_life_years_at_13c": 521,
        "temperature_c": 13.1,
        "activation_energy_kj_per_mol": 110,  # Allentoft 2012 estimate
        "citation": "Allentoft et al. Proc R Soc B 2012 279:4724-4733",
    },
    "silica_encapsulated": {
        "half_life_years_at_70c": 2000,
        "half_life_years_at_9c":  2_000_000,
        "citation": "Grass et al. Angew Chem Int Ed 2015 54:2552-2555",
    },
    "frozen_minus_20": {
        "half_life_years_estimated": 10_000,  # estimate
        "temperature_c": -20,
        "citation": "estimated, based on Allentoft Arrhenius extrapolation",
    },
}


def dna_decay_half_life(temperature_c: float,
                        encapsulated: bool = False) -> float:
    """Compute the DNA half-life (years).

    Based on the Allentoft 2012 Arrhenius model:
        k(T) = A * exp(-Ea / (R * T))
        t_half(T2) / t_half(T1) = exp(Ea/R * (1/T2 - 1/T1))

    Allentoft 2012: bone DNA 13.1C -> 521 years, Ea ~ 110 kJ/mol
    Grass 2015: silica-encapsulated 70C -> 2000 years (more stable)

    encapsulated=False: bare DNA / aqueous solution (Allentoft model)
    encapsulated=True: silica-encapsulated (Grass 2015 model, slower
                       decay)
    """
    R = 8.314e-3  # kJ/(mol*K)
    T1 = 13.1 + 273.15
    T2 = temperature_c + 273.15

    if encapsulated:
        # Grass 2015: silica-encapsulated 70C -> 2000 years, 9C -> ~2
        # million years
        # back-calculate Ea: ln(2e6/2e3) = (Ea/R) * (1/282.15 - 1/343.15)
        #   Ea ~ 91 kJ/mol (lower than bone DNA's 110 kJ/mol;
        #   encapsulation lowers the activation energy)
        t1 = 2000.0
        T1 = 70.0 + 273.15
        Ea = 91.0
    else:
        # Allentoft 2012: bone DNA 13.1C -> 521 years
        t1 = 521.0
        Ea = 110.0

    # Arrhenius: ln(t2/t1) = (Ea/R) * (1/T2 - 1/T1)
    # half-life is inversely proportional to rate, so
    # t2/t1 = exp((Ea/R) * (1/T2 - 1/T1))
    ratio = math.exp((Ea / R) * (1.0 / T2 - 1.0 / T1))
    return t1 * ratio


def dna_survival_fraction(years: float, temperature_c: float = 13.1,
                          encapsulated: bool = False) -> float:
    """Fraction of DNA surviving after `years` at the given temperature.

    Uses the exponential decay model: N(t) = N0 * exp(-t * ln2 / t_half)
    """
    t_half = dna_decay_half_life(temperature_c, encapsulated=encapsulated)
    return math.exp(-years * math.log(2.0) / t_half)


# ============================================================================
# Convenience: map real promoter strength to a GRN threshold
# ============================================================================

def mu_to_grn_strength(mu: float, max_mu: float = 3000.0) -> float:
    """Miller units -> GRN threshold (range 0..1).

    GRN uses sigmoid(x - threshold); the lower the threshold, the
    stronger the promoter. So we return (1 - mu/max_mu) as the
    threshold: a strong promoter -> low threshold. A constitutive weak
    promoter (mu < max_mu*0.01) -> threshold near 1.0 (weak
    expression).
    """
    return max(0.0, min(1.0, 1.0 - mu / max_mu))


# ============================================================================
# E. coli tRNA abundance table (based on Dong 1996 J Mol Biol
# 260:649-663)
# ============================================================================
# Data source: Dong, Kirsebom, Nomenclature et al., J Mol Biol 1996
# 260:649-663
# Measurement conditions: E. coli K-12 MG1655, MOPS glucose medium, 37C,
# OD600~0.6
# Field: codon -> tRNA copies/cell (approximate)
#
# Key features:
# - wobble pairing lets one tRNA read multiple codons (e.g.
#   tRNA-Leu-CAG reads CUG/CUU/CUC)
# - high-frequency codons correspond to high-abundance tRNAs (e.g.
#   CUG->tRNA-Leu-CAG ~3500/cell)
# - rare codons correspond to low-abundance tRNAs (e.g.
#   CUA->tRNA-Leu-UAG ~200/cell)
# - stop codons have no cognate tRNA (abundance = 0)
#
# Note: this table originally lived in central_dogma.py; it was
# physically moved here to remove the bio_data<->central_dogma circular
# dependency (bio_data is the low-level base module; central_dogma
# imports it one way).
TRNA_ABUNDANCE: dict[str, int] = {
    # Phe (1 tRNA: Phe-GAA, wobble reads UUU/UUC)
    "TTT": 2800, "TTC": 2800,
    # Leu (5 tRNAs; Leu-CAG high-abundance reads CUG/CUU/CUC, Leu-UAG
    # low-abundance reads CUA)
    "TTA": 700,  "TTG": 1000,
    "CTT": 800,  "CTC": 800,
    "CTA": 200,  # rare! tRNA-Leu-UAG
    "CTG": 3500, # most abundant! tRNA-Leu-CAG (E. coli's most common
                 # Leu codon)
    # Ile (3 codons; Ile-UAU is rare and reads AUA, requiring lysidine
    # modification)
    "ATT": 2300, "ATC": 2300,
    "ATA": 200,  # rare! tRNA-Ile-UAU (can only read AUA after lysidine
                 # modification)
    # Met (1 codon; tRNA-Met-CAU elongator, plus initiator fMet-tRNA)
    "ATG": 2000,
    # Val (2 tRNAs: Val-GAC reads GUU/GUC at high levels, Val-UAC reads
    # GUA/GUG at moderate levels)
    "GTT": 2300, "GTC": 2300,
    "GTA": 1500, "GTG": 1500,
    # Ser (5 tRNAs for 6 codons, extensive wobble pairing)
    "TCT": 1500, "TCC": 1500,
    "TCA": 700,  "TCG": 1000,
    "AGT": 800,  "AGC": 800,
    # Pro (2 tRNAs)
    "CCT": 1000, "CCC": 1000,
    "CCA": 2200, "CCG": 2200,
    # Thr (2 tRNAs)
    "ACT": 1600, "ACC": 1600,
    "ACA": 1000, "ACG": 1000,
    # Ala (2 tRNAs; Ala-GGC high-abundance reads GGC/GGU)
    "GCT": 1500, "GCC": 1500,
    "GCA": 2500, "GCG": 2500,
    # Tyr (1 tRNA: Tyr-GUA, wobble reads UAU/UAC)
    "TAT": 1700, "TAC": 1700,
    # His (1 tRNA: His-GUG)
    "CAT": 1100, "CAC": 1100,
    # Gln (2 tRNAs; Gln-CUG reads CAG at high levels)
    "CAA": 1500, "CAG": 2800,
    # Asn (1 tRNA: Asn-GUU)
    "AAT": 2500, "AAC": 2500,
    # Lys (2 tRNAs; Lys-UUU reads AAA at high levels)
    "AAA": 3000, "AAG": 1000,
    # Asp (1 tRNA: Asp-GUC)
    "GAT": 2500, "GAC": 2500,
    # Glu (2 tRNAs; Glu-UUC reads GAA at high levels)
    "GAA": 3000, "GAG": 800,
    # Cys (1 tRNA: Cys-GCA)
    "TGT": 800,  "TGC": 800,
    # Trp (1 tRNA, only 1 codon)
    "TGG": 1000,
    # Arg (5 tRNAs for 6 codons; Arg-ICG reads CGU/CGC/CGA, Arg-UCU/CCU
    # are rare)
    "CGT": 2000, "CGC": 2000, "CGA": 2000,
    "CGG": 600,
    "AGA": 150,  # rare! tRNA-Arg-UCU
    "AGG": 100,  # rare! tRNA-Arg-CCU
    # Gly (2 tRNAs; Gly-GCC high-abundance reads GGC/GGU)
    "GGT": 3200, "GGC": 3200,
    "GGA": 800,  "GGG": 800,
    # Stop codons - no cognate tRNA (recognized by release factors)
    "TAA": 0, "TAG": 0, "TGA": 0,
}

# max tRNA abundance (used to normalize codon-specific translation
# rates)
MAX_TRNA_ABUNDANCE = max(TRNA_ABUNDANCE.values())  # = 3500 (CTG)


# Yeast (S. cerevisiae) tRNA gene copy numbers (Chan & Lowe GtRNAdb)
# ~274 tRNA genes total; abundance estimated as gene copy number x
# expression coefficient
# Reference: Tuller et al. Cell 2010 141:342-354 (yeast translation
# efficiency positively correlates with tRNA abundance)
YEAST_TRNA_ABUNDANCE: dict[str, int] = {
    # Phe (2 genes: GAA)
    "TTT": 12, "TTC": 12,
    # Leu (9 genes total; CAG reads CUG/CUU/CUC, UAG reads CUA)
    "TTA": 4,  "TTG": 10,
    "CTT": 8,  "CTC": 8,
    "CTA": 3,  # rare
    "CTG": 14, # most abundant (yeast's most common Leu codon)
    # Ile (3 genes; UAU rare due to lysidine-like modification)
    "ATT": 10, "ATC": 10,
    "ATA": 2,  # rare
    # Met (4 genes: CAU, incl. initiator)
    "ATG": 13,
    # Val (6 genes; GAC reads GUU/GUC, UAC reads GUA/GUG)
    "GTT": 10, "GTC": 10,
    "GTA": 6,  "GTG": 6,
    # Ser (8 genes for 6 codons)
    "TCT": 7,  "TCC": 7,
    "TCA": 5,  "TCG": 4,
    "AGT": 6,  "AGC": 6,
    # Pro (4 genes)
    "CCT": 6,  "CCC": 4,
    "CCA": 10, "CCG": 4,
    # Thr (5 genes)
    "ACT": 8,  "ACC": 8,
    "ACA": 7,  "ACG": 5,
    # Ala (6 genes; GGC high-abundance)
    "GCT": 8,  "GCC": 8,
    "GCA": 12, "GCG": 6,
    # Tyr (3 genes: GUA)
    "TAT": 8,  "TAC": 8,
    # His (2 genes: GUG)
    "CAT": 5,  "CAC": 5,
    # Gln (5 genes; CUG reads CAG)
    "CAA": 9,  "CAG": 11,
    # Asn (4 genes: GUU)
    "AAT": 10, "AAC": 10,
    # Lys (7 genes; UUU reads AAA, CUU reads AAG)
    "AAA": 14, "AAG": 7,
    # Asp (4 genes: GUC)
    "GAT": 10, "GAC": 10,
    # Glu (6 genes; UUC reads GAA, CUC reads GAG)
    "GAA": 14, "GAG": 6,
    # Cys (2 genes: GCA)
    "TGT": 4,  "TGC": 4,
    # Trp (1 gene: CCA)
    "TGG": 6,
    # Arg (7 genes for 6 codons)
    "CGT": 6,  "CGC": 4,
    "CGA": 3,  "CGG": 2,
    "AGA": 11, # AGA is high-frequency in yeast
    "AGG": 4,
    # Gly (6 genes; GCC reads GGC/GGU)
    "GGT": 10, "GGC": 8,
    "GGA": 8,  "GGG": 4,
    # Stop codons - no cognate tRNA
    "TAA": 0, "TAG": 0, "TGA": 0,
}


# Human tRNA abundance (based on gene copy number + expression data)
# Reference: Chan & Lowe GtRNAdb; Dittmar KA et al. PLoS Genet 2006
# 2:e221
# Humans have ~610 tRNA genes (incl. pseudogenes), ~400-500
# functionally active
# abundance pattern is consistent with codon usage bias
HUMAN_TRNA_ABUNDANCE: dict[str, int] = {
    # Phe (2 genes: GAA, ~14 copies)
    "TTT": 14, "TTC": 14,
    # Leu (many genes; CAG reads CTG at high frequency)
    "TTA": 7,  "TTG": 12,
    "CTT": 10, "CTC": 12,
    "CTA": 4,  # rare
    "CTG": 20, # human's most common Leu codon
    # Ile (3 genes; AUA is not rare in humans because of the
    # tRNA-Ile-UAU modification)
    "ATT": 15, "ATC": 15,
    "ATA": 8,
    # Met (16 genes: CAU, incl. many initiators)
    "ATG": 22,
    # Val (many genes; GAC reads GTG/GTC)
    "GTT": 11, "GTC": 14,
    "GTA": 6,  "GTG": 16,
    # Ser (many genes for 6 codons)
    "TCT": 15, "TCC": 15,
    "TCA": 12, "TCG": 5,
    "AGT": 12, "AGC": 15,
    # Pro (many genes)
    "CCT": 12, "CCC": 12,
    "CCA": 17, "CCG": 7,
    # Thr (many genes)
    "ACT": 13, "ACC": 18,
    "ACA": 15, "ACG": 6,
    # Ala (many genes; GGC high-abundance)
    "GCT": 15, "GCC": 18,
    "GCA": 16, "GCG": 8,
    # Tyr (3 genes: GUA)
    "TAT": 10, "TAC": 12,
    # His (2 genes: GUG)
    "CAT": 7,  "CAC": 11,
    # Gln (many genes; CUG reads CAG)
    "CAA": 12, "CAG": 20,
    # Asn (3 genes: GUU)
    "AAT": 13, "AAC": 15,
    # Lys (many genes; UUU reads AAA)
    "AAA": 16, "AAG": 17,
    # Asp (4 genes: GUC)
    "GAT": 13, "GAC": 15,
    # Glu (many genes; CUC reads GAG at high frequency in humans)
    "GAA": 15, "GAG": 20,
    # Cys (3 genes: GCA)
    "TGT": 7,  "TGC": 10,
    # Trp (2 genes: CCA)
    "TGG": 7,
    # Arg (many genes for 6 codons; CGG is more frequent in humans than
    # in E. coli)
    "CGT": 8,  "CGC": 10,
    "CGA": 5,  "CGG": 11,
    "AGA": 12, "AGG": 11,
    # Gly (many genes; GCC reads GGC/GGU)
    "GGT": 13, "GGC": 16,
    "GGA": 15, "GGG": 11,
    # Stop codons - no cognate tRNA
    "TAA": 0, "TAG": 0, "TGA": 0,
}


#: species -> tRNA abundance table
SPECIES_TRNA_ABUNDANCE: dict[str, dict[str, int]] = {
    "ecoli": TRNA_ABUNDANCE,
    "yeast": YEAST_TRNA_ABUNDANCE,
    "human": HUMAN_TRNA_ABUNDANCE,
}


def get_species_trna(species: str = "ecoli") -> dict[str, int]:
    """Get the tRNA abundance table for the given species (codon ->
    tRNA copies/cell).

    Data sources:
    - E. coli: Dong et al. J Mol Biol 1996 260:649-663
    - Yeast: Chan & Lowe GtRNAdb 2009 (tRNA gene copy numbers)
    - Human: Chan & Lowe GtRNAdb 2009; Dittmar et al. PLoS Genet 2006

    Args:
        species: species name ("ecoli" / "yeast" / "human")

    Returns:
        {codon -> tRNA abundance}, stop codons have abundance 0
    """
    if species not in SPECIES_TRNA_ABUNDANCE:
        raise ValueError(
            f"unknown species {species!r}; "
            f"available: {list(SPECIES_TRNA_ABUNDANCE.keys())}"
        )
    return SPECIES_TRNA_ABUNDANCE[species]
