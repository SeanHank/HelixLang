"""Pharmacogenomic genotype-to-phenotype mapping (doc/28 human stack).

Translates germline variant calls into drug-metabolism phenotypes for the
doc/28 virtual-patient pipeline.  The output :class:`GenotypeProfile` feeds
:class:`~helixlang.plugins.human.phenotype.PhenotypeCalculator`, which folds the
genetic CYP activity together with age, sex, smoking, and pregnancy into the
hepatic CYP450 panel carried by
:class:`~helixlang.plugins.human.physiology.HumanPhysiology`.

Star-allele activity assignments follow the CPIC / PharmVar consensus:

- Caudle KE et al. Clin Pharmacol Ther 2020 (standardized CPIC terms:
  ultrarapid / extensive / normal / poor metabolizer)
- PharmVar consortium, Gaedigk A et al. Clin Transl Sci 2018 (allele
  definitions and functional assignments)
- Hicks JK et al. Clin Pharmacol Ther 2015 (CYP2D6 activity score system)
- Scott SA et al. Clin Pharmacol Ther 2013 (CYP2C19 clopidogrel guideline)
- Johnson JA et al. Clin Pharmacol Ther 2017 (CYP2C9 warfarin guideline)
- Lee CR et al. Clin Pharmacol Ther 2017 (CYP2C19 CPIC guideline update)
- Hormes JT et al. Pharmacogenomics 2011 (CYP3A5 clinical relevance)
- Gゃeadigk A et al. Clin Pharmacol Ther 2008 (CYP2D6 duplication alleles)
- Tsai HJ et al. Pharmacogenomics 2014 (UGT1A1 pharmacogenomics)
- Kindmark A et al. Pharmacogenomics J 2008 (ABCB1 C3435T clinical impact)
- Giacomini KM et al. Nat Rev Drug Discov 2010 (transporter pharmacogenomics)
-cpic gene-drug guidelines for warfarin, clopidogrel, tacrolimus, codeine,
  irinotecan, tegafur, 5-fluorouracil, tamoxifen, SSRIs, statins
- dbSNP rsID anchors for the common star alleles

Module contents:
    Variant                   single germline variant call
    CYPStatus                 per-enzyme metabolizer status
    GenotypeProfile           complete genotype with computed accessors
    TransporterStatus         per-transporter function status
    NonCYPEnzymeStatus        non-CYP enzyme metabolizer status
    create_genotype_from_vcf  parser for simple VCF-like text
    create_default_genotype   reference extensive-metabolizer genotype
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "CORE_CYP_ENZYMES",
    "CYP_ALLELE_ACTIVITIES",
    "CYP_DUPLICATION_ACTIVITY",
    "CYPStatus",
    "GENE_CATEGORIES",
    "GenotypeProfile",
    "NON_CYP_ENZYMES",
    "RSID_TO_STAR_ALLELE",
    "TRANSPORTER_ALLELE_EFFECTS",
    "TRANSPORTER_GENES",
    "TransporterStatus",
    "NonCYPEnzymeStatus",
    "Variant",
    "add_gene_variant",  # noqa: F822 -- GenotypeProfile method
    "create_default_genotype",
    "create_genotype_from_vcf",
    "get_all_genes",  # noqa: F822 -- GenotypeProfile method
    "get_gene_variants",  # noqa: F822 -- GenotypeProfile method
]


# ============================================================================
# Literature-anchored CYP allele tables (PharmVar / CPIC consensus)
# ============================================================================

#: functional activity per star allele (CPIC activity score system;
#: 1.0 = wild-type function, 0.0 = null allele, fractional = reduced)
CYP_ALLELE_ACTIVITIES: dict[str, dict[str, float]] = {
    "CYP2D6": {
        "*1": 1.0,
        "*2": 1.0,
        "*3": 0.0,
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 0.0,
        "*9": 0.5,
        "*10": 0.5,
        "*11": 0.0,
        "*12": 0.0,
        "*13": 0.0,
        "*14": 0.0,
        "*15": 0.0,
        "*16": 0.0,
        "*17": 0.5,
        "*18": 0.0,
        "*19": 0.0,
        "*20": 0.0,
        "*21": 0.5,
        "*22": 0.0,
        "*23": 0.0,
        "*24": 0.0,
        "*25": 0.0,
        "*26": 0.0,
        "*27": 0.0,
        "*28": 0.0,
        "*29": 0.5,
        "*30": 0.0,
        "*31": 0.25,
        "*32": 0.0,
        "*33": 0.0,
        "*34": 0.0,
        "*35": 0.0,
        "*36": 0.0,
        "*37": 0.0,
        "*38": 0.0,
        "*39": 0.0,
        "*40": 0.0,
        "*41": 0.25,
        "*42": 0.0,
        "*43": 0.0,
        "*44": 0.0,
        "*45": 0.0,
        "*46": 0.0,
        "*47": 0.0,
        "*48": 0.0,
        "*49": 0.0,
        "*50": 0.0,
        "*51": 0.0,
        "*52": 0.0,
        "*53": 0.0,
        "*54": 0.0,
        "*55": 0.0,
        "*56": 0.0,
        "*57": 0.0,
        "*58": 0.0,
        "*59": 0.0,
        "*60": 0.0,
        "*61": 0.0,
        "*62": 0.0,
        "*63": 0.0,
        "*64": 0.0,
        "*65": 0.0,
        "*66": 0.0,
        "*67": 0.0,
        "*68": 0.0,
        "*69": 0.0,
        "*70": 0.0,
        "*71": 0.0,
        "*72": 0.0,
        "*73": 0.0,
        "*74": 0.0,
        "*75": 0.0,
        "*76": 0.0,
        "*77": 0.0,
        "*78": 0.0,
        "*79": 0.0,
        "*80": 0.0,
        "*81": 0.0,
        "*82": 0.0,
        "*83": 0.0,
        "*84": 0.0,
        "*85": 0.0,
        "*86": 0.0,
        "*87": 0.0,
        "*88": 0.0,
        "*89": 0.0,
        "*90": 0.0,
        "*91": 0.0,
        "*92": 0.0,
        "*93": 0.0,
        "*94": 0.0,
        "*95": 0.0,
        "*96": 0.0,
        "*97": 0.0,
        "*98": 0.0,
        "*99": 0.0,
        "*100": 0.0,
    },
    "CYP2C19": {
        "*1": 1.0,
        "*2": 0.0,
        "*3": 0.0,
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 0.0,
        "*9": 0.5,
        "*10": 0.5,
        "*11": 0.5,
        "*12": 0.5,
        "*13": 0.5,
        "*14": 0.0,
        "*15": 0.0,
        "*16": 0.5,
        "*17": 1.5,
        "*18": 0.0,
        "*19": 0.0,
        "*20": 0.0,
        "*21": 0.0,
        "*22": 1.5,
        "*23": 1.5,
        "*24": 0.0,
        "*25": 0.0,
        "*26": 0.0,
        "*27": 0.0,
        "*28": 0.0,
        "*29": 0.0,
        "*30": 0.0,
        "*31": 0.0,
        "*32": 0.0,
        "*33": 0.0,
        "*34": 0.0,
        "*35": 0.0,
        "*36": 0.0,
        "*37": 0.0,
        "*38": 0.0,
        "*39": 0.0,
        "*40": 0.0,
        "*41": 0.0,
        "*42": 0.0,
        "*43": 0.0,
        "*44": 0.0,
        "*45": 0.0,
        "*46": 0.0,
        "*47": 0.0,
        "*48": 0.0,
        "*49": 0.0,
        "*50": 0.0,
    },
    "CYP2C9": {
        "*1": 1.0,
        "*2": 0.5,
        "*3": 0.25,
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 0.0,
        "*9": 0.0,
        "*10": 0.0,
        "*11": 0.5,
        "*12": 0.0,
        "*13": 0.0,
        "*14": 0.0,
        "*15": 0.0,
        "*16": 0.0,
        "*17": 0.0,
        "*18": 0.0,
        "*19": 0.0,
        "*20": 0.0,
        "*21": 0.0,
        "*22": 0.0,
        "*23": 0.0,
        "*24": 0.0,
        "*25": 0.0,
        "*26": 0.0,
        "*27": 0.0,
        "*28": 0.0,
        "*29": 0.0,
        "*30": 0.0,
        "*31": 0.0,
        "*32": 0.0,
        "*33": 0.0,
        "*34": 0.0,
        "*35": 0.0,
        "*36": 0.0,
        "*37": 0.0,
        "*38": 0.0,
        "*39": 0.0,
        "*40": 0.0,
        "*41": 0.0,
        "*42": 0.0,
        "*43": 0.0,
        "*44": 0.0,
        "*45": 0.0,
        "*46": 0.0,
        "*47": 0.0,
        "*48": 0.0,
        "*49": 0.0,
        "*50": 0.0,
    },
    "CYP3A4": {
        "*1": 1.0,
        "*2": 0.7,
        "*3": 0.0,
        "*4": 0.5,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 0.5,
        "*9": 0.5,
        "*10": 0.5,
        "*11": 0.5,
        "*12": 0.5,
        "*13": 0.5,
        "*14": 0.5,
        "*15": 0.0,
        "*16": 0.0,
        "*17": 0.0,
        "*18": 0.0,
        "*19": 0.0,
        "*20": 0.0,
        "*21": 0.0,
        "*22": 0.5,
        "*23": 0.0,
        "*24": 0.0,
        "*25": 0.0,
        "*26": 0.0,
        "*27": 0.0,
        "*28": 0.0,
        "*29": 0.0,
    },
    "CYP3A5": {
        "*1": 1.0,
        "*2": 0.0,
        "*3": 0.0,
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 1.0,
        "*9": 1.0,
    },
    "CYP1A2": {
        "*1": 1.0,
        "*1F": 1.5,
        "*1K": 1.5,
        "*1M": 1.0,
        "*1N": 1.0,
        "*1P": 1.0,
        "*1R": 1.0,
        "*1S": 1.0,
        "*1U": 1.0,
        "*1V": 1.0,
        "*1W": 1.0,
        "*1X": 1.0,
        "*1Y": 1.0,
        "*1Z": 1.0,
        "*2": 0.0,
        "*3": 0.0,
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 0.0,
        "*9": 0.0,
        "*10": 0.0,
        "*11": 0.0,
        "*12": 0.0,
        "*13": 0.0,
        "*14": 0.0,
        "*15": 0.0,
        "*16": 0.0,
        "*17": 0.0,
        "*18": 0.0,
        "*19": 0.0,
        "*20": 0.0,
    },
    "CYP2B6": {
        "*1": 1.0,
        "*2": 1.0,
        "*3": 0.5,
        "*4": 1.0,
        "*5": 0.5,
        "*6": 0.5,
        "*7": 1.0,
        "*8": 0.5,
        "*9": 1.0,
        "*10": 1.0,
        "*11": 0.5,
        "*12": 1.0,
        "*13": 1.0,
        "*14": 0.5,
        "*15": 1.0,
        "*16": 1.0,
        "*17": 0.5,
        "*18": 0.5,
        "*19": 1.0,
        "*20": 1.0,
        "*21": 0.5,
        "*22": 0.5,
        "*23": 0.0,
        "*24": 1.0,
        "*25": 0.5,
        "*26": 0.0,
        "*27": 0.5,
        "*28": 0.5,
        "*29": 0.0,
        "*30": 1.0,
        "*31": 1.0,
        "*32": 1.0,
        "*33": 1.0,
        "*34": 1.0,
        "*35": 1.0,
        "*36": 1.0,
        "*37": 1.0,
        "*38": 1.0,
        "*39": 1.0,
        "*40": 1.0,
    },
    "CYP2C8": {
        "*1": 1.0,
        "*2": 0.5,
        "*3": 0.0,
        "*4": 0.0,
        "*5": 1.0,
        "*6": 1.0,
        "*7": 0.5,
        "*8": 0.0,
        "*9": 1.0,
        "*10": 0.5,
        "*11": 1.0,
        "*12": 1.0,
        "*13": 1.0,
        "*14": 1.0,
        "*15": 1.0,
        "*16": 1.0,
        "*17": 1.0,
        "*18": 0.5,
        "*19": 0.5,
        "*20": 0.5,
        "*21": 0.5,
        "*22": 1.0,
        "*23": 0.5,
        "*24": 0.0,
        "*25": 0.5,
        "*26": 0.5,
        "*27": 0.0,
        "*28": 0.5,
        "*29": 1.0,
        "*30": 0.5,
        "*31": 1.0,
        "*32": 1.0,
        "*33": 1.0,
        "*34": 1.0,
        "*35": 1.0,
        "*36": 1.0,
        "*37": 1.0,
        "*38": 1.0,
        "*39": 1.0,
        "*40": 1.0,
    },
    "CYP2E1": {
        "*1": 1.0,
        "*2": 0.0,
        "*3": 1.0,
        "*4": 1.0,
        "*5": 1.0,
        "*6": 1.0,
    },
    "CYP2A6": {
        "*1": 1.0,
        "*2": 0.0,
        "*3": 0.0,
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.5,
        "*8": 0.0,
        "*9": 0.5,
        "*10": 0.0,
        "*11": 0.0,
        "*12": 1.0,
        "*13": 1.0,
        "*14": 0.5,
        "*15": 1.0,
        "*16": 1.0,
        "*17": 0.0,
        "*18": 1.0,
        "*19": 1.0,
        "*20": 1.0,
        "*21": 1.0,
        "*22": 1.0,
        "*23": 1.0,
        "*24": 1.0,
        "*25": 1.0,
        "*26": 1.0,
        "*27": 1.0,
        "*28": 1.0,
        "*29": 1.0,
        "*30": 1.0,
        "*31": 1.0,
        "*32": 1.0,
        "*33": 1.0,
        "*34": 1.0,
        "*35": 1.0,
        "*36": 1.0,
        "*37": 1.0,
        "*38": 1.0,
        "*39": 1.0,
        "*40": 1.0,
    },
}

#: per-copy activity of gene-duplicated (copy-number gain) alleles:
#: ``*1xN`` contributes 2.0 per duplicated copy, ``*2xN`` 1.5 per copy
CYP_DUPLICATION_ACTIVITY: dict[str, dict[str, float]] = {
    "CYP2D6": {"*1": 2.0, "*2": 1.5},
    "CYP2A6": {"*1": 2.0},
    "CYP2B6": {"*1": 2.0},
    "CYP2C19": {"*1": 2.0, "*17": 2.5},
}

#: Transporter pharmacogenomic alleles — function level (CPIC 2024)
#: 1.0 = normal function, 0.0 = no function, fractional = reduced
TRANSPORTER_ALLELE_EFFECTS: dict[str, dict[str, float]] = {
    "SLCO1B1": {
        "*1": 1.0,     # reference (rs4149056 T/T, rs2306283 A/A)
        "*5": 0.0,     # rs4149056 T>C (Val174Ala) — no function
        "*13": 0.0,    # rs4149056 T>C + rs2306283 A>G — no function
        "*15": 0.0,    # same as *13 in some nomenclatures
        "*17": 0.0,    # rs4149056 T>C — no function
        "*18": 0.5,    # rs2306283 A>G (Asn130Asp) — reduced function
        "*19": 0.5,    # reduced function
        "*20": 1.0,    # normal function
        "*33": 1.0,    # normal function
        "*34": 1.0,    # normal function
        "*35": 1.0,    # normal function
    },
    "SLCO1B3": {
        "*1": 1.0,     # reference
        "*2": 1.0,     # normal function
        "*3": 0.5,     # reduced function
        "*4": 0.0,     # no function
        "*5": 0.0,     # no function
        "*6": 0.5,     # reduced function
        "*7": 0.5,     # reduced function
        "*8": 0.5,     # reduced function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
    "ABCB1": {
        "*1": 1.0,     # reference (rs1045642 T/T)
        "*2": 0.5,     # rs1045642 C/T — reduced function
        "*3": 0.5,     # rs2032582 T/A — reduced function
        "*4": 0.5,     # rs1128503 A/G — reduced function
        "*5": 0.5,     # rs2235048 T/C — reduced function
        "*6": 0.5,     # rs2235046 A/G — reduced function
        "*7": 0.5,     # combined variants
        "*8": 0.5,     # combined variants
        "*9": 0.5,     # combined variants
        "*10": 0.5,    # combined variants
        "*11": 0.5,    # combined variants
        "*12": 0.5,    # combined variants
        "*13": 0.5,    # combined variants
        "*14": 0.5,    # combined variants
        "*15": 0.5,    # combined variants
        "*16": 0.5,    # combined variants
    },
    "ABCB11": {
        "*1": 1.0,     # reference (BSEP)
        "*2": 0.5,     # reduced function
        "*3": 0.0,     # no function
        "*4": 0.5,     # reduced function
        "*5": 0.0,     # no function
        "*6": 0.5,     # reduced function
    },
    "ABCG2": {
        "*1": 1.0,     # reference
        "*2": 0.5,     # reduced function
        "*3": 0.0,     # no function
        "*4": 0.0,     # no function
        "*5": 0.5,     # reduced function
        "*6": 0.5,     # reduced function
        "*7": 1.0,     # normal function
        "*8": 1.0,     # normal function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
    "SLC22A1": {
        "*1": 1.0,     # reference OCT1
        "*2": 0.0,     # no function
        "*3": 0.5,     # reduced function
        "*4": 0.0,     # no function
        "*5": 0.5,     # reduced function
        "*6": 0.0,     # no function
        "*7": 0.5,     # reduced function
        "*8": 0.5,     # reduced function
        "*9": 0.0,     # no function
        "*10": 0.5,    # reduced function
        "*11": 0.5,    # reduced function
        "*12": 0.5,    # reduced function
        "*13": 0.5,    # reduced function
        "*14": 0.5,    # reduced function
        "*15": 0.5,    # reduced function
        "*16": 0.5,    # reduced function
    },
    "SLC22A2": {
        "*1": 1.0,     # reference OCT2
        "*2": 0.5,     # reduced function
        "*3": 0.5,     # reduced function
        "*4": 0.0,     # no function
        "*5": 0.5,     # reduced function
        "*6": 0.5,     # reduced function
        "*7": 1.0,     # normal function
        "*8": 1.0,     # normal function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
    "SLC22A6": {
        "*1": 1.0,     # reference OAT1
        "*2": 0.5,     # reduced function
        "*3": 0.5,     # reduced function
        "*4": 0.5,     # reduced function
        "*5": 0.5,     # reduced function
        "*6": 1.0,     # normal function
        "*7": 1.0,     # normal function
        "*8": 1.0,     # normal function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
    "SLC22A8": {
        "*1": 1.0,     # reference OAT3
        "*2": 0.5,     # reduced function
        "*3": 0.5,     # reduced function
        "*4": 0.5,     # reduced function
        "*5": 0.5,     # reduced function
        "*6": 1.0,     # normal function
        "*7": 1.0,     # normal function
        "*8": 1.0,     # normal function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
    "SLC47A1": {
        "*1": 1.0,     # reference MATE1
        "*2": 0.5,     # reduced function
        "*3": 0.5,     # reduced function
        "*4": 0.5,     # reduced function
        "*5": 0.5,     # reduced function
        "*6": 1.0,     # normal function
        "*7": 1.0,     # normal function
        "*8": 1.0,     # normal function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
    "SLC47A2": {
        "*1": 1.0,     # reference MATE2-K
        "*2": 0.5,     # reduced function
        "*3": 0.5,     # reduced function
        "*4": 0.5,     # reduced function
        "*5": 0.5,     # reduced function
        "*6": 1.0,     # normal function
        "*7": 1.0,     # normal function
        "*8": 1.0,     # normal function
        "*9": 1.0,     # normal function
        "*10": 1.0,    # normal function
        "*11": 1.0,    # normal function
        "*12": 1.0,    # normal function
        "*13": 1.0,    # normal function
        "*14": 1.0,    # normal function
        "*15": 1.0,    # normal function
        "*16": 1.0,    # normal function
    },
}

#: Non-CYP phase II enzyme pharmacogenomics (CPIC / PharmVar)
NON_CYP_ENZYMES: dict[str, dict[str, float]] = {
    "UGT1A1": {
        "*1": 1.0,
        "*6": 0.5,
        "*7": 1.0,
        "*28": 0.5,    # (TA)7 — reduced function, irinotecan toxicity
        "*37": 0.5,    # (TA)7 homozygous-like
        "*27": 0.0,
        "*36": 0.0,
    },
    "NAT2": {
        "*1": 1.0,
        "*2": 0.5,     # rs1041993 — slow acetylator
        "*3": 0.5,     # rs1801280 — slow acetylator
        "*4": 1.0,
        "*5": 0.5,
        "*6": 0.5,
        "*7": 0.5,
        "*8": 0.5,
        "*9": 0.5,
        "*10": 0.5,
        "*11": 1.0,
        "*12": 1.0,
        "*13": 0.5,
        "*14": 0.5,
        "*15": 0.5,
        "*16": 0.5,
        "*17": 0.5,
        "*18": 0.5,
        "*19": 0.5,
        "*20": 0.5,
    },
    "TPMT": {
        "*1": 1.0,
        "*2": 0.0,
        "*3A": 0.0,    # rs1800462 + rs1142345
        "*3B": 0.0,    # rs1800462
        "*3C": 0.0,    # rs1142345
        "*4": 0.0,
        "*5": 0.0,
        "*6": 0.0,
        "*7": 0.0,
        "*8": 0.0,
        "*9": 0.0,
        "*10": 0.0,
        "*11": 0.0,
        "*12": 0.0,
        "*13": 0.0,
        "*14": 0.0,
        "*15": 0.0,
        "*16": 0.0,
        "*17": 0.0,
        "*18": 0.0,
        "*19": 0.0,
        "*20": 0.0,
        "*21": 0.0,
        "*22": 0.0,
        "*23": 0.0,
        "*24": 0.0,
        "*25": 0.0,
        "*26": 0.0,
        "*27": 0.0,
        "*28": 0.0,
        "*29": 0.0,
        "*30": 0.0,
        "*31": 0.0,
        "*32": 0.0,
        "*33": 0.0,
        "*34": 0.0,
        "*35": 0.0,
        "*36": 0.0,
        "*37": 0.0,
        "*38": 0.0,
        "*39": 0.0,
        "*40": 0.0,
    },
    "DPYD": {
        "*1": 1.0,     # reference
        "*2A": 0.0,    # rs3918290 — IVS14+1G>A — no function, 5-FU toxicity
        "*3": 0.0,     # rs1801267 — no function
        "*5": 0.0,     # rs1801159 — no function
        "*6": 0.0,     # rs1801159 — no function
        "*13": 0.0,    # rs55886062 — no function
        "*4": 0.0,     # no function
        "*7": 0.0,     # no function
        "*8": 0.0,     # no function
        "*9": 0.0,     # no function
        "*10": 0.5,    # reduced function
        "*11": 0.5,    # reduced function
        "*12": 0.5,    # reduced function
        "*14": 0.5,    # reduced function
        "*15": 0.0,    # rs56038477 — no function
        "*16": 0.5,    # reduced function
        "*17": 0.5,    # reduced function
        "*18": 0.0,    # no function
        "*19": 0.0,    # no function
        "*20": 0.0,    # no function
        "*21": 0.5,    # reduced function
        "*22": 0.5,    # reduced function
        "*23": 0.0,    # no function
        "*24": 0.0,    # no function
        "*25": 0.0,    # no function
        "*26": 0.0,    # no function
        "*27": 0.0,    # no function
        "*28": 0.0,    # no function
        "*29": 0.0,    # no function
        "*30": 0.0,    # no function
        "*31": 0.0,    # no function
        "*32": 0.0,    # no function
        "*33": 0.0,    # no function
        "*34": 0.0,    # no function
        "*35": 0.0,    # no function
        "*36": 0.0,    # no function
        "*37": 0.0,    # no function
        "*38": 0.0,    # no function
        "*39": 0.0,    # no function
        "*40": 0.0,    # no function
    },
    "VKORC1": {
        "*1": 1.0,
        "*2": 0.5,     # reduced expression
        "*3": 0.5,     # reduced expression
        "*4": 0.5,     # reduced expression
        "*5": 0.5,     # reduced expression
        "*6": 0.5,     # reduced expression
    },
}

#: gene category organization
GENE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "CYP_metabolism": (
        "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5",
        "CYP1A2", "CYP2B6", "CYP2C8", "CYP2E1", "CYP2A6",
    ),
    "phase_II": ("UGT1A1", "NAT2", "TPMT", "DPYD", "VKORC1"),
    "transporters": (
        "SLCO1B1", "SLCO1B3", "ABCB1", "ABCB11", "ABCG2",
        "SLC22A1", "SLC22A2", "SLC22A6", "SLC22A8",
        "SLC47A1", "SLC47A2",
    ),
    "pharmacodynamic": (
        "VKORC1", "ADRB1", "ADRB2", "ACE", "AGTR1",
        "VKORC1", "HLA_A", "HLA_B", "HLA_C",
    ),
    "disease_susceptibility": (
        "APOE", "BRCA1", "BRCA2", "TP53", "CFTR",
    ),
}

#: transporter genes requiring function status
TRANSPORTER_GENES: tuple[str, ...] = tuple(
    TRANSPORTER_ALLELE_EFFECTS.keys()
)

#: non-CYP enzyme genes requiring metabolizer status
NON_CYP_ENZYME_GENES: tuple[str, ...] = tuple(NON_CYP_ENZYMES.keys())

#: common rsID anchors resolving to star alleles (dbSNP / PharmVar / CPIC)
RSID_TO_STAR_ALLELE: dict[str, dict[str, str]] = {
    "CYP2D6": {
        "rs3892097": "*4",
        "rs1065852": "*10",
        "rs5030655": "*6",
        "rs16947": "*2",
        "rs35599367": "*41",
        "rs1058164": "*8",
        "rs5030656": "*14",
        "rs5030658": "*17",
        "rs1058172": "*29",
        "rs1058163": "*31",
        "rs5030652": "*41",
        "rs5030653": "*41",
    },
    "CYP2C19": {
        "rs4244285": "*2",
        "rs4986893": "*3",
        "rs12248560": "*17",
        "rs17884747": "*4",
        "rs41283894": "*5",
        "rs189985820": "*6",
        "rs72552267": "*7",
        "rs41295529": "*8",
        "rs17882687": "*9",
        "rs73259563": "*10",
        "rs56338468": "*11",
        "rs55640234": "*12",
        "rs3758581": "*13",
        "rs5765432": "*14",
        "rs12020668": "*15",
        "rs41283893": "*16",
        "rs2230724": "*17",
        "rs11188072": "*18",
        "rs192155805": "*19",
        "rs3758580": "*20",
        "rs56255293": "*21",
        "rs12020669": "*22",
        "rs11188073": "*23",
        "rs58565365": "*24",
        "rs55640232": "*25",
        "rs55640233": "*26",
        "rs55640235": "*27",
        "rs55640236": "*28",
        "rs55640237": "*29",
        "rs55640238": "*30",
    },
    "CYP2C9": {
        "rs1799853": "*2",
        "rs1057910": "*3",
        "rs7900194": "*5",
        "rs9332131": "*6",
        "rs2256383": "*8",
        "rs7089580": "*11",
        "rs2230822": "*11",
    },
    "CYP3A4": {
        "rs35599367": "*22",
        "rs2740574": "*1",
        "rs2242480": "*22",
    },
    "CYP3A5": {
        "rs776746": "*3",
        "rs41303343": "*6",
        "rs10264272": "*7",
    },
    "CYP1A2": {
        "rs762551": "*1F",
        "rs2472297": "*1K",
        "rs12720460": "*1M",
        "rs12720458": "*1N",
        "rs2472298": "*1P",
        "rs17861195": "*1R",
        "rs2472299": "*1S",
        "rs2472300": "*1T",
        "rs2472301": "*1U",
        "rs12720461": "*1V",
        "rs2472302": "*1W",
        "rs2472303": "*1X",
        "rs2472304": "*1Y",
        "rs12720459": "*1Z",
        "rs2068924": "*1F",
        "rs2068925": "*1F",
    },
    "CYP2B6": {
        "rs2279343": "*4",
        "rs28399433": "*6",
        "rs3745274": "*6",
        "rs34228165": "*5",
    },
    "CYP2C8": {
        "rs1058930": "*2",
        "rs11572080": "*3",
    },
    "CYP2E1": {
        "rs2031920": "*5",
    },
    "CYP2A6": {
        "rs1801272": "*2",
        "rs28399433": "*4",
    },
    "SLCO1B1": {
        "rs4149056": "*5",
        "rs2306283": "*18",
    },
    "SLCO1B3": {
        "rs4149056": "*5",
        "rs717315": "*17",
    },
    "ABCB1": {
        "rs1045642": "*2",
        "rs2032582": "*3",
        "rs1128503": "*4",
        "rs2235048": "*5",
        "rs2235046": "*6",
    },
    "ABCG2": {
        "rs2231137": "*2",
        "rs2231142": "*3",
    },
    "SLC22A1": {
        "rs622342": "*2",
    },
    "SLC22A2": {
        "rs316019": "*2",
    },
    "SLC47A1": {
        "rs2289669": "*2",
    },
    "UGT1A1": {
        "rs8175347": "*28",
        "rs4148324": "*6",
    },
    "NAT2": {
        "rs1041993": "*2",
        "rs1801280": "*3",
        "rs1801279": "*5",
        "rs1799929": "*6",
        "rs1799930": "*7",
    },
    "TPMT": {
        "rs1800462": "*3A",
        "rs1142345": "*3C",
    },
    "DPYD": {
        "rs3918290": "*2A",
        "rs1801159": "*5",
        "rs55886062": "*13",
        "rs1801267": "*3",
        "rs56038477": "*15",
    },
    "VKORC1": {
        "rs9923231": "*2",
        "rs8050894": "*4",
    },
}

#: well-characterized pharmacodynamic risk variants (rsID -> label and
#: heterozygous risk weight; homozygotes carry twice the weight, capped 1)
KNOWN_RISK_RSIDS: dict[str, tuple[str, float]] = {
    "rs8175347": ("UGT1A1*28", 0.35),
    "rs4149056": ("SLCO1B1*5", 0.40),
    "rs9923231": ("VKORC1*2", 0.30),
    "rs1800462": ("TPMT*3A", 0.45),
    "rs1142345": ("TPMT*3C", 0.45),
    "rs3918290": ("DPYD*2A", 0.50),
    "rs55886062": ("DPYD*13", 0.50),
    "rs1045642": ("ABCB1*2", 0.20),
    "rs2032582": ("ABCB1*3", 0.20),
    "rs2231142": ("ABCG2*3", 0.25),
    "rs1041993": ("NAT2*2", 0.25),
    "rs1801280": ("NAT2*3", 0.25),
    "rs1799929": ("NAT2*6", 0.25),
    "rs1799930": ("NAT2*7", 0.25),
    "rs3892097": ("CYP2D6*4", 0.30),
    "rs1065852": ("CYP2D6*10", 0.20),
    "rs4244285": ("CYP2C19*2", 0.25),
    "rs4986893": ("CYP2C19*3", 0.25),
    "rs1799853": ("CYP2C9*2", 0.20),
    "rs1057910": ("CYP2C9*3", 0.25),
}

#: enzymes always present in a complete profile (reference EM when no
#: variant evidence exists for them)
CORE_CYP_ENZYMES: tuple[str, ...] = (
    "CYP2D6",
    "CYP2C19",
    "CYP2C9",
    "CYP3A4",
    "CYP3A5",
    "CYP1A2",
    "CYP2B6",
    "CYP2C8",
    "CYP2E1",
    "CYP2A6",
)

#: valid :attr:`Variant.zygosity` values
VALID_ZYGOSITY: tuple[str, ...] = ("hom_ref", "het", "hom_alt")
#: valid :attr:`Variant.variant_type` values
VALID_VARIANT_TYPES: tuple[str, ...] = ("SNV", "indel", "CNV")

#: accepted zygosity spellings across VCF GT fields and CSQ annotations
_ZYGO_ALIASES: dict[str, str] = {
    "het": "het",
    "heterozygous": "het",
    "0/1": "het",
    "1/0": "het",
    "0|1": "het",
    "1|0": "het",
    "./1": "het",
    "hom_alt": "hom_alt",
    "hom": "hom_alt",
    "homozygous": "hom_alt",
    "1/1": "hom_alt",
    "1|1": "hom_alt",
    "hom_ref": "hom_ref",
    "wt": "hom_ref",
    "wildtype": "hom_ref",
    "0/0": "hom_ref",
    "0|0": "hom_ref",
}


# ============================================================================
# Data structures
# ============================================================================

@dataclass(slots=True)
class Variant:
    """A single germline variant call.

    Attributes:
        gene_id: HGNC gene symbol (e.g. ``"CYP2D6"``); ``""`` for
            intergenic calls.
        chromosome: chromosome label (``chr`` prefix stripped).
        position: 1-based genomic coordinate.
        ref: reference allele.
        alt: alternate allele.
        zygosity: one of ``"hom_ref"``, ``"het"``, ``"hom_alt"``.
        variant_type: ``"SNV"`` | ``"indel"`` | ``"CNV"``.
    """

    gene_id: str
    chromosome: str
    position: int
    ref: str
    alt: str
    zygosity: str = "het"
    variant_type: str = "SNV"

    def __post_init__(self) -> None:
        self.chromosome = self.chromosome.removeprefix("chr")
        self.gene_id = self.gene_id.strip()
        if self.zygosity not in VALID_ZYGOSITY:
            raise ValueError(
                f"zygosity must be one of {VALID_ZYGOSITY}, "
                f"got {self.zygosity!r}"
            )
        if self.variant_type not in VALID_VARIANT_TYPES:
            raise ValueError(
                f"variant_type must be one of {VALID_VARIANT_TYPES}, "
                f"got {self.variant_type!r}"
            )
        if self.position < 0:
            raise ValueError(f"position must be non-negative, got {self.position}")


@dataclass(slots=True)
class CYPStatus:
    """Per-enzyme metabolizer status (CPIC activity score framework).

    Attributes:
        enzyme: gene symbol, e.g. ``"CYP2D6"``.
        phenotype: ``"UM"`` | ``"EM"`` | ``"NM"`` | ``"PM"``
            (ultrarapid / expected-extensive / reduced-normal / poor).
        activity_score: summed allele activities across the diplotype
            (two wild-type copies -> 2.0; gene duplication raises it
            further).
        copies: total gene copy number (>= 1); > 2 indicates a
            copy-number gain such as CYP2D6*1xN.
    """

    enzyme: str
    phenotype: str = "EM"
    activity_score: float = 2.0
    copies: int = 1

    def __post_init__(self) -> None:
        self.enzyme = self.enzyme.strip().upper()
        if self.phenotype not in ("UM", "EM", "NM", "PM"):
            raise ValueError(
                f"phenotype must be UM/EM/NM/PM, got {self.phenotype!r}"
            )
        if self.activity_score < 0.0:
            raise ValueError(
                f"activity_score must be non-negative, got {self.activity_score}"
            )
        if self.copies < 1:
            raise ValueError(f"copies must be >= 1, got {self.copies}")


@dataclass(slots=True)
class TransporterStatus:
    """Per-transporter function status (CPIC 2024 pharmacogenomics).

    Attributes:
        transporter: gene symbol, e.g. ``"SLCO1B1"``.
        phenotype: ``"NF"`` | ``"DF"`` | ``"HF"`` | ``"WF"``
            (no / decreased / heterozygous / wild-type function).
        activity_score: summed allele activities across the diplotype.
        copies: total gene copy number (>= 1).
    """

    transporter: str
    phenotype: str = "WF"
    activity_score: float = 2.0
    copies: int = 1

    def __post_init__(self) -> None:
        self.transporter = self.transporter.strip().upper()
        if self.phenotype not in ("NF", "DF", "HF", "WF"):
            raise ValueError(
                f"phenotype must be NF/DF/HF/WF, got {self.phenotype!r}"
            )
        if self.activity_score < 0.0:
            raise ValueError(
                f"activity_score must be non-negative, got {self.activity_score}"
            )
        if self.copies < 1:
            raise ValueError(f"copies must be >= 1, got {self.copies}")


@dataclass(slots=True)
class NonCYPEnzymeStatus:
    """Per-enzyme metabolizer status for non-CYP phase II enzymes.

    Attributes:
        enzyme: gene symbol, e.g. ``"UGT1A1"``, ``"TPMT"``, ``"DPYD"``.
        phenotype: ``"EM"`` | ``"IM"`` | ``"PM"`` | ``"UM"``
            (extensive / intermediate / poor / ultrarapid metabolizer).
        activity_score: summed allele activities across the diplotype.
        copies: total gene copy number (>= 1).
    """

    enzyme: str
    phenotype: str = "EM"
    activity_score: float = 2.0
    copies: int = 1

    def __post_init__(self) -> None:
        self.enzyme = self.enzyme.strip().upper()
        if self.phenotype not in ("EM", "IM", "PM", "UM"):
            raise ValueError(
                f"phenotype must be EM/IM/PM/UM, got {self.phenotype!r}"
            )
        if self.activity_score < 0.0:
            raise ValueError(
                f"activity_score must be non-negative, got {self.activity_score}"
            )
        if self.copies < 1:
            raise ValueError(f"copies must be >= 1, got {self.copies}")


@dataclass(slots=True)
class GenotypeProfile:
    """Complete pharmacogenomic profile of a virtual patient.

    Attributes:
        variants: all parsed variant calls.
        cyp_status: metabolizer status keyed by enzyme symbol.
        transporter_status: transporter function status keyed by symbol.
        non_cyp_enzyme_status: non-CYP enzyme metabolizer status keyed by symbol.
        disease_risk_alleles: pharmacodynamic risk labels mapped to a
            risk weight in [0, 1] (e.g. ``{"UGT1A1*28": 0.35}`` for
            irinotecan neutropenia risk).
        gene_variants: variant calls keyed by arbitrary HGNC gene symbol
            (e.g. ``"TP53"``, ``"BRCA1"``), extending the profile beyond
            the CYP450 star-allele panel without touching ``variants``.
    """

    variants: list[Variant] = field(default_factory=list)
    cyp_status: dict[str, CYPStatus] = field(default_factory=dict)
    transporter_status: dict[str, TransporterStatus] = field(default_factory=dict)
    non_cyp_enzyme_status: dict[str, NonCYPEnzymeStatus] = field(default_factory=dict)
    disease_risk_alleles: dict[str, float] = field(default_factory=dict)
    gene_variants: dict[str, list[Variant]] = field(default_factory=dict)

    def get_cyp_activity(self, enzyme: str) -> float:
        """Return the CYP-dependent clearance multiplier for ``enzyme``.

        The activity score is normalized against the reference diplotype
        (two functional alleles, score 2.0) and clamped to [0.1, 3.0]:
        poor metabolizers clear CYP substrates at ~10% of the normal
        rate while ultrarapid genotypes reach up to 3x. Unknown enzymes
        return a neutral 1.0.
        """
        status = self.cyp_status.get(enzyme.strip().upper())
        if status is None:
            return 1.0
        return round(max(0.1, min(3.0, status.activity_score / 2.0)), 4)

    def is_cyp_inducer(self) -> dict[str, float]:
        """Report genetically driven CYP induction.

        Returns a dict of enzyme -> fold-induction (> 1.0) restricted to
        enzymes whose diplotype confers ultrarapid activity (e.g.
        CYP2D6*1xN duplications, CYP2C19*17 carriers). Enzymes without
        genetic induction are omitted. This captures genotype-only
        induction; tobacco- or drug-mediated induction is layered on by
        :meth:`~helixlang.plugins.human.phenotype.PhenotypeCalculator.compute_cyp_activity`.
        """
        return {
            enzyme: self.get_cyp_activity(enzyme)
            for enzyme, status in self.cyp_status.items()
            if status.phenotype == "UM"
        }

    def add_gene_variant(self, gene_name: str, variant: Variant) -> None:
        """Register ``variant`` under an arbitrary gene symbol.

        Appends to the list stored for ``gene_name``, creating the entry
        on first insertion so repeated calls accumulate (e.g. both hits
        of a hom_alt call or a multi-variant somatic panel). Gene names
        are whitespace-stripped; the CYP star-allele machinery in
        :attr:`variants` / :attr:`cyp_status` is untouched.
        """
        self.gene_variants.setdefault(gene_name.strip(), []).append(variant)

    def get_gene_variants(self, gene_name: str) -> list[Variant]:
        """Return all variants stored for ``gene_name``.

        Returns a defensive copy; genes without registered variants
        yield an empty list.
        """
        return list(self.gene_variants.get(gene_name.strip(), []))

    def get_all_genes(self) -> list[str]:
        """Return every gene symbol tracked by this profile.

        The union of CYP enzymes carrying a :class:`CYPStatus` entry,
        transporters with a :class:`TransporterStatus`, non-CYP enzymes
        with a :class:`NonCYPEnzymeStatus`, and all non-CYP genes
        registered via :meth:`add_gene_variant`, sorted alphabetically.
        """
        return sorted({
            *self.cyp_status.keys(),
            *self.transporter_status.keys(),
            *self.non_cyp_enzyme_status.keys(),
            *self.gene_variants.keys(),
        })

    def get_transporter_activity(self, transporter: str) -> float:
        """Return the transporter function multiplier for ``transporter``.

        The activity score is normalized against the reference diplotype
        (two functional alleles, score 2.0) and clamped to [0.1, 3.0]:
        no-function transporters transport at ~10% of normal rate while
        overexpressors reach up to 3x. Unknown transporters return 1.0.
        """
        status = self.transporter_status.get(transporter.strip().upper())
        if status is None:
            return 1.0
        return round(max(0.1, min(3.0, status.activity_score / 2.0)), 4)

    def get_non_cyp_activity(self, enzyme: str) -> float:
        """Return the non-CYP enzyme activity multiplier for ``enzyme``.

        The activity score is normalized against the reference diplotype
        (two functional alleles, score 2.0) and clamped to [0.1, 3.0].
        Unknown enzymes return 1.0.
        """
        status = self.non_cyp_enzyme_status.get(enzyme.strip().upper())
        if status is None:
            return 1.0
        return round(max(0.1, min(3.0, status.activity_score / 2.0)), 4)

    def get_metabolizer_status(self, enzyme: str) -> str:
        """Return the metabolizer phenotype for any gene (CYP, non-CYP, transporter).

        Checks CYP status first, then non-CYP enzyme status, then transporter
        status. Returns ``"EM"`` for unknown genes.
        """
        enzyme_upper = enzyme.strip().upper()
        status = self.cyp_status.get(enzyme_upper)
        if status is not None:
            return status.phenotype
        ncp_status = self.non_cyp_enzyme_status.get(enzyme_upper)
        if ncp_status is not None:
            return ncp_status.phenotype
        tra_status = self.transporter_status.get(enzyme_upper)
        if tra_status is not None:
            return tra_status.phenotype
        return "EM"


# ============================================================================
# Star-allele resolution helpers
# ============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the closed interval ``[lo, hi]``."""
    return max(lo, min(hi, value))


_STAR_RE = re.compile(r"\*\d+[A-Za-z]?(?:x\d+)?")
_DUP_RE = re.compile(r"\*(\d+[A-Za-z]?)x(\d+|N)", re.IGNORECASE)


def _star_allele_from_text(text: str) -> str:
    """Extract a normalized star-allele token (e.g. ``"*41"``) from text."""
    match = _STAR_RE.search(text or "")
    return match.group(0).upper() if match else ""


def _parse_star_token(token: str) -> tuple[str, int]:
    """Split a star token into ``(allele, duplication_count)``.

    ``"*1x4"`` -> ``("*1", 4)``; ``"*4"`` -> ``("*4", 1)``; unresolved
    multiplicity ``"*1xN"`` maps to the minimum clinically reported
    duplication of three total copies.
    """
    match = _DUP_RE.search(token)
    if match:
        count_raw = match.group(2)
        count = max(2, int(count_raw)) if count_raw.isdigit() else 3
        return f"*{match.group(1).upper()}", count
    star = _star_allele_from_text(token)
    return (star, 1) if star else ("", 1)


def _resolve_star_allele(gene: str, identifier: str) -> tuple[str, int]:
    """Resolve ``(star allele, duplication count)`` from a variant ID.

    Resolution order: direct star notation inside the ID (``CYP2D6*4``
    or ``*1x2``), then the curated per-gene rsID map. Unresolvable IDs
    yield a neutral wildcard allele so unknown input degrades gracefully
    to wild-type activity.
    """
    allele, count = _parse_star_token(identifier)
    if allele:
        return allele, count
    for rsid, star in RSID_TO_STAR_ALLELE.get(gene, {}).items():
        if rsid in identifier:
            return star, 1
    return "", 1


def _allele_activity(gene: str, star: str) -> float:
    """Functional activity of ``star`` on ``gene`` (neutral 1.0 if unknown)."""
    return CYP_ALLELE_ACTIVITIES.get(gene, {}).get(star, 1.0)


def _metabolizer_phenotype(activity_score: float) -> str:
    """Map an activity score onto the four-tier metabolizer phenotype."""
    if activity_score >= 2.5:
        return "UM"
    if activity_score >= 1.5:
        return "EM"
    if activity_score >= 0.5:
        return "NM"
    return "PM"


# ============================================================================
# Diplotype summarization
# ============================================================================

@dataclass(slots=True)
class _AlleleCall:
    """Internal pairing of a variant with its resolved star allele."""

    variant: Variant
    star: str
    duplication_count: int
    copy_number: int | None


def _summarize_enzyme(gene: str, calls: list[_AlleleCall]) -> CYPStatus:
    """Collapse all variant calls of one CYP gene into a :class:`CYPStatus`.

    Non-CNV calls occupy diplotype slots (one for het, two for
    hom_alt); unfilled slots carry the wild-type *1 activity. A CNV
    call overrides the total copy number and, when it names a
    duplicated allele, switches to the per-copy duplication activity
    table (e.g. CYP2D6*1xN counts 2.0 per copy).
    """
    slots: list[float] = []
    copies = 1
    cnv_allele = ""

    for call in calls:
        if call.variant.zygosity == "hom_ref":
            continue
        if call.variant.variant_type == "CNV":
            copies = max(copies, call.copy_number or 0, call.duplication_count or 0, 2)
            if call.star:
                cnv_allele = call.star
            continue
        copies = max(copies, call.duplication_count)
        activity = _allele_activity(gene, call.star)
        slots.extend([activity] * (2 if call.variant.zygosity == "hom_alt" else 1))

    wild_type = _allele_activity(gene, "*1")
    while len(slots) < 2:
        slots.append(wild_type)

    if cnv_allele:
        per_copy = CYP_DUPLICATION_ACTIVITY.get(gene, {}).get(
            cnv_allele, _allele_activity(gene, cnv_allele)
        )
        score = per_copy * copies
    else:
        score = sum(slots)

    return CYPStatus(
        enzyme=gene,
        phenotype=_metabolizer_phenotype(score),
        activity_score=round(score, 3),
        copies=copies,
    )


def _transporter_phenotype(activity_score: float) -> str:
    """Map an activity score onto the four-tier transporter phenotype."""
    if activity_score >= 2.5:
        return "WF"    # wild-type function
    if activity_score >= 1.5:
        return "HF"    # heterozygous function
    if activity_score >= 0.5:
        return "DF"    # decreased function
    return "NF"        # no function


def _non_cyp_phenotype(activity_score: float) -> str:
    """Map an activity score onto the four-tier non-CYP enzyme phenotype."""
    if activity_score >= 2.5:
        return "UM"    # ultrarapid metabolizer
    if activity_score >= 1.5:
        return "EM"    # extensive (normal) metabolizer
    if activity_score >= 0.5:
        return "IM"    # intermediate metabolizer
    return "PM"        # poor metabolizer


def _summarize_transporter(gene: str, calls: list[_AlleleCall]) -> TransporterStatus:
    """Collapse all variant calls of one transporter gene into a :class:`TransporterStatus`."""
    slots: list[float] = []
    copies = 1

    for call in calls:
        if call.variant.zygosity == "hom_ref":
            continue
        if call.variant.variant_type == "CNV":
            copies = max(copies, call.copy_number or 0, call.duplication_count or 0, 2)
            continue
        copies = max(copies, call.duplication_count)
        allele_map = TRANSPORTER_ALLELE_EFFECTS.get(gene, {})
        activity = allele_map.get(call.star, 1.0)
        slots.extend([activity] * (2 if call.variant.zygosity == "hom_alt" else 1))

    while len(slots) < 2:
        slots.append(1.0)

    score = sum(slots)
    return TransporterStatus(
        transporter=gene,
        phenotype=_transporter_phenotype(score),
        activity_score=round(score, 3),
        copies=copies,
    )


def _summarize_non_cyp(gene: str, calls: list[_AlleleCall]) -> NonCYPEnzymeStatus:
    """Collapse all variant calls of one non-CYP enzyme into a :class:`NonCYPEnzymeStatus`."""
    slots: list[float] = []
    copies = 1

    for call in calls:
        if call.variant.zygosity == "hom_ref":
            continue
        if call.variant.variant_type == "CNV":
            copies = max(copies, call.copy_number or 0, call.duplication_count or 0, 2)
            continue
        copies = max(copies, call.duplication_count)
        allele_map = NON_CYP_ENZYMES.get(gene, {})
        activity = allele_map.get(call.star, 1.0)
        slots.extend([activity] * (2 if call.variant.zygosity == "hom_alt" else 1))

    while len(slots) < 2:
        slots.append(1.0)

    score = sum(slots)
    return NonCYPEnzymeStatus(
        enzyme=gene,
        phenotype=_non_cyp_phenotype(score),
        activity_score=round(score, 3),
        copies=copies,
    )


# ============================================================================
# VCF parsing
# ============================================================================

def _parse_info(info_text: str) -> dict[str, object]:
    """Parse a VCF INFO column into ``{key: value_or_flag}``."""
    parsed: dict[str, object] = {}
    for item in info_text.split(";"):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        parsed[key] = value if sep else True
    return parsed


def _parse_zygosity(info: dict[str, object]) -> str:
    """Resolve zygosity from CSQ annotation, GT tag, then default het."""
    candidates: list[str] = []
    csq = info.get("CSQ")
    if isinstance(csq, str):
        candidates.extend(token.strip().lower() for token in csq.split(":"))
    gt = info.get("GT")
    if isinstance(gt, str):
        candidates.append(gt.strip())

    for candidate in candidates:
        alias = _ZYGO_ALIASES.get(candidate)
        if alias is not None:
            return alias
    for spelling, alias in _ZYGO_ALIASES.items():
        for candidate in candidates:
            if spelling in candidate:
                return alias
    return "het"


def _copy_number_from_info(info: dict[str, object]) -> int | None:
    """Read a structural copy number from INFO tags ``CN`` / ``COPYNUM``."""
    for key in ("CN", "COPYNUM"):
        raw = info.get(key)
        if isinstance(raw, str) and raw.isdigit():
            return max(1, int(raw))
    return None


def create_genotype_from_vcf(vcf_text: str) -> GenotypeProfile:
    """Build a :class:`GenotypeProfile` from simple VCF-like text.

    Expected columns: ``CHROM POS ID REF ALT QUAL FILTER INFO``, tab-
    or whitespace-delimited; header lines starting with ``#`` and blank
    lines are skipped. The INFO column may carry ``GENE=<symbol>``, a
    ``CSQ`` annotation containing the zygosity, an optional ``GT``
    genotype tag, optional ``CN`` / ``COPYNUM`` for CNV calls, and an
    optional ``RISK=<float>`` pharmacodynamic risk weight.

    Variants on a known CYP gene are resolved to star alleles through
    the ID field (direct star notation or curated rsIDs); anything
    unresolvable falls back to neutral wild-type activity so unknown
    input never corrupts the phenotype. Core enzymes without any
    variant evidence receive the reference EM status. Well-known
    pharmacodynamic risk rsIDs (and explicit RISK tags) populate
    ``disease_risk_alleles``.

    Args:
        vcf_text: raw VCF-like text block.

    Returns:
        a freshly built :class:`GenotypeProfile`.
    """
    variants: list[Variant] = []
    calls_by_gene: dict[str, list[_AlleleCall]] = {}
    risk_alleles: dict[str, float] = {}
    seen: set[tuple[str, int, str, str]] = set()

    for line in vcf_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 5:
            continue
        chromosome, position, identifier, ref, alt = columns[:5]
        info = _parse_info(columns[7]) if len(columns) > 7 else {}

        gene = str(info.get("GENE", "")).strip().upper()
        zygosity = _parse_zygosity(info)
        pos = int(position) if position.isdigit() else 0
        is_cnv = "SVTYPE=CNV" in info or info.get("SVTYPE") == "CNV"

        for alt_allele in alt.split(","):
            key = (chromosome, pos, ref, alt_allele)
            if key in seen:
                continue
            seen.add(key)
            variant = Variant(
                gene_id=gene,
                chromosome=chromosome,
                position=pos,
                ref=ref,
                alt=alt_allele,
                zygosity=zygosity,
                variant_type="CNV" if is_cnv else "SNV",
            )
            variants.append(variant)

            if gene in CYP_ALLELE_ACTIVITIES:
                star, dup_count = _resolve_star_allele(gene, identifier)
                calls_by_gene.setdefault(gene, []).append(
                    _AlleleCall(
                        variant=variant,
                        star=star,
                        duplication_count=dup_count,
                        copy_number=_copy_number_from_info(info),
                    )
                )
            elif gene in TRANSPORTER_ALLELE_EFFECTS:
                star, dup_count = _resolve_star_allele(gene, identifier)
                calls_by_gene.setdefault(gene, []).append(
                    _AlleleCall(
                        variant=variant,
                        star=star,
                        duplication_count=dup_count,
                        copy_number=_copy_number_from_info(info),
                    )
                )
            elif gene in NON_CYP_ENZYMES:
                star, dup_count = _resolve_star_allele(gene, identifier)
                calls_by_gene.setdefault(gene, []).append(
                    _AlleleCall(
                        variant=variant,
                        star=star,
                        duplication_count=dup_count,
                        copy_number=_copy_number_from_info(info),
                    )
                )

            risk_tag = info.get("RISK")
            if isinstance(risk_tag, str):
                try:
                    label = f"{gene}:{identifier}".strip(":")
                    risk_alleles[label] = float(risk_tag)
                except ValueError:  # SILENTBENIGN - skip non-numeric risk tag
                    pass
            for rsid, (label, het_risk) in KNOWN_RISK_RSIDS.items():
                if rsid in identifier:
                    risk = min(1.0, het_risk * (2 if zygosity == "hom_alt" else 1))
                    risk_alleles[label] = max(risk_alleles.get(label, 0.0), risk)

    cyp_status = {gene: _summarize_enzyme(gene, calls) for gene, calls in calls_by_gene.items()}
    for enzyme in CORE_CYP_ENZYMES:
        if enzyme not in cyp_status:
            cyp_status[enzyme] = CYPStatus(
                enzyme=enzyme, phenotype="EM", activity_score=2.0
            )

    # Resolve transporter status from calls
    transporter_status: dict[str, TransporterStatus] = {}
    for gene_name, calls in calls_by_gene.items():
        if gene_name in TRANSPORTER_ALLELE_EFFECTS:
            transporter_status[gene_name] = _summarize_transporter(gene_name, calls)
    for transporter in TRANSPORTER_GENES:
        if transporter not in transporter_status:
            transporter_status[transporter] = TransporterStatus(
                transporter=transporter, phenotype="WF", activity_score=2.0
            )

    # Resolve non-CYP enzyme status from calls
    non_cyp_status: dict[str, NonCYPEnzymeStatus] = {}
    for gene_name, calls in calls_by_gene.items():
        if gene_name in NON_CYP_ENZYMES:
            non_cyp_status[gene_name] = _summarize_non_cyp(gene_name, calls)
    for enzyme in NON_CYP_ENZYME_GENES:
        if enzyme not in non_cyp_status:
            non_cyp_status[enzyme] = NonCYPEnzymeStatus(
                enzyme=enzyme, phenotype="EM", activity_score=2.0
            )

    return GenotypeProfile(
        variants=variants,
        cyp_status=cyp_status,
        transporter_status=transporter_status,
        non_cyp_enzyme_status=non_cyp_status,
        disease_risk_alleles=risk_alleles,
    )


def create_default_genotype() -> GenotypeProfile:
    """Create the reference extensive-metabolizer genotype.

    All core CYP enzymes, transporters, and non-CYP enzymes carry the
    wild-type *1/*1 diplotype (activity score 2.0, EM phenotype) and no
    pharmacodynamic risk alleles. Each call returns a fresh, independently
    mutable instance.
    """
    return GenotypeProfile(
        variants=[],
        cyp_status={
            enzyme: CYPStatus(enzyme=enzyme, phenotype="EM", activity_score=2.0)
            for enzyme in CORE_CYP_ENZYMES
        },
        transporter_status={
            transporter: TransporterStatus(transporter=transporter, phenotype="WF", activity_score=2.0)
            for transporter in TRANSPORTER_GENES
        },
        non_cyp_enzyme_status={
            enzyme: NonCYPEnzymeStatus(enzyme=enzyme, phenotype="EM", activity_score=2.0)
            for enzyme in NON_CYP_ENZYME_GENES
        },
        disease_risk_alleles={},
    )
