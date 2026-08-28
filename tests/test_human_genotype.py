"""Tests for helixlang.plugins.human.genotype (doc/28 pharmacogenomics layer).

Covers star-allele diplotypes (CYP2D6*4 poor metabolizer, *1xN
ultrarapid), the CPIC activity-score mapping, and the simple VCF
parser that feeds :class:`~helixlang.plugins.human.phenotype.PhenotypeCalculator`.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.human.genotype import (
    CORE_CYP_ENZYMES,
    CYPStatus,
    GenotypeProfile,
    Variant,
    create_default_genotype,
    create_genotype_from_vcf,
)


def test_default_genotype():
    """create_default_genotype() carries every core CYP enzyme."""
    geno = create_default_genotype()
    assert isinstance(geno, GenotypeProfile)
    assert set(geno.cyp_status) == set(CORE_CYP_ENZYMES)
    assert geno.variants == []
    assert geno.disease_risk_alleles == {}


def test_cyp_status_em():
    """The reference genotype reports EM status across the panel."""
    geno = create_default_genotype()
    for enzyme in CORE_CYP_ENZYMES:
        assert geno.get_metabolizer_status(enzyme) == "EM"


def test_cyp_activity_normal():
    """Wild-type *1/*1 diplotype normalizes to unit clearance (2.0 / 2.0)."""
    geno = create_default_genotype()
    assert geno.get_cyp_activity("CYP2D6") == pytest.approx(1.0)


def test_variant_creation():
    """Variant stores its fields and normalizes chr-prefixed contigs."""
    var = Variant(
        gene_id=" CYP2D6 ",
        chromosome="chr22",
        position=42526693,
        ref="G",
        alt="A",
        zygosity="hom_alt",
        variant_type="indel",
    )
    assert var.gene_id == "CYP2D6"
    assert var.chromosome == "22"
    assert var.position == 42526693
    assert var.ref == "G"
    assert var.alt == "A"
    assert var.zygosity == "hom_alt"
    assert var.variant_type == "indel"
    with pytest.raises(ValueError):
        Variant("GENE", "7", 1, "A", "T", zygosity="half")
    with pytest.raises(ValueError):
        Variant("GENE", "7", -5, "A", "T")


def test_genotype_profile_empty():
    """An empty GenotypeProfile has no variants and neutral activity."""
    geno = GenotypeProfile()
    assert geno.variants == []
    assert geno.cyp_status == {}
    assert geno.disease_risk_alleles == {}
    assert geno.get_cyp_activity("CYP2D6") == pytest.approx(1.0)
    assert geno.get_metabolizer_status("CYP3A4") == "EM"


def test_cyp2d6_pm():
    """CYP2D6*4/*4 null diplotype yields PM status at floor clearance."""
    vcf = "\n".join(
        [
            "##fileformat=VCFv4.2",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            "7\t86478423\tCYP2D6*4\tC\tT\t.\tPASS\tGENE=CYP2D6;GT=1/1",
        ]
    )
    geno = create_genotype_from_vcf(vcf)
    assert geno.get_metabolizer_status("CYP2D6") == "PM"
    assert geno.get_cyp_activity("CYP2D6") == pytest.approx(0.1)
    assert "CYP2D6" not in geno.is_cyp_inducer()


def test_cyp2d6_um():
    """CYP2D6*1xN copy-number gain yields UM status at ceiling clearance."""
    vcf = "\n".join(
        [
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            "22\t42524000\tCYP2D6*1xN\tC\t<DEL>\t.\tPASS"
            "\tSVTYPE=CNV;GENE=CYP2D6;GT=1/1;CN=3",
        ]
    )
    geno = create_genotype_from_vcf(vcf)
    status = geno.cyp_status["CYP2D6"]
    assert status.phenotype == "UM"
    assert status.activity_score > 2.5
    assert status.copies >= 3
    assert geno.get_cyp_activity("CYP2D6") == pytest.approx(3.0)
    assert geno.is_cyp_inducer() == {"CYP2D6": 3.0}


def test_cyp2c19_poor():
    """CYP2C19*2/*2 loss-of-function diplotype maps to PM."""
    vcf = (
        "10\t94781859\tCYP2C19*2\tG\tA\t.\tPASS\tGENE=CYP2C19;GT=1/1\n"
    )
    geno = create_genotype_from_vcf(vcf)
    assert geno.get_metabolizer_status("CYP2C19") == "PM"
    assert geno.get_cyp_activity("CYP2C19") == pytest.approx(0.1)
    # untouched enzymes still carry reference EM status
    assert geno.get_metabolizer_status("CYP2D6") == "EM"


def test_parse_vcf_basic():
    """create_genotype_from_vcf parses columns, INFO tags, and GT zygosity."""
    vcf = "\n".join(
        [
            "##fileformat=VCFv4.2",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            "7\t86478423\trs3892097\tC\tT\t.\tPASS\tGENE=CYP2D6;GT=0|1",
            "",
        ]
    )
    geno = create_genotype_from_vcf(vcf)
    assert len(geno.variants) == 1
    var = geno.variants[0]
    assert var.gene_id == "CYP2D6"
    assert var.position == 86478423
    assert var.zygosity == "het"
    # het *4 + wild-type slot -> reduced-normal metabolizer
    assert geno.get_metabolizer_status("CYP2D6") == "NM"
    assert geno.get_cyp_activity("CYP2D6") == pytest.approx(0.5)
    assert set(geno.cyp_status) == set(CORE_CYP_ENZYMES)


def test_parse_vcf_empty():
    """Empty or header-only VCF input degrades to the default EM profile."""
    geno = create_genotype_from_vcf("")
    assert geno.variants == []
    assert geno.disease_risk_alleles == {}
    assert set(geno.cyp_status) == set(CORE_CYP_ENZYMES)
    for enzyme in CORE_CYP_ENZYMES:
        assert geno.get_metabolizer_status(enzyme) == "EM"

    header_only = create_genotype_from_vcf("#CHROM\tPOS\tID\tREF\tALT\n")
    assert header_only.variants == []


def test_cyp_activity_clamp():
    """Activity scores clamp into [0.1, 3.0] regardless of genotype."""
    geno = GenotypeProfile(
        cyp_status={
            "CYP2D6": CYPStatus("CYP2D6", phenotype="UM", activity_score=100.0),
            "CYP2C19": CYPStatus("CYP2C19", phenotype="PM", activity_score=0.0),
        }
    )
    assert geno.get_cyp_activity("CYP2D6") == pytest.approx(3.0)
    assert geno.get_cyp_activity("CYP2C19") == pytest.approx(0.1)
