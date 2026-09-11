"""Branch-completion tests for helixlang.plugins.human.genotype."""
from __future__ import annotations

import pytest

from helixlang.plugins.human.genotype import (
    CYPStatus,
    GenotypeProfile,
    NonCYPEnzymeStatus,
    TransporterStatus,
    Variant,
    _AlleleCall,
    _copy_number_from_info,
    _parse_info,
    _parse_zygosity,
    _resolve_star_allele,
    _summarize_enzyme,
    _summarize_non_cyp,
    _summarize_transporter,
    create_genotype_from_vcf,
)


def _variant(gene="CYP2D6", zygosity="het", vtype="SNV", alt="A"):
    return Variant(
        gene_id=gene, chromosome="1", position=100, ref="C", alt=alt,
        zygosity=zygosity, variant_type=vtype,
    )


class TestStatusValidations:
    def test_variant_bad_type(self):
        with pytest.raises(ValueError, match="variant_type"):
            _variant(vtype="SVA")

    def test_cyp_bad_phenotype(self):
        with pytest.raises(ValueError, match="phenotype"):
            CYPStatus(enzyme="CYP2D6", phenotype="XX")

    def test_cyp_bad_activity(self):
        with pytest.raises(ValueError, match="activity_score"):
            CYPStatus(enzyme="CYP2D6", activity_score=-1.0)

    def test_cyp_bad_copies(self):
        with pytest.raises(ValueError, match="copies"):
            CYPStatus(enzyme="CYP2D6", copies=0)

    def test_transporter_bad_phenotype(self):
        with pytest.raises(ValueError, match="phenotype"):
            TransporterStatus(transporter="SLCO1B1", phenotype="ZZ")

    def test_transporter_bad_activity(self):
        with pytest.raises(ValueError, match="activity_score"):
            TransporterStatus(transporter="SLCO1B1", activity_score=-0.5)

    def test_transporter_bad_copies(self):
        with pytest.raises(ValueError, match="copies"):
            TransporterStatus(transporter="SLCO1B1", copies=0)

    def test_non_cyp_bad_phenotype(self):
        with pytest.raises(ValueError, match="phenotype"):
            NonCYPEnzymeStatus(enzyme="UGT1A1", phenotype="Q1")

    def test_non_cyp_bad_activity(self):
        with pytest.raises(ValueError, match="activity_score"):
            NonCYPEnzymeStatus(enzyme="UGT1A1", activity_score=-1.0)

    def test_non_cyp_bad_copies(self):
        with pytest.raises(ValueError, match="copies"):
            NonCYPEnzymeStatus(enzyme="UGT1A1", copies=0)


class TestProfileAccessors:
    def test_gene_variant_registry(self):
        geno = GenotypeProfile()
        geno.add_gene_variant(" XYZ ", _variant())
        assert len(geno.get_gene_variants("  XYZ ")) == 1
        assert "XYZ" in geno.get_all_genes()

    def test_unknown_transporter_activity(self):
        assert GenotypeProfile().get_transporter_activity("NOPE") == 1.0

    def test_unknown_non_cyp_activity(self):
        assert GenotypeProfile().get_non_cyp_activity("NOPE") == 1.0

    def test_known_non_cyp_status(self):
        geno = GenotypeProfile()
        geno.non_cyp_enzyme_status["UGT1A1"] = NonCYPEnzymeStatus(
            enzyme="UGT1A1", phenotype="IM")
        assert geno.get_metabolizer_status("UGT1A1") == "IM"

    def test_known_transporter_status(self):
        geno = GenotypeProfile()
        geno.transporter_status["SLCO1B1"] = TransporterStatus(
            transporter="SLCO1B1", phenotype="NF")
        assert geno.get_metabolizer_status("SLCO1B1") == "NF"

    def test_known_transporter_activity(self):
        geno = GenotypeProfile()
        geno.transporter_status["SLCO1B1"] = TransporterStatus(
            transporter="SLCO1B1", activity_score=1.0)
        assert geno.get_transporter_activity("SLCO1B1") == pytest.approx(0.5)

    def test_known_non_cyp_activity(self):
        geno = GenotypeProfile()
        geno.non_cyp_enzyme_status["UGT1A1"] = NonCYPEnzymeStatus(
            enzyme="UGT1A1", activity_score=8.0)
        assert geno.get_non_cyp_activity("UGT1A1") == pytest.approx(3.0)

    def test_clamp_direct(self):
        from helixlang.plugins.human.genotype import _clamp
        assert _clamp(5.0, 0.0, 3.0) == 3.0


class TestStarResolution:
    def test_direct_star(self):
        assert _resolve_star_allele("CYP2D6", "CYP2D6*41") == ("*41", 1)

    def test_dup_notation(self):
        assert _resolve_star_allele("CYP2D6", "*1x4") == ("*1", 4)

    def test_dup_unknown_multiplicity(self):
        assert _resolve_star_allele("CYP2D6", "*2xN") == ("*2", 3)

    def test_rsid_hit(self):
        assert _resolve_star_allele("SLCO1B1", "rs4149056") == ("*5", 1)

    def test_unresolved(self):
        assert _resolve_star_allele("CYP2D6", "rs987654321") == ("", 1)


class TestSummarizers:
    def test_enzyme_hom_ref_skip_and_wildtype(self):
        calls = [_AlleleCall(_variant(zygosity="hom_ref"), "*4", 1, None)]
        status = _summarize_enzyme("CYP2D6", calls)
        assert status.phenotype == "EM" and status.activity_score == pytest.approx(2.0)

    def test_enzyme_cnv_without_star(self):
        calls = [_AlleleCall(
            _variant(vtype="CNV", alt="<DEL>"), "", 2, 3)]
        status = _summarize_enzyme("CYP2D6", calls)
        assert status.copies >= 2

    def test_transporter_tiers(self):
        low = _summarize_transporter("SLCO1B1", [
            _AlleleCall(_variant(), "*5", 1, None),
            _AlleleCall(_variant(), "*5", 1, None),
        ])
        assert low.phenotype == "NF"
        df = _summarize_transporter("SLCO1B1", [
            _AlleleCall(_variant(), "*18", 1, None),
            _AlleleCall(_variant(), "*18", 1, None),
        ])
        assert df.phenotype == "DF"
        hf = _summarize_transporter("SLCO1B1", [
            _AlleleCall(_variant(), "*1", 1, None),
            _AlleleCall(_variant(), "*34", 1, None),
        ])
        assert hf.phenotype == "HF"
        wf = _summarize_transporter("SLCO1B1", [
            _AlleleCall(_variant(zygosity="hom_alt"), "*33", 1, None),
            _AlleleCall(_variant(zygosity="hom_alt"), "*33", 1, None),
        ])
        assert wf.phenotype == "WF"

    def test_transporter_hom_ref_and_cnv(self):
        calls = [
            _AlleleCall(_variant(zygosity="hom_ref"), "*5", 1, None),
            _AlleleCall(_variant(vtype="CNV", alt="<DEL>"), "*20", 3, None),
        ]
        status = _summarize_transporter("SLCO1B1", calls)
        assert status.copies >= 3 and status.activity_score == pytest.approx(2.0)

    def test_non_cyp_tiers(self):
        pm = _summarize_non_cyp("UGT1A1", [
            _AlleleCall(_variant(), "*27", 1, None),
            _AlleleCall(_variant(), "*27", 1, None),
        ])
        assert pm.phenotype == "PM"
        im = _summarize_non_cyp("UGT1A1", [
            _AlleleCall(_variant(), "*28", 1, None),
            _AlleleCall(_variant(), "*28", 1, None),
        ])
        assert im.phenotype == "IM"
        em = _summarize_non_cyp("UGT1A1", [
            _AlleleCall(_variant(), "*1", 1, None),
            _AlleleCall(_variant(), "*7", 1, None),
        ])
        assert em.phenotype == "EM"
        um = _summarize_non_cyp("UGT1A1", [
            _AlleleCall(_variant(zygosity="hom_alt"), "*1", 3, None),
            _AlleleCall(_variant(zygosity="hom_alt"), "*1", 3, None),
        ])
        assert um.phenotype == "UM"

    def test_non_cyp_cnv_and_hom_ref(self):
        calls = [
            _AlleleCall(_variant(zygosity="hom_ref"), "*1", 1, None),
            _AlleleCall(_variant(vtype="CNV", alt="<DEL>"), "*1", 4, None),
        ]
        status = _summarize_non_cyp("UGT1A1", calls)
        assert status.copies >= 4


class TestVcfInfoParsing:
    def test_parse_info_empty_items(self):
        assert _parse_info("A=1;;B=2;") == {"A": "1", "B": "2"}

    def test_parse_info_flag(self):
        assert _parse_info("X") == {"X": True}

    def test_parse_zygosity_csq(self):
        assert _parse_zygosity({"CSQ": "heterozygous"}) == "het"

    def test_parse_zygosity_gt_substring(self):
        assert _parse_zygosity({"GT": "X/0/1Y"}) == "het"

    def test_parse_zygosity_default(self):
        assert _parse_zygosity({}) == "het"

    def test_copy_number_from_info(self):
        assert _copy_number_from_info({"COPYNUM": "5"}) == 5
        assert _copy_number_from_info({"CN": "0"}) == 1
        assert _copy_number_from_info({"CN": "abc"}) is None


class TestVcfPipeline:
    def test_short_lines_skipped(self):
        geno = create_genotype_from_vcf("1 2 3 4\n")
        assert geno.variants == []

    def test_duplicate_variant_skipped(self):
        vcf = (
            "1\t100\trs1\tC\tT\t.\tPASS\tGENE=CYP2D6\n"
            "1\t100\trs1\tC\tT\t.\tPASS\tGENE=CYP2D6\n"
        )
        geno = create_genotype_from_vcf(vcf)
        assert len(geno.variants) == 1

    def test_transporter_variant(self):
        geno = create_genotype_from_vcf(
            "12\t21222820\trs4149056\tT\tC\t.\tPASS\tGENE=SLCO1B1;GT=1/1\n")
        assert geno.transporter_status["SLCO1B1"].phenotype == "NF"

    def test_non_cyp_variant(self):
        geno = create_genotype_from_vcf(
            "2\t234668879\trs8175347\tA\tC\t.\tPASS\tGENE=UGT1A1;GT=1/1\n")
        assert geno.non_cyp_enzyme_status["UGT1A1"].phenotype == "IM"

    def test_unknown_gene_variant(self):
        geno = create_genotype_from_vcf(
            "1\t100\trs1\tC\tT\t.\tPASS\tGENE=LRG1\n")
        assert len(geno.variants) == 1
        assert "LRG1" not in geno.cyp_status
        assert "LRG1" not in geno.transporter_status
        assert "LRG1" not in geno.non_cyp_enzyme_status

    def test_non_numeric_position(self):
        geno = create_genotype_from_vcf(
            "12\txxx\trs4149056\tT\tC\t.\tPASS\tGENE=SLCO1B1\n")
        assert geno.variants[0].position == 0

    def test_risk_tag_numeric(self):
        geno = create_genotype_from_vcf(
            "1\t100\trs999\tC\tT\t.\tPASS\tGENE=TPMT;RISK=0.7\n")
        assert geno.disease_risk_alleles["TPMT:rs999"] == pytest.approx(0.7)

    def test_risk_tag_non_numeric(self):
        geno = create_genotype_from_vcf(
            "1\t100\trs999\tC\tT\t.\tPASS\tGENE=TPMT;RISK=silent\n")
        assert geno.disease_risk_alleles == {}
