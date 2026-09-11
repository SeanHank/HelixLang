"""Targeted tests to close all remaining coverage gaps in plugins/annotation/.

Covers:
  - annotation/__init__.py: GeneAnnotation, to_dict, _check, _make_backend, _load
  - annotation/kegg_mapping.py: KOReactionDB, build_ko_db
  - annotation/ec_mapping.py: ECReactionDB, build_ec_db
  - annotation/vector.py: grammar registration, _parse_vector
  - annotation/sequences.py: reverse_complement, translate, FASTA/GFF3 extraction
"""
from __future__ import annotations

import pytest

# ============================================================================
# annotation/__init__.py — GeneAnnotation, _check, _make_backend, _load
# ============================================================================

class TestGeneAnnotation:
    def test_init_defaults(self):
        from helixlang.plugins.annotation import GeneAnnotation
        ga = GeneAnnotation(gene_id="g1")
        assert ga.gene_id == "g1"
        assert ga.protein_seq == ""
        assert ga.ec_numbers == []
        assert ga.kegg_ko == []
        assert ga.go_terms == []
        assert ga.subsystem == ""
        assert ga.is_transporter is False
        assert ga.transport_substrate is None
        assert ga.is_transcription_factor is False
        assert ga.tf_family is None
        assert ga.confidence == 0.0

    def test_init_all_args(self):
        from helixlang.plugins.annotation import GeneAnnotation
        ga = GeneAnnotation(
            gene_id="g2",
            protein_seq="MKTLYF",
            ec_numbers=["1.2.3.4"],
            kegg_ko=["K00001"],
            go_terms=["GO:0003674"],
            subsystem="glycolysis",
            is_transporter=True,
            transport_substrate="glucose",
            is_transcription_factor=True,
            tf_family="HTH_LacI",
            confidence=0.95,
        )
        assert ga.gene_id == "g2"
        assert ga.protein_seq == "MKTLYF"
        assert ga.ec_numbers == ["1.2.3.4"]
        assert ga.kegg_ko == ["K00001"]
        assert ga.go_terms == ["GO:0003674"]
        assert ga.subsystem == "glycolysis"
        assert ga.is_transporter is True
        assert ga.transport_substrate == "glucose"
        assert ga.is_transcription_factor is True
        assert ga.tf_family == "HTH_LacI"
        assert ga.confidence == 0.95

    def test_to_dict(self):
        from helixlang.plugins.annotation import GeneAnnotation
        ga = GeneAnnotation(
            gene_id="g3",
            ec_numbers=["2.7.1.1"],
            kegg_ko=["K00844"],
            go_terms=["GO:0004331"],
            subsystem="glycolysis",
            is_transporter=False,
            is_transcription_factor=True,
            tf_family="HTH_GntR",
            confidence=0.8,
        )
        d = ga.to_dict()
        assert d["gene_id"] == "g3"
        assert d["ec_numbers"] == ["2.7.1.1"]
        assert d["kegg_ko"] == ["K00844"]
        assert d["go_terms"] == ["GO:0004331"]
        assert d["subsystem"] == "glycolysis"
        assert d["is_transporter"] is False
        assert d["is_transcription_factor"] is True
        assert d["tf_family"] == "HTH_GntR"
        assert d["confidence"] == 0.8


class TestAnnotationCheck:
    def test_check_returns_true_for_existing_package(self):
        from helixlang.plugins.annotation import _check
        assert _check("os") is True

    def test_check_returns_false_for_missing_package(self):
        from helixlang.plugins.annotation import _check
        assert _check("nonexistent_package_xyz_123") is False


class TestAnnotationMakeBackend:
    def test_make_backend_returns_gene_annotation_class(self):
        from helixlang.plugins.annotation import GeneAnnotation, _make_backend
        assert _make_backend() is GeneAnnotation

    def test_make_backend_with_config(self):
        from helixlang.plugins.annotation import GeneAnnotation, _make_backend
        assert _make_backend({"some": "config"}) is GeneAnnotation


class TestAnnotationLoad:
    def test_load_returns_make_backend(self):
        from helixlang.plugins.annotation import _load
        backend = _load()
        assert callable(backend)

    def test_load_missing_numpy_raises(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("no numpy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from helixlang.core.errors import PluginDependencyError
        from helixlang.plugins.annotation import _load
        with pytest.raises(PluginDependencyError):
            _load()


# ============================================================================
# annotation/kegg_mapping.py
# ============================================================================

class TestKOMapping:
    def test_ko_reaction_db_init(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        assert db.size == 0

    def test_load_from_dict(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        db.load_from_dict({
            "K00844": {"reactions": ["HEX1"], "pathways": ["map00010"]},
            "K01810": {"reactions": ["PGI"], "pathways": []},
        })
        assert db.size == 2

    def test_lookup_found(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        db.load_from_dict({
            "K00844": {"reactions": ["HEX1"], "pathways": ["map00010"]},
        })
        m = db.lookup("K00844")
        assert m is not None
        assert m.ko_id == "K00844"
        assert m.reaction_ids == ["HEX1"]
        assert m.pathway_ids == ["map00010"]
        assert m.confidence == 1.0

    def test_lookup_not_found(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        assert db.lookup("NOPE") is None

    def test_has_ko_true(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        db.load_from_dict({"K00844": {"reactions": []}})
        assert db.has_ko("K00844") is True

    def test_has_ko_false(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        assert db.has_ko("NOPE") is False

    def test_build_ko_db(self):
        from helixlang.plugins.annotation.kegg_mapping import build_ko_db
        db = build_ko_db()
        assert db.size > 0
        assert db.has_ko("K00844")
        m = db.lookup("K00844")
        assert m.reaction_ids == ["HEX1"]

    def test_load_from_dict_empty_key(self):
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB
        db = KOReactionDB()
        db.load_from_dict({"K01810": {}})
        m = db.lookup("K01810")
        assert m.reaction_ids == []
        assert m.pathway_ids == []


# ============================================================================
# annotation/ec_mapping.py
# ============================================================================

class TestECReactionDB:
    def test_init(self):
        from helixlang.plugins.annotation.ec_mapping import ECReactionDB
        db = ECReactionDB()
        assert db.size == 0

    def test_load_from_dict(self):
        from helixlang.plugins.annotation.ec_mapping import ECReactionDB
        db = ECReactionDB()
        db.load_from_dict({
            "2.7.1.1": ["HEX1"],
            "5.3.1.9": ["PGI"],
        })
        assert db.size == 2

    def test_lookup_found(self):
        from helixlang.plugins.annotation.ec_mapping import ECReactionDB
        db = ECReactionDB()
        db.load_from_dict({"2.7.1.1": ["HEX1"]})
        m = db.lookup("2.7.1.1")
        assert m is not None
        assert m.ec_number == "2.7.1.1"
        assert m.reaction_ids == ["HEX1"]
        assert m.confidence == 1.0

    def test_lookup_not_found(self):
        from helixlang.plugins.annotation.ec_mapping import ECReactionDB
        db = ECReactionDB()
        assert db.lookup("99.99.99") is None

    def test_has_ec_true(self):
        from helixlang.plugins.annotation.ec_mapping import ECReactionDB
        db = ECReactionDB()
        db.load_from_dict({"2.7.1.1": ["HEX1"]})
        assert db.has_ec("2.7.1.1") is True

    def test_has_ec_false(self):
        from helixlang.plugins.annotation.ec_mapping import ECReactionDB
        db = ECReactionDB()
        assert db.has_ec("99.99.99") is False

    def test_build_ec_db(self):
        from helixlang.plugins.annotation.ec_mapping import build_ec_db
        db = build_ec_db()
        assert db.size > 0
        assert db.has_ec("2.7.1.1")
        m = db.lookup("2.7.1.1")
        assert m.reaction_ids == ["HEX1"]


# ============================================================================
# annotation/vector.py — grammar registration
# ============================================================================

class TestVectorGrammar:
    def test_import_registers_grammar(self):
        from helixlang.api.registry import grammar_registry
        from helixlang.plugins.annotation import vector
        assert vector.VECTOR_GRAMMAR.keyword == "vector"
        grammar_registry.register(vector.VECTOR_GRAMMAR)

    def test_parse_vector_missing_gene_raises(self):
        from unittest.mock import MagicMock

        from helixlang.api.ast import Program
        from helixlang.core.errors import ParseError
        from helixlang.plugins.annotation.vector import _parse_vector

        parser = MagicMock()
        token = MagicMock()
        token.line = 1
        parser._advance.return_value = token
        parser._collect_fields_until_block_end.return_value = {"plasmid": "pUC19"}

        prog = Program()
        with pytest.raises(ParseError, match="gene="):
            _parse_vector(parser, prog)

    def test_parse_vector_with_gene(self):
        from unittest.mock import MagicMock

        from helixlang.api.ast import Program
        from helixlang.plugins.annotation.vector import _parse_vector

        parser = MagicMock()
        token = MagicMock()
        token.line = 1
        parser._advance.return_value = token
        parser._collect_fields_until_block_end.return_value = {"gene": "TP53"}

        prog = Program()
        _parse_vector(parser, prog)
        parser._append_sim_list.assert_called_once()

    def test_register_function(self):
        from helixlang.plugins.annotation.vector import register
        register()

    def test_decompile(self):
        from helixlang.api.ast import Program
        from helixlang.plugins.annotation.vector import VECTOR_GRAMMAR

        prog = Program()
        prog.extensions["vectors"] = [{"gene": "TP53", "plasmid": "pUC19"}]
        result = VECTOR_GRAMMAR.decompile(prog)
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("vector" in line for line in result)


# ============================================================================
# annotation/sequences.py
# ============================================================================

class TestReverseComplement:
    def test_basic(self):
        from helixlang.plugins.annotation.sequences import reverse_complement
        assert reverse_complement("ATCG") == "CGAT"

    def test_all_bases(self):
        from helixlang.plugins.annotation.sequences import reverse_complement
        assert reverse_complement("ACGT") == "ACGT"

    def test_lowercase(self):
        from helixlang.plugins.annotation.sequences import reverse_complement
        assert reverse_complement("atcg") == "CGAT"

    def test_unknown_base(self):
        from helixlang.plugins.annotation.sequences import reverse_complement
        assert reverse_complement("ANG") == "CNT"

    def test_empty(self):
        from helixlang.plugins.annotation.sequences import reverse_complement
        assert reverse_complement("") == ""


class TestTranslate:
    def test_basic_codons(self):
        from helixlang.plugins.annotation.sequences import translate
        # ATG = M, TTT = F, GAA = E
        result = translate("ATGTTTGAA")
        assert result.startswith("M")
        assert "F" in result
        assert "E" in result

    def test_stop_codon(self):
        from helixlang.plugins.annotation.sequences import translate
        # TAA is a stop codon
        result = translate("ATGTAA")
        assert result == "M"

    def test_incomplete_codon(self):
        from helixlang.plugins.annotation.sequences import translate
        result = translate("ATGTT")
        assert len(result) >= 1

    def test_unknown_codon(self):
        from helixlang.plugins.annotation.sequences import translate
        result = translate("NNN")
        assert result == "X"

    def test_empty(self):
        from helixlang.plugins.annotation.sequences import translate
        assert translate("") == ""


class TestExtractProteinsFromFasta:
    def test_valid_fasta(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_fasta
        fa = tmp_path / "proteins.fa"
        fa.write_text(">gene1\nMKTLYF\n>gene2\nACDEF\n")
        proteins = extract_proteins_from_fasta(str(fa))
        assert proteins["gene1"] == "MKTLYF"
        assert proteins["gene2"] == "ACDEF"

    def test_multiline_sequence(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_fasta
        fa = tmp_path / "proteins.fa"
        fa.write_text(">gene1\nMKT\nLYF\n")
        proteins = extract_proteins_from_fasta(str(fa))
        assert proteins["gene1"] == "MKTLyf".upper()

    def test_missing_file(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_fasta
        proteins = extract_proteins_from_fasta(str(tmp_path / "nope.fa"))
        assert proteins == {}

    def test_empty_header(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_fasta
        fa = tmp_path / "proteins.fa"
        fa.write_text(">\nMKT\n")
        proteins = extract_proteins_from_fasta(str(fa))
        assert isinstance(proteins, dict)

    def test_last_sequence_captured(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_fasta
        fa = tmp_path / "proteins.fa"
        fa.write_text(">gene1\nMKT\n>gene2\nACD\n")
        proteins = extract_proteins_from_fasta(str(fa))
        assert len(proteins) == 2


class TestExtractProteinsFromGff3:
    def test_basic_extraction(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAAGAATTTTAA\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t15\t.\t+\t0\tID=gene1\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert "gene1" in proteins
        ps = proteins["gene1"]
        assert ps.gene_id == "gene1"
        assert ps.strand == "+"
        assert ps.contig == "contig1"
        assert ps.start == 1
        assert ps.end == 15

    def test_minus_strand(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        # ATGAAATTT on + strand -> reverse complement: AAATTCAT
        genome.write_text(">contig1\nATGAAATTT\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t9\t.\t-\t0\tID=gene_minus\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert "gene_minus" in proteins

    def test_non_cds_features_skipped(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAATTT\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tgene\t1\t9\t.\t+\t0\tID=gene1\n"
            "contig1\t.\tmRNA\t1\t9\t.\t+\t0\tID=mrna1\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert len(proteins) == 0

    def test_missing_files(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        proteins = extract_proteins_from_gff3(
            str(tmp_path / "no.fa"), str(tmp_path / "no.gff3")
        )
        assert proteins == {}

    def test_no_id_attribute_skipped(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAATTT\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t9\t.\t+\t0\tParent=mrna1\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert len(proteins) == 0

    def test_short_gff_line_skipped(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAATTT\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert len(proteins) == 0

    def test_comment_lines_skipped(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAATTT\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "# this is a comment\n"
            "\n"
            "contig1\t.\tCDS\t1\t9\t.\t+\t0\tID=gene1\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert "gene1" in proteins

    def test_contig_not_in_genome(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig2\nATGAAATTT\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t9\t.\t+\t0\tID=gene1\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert len(proteins) == 0

    def test_multiline_genome(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAA\nTTTTAA\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t12\t.\t+\t0\tID=gene1\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert "gene1" in proteins

    def test_multiple_contigs(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3
        genome = tmp_path / "genome.fa"
        genome.write_text(
            ">contig1\nATGAAATTT\n"
            ">contig2\nAAACCCTTT\n"
        )
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t9\t.\t+\t0\tID=gene1\n"
            "contig2\t.\tCDS\t1\t9\t.\t+\t0\tID=gene2\n"
        )
        proteins = extract_proteins_from_gff3(str(genome), str(gff))
        assert "gene1" in proteins
        assert "gene2" in proteins


class TestExtractProteinSequences:
    def test_with_gff3(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_protein_sequences
        genome = tmp_path / "genome.fa"
        genome.write_text(">contig1\nATGAAAGAATTTTAA\n")
        gff = tmp_path / "ann.gff3"
        gff.write_text(
            "contig1\t.\tCDS\t1\t15\t.\t+\t0\tID=gene1\n"
        )
        proteins = extract_protein_sequences(str(genome), str(gff))
        assert "gene1" in proteins
        assert isinstance(proteins["gene1"], str)

    def test_without_gff3(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_protein_sequences
        fa = tmp_path / "proteins.fa"
        fa.write_text(">gene1\nMKTLYF\n")
        proteins = extract_protein_sequences(str(fa))
        assert proteins["gene1"] == "MKTLYF"

    def test_none_gff3(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_protein_sequences
        fa = tmp_path / "proteins.fa"
        fa.write_text(">gene1\nMKTLYF\n")
        proteins = extract_protein_sequences(str(fa), gff3_path=None)
        assert proteins["gene1"] == "MKTLYF"

    def test_empty_gff3_string(self, tmp_path):
        from helixlang.plugins.annotation.sequences import extract_protein_sequences
        fa = tmp_path / "proteins.fa"
        fa.write_text(">gene1\nMKTLYF\n")
        proteins = extract_protein_sequences(str(fa), gff3_path="")
        assert proteins["gene1"] == "MKTLYF"
