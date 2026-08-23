"""Tests for full-chain custom organism pipeline (doc/26 Phase F)."""
from __future__ import annotations

import tempfile

from helixlang.apps.full_pipeline import (
    PipelineConfig,
    PipelineResult,
    _is_nucleotide_sequence,
    _translate_dna_to_protein,
    run_full_pipeline,
)


def _write_fasta(sequences: dict[str, str], suffix: str = ".fasta") -> str:
    path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    for name, seq in sequences.items():
        path.write(f">{name}\n{seq}\n")
    path.close()
    return path.name


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.organism_name == "custom_organism"
        assert c.ecgem is True
        assert c.community is False

    def test_custom(self):
        c = PipelineConfig(organism_name="my_org", ecgem=False)
        assert c.organism_name == "my_org"
        assert c.ecgem is False


class TestPipelineResult:
    def test_creation(self):
        r = PipelineResult()
        assert len(r.stages_completed) == 0
        assert r.pipeline_time == 0.0


class TestNucleotideDetection:
    def test_protein_not_nucleotide(self):
        assert not _is_nucleotide_sequence("MKWVTFISLLFLFSSAYS")

    def test_dna_is_nucleotide(self):
        assert _is_nucleotide_sequence("ATGAAATTTGTTCTGGGG")

    def test_mixed_is_nucleotide(self):
        assert _is_nucleotide_sequence("ATGCATGC")

    def test_empty(self):
        assert not _is_nucleotide_sequence("")


class TestDNATranslation:
    def test_basic_translation(self):
        protein = _translate_dna_to_protein("ATGAAATTTGTTCTG")
        assert protein == "MKFVL"

    def test_stop_codon(self):
        protein = _translate_dna_to_protein("ATGTAA")
        assert protein == "M"

    def test_unknown_codon(self):
        protein = _translate_dna_to_protein("NNN")
        assert protein == "X"

    def test_empty(self):
        assert _translate_dna_to_protein("") == ""


class TestRunFullPipeline:
    def test_protein_fasta(self):
        path = _write_fasta({
            "prot1": "MKWVTFISLLFLFSSAYSAVA",
            "prot2": "ACDEFGHIKLMNPQRSTVWY",
        })
        config = PipelineConfig(ecgem=False, community=False)
        result = run_full_pipeline(path, config)
        assert isinstance(result, PipelineResult)
        assert "A_fasta" in result.stages_completed
        assert result.pipeline_time > 0

    def test_empty_fasta(self):
        path = _write_fasta({})
        result = run_full_pipeline(path)
        assert isinstance(result, PipelineResult)
        assert len(result.proteins) == 0

    def test_with_ecgem(self):
        path = _write_fasta({"prot1": "MKWVTFISLLFLFSSAYS"})
        config = PipelineConfig(ecgem=True, community=False)
        result = run_full_pipeline(path, config)
        assert "D_ecgem" in result.stages_completed or "C_kinetics" in result.stages_completed

    def test_structure_stage(self):
        path = _write_fasta({"prot1": "MKWVTFISLLFLFSSAYS"})
        config = PipelineConfig(ecgem=False)
        result = run_full_pipeline(path, config)
        assert "B_structure" in result.stages_completed or "A_fasta" in result.stages_completed

    def test_kinetics_predictions(self):
        path = _write_fasta({"prot1": "MKWVTFISLLFLFSSAYS"})
        config = PipelineConfig(ecgem=False)
        result = run_full_pipeline(path, config)
        assert len(result.kcat_predictions) > 0

    def test_nucleotide_fasta_translated(self):
        path = _write_fasta({"glucokinase": "ATGAAATTTGTTCTGGGG"})
        config = PipelineConfig(ecgem=False)
        result = run_full_pipeline(path, config)
        assert len(result.proteins) > 0
        assert result.proteins[0].sequence == "MKFVLG"

    def test_dFBA_simulation_output(self):
        path = _write_fasta({"prot1": "MKWVTFISLLFLFSSAYS"})
        config = PipelineConfig(ecgem=True, community=False, ticks=100)
        result = run_full_pipeline(path, config)
        assert "dfba" in result.simulation

    def test_community_stage(self):
        path = _write_fasta({"prot1": "MKWVTFISLLFLFSSAYS"})
        config = PipelineConfig(ecgem=True, community=True, ticks=10)
        result = run_full_pipeline(path, config)
        if "E_community" in result.stages_completed:
            assert result.community is not None
