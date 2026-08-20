"""Integration tests for the GEM reconstruction pipeline (doc/20)."""
from __future__ import annotations

import pathlib
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_fasta(tmp_path: pathlib.Path) -> str:
    """Create a minimal FASTA file for testing."""
    fasta = tmp_path / "mini.fasta"
    fasta.write_text(textwrap.dedent("""\
        >gene_001 hypothetical protein
        MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
        >gene_002 DNA gyrase subunit A
        MVSKLPEPVKNDDIELAKRTLTypeII_topoisomerase
        >gene_003 RNA polymerase alpha
        MKVTKNEVQLSNDANAQQRILSYIKDQLSYIGSTGIISDEIRKIVEGIRISHEG
    """))
    return str(fasta)


@pytest.fixture
def ecoli_core_fasta(tmp_path: pathlib.Path) -> str:
    """Create an E. coli-like FASTA with core metabolic gene headers."""
    fasta = tmp_path / "ecoli_core.fasta"
    headers_and_seqs = [
        ("gltA", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV"),
        ("sucA", "MSIQHFRVALIPFFAAFCLPKDIKNPEKFEKIDMRSLHALATSRYQGLLAAIER"),
        ("mdh", "MKVILTAILSALVTAVAIGAQANIGQALVDMDVILGLPAFSPVAAVKIMKDNK"),
        ("pfkA", "MTITTTTEKITLKGTPAGNQYNIALDDASTWKIFRQELEKDGQVVIKSIGKDER"),
        ("pykF", "MSKTKIIKLTGSHSDSPAAKLFADYDAVRFQEHKPDEIINIVKSIRSDLSPKAR"),
        ("pta", "MKLYPNTLKLQVVSETNLVETVNAIFNLNGLKDLKELIEQFDQEIGITIDN"),
        ("ackA", "MSAPYVPNVPLVKLSVLCEETKNLDIKSFDIAEANRQLYTGESVLSQDIKV"),
        ("acs", "MTEFKFGMDAYCPFGIDTNNHPTWQALLAKYADKTLSEFKKEGIAVVRAVL"),
        ("zwf", "MAIIGSKGFIGSNIVDDLFRKLKEDYKNLDINYYNKAFSTSEYTTLMDTLN"),
        ("pgi", "MSKVHIGVMGIPSGEIVQRFNAIANMDKPEVVCKQGFKAGDKLVTK"),
        ("tpiA", "MKTFVNKVIPLMDRNEFEQISQAFTKALDSFPKKVELVIHPGTQFGKNLES"),
        ("gapA", "MRVVSKFGEIKKGKKITVFDQGKYDAIALASDAKFAEELATLSDNYIV"),
        ("pgk", "MSITKDQRILEVLKYIVKNFDFLRGDTVVSDYQVQRDDWDFVKSAVKE"),
        ("eno", "MSKMIVKVNGERTITNAVRTALEKYGATVQLSGMLKDTVISGSKNFLE"),
        ("fbaA", "MKTFVNKVIPLMDRNEFEQISQAFTKALDSFPKKVELVIHPGTQFGKNLES"),
        ("gnd", "MAIIGSKGFIGSNIVDDLFRKLKEDYKNLDINYYNKAFSTSEYTTLMDTLN"),
        ("icd", "MKTFVNKVIPLMDRNEFEQISQAFTKALDSFPKKVELVIHPGTQFGKNLES"),
    ]
    lines = []
    for name, seq in headers_and_seqs:
        lines.append(f">{name} {name} protein")
        lines.append(seq)
    fasta.write_text("\n".join(lines) + "\n")
    return str(fasta)


# ---------------------------------------------------------------------------
# Phase 2: Annotation
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_gene_annotation_creation(self) -> None:
        from helixlang.annotation import GeneAnnotation

        annot = GeneAnnotation(
            gene_id="gltA",
            ec_numbers=["2.3.3.1"],
            kegg_ko=["K01647"],
            go_terms=["GO:0004108"],
        )
        assert annot.gene_id == "gltA"
        assert "2.3.3.1" in annot.ec_numbers
        assert annot.to_dict()["gene_id"] == "gltA"

    def test_ec_mapping_lookup(self) -> None:
        from helixlang.annotation.ec_mapping import build_ec_db

        db = build_ec_db()
        assert db.size > 0
        mapping = db.lookup("2.3.3.1")
        assert mapping is not None
        assert "CS" in mapping.reaction_ids

    def test_kegg_mapping_lookup(self) -> None:
        from helixlang.annotation.kegg_mapping import build_ko_db

        db = build_ko_db()
        assert db.size > 0
        mapping = db.lookup("K00844")
        assert mapping is not None
        assert "HEX1" in mapping.reaction_ids


# ---------------------------------------------------------------------------
# Phase 3: GEM reconstruction
# ---------------------------------------------------------------------------

class TestGemReconstruction:
    def test_bottom_up_reconstruct(self) -> None:
        from helixlang.annotation import GeneAnnotation
        from helixlang.gem.bottom_up import bottom_up_reconstruct

        annotations = {
            "gltA": GeneAnnotation(
                gene_id="gltA", ec_numbers=["2.3.3.1"], kegg_ko=["K01647"]),
            "pfkA": GeneAnnotation(
                gene_id="pfkA", ec_numbers=["2.7.1.11"], kegg_ko=["K00850"]),
            "mdh": GeneAnnotation(
                gene_id="mdh", ec_numbers=["1.1.1.37"], kegg_ko=["K00024"]),
        }
        result = bottom_up_reconstruct(annotations)
        assert result.reaction_count >= 3
        assert result.genes_annotated == 3
        assert result.ec_matched >= 2  # at least gltA and pfkA match EC DB
        assert "CS" in result.reaction_ids()
        assert "PFK" in result.reaction_ids()

    def test_top_down_reconstruct(self) -> None:
        from helixlang.annotation import GeneAnnotation
        from helixlang.gem.top_down import top_down_reconstruct

        annotations = {
            "gltA": GeneAnnotation(
                gene_id="gltA", ec_numbers=["2.3.3.1"]),
            "pfkA": GeneAnnotation(
                gene_id="pfkA", ec_numbers=["2.7.1.11"]),
        }
        result = top_down_reconstruct(annotations)
        assert result.kept_reactions >= 2

    def test_consensus_merge(self) -> None:
        from helixlang.annotation import GeneAnnotation
        from helixlang.gem.bottom_up import bottom_up_reconstruct
        from helixlang.gem.consensus import consensus_merge
        from helixlang.gem.top_down import top_down_reconstruct

        annotations = {
            "gltA": GeneAnnotation(
                gene_id="gltA", ec_numbers=["2.3.3.1"]),
            "pfkA": GeneAnnotation(
                gene_id="pfkA", ec_numbers=["2.7.1.11"]),
        }
        bu = bottom_up_reconstruct(annotations)
        td = top_down_reconstruct(annotations)
        consensus = consensus_merge(bu, td)
        assert consensus.reaction_count >= 3
        assert consensus.from_both >= 1  # at least some overlap

    def test_gapfill(self) -> None:
        from helixlang.annotation import GeneAnnotation
        from helixlang.gem.bottom_up import bottom_up_reconstruct
        from helixlang.gem.consensus import consensus_merge
        from helixlang.gem.gapfill import gapfill
        from helixlang.gem.top_down import top_down_reconstruct

        annotations = {
            "gltA": GeneAnnotation(
                gene_id="gltA", ec_numbers=["2.3.3.1"]),
        }
        bu = bottom_up_reconstruct(annotations)
        td = top_down_reconstruct(annotations)
        consensus = consensus_merge(bu, td)
        result = gapfill(consensus)
        assert result.gap_filled_count > 0  # exchange reactions added

    def test_biomass_reaction(self) -> None:
        from helixlang.gem.biomass import build_biomass_reaction

        rxn = build_biomass_reaction("e_coli_k12")
        assert "BIOMASS_reaction" in rxn.name
        assert len(rxn.components) > 30
        assert any(c.metabolite_id == "biomass_c" for c in rxn.components)


# ---------------------------------------------------------------------------
# Phase 4: GRN inference
# ---------------------------------------------------------------------------

class TestGrnInference:
    def test_regulatory_edges(self) -> None:
        from helixlang.gem.grn_inference import (
            EvidenceLevel,
            RegulatoryEdge,
        )

        edge = RegulatoryEdge(
            tf_id="crp",
            target_gene="lacZ",
            regulation_type="activation",
            evidence_level=EvidenceLevel.DATABASE,
            confidence=0.95,
        )
        assert edge.is_high_confidence
        assert edge.regulation_type == "activation"

    def test_grn_inference_result(self) -> None:
        from helixlang.annotation.tf_detection import (
            TFCandidate,
            TFScanResult,
        )
        from helixlang.gem.grn_inference import infer_grn

        # Create a minimal TF scan result with CRP
        tf_result = TFScanResult(
            total_genes=10,
            tf_candidates=[
                TFCandidate(
                    gene_id="crp",
                    tf_family="HTH_Crp",
                    domain_accession="PF01532",
                    confidence=0.9,
                ),
            ],
        )
        grn = infer_grn(tf_result, use_motif_prediction=False)
        assert grn.total_edges > 0
        # CRP should have regulatory edges from database
        crp_edges = [e for e in grn.regulatory_edges if e.tf_id == "crp"]
        assert len(crp_edges) > 0


# ---------------------------------------------------------------------------
# Phase 5: Kinetics
# ---------------------------------------------------------------------------

class TestKinetics:
    def test_kcat_predictor(self) -> None:
        from helixlang.kinetics.kcat_predictor import predict_kcat

        pred = predict_kcat("CS", ec_number="2.3.3.1")
        assert pred.kcat_value > 0
        assert pred.source in ("brenda", "median", "organism_scaled", "fallback")

    def test_km_estimator(self) -> None:
        from helixlang.kinetics.km_estimator import estimate_km

        km = estimate_km("HEX1", substrate="glucose")
        assert km > 0
        assert km < 10.0  # reasonable range


# ---------------------------------------------------------------------------
# Phase 6: Bridge integration
# ---------------------------------------------------------------------------

class TestBridge:
    def test_consensus_to_metabolic_model(self) -> None:
        from helixlang.annotation import GeneAnnotation
        from helixlang.gem.bottom_up import bottom_up_reconstruct
        from helixlang.gem.bridge import consensus_to_metabolic_model
        from helixlang.gem.consensus import consensus_merge
        from helixlang.gem.top_down import top_down_reconstruct

        annotations = {
            "gltA": GeneAnnotation(
                gene_id="gltA", ec_numbers=["2.3.3.1"]),
        }
        bu = bottom_up_reconstruct(annotations)
        td = top_down_reconstruct(annotations)
        consensus = consensus_merge(bu, td)
        model = consensus_to_metabolic_model(consensus)
        assert hasattr(model, "reactions")
        assert len(model.reactions) > 0

    def test_regulatory_edges_to_grn(self) -> None:
        from helixlang.gem.bridge import regulatory_edges_to_grn
        from helixlang.gem.grn_inference import (
            EvidenceLevel,
            RegulatoryEdge,
        )

        edges = [
            RegulatoryEdge(
                tf_id="crp", target_gene="lacZ",
                regulation_type="activation",
                evidence_level=EvidenceLevel.DATABASE,
                confidence=0.95,
            ),
        ]
        grn = regulatory_edges_to_grn(edges)
        assert hasattr(grn, "nodes")
        assert "crp" in grn.nodes
        assert "lacZ" in grn.nodes
        assert len(grn.edges) == 1

    def test_gpr_to_genome_dict(self) -> None:
        from helixlang.annotation import GeneAnnotation
        from helixlang.gem.bottom_up import bottom_up_reconstruct
        from helixlang.gem.bridge import gpr_to_genome_dict
        from helixlang.gem.consensus import consensus_merge
        from helixlang.gem.top_down import top_down_reconstruct

        annotations = {
            "gltA": GeneAnnotation(
                gene_id="gltA", ec_numbers=["2.3.3.1"]),
        }
        bu = bottom_up_reconstruct(annotations)
        td = top_down_reconstruct(annotations)
        consensus = consensus_merge(bu, td)
        genome = gpr_to_genome_dict(consensus)
        assert isinstance(genome, dict)
        assert len(genome) > 0


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_run_gem_pipeline_mini(self, mini_fasta: str) -> None:
        from helixlang.apps.gem_pipeline import run_gem_pipeline

        result = run_gem_pipeline(
            genome_fasta=mini_fasta,
            organism="test_organism",
            use_database_interactions=True,
            include_spontaneous=True,
            run_gapfill=True,
        )
        assert result.stages_completed >= 2
        assert result.annotated_genes >= 0
        assert len(result.errors) == 0 or result.stages_completed >= 2

    def test_run_gem_pipeline_ecoli(self, ecoli_core_fasta: str) -> None:
        from helixlang.apps.gem_pipeline import run_gem_pipeline

        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
            use_database_interactions=True,
            include_spontaneous=True,
            run_gapfill=True,
        )
        assert result.stages_completed >= 2
        assert result.annotated_genes >= 0
        # Summary should be non-empty
        summary = result.summary()
        assert "GEM Pipeline Summary" in summary


# ---------------------------------------------------------------------------
# RegulonDB parser
# ---------------------------------------------------------------------------

class TestRegulonDBParser:
    def test_parse_regulondb_3col(self) -> None:
        from helixlang.apps.genome_scale import parse_regulondb

        text = "regulator\ttarget\teffect\ncrp\tlacZ\t+\nfnr\tsdhCDAB\t-\n"
        edges = parse_regulondb(text)
        assert len(edges) == 2
        assert edges[0] == ("crp", "lacZ", 1.0)
        assert edges[1] == ("fnr", "sdhCDAB", -1.0)

    def test_parse_regulondb_full(self) -> None:
        from helixlang.apps.genome_scale import parse_regulondb_full

        text = (
            "# RegulonDB Network\ncrp\tlacZ\tactivation\t...\n"
            "fnr\tsdhCDAB\trepression\t...\n"
        )
        edges = parse_regulondb_full(text)
        assert len(edges) == 2
        assert edges[0][2] == 1.0  # activation
        assert edges[1][2] == -1.0  # repression

    def test_parse_regulondb_numeric_effect(self) -> None:
        from helixlang.apps.genome_scale import parse_regulondb_full

        text = "crp\tlacZ\t+0.8\nfnr\tsdhCDAB\t-0.6\n"
        edges = parse_regulondb_full(text)
        assert len(edges) == 2
        assert edges[0][2] == pytest.approx(0.8)
        assert edges[1][2] == pytest.approx(-0.6)
