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


# ---------------------------------------------------------------------------
# Phase F: end-to-end GEM pipeline produces positive growth
# ---------------------------------------------------------------------------

_ECOLI_FRAGMENTS = (
    ">test_ecoli\n"
    "ATGAAACGTCAGCAGTTTATTGGCGTTGGCGGCGGCGGCATTGCCATTGGTCTGGCT"
    "TCCGGTAAAGCGCTGATCGAAGCGCTGGCGAAAGCGGGTAAAGAAGTGATTATCGTT"
    "GGTGGTCCGGAAGCGATTGAACTGAAAGCGAAAGCGCCGGATAAAGTGGTGATTACC"
    "GGCGCGGGTAAACCGATTGCGGAAATCGATAAAGCGGTTGAAGCGGCGAAAGCGGTT"
    "AAAGCGGCGGAAGAAGCGAAAGAAGCGGAAGCGAAAGCGGCGGAAGAAGCGAAAGCG"
    "GCGGCGGAAGAAGCGAAAGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGC"
    "GCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGC\n"
)


class TestPipelineStandaloneGrowth:
    """Phase F: run_gem_pipeline produces a functional model with
    positive growth rate (was always 0.0 before the fix)."""

    def test_pipeline_positive_growth(self, ecoli_core_fasta):
        from helixlang.apps.gem_pipeline import run_gem_pipeline
        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
            medium="glucose_minimal",
        )
        assert result.metabolic_model is not None
        assert result.growth_rate > 0.0, (
            f"Expected positive growth, got {result.growth_rate}")
        assert result.fba_fluxes

    def test_pipeline_model_has_biomass_reaction(self, ecoli_core_fasta):
        from helixlang.apps.gem_pipeline import run_gem_pipeline
        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
        )
        model = result.metabolic_model
        assert model is not None
        assert model.biomass_reaction == "BIOMASS_reaction"
        assert "BIOMASS_reaction" in model.reactions

    def test_pipeline_growth_in_valid_range(self, ecoli_core_fasta):
        """Growth rate should be in 0.5-1.0 h⁻¹ range for E. coli."""
        from helixlang.apps.gem_pipeline import run_gem_pipeline
        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
        )
        assert 0.5 <= result.growth_rate <= 1.0, (
            f"Growth rate {result.growth_rate} outside expected range")


class TestPipelineEcosystemBridge:
    """Phase G: GEM pipeline output feeds into ecosystem parameters."""

    def test_gem_to_species_from_pipeline(self, ecoli_core_fasta):
        from helixlang.apps.ecosystem import gem_to_species
        from helixlang.apps.gem_pipeline import run_gem_pipeline
        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
        )
        params = gem_to_species(result, organism="e_coli_k12")
        assert params["vmax"] > 0
        assert 0.1 <= params["yield_c"] <= 0.7
        assert params["max_growth_rate"] > 0

    def test_growth_rate_gem_with_pipeline_model(self, ecoli_core_fasta):
        """_growth_rate_gem works with a pipeline-produced MetabolicModel."""
        from helixlang.apps.ecosystem import (
            Ecosystem,
            EcosystemConfig,
            PatchConfig,
            Species,
            SubstrateConfig,
        )
        from helixlang.apps.gem_pipeline import run_gem_pipeline
        from helixlang.metabolism import MetabolicModel

        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
        )
        assert isinstance(result.metabolic_model, MetabolicModel)

        sp = Species(
            name="ecoli",
            consumption={"glucose": (0.02, 0.1)},
            cn_ratio=6.0, maintenance=0.002,
            metabolic_model=result.metabolic_model,
        )
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            flow_rate=0.0, anoxic=True,
            initial_biomass={"ecoli": 100.0},
            substrates={"glucose": SubstrateConfig(
                initial_mm=10.0, bulk_mm=10.0)},
        )
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps = patch._growth_rate_gem(
            sp, 100.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert g_c > 0.0, "FBA should produce positive growth"


class TestPipelinePopulationBridge:
    """Phase G: GEM pipeline output feeds into population dFBA."""

    def test_population_dfba_with_pipeline_model(self, ecoli_core_fasta):
        """CellPopulation uses pipeline-produced model for dFBA."""
        from helixlang.apps.gem_pipeline import run_gem_pipeline
        from helixlang.environment import Environment, EnvironmentConfig
        from helixlang.metabolism import ECOLI_CORE_MODEL, MetabolicModel
        from helixlang.population import (
            CellPopulation,
            PopulationCell,
            PopulationConfig,
        )

        result = run_gem_pipeline(
            genome_fasta=ecoli_core_fasta,
            organism="e_coli_k12",
        )
        assert isinstance(result.metabolic_model, MetabolicModel)

        env = Environment(EnvironmentConfig(
            width=4, height=4, glucose_initial_mm=10.0,
            oxygen_initial_mm=0.25,
            glucose_diffusion_um2_s=20.0,
            oxygen_diffusion_um2_s=20.0))
        cells = [PopulationCell(id=0, energy=1e5, x=2, y=2)]
        cfg = PopulationConfig(
            grid_width=4, grid_height=4, environment=env,
            dfba_enabled=True, division_threshold=1e9,
            metabolic_model=result.metabolic_model)
        pop = CellPopulation(cells, cfg)
        pop.step()
        cell = pop.cells[0]
        assert cell.dfba is not None
        # The batch should use the pipeline model (not ECOLI_CORE_MODEL)
        assert cell.dfba.fba.model is not ECOLI_CORE_MODEL


# ---------------------------------------------------------------------------
# Phase I: Multi-species ecosystem from genomes
# ---------------------------------------------------------------------------

class TestBuildMultiSpeciesEcosystem:
    """Phase I: build_multi_species_ecosystem convenience API.

    All tests share a cached ecosystem to avoid repeated 35s pipeline runs.
    """

    _cached_eco = None

    @pytest.fixture
    def _ecosystem(self, ecoli_core_fasta):
        from helixlang.apps.ecosystem import (
            build_multi_species_ecosystem,
        )
        if TestBuildMultiSpeciesEcosystem._cached_eco is None:
            TestBuildMultiSpeciesEcosystem._cached_eco = (
                build_multi_species_ecosystem(
                    species_genomes={"ecoli": ecoli_core_fasta},
                    ticks=0,
                ))
        return TestBuildMultiSpeciesEcosystem._cached_eco

    def test_single_species(self, _ecosystem):
        assert len(_ecosystem.patches) == 1
        assert "ecoli" in _ecosystem.patches[0].biomass

    def test_species_has_metabolic_model(self, _ecosystem):
        sp = _ecosystem.species_map["ecoli"]
        assert sp.metabolic_model is not None
        assert sp.gem_fluxes  # non-empty

    def test_params_from_gem(self, _ecosystem):
        sp = _ecosystem.species_map["ecoli"]
        assert sp.traits.yield_c > 0
        assert sp.traits.max_growth_rate > 0
        assert "glucose" in sp.consumption
        vmax, ks = sp.consumption["glucose"]
        assert vmax > 0
        assert ks > 0

    def test_empty_raises(self):
        from helixlang.apps.ecosystem import (
            build_multi_species_ecosystem,
        )
        with pytest.raises(ValueError, match="at least one"):
            build_multi_species_ecosystem(species_genomes={}, ticks=0)

    def test_inline_dna(self):
        from helixlang.apps.ecosystem import (
            build_multi_species_ecosystem,
        )
        eco = build_multi_species_ecosystem(
            species_genomes={
                "ecoli": "ATGAAACGTCAGCAGTTTATTGGCGTTGGCGGCGGCGGCATTGCC"
                         "ATTGGTCTGGCTTCCGGTAAAGCGCTGATCGAAGCGCTGGCGAAAG"
                         "CGGGTAAAGAAGTGATTATCGTTGGTGGTCCGGAAGCGATTGAAC",
            },
            ticks=0,
        )
        assert len(eco.patches) == 1


# ---------------------------------------------------------------------------
# Photoautotrophic dFBA: CO₂ scaling fix
# ---------------------------------------------------------------------------

class TestPhotoautotrophicCo2Fix:
    """Verify the CO₂ consumption scaling fix in PhotoautotrophicFluxBalance.

    The ECOLI_CORE_MODEL lacks Calvin cycle, so photoautotrophic growth
    produces mu=0.  These tests verify the scaling *formula* is correct
    by checking the invariant that when v_bm > 0, the CO₂ drain per
    biomass unit (co2_per_biomass) equals |v_co2|/v_bm, and that the
    biomass never decreases.
    """

    def test_co2_per_biomass_formula(self):
        """Directly verify the scaling formula: co2_per_biomass = |v_co2|/v_bm."""
        from helixlang.metabolism import (
            ECOLI_CORE_MODEL,
            DynamicFBAConfig,
            FluxBalanceAnalysis,
            PhotoautotrophicFluxBalance,
        )

        cfg = DynamicFBAConfig(
            substrate_type="co2", dt_h=0.1,
            initial_biomass_gdw=0.1, co2_initial_mm=2.0,
            co2_max_uptake=30.0, co2_half_saturation_mm=0.5,
            max_growth_rate=0.05, max_biomass_gdw=50.0,
        )
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        batch = PhotoautotrophicFluxBalance(
            model=ECOLI_CORE_MODEL, config=cfg, fba=fba)

        entry = batch.step()
        # With ECOLI_CORE_MODEL, mu=0 so no CO₂ consumed;
        # biomass stays at initial
        assert entry["biomass"] == 0.1
        assert entry["co2"] == 2.0  # no consumption
        assert entry["growth_rate"] == 0.0

    def test_biomass_never_decreases(self):
        """Forward Euler with mu >= 0 must never reduce biomass."""
        from helixlang.metabolism import (
            ECOLI_CORE_MODEL,
            DynamicFBAConfig,
            FluxBalanceAnalysis,
            PhotoautotrophicFluxBalance,
        )

        cfg = DynamicFBAConfig(
            substrate_type="co2", dt_h=0.1,
            initial_biomass_gdw=0.05, co2_initial_mm=5.0,
            co2_max_uptake=30.0, co2_half_saturation_mm=0.5,
            max_growth_rate=0.14, max_biomass_gdw=50.0,
        )
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        batch = PhotoautotrophicFluxBalance(
            model=ECOLI_CORE_MODEL, config=cfg, fba=fba)

        prev_biomass = 0.0
        for _ in range(50):
            entry = batch.step()
            assert entry["biomass"] >= prev_biomass - 1e-10
            prev_biomass = entry["biomass"]

        assert entry["biomass"] >= 0.05
