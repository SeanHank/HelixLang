"""End-to-end integration validation: HelixLang full-pipeline tests.

Covers the full pipeline from HelixLang source to biological DNA and back to
execution:

1. **Compile pipeline**: source -> Lexer -> Parser -> Semantic -> Compiler -> VM
2. **Physical storage pipeline**: source -> Goldman/Erlich encoding -> DNA -> decode -> execute
3. **Bio-compile pipeline**: DNA -> ORF detection -> HelixLang -> compile -> VM
4. **Reverse bio pipeline**: HelixLang -> DNA (promoter/terminator/codon optimization) -> validate
5. **Full lifecycle**: source -> DNA -> synthesis -> decay -> PCR -> sequencing -> decode -> execute
6. **CLI integration**: command-line tool end-to-end
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("Bio")
pytest.importorskip("reedsolo")

from helixlang.biocodec import (
    LAC_PROMOTER,
    RRNB_T1_TERMINATOR,
    back_translate,
    find_orfs,
    validate_biological,
)
from helixlang.biocodec import (
    dna_to_helix as bio_dna_to_helix,
)
from helixlang.biocodec import (
    helix_to_dna as bio_helix_to_dna,
)
from helixlang.codon_table import Op, get_table
from helixlang.compiler import Compiler
from helixlang.dna_codec import (
    decay_dna,
    dna_to_helix,
    erlich_decode,
    erlich_encode,
    helix_to_dna,
    pcr_amplify,
    sequence_dna,
    synthesize_dna,
)
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.semantic import SemanticAnalyzer
from helixlang.vm import CellVM

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PYTHON = sys.executable


def _run_cli(filename, *args):
    """Run the helixlang CLI, returning (returncode, stdout, stderr)."""
    cmd = [PYTHON, "-m", "helixlang", str(EXAMPLES / filename), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


def _compile_and_run(src: str, table_name: str = "standard",
                     ticks: int | None = None):
    """Compile and run HelixLang source, returning (chunk, vm, trace)."""
    table = get_table(table_name)
    stop_codons = {c for c, op in table.items() if op == Op.OP_HALT}
    tokens = list(Lexer(src).tokens())
    program = Parser(tokens, stop_codons=stop_codons).parse()
    SemanticAnalyzer(program).check()
    chunk = Compiler(table).compile(program)
    vm = CellVM(chunk, program)
    if ticks is not None:
        program.config.ticks = ticks
    trace = vm.run(program.config.ticks)
    return chunk, vm, trace


# ============================================================================
# 1. Compile pipeline: all examples can compile and run
# ============================================================================

class TestCompilePipeline:
    """Verify the integrity of the HelixLang compile pipeline."""

    @pytest.mark.parametrize("example", [
        "01_hello_dna.helix",
        "02_lac_operon.helix",
        "03_plant_growth.helix",
        "04_turing_pattern.helix",
        "05_table_switch.helix",
    ])
    def test_all_examples_compile_and_run(self, example):
        """All examples compile and run without errors."""
        rc, out, err = _run_cli(example)
        assert rc == 0, f"{example} failed: {err}"

    def test_hello_dna_produces_proteins(self):
        """01_hello_dna should produce proteins."""
        chunk, vm, trace = _compile_and_run(
            (EXAMPLES / "01_hello_dna.helix").read_text())
        # Proteins should be produced
        assert any(s["proteins"] for s in trace)

    def test_lac_operon_has_dynamics(self):
        """02_lac_operon should have dynamic changes (protein or morphology)."""
        chunk, vm, trace = _compile_and_run(
            (EXAMPLES / "02_lac_operon.helix").read_text())
        # Proteins should accumulate or change over time
        all_prots = [str(s["proteins"]) for s in trace]
        assert len(set(all_prots)) > 1, "protein dynamics should change over time"

    def test_plant_growth_produces_morphology(self):
        """03_plant_growth should produce morphology points."""
        chunk, vm, trace = _compile_and_run(
            (EXAMPLES / "03_plant_growth.helix").read_text())
        assert len(vm.cell.morphology_points) > 0

    def test_turing_pattern_produces_field(self):
        """04_turing_pattern should produce a reaction-diffusion field."""
        chunk, vm, trace = _compile_and_run(
            (EXAMPLES / "04_turing_pattern.helix").read_text())
        assert vm.field is not None
        assert vm.field.total_v() > 0


# ============================================================================
# 2. Physical storage pipeline: HelixLang -> DNA -> HelixLang -> execute
# ============================================================================

class TestPhysicalStoragePipeline:
    """Verify that DNA physical storage roundtrips still compile and run."""

    def test_goldman_roundtrip_preserves_semantics(self):
        """After a Goldman encoding roundtrip, the source semantics are unchanged."""
        src = (EXAMPLES / "01_hello_dna.helix").read_text()
        # Encode -> decode
        enc = helix_to_dna(src, scheme="goldman")
        dec = dna_to_helix(enc, scheme="goldman")
        assert dec == src, "Goldman roundtrip altered source"
        # The decoded source should compile and run
        chunk, vm, trace = _compile_and_run(dec)
        assert any(s["proteins"] for s in trace)

    def test_erlich_roundtrip_preserves_semantics(self):
        """After an Erlich fountain-code roundtrip, the source semantics are unchanged."""
        src = (EXAMPLES / "01_hello_dna.helix").read_text()
        # Erlich needs a sufficiently large amount of data (K>=4)
        # If the source is too short, add supplementary data
        data = src.encode("utf-8")
        if len(data) < 128:
            # Use a longer source
            src = (EXAMPLES / "02_lac_operon.helix").read_text()
        enc = helix_to_dna(src, scheme="erlich")
        dec = dna_to_helix(enc, scheme="erlich")
        assert dec == src, "Erlich roundtrip altered source"
        # Compile and run
        chunk, vm, trace = _compile_and_run(dec)
        assert len(trace) > 0

    def test_goldman_with_moderate_pcr_recovers(self):
        """Goldman 4x redundancy should still recover under moderate PCR errors."""
        src = "#gene name=test\nATG GCT GGT TAA\n#end\n#config ticks=5\n"
        enc = helix_to_dna(src, scheme="goldman")
        # Inject PCR errors (low cycles; Goldman redundancy should correct them)
        rng = random.Random(42)
        for o in enc["oligos"]:
            o["full"] = pcr_amplify(o["full"], cycles=3, rng=rng, polymerase="taq")
        dec = dna_to_helix(enc, scheme="goldman")
        # Key syntax should be preserved
        assert "#gene" in dec
        assert "#end" in dec
        # Should compile
        chunk, vm, trace = _compile_and_run(dec)
        assert len(trace) > 0


# ============================================================================
# 3. Bio-compile pipeline: DNA -> HelixLang -> compile -> VM
# ============================================================================

class TestBioCompilePipeline:
    """Verify the biological DNA -> HelixLang -> compile and run pipeline."""

    def test_dna_to_helix_compiles_and_runs(self):
        """DNA -> HelixLang -> compile -> VM execution."""
        # Construct DNA containing an ORF
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATY"
        dna = back_translate(protein, optimize="cai") + "TAA"
        # DNA -> HelixLang
        result = bio_dna_to_helix(dna, min_length_aa=10)
        assert len(result.orfs) >= 1
        # HelixLang -> compile and run
        chunk, vm, trace = _compile_and_run(result.helix_source, ticks=5)
        assert len(trace) > 0

    def test_bio_roundtrip_helix_to_dna_to_helix(self):
        """HelixLang -> DNA (with regulatory elements) -> extract ORF -> HelixLang."""
        # Original HelixLang source
        src = "#gene name=gfp\nATG GCT TCT AAA GGT GAA GAA CTG TTC ACC GGT TAA\n#end"
        # HelixLang -> biological DNA
        bio_dna = bio_helix_to_dna(src, promoter="lac", terminator="rrnB_T1",
                                    optimize_codons=True, avoid_restriction=True)
        # Extract the ORF portion
        orf_dna = bio_dna[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        # DNA -> HelixLang
        result = bio_dna_to_helix(orf_dna, min_length_aa=3)
        assert len(result.orfs) >= 1
        # Should detect the start ATG
        assert result.orfs[0].start_codon == "ATG"
        # Compile and run
        chunk, vm, trace = _compile_and_run(result.helix_source, ticks=5)
        assert len(trace) > 0

    def test_cai_optimized_dna_validates_biological(self):
        """CAI-optimized DNA passes biological plausibility validation."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
        dna = back_translate(protein, optimize="cai") + "TAA"
        report = validate_biological(dna)
        assert report.has_start_codon
        assert report.has_stop_codon
        assert report.no_internal_stop
        assert report.cai_adequate


# ============================================================================
# 4. Full lifecycle: synthesis -> storage -> PCR -> sequencing -> decode -> execute
# ============================================================================

class TestFullLifecycle:
    """Full DNA storage lifecycle: simulates a realistic storage scenario."""

    def test_full_lifecycle_goldman_high_fidelity(self):
        """High-fidelity full pipeline: synthesis -> PCR(Q5) -> sequencing (PacBio HiFi) -> decode -> execute."""
        src = (EXAMPLES / "01_hello_dna.helix").read_text()
        # 1. Goldman encode
        enc = helix_to_dna(src, scheme="goldman")
        # 2. Chemical synthesis (high quality, 99.5% coupling)
        rng = random.Random(42)
        for o in enc["oligos"]:
            o["full"] = synthesize_dna(o["full"], rng=rng, quality="high")
        # 3. PCR amplification (Q5 high fidelity, 10 cycles)
        rng = random.Random(43)
        for o in enc["oligos"]:
            o["full"] = pcr_amplify(o["full"], cycles=10, rng=rng, polymerase="q5")
        # 4. PacBio HiFi sequencing
        rng = random.Random(44)
        for o in enc["oligos"]:
            o["full"] = sequence_dna(o["full"], platform="pacbio_hifi", rng=rng)
        # 5. Decode
        dec = dna_to_helix(enc, scheme="goldman")
        # 6. Verify it compiles and runs
        assert "#gene" in dec
        chunk, vm, trace = _compile_and_run(dec)
        assert len(trace) > 0

    def test_full_lifecycle_with_decay_cold_storage(self):
        """Still decodable after cold-storage decay (low temperature, very slow decay)."""
        src = "#gene name=test\nATG GCT GGT TAA\n#end\n#config ticks=5\n"
        # 1. Encode
        enc = helix_to_dna(src, scheme="goldman")
        # 2. Cold storage for 100 years at -20 deg C (very slow decay)
        rng = random.Random(42)
        for o in enc["oligos"]:
            o["full"] = decay_dna(o["full"], years=100, temperature_c=-20, rng=rng)
        # 3. PCR amplification
        rng = random.Random(43)
        for o in enc["oligos"]:
            o["full"] = pcr_amplify(o["full"], cycles=5, rng=rng, polymerase="q5")
        # 4. Decode
        dec = dna_to_helix(enc, scheme="goldman")
        # At -20 deg C, 100 years of decay is negligible -> should recover perfectly
        assert "#gene" in dec
        assert "#end" in dec

    def test_erlich_lifecycle_with_rs_correction(self):
        """Erlich + RS inner code should still decode after correcting a few errors."""
        # Use a source that is long enough (K>=4)
        src = (EXAMPLES / "02_lac_operon.helix").read_text()
        data = src.encode("utf-8")
        K = math.ceil(len(data) / 32)
        # 1. Erlich encode (K < 50 needs redundancy >= 2.5 for reliable BP decoding)
        oligos = erlich_encode(data, redundancy=2.5)
        # 2. Inject 1 substitution per oligo (RS should correct 1 symbol error)
        rng = random.Random(42)
        for o in oligos:
            pos = rng.randint(0, len(o.payload) - 1)
            bad = list(o.payload)
            bad[pos] = "A" if bad[pos] != "A" else "T"
            o.payload = "".join(bad)
        # 3. Decode (RS inner code corrects 1 error + BP fountain decode)
        decoded = erlich_decode(oligos, K=K, total_len=len(data))
        assert decoded == data
        # 4. Verify it compiles and runs
        recovered_src = decoded.decode("utf-8")
        chunk, vm, trace = _compile_and_run(recovered_src)
        assert len(trace) > 0


# ============================================================================
# 5. CLI integration tests
# ============================================================================

class TestCLIIntegration:
    """Verify the CLI tool end-to-end."""

    def test_cli_encode_decode_goldman(self, tmp_path):
        """CLI: --encode-dna=goldman successfully outputs FASTA."""
        # Encode
        rc, out, err = _run_cli("01_hello_dna.helix", "--encode-dna=goldman")
        assert rc == 0, f"encode failed: {err}"
        assert "oligos" in out
        assert ">oligo_" in out

    def test_cli_encode_dna_with_pcr(self):
        """CLI: --encode-dna + --pcr-cycles injects PCR errors."""
        rc, out, err = _run_cli("01_hello_dna.helix",
                                "--encode-dna=goldman", "--pcr-cycles=5")
        assert rc == 0, f"encode with PCR failed: {err}"
        assert ">oligo_" in out

    def test_cli_disassemble_all_examples(self):
        """CLI: --disassemble succeeds for all examples."""
        for f in sorted(EXAMPLES.glob("*.helix")):
            rc, out, err = _run_cli(f.name, "--disassemble")
            assert rc == 0, f"{f.name} disassemble failed: {err}"
            # annotation-only sim examples carry no bytecode; the classic
            # (default) examples must disassemble to real opcodes
            src = f.read_text()
            if "backend=" not in src or "#config backend=classic" in src:
                assert "OP_" in out, f"{f.name} has no disassembled opcodes"

    def test_cli_table_switch(self):
        """CLI: --table=mito_vertebrate switches the translation table."""
        rc_std, out_std, _ = _run_cli("05_table_switch.helix", "--disassemble")
        rc_mito, out_mito, _ = _run_cli("05_table_switch.helix",
                                         "--table=mito_vertebrate", "--disassemble")
        assert rc_std == 0 and rc_mito == 0
        # The standard and mitochondrial tables should produce different bytecode
        assert out_std != out_mito


# ============================================================================
# 6. Multi-layer integration: biological DNA + physical storage + execution
# ============================================================================

class TestMultiLayerIntegration:
    """Multi-layer integration test: biological DNA -> physical storage -> execution."""

    def test_bio_dna_stored_physically(self):
        """Biological DNA -> Goldman physical storage -> decode -> biological validation."""
        # 1. Construct biological DNA (promoter + ORF + terminator)
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATY"
        gfp_orf = back_translate(protein, optimize="cai") + "TAA"
        bio_dna = LAC_PROMOTER + gfp_orf + RRNB_T1_TERMINATOR
        # 2. Physical storage encode (store bio_dna as text)
        enc = helix_to_dna(bio_dna, scheme="goldman")
        # 3. Decode
        dec = dna_to_helix(enc, scheme="goldman")
        assert dec == bio_dna, "physical storage altered bio DNA"
        # 4. Extract the ORF from the decoded DNA
        orf_start = len(LAC_PROMOTER)
        orf_end = len(dec) - len(RRNB_T1_TERMINATOR)
        recovered_orf = dec[orf_start:orf_end]
        # 5. Biological validation
        report = validate_biological(recovered_orf)
        assert report.has_start_codon
        assert report.has_stop_codon
        assert report.no_internal_stop

    def test_helix_source_through_bio_compiler(self):
        """HelixLang source -> bio-compile -> DNA -> bio-decompile -> execute."""
        # Original HelixLang
        src = "#gene name=test\nATG GCT TCT AAA GGT GAA TAA\n#end"
        # Bio-compile -> DNA (promoter + codon optimization + restriction-site removal)
        bio_dna = bio_helix_to_dna(src, promoter="lac", terminator="rrnB_T1",
                                    optimize_codons=True, avoid_restriction=True)
        # Extract the ORF
        orf_dna = bio_dna[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        # Bio-decompile -> HelixLang
        result = bio_dna_to_helix(orf_dna, min_length_aa=3)
        assert len(result.orfs) >= 1
        # Compile and run
        chunk, vm, trace = _compile_and_run(result.helix_source, ticks=3)
        assert len(trace) > 0

    def test_error_cascade_tolerance(self):
        """Error-cascade tolerance: low error across the whole synthesis + PCR + sequencing chain, corrected by Goldman redundancy."""
        src = "#gene name=robust\nATG GCT ACC GGT TCT AAA GAA CTG TTC ACC GGT GCT TAA\n#end\n#config ticks=5\n"
        # Encode
        enc = helix_to_dna(src, scheme="goldman")
        # Inject errors across the whole chain (high quality, low error rate)
        rng = random.Random(42)
        for o in enc["oligos"]:
            # Synthesis (high quality, 99.5% coupling)
            o["full"] = synthesize_dna(o["full"], rng=rng, quality="high")
            # PCR (Q5 high fidelity, 5 cycles)
            o["full"] = pcr_amplify(o["full"], cycles=5, rng=rng, polymerase="q5")
            # Sequencing (PacBio HiFi Q40+)
            o["full"] = sequence_dna(o["full"], platform="pacbio_hifi", rng=rng)
        # Decode (Goldman 4x redundancy voting)
        dec = dna_to_helix(enc, scheme="goldman")
        # Core syntax should be preserved
        assert "#gene" in dec or "ATG" in dec


# ============================================================================
# 7. Realistic gene scenario (E. coli-like genes)
# ============================================================================

class TestRealisticGeneScenario:
    """Realistic gene scenario: full processing of an E. coli-like gene."""

    def test_ecoli_like_gene_full_pipeline(self):
        """E. coli-like gene: UTR + ORF + UTR -> detect -> compile -> execute."""
        # Construct an E. coli-like gene region
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
        orf_dna = back_translate(protein, optimize="cai") + "TAA"
        # Add UTRs (non-coding regions)
        utr5 = "ACGTACGTAC" * 5  # 50 bp 5'UTR
        utr3 = "TTTTACGTTT" * 3  # 30 bp 3'UTR
        gene_region = utr5 + orf_dna + utr3
        # Detect ORFs
        orfs = find_orfs(gene_region, min_length_aa=10)
        assert len(orfs) >= 1
        # The longest ORF should be GFP
        longest = max(orfs, key=lambda o: len(o.sequence))
        assert len(longest.sequence) >= len(orf_dna)
        # Convert to HelixLang and execute
        result = bio_dna_to_helix(gene_region, min_length_aa=10)
        chunk, vm, trace = _compile_and_run(result.helix_source, ticks=3)
        assert len(trace) > 0

    def test_multi_orf_gene_cluster(self):
        """Multi-ORF gene cluster: detection and compilation of multiple concatenated genes."""
        # Construct two concatenated ORFs
        protein1 = "MASKGEELFTGVPVPILVEL"
        protein2 = "DGDVNGHKFSVSGEGEGDATY"
        orf1 = back_translate(protein1, optimize="cai") + "TAA"
        orf2 = back_translate(protein2, optimize="cai") + "TAA"
        # Concatenate (add a long enough spacer to avoid spurious ATGs across ORFs)
        gene_cluster = orf1 + "ACGTACGTACGTACGTACGT" + orf2
        # Detect ORFs (use a higher min_length_aa to filter out spurious ORFs)
        orfs = find_orfs(gene_cluster, min_length_aa=10)
        assert len(orfs) >= 1, f"should detect >=1 ORF, got {len(orfs)}"
        # Only use ATG-started ORFs to generate HelixLang source (parser requires ATG)
        atg_orfs = [o for o in orfs if o.start_codon == "ATG"]
        if not atg_orfs:
            pytest.skip("no ATG-started ORF detected")
        # Manually construct HelixLang source (only ATG ORFs)
        helix_lines = ["# HelixLang bio-compiled from gene cluster"]
        for i, orf in enumerate(atg_orfs):
            codons = [orf.sequence[j:j+3] for j in range(0, len(orf.sequence), 3)]
            helix_lines.append(f"#gene name=orf_{i+1}")
            helix_lines.append(" ".join(codons))
            helix_lines.append("#end")
        helix_src = "\n".join(helix_lines)
        # Compile and run
        chunk, vm, trace = _compile_and_run(helix_src, ticks=3)
        assert len(trace) > 0

    def test_promoter_strength_integrates_with_grn(self):
        """The lac promoter strength parameter can be passed into the GRN model."""
        from helixlang.bio_data import lac_promoter_strength, mu_to_grn_strength
        # Repressed/induced promoter strength (verify they are callable)
        lac_promoter_strength(induced=False)
        lac_promoter_strength(induced=True)
        # Convert to a GRN threshold
        threshold_uninduced = mu_to_grn_strength(3.0)  # 3 MU
        threshold_induced = mu_to_grn_strength(3000.0)  # 3000 MU
        # The induced threshold should be lower (stronger expression)
        assert threshold_induced < threshold_uninduced
        # Build a GRN for the test
        from helixlang.grn import GRN
        grn = GRN()
        grn.add_gene("lacZ", threshold=threshold_induced)
        # Simulate a few steps (GRN is self-driving; an initial level of 0 only activates with an input edge)
        # Directly set the level to simulate induced expression
        grn.set_level("lacZ", 0.8)
        triggered = grn.step()
        # lacZ should be triggered
        assert "lacZ" in triggered
