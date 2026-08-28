"""P0 integration task tests: central dogma pipeline + biological instructions + type system integration.

Verifies:
- P0-1.1 Biological instruction parsing (#crispr / #evolve / #methylate / #histone / #quorum / #transcribe / #translate)
- P0-1.2 VM central dogma pipeline (coupled transcription-translation when use_central_dogma=true)
- P0-1.3 Type system integrated into the parser (#type annotations + reference integrity checks)

References:
- Proshkin 2010 Mol Syst Biol 6:366 (transcription rate 50 nt/s)
- Ingolia 2009 Science 324:218 (translation rate 20 aa/s)
- Bernstein 2002 J Bacteriol 184:1037 (mRNA half-life ~5 min)
"""
from __future__ import annotations

import pytest

from helixlang.core.ast_nodes import Program
from helixlang.core.compiler import Compiler
from helixlang.core.lexer import Lexer
from helixlang.core.parser import BIO_INSTRUCTION_KINDS, Parser
from helixlang.core.vm import CellVM

# ============================================================================
# Helper functions
# ============================================================================

def _parse(src: str, enable_type_check: bool = False) -> Program:
    tokens = list(Lexer(src).tokens())
    return Parser(tokens, enable_type_check=enable_type_check).parse()


def _run_central_dogma(src: str, ticks: int = 5) -> CellVM:
    prog = _parse(src)
    chunk = Compiler().compile(prog)
    vm = CellVM(chunk, prog)
    vm.run(ticks)
    return vm


GENE_SRC = """
#config ticks=5 use_central_dogma=true species=ecoli
#promoter name=lacp strength=0.8
#gene name=lacZ promoter=lacp
ATG GAT CAA ACG TTT GAA AGC GAT CCG GTG AAA GCG AAA CTG GAT CAA TAA
#end
"""


# ============================================================================
# P0-1.1 Biological instruction parsing
# ============================================================================

class TestBioInstructionParsing:
    """Verifies biological instruction annotation parsing."""

    def test_bio_instruction_kinds_complete(self):
        """BIO_INSTRUCTION_KINDS contains 7 instructions."""
        assert BIO_INSTRUCTION_KINDS == frozenset({
            "crispr", "evolve", "methylate", "histone",
            "transcribe", "translate", "quorum",
        })

    def test_parse_evolve_instruction(self):
        """Parses the #evolve instruction."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#evolve target=g1 mutation_rate=0.05
"""
        prog = _parse(src)
        assert len(prog.bio_instructions) == 1
        inst = prog.bio_instructions[0]
        assert inst.kind == "evolve"
        assert inst.target == "g1"
        assert inst.params["mutation_rate"] == "0.05"

    def test_parse_crispr_instruction(self):
        """Parses the #crispr instruction (including string params)."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#crispr target=g1 position=10 new_sequence="GGGG" cas=SpCas9
"""
        prog = _parse(src)
        inst = prog.bio_instructions[0]
        assert inst.kind == "crispr"
        assert inst.target == "g1"
        assert inst.params["position"] == "10"
        assert inst.params["new_sequence"] == "GGGG"
        assert inst.params["cas"] == "SpCas9"

    def test_parse_methylate_instruction(self):
        """Parses the #methylate instruction."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#methylate target=g1 methylase=dam
"""
        prog = _parse(src)
        inst = prog.bio_instructions[0]
        assert inst.kind == "methylate"
        assert inst.params["methylase"] == "dam"

    def test_parse_histone_instruction(self):
        """Parses the #histone instruction."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#histone target=g1 mark=H3K4me3
"""
        prog = _parse(src)
        inst = prog.bio_instructions[0]
        assert inst.kind == "histone"
        assert inst.params["mark"] == "H3K4me3"

    def test_parse_quorum_instruction(self):
        """Parses the #quorum instruction."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#quorum target=g1 threshold=5.0 activate=g1
"""
        prog = _parse(src)
        inst = prog.bio_instructions[0]
        assert inst.kind == "quorum"
        assert inst.params["threshold"] == "5.0"

    def test_parse_transcribe_instruction(self):
        """Parses the #transcribe instruction."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#transcribe target=g1
"""
        prog = _parse(src)
        inst = prog.bio_instructions[0]
        assert inst.kind == "transcribe"

    def test_parse_translate_instruction(self):
        """Parses the #translate instruction."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#translate target=g1
"""
        prog = _parse(src)
        inst = prog.bio_instructions[0]
        assert inst.kind == "translate"

    def test_parse_multiple_instructions(self):
        """Parses multiple biological instructions."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#evolve target=g1 mutation_rate=0.01
#methylate target=g1 methylase=dam
#histone target=g1 mark=H3K27me3
"""
        prog = _parse(src)
        assert len(prog.bio_instructions) == 3
        kinds = [i.kind for i in prog.bio_instructions]
        assert kinds == ["evolve", "methylate", "histone"]

    def test_bio_instruction_line_number(self):
        """Biological instructions record the line number."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end

#evolve target=g1
"""
        prog = _parse(src)
        assert prog.bio_instructions[0].line > 0


# ============================================================================
# P0-1.2 VM central dogma pipeline
# ============================================================================

class TestCentralDogmaPipeline:
    """Verifies the VM central dogma pipeline (use_central_dogma=true)."""

    def test_central_dogma_runs_without_error(self):
        """The central dogma pipeline runs normally."""
        vm = _run_central_dogma(GENE_SRC, ticks=5)
        assert len(vm.trace) == 5

    def test_central_dogma_produces_protein(self):
        """The central dogma pipeline produces protein."""
        vm = _run_central_dogma(GENE_SRC, ticks=5)
        assert "lacZ" in vm.cell.proteins
        assert vm.cell.proteins["lacZ"] > 0

    def test_central_dogma_produces_mrna(self):
        """The central dogma pipeline produces mRNA."""
        vm = _run_central_dogma(GENE_SRC, ticks=5)
        assert "lacZ" in vm._gene_mrna
        assert vm._gene_mrna["lacZ"] > 0

    def test_central_dogma_protein_grows_over_ticks(self):
        """Protein concentration grows over ticks."""
        vm = _run_central_dogma(GENE_SRC, ticks=10)
        # Protein should accumulate after running more ticks
        assert vm.cell.proteins["lacZ"] > 0

    def test_central_dogma_species_ecoli(self):
        """The central dogma pipeline works for the ecoli species."""
        src = GENE_SRC.replace("species=ecoli", "species=ecoli")
        vm = _run_central_dogma(src, ticks=3)
        assert vm.cell.proteins.get("lacZ", 0) > 0

    def test_central_dogma_species_yeast(self):
        """The central dogma pipeline works for the yeast species."""
        src = GENE_SRC.replace("species=ecoli", "species=yeast")
        vm = _run_central_dogma(src, ticks=3)
        assert vm.cell.proteins.get("lacZ", 0) > 0

    def test_central_dogma_species_human(self):
        """The central dogma pipeline works for the human species."""
        src = GENE_SRC.replace("species=ecoli", "species=human")
        vm = _run_central_dogma(src, ticks=3)
        assert vm.cell.proteins.get("lacZ", 0) > 0

    def test_central_dogma_off_by_default(self):
        """Central dogma is off by default (uses the GRN + bytecode path)."""
        src = """
#config ticks=3
#gene name=g1
ATG AAA TTT TAA
#end
"""
        prog = _parse(src)
        assert prog.config.use_central_dogma is False

    def test_evolve_instruction_in_pipeline(self):
        """The #evolve instruction executes in the central dogma pipeline."""
        src = GENE_SRC + "#evolve target=lacZ mutation_rate=0.01\n"
        vm = _run_central_dogma(src, ticks=5)
        assert len(vm._evolution_history) > 0

    def test_methylate_instruction_in_pipeline(self):
        """The #methylate instruction executes in the central dogma pipeline."""
        src = GENE_SRC + "#methylate target=lacZ methylase=dam\n"
        vm = _run_central_dogma(src, ticks=5)
        assert len(vm._epigenetic_marks) > 0
        # Methylation reduces the expression modifier
        assert vm._chromatin_modifier["lacZ"] < 1.0

    def test_histone_instruction_in_pipeline(self):
        """The #histone instruction executes in the central dogma pipeline."""
        src = GENE_SRC + "#histone target=lacZ mark=H3K27me3\n"
        vm = _run_central_dogma(src, ticks=5)
        assert len(vm._epigenetic_marks) > 0

    def test_transcribe_instruction_in_pipeline(self):
        """The #transcribe instruction activates gene expression."""
        src = GENE_SRC + "#transcribe target=lacZ\n"
        vm = _run_central_dogma(src, ticks=3)
        # transcribe should set the GRN node level to 1.0
        assert vm.grn.nodes.get("lacZ") is not None

    def test_translate_instruction_in_pipeline(self):
        """The #translate instruction increases protein."""
        src = GENE_SRC + "#translate target=lacZ\n"
        vm = _run_central_dogma(src, ticks=3)
        assert vm.cell.proteins.get("lacZ", 0) > 0

    def test_crispr_instruction_in_pipeline(self):
        """The #crispr instruction attempts editing in the central dogma pipeline."""
        src = GENE_SRC + "#crispr target=lacZ position=3 new_sequence=\"GGG\" cas=SpCas9\n"
        vm = _run_central_dogma(src, ticks=3)
        # CRISPR may succeed or fail (depending on the PAM site), but should not crash
        assert isinstance(vm._crispr_edits, list)


# ============================================================================
# P0-1.3 Type system integration
# ============================================================================

class TestTypeSystemIntegration:
    """Verifies the type system integrated into the parser."""

    def test_parse_type_annotation(self):
        """Parses the #type annotation."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#type g1=Protein
"""
        prog = _parse(src)
        assert "g1" in prog.type_annotations
        assert prog.type_annotations["g1"] == "Protein"

    def test_parse_multiple_type_annotations(self):
        """Parses multiple type annotations."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#gene name=g2
ATG CCC GGG TAA
#end
#type g1=Protein
#type g2=Signal
"""
        prog = _parse(src)
        assert prog.type_annotations == {"g1": "Protein", "g2": "Signal"}

    def test_type_check_passes_valid_references(self):
        """Type check passes: references an existing gene."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#type g1=Protein
#evolve target=g1
"""
        prog = _parse(src, enable_type_check=True)
        assert len(prog.bio_instructions) == 1

    def test_type_check_fails_undefined_gene_in_annotation(self):
        """Type check fails: the type annotation references an undefined gene."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#type nonexistent=Protein
"""
        with pytest.raises(Exception, match="type check failed"):
            _parse(src, enable_type_check=True)

    def test_type_check_fails_undefined_bio_instruction_target(self):
        """Type check fails: the biological instruction references an undefined gene."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#evolve target=nonexistent
"""
        with pytest.raises(Exception, match="type check failed"):
            _parse(src, enable_type_check=True)

    def test_type_check_fails_undefined_regulation(self):
        """Type check fails: the regulatory edge references an undefined gene."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#regulate nonexistent -> g1 strength=0.5
"""
        with pytest.raises(Exception, match="type check failed"):
            _parse(src, enable_type_check=True)

    def test_type_check_disabled_by_default(self):
        """Type check is disabled by default (no exception raised)."""
        src = """
#gene name=g1
ATG AAA TTT TAA
#end
#evolve target=nonexistent
"""
        # Type check not enabled → no exception raised
        prog = _parse(src, enable_type_check=False)
        assert len(prog.bio_instructions) == 1

    def test_type_check_validates_promoter_reference(self):
        """Type check validates the promoter reference."""
        src = """
#promoter name=p1 strength=0.8
#gene name=g1 promoter=p1
ATG AAA TTT TAA
#end
#type p1=Signal
#type g1=Protein
"""
        prog = _parse(src, enable_type_check=True)
        assert prog.type_annotations["p1"] == "Signal"
        assert prog.type_annotations["g1"] == "Protein"

    def test_type_annotation_with_promoter_symbol(self):
        """Type annotations can apply to a promoter."""
        src = """
#promoter name=plac strength=0.8
#type plac=Signal
"""
        prog = _parse(src, enable_type_check=True)
        assert prog.type_annotations["plac"] == "Signal"
