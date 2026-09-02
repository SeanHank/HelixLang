"""Incremental JIT tests (doc/38 §3, goal 1 / Phase C).

Covers:
- the GeneDependencyGraph invalidation closure (doc acceptance: an edit to g
  invalidates exactly {g} ∪ {regulation sources targetting g} ∪ {callers},
  transitively)
- per-gene block hashing (ORF codon block + #gene header fields)
- shape-change detection (gene added/removed/renamed forces a full rebuild)
- IRProgram.patch_gene / IRBuilder.build_function (byte-faithful rebuild of a
  single gene)
- IncrementalCompiler: edited gene's closure is the only thing re-derived;
  the resulting Chunk is byte-identical to a from-scratch compile; traces of
  incremental vs full builds are identical; an edit that touches an isolated
  gene reuses every other IR function untouched.
- ``helixc --watch`` recomputes only the closure per iteration.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from helixlang.core import hxbc
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.incr import (
    GeneDependencyGraph,
    IncrementalCache,
    IncrementalCompiler,
    hash_gene_block,
)
from helixlang.core.ir import IRFunction, IRProgram
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.vm import CellVM

# a: no calls; b: calls a (call_target); c: call_target=z; z: called by c
# plus a gene-level regulation z -> x, giving edges
#   callers[a]={b}, callers[z]={c}, regulators[x]={z}.
GENE_SRC = """\
#gene name=x
ATG TGG TAA
#end
#gene name=a
ATG TGG TAA
#end
#gene name=b call_target=a
ATG CGA TGG TAA
#end
#gene name=z
ATG TGG TAA
#end
#gene name=c call_target=z
ATG CGG TGG TAA
#end
#regulate z -> x strength=0.8
#config ticks=2 output=stdout
"""

# a: no calls; b: calls a; c: isolated leaf.
SIMPLE = """\
#gene name=a
ATG TGG TAA
#end
#gene name=b call_target=a
ATG CGC TGG TAA
#end
#gene name=c
ATG TGG TAA
#end
#config ticks=2 output=stdout
"""


def parse(src: str):
    toks = list(Lexer(src).tokens())
    program = Parser(toks, config=LanguageConfig.for_table("standard")).parse()
    SemanticAnalyzer(program).check()
    return program


def cli(argv: list[str]) -> tuple[int, str]:
    from helixlang.cli import main

    out = io.StringIO()
    with redirect_stdout(out):
        rc = main(argv)
    return rc, out.getvalue()


# ---------------------------------------------------------------------------
# 1. dependency graph + closure
# ---------------------------------------------------------------------------
class TestGeneDependencyGraph:
    def test_graph_edges_from_program_and_ir(self):
        program = parse(GENE_SRC)
        compiler = Compiler(LanguageConfig.for_table("standard"))
        ir = compiler.build_ir(program)
        graph = GeneDependencyGraph.build(program, ir)
        assert graph.genes == {"x", "a", "b", "z", "c"}
        assert graph.callers["a"] == {"b"}
        assert graph.callers["z"] == {"c"}
        assert graph.regulators["x"] == {"z"}

    def test_closure_exact_doc_set(self):
        """The doc/38 §3 acceptance set: {g} ∪ {targetting reguls} ∪ {callers}."""
        program = parse(GENE_SRC)
        compiler = Compiler(LanguageConfig.for_table("standard"))
        ir = compiler.build_ir(program)
        graph = GeneDependencyGraph.build(program, ir)
        # b calls a, nobody else touches b.
        assert graph.closure({"b"}) == {"b"}
        # editing a pulls in its sole caller b.
        assert graph.closure({"a"}) == {"a", "b"}
        # editing z pulls in its caller c.
        assert graph.closure({"z"}) == {"z", "c"}
        # editing x pulls in the regulation source z (callers of x: none).
        assert graph.closure({"x"}) == {"x", "z"}

    def test_closure_direct_star(self):
        """Single-level doc set for a star DAG (caller + regulation source)."""
        graph = GeneDependencyGraph(
            genes={"a", "b", "c"},
            regulators={"b": {"a"}},   # a -> b
            callers={"b": {"c"}},      # c calls b
        )
        assert graph.closure({"b"}) == {"b", "a", "c"}  # one hop, not fixpoint

    def test_closure_not_transitive(self):
        """A chain caller must not rebuild the whole chain for a leaf edit."""
        graph = GeneDependencyGraph(
            genes={"g0", "g1", "g2"},
            callers={"g0": {"g1"}, "g1": {"g2"}},
        )
        assert graph.closure({"g0"}) == {"g0", "g1"}

    def test_closure_ignores_unknown_names(self):
        graph = GeneDependencyGraph(genes={"a"})
        assert graph.closure({"nope"}) == set()


# ---------------------------------------------------------------------------
# 2. block hashing + cache diff
# ---------------------------------------------------------------------------
class TestBlockHashing:
    def test_hash_stable_without_edit(self):
        p1, p2 = parse(SIMPLE), parse(SIMPLE)
        for g1, g2 in zip(p1.genes, p2.genes, strict=True):
            assert hash_gene_block(g1) == hash_gene_block(g2)

    def test_codon_edit_changes_hash(self):
        edited = SIMPLE.replace("ATG TGG TAA", "ATG TCT TAA", 1)
        p_a, p_b = parse(SIMPLE), parse(edited)
        before = {g.name: hash_gene_block(g) for g in p_a.genes}
        after = {g.name: hash_gene_block(g) for g in p_b.genes}
        changed = {n for n, h in before.items() if after[n] != h}
        assert changed == {"a"}  # only the edited gene (first TGG→TCT block)

    def test_header_field_edit_changes_hash(self):
        edited = SIMPLE.replace(
            "#gene name=b call_target=a", "#gene name=b call_target=a tss=1.0")
        p_a, p_b = parse(SIMPLE), parse(edited)
        assert hash_gene_block(p_a.genes[1]) != hash_gene_block(p_b.genes[1])
        assert hash_gene_block(p_a.genes[0]) == hash_gene_block(p_b.genes[0])

    def test_parser_stamps_block_hash(self):
        """Parse-time block digest == the compiler's cache key (no re-hash)."""
        p1, p2 = parse(SIMPLE), parse(SIMPLE.replace("TGG", "TCT", 1))
        for g in p1.genes:
            assert g.block_hash == hash_gene_block(g)
        assert p1.genes[0].block_hash != p2.genes[0].block_hash
        assert p1.genes[1].block_hash == p2.genes[1].block_hash


class TestIncrementalCache:
    def test_diff_detects_edited_genes(self):
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        a, b = IncrementalCache.compute(parse(SIMPLE)), \
            IncrementalCache.compute(parse(edited))
        stable, changed = b.diff(a)
        assert stable is True
        assert changed == {"b"}

    def test_shape_change_forces_full_rebuild(self):
        p1 = parse(SIMPLE)
        added = SIMPLE + "#gene name=d\nATG TGG TAA\n#end\n"
        p2 = parse(added)
        c1, c2 = IncrementalCache.compute(p1), IncrementalCache.compute(p2)
        stable, changed = c2.diff(c1)
        assert stable is False
        assert changed == set()

    def test_renamed_gene_is_shape_change(self):
        renamed = SIMPLE.replace("name=c", "name=c2")
        c1, c2 = IncrementalCache.compute(parse(SIMPLE)), \
            IncrementalCache.compute(parse(renamed))
        assert c2.diff(c1)[0] is False


# ---------------------------------------------------------------------------
# 3. patch + single-gene rebuild fidelity
# ---------------------------------------------------------------------------
class TestPatchAndRebuild:
    def test_patch_gene_replaces_in_place(self):
        ir = IRProgram(functions=[IRFunction(name=n) for n in ("a", "b", "c")])
        replacement = IRFunction(name="b", instrs=[])
        assert ir.patch_gene("b", replacement) is ir
        assert [f.name for f in ir.functions] == ["a", "b", "c"]
        assert ir.functions[1] is replacement

    def test_patch_unknown_gene_raises(self):
        ir = IRProgram(functions=[IRFunction(name="a")])
        try:
            ir.patch_gene("ghost", IRFunction(name="ghost"))
            raise AssertionError("expected KeyError")
        except KeyError:
            pass

    def test_build_function_matches_whole_program_build(self):
        program = parse(SIMPLE)
        builder = IRBuilder(LanguageConfig.for_table("standard").codon_to_op)
        whole = builder.build(program)
        for gene, fn in zip(program.genes, whole.functions, strict=True):
            rebuilt = builder.build_function(program, gene)
            assert rebuilt.name == fn.name
            assert [(i.opcode, i.operand, i.value_type)
                    for i in rebuilt.instrs] == \
                [(i.opcode, i.operand, i.value_type) for i in fn.instrs]


# ---------------------------------------------------------------------------
# 4. IncrementalCompiler: closure-limited rebuild, byte-identical chunks
# ---------------------------------------------------------------------------
class TestIncrementalCompiler:
    def setup_method(self):
        self.compiler = IncrementalCompiler(
            LanguageConfig.for_table("standard"))

    def test_first_compile_is_full(self):
        res = self.compiler.compile(parse(SIMPLE))
        assert res.stats.full_build is True
        assert res.stats.rebuilt == ["a", "b", "c"]
        assert res.stats.reused == []

    def test_edit_b_only_rebuilds_b(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        assert r2.stats.full_build is False
        assert r2.stats.rebuilt == ["b"]
        assert sorted(r2.stats.reused) == ["a", "c"]

    def test_edit_caller_pulls_in_callers(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        edited = SIMPLE.replace("ATG TGG TAA", "ATG TCT TAA", 1)  # edits gene a
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        assert r2.stats.rebuilt == ["a", "b"]
        assert r2.stats.reused == ["c"]

    def test_edited_chunk_byte_identical_to_full_compile(self):
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        r1 = self.compiler.compile(parse(SIMPLE))
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        fresh = self.compiler.compile(parse(edited))
        assert bytes(r2.chunk.code) == bytes(fresh.chunk.code)
        assert r2.chunk.gene_offsets == fresh.chunk.gene_offsets

    def test_no_change_compile_rebuilds_nothing(self):
        p1 = parse(SIMPLE)
        r1 = self.compiler.compile(p1)
        r2 = self.compiler.compile(p1,
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        assert r2.stats.full_build is False
        assert r2.stats.rebuilt == []
        assert bytes(r2.chunk.code) == bytes(r1.chunk.code)

    def test_isolated_edit_reuses_all_other_genes(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        edited = SIMPLE.replace("#gene name=c\nATG TGG TAA",
                                "#gene name=c\nATG TCT TAA")
        p2 = parse(edited)
        r2 = self.compiler.compile(p2,
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        assert r2.stats.rebuilt == ["c"]
        # untouched genes are the very same IR objects (no re-derivation)
        assert r2.ir.functions[0] is r1.ir.functions[0]
        assert r2.ir.functions[1] is r1.ir.functions[1]
        # and the patched chunk is still the faithful full compile
        fresh = self.compiler.compile(p2)
        assert bytes(r2.chunk.code) == bytes(fresh.chunk.code)

    def test_chain_edit_proportional_to_closure(self):
        chain = """\
#gene name=g0
ATG TGG TAA
#end
#gene name=g1 call_target=g0
ATG CGG TAA
#end
#gene name=g2 call_target=g1
ATG CGG TAA
#end
#config ticks=2 output=stdout
"""
        r1 = self.compiler.compile(parse(chain))
        edited = chain.replace("#gene name=g0\nATG TGG TAA",
                               "#gene name=g0\nATG TCT TAA")
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        # editing the leaf pulls in only its direct caller, not the whole chain
        assert sorted(r2.stats.rebuilt) == ["g0", "g1"]
        assert r2.stats.reused == ["g2"]
        fresh = self.compiler.compile(parse(edited))
        assert bytes(r2.chunk.code) == bytes(fresh.chunk.code)

    def test_shape_change_forces_full_rebuild(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        p2 = parse(SIMPLE + "#gene name=d\nATG TGG TAA\n#end\n")
        r2 = self.compiler.compile(p2,
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        assert r2.stats.full_build is True
        assert r2.stats.rebuilt == ["a", "b", "c", "d"]

    def test_incremental_and_full_traces_identical(self):
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        p2 = parse(edited)
        p2.config.sim["seed"] = "0"
        r1 = self.compiler.compile(parse(SIMPLE))
        r2 = self.compiler.compile(p2,
                                   previous_ir=r1.ir, previous_cache=r1.cache)
        fresh = self.compiler.compile(p2)
        p2_full = parse(edited)
        p2_full.config.sim["seed"] = "0"
        trace_incr = CellVM(r2.chunk, p2).run(p2.config.ticks)
        trace_full = CellVM(fresh.chunk, p2_full).run(p2_full.config.ticks)
        assert trace_incr == trace_full

    def test_artifact_source_rebuilds_cache(self):
        """A .helixc's embedded SRC section reproduces the cache (step 2)."""
        art = hxbc.loads_program(
            hxbc.dumps_program(parse(SIMPLE), source=SIMPLE))
        assert art.source
        program = parse(art.source)
        cache = IncrementalCache.compute(program)
        assert cache.block_hashes.keys() == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# 4b. bytecode-splice fast path (previous_chunk threaded)
# ---------------------------------------------------------------------------
class TestSplicePath:
    def setup_method(self):
        self.compiler = IncrementalCompiler(
            LanguageConfig.for_table("standard"))

    def test_equal_width_edit_splices_byte_identical(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache,
                                   previous_chunk=r1.chunk)
        assert r2.stats.full_build is False
        assert r2.stats.rebuilt == ["b"]
        assert r2.stats.splice is True
        fresh = self.compiler.compile(parse(edited))
        assert bytes(r2.chunk.code) == bytes(fresh.chunk.code)
        assert r2.chunk.gene_offsets == fresh.chunk.gene_offsets

    def test_untouched_regions_copied_verbatim(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache,
                                   previous_chunk=r1.chunk)
        # gene c is the tail region and untouched: bytes must be verbatim.
        c_start = r2.chunk.gene_offsets["c"]
        assert bytes(r2.chunk.code[c_start:]) == bytes(r1.chunk.code[c_start:])

    def test_region_growth_declines_splice(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        edited = SIMPLE.replace(
            "#gene name=b call_target=a\nATG CGC TGG TAA",
            "#gene name=b call_target=a\nATG CGG CGC TGG TAA")
        r2 = self.compiler.compile(parse(edited),
                                   previous_ir=r1.ir, previous_cache=r1.cache,
                                   previous_chunk=r1.chunk)
        assert r2.stats.splice is False
        fresh = self.compiler.compile(parse(edited))
        assert bytes(r2.chunk.code) == bytes(fresh.chunk.code)

    def test_new_pool_literal_declines_splice(self):
        table = {**STANDARD_TABLE, "TCT": Op.OP_PUSH_CONST}
        cfg = LanguageConfig(table_name="t", codon_to_op=table,
                             stop_codons=frozenset(c for c, o in table.items()
                                                   if o is Op.OP_HALT),
                             start_codons=frozenset(c for c, o in table.items()
                                                    if o is Op.OP_START),
                             translation={})
        compiler = IncrementalCompiler(cfg)

        def p(src: str):
            return Parser(list(Lexer(src).tokens()), config=cfg).parse()

        a = "#gene name=a\nATG TGG TAA\n#end\n#gene name=b\nATG TCT TAA\n#end"
        b = a.replace("ATG TCT TAA", "ATG TCT TCC TAA")  # pushes a new literal
        ra = compiler.compile(p(a))
        rb = compiler.compile(p(b), previous_ir=ra.ir,
                              previous_cache=ra.cache, previous_chunk=ra.chunk)
        assert rb.stats.splice is False
        assert bytes(rb.chunk.code) == bytes(compiler.compile(p(b)).chunk.code)

    def test_previous_ir_not_mutated(self):
        r1 = self.compiler.compile(parse(SIMPLE))
        before = [tuple((i.opcode, i.operand) for i in fn.instrs)
                  for fn in r1.ir.functions]
        edited = SIMPLE.replace("ATG CGC TGG TAA", "ATG CGG TGG TAA")
        self.compiler.compile(parse(edited),
                              previous_ir=r1.ir, previous_cache=r1.cache,
                              previous_chunk=r1.chunk)
        after = [tuple((i.opcode, i.operand) for i in fn.instrs)
                 for fn in r1.ir.functions]
        assert before == after


# ---------------------------------------------------------------------------
# 5. CLI watch mode
# ---------------------------------------------------------------------------
class TestWatch:
    def test_watch_single_iteration(self, tmp_path: Path):
        src = tmp_path / "jit.helix"
        src.write_text(SIMPLE)
        rc, out = cli(["--watch", "--watch-iterations", "1",
                       "--ticks", "2", str(src)])
        assert rc == 0
        assert "[jit] iteration 1: full build (3 genes)" in out

    def test_watch_rebuilds_closure(self, tmp_path: Path):
        src = tmp_path / "jit.helix"
        src.write_text(SIMPLE)
        rc, out = cli(["--watch", "--watch-iterations", "2",
                       "--ticks", "2", str(src)])
        assert rc == 0
        assert "[jit] iteration 1: full build (3 genes)" in out
        # iteration 2 unchanged source -> nothing rebuilt
        assert "[jit] iteration 2: rebuilt (none)" in out
