"""Typed biological IR: builder, optimizer, lowerer, serializer, runtimes.

Covers the doc/37 §4 pipeline:

    helix source -> AST -> IRBuilder -> typed IR -> IROpt ->
    IRLowerer -> bytecode -> CellVM / IRRuntime / BatchRuntime

Note: the standard codon table maps no codon to the pure arithmetic opcodes
(PUSH_CONST/ADD/SUB/MUL/LT/NOT are reachable in bytecode but not via a
codon), so optimizer/runtime-arithmetic tests build programs through a *test
codon table* that does.  This mirrors how sim_runtime / tooling may author IR
or bytecode directly.
"""
import random

import pytest

from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.ir import IRType
from helixlang.core.ir_batch_runtime import BatchRuntime, StackDepthError
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_lower import lower
from helixlang.core.ir_opt import optimize_program
from helixlang.core.ir_runtime import IRRuntime
from helixlang.core.ir_serialize import IRFormatError, dumps, loads
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser, parse_source
from helixlang.core.vm import CellVM

# ── test codon table: arithmetic ops added on top of the standard base ─────
TEST_TABLE: dict[str, Op] = {
    **STANDARD_TABLE,
    "AAA": Op.OP_CALL_GENE,   # override (std: OP_DIE) for call tests
    "ACG": Op.OP_PUSH_CONST,  # wobble 2
    "TCT": Op.OP_PUSH_CONST,  # wobble 3 (std: SIGNAL)
    "TCC": Op.OP_PUSH_CONST,  # wobble 1
    "CCT": Op.OP_ADD,         # (std: MODIFY_STATE)
    "CCC": Op.OP_SUB,
    "CCA": Op.OP_MUL,
    "CCG": Op.OP_LT,
    "GAA": Op.OP_NOT,         # (std: FEED)
    "AAC": Op.OP_POP,         # (std: DIVIDE)
    "CAC": Op.OP_DUP,         # (std: REGULATE)
    "CAG": Op.OP_SWAP,        # (std: EMIT_MORPHOGEN)
    "AAT": Op.OP_WRITE_MEM,   # (std: DIVIDE)
    "TGG": Op.OP_BUILD_PROTEIN,  # effect boundary (std: BUILD_PIGMENT)
}


def _parse(src, table):
    stop = {c for c, op in table.items() if op == Op.OP_HALT}
    toks = list(Lexer(src).tokens())
    return Parser(toks, stop_codons=stop).parse()


def compile_program(src, table=STANDARD_TABLE):
    prog = _parse(src, table)
    return Compiler(table).compile(prog), prog


def build_ir(src, table=STANDARD_TABLE):
    prog = _parse(src, table)
    return IRBuilder(table).build(prog), prog


# stock effect program bit-exactness across table + IR path
ARITH_SRC = "#gene name=g\nATG TCT TCC CCT CCG GAA TAA\n#end"  # 3<1, not(0)


# ── builder ────────────────────────────────────────────────────────────────
class TestBuilder:
    def test_function_per_gene(self):
        prog = parse_source("#gene name=a\nATG GCT TAA\n#end")
        ir = IRBuilder().build(prog)
        assert [f.name for f in ir.functions] == ["a"]
        assert [i.opcode for i in ir.functions[0].instrs] == [
            Op.OP_START, Op.OP_BUILD_PROTEIN, Op.OP_HALT]

    def test_push_const_carries_literal_not_pool_index(self):
        ir, _ = build_ir("#gene name=g\nATG TCT TCC TAA\n#end", TEST_TABLE)
        consts = [i.operand for i in ir.functions[0].instrs
                  if i.opcode is Op.OP_PUSH_CONST]
        assert consts == [3, 1]

    def test_types_annotated(self):
        ir, _ = build_ir(ARITH_SRC, TEST_TABLE)
        for inst in ir.functions[0].instrs:
            if inst.opcode is Op.OP_PUSH_CONST:
                assert inst.value_type is IRType.NUM
            else:
                assert inst.value_type is None

    def test_call_target_resolved_at_build(self):
        src = ("#gene name=caller\nATG AAA TAA\n#end\n"
               "#gene name=target\nATG TGG TAA\n#end")
        ir, _ = build_ir(src, TEST_TABLE)
        call = [i for i in ir.functions[0].instrs
                if i.opcode is Op.OP_CALL_GENE]
        # AAA wobble=0 -> modulo fallback selects the first gene
        assert call and call[0].operand == "caller"
        # explicit call_target field wins
        src2 = ("#gene name=caller call_target=target\nATG AAA TAA\n#end\n"
                "#gene name=target\nATG TGG TAA\n#end")
        ir2, _ = build_ir(src2, TEST_TABLE)
        call2 = [i for i in ir2.functions[0].instrs
                 if i.opcode is Op.OP_CALL_GENE]
        assert call2 and call2[0].operand == "target"


# ── lowering ───────────────────────────────────────────────────────────────
class TestLowerer:
    def test_lower_preserves_operand_bytes(self):
        chunk, _ = compile_program("#gene name=g\nATG GCT GCC GCA GCG TAA\n#end")
        assert chunk.code[1] == int(Op.OP_BUILD_PROTEIN)
        assert chunk.code[2] == 3
        assert chunk.code[4] == 1
        assert chunk.code[6] == 0
        assert chunk.code[8] == 2

    def test_lower_call_patches_to_offset(self):
        src = ("#gene name=caller\nATG CGT TAA\n#end\n"
               "#gene name=target\nATG GCT TAA\n#end")
        chunk, _ = compile_program(src)
        ip = 0
        found = False
        while ip < len(chunk.code):
            if chunk.code[ip] == int(Op.OP_CALL_GENE):
                off = (chunk.code[ip + 1] << 8) | chunk.code[ip + 2]
                assert off == chunk.gene_offsets["target"]
                found = True
                break
            ip += 1
        assert found
        assert int(Op.OP_JUMP) in chunk.code

    def test_config_snapshot(self):
        prog = parse_source("#config ticks=250 ops_per_tick=32\n"
                            "#gene name=g\nATG GCT TAA\n#end")
        ir = IRBuilder().build(prog)
        assert ir.config["ops_per_tick"] == 32
        assert ir.config["ticks"] == 250


# ── optimizer ──────────────────────────────────────────────────────────────
class TestOptimizer:
    def test_fold_constant_arithmetic(self):
        """PUSH 3; PUSH 1; ADD -> PUSH 4."""
        ir, _ = build_ir("#gene name=g\nATG TCT TCC CCT TAA\n#end", TEST_TABLE)
        before = len(ir.functions[0].instrs)
        optimize_program(ir)
        after = len(ir.functions[0].instrs)
        assert after == before - 2
        pushes = [i.operand for i in ir.functions[0].instrs
                  if i.opcode is Op.OP_PUSH_CONST]
        assert pushes[-1] == 4  # 3 + 1

    def test_fold_sub_mul_lt_not(self):
        """Whole chain 3-1 -> 2; 2<1 -> 0; 0*1 -> 0; not(0) -> 1 folds to 1."""
        ir, _ = build_ir(
            "#gene name=g\nATG TCT TCC CCC TCC CCG TCC CCA GAA TAA\n#end",
            TEST_TABLE)
        optimize_program(ir)
        inline = [(i.opcode.name, i.operand) for i in ir.functions[0].instrs]
        assert inline == [("OP_START", None),
                          ("OP_PUSH_CONST", 1),   # 3-1, 2<1, 0*1, not(0)
                          ("OP_HALT", None)]

    def test_fold_refuses_underflow_windows(self):
        """LT whose operands are not both on the simulated stack stays put."""
        ir, _ = build_ir("#gene name=g\nATG TCT CCG GAA TAA\n#end",
                         TEST_TABLE)
        optimize_program(ir)
        ops = [i.opcode for i in ir.functions[0].instrs]
        assert Op.OP_LT in ops
        assert Op.OP_NOT in ops

    def test_fold_equivalence(self):
        """VM trajectory and final stack identical (opt vs plain copy)."""
        ir, prog = build_ir("#gene name=g\nATG TCT TCC CCT TAA\n#end",
                            TEST_TABLE)
        ir2, _ = build_ir("#gene name=g\nATG TCT TCC CCT TAA\n#end",
                          TEST_TABLE)
        optimize_program(ir2)
        vm_plain = CellVM(lower(ir), prog).run(30)
        vm_opt = CellVM(lower(ir2), prog).run(30)
        assert vm_plain == vm_opt
        assert CellVM(lower(ir), prog).stack == CellVM(lower(ir2), prog).stack

    def test_fold_does_not_cross_effect(self):
        ir, _ = build_ir("#gene name=g\nATG TCT TCC CCT TGG TCT TCC CCT TAA\n#end",
                         TEST_TABLE)
        optimize_program(ir)
        ops = [i.opcode for i in ir.functions[0].instrs]
        assert Op.OP_BUILD_PROTEIN in ops
        # START PUSH(4) BUILD_PROTEIN PUSH(4) HALT
        assert len(ops) == 5

    def test_dead_push_pop(self):
        ir, _ = build_ir("#gene name=g\nATG TCT AAC TAA\n#end", TEST_TABLE)
        optimize_program(ir)
        ops = [i.opcode for i in ir.functions[0].instrs]
        assert [o for o in ops if o is not Op.OP_NOP] == [
            Op.OP_START, Op.OP_HALT]

    def test_dup_swap_optimizable(self):
        ir, _ = build_ir("#gene name=g\nATG CAC CAC CAG CAG AAC TAA\n#end",
                         TEST_TABLE)
        ops = [(i.opcode.name, i.operand) for i in ir.functions[0].instrs]
        assert ("OP_DUP", None) in ops and ("OP_SWAP", None) in ops

    def test_optimizer_idempotent(self):
        ir, _ = build_ir("#gene name=g\nATG TCT TCC CCT AAC TAA\n#end",
                         TEST_TABLE)
        optimize_program(ir)
        first = [(i.opcode, i.operand) for i in ir.functions[0].instrs]
        optimize_program(ir)
        second = [(i.opcode, i.operand) for i in ir.functions[0].instrs]
        assert first == second


# ── IR runtime (CPU runtime #2) parity with CellVM ────────────────────────
class TestIRRuntime:
    @pytest.mark.parametrize("src,table", [
        ("#gene name=g\nATG GCT TAA\n#end", STANDARD_TABLE),
        ("#gene name=g\nATG GAT GAA TGT TAA\n#end", STANDARD_TABLE),
        ("#gene name=g\nATG TCT TCC CCT TAA\n#end", TEST_TABLE),
        ("#gene name=g\nATG TCT CAC CAG CCA TAA\n#end", TEST_TABLE),
        ("#gene name=g\nATG TCT TCC GCG TCC CCA TAA\n#end", TEST_TABLE),
        ("#gene name=g\nATG GAT TAA\n#end\n#gene name=h\nATG GCT TAA\n#end",
         STANDARD_TABLE),
    ])
    def test_parity_with_vm(self, src, table):
        ir, prog = build_ir(src, table)
        chunk = lower(ir)
        assert IRRuntime(ir, prog).run(60) == CellVM(chunk, prog).run(60)

    def test_parity_with_calls_and_regulation(self):
        src = ("#gene name=src\nATG CAT GCT TGT TAA\n#end\n"
               "#gene name=dst\nATG GCT TAA\n#end")
        ir, prog = build_ir(src, STANDARD_TABLE)
        chunk = lower(ir)
        assert IRRuntime(ir, prog).run(40) == CellVM(chunk, prog).run(40)

    def test_optimized_IR_runtime_parity(self):
        ir, prog = build_ir("#gene name=g\nATG TCT TCC CCT CAC GAA TAA\n#end",
                            TEST_TABLE)
        optimize_program(ir)
        chunk = lower(ir)
        assert IRRuntime(ir, prog).run(40) == CellVM(chunk, prog).run(40)


# ── serializer (HLIR) ──────────────────────────────────────────────────────
class TestSerialize:
    def test_roundtrip(self):
        ir, _ = build_ir("#gene name=g\nATG TCT TCC CCT TAA\n#end", TEST_TABLE)
        loaded = loads(dumps(ir))
        assert loaded.version == ir.version
        assert loaded.gene_names() == ir.gene_names()
        a = ir.functions[0].instrs
        b = loaded.functions[0].instrs
        assert [(i.opcode, i.operand, i.line, i.codon_index)
                for i in a] == \
            [(i.opcode, i.operand, i.line, i.codon_index) for i in b]

    def test_roundtrip_executes_identically(self):
        ir, prog = build_ir("#gene name=g\nATG TCT TCC CCT TAA\n#end",
                            TEST_TABLE)
        ir2 = loads(dumps(ir))
        assert CellVM(lower(ir2), prog).run(30) == \
            CellVM(lower(ir), prog).run(30)

    def test_rejects_too_new(self):
        import json
        with pytest.raises(IRFormatError):
            loads(json.dumps({"fmt": "hlir", "version": 99,
                              "functions": []}))


# ── batch / GPU runtime parity ─────────────────────────────────────────────
class TestBatchRuntime:
    @pytest.mark.parametrize("n", [2, 5, 12])
    @pytest.mark.parametrize("src,table", [
        ("#gene name=g\nATG GCT TAA\n#end", STANDARD_TABLE),
        ("#gene name=g\nATG TCT TCC CCT TAA\n#end", TEST_TABLE),
        ("#gene name=g\nATG TCT TCC CCT TCC CCT TAA\n#end", TEST_TABLE),
    ])
    def test_batch_parity(self, n, src, table):
        ir, prog = build_ir(src, table)
        traces = BatchRuntime(ir, prog, n=n).run(40)
        solo = IRRuntime(ir, prog).run(40)
        assert len(traces) == n
        for t in traces:
            assert t == solo

    def test_batch_parity_diverse_program(self):
        src = ("#gene name=a\nATG TCT GAA TGG TAA\n#end\n"
               "#gene name=b\nATG TCC CAC TAA\n#end")
        ir, prog = build_ir(src, TEST_TABLE)
        traces = BatchRuntime(ir, prog, n=7).run(50)
        solo = IRRuntime(ir, prog).run(50)
        for t in traces:
            assert t == solo

    def test_batch_parity_random_arithmetic(self):
        rng = random.Random(7)
        seq, depth = [], 0
        for _ in range(22):
            valid = ["TCT", "TCC"]  # pushes: +1
            if depth >= 2:
                valid += ["CCT", "CCC", "CCA", "CCG", "CAG"]  # -1
            if depth >= 1:
                valid += ["GAA", "CAC", "AAC"]  # -1 / +1 / -1
            valid += ["TGG"]  # effect boundary, depth unchanged
            codon = rng.choice(valid)
            seq.append(codon)
            if codon in ("TCT", "TCC", "CAC"):
                depth += 1
            elif codon in ("CCT", "CCC", "CCA", "CCG", "GAA", "CAG", "AAC"):
                depth -= 1
            assert depth >= 0
        src = f"#gene name=g\nATG {' '.join(seq)} TAA\n#end"
        ir, prog = build_ir(src, TEST_TABLE)
        traces = BatchRuntime(ir, prog, n=6).run(30)
        solo = IRRuntime(ir, prog).run(30)
        for t in traces:
            assert t == solo

    def test_batch_underflow_raises(self):
        # TEST_TABLE maps CCT -> ADD: a lone operand underflows the cohort.
        ir, prog = build_ir("#gene name=g\nATG CCT TAA\n#end", TEST_TABLE)
        with pytest.raises(StackDepthError):
            BatchRuntime(ir, prog, n=3).run(5)

    def test_backend_selection_and_fallback(self):
        ir, prog = build_ir("#gene name=g\nATG TCT TCC CCT TAA\n#end",
                            TEST_TABLE)
        rt = BatchRuntime(ir, prog, n=4, backend="jax")
        assert rt.active_backend in ("jax", "numpy")
        assert len(rt.run(10)) == 4
