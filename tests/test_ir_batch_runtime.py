"""Direct unit coverage of the vectorised batch runtime internals.

These tests drive ``BatchRuntime`` and the private ``_vector_apply`` /
``_scalar_step`` / ``_scalar_pop_frames`` / ``_make_engine`` kernels directly
with controlled inputs to close residual branch coverage of
:mod:`helixlang.core.ir_batch_runtime` (stack over/underflow, engine
selection, central-dogma tick, run-loop termination, operand-byte width).
"""
import numpy as np
import pytest
from test_ir import STANDARD_TABLE, TEST_TABLE, build_ir

from helixlang.core.codon_table import Op
from helixlang.core.ir import IRInst
from helixlang.core.ir_batch_runtime import (
    BatchRuntime,
    StackDepthError,
    _make_engine,
    _NumpyEngine,
    _op_operand_bytes,
    _operand_bytes,
    _vector_apply,
)
from helixlang.core.vm import Frame


def _rt(n=3, src="#gene name=g\nATG GCT TAA\n#end", table=STANDARD_TABLE):
    ir, prog = build_ir(src, table)
    return ir, prog, BatchRuntime(ir, prog, n=n)


# ── engine selection ───────────────────────────────────────────────────────
class TestMakeEngine:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            _make_engine("quux", 3)

    def test_jax_engine_selected(self):
        try:
            import jax.numpy  # noqa: F401
        except ImportError:
            pytest.skip("jax not installed")
        eng, family, active = _make_engine("jax", 2)
        assert family == "jax"
        assert active == "jax"
        assert eng.__class__.__name__ == "_JAXEngine"

    def test_numpy_engine_selected(self):
        eng, family, active = _make_engine("numpy", 2)
        assert family == "numpy"
        assert eng.__class__.__name__ == "_NumpyEngine"

    def test_jax_import_error_falls_back_to_numpy(self):
        import builtins
        from unittest import mock

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "jax" or name.startswith("jax."):
                raise ImportError("jax unavailable for test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            eng, family, active = _make_engine("jax", 2)
        assert family == "numpy"
        assert active == "numpy"
        assert eng.__class__.__name__ == "_NumpyEngine"


# ── engine methods unit coverage ───────────────────────────────────────────
class TestEngines:
    def test_numpy_load_rows(self):
        eng = _NumpyEngine(2)
        full = np.zeros((2, 512))
        full[0, 0] = 5.0
        full[0, 1] = 6.0
        full[1, 0] = 7.0
        eng.store_rows(np.array([0, 1]), full)
        out = eng.load_rows(np.array([0, 1]))
        np.testing.assert_array_equal(out, full)
        assert eng.load_rows(np.array([1]))[0, 0] == 7.0

    def test_jax_engine_methods(self):
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("jax not installed")
        from helixlang.core.ir_batch_runtime import _JAXEngine

        eng = _JAXEngine(2, 4)
        rows = jnp.array([0, 1])
        eng.write_col(rows, 0, jnp.array([5.0, 7.0]))
        eng.write_col(rows, 1, jnp.array([6.0, 8.0]))
        eng.swap_cols(rows, 0, 1)
        np.testing.assert_array_equal(
            np.asarray(eng.read_col(rows, 0)), np.array([6.0, 8.0]))
        np.testing.assert_array_equal(
            np.asarray(eng.read_col(rows, 1)), np.array([5.0, 7.0]))
        eng.set_depth(rows, jnp.array([2, 2]))
        eng.bump_depth(rows, -1)
        np.testing.assert_array_equal(np.asarray(eng.depth), np.array([1, 1]))
        eng.store_rows(rows, jnp.ones((2, 4)))
        np.testing.assert_array_equal(
            np.asarray(eng.load_rows(rows)), np.ones((2, 4)))


# ── module-level vector kernel: under/overflow branches ────────────────────
class TestVectorApplyErrors:
    def test_push_const_none_operand(self):
        eng = _NumpyEngine(2)
        with pytest.raises(StackDepthError):
            _vector_apply(eng, Op.OP_PUSH_CONST, None, np.array([0, 1]))

    def test_push_const_over_bank(self):
        eng = _NumpyEngine(2)
        eng.set_depth(np.array([0, 1], dtype=np.int64), np.array([511, 511]))
        with pytest.raises(StackDepthError):
            _vector_apply(eng, Op.OP_PUSH_CONST, 3, np.array([0, 1]))

    def test_dup_underflow(self):
        eng = _NumpyEngine(2)
        eng.set_depth(np.array([0, 1], dtype=np.int64), np.array([0, 0]))
        with pytest.raises(StackDepthError):
            _vector_apply(eng, Op.OP_DUP, None, np.array([0, 1]))

    def test_swap_underflow(self):
        eng = _NumpyEngine(2)
        eng.set_depth(np.array([0, 1], dtype=np.int64), np.array([1, 1]))
        with pytest.raises(StackDepthError):
            _vector_apply(eng, Op.OP_SWAP, None, np.array([0, 1]))


class TestVectorApplyArith:
    def test_mul_branch(self):
        eng = _NumpyEngine(2)
        eng.set_depth(np.array([0, 1], dtype=np.int64), np.array([2, 2]))
        eng.write_col(np.array([0, 1]), 0, np.array([3.0, 4.0]))
        eng.write_col(np.array([0, 1]), 1, np.array([5.0, 6.0]))
        _vector_apply(eng, Op.OP_MUL, None, np.array([0, 1]))
        np.testing.assert_array_equal(eng.S[0, :2], np.array([15.0, 5.0]))
        np.testing.assert_array_equal(eng.S[1, :2], np.array([24.0, 6.0]))
        np.testing.assert_array_equal(eng.depth, np.array([1, 1]))

    def test_add_sub_lt(self):
        cases = [(Op.OP_ADD, np.array([8.0, 10.0])),
                 (Op.OP_SUB, np.array([-2.0, -2.0])),
                 (Op.OP_LT, np.array([1.0, 1.0]))]
        for op, expected in cases:
            eng = _NumpyEngine(2)
            eng.set_depth(np.array([0, 1], dtype=np.int64), np.array([2, 2]))
            eng.write_col(np.array([0, 1]), 0, np.array([3.0, 4.0]))
            eng.write_col(np.array([0, 1]), 1, np.array([5.0, 6.0]))
            _vector_apply(eng, op, None, np.array([0, 1]))
            np.testing.assert_allclose(eng.S[:, 0], expected)

    def test_not_branch(self):
        eng = _NumpyEngine(3)
        eng.set_depth(np.array([0, 1, 2], dtype=np.int64), np.array([1, 1, 1]))
        eng.write_col(np.array([0, 1, 2]), 0, np.array([0.0, 3.0, 0.0]))
        _vector_apply(eng, Op.OP_NOT, None, np.array([0, 1, 2]))
        np.testing.assert_array_equal(eng.S[:, 0], np.array([1.0, 0.0, 1.0]))

    def test_push_const_ok_and_pop(self):
        eng = _NumpyEngine(2)
        _vector_apply(eng, Op.OP_PUSH_CONST, 4, np.array([0, 1]))
        np.testing.assert_array_equal(eng.S[:, 0], np.array([4.0, 4.0]))
        _vector_apply(eng, Op.OP_POP, None, np.array([0, 1]))
        np.testing.assert_array_equal(eng.depth, np.array([0, 0]))


# ── BatchRuntime construction guard ────────────────────────────────────────
class TestConstructor:
    def test_n_zero_raises(self):
        ir, prog = build_ir("#gene name=g\nATG GCT TAA\n#end", STANDARD_TABLE)
        with pytest.raises(ValueError):
            BatchRuntime(ir, prog, n=0)


# ── run loop termination / central dogma ───────────────────────────────────
class TestRunLoop:
    def test_break_when_all_dead(self):
        # AAA -> OP_DIE: cells die, then the run loop breaks early.
        ir, prog, rt = _rt(n=3, src="#gene name=g\nATG AAA TAA\n#end",
                           table=STANDARD_TABLE)
        traces = rt.run(20)
        assert all(not c.cell.alive for c in rt.cells)
        assert all(len(t) < 20 for t in traces)

    def test_central_dogma_tick(self):
        ir, prog, rt = _rt(
            n=2, src="#config use_central_dogma=true\n"
                     "#gene name=g\nATG GCT TAA\n#end",
            table=STANDARD_TABLE)
        assert prog.config.use_central_dogma is True
        traces = rt.run(4)
        assert len(traces) == 2
        for t in traces:
            assert len(t) == 4


# ── execute_pending_all: 'end' key / scalar pop frames ─────────────────────
class TestExecutePendingEnd:
    def test_end_key_scalar_pop_frames(self):
        ir, prog, rt = _rt(n=2, src=(
            "#gene name=a\nATG GCT TAA\n#end\n"
            "#gene name=b\nATG GCT TAA\n#end"))
        flat = len(rt.cells[0]._flat)
        rt._quotas = [64, 64]
        # cell 0: finished function, single frame -> pop to empty (no ip set)
        rt.cells[0].frames.append(Frame(return_ip=0, gene_name="a"))
        rt.cells[0].ip = flat
        # cell 1: finished function, two frames -> pop keep one, restore ip
        rt.cells[1].frames.append(Frame(return_ip=1, gene_name="a"))
        rt.cells[1].frames.append(Frame(return_ip=9, gene_name="b"))
        rt.cells[1].ip = flat
        rt._execute_pending_all()
        assert len(rt.cells[0].frames) == 0
        # cell1's top frame was popped and ip restored; remaining frame then
        # again sits at the (still past-end) ip and is popped too
        assert len(rt.cells[1].frames) == 0


class TestScalarPopFrames:
    def test_pop_with_and_without_frame(self):
        ir, prog, rt = _rt(n=2)
        rt.cells[0].frames.append(Frame(return_ip=3, gene_name="a"))
        rt.cells[0].frames.append(Frame(return_ip=10, gene_name="b"))
        rt.cells[0].ip = 99
        rt.cells[1].frames.append(Frame(return_ip=1, gene_name="a"))
        rt.cells[1].ip = 50
        rt._scalar_pop_frames([0, 1])
        assert len(rt.cells[0].frames) == 1
        assert rt.cells[0].ip == 3
        assert len(rt.cells[1].frames) == 0
        assert rt.cells[1].ip == 50


class TestScalarStep:
    def test_frames_over_256_cleared(self):
        ir, prog, rt = _rt(n=1,
                           src="#gene name=g\nATG GCT TAA\n#end")
        c = rt.cells[0]
        c.frames = [Frame(return_ip=0, gene_name="a") for _ in range(300)]
        c.ip = 1  # GCT instruction index
        rt._scalar_step(0)
        assert c.frames == []
        assert c.ip == 2
        # _scalar_step sets _current_inst / _current_operand_bytes each time
        assert c._current_inst is not None


# ── operand byte width ─────────────────────────────────────────────────────
class TestOperandBytes:
    def test_two_byte_operand(self):
        inst = IRInst(Op.OP_CALL_GENE, operand=0xABCD)
        assert _op_operand_bytes(Op.OP_CALL_GENE) >= 2
        assert _operand_bytes(inst) == [0xAB, 0xCD]

    def test_one_byte_operand(self):
        inst = IRInst(Op.OP_PUSH_CONST, operand=0x07)
        assert _operand_bytes(inst) == [0x07]

    def test_non_int_operand(self):
        inst = IRInst(Op.OP_CALL_GENE, operand="g")
        assert _operand_bytes(inst) == []


# ── demonstration of engine-driven vector ops through the batch runtime ────
class TestVectorBackend:
    def test_runtime_uses_numpy_backend_for_vector_ops(self):
        # A pure-arithmetic cohort exercises the vectorised kernel end to end.
        ir, prog, rt = _rt(n=3, src=(
            "#gene name=g\nATG TCT TCC CCA GAA TAA\n#end"), table=TEST_TABLE)
        assert rt.active_backend == "numpy"
        traces = rt.run(20)
        assert len(traces) == 3
