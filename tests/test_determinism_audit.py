"""Tests for doc/34 P0: determinism audit.

Verifies that simulations produce identical results when run with the same seed.
"""
from __future__ import annotations

from helixlang.core.bytecode import OPCODE_VERSION


class TestBytecodeABIVersion:
    """P0.1: Bytecode ABI is versioned and frozen."""

    def test_opcode_version_exists(self) -> None:
        assert isinstance(OPCODE_VERSION, int)

    def test_opcode_version_is_one(self) -> None:
        assert OPCODE_VERSION == 1

    def test_opcode_version_in_bytecode_module(self) -> None:
        import helixlang.core.bytecode as bc
        assert hasattr(bc, "OPCODE_VERSION")

    def test_hxbc_format_version_matches(self) -> None:
        from helixlang.core.hxbc import FORMAT_VERSION
        assert FORMAT_VERSION >= OPCODE_VERSION


class TestDeterminismFBA:
    """FBA is deterministic — same model, same result."""

    def test_fba_deterministic(self) -> None:
        from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis

        results = []
        for _ in range(3):
            fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
            fba.set_uptake("GLC", 10.0)
            r = fba.solve(objective="biomass")
            results.append(r.get("BIOMASS", 0.0))

        assert results[0] == results[1] == results[2], (
            f"FBA not deterministic: {results}"
        )

    def test_fba_nonzero_growth(self) -> None:
        from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis

        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        r = fba.solve(objective="biomass")
        assert r.get("BIOMASS", 0.0) > 0.0


class TestDeterminismGRN:
    """GRN is deterministic for same input."""

    def test_grn_deterministic(self) -> None:
        from helixlang.plugins.runtime.grn import GRN

        edges = [("A", "B", -0.8), ("B", "C", 0.5)]
        results = []
        for _ in range(3):
            grn = GRN()
            grn.add_gene("A", threshold=0.5, initial_level=1.0)
            grn.add_gene("B", threshold=0.5, initial_level=0.0)
            grn.add_gene("C", threshold=0.5, initial_level=0.0)
            for src, tgt, strength in edges:
                grn.add_edge(src, tgt, strength)
            grn.step()
            levels = {name: node.level for name, node in grn.nodes.items()}
            results.append(levels)

        assert results[0] == results[1] == results[2]


class TestDeterminismVM:
    """Bytecode VM is deterministic with same seed."""

    def test_vm_deterministic_same_seed(self) -> None:
        from helixlang.core.ast_nodes import Program
        from helixlang.core.bytecode import Chunk
        from helixlang.core.vm import CellVM

        def _build() -> tuple[Chunk, Program]:
            c = Chunk()
            prog = Program()
            return c, prog

        results = []
        for _ in range(3):
            c, p = _build()
            vm = CellVM(c, p)
            results.append(vm.tick)

        assert all(r == results[0] for r in results), (
            f"VM not deterministic: {results}"
        )

    def test_vm_produces_same_trace(self) -> None:
        """Two VM instances with same bytecode produce identical traces."""
        from helixlang.core.codon_table import STANDARD_TABLE, Op
        from helixlang.core.compiler import Compiler
        from helixlang.core.lexer import Lexer
        from helixlang.core.parser import Parser
        from helixlang.core.vm import CellVM

        src = "#gene name=g1\nATG GCT TAA\n#end\n#config ticks=3"
        stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
        toks = list(Lexer(src).tokens())
        prog = Parser(toks, stop_codons=stop).parse()
        chunk = Compiler(STANDARD_TABLE).compile(prog)

        vm1 = CellVM(chunk, prog)
        vm1.run(3)
        t1 = [dict(s) for s in vm1.trace]

        vm2 = CellVM(chunk, prog)
        vm2.run(3)
        t2 = [dict(s) for s in vm2.trace]

        assert t1 == t2, "Two VMs with same code produced different traces"


class TestDeterminismStochastic:
    """Stochastic module is deterministic with same seed."""

    def test_gillespie_telegraph_deterministic(self) -> None:
        from helixlang.plugins.runtime.stochastic import gillespie_telegraph

        results = []
        for _ in range(3):
            r = gillespie_telegraph(
                k_on=0.1, k_off=0.05, burst_size=1.0,
                degradation_rate=0.02, t_max=10.0, seed=42,
            )
            results.append(r)

        assert results[0] == results[1] == results[2]
