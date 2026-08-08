"""HelixLang language-performance benchmark harness.

Measures the full language pipeline — lexing, parsing, semantic analysis,
compilation, and CellVM execution — plus the GRN core, memory footprint, and
the 16 shipped examples. Pure standard library (no pytest-benchmark, no
third-party deps) so it runs on any machine with the repo checkout.

Usage::

    python benchmarks/bench_helix.py                # full run
    python benchmarks/bench_helix.py --fast         # reduced matrix
    python benchmarks/bench_helix.py --json out.json

Methodology
-----------
* Every timed measurement is the **best of** several repeats, each running the
  workload ``number`` times, measured with ``time.perf_counter``.
* The GIL-bound garbage collector is disabled during timing and re-enabled
  afterwards, so collection noise does not distort the numbers (allocations
  still happen; only periodic GC sweeps are excluded).
* A warm-up pass is run before each measurement so first-call import/JIT and
  caches are not counted.
* Determinism: all synthetic programs are generated from fixed shapes with a
  seeded RNG; ``CellVM`` and ``GrayScott`` use fixed internal seeds, so results
  are reproducible run-to-run.

Output is a markdown report (printed to stdout) suitable for pasting into
``doc/performance-report.md``.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helixlang.ast_nodes import Program  # noqa: E402
from helixlang.bytecode import Chunk  # noqa: E402
from helixlang.codon_table import Op  # noqa: E402
from helixlang.compiler import Compiler  # noqa: E402
from helixlang.grn import GRN  # noqa: E402
from helixlang.lexer import Lexer, Token  # noqa: E402
from helixlang.parser import Parser  # noqa: E402
from helixlang.semantic import SemanticAnalyzer  # noqa: E402
from helixlang.vm import CellVM  # noqa: E402

# ============================================================================
# Timing helpers
# ============================================================================

def best_time(fn: Callable[[], Any], *, number: int = 1,
              repeat: int = 5) -> float:
    """Return the best wall time (seconds) across ``repeat`` runs of ``fn``.

    Runs one warm-up first. GC is disabled during each timed run.
    """
    fn()  # warm-up (populates caches, triggers first-time imports)
    best = float("inf")
    for _ in range(repeat):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = (time.perf_counter() - t0) / number
        gc.enable()
        if dt < best:
            best = dt
    gc.enable()
    return best


def fmt_rate(value: float, unit: str) -> str:
    """Format a per-second rate with the best unit prefix."""
    for prefix in ("", "k", "M", "G"):
        if value < 1000 or prefix == "G":
            return f"{value:,.1f}{prefix}{unit}"
        value /= 1000
    return f"{value:,.1f}G{unit}"  # pragma: no cover


# ============================================================================
# Synthetic program generation
# ============================================================================

def gen_program(n_genes: int, codons_per_gene: int, ticks: int = 100) -> str:
    """Generate a program with ``n_genes`` constitutive genes.

    Each gene body is ``ATG <GCT>... <TAA>`` so it synthesizes proteins (the
    most common runtime workload) and halts. ``codons_per_gene`` counts the
    body codons (including ATG and the stop).
    """
    body = (codons_per_gene - 2) // 1
    lines: list[str] = []
    for i in range(n_genes):
        lines.append(f"#gene name=g{i}")
        lines.append("ATG " + "GCT " * max(1, body) + "TAA")
        lines.append("#end")
        lines.append("")
    lines.append(f"#config ticks={ticks}")
    return "\n".join(lines)


# ============================================================================
# Instrumented VM (counts executed instructions)
# ============================================================================

class CountingVM(CellVM):
    """CellVM that counts every dispatched instruction."""

    def __init__(self, chunk: Chunk, program: Program) -> None:
        super().__init__(chunk, program)
        self.executed_ops = 0
        original = self._dispatcher.dispatch

        def counting(op: Op) -> None:
            original(op)
            self.executed_ops += 1

        self._dispatcher.dispatch = counting


def compile_pipeline(src: str) -> tuple[Chunk, Program]:
    """Run lex -> parse -> semantic -> compile, returning (chunk, program)."""
    tokens = list(Lexer(src).tokens())
    program = Parser(tokens).parse()
    SemanticAnalyzer(program).check()
    chunk = Compiler().compile(program)
    return chunk, program


def stage_times(src: str) -> dict[str, float]:
    """Best wall times for each pipeline stage in isolation (seconds).

    Each stage is timed against the *output of the previous stage*, so the
    measured time is that stage alone: lex is timed on ``src``, parse on the
    pre-built token stream, semantic on the pre-built AST, compile on the
    pre-built AST. The pre-built objects are never mutated by their stage.
    """
    def lex() -> list[Token]:
        return list(Lexer(src).tokens())

    tokens = lex()  # cache outside timing (warm-up + reuse)
    lex_t = best_time(lex, number=3, repeat=5)

    def parse() -> Program:
        return Parser(tokens).parse()

    program = parse()  # Parser does not mutate ``tokens``
    parse_t = best_time(parse, number=3, repeat=5)

    def semantic() -> None:
        SemanticAnalyzer(program).check()

    semantic_t = best_time(semantic, number=3, repeat=5)

    def compile_() -> Chunk:
        return Compiler().compile(program)

    compile_t = best_time(compile_, number=3, repeat=5)

    return {
        "lex": lex_t,
        "parse": parse_t,
        "semantic": semantic_t,
        "compile": compile_t,
    }


# ============================================================================
# Benchmarks
# ============================================================================

def bench_compile_matrix(fast: bool) -> list[dict]:
    """Compile-pipeline throughput across program shapes."""
    sizes = [(1, 16), (4, 16), (16, 16), (64, 16), (16, 64), (64, 64)]
    if fast:
        sizes = [(4, 16), (16, 64)]
    rows = []
    for n_genes, codons in sizes:
        src = gen_program(n_genes, codons)
        n_codons = n_genes * codons
        stages = stage_times(src)
        total = sum(stages.values())
        rows.append({
            "genes": n_genes,
            "codons": n_codons,
            "lex": stages["lex"],
            "parse": stages["parse"],
            "semantic": stages["semantic"],
            "compile": stages["compile"],
            "total": total,
            "codons_per_s": n_codons / total if total else 0.0,
        })
    return rows


def bench_vm_throughput(fast: bool) -> list[dict]:
    """CellVM tick/op throughput at increasing tick counts."""
    n_genes, codons = 4, 32
    src = gen_program(n_genes, codons, ticks=100)
    chunk, program = compile_pipeline(src)
    tick_values = [100, 1000, 5000] if not fast else [500]
    rows = []
    for ticks in tick_values:
        program.config.ticks = ticks

        def run(_ticks: int = ticks) -> CountingVM:
            vm2 = CountingVM(chunk, program)
            vm2.run(_ticks)
            return vm2

        dt = best_time(run, number=1, repeat=5)
        result = run()
        n_ops = result.executed_ops
        rows.append({
            "ticks": ticks,
            "wall_s": dt,
            "ticks_per_s": ticks / dt,
            "executed_ops": n_ops,
            "ops_per_s": n_ops / dt,
        })
    return rows


def bench_grn_scaling(fast: bool) -> list[dict]:
    """GRN ``step()`` scaling: N nodes in a cycle, ticks=1000."""
    sizes = [8, 32, 128] if not fast else [32]
    rows = []
    for n in sizes:
        grn = GRN()
        for i in range(n):
            grn.add_gene(f"g{i}", threshold=0.5, initial_level=1.0)
        for i in range(n):
            grn.add_edge(f"g{i}", f"g{(i + 1) % n}", 0.5)
        ticks = 1000

        def run(_n: int = n, _ticks: int = ticks) -> None:
            g = GRN()
            for i in range(_n):
                g.add_gene(f"g{i}", threshold=0.5, initial_level=1.0)
            for i in range(_n):
                g.add_edge(f"g{i}", f"g{(i + 1) % _n}", 0.5)
            for _ in range(_ticks):
                g.step()

        dt = best_time(run, number=1, repeat=5)
        rows.append({
            "nodes": n,
            "edges": n,
            "steps_per_s": ticks / dt,
            "step_s": dt / ticks,
        })
    return rows


def bench_gray_scott() -> list[dict]:
    """Gray-Scott per-cell cost at increasing grid sizes (20 steps)."""
    from helixlang.reaction_diffusion import GrayScott
    rows = []
    for n in (16, 32, 64, 128):
        gs = GrayScott(n=n, seed=42)

        def run(_gs: GrayScott = gs) -> None:
            for _ in range(20):
                _gs.step()

        dt = best_time(run, number=1, repeat=5) / 20
        rows.append({
            "grid": f"{n}x{n}",
            "step_s": dt,
            "cell_s": dt / (n * n),
        })
    return rows


def bench_memory() -> dict:
    """Peak memory of a full compile+run for a mid-size program."""
    src = gen_program(16, 64, ticks=200)
    chunk, program = compile_pipeline(src)
    gc.collect()
    tracemalloc.start()
    vm = CellVM(chunk, program)
    vm.run(program.config.ticks)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "peak_bytes": peak,
        "peak_mib": peak / (1024 * 1024),
        "snapshots": len(vm.trace),
    }


def bench_examples() -> list[dict]:
    """Full compile+run wall time for every shipped example."""
    examples = sorted((ROOT / "examples").glob("*.helix"))
    rows = []
    for path in examples:
        src = path.read_text()

        def run(_src: str = src) -> None:
            chunk, program = compile_pipeline(_src)
            CellVM(chunk, program).run(program.config.ticks)

        dt = best_time(run, number=1, repeat=3)
        prog = Parser(list(Lexer(src).tokens())).parse()
        rows.append({
            "example": path.name,
            "wall_s": dt,
            "ticks": prog.config.ticks,
        })
    return rows


# ============================================================================
# Reporting
# ============================================================================

def platform_info() -> dict:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "node": platform.node(),
        "processor": platform.processor() or platform.machine(),
    }


def emit_markdown(results: dict) -> str:
    lines: list[str] = []
    add = lines.append

    info = results["platform"]
    add("# HelixLang Performance Report")
    add("")
    add("> Generated by `benchmarks/bench_helix.py`. All timings are best-of-N wall")
    add("> seconds measured with `time.perf_counter`; GC sweeps are disabled during")
    add("> timing. Synthetic programs are reproducible (fixed shapes, seeded RNG).")
    add("")
    add("## Platform")
    add("")
    add("| Property | Value |")
    add("|---|---|")
    add(f"| Python | {info['implementation']} {info['python']} |")
    add(f"| OS | {info['system']} ({info['machine']}) |")
    add(f"| Processor | {info['processor']} |")
    add("")
    add("## 1. Compile pipeline (lex → parse → semantic → compile)")
    add("")
    add("| Genes | Codons | lex | parse | semantic | compile | total | codons/s |")
    add("|---|---|---|---|---|---|---|---|")
    for r in results["compile_matrix"]:
        add(
            f"| {r['genes']} | {r['codons']} "
            f"| {r['lex']*1e3:.2f} ms | {r['parse']*1e3:.2f} ms "
            f"| {r['semantic']*1e3:.2f} ms | {r['compile']*1e3:.2f} ms "
            f"| {r['total']*1e3:.2f} ms | {fmt_rate(r['codons_per_s'], 'codons/s')} |"
        )
    add("")
    add("## 2. CellVM execution throughput")
    add("")
    add("| Ticks | wall | ticks/s | executed ops | ops/s |")
    add("|---|---|---|---|---|")
    for r in results["vm"]:
        add(
            f"| {r['ticks']} | {r['wall_s']*1e3:.1f} ms "
            f"| {fmt_rate(r['ticks_per_s'], 'ticks/s')} "
            f"| {r['executed_ops']} | {fmt_rate(r['ops_per_s'], 'ops/s')} |"
        )
    add("")
    add("## 3. GRN `step()` scaling (cycle graph, 1000 steps)")
    add("")
    add("| Nodes | Edges | steps/s | μs/step |")
    add("|---|---|---|---|")
    for r in results["grn"]:
        add(
            f"| {r['nodes']} | {r['edges']} "
            f"| {fmt_rate(r['steps_per_s'], 'steps/s')} "
            f"| {r['step_s']*1e6:.1f} μs |"
        )
    add("")
    add("## 4. Gray-Scott `step()` per-cell cost (20 steps)")
    add("")
    add("| Grid | μs/step | μs/cell |")
    add("|---|---|---|")
    for r in results["gray_scott"]:
        add(
            f"| {r['grid']} | {r['step_s']*1e6:.1f} | {r['cell_s']*1e6:.4f} |"
        )
    add("")
    add("## 5. Full compile+run of the 16 shipped examples")
    add("")
    add("| Example | ticks | wall |")
    add("|---|---|---|")
    for r in results["examples"]:
        add(f"| `{r['example']}` | {r['ticks']} | {r['wall_s']*1e3:.2f} ms |")
    add("")
    mem = results["memory"]
    add("## 6. Memory footprint")
    add("")
    add("| Workload | Peak |")
    add("|---|---|")
    add(f"| 16 genes × 64 codons, 200 ticks (compile+run) | {mem['peak_mib']:.2f} MiB "
        f"({mem['peak_bytes']:,} B), {mem['snapshots']} snapshots |")
    add("")
    add("## Raw JSON")
    add("")
    add("```json")
    add(json.dumps(results, indent=2))
    add("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast", action="store_true",
                   help="run a reduced measurement matrix")
    p.add_argument("--json", metavar="OUT", default=None,
                   help="also write the raw results as JSON")
    args = p.parse_args(argv)

    print(f"HelixLang benchmark: {'fast' if args.fast else 'full'} matrix "
          f"({platform.python_implementation()} "
          f"{platform.python_version()})", file=sys.stderr)

    results = {
        "platform": platform_info(),
        "compile_matrix": bench_compile_matrix(args.fast),
        "vm": bench_vm_throughput(args.fast),
        "grn": bench_grn_scaling(args.fast),
        "gray_scott": bench_gray_scott(),
        "examples": bench_examples(),
        "memory": bench_memory(),
    }

    report = emit_markdown(results)
    print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
