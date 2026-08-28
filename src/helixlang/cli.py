"""HelixLang command line entry point.

Usage:
  helixlang <source.helix> [--table=standard|mito_vertebrate|ciliate]
                            [--disassemble] [--debug] [--csv] [--png PREFIX]
                            [--ticks N]
  helixlang --serve [--port 5000] [--host 127.0.0.1]   # web visualization
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from helixlang.core import hxbc
from helixlang.core.ast_nodes import Program
from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import Op, get_table
from helixlang.core.compiler import Compiler
from helixlang.core.disassembler import disassemble
from helixlang.core.errors import (
    CompileError,
    LexError,
    ParseError,
    RegulationError,
    RuntimeHelixError,
    SemanticError,
    SimConfigError,
)
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.vm import CellVM
from helixlang.plugins.runtime.seq_utils import stop_codons_from_table as _stop_codons_from_table
from helixlang.sim_runtime import _SIM_BACKENDS, BACKENDS, SimResult, run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="helixlang",
        description="HelixLang: DNA codons → bytecode → biological simulation",
    )
    p.add_argument("source", type=Path, nargs="?", default=None,
                   help=".helix source file (omitted with --serve starts web mode)")
    p.add_argument("--table", default="standard",
                   choices=["standard", "mito_vertebrate", "ciliate"],
                   help="translation table (default: standard)")
    p.add_argument("--disassemble", action="store_true",
                   help="print disassembled bytecode and exit")
    p.add_argument("--debug", action="store_true",
                   help="trace VM execution")
    p.add_argument("--csv", action="store_true",
                   help="emit CSV trace to stdout")
    p.add_argument("--png", metavar="PREFIX", default=None,
                   help="write morphology field PNG (PPM format)")
    p.add_argument("--ticks", type=int, default=None,
                   help="override #config ticks")
    # Simulation backends (wiring.md §6.1, §9)
    p.add_argument("--backend", choices=sorted(BACKENDS), default=None,
                   help="override #config backend (classic keeps the bytecode "
                        "VM; whole_cell/population/fba/calibration/benchmark "
                        "run the simulation library)")
    p.add_argument("--json", action="store_true",
                   help="emit the SimResult payload as JSON (sim backends)")
    # Web visualization
    p.add_argument("--serve", action="store_true",
                   help="launch Web visualization server")
    p.add_argument("--port", type=int, default=5000,
                   help="web server port (default: 5000)")
    p.add_argument("--host", default="127.0.0.1",
                   help="web server bind host (default: 127.0.0.1)")
    # DNA physical encoding/decoding
    p.add_argument("--encode-dna", metavar="SCHEME",
                   choices=["goldman", "erlich"], default=None,
                   help="encode source to DNA (goldman/erlich), output FASTA")
    p.add_argument("--decode-dna", metavar="FILE", default=None,
                   help="decode DNA file back to helix source")
    p.add_argument("--pcr-cycles", type=int, default=0,
                   help="simulate PCR error injection (0=none, 30=standard PCR)")
    # Binary artifact (.helixc) tooling (doc/11-helixc-binary-format.md)
    p.add_argument("--compile", action="store_true",
                   help="compile .helix source into a .helixc binary artifact")
    p.add_argument("--decompile", action="store_true",
                   help="decompile a .helixc artifact back to .helix source")
    p.add_argument("--compare", metavar="ARTIFACT", default=None,
                   help="run a .helix source and a .helixc artifact, diff traces")
    p.add_argument("-o", "--output", metavar="OUT", default=None,
                   help="output path for --compile / --decompile")
    p.add_argument("--no-chunk", action="store_true",
                   help="--compile without embedding the precompiled chunk")
    p.add_argument("--no-source", action="store_true",
                    help="--compile without embedding the original source")
    p.add_argument("--check-bytecode-version", action="store_true",
                    help="print OPCODE_VERSION and exit")
    # GEM reconstruction pipeline (doc/20)
    p.add_argument("--gem", action="store_true",
                   help="run GEM reconstruction pipeline from #species genome data")
    p.add_argument("--dynamic", action="store_true",
                   help="use dynamic FBA (time-course) mode with GEM backend")
    p.add_argument("--duration", metavar="HOURS", type=float, default=24.0,
                   help="simulation duration in hours for --dynamic mode (default: 24)")
    p.add_argument("--dt", metavar="HOURS", type=float, default=0.1,
                   help="time step in hours for --dynamic mode (default: 0.1)")
    # Full-chain pipeline (doc/26)
    p.add_argument("--full-pipeline", action="store_true",
                   help="run full-chain custom organism pipeline from FASTA "
                        "(DNA → structure → kinetics → ecGEM → simulation)")
    args = p.parse_args(argv)

    # ----- Web mode -----
    if args.serve:
        try:
            from helixlang.server import run_server
        except ImportError as e:
            print(f"error: web mode requires Flask, run: "
                  f"pip install 'helixlang[web]'\n  ({e})",
                  file=sys.stderr)
            return 1
        print(f"HelixLang web visualization running at: http://{args.host}:{args.port}")
        print("  press Ctrl+C to exit")
        return run_server(host=args.host, port=args.port, debug=False)

    # ----- Bytecode version check -----
    if args.check_bytecode_version:
        from helixlang.core.bytecode import OPCODE_VERSION
        print(f"OPCODE_VERSION={OPCODE_VERSION}")
        return 0

    # ----- compile/run mode -----
    if args.source is None:
        p.error("missing source file (or use --serve for web mode)")
        return 2

    # ----- DNA physical codec mode -----
    if args.encode_dna or args.decode_dna:
        try:
            from helixlang.plugins.runtime.dna_codec import (
                dna_to_helix,
                helix_to_dna,
                pcr_amplify,
            )
        except ImportError as e:
            print(f"error: DNA codec requires bio dependencies: "
                  f"pip install 'helixlang[bio]'\n  ({e})",
                  file=sys.stderr)
            return 1

        if args.encode_dna:
            # encode source → DNA
            src = args.source.read_text()
            enc = helix_to_dna(src, scheme=args.encode_dna)
            print(f"=== {args.encode_dna} encoding: {enc['stats']['num_oligos']} oligos, "
                  f"{enc['stats']['total_bp']} bp, "
                  f"{enc['stats']['density_bit_per_nt']} bit/nt ===")
            import random
            rng = random.Random(42)
            for o in enc["oligos"]:
                seq = o["full"] if "full" in o else o["payload"]
                if args.pcr_cycles > 0:
                    seq = pcr_amplify(seq, cycles=args.pcr_cycles, rng=rng)
                print(f">oligo_{o['index']} len={len(seq)}")
                print(seq)
            return 0

        if args.decode_dna:
            # decode DNA file → helix
            dna_file = Path(args.decode_dna)
            if not dna_file.exists():
                print(f"error: DNA file not found: {dna_file}", file=sys.stderr)
                return 2
            # read FASTA
            oligos = []
            cur_idx = 0
            cur_seq = ""
            for line in dna_file.read_text().splitlines():
                if line.startswith(">"):
                    if cur_seq:
                        oligos.append({"index": cur_idx, "full": cur_seq,
                                       "payload": cur_seq, "overhang": ""})
                        cur_idx += 1
                    cur_seq = ""
                else:
                    cur_seq += line.strip()
            if cur_seq:
                oligos.append({"index": cur_idx, "full": cur_seq,
                               "payload": cur_seq, "overhang": ""})
            enc_data = {"oligos": oligos,
                        "stats": {"num_segments": len(oligos)}}
            # try goldman decode
            try:
                result = dna_to_helix(enc_data, scheme="goldman")
                print(result)
                return 0
            except (ValueError, KeyError, IndexError, RuntimeError) as e:
                print(f"decode error: {e}", file=sys.stderr)
                return 1

    if not args.source.exists():
        print(f"error: source file not found: {args.source}",
              file=sys.stderr)
        return 2

    # ----- Full-chain pipeline (doc/26) -----
    if args.full_pipeline:
        return _run_full_pipeline(args)

    # ----- binary artifact modes (doc/11-helixc-binary-format.md) -----
    if args.compile or args.decompile or args.compare:
        return _run_artifact_mode(args)

    if args.source.suffix == ".helixc":
        return _run_artifact(args)

    src = args.source.read_text()

    try:
        # parse + semantic check (shared by every backend)
        table = get_table(args.table)
        stop_codons = _stop_codons_from_table(table)
        tokens = list(Lexer(src).tokens())
        program = Parser(tokens, stop_codons=stop_codons).parse()
        SemanticAnalyzer(program).check()
    except (LexError, ParseError, SemanticError, CompileError) as e:
        print(f"compile error: {e}", file=sys.stderr)
        return 1

    # ----- Simulation backends (wiring.md §9) -----
    if args.ticks is not None:
        program.config.ticks = args.ticks
    # --gem flag forces GEM backend
    if args.gem:
        effective_backend = "gem"
    else:
        effective_backend = args.backend or program.config.backend
    # Wire --dynamic through sim_extensions
    if args.dynamic:
        program.sim_extensions["gem_dynamic"] = "true"
        program.sim_extensions["gem_duration"] = str(args.duration)
        program.sim_extensions["gem_dt"] = str(args.dt)
    if not args.disassemble and (
            program.sim_extensions.get("kind") in _SIM_BACKENDS
            or effective_backend != "classic"):
        return _run_sim(program, args, effective_backend)

    # ----- classic bytecode pipeline (bit-identical to before) -----
    try:
        chunk = Compiler(table).compile(program)
    except CompileError as e:
        print(f"compile error: {e}", file=sys.stderr)
        return 1

    if args.disassemble:
        print(disassemble(chunk, args.source.name))
        return 0

    vm = CellVM(chunk, program)
    vm.debug = args.debug

    try:
        trace = vm.run(program.config.ticks)
    except (RuntimeHelixError, ValueError, IndexError, KeyError) as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 1

    # output
    if args.csv:
        _emit_csv(trace)
    if args.png:
        _emit_ppm(vm, args.png)
    if not args.csv and not args.png:
        print(f"=== {args.source.name} | table={args.table} "
              f"ticks={program.config.ticks} ===")
        for s in trace[-5:]:
            print(f"  tick={s['tick']:>3} pos=({s['x']},{s['y']}) "
                  f"energy={s['energy']} alive={int(s['alive'])} "
                  f"proteins={s['proteins']}")
        if vm.cell.morphology_points:
            print(f"  morphology points: {len(vm.cell.morphology_points)}")
        if vm.field:
            print(f"  field total V: {vm.field.total_v():.3f}")
    return 0


def _run_full_pipeline(args: argparse.Namespace) -> int:
    """Run full-chain custom organism pipeline from FASTA (doc/26)."""
    try:
        from helixlang.plugins.apps.full_pipeline import PipelineConfig, run_full_pipeline
    except ImportError as e:
        print(f"error: full pipeline requires: {e}", file=sys.stderr)
        return 1
    config = PipelineConfig()
    result = run_full_pipeline(str(args.source), config)
    print(f"=== Full Pipeline: {args.source.name} ===")
    print(f"  stages completed: {', '.join(result.stages_completed)}")
    print(f"  proteins: {len(result.proteins)}")
    print(f"  structures: {len(result.structures)}")
    print(f"  kcat predictions: {len(result.kcat_predictions)}")
    if result.ecgem is not None:
        print(f"  ecGEM growth rate: {result.ecgem.growth_rate:.4f} h^-1")
        print(f"  ecGEM unconstrained: {result.ecgem.growth_rate_unconstrained:.4f} h^-1")
    if result.community is not None:
        print(f"  community biomass: {result.community.total_biomass:.4f}")
        print(f"  community converged: {result.community.converged}")
    print(f"  pipeline time: {result.pipeline_time:.2f}s")
    if result.warnings:
        for w in result.warnings:
            print(f"  warning: {w}")
    return 0


def _run_sim(program: Program, args: argparse.Namespace, backend: str) -> int:
    """Dispatch a sim backend and render the SimResult (wiring.md §9)."""
    try:
        result = run(program, backend=backend)
    except (SimConfigError, ValueError, KeyError, IndexError) as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 1
    if result is None:
        return 0
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    elif args.csv:
        _emit_sim_csv(result)
    else:
        _emit_sim_table(result, args.source.name)
    return 0


def _run_artifact_mode(args: argparse.Namespace) -> int:
    """Handle --compile / --decompile / --compare (doc/11-helixc-binary-format.md)."""
    if args.compile:
        if args.output is None:
            print("error: --compile requires -o OUT", file=sys.stderr)
            return 2
        try:
            info = hxbc.compile_file(
                args.source, args.output,
                include_chunk=not args.no_chunk,
                include_source=not args.no_source,
                table=args.table)
        except (LexError, ParseError, SemanticError, CompileError,
                hxbc.BinaryError, OSError) as e:
            print(f"compile error: {e}", file=sys.stderr)
            return 1
        size = info.path.stat().st_size
        print(f"=== compiled {args.source.name} -> {args.output} ===")
        print(f"  {size} bytes | table={info.table} | "
              f"chunk={'embedded' if info.chunk is not None else 'omitted'} | "
              f"source={'embedded' if info.source is not None else 'omitted'}")
        return 0

    if args.decompile:
        if args.output is None:
            print("error: --decompile requires -o OUT", file=sys.stderr)
            return 2
        try:
            art = hxbc.load_program(args.source)
        except hxbc.BinaryError as e:
            print(f"binary error: {e}", file=sys.stderr)
            return 1
        text = art.source if art.source is not None else hxbc.decompile(art.program)
        Path(args.output).write_text(text)
        print(f"=== decompiled {args.source.name} -> {args.output} ===")
        print(f"  {len(text)} bytes | "
              f"byte-for-byte={'yes' if art.source is not None else 'no'} "
              f"(embedded source)")
        return 0

    # --compare <source.helix> <artifact.helixc>  (either flag order)
    artifact_path = Path(args.compare)
    source_path = args.source
    if artifact_path.suffix != ".helixc" and source_path.suffix == ".helixc":
        source_path, artifact_path = artifact_path, source_path
    try:
        hxbc.load_program(artifact_path)
    except hxbc.BinaryError as e:
        print(f"binary error: {e}", file=sys.stderr)
        return 1
    rc1, out1 = _capture_main([str(source_path), "--csv"])
    rc2, out2 = _capture_main([str(artifact_path), "--csv"])
    if rc1 != 0:
        print(f"error: source run failed (rc={rc1})", file=sys.stderr)
        return rc1
    if rc2 != 0:
        print(f"error: artifact run failed (rc={rc2})", file=sys.stderr)
        return rc2
    if out1 != out2:
        print(f"=== compare MISMATCH {source_path} vs {artifact_path} ===")
        _print_diff(out1.splitlines(), out2.splitlines())
        return 1
    print(f"=== compare OK: {source_path} == {artifact_path} ===")
    return 0


def _capture_main(argv: list[str]) -> tuple[int, str]:
    """Run main() with stdout redirected, returning (rc, output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def _print_diff(source_lines: list[str], artifact_lines: list[str]) -> None:
    n = max(len(source_lines), len(artifact_lines))
    for i in range(n):
        sl = source_lines[i] if i < len(source_lines) else "<missing>"
        al = artifact_lines[i] if i < len(artifact_lines) else "<missing>"
        if sl != al:
            print(f"  line {i}: source:  {sl}")
            print(f"          artifact: {al}")


def _run_artifact(args: argparse.Namespace) -> int:
    """Run a .helixc artifact: classic VM (cached chunk) or sim backend."""
    try:
        art = hxbc.load_program(args.source)
    except hxbc.BinaryError as e:
        print(f"binary error: {e}", file=sys.stderr)
        return 1
    program = art.program
    try:
        SemanticAnalyzer(program).check()
    except (SemanticError, RegulationError) as e:
        print(f"semantic error: {e}", file=sys.stderr)
        return 1

    if art.chunk_stale:
        print(f"warning: stale compiled chunk in {args.source.name}; "
              f"recompiling from the program AST", file=sys.stderr)

    if args.ticks is not None:
        program.config.ticks = args.ticks
    effective_backend = args.backend or program.config.backend
    if not args.disassemble and (
            program.sim_extensions.get("kind") in _SIM_BACKENDS
            or effective_backend != "classic"):
        return _run_sim(program, args, effective_backend)

    # classic bytecode path
    if args.table != "standard":
        # explicit --table override -> recompile from the AST
        chunk = _compile_from_program(program, get_table(args.table))
        if chunk is None:
            return 1
    elif art.chunk is not None:
        chunk = art.chunk
    else:
        chunk = _compile_from_program(program, get_table(art.table))
        if chunk is None:
            return 1

    if args.disassemble:
        print(disassemble(chunk, args.source.name))
        return 0

    vm = CellVM(chunk, program)
    vm.debug = args.debug
    try:
        trace = vm.run(program.config.ticks)
    except (RuntimeHelixError, ValueError, IndexError, KeyError) as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 1

    if args.csv:
        _emit_csv(trace)
    if args.png:
        _emit_ppm(vm, args.png)
    if not args.csv and not args.png:
        print(f"=== {args.source.name} | table={art.table} "
              f"ticks={program.config.ticks} ===")
        for s in trace[-5:]:
            print(f"  tick={s['tick']:>3} pos=({s['x']},{s['y']}) "
                  f"energy={s['energy']} alive={int(s['alive'])} "
                  f"proteins={s['proteins']}")
        if vm.cell.morphology_points:
            print(f"  morphology points: {len(vm.cell.morphology_points)}")
        if vm.field:
            print(f"  field total V: {vm.field.total_v():.3f}")
    return 0


def _compile_from_program(program: Program,
                          table: dict[str, Op]) -> Chunk | None:
    """Compile a chunk, returning None (after reporting) on CompileError."""
    try:
        return Compiler(table).compile(program)
    except CompileError as e:
        print(f"compile error: {e}", file=sys.stderr)
        return None


def _cell(value: object) -> str:
    """Render a cell value for text/CSV output."""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def _emit_sim_csv(result: SimResult) -> None:
    print(",".join(["backend"] + result.columns))
    for row in result.rows:
        print(",".join([result.backend] + [_cell(row.get(c))
                                           for c in result.columns]))


def _emit_sim_table(result: SimResult, source_name: str) -> None:
    n = len(result.rows)
    print(f"=== {source_name} | backend={result.backend} rows={n} ===")
    print("  " + " | ".join(result.columns))
    for row in result.rows[:30]:
        print("  " + " | ".join(_cell(row.get(c)) for c in result.columns))
    if n > 30:
        print(f"  ... {n - 30} more rows")


def _emit_csv(trace: list[dict]) -> None:
    print("tick,x,y,energy,alive,proteins,morphology_points,field_total_v")
    for s in trace:
        prots = ';'.join(f"{k}:{v:.2f}" for k, v in s["proteins"].items())
        print(f"{s['tick']},{s['x']},{s['y']},{s['energy']},"
              f"{int(s['alive'])},\"{prots}\","
              f"{s['morphology_points_count']},{s['field_total_v']:.4f}")


def _emit_ppm(vm: CellVM, prefix: str) -> None:
    """Output PPM (P3) format, no dependencies."""
    if vm.field is None:
        # output L-system morphology (simple binary image)
        points = vm.cell.morphology_points
        if not points:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w = max(int(maxx - minx) + 2, 1)
        h = max(int(maxy - miny) + 2, 1)
        grid = [['255 255 255'] * w for _ in range(h)]
        for x, y in points:
            ix = int(x - minx)
            iy = int(y - miny)
            if 0 <= ix < w and 0 <= iy < h:
                grid[iy][ix] = '0 0 0'
        lines = ["P3", f"{w} {h}", "255"]
        for row in grid:
            lines.append(' '.join(row))
        Path(f"{prefix}.ppm").write_text('\n'.join(lines))
        return
    # reaction-diffusion field: V concentration mapped to blue channel
    n = vm.field.n
    lines = ["P3", f"{n} {n}", "255"]
    for field_row in vm.field.v:
        line_parts = []
        for v in field_row:
            b = int(v * 255)
            line_parts.append(f"0 0 {b}")
        lines.append(' '.join(line_parts))
    Path(f"{prefix}.ppm").write_text('\n'.join(lines))
