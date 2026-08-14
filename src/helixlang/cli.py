"""HelixLang command line entry point.

Usage:
  helixlang <source.helix> [--table=standard|mito_vertebrate|ciliate]
                            [--disassemble] [--debug] [--csv] [--png PREFIX]
                            [--ticks N]
  helixlang --serve [--port 5000] [--host 127.0.0.1]   # web visualization
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from helixlang.ast_nodes import Program
from helixlang.codon_table import get_table
from helixlang.compiler import Compiler
from helixlang.disassembler import disassemble
from helixlang.errors import (
    CompileError,
    LexError,
    ParseError,
    RuntimeHelixError,
    SemanticError,
    SimConfigError,
)
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.semantic import SemanticAnalyzer
from helixlang.seq_utils import stop_codons_from_table as _stop_codons_from_table
from helixlang.sim_runtime import _SIM_BACKENDS, BACKENDS, SimResult, run
from helixlang.vm import CellVM


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

    # ----- compile/run mode -----
    if args.source is None:
        p.error("missing source file (or use --serve for web mode)")
        return 2

    # ----- DNA physical codec mode -----
    if args.encode_dna or args.decode_dna:
        try:
            from helixlang.dna_codec import (
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
    effective_backend = args.backend or program.config.backend
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
