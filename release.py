#!/usr/bin/env python3
"""HelixLang Release Script

One-command release: sync version, run all gates, build.

Usage:
    python release.py <version>

Example:
    python release.py 2026.8.5

What this script does:
    1. Validates version format
    2. Syncs version to pyproject.toml, __init__.py, server.py
    3. Runs all quality gates in parallel (ruff, mypy, pytest -n auto, validation, examples)
    4. Syncs metrics to README.md, README_PYPI.md, CONTRIBUTING.md
    5. Builds sdist + wheel
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# ─── Colors ───────────────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"
BOLD = "\033[1m"

# ─── Globals ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("PYTHON", sys.executable)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(f"{BLUE}[release]{NC} {msg}")


def ok(msg: str) -> None:
    print(f"{GREEN}  ✓{NC} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}  ⚠{NC} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}  ✗{NC} {msg}")


def banner(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}═══ {msg} ═══{NC}\n")


# ─── Gate runner ──────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    name: str
    exit_code: int
    log_file: Path


def run_gate(name: str, cmd: list[str], gate_dir: Path, cwd: Path | None = None) -> GateResult:
    """Run a command, capture output to gate_dir, return exit code."""
    log_file = gate_dir / f"{name}.log"
    exit_file = gate_dir / f"{name}.exit"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=cwd or ROOT,
        )
        log_file.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
        exit_file.write_text(str(result.returncode))
        return GateResult(name=name, exit_code=result.returncode, log_file=log_file)
    except subprocess.TimeoutExpired:
        log_file.write_text("TIMEOUT: gate exceeded 1800s")
        exit_file.write_text("124")
        return GateResult(name=name, exit_code=124, log_file=log_file)
    except Exception as e:
        log_file.write_text(f"ERROR: {e}")
        exit_file.write_text("1")
        return GateResult(name=name, exit_code=1, log_file=log_file)


def report_gate(result: GateResult) -> bool:
    """Report gate result. Returns True if passed."""
    if result.exit_code == 0:
        ok(f"{result.name} — PASS")
        return True
    else:
        fail(f"{result.name} — FAIL (exit {result.exit_code})")
        if result.log_file.exists():
            lines = result.log_file.read_text().splitlines()
            if lines:
                print(f"    {RED}Last 15 lines:{NC}")
                for line in lines[-15:]:
                    print(f"    {line}")
        return False


# ─── Step 1: Version sync ────────────────────────────────────────────────────


def sync_version(version: str) -> bool:
    """Sync version to all source files. Returns True on success."""
    banner(f"Step 1: Sync version → {version}")

    replacements = [
        (ROOT / "pyproject.toml", re.compile(r'^version = ".*"', re.MULTILINE), f'version = "{version}"'),
        (ROOT / "src" / "helixlang" / "__init__.py", re.compile(r'^__version__ = ".*"', re.MULTILINE), f'__version__ = "{version}"'),
        (ROOT / "src" / "helixlang" / "server.py", re.compile(r'"version": ".*"'), f'"version": "{version}"'),
    ]

    all_ok = True
    for path, pattern, replacement in replacements:
        if path.exists():
            content = path.read_text()
            new_content = pattern.sub(replacement, content)
            path.write_text(new_content)
            ok(f"{path.name} → {version}")
        else:
            fail(f"File not found: {path}")
            all_ok = False

    # bytecode.py comment only
    bp = ROOT / "src" / "helixlang" / "bytecode.py"
    if bp.exists():
        content = bp.read_text()
        if "Frozen as of HelixLang" in content:
            new_content = re.sub(
                r"Frozen as of HelixLang .*",
                f"Frozen as of HelixLang {version}.",
                content,
            )
            bp.write_text(new_content)
            ok(f"bytecode.py comment → {version}")

    # Verify sync
    for path in [ROOT / "pyproject.toml", ROOT / "src" / "helixlang" / "__init__.py", ROOT / "src" / "helixlang" / "server.py"]:
        if path.exists() and version not in path.read_text():
            fail(f"Version mismatch in {path}")
            all_ok = False

    if not all_ok:
        print(f"\n{RED}Version sync failed. Aborting.{NC}", file=sys.stderr)
        return False

    ok("All version strings synchronized")
    return True


# ─── Step 2: Quality gates ───────────────────────────────────────────────────


def run_quality_gates(gate_dir: Path) -> list[GateResult]:
    """Run all quality gates in parallel. Returns list of results."""
    banner("Step 2: Quality gates (running in parallel)")

    log(f"Gate results → {gate_dir}/")

    gates = [
        ("ruff", [PYTHON, "-m", "ruff", "check", "src", "tests"], ROOT),
        ("mypy", [PYTHON, "-m", "mypy", "src/helixlang/"], ROOT),
        ("pytest", [PYTHON, "-B", "-m", "pytest", "tests/", "-n", "auto", "--tb=short", "-q"], ROOT),
    ]

    # Validation gate
    val_runner = ROOT / "validation" / "run_all.py"
    if val_runner.exists():
        gates.append(("val", [PYTHON, "-B", str(val_runner)], ROOT))

    # Examples gate
    examples_dir = ROOT / "examples"
    helix_files = list(examples_dir.glob("*.helix")) if examples_dir.exists() else []
    if helix_files:
        gates.append(("examples", None, ROOT))  # handled separately

    results: list[GateResult] = []

    with ThreadPoolExecutor(max_workers=len(gates)) as pool:
        futures = {}
        for name, cmd, cwd in gates:
            if cmd is not None:
                futures[pool.submit(run_gate, name, cmd, gate_dir, cwd)] = name

        # Run examples gate in the same pool
        if helix_files:
            def _run_examples() -> GateResult:
                log_file = gate_dir / "examples.log"
                exit_file = gate_dir / "examples.exit"
                fail_count = 0
                lines = []
                for f in helix_files:
                    r = subprocess.run(
                        [PYTHON, "-B", "-m", "helixlang", str(f), "--disassemble"],
                        capture_output=True, text=True, timeout=60, cwd=ROOT,
                    )
                    if r.returncode != 0:
                        lines.append(f"COMPILE FAIL: {f}")
                        fail_count += 1
                # Run minimal example
                hello = examples_dir / "01_hello_dna.helix"
                if hello.exists():
                    r = subprocess.run(
                        [PYTHON, "-B", "-m", "helixlang", str(hello)],
                        capture_output=True, text=True, timeout=60, cwd=ROOT,
                    )
                    if r.returncode != 0:
                        lines.append(f"RUN FAIL: {hello}")
                        fail_count += 1
                log_file.write_text("\n".join(lines) if lines else "OK")
                exit_file.write_text(str(fail_count))
                return GateResult(name="examples", exit_code=fail_count, log_file=log_file)

            futures[pool.submit(_run_examples)] = "examples"

        for future in as_completed(futures):
            results.append(future.result())

    # Sort by original order
    order = {"ruff": 0, "mypy": 1, "pytest": 2, "val": 3, "examples": 4}
    results.sort(key=lambda r: order.get(r.name, 99))

    log("Waiting for gates...")
    print()
    banner("Gate Results")

    all_passed = True
    for r in results:
        if not report_gate(r):
            all_passed = False

    if not all_passed:
        print(f"\n{RED}Some gates failed. Fix them before releasing.{NC}")
        print(f"Full logs in: {gate_dir}/")
        return results

    ok("All gates passed")
    return results


# ─── Step 2b: Generate validation report ─────────────────────────────────────


def generate_report() -> None:
    banner("Step 2b: Generate validation report")
    runner = ROOT / "validation" / "run_all.py"
    if runner.exists():
        r = subprocess.run(
            [PYTHON, "-B", str(runner), "--report-only"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if r.returncode == 0:
            ok("validation/report.md regenerated")
        else:
            warn("Report generation had warnings (non-fatal)")
    else:
        warn("No report generator found — skipping")


# ─── Step 3: Sync metrics ────────────────────────────────────────────────────


def _replacer(pattern: str, replacement: str, text: str) -> str:
    return re.sub(pattern, replacement, text)


def sync_metrics(gate_dir: Path) -> None:
    banner("Step 3: Sync metrics to README / README_PYPI / CONTRIBUTING")

    # Extract pytest stats
    pytest_log = gate_dir / "pytest.log"
    test_count = ""
    coverage = ""
    if pytest_log.exists():
        content = pytest_log.read_text()
        m = re.search(r"(\d+) passed", content)
        if m:
            test_count = m.group(1)
        m = re.search(r"TOTAL.*?(\d+)%", content)
        if m:
            coverage = m.group(1) + "%"

    # Extract validation stats from report.md
    val_pass = ""
    val_total = ""
    report_path = ROOT / "validation" / "report.md"
    if report_path.exists():
        content = report_path.read_text()
        m = re.search(r"\*\*(\d+)/(\d+)\*\* pass", content)
        if m:
            val_pass = m.group(1)
            val_total = m.group(2)

    # Fallback: extract from gate log
    if not val_pass:
        val_log = gate_dir / "val.log"
        if val_log.exists():
            content = val_log.read_text()
            m = re.search(r"(\d+)/(\d+) PASS", content)
            if m:
                val_pass = m.group(1)
                val_total = m.group(2)

    # Count files
    test_files = len(list((ROOT / "tests").glob("test_*.py"))) if (ROOT / "tests").exists() else 0
    src_modules = len(list((ROOT / "src" / "helixlang").rglob("*.py"))) if (ROOT / "src" / "helixlang").exists() else 0
    doc_count = len(list((ROOT / "doc").glob("*.md"))) if (ROOT / "doc").exists() else 0
    example_count = len(list((ROOT / "examples").glob("*.helix"))) if (ROOT / "examples").exists() else 0

    log(f"Discovered: {test_count or '?'} tests, {test_files} test files, {src_modules} modules, {doc_count} docs, {example_count} examples")

    # Sync to docs
    for fname in ["README.md", "README_PYPI.md", "CONTRIBUTING.md"]:
        fpath = ROOT / fname
        if not fpath.exists():
            continue

        content = fpath.read_text()

        # 1. Test count — table format: "| Test cases | 3,192 (81% coverage) |"
        if test_count:
            content = re.sub(
                r"(\| Test cases \| )\d[\d,]*( \(.*?\))?( \||\|)",
                rf"\g<1>{test_count}\g<2>\g<3>",
                content,
            )
            # bullet format: "- **3,192 test cases**(all passing..."
            content = re.sub(
                r"\*\*[\d,]+ test cases\*\*",
                f"**{test_count} test cases**",
                content,
            )

        # 2. Validation benchmarks — table format: "| Validation benchmarks | 67 (67 pass) |"
        if val_pass and val_total:
            content = re.sub(
                r"(\| Validation benchmarks \| )\d+ \(\d+ pass[^)]*\)( \||\|)",
                rf"\g<1>{val_total} ({val_pass} pass)\g<2>",
                content,
            )

        # 3. "Benchmarks passing" — "**67/67**"
        if val_pass and val_total:
            content = re.sub(
                r"Benchmarks passing \| \*\*\d+/\d+\*\*",
                f"Benchmarks passing | **{val_pass}/{val_total}**",
                content,
            )

        # 4. "N reproducible benchmarks" heading — "67 reproducible benchmarks"
        if val_total:
            content = re.sub(
                r"\d+ reproducible benchmarks validating",
                f"{val_total} reproducible benchmarks validating",
                content,
            )

        # 5. Benchmark range in table — "| 11-45 |" → "| 11-67 |"
        if val_total:
            content = re.sub(
                r"(\| 11-)\d+ (\|)",
                rf"\g<1>{val_total}\g<2>",
                content,
            )

        # 6. Source modules
        content = re.sub(
            r"(\| Source modules \| )\d+(\s*\|)",
            rf"\g<1>{src_modules}\g<2>",
            content,
        )

        # 7. Test files count — "pytest suite (N files"
        content = re.sub(
            r"pytest suite \(\d+ files",
            f"pytest suite ({test_files} files",
            content,
        )

        # 8. Doc count — "(N files, 25,000+"
        content = re.sub(
            r"\(\d+ files, 25,000\+",
            f"({doc_count} files, 25,000+",
            content,
        )

        # 9. Example count — "60 runnable .helix" and "| `.helix` examples | 60 |"
        content = re.sub(
            r"\d+ runnable \.helix",
            f"{example_count} runnable .helix",
            content,
        )
        content = re.sub(
            r"(\| `\.helix` examples \| )\d+(\s*\|)",
            rf"\g<1>{example_count}\g<2>",
            content,
        )

        fpath.write_text(content)

    ok(f"Docs synced: tests={test_count or '?'}, val={val_pass or '?'}/{val_total or '?'}, modules={src_modules}, examples={example_count}")


# ─── Step 4: Build ───────────────────────────────────────────────────────────


def build() -> bool:
    banner("Step 4: Build")

    # Clean old artifacts
    for d in [ROOT / "dist", ROOT / "build"]:
        if d.exists():
            shutil.rmtree(d)
    for egg in ROOT.glob("*.egg-info"):
        if egg.is_dir():
            shutil.rmtree(egg)

    r = subprocess.run(
        [PYTHON, "-B", "-m", "build"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)

    whl = list((ROOT / "dist").glob("*.whl"))
    tar = list((ROOT / "dist").glob("*.tar.gz"))
    artifacts = " ".join(str(p.name) for p in whl + tar)
    ok(f"Built: {artifacts}")
    return r.returncode == 0


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HelixLang release script — sync version, gate, build.",
        usage="python release.py <version>",
    )
    parser.add_argument("version", help="Version in YYYY.M.D or YYYY.M.D.N format")
    args = parser.parse_args()

    version = args.version

    # Validate version format
    if not re.match(r"^\d{4}\.\d{1,2}\.\d{1,2}(\.\d+)?$", version):
        print(f"Error: Version must match YYYY.M.D or YYYY.M.D.N (got: {version})", file=sys.stderr)
        return 1

    os.chdir(ROOT)

    # Step 1: Sync version
    if not sync_version(version):
        return 1

    # Step 2: Quality gates
    with tempfile.TemporaryDirectory(prefix="release_gates_") as tmpdir:
        gate_dir = Path(tmpdir)
        results = run_quality_gates(gate_dir)

        failed = [r for r in results if r.exit_code != 0]
        if failed:
            return 1

        # Step 2b: Generate report
        generate_report()

        # Step 3: Sync metrics
        sync_metrics(gate_dir)

    # Step 4: Build
    if not build():
        return 1

    # Summary
    banner(f"Release Complete: {version}")

    print(f"  {BOLD}Files modified:{NC}")
    r = subprocess.run(["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines():
            print(f"    {line}")
    else:
        print("    (none)")

    print()
    print(f"  {BOLD}Next steps:{NC}")
    print("    1. Review changes:  git diff")
    print(f"    2. Commit:          git add -A && git commit -m 'release: v{version}'")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
