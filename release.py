#!/usr/bin/env python3
"""HelixLang Release Script

One-command release: sync version, run all gates, build.

Usage:
    python release.py <version>

Example:
    python release.py 2026.9.0

Version format: YYYY.M.D or YYYY.M.D.N (e.g. 2026.9.1, 2026.9.1.2):
    D = iteration release of the month, starting from 0 (first release is YYYY.M.0)
    N (optional) = patch version of that iteration release (2026.9.1.2 is the 2nd
                   patch of the month's 2nd release)

What this script does:
    1. Validates version format
    2. Checks every version-bearing source for drift and fails fast on mismatch (doc/38 §2.3)
    3. Syncs version to pyproject.toml, core/version.py, server/app.py (+ bytecode.py comment)
    4. Runs all quality gates in parallel (ruff, mypy, pytest -n auto, validation, examples)
    5. Syncs metrics to README.md, README_PYPI.md, CONTRIBUTING.md
    6. Builds sdist + wheel

Every run persists its complete results — the full transcript, each gate's
stdout/stderr, and a per-gate exit-code summary — under ``release_logs/``
(overridable with ``--log-dir DIR``), one directory per run:

    release_logs/<version>-<YYYYMMDD-HHMMSS>/
        release.log     full transcript of this run (ANSI-stripped)
        gates/          <gate>.log + <gate>.exit for every gate
        summary.txt     version, per-gate status, overall result, artifacts

Nothing is deleted after a run, including on failure, so a failed release can
always be diagnosed from disk.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
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


# ─── Run log persistence ─────────────────────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_DEFAULT_LOG_ROOT = ROOT / "release_logs"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _Tee:
    """A file-like writer that mirrors one stream to the terminal AND to an
    ANSI-stripped log file, so the full transcript survives the run."""

    def __init__(self, real: object, log_file: object) -> None:
        self._real = real
        self._log = log_file

    def write(self, s: str) -> int:
        self._real.write(s)
        self._log.write(strip_ansi(s))
        self._log.flush()
        return len(s)

    def flush(self) -> None:
        self._real.flush()
        self._log.flush()

    def isatty(self) -> bool:
        try:
            return bool(self._real.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._real.fileno()

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def setup_run_log(version: str, log_root: Path | None = None) -> tuple[Path, Path]:
    """Create a per-run directory and redirect stdout/stderr into a release.log.

    Returns ``(run_dir, gates_dir)``.  Nothing is cleaned up afterwards — the
    transcripts and gate logs persist even when the release fails.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (log_root or _DEFAULT_LOG_ROOT) / f"{version}-{ts}"
    gates_dir = run_dir / "gates"
    run_dir.mkdir(parents=True, exist_ok=True)
    gates_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(run_dir / "release.log", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    log(f"Run directory : {run_dir}/")
    log(f"Full transcript: {run_dir / 'release.log'}")
    log(f"Gate logs     : {gates_dir}/")
    return run_dir, gates_dir


def write_summary(run_dir: Path, version: str, *,
                  result: str, gates: list[GateResult] | None = None,
                  elapsed: float) -> None:
    """Write an ANSI-free summary.txt describing the whole run."""
    lines = [
        "HelixLang release run — summary",
        f"Version     : {version}",
        f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed     : {elapsed:.1f}s",
        f"Result      : {result}",
        f"Transcript  : {run_dir / 'release.log'}",
        "",
    ]
    if gates:
        lines.append(f"{'gate':<10} {'exit':>4}  status")
        lines.append(f"{'-'*10} {'-'*4}  {'-'*6}")
        order = {"ruff": 0, "mypy": 1, "boundary": 2, "pytest": 3, "val": 4,
                 "examples": 5}
        for g in sorted(gates, key=lambda r: order.get(r.name, 99)):
            status = "PASS" if g.exit_code == 0 else "FAIL"
            lines.append(f"{g.name:<10} {g.exit_code:>4}  {status}")
        lines.append("")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")


# ─── Gate runner ──────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    name: str
    exit_code: int
    log_file: Path


def run_gate(name: str, cmd: list[str], gate_dir: Path, cwd: Path | None = None) -> GateResult:
    """Run a command, capture output to gate_dir, return exit code.

    No timeout gate: every gate (including the full pytest suite) runs to
    completion, however long it takes (doc/38 release notes).
    """
    log_file = gate_dir / f"{name}.log"
    exit_file = gate_dir / f"{name}.exit"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or ROOT,
        )
        log_file.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
        exit_file.write_text(str(result.returncode))
        return GateResult(name=name, exit_code=result.returncode, log_file=log_file)
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
                for line in lines:
                    print(f"    {line}")
        return False


# ─── Step 0: Version drift check ─────────────────────────────────────────────

#: Every place a release version is written, so a partial sync can be detected
#: before release.py overwrites anything (doc/38 §2.3).
_VERSION_LOCATIONS: list[tuple[str, Path, re.Pattern[str]]] = [
    ("pyproject.toml", ROOT / "pyproject.toml",
     re.compile(r'^version = "([^"]+)"', re.MULTILINE)),
    ("core/version.py", ROOT / "src" / "helixlang" / "core" / "version.py",
     re.compile(r'^__version__ = "([^"]+)"', re.MULTILINE)),
    ("server/app.py", ROOT / "src" / "helixlang" / "server" / "app.py",
     re.compile(r'"version": "([^"]+)"')),
    ("core/bytecode.py", ROOT / "src" / "helixlang" / "core" / "bytecode.py",
     re.compile(r"Frozen as of HelixLang (\d+\.\d+\.\d+(?:\.\d+)?)")),
]


def read_versions() -> dict[str, str]:
    """Current version string from every location, keyed by a short name."""
    found: dict[str, str] = {}
    for name, path, pattern in _VERSION_LOCATIONS:
        if path.exists():
            m = pattern.search(path.read_text())
            if m:
                found[name] = m.group(1)
    return found


def check_versions() -> bool:
    """Fail-fast drift gate: every version-bearing source must agree.

    Returns True when all present locations carry the same version and every
    location is readable; otherwise the release aborts before any write.
    """
    banner("Step 0: Version drift check")
    found = read_versions()
    if not found:
        fail("No version strings found — nothing to check")
        return False

    first = next(iter(found.values()))
    all_ok = True
    for name in ("pyproject.toml", "core/version.py", "server/app.py",
                 "core/bytecode.py"):
        if name in found:
            if found[name] == first:
                ok(f"{name} → {found[name]}")
            else:
                fail(f"{name} → {found[name]} (drifted from {first})")
                all_ok = False
        else:
            fail(f"{name}: version string not found")
            all_ok = False

    if all_ok:
        ok(f"All {len(found)} version strings agree ({first})")
    else:
        print(f"\n{RED}Version drift detected. Fix it before releasing.{NC}",
              file=sys.stderr)
    return all_ok


# ─── Step 1: Version sync ────────────────────────────────────────────────────


def sync_version(version: str) -> bool:
    """Sync version to all source files. Returns True on success."""
    banner(f"Step 1: Sync version → {version}")

    replacements = [
        (ROOT / "pyproject.toml", re.compile(r'^version = ".*"', re.MULTILINE), f'version = "{version}"'),
        (ROOT / "src" / "helixlang" / "core" / "version.py", re.compile(r'^__version__ = ".*"', re.MULTILINE), f'__version__ = "{version}"'),
        (ROOT / "src" / "helixlang" / "server" / "app.py", re.compile(r'"version": ".*"'), f'"version": "{version}"'),
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
    bp = ROOT / "src" / "helixlang" / "core" / "bytecode.py"
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
    for path in [ROOT / "pyproject.toml", ROOT / "src" / "helixlang" / "core" / "version.py", ROOT / "src" / "helixlang" / "server" / "app.py"]:
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
        ("boundary", [PYTHON, "-m", "helixlang.core.find_core_imports", "--strict"], ROOT),
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
                        capture_output=True, text=True, cwd=ROOT,
                    )
                    if r.returncode != 0:
                        lines.append(f"COMPILE FAIL: {f}")
                        fail_count += 1
                # Run minimal example
                hello = examples_dir / "01_hello_dna.helix"
                if hello.exists():
                    r = subprocess.run(
                        [PYTHON, "-B", "-m", "helixlang", str(hello)],
                        capture_output=True, text=True, cwd=ROOT,
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
    order = {"ruff": 0, "mypy": 1, "boundary": 2, "pytest": 3, "val": 4,
             "examples": 5}
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


def _stale_metric_values(content: str, val_pass: str | int,
                         val_total: str | int) -> list[str]:
    """Confirm every validation-benchmark count occurrence matches the target
    (val_pass/val_total read from validation/report.md)."""
    vp, vt = int(val_pass), int(val_total)
    stale: list[str] = []
    for m in re.finditer(r"\| Validation benchmarks \| (\d+) \((\d+) pass\)", content):
        if (int(m.group(2)), int(m.group(1))) != (vp, vt):
            stale.append(f"table {m.group(1)} ({m.group(2)} pass)")
    for m in re.finditer(r"\| Benchmarks passing \| \*\*(\d+)/(\d+)\*\*", content):
        if (int(m.group(1)), int(m.group(2))) != (vp, vt):
            stale.append(f"passing {m.group(1)}/{m.group(2)}")
    for m in re.finditer(r"\[(\d+)/(\d+) validation benchmarks\]", content):
        if (int(m.group(1)), int(m.group(2))) != (vp, vt):
            stale.append(f"badge {m.group(1)}/{m.group(2)}")
    for m in re.finditer(r"(\d+) reproducible benchmarks", content):
        if int(m.group(1)) != vt:
            stale.append(f"reproducible {m.group(1)}")
    for m in re.finditer(r"\| 11-(\d+)\s*\|", content):
        if int(m.group(1)) != vt:
            stale.append(f"range 11-{m.group(1)}")
    for m in re.finditer(r"\*\*(\d+)/(\d+)\*\* benchmarks PASS\b", content):
        if (int(m.group(1)), int(m.group(2))) != (vp, vt):
            stale.append(f"PASS-bullet {m.group(1)}/{m.group(2)}")
    return stale


def _replacer(pattern: str, replacement: str, text: str) -> str:
    return re.sub(pattern, replacement, text)


def sync_metrics(gate_dir: Path) -> bool:
    banner("Step 3: Sync metrics to README / README_PYPI / CONTRIBUTING")

    # Extract pytest stats
    pytest_log = gate_dir / "pytest.log"
    test_count = ""
    if pytest_log.exists():
        content = pytest_log.read_text()
        m = re.search(r"(\d+) passed", content)
        if m:
            test_count = m.group(1)

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
    sync_failed = False
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

        # 4b. "N reproducible benchmarks with" (CONTRIBUTING) — "67 reproducible benchmarks with"
        if val_total:
            content = re.sub(
                r"\d+ reproducible benchmarks with",
                f"{val_total} reproducible benchmarks with",
                content,
            )

        # 5. Benchmark range in table — "| 11-45 |" or "| 11-67|" → "| 11-73 |"/"| 11-73|"
        if val_total:
            content = re.sub(
                r"(\| 11-)\d+(\s*\|)",
                rf"\g<1>{val_total}\g<2>",
                content,
            )

        # 5b. Linked badge — "[67/67 validation benchmarks]" → "[73/73 validation benchmarks]"
        if val_pass and val_total:
            content = re.sub(
                r"\[(\d+)/(\d+) validation benchmarks\]",
                f"[{val_pass}/{val_total} validation benchmarks]",
                content,
            )

        # 5c. PASS bullet (CONTRIBUTING) — "**67/67** benchmarks PASS" → "**73/73** benchmarks PASS"
        if val_pass and val_total:
            content = re.sub(
                r"\*\*(\d+)/(\d+)\*\* benchmarks PASS\b",
                f"**{val_pass}/{val_total}** benchmarks PASS",
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

        # Confirm every validation-benchmark count in the file is now synced.
        if val_pass and val_total:
            stale = _stale_metric_values(content, val_pass, val_total)
            if stale:
                for s in stale:
                    fail(f"{fname}: stale validation-benchmark count: {s}")
                sync_failed = True
            else:
                ok(f"{fname}: validation-benchmark counts synced "
                   f"({val_pass}/{val_total})")

    ok(f"Docs synced: tests={test_count or '?'}, val={val_pass or '?'}/{val_total or '?'}, modules={src_modules}, examples={example_count}")
    return not sync_failed


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

    # sdist
    r = subprocess.run(
        [PYTHON, "-B", "-m", "build", "--sdist"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)

    # Pure-Python wheel
    r = subprocess.run(
        [PYTHON, "-B", "-m", "build", "--wheel"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)

    # Native wheel (dual-wheel shipping, doc/36 Phase 4 item 2).  Requires
    # Cython on the build host and --no-isolation because Cython is not in
    # [build-system] requires (doc/36 §4.2.1).
    native_env = dict(os.environ, HELIX_BUILD_NATIVE="1")
    r = subprocess.run(
        [PYTHON, "-B", "-m", "build", "--wheel", "--no-isolation"],
        capture_output=True, text=True, cwd=ROOT, env=native_env,
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
    parser.add_argument(
        "version",
        help="Version in YYYY.M.D or YYYY.M.D.N format: D = 0-based iteration "
             "release of the month, N (optional) = patch of that iteration "
             "(e.g. 2026.9.1, 2026.9.1.2)",
    )
    parser.add_argument(
        "--check-versions", action="store_true",
        help="Only verify every version-bearing source agrees; write nothing.",
    )
    parser.add_argument(
        "--log-dir", metavar="DIR", default=None,
        help="Persist run + gate logs under DIR (default: release_logs/).",
    )
    args = parser.parse_args()

    version = args.version

    # Validate version format
    if not re.match(r"^\d{4}\.\d{1,2}\.\d{1,2}(\.\d+)?$", version):
        print(f"Error: Version must match YYYY.M.D or YYYY.M.D.N (got: {version})", file=sys.stderr)
        return 1

    os.chdir(ROOT)

    # Everything from here on is mirrored to a persistent per-run release.log,
    # and gate logs live in the same directory, so a failed run can be
    # diagnosed from disk afterwards.
    log_root = Path(args.log_dir) if args.log_dir else None
    run_dir, gates_dir = setup_run_log(version, log_root)
    start = datetime.now()

    results: list[GateResult] = []

    def finish(code: int, result: str) -> int:
        elapsed = (datetime.now() - start).total_seconds()
        write_summary(run_dir, version, result=result,
                      gates=results or None, elapsed=elapsed)
        sys.stdout.flush()
        sys.stderr.flush()
        return code

    try:
        # Step 0: Fail fast if a previous sync partially applied (doc/38 §2.3)
        if not check_versions():
            return finish(1, "ABORTED — version drift")

        if args.check_versions:
            return finish(0, "PASS — check-versions only")

        # Step 1: Sync version
        if not sync_version(version):
            return finish(1, "FAILED — version sync")

        # Step 2: Quality gates (logs persist in gates_dir/)
        results = run_quality_gates(gates_dir)

        failed = [r for r in results if r.exit_code != 0]
        if failed:
            return finish(1, "FAILED — some gates failed")

        # Step 2b: Generate report
        generate_report()

        # Step 3: Sync metrics
        if not sync_metrics(gates_dir):
            return finish(1, "FAILED — metric sync")

        # Step 4: Build
        if not build():
            return finish(1, "FAILED — build")

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

        return finish(0, "SUCCESS — release complete")
    except KeyboardInterrupt:
        return finish(2, "ABORTED — keyboard interrupt")
    except Exception as e:  # noqa: BLE001 — top-level release guard
        import traceback
        traceback.print_exc()
        fail(f"Unhandled error: {e}")
        return finish(1, f"ERROR — {type(e).__name__}")


if __name__ == "__main__":
    sys.exit(main())
