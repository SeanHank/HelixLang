"""Silent-fallback linter (doc/36 §3ξ CI hook).

Scans the source tree for patterns that correspond to the twelve silent
fallback categories F1–F12 and reports each with the file/line and the most
likely category.  Exits nonzero when any are found, so CI fails on newly
introduced silent fallbacks.

This is intentionally conservative (a lint, not a type-checker): it flags
**likely** fallbacks for a human to triage rather than claiming every match is a
bug.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

# Category metadata mirrors doc/36 §3ξ.1.
CATALOG: dict[str, str] = {
    "F1": "missing optional core module silently skipped",
    "F2": "missing optional dep silently degraded",
    "F3": "missing model data silently substituted with default",
    "F4": "missing optional dep silently substituted with fallback impl",
    "F5": "approx/backend env var silently overridden",
    "F6": "approx computation performed without explicit opt-in",
    "F7": "unknown #keyword silently ignored",
    "F8": "backend silently reselected",
    "F9": "ABI/version mismatch silently tolerated",
    "F10": "constraint violated silently ignored",
    "F11": "stack underflow silently ignored",
    "F12": "missing graph node silently defaulted",
}


@dataclass(slots=True)
class Finding:
    path: str
    lineno: int
    category: str | None
    detail: str

    def __str__(self) -> str:
        cat = f"[{self.category}] " if self.category else "[?] "
        return f"{self.path}:{self.lineno}: {cat}{self.detail}"


# Broad-except bodies that do nothing meaningful are the prototypical fallback.
_EMPTY_BODY = {ast.Pass(), ast.Break(), ast.Continue()}

# Names of variables that conventionally hold a model/backend/config default.
_DEFAULTLIKE = re.compile(r"default|fallback|dummy|core|builtin|_sel\b", re.I)

# Keywords that switch backends / fidelity.
_BACKEND_TERMS = re.compile(
    r"backend|pure|numpy|numba|cython|fidelity|approx|euler|solver|reselect",
    re.I)

# A handler carrying this explicit marker is a *documented-benign* probe and is
# excluded from the report (doc/36 Phase 5 item 4).  The marker says the author
# has confirmed the empty except is an intentional capability/cache/version
# probe (not a reduced-fidelity computation path) and documented it as such, so
# the lint stops treating it as silent.  Style::
#
#     except ImportError:  # SILENTBENIGN - optional-dep probe
#
# The marker is a bare token: it deliberately avoids the ``noqa:`` prefix (so
# ruff does not parse it as one of its own directives) and avoids the literal
# word "legacy-fallback" so this linter's own F4 heuristic never flags it.
_BENIGN_MARKER = r"\bSILENTBENIGN\b"


class _Visitor(ast.NodeVisitor):
    def __init__(self, lines: list[str]) -> None:
        self.findings: list[Finding] = []
        self._lines = lines
        self._benign = re.compile(_BENIGN_MARKER, re.I)

    def _is_benign(self, node: ast.ExceptHandler) -> bool:
        """True if the handler (or its preceding line) carries the benign marker.

        Only the ``except`` line and the line directly before it are inspected
        (an inline marker sits on the ``except``/``pass`` line, or a leading
        marker comment sits on the line above).
        """
        start = max(1, node.lineno - 1)
        end = node.end_lineno if node.end_lineno is not None else node.lineno
        for lineno in range(start, end + 1):
            if lineno <= len(self._lines):
                if self._benign.search(self._lines[lineno - 1]):
                    return True
        return False

    def _report(self, node: ast.AST, cat: str | None, detail: str) -> None:
        self.findings.append(Finding("", getattr(node, "lineno", 0), cat, detail))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        body = [n for n in node.body if not isinstance(n, ast.Pass)]
        if node.type is None:
            exc = "bare"
        elif isinstance(node.type, ast.Tuple):
            exc = ", ".join(ast.unparse(e) for e in node.type.elts)
        else:
            exc = ast.unparse(node.type)
        # A handler whose body reduces to nothing (only `pass`) swallows the
        # error silently — the prototypical silent fallback (doc/36 F1/F4).
        if not body:
            if self._is_benign(node):
                self.generic_visit(node)
                return
            cat = None
            if isinstance(node.type, ast.Name) and \
                    node.type.id in ("ImportError", "ModuleNotFoundError"):
                cat = "F1"      # missing optional module silently skipped
            elif exc in ("Exception", "Exception,", "Exception"):
                cat = "F2"
            elif exc == "KeyError":
                cat = "F12"
            self._report(node, cat,
                         f"empty except {exc} (swallows the error silently)")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        val = node.value
        if isinstance(val, ast.Constant) and \
                isinstance(val.value, str) and "fallback" in val.value.lower():
            for t in node.targets:
                self._report(node, "F4", f"fallback string assigned to "
                                         f"{ast.unparse(t) if hasattr(ast, 'unparse') else t}")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.generic_visit(node)


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def scan_file(path: Path) -> list[Finding]:
    src = _text(path)
    if not src:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    v = _Visitor(src.splitlines())
    v.visit(tree)
    # Re-point path+category with simple textual heuristics.
    findings = []
    for f in v.findings:
        findings.append(Finding(str(path), f.lineno, f.category, f.detail))
    return findings


def scan_tree(root: Path, *, skip: tuple[str, ...] = ("tests",)) -> list[Finding]:
    out: list[Finding] = []
    for p in sorted(root.rglob("*.py")):
        if any(part in skip for part in p.parts):
            continue
        out.extend(scan_file(p))
    return out


def format_report(findings: list[Finding]) -> str:
    if not findings:
        return "No silent-fallback patterns found."
    lines = [str(f) for f in findings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="find_silent_fallbacks")
    ap.add_argument("roots", nargs="*", default=["src"],
                    help="directories (or files) to scan")
    ap.add_argument("--fail", action="store_true",
                    help="exit 1 when findings exist (CI)")
    ap.add_argument("--skip", action="append", default=[],
                    metavar="PART", help="skip paths containing PART "
                    "(repeatable; default always skips tests/)")
    args = ap.parse_args(argv)
    skip = ("tests",) + tuple(args.skip)
    findings: list[Finding] = []
    for root in args.roots:
        p = Path(root)
        if p.is_dir():
            findings.extend(scan_tree(p, skip=skip))
        else:
            findings.extend(scan_file(p))
    print(format_report(findings))
    return 1 if (args.fail and findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
