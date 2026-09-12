"""Stub / unimplemented / deferred-code linter (CI + release quality gate).

Companion to :mod:`helixlang.core.find_silent_fallbacks` (which hunts for
*silently* degraded computations).  This linter hunts the inverse failure
mode: code that was left *visibly* unfinished -- ``...``/``pass``-only
bodies, ``raise NotImplementedError``, and comment markers confessing that a
piece was simplified, stubbed, or deferred.

CI and ``release.py`` run it with ``--fail``; any finding fails the gate so a
stub can never ship silently.  Resolve findings the honest way:

* implement the thing (preferred), or
* if a case is intentionally kept -- a documented no-op hook, a stand-in
  error type, a deliberately defensive default -- audit it explicitly with
  the ``STUBBENIGN`` token on that line (or the line directly above) plus a
  one-line reason, e.g.::

      def _flush_morphology(self) -> None:
          # STUBBENIGN no-op hook: morphology handled inline in _dispatch
          pass

The token is a bare word (like the silent-fallback linter's SILENTBENIGN) so
ruff does not parse it as one of its own ``noqa`` directives, and this
linter's own stub-family regexes never match it.

Not flagged by design: docstring prose (including marker words used as
examples), ``...`` bodies inside :class:`~typing.Protocol` classes (canonical
typing syntax, not a stub), and ``tests/`` (mocks/fakes are normal test
tooling, not shipped product code).
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

# Category catalog (mirrors find_silent_fallbacks' CATALOG style).
CATALOG: dict[str, str] = {
    "S1": "function body is only an ellipsis (``...``) — unimplemented",
    "S2": "function body is only ``pass`` — empty/stub function",
    "S3": "raises NotImplementedError — explicitly unimplemented path",
    "S4": "comment marker TODO/FIXME/XXX/HACK/TBD/WIP — deferred work item",
    "S5": "comment admits \"not implemented\"/\"unimplemented\" — deferred item",
    "S6": "comment marker stub/placeholder/dummy/deferred — simplified/stub item",
}

# An explicit audit token; see module docstring.  Bare word on purpose.
_BENIGN = re.compile(r"\bSTUBBENIGN\b")

# Bare comment markers that confess unfinished work still in the code.
_WORK_MARKERS = re.compile(r"\b(TODO|FIXME|XXX|HACK|TBD|WIP)\b")
_UNIMPL_MARKERS = re.compile(r"\b(not yet implemented|not implemented|unimplemented)\b")
_STUB_MARKERS = re.compile(r"\b(stub|placeholder|dummy|deferred)\b")

# A negation must DIRECTLY precede the marker (within a short gap) to count as
# "this is NOT a stub" prose (e.g. "never a placeholder"); negation elsewhere
# in the comment does not suppress the finding ("placeholder, not used if IV"
# is still a real stand-in worth auditing).
_NEGATED_GAP = r"[^#!]{0,32}?"
_NEGATION = re.compile(
    rf"\b(no|never|not|n't|without|rather than|instead of|no longer|no more)\b"
    rf"{_NEGATED_GAP}(stub|placeholder|dummy|deferred)\b",
    re.I,
)


@dataclass(slots=True)
class Finding:
    path: str
    lineno: int
    category: str | None
    detail: str

    def __str__(self) -> str:
        cat = f"[{self.category}] " if self.category else "[?] "
        return f"{self.path}:{self.lineno}: {cat}{self.detail}"


class _Visitor(ast.NodeVisitor):
    """AST pass: ellipsis/pass-only bodies and explicit NotImplementedError."""

    def __init__(self, lines: list[str]) -> None:
        self.findings: list[Finding] = []
        self._lines = lines
        self._protocol_depth = 0

    def _report(self, node: ast.AST, cat: str, detail: str) -> None:
        self.findings.append(
            Finding("", int(getattr(node, "lineno", 0)), cat, detail))

    @staticmethod
    def _is_protocol_class(node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Protocol":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "Protocol":
                return True
        return False

    def _has_benign(self, node: ast.AST) -> bool:
        start = max(1, getattr(node, "lineno", 1) - 6)
        end = getattr(node, "end_lineno", None)
        end = end if end is not None else start
        for i in range(start, end + 1):
            if i <= len(self._lines) and _BENIGN.search(self._lines[i - 1]):
                return True
        return False

    def _body_wo_docstring(self, node: ast.FunctionDef) -> list[ast.stmt]:
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            body = body[1:]
        return body

    def _check_fn(self, node: ast.FunctionDef) -> None:
        if self._protocol_depth:
            return
        body = self._body_wo_docstring(node)
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass) and not self._has_benign(node):
                self._report(node, "S2",
                             "function body is only `pass` (empty/stub)")
            elif isinstance(stmt, ast.Expr) and \
                    isinstance(stmt.value, ast.Constant) and \
                    stmt.value.value is Ellipsis and not self._has_benign(node):
                self._report(node, "S1",
                             "function body is only `...` (unimplemented stub)")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_fn(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_fn(node)  # type: ignore[arg-type]
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._is_protocol_class(node):
            self._protocol_depth += 1
        try:
            self.generic_visit(node)
        finally:
            if self._is_protocol_class(node):
                self._protocol_depth -= 1

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        name = None
        if isinstance(exc, ast.Name):
            name = exc.id
        elif isinstance(exc, ast.Call):
            fn = exc.func
            if isinstance(fn, ast.Name):
                name = fn.id
        if name == "NotImplementedError" and not self._has_benign(node):
            self._report(node, "S3",
                         "raises NotImplementedError (explicitly "
                         "unimplemented path)")
        self.generic_visit(node)


def _scan_comments(lines: list[str]) -> list[Finding]:
    """Tokenize pass: comment markers that confess a stub/deferred item."""
    findings: list[Finding] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO("\n".join(lines)).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            row = tok.start[0]
            text = tok.string
            if _BENIGN.search(text):
                continue
            if row > 1 and _BENIGN.search(lines[row - 2]):
                continue
            cat: str | None = None
            if _WORK_MARKERS.search(text):
                cat = "S4"
            elif _UNIMPL_MARKERS.search(text):
                cat = "S5"
            elif _STUB_MARKERS.search(text) and not _NEGATION.search(text):
                cat = "S6"
            if cat:
                findings.append(Finding("", row, cat, text.strip()))
    except (tokenize.TokenError, IndentationError):  # SILENTBENIGN parse-tolerance probe
        pass
    return findings


def scan_file(path: Path) -> list[Finding]:
    """Scan one Python file for stub/unimplemented/deferred patterns."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not src:
        return []
    lines = src.splitlines()
    findings: list[Finding] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        v = _Visitor(lines)
        v.visit(tree)
        findings.extend(v.findings)
    findings.extend(_scan_comments(lines))
    findings.sort(key=lambda f: f.lineno)
    return [Finding(str(path), f.lineno, f.category, f.detail)
            for f in findings]


def scan_tree(root: Path, *, skip: tuple[str, ...] = ("tests",)) -> list[Finding]:
    out: list[Finding] = []
    for p in sorted(root.rglob("*.py")):
        if any(part in skip for part in p.parts):
            continue
        out.extend(scan_file(p))
    return out


def format_report(findings: list[Finding]) -> str:
    if not findings:
        return "No stub/deferred-code patterns found."
    return "\n".join(str(f) for f in findings)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="find_stubs")
    ap.add_argument("roots", nargs="*", default=["src"],
                    help="directories (or files) to scan")
    ap.add_argument("--fail", action="store_true",
                    help="exit 1 when findings exist (CI / release)")
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
