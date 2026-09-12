"""Documentation code-anchor truth gate (CI + release quality gate).

Guards the project rule "code is the only source of truth": every ``.py:LINE``
anchor in the design documents must resolve to a real file and a real line in
the current tree.  An anchor that names a file which no longer exists, lands on
a path that has moved, or points past the end of a file means the prose has
drifted from the implementation and the document must be corrected — the doc is
fixed to match the code, never the other way round.

Resolution rules (mirroring how the docs address code):

- an anchor path containing a directory (e.g. ``plugins/human/__init__.py``)
  is resolved against ``src/helixlang/`` first and ``<repo root>/`` second;
- a bare filename (e.g. ``population.py``) is resolved by unique basename
  search underneath ``src/helixlang/`` — an anchor that matches several files
  is reported as ambiguous, never silently resolved;
- the ``:START`` and optional ``:START-END`` line span must lie inside the
  bounds of the resolved file.

CI and ``release.py`` run it with ``--fail``; any finding fails the gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "src" / "helixlang"
VALIDATION = ROOT / "validation"
VALIDATION_BENCHMARKS = VALIDATION / "benchmarks"

#: Bases a slash-bearing anchor path may be rooted at, in priority order.
_SEARCH_BASES = (SRC, ROOT, VALIDATION_BENCHMARKS, VALIDATION)

#: Bases a bare-filename anchor is searched under, in priority order.
_BASENAME_ROOTS = (SRC, VALIDATION_BENCHMARKS, VALIDATION)

# ``path.py:12`` or ``path.py:12-34``; the basename chunk accepts dots, slashes,
# letter/digit/underscore/hyphen so both ``parser.py:60`` and
# ``plugins/human/__init__.py:34`` are captured.
_ANCHOR = re.compile(r"(?P<file>[A-Za-z0-9_./+-]+\.py):"
                     r"(?P<start>\d+)(?:-(?P<end>\d+))?")


@dataclass(slots=True)
class Anchor:
    """One code anchor extracted from a document line."""
    source: str
    lineno: int
    raw: str
    file: str
    start: int
    end: int | None

    def display(self) -> str:
        if self.end is None:
            return f"{self.file}:{self.start}"
        return f"{self.file}:{self.start}-{self.end}"


@dataclass(slots=True)
class Finding:
    """A non-resolvable (or ambiguous / out-of-bounds) code anchor."""
    path: str
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.detail}"


def _match_text(text: str, source: str | Path) -> list[Anchor]:
    out: list[Anchor] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        for m in _ANCHOR.finditer(raw):
            end = int(m.group("end")) if m.group("end") else None
            out.append(Anchor(
                source=str(source),
                lineno=i,
                raw=raw.strip(),
                file=m.group("file"),
                start=int(m.group("start")),
                end=end,
            ))
    return out


def resolve_file(filepath: str) -> str | None:
    """Resolve an anchor's file path to an absolute path.

    Returns the resolved absolute path, ``"AMBIGUOUS"`` when a basename
    matches more than one module, or ``None`` when nothing resolves.
    """
    anchor = Path(filepath)
    if "/" in filepath or "\\" in filepath:
        for base in _SEARCH_BASES:
            candidate = base / anchor
            if candidate.is_file():
                return str(candidate.resolve())
        return None
    return _resolve_basename(anchor.name)


def _resolve_basename(name: str) -> str | None:
    for base in _BASENAME_ROOTS:
        if not base.is_dir():
            continue
        matches = list(base.rglob(name))
        if len(matches) > 1:
            return "AMBIGUOUS"
        if len(matches) == 1:
            return str(matches[0].resolve())
    shallow = list(ROOT.glob(name))
    if len(shallow) == 1:
        return str(shallow[0].resolve())
    return None


def validate_anchor(anchor: Anchor) -> str | None:
    """Return a finding detail for an unverifiable anchor, else ``None``."""
    target = resolve_file(anchor.file)
    if target is None:
        return f"anchor {anchor.display()!r} → no such file in the tree"
    if target == "AMBIGUOUS":
        return f"anchor {anchor.display()!r} → ambiguous basename match"
    try:
        with open(target, encoding="utf-8") as fh:
            line_count = sum(1 for _ in fh)
    except OSError as exc:
        return f"anchor {anchor.display()!r} → unreadable: {exc}"
    if anchor.start < 1:
        return f"anchor {anchor.display()!r} → start line below 1"
    if anchor.start > line_count:
        return (f"anchor {anchor.display()!r} → start line {anchor.start} "
                f"beyond file length ({line_count} lines)")
    if anchor.end is not None:
        if anchor.end < anchor.start:
            return f"anchor {anchor.display()!r} → end before start"
        if anchor.end > line_count:
            return (f"anchor {anchor.display()!r} → end line {anchor.end} "
                    f"beyond file length ({line_count} lines)")
    return None


def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    findings: list[Finding] = []
    for anchor in _match_text(text, path):
        detail = validate_anchor(anchor)
        if detail is not None:
            findings.append(Finding(str(path), anchor.lineno, detail))
    return findings


def scan_tree(root: Path, *, skip: tuple[str, ...] = ()) -> list[Finding]:
    out: list[Finding] = []
    for p in sorted(root.rglob("*.md")):
        if any(part in skip for part in p.parts):
            continue
        out.extend(scan_file(p))
    return out


def format_report(findings: list[Finding]) -> str:
    if not findings:
        return "All doc code anchors resolve to the current tree."
    return "\n".join(str(f) for f in findings)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="docs_truth")
    ap.add_argument("roots", nargs="*", default=["doc"],
                    help="directories (or files) to scan")
    ap.add_argument("--fail", action="store_true",
                    help="exit 1 when findings exist (CI / release)")
    ap.add_argument("--skip", action="append", default=[],
                    metavar="PART", help="skip paths containing PART "
                    "(repeatable)")
    args = ap.parse_args(argv)
    skip = tuple(args.skip)
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
