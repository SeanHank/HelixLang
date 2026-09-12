"""Documentation implementation-claim audit (CI + release quality gate).

Scans markdown documents (default ``doc/``) and flags any prose that presents
Helix functionality as *planned, proposed, deferred, blocked, out of scope,
partial, simplified, or otherwise not truly implemented and wired in*
(English and Chinese markers).  The invarant it guards (doc/06 §12, project
rule): every statement about the compiler / VM / plugins must describe the
implemented reality in detail — not a to-do, a deferral, a roadmap promise,
or a "not in scope" carve-out.  A statement that cannot be made as an
implemented fact must be removed, not hedged.

CI and ``release.py`` run it with ``--fail``; any finding fails the gate so a
non-implemented claim can never ship in the documents.

By design *not* flagged — these are not implementation claims:

- fenced code blocks (```…```) and inline code spans (``` ``…`` ``) — example
  configs / identifiers, not claims;
- HTML comments (``<!-- … -->``).

Genuinely-exceptional prose (a verbatim external quote, a documented third-party
dependency fence) may be marked with the bare token ``DOCBENIGN`` on that line
plus a one-line reason, mirroring the ``STUBBENIGN`` / ``SILENTBENIGN`` tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Category metadata.  The web-style catalog makes the report self-explanatory.
CATALOG: dict[str, str] = {
    "D1": "planned/proposed/to-be-implemented claim",
    "D2": "deferred/postponed/on-hold claim",
    "D3": "blocked-as-status claim",
    "D4": "out-of-scope / not-in-this-revision carve-out",
    "D5": "partial-implementation admission",
    "D6": "work marker (TODO/TBD/- [ ] unchecked item)",
    "D7": "roadmap / future-work / not-yet claim (EN + 中文)",
    "D8": "status banner (PROPOSED / Draft for review / pending)",
}

# An explicit audit token; see module docstring.  Bare word on purpose.
_BENIGN = re.compile(r"\bDOCBENIGN\b")

# Inline code spans are stripped before matching: ``#!/bin/sh`` is an example,
# not a claim.
_INLINE_CODE = re.compile(r"`[^`]*`")

# Fence / HTML-comment delimiters live on their own stripped lines.
_FENCE = re.compile(r"^```")
_HTML_OPEN = re.compile(r"^<!--")
_HTML_CLOSE = re.compile(r"-->\s*$")

# (category, regex) in priority order; first match wins per line.
_RULES: list[tuple[str, str]] = [
    # D1 — planned / proposed / will-be…
    ("D1",
     r"\b(?:planned|planning to|plan to|to[- ]be[- ]implemented|"
     r"will (?:be )?(?:implemented|shipped|landed|added|built|wired|supported|"
     r"released|available|provided)|"
     r"not yet implemented|not implemented|unimplemented|"
     r"proposal only|design proposal only)\b"),
    # D2 — deferred / postponed / on-hold…  (STUBBENIGN: "deferred" is the
    # audit catalog's own category name here, not a shipped-code marker)
    ("D2",
     r"\b(?:deferred|postponed|on hold|on-hold|carry forward|carried forward)\b"),
    # D3 — status "blocked" (word-sense guarded: “flux fully blocked” is not a
    # claim; “is blocked / blocked by X / still blocked / remains blocked” is).
    ("D3",
     r"\b(?:is|remains?|still|currently|now|was)\s+blocked\b|\bblocked by\b|\b"
     r"block(?:s|ed)?\s+(?:this|the|its)\s+(?:release|goal|work|landing)\b"),
    # D4 — scope carve-outs.
    ("D4",
     r"\bout[- ]of[- ]scope\b|\bnot in scope\b|\boutside (?:the|this)? scope\b|"
     r"\bnot in this (?:revision|release|version|doc)\b"),
    # D5 — partial-implementation admissions.
    ("D5",
     r"\bpartial(?:ly)?\s+(?:implementation|implemented|shipping|shipped|"
     r"vectorized|vectorised|coverage|support(?:ed)?|landed|done|status|"
     r"completion|degree)\b|partially shipped|⚠️\s*partial|🟨\s*partial|"
     r"部分(?:落实|实现|接入|覆盖|支持)"),
    # D6 — work markers and unchecked task checkboxes.
    ("D6",
     r"\b(?:TODO|FIXME|TBD|WIP)\b|todo:|to[-\s]do:\s|not (?:yet )?started|"
     r"left unchecked|- \[ \]"),
    # D7 — roadmap / future / not-yet claims.
    ("D7",
     r"\b(?:roadmap|future work|future expansion|extension roadmap|"
     r"subsequent roadmap|later migrate|to be extended)\b|"
     r"计划内|计划中|后续计划|实施计划|待实现|尚未(?:实现|落地|接入)|"
     r"暂(?:缓|不|未)|预留|占位|简化落实|简化实现|未接入"),
    # D8 — status banners.  “pending” is only a claim as a *status* (“pending
    # follow-ups”, “still pending”); VM terms like “pending frames” are not.
    ("D8",
     r"\b(?:PROPOSED|Draft for review|pending (?:follow-ups?|list|work|items?)|"
     r"still pending|awaiting)\b"),
]

_RULE_RE: list[tuple[str, re.Pattern[str]]] = [
    (cat, re.compile(expr, re.I)) for cat, expr in _RULES
]


@dataclass(slots=True)
class Finding:
    path: str
    lineno: int
    category: str | None
    detail: str

    def __str__(self) -> str:
        cat = f"[{self.category}] " if self.category else "[?] "
        return f"{self.path}:{self.lineno}: {cat}{self.detail}"


def scan_text(text: str, path: str | Path) -> list[Finding]:
    """Scan in-memory markdown for non-implementation claims."""
    findings: list[Finding] = []
    in_fence = False
    in_html = False
    for i, raw in enumerate(text.splitlines(), start=1):
        if _BENIGN.search(raw):
            continue
        stripped = raw.strip()
        if _HTML_OPEN.search(stripped):
            in_html = True
            if _HTML_CLOSE.search(stripped):
                in_html = False
            continue
        if in_html:
            if _HTML_CLOSE.search(stripped):
                in_html = False
            continue
        if _FENCE.search(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = _INLINE_CODE.sub(" ", raw)
        for cat, rx in _RULE_RE:
            if rx.search(line):
                findings.append(
                    Finding(str(path), i, cat, raw.strip()))
                break
    return findings


def scan_file(path: Path) -> list[Finding]:
    """Scan one markdown file for non-implementation claims."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not text.strip():
        return []
    return scan_text(text, path)


def scan_tree(root: Path, *, skip: tuple[str, ...] = ()) -> list[Finding]:
    """Recursively scan ``*.md`` files under ``root``."""
    out: list[Finding] = []
    for p in sorted(root.rglob("*.md")):
        if any(part in skip for part in p.parts):
            continue
        out.extend(scan_file(p))
    return out


def format_report(findings: list[Finding]) -> str:
    if not findings:
        return "No non-implementation claims found."
    return "\n".join(str(f) for f in findings)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="audit_docs")
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
