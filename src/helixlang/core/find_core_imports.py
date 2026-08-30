"""Plugin import-boundary audit (doc/38 §6.2 / §6.9 E5).

A mechanical — not conventional — enforceability rule for the §6 design
constraint: a plugin may import *only* from ``helixlang.api.*``,
``helixlang.core.errors`` (the two halves of the frozen §6.2 surface), stdlib
and third-party packages.  A naked import of ``helixlang.core.*``,
``helixlang.interop`` or ``helixlang.sim_runtime.*`` is a violation.

Bundled plugins that still reach into core are listed in
``KNOWN_COMPLIANT_EXCEPTIONS`` while the §6.9 migration (E2→E4) moves them to
the public surface; the E5 gate runs this scanner in ``--strict`` mode, in
which *any* known exception becomes a hard failure — the allowlist ends at
zero exemptions.

Usage::

    python -m helixlang.core.find_core_imports [root] [--strict] [--json]

Exit code is non-zero when any hard violation is found (or any known
exception under ``--strict``).
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path

#: The frozen public surface a plugin may legally import (§6.2).
ALLOWED_ROOTS = ("helixlang.api", "helixlang.core.errors")

#: Modules migrated in §6.9 E2–E5; the E5 gate must drop these to zero.
#: Grouped by the migration step that removes them:
#:   E4 -> core.{compiler,lexer,parser,semantic} (apps/consortium.py re-run),
#:         core.{opcode_semantics} + ast_nodes/bytecode (runtime/population.py),
#:         sim_runtime (gem/bridge.py), bare core namespace (gem/*, human/*),
#:         helixlang._accel grn native backend (runtime/grn.py)
KNOWN_COMPLIANT_EXCEPTIONS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Violation:
    module: str
    path: Path
    lineno: int
    known: bool


def _module_names(tree: ast.AST) -> list[tuple[str | None, int]]:
    """Yield (module, lineno) for every import statement in ``tree``."""
    names: list[tuple[str | None, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative intra-package import: allowed
                continue
            if node.module:
                names.append((node.module, node.lineno))
            else:
                for alias in node.names:
                    names.append((alias.name, node.lineno))
    return names


def scan(paths: list[Path]) -> list[Violation]:
    """Scan source trees; return import-boundary violations."""
    violations: list[Violation] = []

    def _scan_file(target: Path) -> None:
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"),
                             filename=str(target))
        except (OSError, SyntaxError):
            return
        for module, lineno in _module_names(tree):
            if not module:
                continue
            root = module.split(".")[0]
            if root not in ("helixlang",) or root == "":
                continue                       # stdlib / third-party: allowed
            if module.startswith(ALLOWED_ROOTS):
                continue                       # frozen public surface
            if module.startswith("helixlang.plugins"):
                continue                       # sibling plugin imports
            violations.append(Violation(module, target, lineno,
                                        module in KNOWN_COMPLIANT_EXCEPTIONS))

    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*.py")):
                _scan_file(f)
        elif p.is_file():
            _scan_file(p)
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="find_core_imports")
    parser.add_argument("roots", nargs="*",
                        default=["src/helixlang/plugins"])
    parser.add_argument("--strict", action="store_true",
                        help="treat KNOWN_COMPLIANT_EXCEPTIONS as violations")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    violations = scan([Path(r) for r in args.roots])
    if not args.strict:
        violations = [v for v in violations if not v.known]

    if args.json:
        print(json.dumps({
            "violations": [
                {"module": v.module, "path": str(v.path), "line": v.lineno,
                 "known": v.known}
                for v in violations
            ],
            "strict": args.strict,
        }, indent=2))
        return 1 if violations else 0

    for v in sorted(violations, key=lambda v: (str(v.path), v.lineno)):
        print(f"{v.path}:{v.lineno}: {v.module}")
    if violations:
        print(f"\n{len(violations)} import-boundary violation(s)")
    else:
        n = sum(1 for v in scan([Path(r) for r in args.roots]))
        print(f"clean: {n} core import(s) outside helixlang.api "
              f"(known-exempt)" if n else "clean: zero open imports")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
