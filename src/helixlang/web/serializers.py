"""HelixLang web serialization helpers.

Split out from ``server.py`` to centralize pure serialization and parsing logic
for Program / CellVM / trace / morphology / field / cell / L-system rules.
``server.py`` imports them via

    from helixlang.web.serializers import (
        _parse_lsystem_rules,
        _serialize_program_summary,
        _serialize_grn,
        _serialize_trace,
        _serialize_morphology,
        _serialize_field,
        _serialize_cell,
    )

These functions duck-type the objects passed in and do not depend directly on
the CellVM/Program types, avoiding circular imports.
"""
from __future__ import annotations

from typing import Any


def _parse_lsystem_rules(s: str) -> dict[str, str]:
    """Parse an L-system rule string into a {symbol: production} dict.

    Supports two separator formats:
      - "F->F[+F]F[-F]F;X->FX"  (arrow-separated, multiple rules separated by ;)
      - "F:F[+F]F[-F]F;X:FX"    (colon-separated)
    """
    if not s:
        return {}
    rules: dict[str, str] = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        sep = "->" if "->" in part else (":" if ":" in part else None)
        if sep is None:
            continue
        sym, _, prod = part.partition(sep)
        sym = sym.strip()
        if sym and len(sym) == 1:
            rules[sym] = prod.strip()
    return rules


def _serialize_program_summary(program: Any) -> dict:
    """Serialize a Program summary for the frontend."""
    return {
        "genes": [
            {
                "name": g.name,
                "promoter": g.promoter,
                "orf": [c.seq for c in g.orf],
                "orf_length": len(g.orf),
            }
            for g in program.genes
        ],
        "promoters": [
            {"name": p.name, "strength": p.strength} for p in program.promoters
        ],
        "regulations": [
            {"source": r.source, "target": r.target, "strength": r.strength}
            for r in program.regulations
        ],
        "lsystems": [
            {"name": n, "axiom": d.axiom, "angle": d.angle, "step": d.step}
            for n, d in program.lsystems.items()
        ],
        "has_field": program.field_decl is not None,
        "field": (
            {
                "size": program.field_decl.size,
                "F": program.field_decl.F,
                "k": program.field_decl.k,
                "Du": program.field_decl.Du,
                "Dv": program.field_decl.Dv,
            }
            if program.field_decl
            else None
        ),
        "config": {
            "ticks": program.config.ticks,
            "table": program.config.table,
            "ops_per_tick": program.config.ops_per_tick,
            "react_steps": program.config.react_steps,
        },
    }


def _serialize_grn(vm: Any) -> dict:
    """GRN nodes + edges, for the frontend network graph."""
    nodes = [
        {
            "id": name,
            "name": name,
            "level": round(node.level, 4),
            "threshold": round(node.threshold, 4),
            "value": round(node.level, 4),  # used for ECharts node size
        }
        for name, node in vm.grn.nodes.items()
    ]
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "weight": round(e.weight, 4),
            "sign": "positive" if e.weight >= 0 else "negative",
        }
        for e in vm.grn.edges
    ]
    return {"nodes": nodes, "edges": edges}


def _serialize_trace(trace: list[dict]) -> dict:
    """Time series data: energy, morphology point count, field_total_v, per-gene level."""
    ticks = [s["tick"] for s in trace]
    # collect all gene/protein keys that appear
    gene_keys: set[str] = set()
    protein_keys: set[int] = set()
    for s in trace:
        gene_keys.update(s["gene_levels"].keys())
        protein_keys.update(s["proteins"].keys())
    gene_levels: dict[str, list[float]] = {}
    for gk in sorted(gene_keys):
        gene_levels[gk] = [
            round(s["gene_levels"].get(gk, 0.0), 4) for s in trace
        ]
    proteins: dict[str, list[float]] = {}
    for pk in sorted(protein_keys):
        proteins[str(pk)] = [
            round(s["proteins"].get(pk, 0.0), 4) for s in trace
        ]
    series = {
        "ticks": ticks,
        "energy": [s["energy"] for s in trace],
        "morphology_points": [s["morphology_points_count"] for s in trace],
        "field_total_v": [round(s["field_total_v"], 4) for s in trace],
        "gene_levels": gene_levels,
        "proteins": proteins,
    }
    return series


def _serialize_morphology(vm: Any) -> dict:
    """L-system morphology point sequence."""
    return {
        "points": [
            [round(x, 2), round(y, 2)] for x, y in vm.cell.morphology_points
        ],
        "count": len(vm.cell.morphology_points),
    }


def _serialize_field(vm: Any) -> dict | None:
    """Reaction-diffusion field U/V 2D arrays (frontend heatmap)."""
    if vm.field is None:
        return None
    n = vm.field.n
    return {
        "n": n,
        "u": [[round(vm.field.u[i][j], 4) for j in range(n)] for i in range(n)],
        "v": [[round(vm.field.v[i][j], 4) for j in range(n)] for i in range(n)],
    }


def _serialize_cell(vm: Any) -> dict:
    c = vm.cell
    return {
        "name": c.name,
        "x": c.x,
        "y": c.y,
        "energy": c.energy,
        "alive": c.alive,
        "color": list(c.color),
        "age": c.age,
        "divisions": c.divisions,
        "proteins": {str(k): round(v, 4) for k, v in c.proteins.items()},
        "slots_used": sum(1 for s in c.slots if s is not None),
    }
