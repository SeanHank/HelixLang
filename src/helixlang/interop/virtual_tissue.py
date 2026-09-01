"""Virtual-tissue / PhysiCell-style spatial interop (doc/42 Phase E, gap RT-6).

A lightweight, stdlib-only exchange format for agent-based tissue state in the
PhysiCell idiom: a JSON document with (a) ``cells`` — each with a 3-D position,
a phenotype (cycle phase, volume, cell type, custom real variables) and an
agent id — and (b) ``substrates`` — a microenvironment concentration field
sampled on a regular grid (origin + spacing + per-substrate 3-D array).

Also provides a simple comma-separated (CSV) cell dump matching the classic
PhysiCell "cells.csv" columns: x, y, z, cell_type, cycle_phase, volume, plus
custom variables.

Stdlib-only (``json`` + ``csv``).
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

from helixlang.core.errors import BioError

#: Canonical PhysiCell-style cell.CSV column header (name -> index).
_CELL_CSV_COLUMNS = (
    "position_x", "position_y", "position_z",
    "cell_type", "cycle_phase", "volume",
)

#: Substrate field may be serialized as a flat list (row-major) or nested lists.
_DEFAULT_SPACING = 20.0


@dataclass
class VirtualCell:
    """One agent: position, phenotype, and custom scalar variables."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    cell_type: int = 0
    cycle_phase: str = "G1"
    volume: float = 1.0
    custom: dict[str, float] = field(default_factory=dict)


@dataclass
class SubstrateField:
    """A microenvironment concentration field on a regular grid."""

    name: str = "substrate"
    ox: float = 0.0
    oy: float = 0.0
    oz: float = 0.0
    dx: float = _DEFAULT_SPACING
    dy: float = _DEFAULT_SPACING
    dz: float = _DEFAULT_SPACING
    nx: int = 1
    ny: int = 1
    nz: int = 1
    data: list[float] = field(default_factory=list)


@dataclass
class VirtualTissue:
    """Full spatial tissue snapshot."""

    cells: list[VirtualCell] = field(default_factory=list)
    substrates: list[SubstrateField] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# JSON encode / decode
# ============================================================================

def _substrate_to_dict(s: SubstrateField) -> dict[str, Any]:
    return {
        "name": s.name,
        "origin": [s.ox, s.oy, s.oz],
        "spacing": [s.dx, s.dy, s.dz],
        "shape": [s.nx, s.ny, s.nz],
        "data": list(s.data),
    }


def _dict_to_substrate(d: dict[str, Any]) -> SubstrateField:
    ox, oy, oz = d.get("origin", [0.0, 0.0, 0.0])
    dx, dy, dz = d.get("spacing", [_DEFAULT_SPACING] * 3)
    nx, ny, nz = d.get("shape", [1, 1, 1])
    data = d.get("data", [0.0] * (nx * ny * nz))
    return SubstrateField(
        name=d.get("name", "substrate"),
        ox=ox, oy=oy, oz=oz, dx=dx, dy=dy, dz=dz,
        nx=int(nx), ny=int(ny), nz=int(nz),
        data=[float(v) for v in data],
    )


def tissue_to_dict(tissue: VirtualTissue) -> dict[str, Any]:
    """Serialize a :class:`VirtualTissue` to a JSON-able dict."""
    cells = []
    for c in tissue.cells:
        cells.append({
            "position": [c.x, c.y, c.z],
            "cell_type": c.cell_type,
            "cycle_phase": c.cycle_phase,
            "volume": c.volume,
            "custom": dict(c.custom),
        })
    return {
        "cells": cells,
        "substrates": [_substrate_to_dict(s) for s in tissue.substrates],
        "meta": dict(tissue.meta),
    }


def dict_to_tissue(data: dict[str, Any]) -> VirtualTissue:
    """Build a :class:`VirtualTissue` from a JSON-able dict."""
    cells = []
    for raw in data.get("cells", []):
        pos = raw.get("position", [0.0, 0.0, 0.0])
        cells.append(VirtualCell(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            cell_type=int(raw.get("cell_type", 0)),
            cycle_phase=str(raw.get("cycle_phase", "G1")),
            volume=float(raw.get("volume", 1.0)),
            custom={k: float(v) for k, v in raw.get("custom", {}).items()},
        ))
    substrates = [_dict_to_substrate(s) for s in data.get("substrates", [])]
    return VirtualTissue(cells=cells, substrates=substrates,
                         meta=dict(data.get("meta", {})))


def tissue_dumps(tissue: VirtualTissue) -> str:
    """Serialize a :class:`VirtualTissue` to a JSON string."""
    try:
        return json.dumps(tissue_to_dict(tissue), indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BioError(f"could not serialize tissue: {exc}") from exc


def tissue_loads(text: str) -> VirtualTissue:
    """Deserialize a :class:`VirtualTissue` from a JSON string."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise BioError(f"malformed tissue JSON: {exc}") from exc
    return dict_to_tissue(data)


# ============================================================================
# CSV (PhysiCell-style) cell dump / load
# ============================================================================

def cells_to_csv(tissue: VirtualTissue) -> str:
    """Dump the cells to a PhysiCell-style CSV string.

    Columns: ``position_x, position_y, position_z, cell_type, cycle_phase,
    volume`` followed by one column per custom variable (sorted for
    determinism).
    """
    custom_keys = sorted({k for c in tissue.cells for k in c.custom})
    fieldnames = list(_CELL_CSV_COLUMNS) + custom_keys
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for c in tissue.cells:
        row = {
            "position_x": c.x,
            "position_y": c.y,
            "position_z": c.z,
            "cell_type": c.cell_type,
            "cycle_phase": c.cycle_phase,
            "volume": c.volume,
        }
        for k in custom_keys:
            row[k] = c.custom.get(k, 0.0)
        writer.writerow(row)
    return buf.getvalue()


def cells_from_csv(text: str) -> list[VirtualCell]:
    """Parse a PhysiCell-style CSV back into cell records."""
    cells: list[VirtualCell] = []
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise BioError("empty cell CSV")
    custom_cols = [c for c in reader.fieldnames if c not in _CELL_CSV_COLUMNS]
    for row in reader:
        def _f(name: str, default: float = 0.0, row: dict[str, str] = row) -> float:
            try:
                return float(row.get(name, default))
            except (TypeError, ValueError):
                return default
        cells.append(VirtualCell(
            x=_f("position_x"),
            y=_f("position_y"),
            z=_f("position_z"),
            cell_type=int(_f("cell_type")),
            cycle_phase=row.get("cycle_phase", "G1") or "G1",
            volume=_f("volume", 1.0),
            custom={k: _f(k) for k in custom_cols},
        ))
    if not cells:
        raise BioError("cell CSV has no data rows")
    return cells


__all__ = [
    "SubstrateField",
    "VirtualCell",
    "VirtualTissue",
    "dict_to_tissue",
    "tissue_to_dict",
    "tissue_dumps",
    "tissue_loads",
    "cells_to_csv",
    "cells_from_csv",
]
