"""Simulation result value objects and backend registry (doc/12 §8)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["BACKENDS", "SimResult", "HistoryResult", "FluxResult", "ColonyResult", "ScoreResult"]

BACKENDS = frozenset({
    "classic", "whole_cell", "population", "fba", "calibration", "benchmark",
    "gem", "ecosystem",
})


@dataclass
class SimResult:
    """Uniform sim-backend output: selected columns + rows (+ metadata).

    ``columns`` names the CSV/JSON columns; ``rows`` are one record each
    (whole-cell minutes, dFBA batch steps, population ticks, or a single
    score row for ``calibration``/``benchmark``).  ``meta`` carries
    non-tabular payloads (colony observables, per-cell traces).
    ``provenance`` records reproducibility metadata (seed, backend,
    source hash, dependency versions, runtime).
    """
    backend: str
    columns: list[str]
    rows: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoryResult(SimResult):
    """``whole_cell`` history: one row per minute of simulated time."""


@dataclass
class FluxResult(SimResult):
    """``fba``: a static flux vector row or a dFBA batch trace."""


@dataclass
class ColonyResult(SimResult):
    """``population``: per-tick colony statistics + observables in meta."""


@dataclass
class ScoreResult(SimResult):
    """``calibration`` / ``benchmark``: fitted/score table."""

