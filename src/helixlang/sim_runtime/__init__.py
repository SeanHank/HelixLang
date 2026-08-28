"""Simulation-library adapter (doc/12-helix-language-wiring.md §8).

Package facade re-exporting the simulation dispatch engine.
"""
from __future__ import annotations

from ._engine import (
    _SIM_BACKENDS,
    _add_gem_core_reactions,
    _add_gem_transport_reactions,
    _build_disease_from_helix,
    _build_drugs_from_helix,
    _build_ecosystem_patches,
    _build_endocrine_config_from_helix,
    _build_genotype_from_helix,
    _build_grn,
    _build_immune_config_from_helix,
    _build_pd_from_helix,
    _build_population_config,
    _build_qsp_bindings_from_helix,
    _build_traits_from_helix,
    _build_virtual_cell_config,
    _seed_cells,
    _set_gem_medium,
    run,
)
from ._types import (
    BACKENDS,
    ColonyResult,
    FluxResult,
    HistoryResult,
    ScoreResult,
    SimResult,
)

__all__ = [
    "BACKENDS", "SimResult", "HistoryResult", "FluxResult", "ColonyResult", "ScoreResult",
    "run", "_SIM_BACKENDS",
    "_add_gem_core_reactions", "_add_gem_transport_reactions",
    "_build_disease_from_helix", "_build_drugs_from_helix", "_build_ecosystem_patches",
    "_build_endocrine_config_from_helix", "_build_genotype_from_helix", "_build_grn",
    "_build_immune_config_from_helix", "_build_pd_from_helix", "_build_population_config",
    "_build_qsp_bindings_from_helix", "_build_traits_from_helix", "_build_virtual_cell_config",
    "_seed_cells", "_set_gem_medium",
]
