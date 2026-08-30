"""Backend adapters for the shared sim_runtime executors (doc/38 §6.5).

E3 registers one :class:`~helixlang.api.backend.Backend` per sim backend here.
§9 splits the executor bodies out of ``_engine`` into
:mod:`helixlang.sim_runtime.backends.pipelines`; the adapters delegate here.
``_engine`` keeps orchestration only (``run()`` + shared config/state helpers).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from helixlang.api.backend import Backend, RunRequest
from helixlang.core.ast_nodes import Program

if TYPE_CHECKING:
    pass

__all__ = ["CORE_BACKENDS", "CORE_IMPL_ATTRS"]


def _executor() -> Any:
    from helixlang.sim_runtime.backends import pipelines

    return pipelines


#: ``name -> (executor attribute, #sim kinds aliases)``.  ``id`` equals ``name``
#: for every backend; ``kinds`` mirror the legacy ``_SIM_BACKENDS`` keys so the
#: ``#sim kind=...`` surface is byte-identical before/after the migration.
CORE_IMPL_ATTRS: dict[str, tuple[str, tuple[str, ...]]] = {
    # ---- first-class "#config backend=" backends (legacy elif chain) ----
    "whole_cell": ("_run_whole_cell", ()),
    "population": ("_run_population", ()),
    "fba": ("_run_fba", ()),
    "calibration": ("_run_calibration", ()),
    "benchmark": ("_run_benchmark", ()),
    "gem": ("_run_gem", ()),
    # ecosystem was reachable both ways (elif + _SIM_BACKENDS) — keep so.
    "ecosystem": ("_run_ecosystem", ("ecosystem",)),
    # ---- long-tail "#sim kind=" backends (legacy _SIM_BACKENDS table) ----
    "3d_morphology": ("_run_3d_morphology", ("3d_morphology",)),
    "codec_benchmark": ("_run_codec_benchmark", ("codec_benchmark",)),
    "codon_usage": ("_run_codon_usage", ("codon_usage",)),
    "cello_workflow": ("_run_cello_workflow", ("cello_workflow",)),
    "consortium": ("_run_consortium", ("consortium",)),
    "digital_evolution": ("_run_digital_evolution", ("digital_evolution",)),
    "directed_evolution": ("_run_directed_evolution", ("directed_evolution",)),
    "fate_analysis": ("_run_fate_analysis", ("fate_analysis",)),
    "human": ("_run_human_simulation", ("human",)),
    "morphogen_gradient": ("_run_morphogen_gradient", ("morphogen_gradient",)),
    "omics_calibration": ("_run_omics_calibration", ("omics_calibration",)),
    "population_calibration": ("_run_population_calibration",
                               ("population_calibration",)),
    "population_dbtl": ("_run_population_dbtl", ("population_dbtl",)),
    "protein_fitness": ("_run_protein_fitness", ("protein_fitness",)),
    "protein_structure": ("_run_protein_structure", ("protein_structure",)),
    "spatial_dfba": ("_run_spatial_dfba", ("spatial_dfba",)),
    "spatial_evolution": ("_run_spatial_evolution", ("spatial_evolution",)),
    "stochastic": ("_run_stochastic", ("stochastic",)),
    "synbio_design": ("_run_synbio_design", ("synbio_design",)),
}


def _make_backend(name: str, attr: str,
                  kinds: tuple[str, ...]) -> type[Backend]:
    """Build a :class:`Backend` subclass delegating to ``_engine.<attr>``."""

    def run(self: Backend, req: RunRequest) -> Any:
        # E3 shim: executors still operate on the raw Program; E4 ports them to
        # ProgramView reads and drops this cast.
        program = cast(Program, req.program)
        return getattr(_executor(), attr)(program)

    return type(name.title().replace("_", ""), (Backend,), {
        "id": name,
        "kinds": kinds,
        "run": run,
    })


CORE_BACKENDS: list[Backend] = [
    _make_backend(name, attr, kinds)()
    for name, (attr, kinds) in CORE_IMPL_ATTRS.items()
]
