"""Result provenance — attach metadata to every simulation output.

Every simulation result carries a provenance dict so that output is
reproducible and auditable.  See doc/34 §3.4 for the schema and doc/41 §7
(Item 6) for the unified 8-field provenance contract.

Usage::

    from helixlang.core.provenance import build_provenance, complete_provenance

    provenance = build_provenance(
        seed=42,
        backend="fba",
        parameters={"organism": "ecoli"},
    )
    provenance = complete_provenance(provenance, backend_name="fba", seed=42)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from helixlang.core.version import __version__

#: doc/41 §7.1 — the unified 8-field provenance contract.  Every simulation
#: result post-run carries all of these keys (defaults when absent).
PROVENANCE_CONTRACT_KEYS: tuple[str, ...] = (
    "source_hash",
    "model_version",
    "parameter_set",
    "literature_references",
    "backend_implementation",
    "solver",
    "random_seed",
    "fidelity_mode",
)


def _source_hash(source: str | bytes) -> str:
    """SHA-256 hash of the source text."""
    if isinstance(source, str):
        source = source.encode()
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _directive_fingerprint(parameters: dict[str, Any] | None) -> str:
    """Stable SHA-256 of the canonicalised parameter dict (doc/41 §7.1)."""
    payload = json.dumps(parameters or {}, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _native_available() -> bool:
    """True when the native ``helixlang._accel`` extension is importable.

    Probes with ``find_spec`` so it never *imports* the extension (which would
    be heavy and could raise platform-specific errors); this drives
    ``backend_implementation.native`` (doc/41 §7.3: pure vs native wheel).
    """
    try:
        return importlib.util.find_spec("helixlang._accel") is not None
    except (ImportError, ValueError):
        return False


def _dependency_versions() -> dict[str, str]:
    """Collect versions of key optional dependencies."""
    deps: dict[str, str] = {"python": platform.python_version()}
    for mod_name in ("numpy", "rdkit", "biopython", "flask", "torch", "esm"):
        try:
            from importlib.metadata import version as pkg_version
            deps[mod_name] = pkg_version(mod_name)
        except Exception:  # SILENTBENIGN - optional-dep version probe
            pass
    return deps


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Normative provenance record (doc/41 §7.2).

    The canonical 8-field contract plus the legacy flat keys that benchmarks
    and the CLI/server still consume (``helix_version``, ``seed``, ``backend``,
    ``parameters``, ``dependencies``, ``timestamp``, ``source_hash``).
    """

    helix_version: str = __version__
    seed: int | None = None
    backend: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=_dependency_versions)
    timestamp: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    source_hash: str = ""
    model_version: str = ""
    parameter_set: dict[str, Any] = field(default_factory=dict)
    literature_references: list[str] = field(default_factory=list)
    backend_implementation: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    random_seed: dict[str, int] = field(default_factory=dict)
    fidelity_mode: str = "full"
    source_path: str | None = None
    runtime_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), **self.extra}


def build_provenance(
    *,
    seed: int | None = None,
    backend: str = "",
    fidelity: str | None = None,
    parameters: dict[str, Any] | None = None,
    source: str | bytes | None = None,
    source_path: str | None = None,
    runtime_seconds: float | None = None,
    model_version: str | None = None,
    solver: dict[str, Any] | None = None,
    seeds: dict[str, int] | None = None,
    references: list[str] | None = None,
    backend_impl: dict[str, Any] | None = None,
    fidelity_mode: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance dict for a simulation result.

    Extends the doc/34 schema with the doc/41 §7.1 canonical contract: every
    call emits all 8 ``PROVENANCE_CONTRACT_KEYS`` with sensible defaults.

    Parameters
    ----------
    seed : int, optional
        RNG seed used for the simulation.
    backend : str
        Simulation backend name (e.g. ``"fba"``, ``"whole_cell"``).
    fidelity : str, optional
        Legacy fidelity class of the backend (doc/36 §3ξ.6): ``"full"``, or a
        reduced-fidelity opt-in.  Kept optional-only for golden stability;
        the canonical ``fidelity_mode`` field is the default-populated one.
    parameters : dict
        Key simulation parameters (the effective directive set).
    source : str or bytes, optional
        Raw .helix source text.  Hashed to produce ``source_hash``.
    source_path : str, optional
        Path to the source file.  Stored for reference.
    runtime_seconds : float, optional
        Wall-clock seconds the simulation took.
    model_version : str, optional
        Model/protocol version (resolved ``PluginManifest.version``).
    solver : dict, optional
        Solver metadata ``{id, tolerances, status}`` for the path used.
    seeds : dict, optional
        Per-role RNG seeds (config + fit/cripple/noise/genome/sde/pool).
    references : list, optional
        Literature references (from plugin manifests / ``benchmark.yaml``).
    backend_impl : dict, optional
        Overrides for ``backend_implementation`` (name/native/module/version).
    fidelity_mode : str, optional
        Canonical fidelity mode; defaults to ``"full"`` (reduced only when a
        program explicitly opted into a reduced-fidelity capability flag).
    extra : dict, optional
        Additional provenance fields.

    Returns
    -------
    dict
        Provenance dictionary conforming to doc/34 §3.4 + doc/41 §7.
    """
    params = parameters or {}
    record = ProvenanceRecord(
        seed=seed,
        backend=backend,
        parameters=params,
        source_hash=_source_hash(source) if source is not None else "",
        model_version=model_version or "",
        parameter_set={
            "fields": params,
            "fingerprint": _directive_fingerprint(params),
        },
        literature_references=list(references or []),
        backend_implementation={
            "name": (backend_impl or {}).get("name", backend),
            "native": (backend_impl or {}).get("native", _native_available()),
            "module": (backend_impl or {}).get("module", ""),
            "version": (backend_impl or {}).get("version", ""),
        },
        solver=dict(solver or {"id": ""}),
        random_seed={**({"seed": seed} if seed is not None else {}), **(seeds or {})},
        fidelity_mode=fidelity_mode if fidelity_mode is not None else "full",
        source_path=source_path,
        runtime_seconds=runtime_seconds,
        extra=extra or {},
    )
    prov = record.to_dict()
    if fidelity is not None:
        prov["fidelity"] = fidelity
    return prov


def complete_provenance(
    prov: dict[str, Any],
    *,
    seed: int | None = None,
    seeds: dict[str, int] | None = None,
    backend_name: str = "",
    backend_impl: dict[str, Any] | None = None,
    source: str | bytes | None = None,
    source_path: str | None = None,
    parameters: dict[str, Any] | None = None,
    model_version: str | None = None,
    references: list[str] | None = None,
    solver: dict[str, Any] | None = None,
    fidelity_mode: str | None = None,
    runtime_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill every doc/41 §7.1 contract key onto an existing provenance dict.

    The engine calls this after every simulator run (``_engine.py:118-125``);
    values already present on the result (e.g. a backend's own fidelity
    declaration) win — this only completes the contract, never overwrites.
    Returns the (merged) provenance dict.
    """
    base = build_provenance(
        seed=seed,
        backend=backend_name,
        parameters=parameters,
        source=source,
        source_path=source_path,
        runtime_seconds=runtime_seconds,
        model_version=model_version,
        solver=solver,
        seeds=seeds,
        references=references,
        backend_impl=backend_impl,
        fidelity_mode=fidelity_mode,
        extra=extra,
    )
    merged = dict(prov) if prov else {}
    for key, value in base.items():
        merged.setdefault(key, value)
    return merged


def provenance_from_registry(
    registry: Any,
    *,
    seed: int | None = None,
    parameters: dict[str, Any] | None = None,
    source: str | bytes | None = None,
    source_path: str | None = None,
    runtime_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance dict recording the registry's active backends.

    Records every plugin backend activated for the program plus the fidelity
    class derived from the declared capability flags (doc/36 §3ξ.6).  The
    ``backend`` field names the active plugin backends and ``fidelity`` states
    whether any reduced-fidelity opt-in was declared — so a result is never an
    ambiguous "default" computation.
    """
    fid = registry.fidelity()
    backend = ",".join(p["name"] + ":" + p["backend"] for p in fid["plugins"])
    return build_provenance(
        seed=seed,
        backend=backend,
        fidelity=fid["fidelity"],
        fidelity_mode=fid["fidelity"],
        parameters=parameters,
        source=source,
        source_path=source_path,
        runtime_seconds=runtime_seconds,
        extra=extra,
    )


def attach_provenance(
    result: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Attach provenance to an existing result dict (mutates in place)."""
    result["provenance"] = build_provenance(**kwargs)
    return result
