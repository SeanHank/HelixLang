"""Result provenance — attach metadata to every simulation output.

Every simulation result carries a provenance dict so that output is
reproducible and auditable.  See doc/34 §3.4 for the schema.

Usage::

    from helixlang.core.provenance import build_provenance

    provenance = build_provenance(
        seed=42,
        backend="fba",
        parameters={"organism": "ecoli"},
    )
"""
from __future__ import annotations

import hashlib
import platform
import time
from typing import Any

from helixlang.core.version import __version__


def _source_hash(source: str | bytes) -> str:
    """SHA-256 hash of the source text."""
    if isinstance(source, str):
        source = source.encode()
    return "sha256:" + hashlib.sha256(source).hexdigest()


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


def build_provenance(
    *,
    seed: int | None = None,
    backend: str = "",
    fidelity: str | None = None,
    parameters: dict[str, Any] | None = None,
    source: str | bytes | None = None,
    source_path: str | None = None,
    runtime_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance dict for a simulation result.

    Parameters
    ----------
    seed : int, optional
        RNG seed used for the simulation.
    backend : str
        Simulation backend name (e.g. ``"fba"``, ``"whole_cell"``).
    fidelity : str, optional
        Fidelity class of the backend that produced the result (doc/36 §3ξ.6):
        ``"full"``, or a reduced-fidelity declaration that was explicitly opted
        into (e.g. ``"reduced"``).  Omitted entirely when not supplied.
    parameters : dict
        Key simulation parameters.
    source : str or bytes, optional
        Raw .helix source text.  Hashed to produce ``source_hash``.
    source_path : str, optional
        Path to the source file.  Stored for reference.
    runtime_seconds : float, optional
        Wall-clock seconds the simulation took.
    extra : dict, optional
        Additional provenance fields.

    Returns
    -------
    dict
        Provenance dictionary conforming to doc/34 §3.4.
    """
    prov: dict[str, Any] = {
        "helix_version": __version__,
        "seed": seed,
        "backend": backend,
        "parameters": parameters or {},
        "dependencies": _dependency_versions(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if fidelity is not None:
        prov["fidelity"] = fidelity
    if source is not None:
        prov["source_hash"] = _source_hash(source)
    if source_path is not None:
        prov["source_path"] = source_path
    if runtime_seconds is not None:
        prov["runtime_seconds"] = runtime_seconds
    if extra:
        prov.update(extra)
    return prov


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
