"""HelixLang application module: wraps core codec capabilities into user-facing applications.

Plugin contract (doc/36 §7: ``apps/*`` -> ``plugins/apps/``).  This package is the
canonical home of the pipeline/synbio/consortium/ecosystem/whole-cell stacks and
exposes the :data:`PLUGIN` provider so the registry discovers the ``apps`` plugin
(``#use apps`` / ``#evolve``/``#lsystem``/``#crispr``/``#morphogen``/``#species``
/``#field``/``#patch`` keywords).
"""
from __future__ import annotations

from collections.abc import Callable

from helixlang.api.registry import PluginProvider


def _check(pkg: str) -> bool:
    def _probe() -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False
    return _probe()


def _make_backend(cfg: dict | None = None) -> type:
    from helixlang.plugins.apps.full_pipeline import PipelineConfig
    return PipelineConfig


def _load() -> Callable[[dict | None], type]:
    if not _check("numpy"):
        from helixlang.core.errors import PluginDependencyError
        raise PluginDependencyError("apps", "numpy", "apps")
    return _make_backend


PLUGIN = PluginProvider(
    name="apps",
    extra="apps",
    keywords=("evolve", "lsystem", "crispr", "morphogen", "species", "field", "patch"),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": lambda: _check("numpy")},
    load=_load,
)
