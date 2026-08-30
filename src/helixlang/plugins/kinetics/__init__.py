"""Kinetic parameter estimation (doc/20 §8, doc/26 Phase C)."""
from __future__ import annotations

from helixlang.plugins.kinetics.kcat_predictor import (
    BRENDAEntry,
    KcatPredictor,
    predict_kcat,
)
from helixlang.plugins.kinetics.km_estimator import (
    KmEstimator,
    estimate_km,
)
from helixlang.plugins.kinetics.sequence_predictor import (
    SequenceKcatPrediction,
    SequenceKcatPredictor,
    SequenceKmEstimator,
    SequenceKmPrediction,
)

__all__ = [
    "BRENDAEntry",
    "KcatPredictor",
    "KmEstimator",
    "SequenceKcatPredictor",
    "SequenceKcatPrediction",
    "SequenceKmEstimator",
    "SequenceKmPrediction",
    "predict_kcat",
    "estimate_km",
]


# ---------------------------------------------------------------------------
# Plugin contract (doc/36 §7: kinetics/* -> plugins/kinetics/; extra "ml")
# ---------------------------------------------------------------------------
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
    from helixlang.plugins.kinetics.kcat_predictor import KcatPredictor
    return KcatPredictor


def _load() -> Callable[[dict | None], type]:
    if not _check("numpy"):
        from helixlang.core.errors import PluginDependencyError
        raise PluginDependencyError("kinetics", "numpy", "ml")
    return _make_backend


PLUGIN = PluginProvider(
    name="kinetics",
    extra="ml",
    keywords=("type",),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": lambda: _check("numpy")},
    load=_load,
)
