"""Omics modules: spatial transcriptomics + expression inference (doc/20 §14)."""
from __future__ import annotations

# Re-export spatial omics (pre-existing module)
from helixlang.plugins.omics._spatial_omics import (  # noqa: F401
    ExpressionMatrix,
    SpatialAtlas,
    adjusted_rand_index,
    apply_fba_bounds,
    build_state_grn,
    compare_heterogeneity,
    expression_to_fba_bounds,
    expression_to_grn_states,
    from_arrays,
    read_expression_matrix,
)

# Re-export expression inference (doc/20 §14)
from helixlang.plugins.omics.expression_inference import (  # noqa: F401
    ExpressionModel,
    ExpressionState,
    build_expression_model,
    hill_function,
    infer_expression,
    infer_expression_at_time,
)

__all__ = [
    # Spatial omics
    "ExpressionMatrix",
    "SpatialAtlas",
    "adjusted_rand_index",
    "apply_fba_bounds",
    "build_state_grn",
    "compare_heterogeneity",
    "expression_to_fba_bounds",
    "expression_to_grn_states",
    "from_arrays",
    "read_expression_matrix",
    # Expression inference
    "ExpressionModel",
    "ExpressionState",
    "build_expression_model",
    "hill_function",
    "infer_expression",
    "infer_expression_at_time",
]


# ---------------------------------------------------------------------------
# Plugin contract (doc/36 §7: omics/* -> plugins/omics/; extra "ml")
# ---------------------------------------------------------------------------
from collections.abc import Callable

from helixlang.core.plugin_registry import PluginProvider


def _check(pkg: str) -> bool:
    def _probe() -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False
    return _probe()


def _make_backend(cfg: dict | None = None) -> type:
    from helixlang.plugins.omics.expression_inference import ExpressionModel
    return ExpressionModel


def _load() -> Callable[[dict | None], type]:
    if not _check("numpy"):
        from helixlang.core.errors import PluginDependencyError
        raise PluginDependencyError("omics", "numpy", "ml")
    return _make_backend


PLUGIN = PluginProvider(
    name="omics",
    extra="ml",
    keywords=(),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": lambda: _check("numpy")},
    load=_load,
)
