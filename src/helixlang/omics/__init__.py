"""Omics modules: spatial transcriptomics + expression inference (doc/20 §14)."""
from __future__ import annotations

# Re-export spatial omics (pre-existing module)
from helixlang.omics._spatial_omics import (  # noqa: F401
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
from helixlang.omics.expression_inference import (  # noqa: F401
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
