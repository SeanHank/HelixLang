"""Public physical-dimension / Quantity surface (doc/38 §6.2 ``api.dimensions``).

Re-exports the full public surface of ``core.dimensions`` so plugins stop
importing ``helixlang.core.dimensions`` directly.  The set below is frozen; it
mirrors ``core.dimensions.__all__``.
"""
from __future__ import annotations

from helixlang.core.dimensions import (  # noqa: F401
    AVOGADRO,
    DIM_AMOUNT,
    DIM_CONCENTRATION,
    DIM_LENGTH,
    DIM_MASS,
    DIM_TIME,
    DIM_VOLUME,
    DIMENSIONLESS,
    Dimension,
    Quantity,
    UnitError,
    compatible,
    convert,
    declare_unit,
    dim_of_unit,
    factor_to_si,
    parse_quantity,
)

__all__ = [
    "AVOGADRO", "Dimension", "Quantity", "UnitError",
    "DIMENSIONLESS", "DIM_LENGTH", "DIM_MASS", "DIM_TIME", "DIM_AMOUNT",
    "DIM_VOLUME", "DIM_CONCENTRATION",
    "compatible", "convert", "declare_unit", "dim_of_unit", "factor_to_si",
    "parse_quantity",
]
