"""Dimensional analysis for the HelixLang type checker (doc/38 §8).

A :class:`Dimension` is a tuple of 7 SI base exponents
``(length, mass, time, current, temperature, amount, intensity)`` and a
:class:`Quantity` is a ``(value, unit)`` pair with dimensional arithmetic:

- ``+`` / ``-`` demand equal dimensions (the user of the quantity decides how
  to convert — quantity arithmetic always works in the left operand's unit);
- ``*`` / ``/`` compose the dimensions of the operands;
- named units resolve to a :class:`Dimension` plus an SI conversion factor,
  with the anchors in :mod:`helixlang.core.units` as the conversion basis
  (1 tick = 1 minute = 60 s, lattice spacing in µm, AI-2 in µM, ATP counted
  as molecules).

The module is **stdlib-only** like :mod:`helixlang.core.units`, so it can be
imported from the checker, the runtime and the plugin side without cycles.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from helixlang.core.units import TIME_TICK_S

# SI base exponent indices.
_L, _M, _T, _I, _TH, _N, _J = range(7)

#: Avogadro's number (mol^-1) — molecule-count <-> amount conversions.
AVOGADRO = 6.02214076e23


class UnitError(ValueError):
    """Raised when a quantity is combined with an incompatible dimension."""


class Dimension(tuple):
    """A 7-exponent SI dimension tuple (length, mass, time, current,
    temperature, amount, intensity)."""

    def __new__(cls, *exponents: int) -> Dimension:
        return tuple.__new__(cls, (int(e) for e in exponents))

    def __add__(self, other: object) -> Dimension:  # compose  A * B
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(*(a + b for a, b in zip(self, other, strict=True)))

    def __sub__(self, other: object) -> Dimension:  # compose  A / B
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(*(a - b for a, b in zip(self, other, strict=True)))

    def scale(self, n: int) -> Dimension:
        """Raise the dimension to the ``n``th power (e.g. volume = length³)."""
        return Dimension(*(e * n for e in self))

    @property
    def dimensionless(self) -> bool:
        return not any(self)

    def tree(self) -> str:
        """Human-readable dimension tree, e.g. ``(amount) * (length)^-3``."""
        terms = []
        names = ("length", "mass", "time", "current", "temperature",
                 "amount", "intensity")
        for exp, name in zip(self, names, strict=True):
            if exp == 1:
                terms.append(name)
            elif exp != 0:
                terms.append(f"{name}^{exp}")
        return " * ".join(terms) if terms else "dimensionless"


DIMENSIONLESS = Dimension(0, 0, 0, 0, 0, 0, 0)
DIM_LENGTH = Dimension(1, 0, 0, 0, 0, 0, 0)
DIM_MASS = Dimension(0, 1, 0, 0, 0, 0, 0)
DIM_TIME = Dimension(0, 0, 1, 0, 0, 0, 0)
DIM_AMOUNT = Dimension(0, 0, 0, 0, 0, 1, 0)
DIM_VOLUME = DIM_LENGTH.scale(3)
DIM_CONCENTRATION = DIM_AMOUNT - DIM_VOLUME

#: ``unit name -> (dimension, factor to SI base, canonical suffix)``.
#: The canonical suffix is the SI-coherent render of the dimension, used as a
#: display unit when a composite quantity has no named unit (e.g. ``mol/min``).
_NAMED_UNITS: dict[str, tuple[Dimension, float]] = {
    "min": (DIM_TIME, 60.0),           # 1 tick = 1 minute (Neidhardt 1996)
    "s": (DIM_TIME, 1.0),
    "h": (DIM_TIME, 3600.0),           # 1 hour
    "d": (DIM_TIME, 86400.0),          # 1 day
    "wk": (DIM_TIME, 7 * 86400.0),     # 1 week
    "tick": (DIM_TIME, TIME_TICK_S),
    "µm": (DIM_LENGTH, 1e-6),
    "um": (DIM_LENGTH, 1e-6),
    "µm³": (DIM_VOLUME, 1e-18),
    "µm3": (DIM_VOLUME, 1e-18),
    "mol": (DIM_AMOUNT, 1.0),
    "mmol": (DIM_AMOUNT, 1e-3),
    "pmol": (DIM_AMOUNT, 1e-12),
    "molecule": (DIM_AMOUNT, 1.0 / AVOGADRO),
    "atom": (DIM_AMOUNT, 1.0 / AVOGADRO),
    "M": (DIM_CONCENTRATION, 1e3),     # mol / L = mol / 1e-3 m^3
    "mM": (DIM_CONCENTRATION, 1.0),
    "µM": (DIM_CONCENTRATION, 1e-3),   # µmol / L
    "uM": (DIM_CONCENTRATION, 1e-3),
    "nM": (DIM_CONCENTRATION, 1e-6),
    "pM": (DIM_CONCENTRATION, 1e-9),
    "L": (DIM_VOLUME, 1e-3),           # 1 litre = 1e-3 m^3
    "l": (DIM_VOLUME, 1e-3),
    "ml": (DIM_VOLUME, 1e-6),
    "uL": (DIM_VOLUME, 1e-9),
    "µL": (DIM_VOLUME, 1e-9),
    "gDW": (DIM_MASS, 1e-3),
    "g": (DIM_MASS, 1e-3),
    "mg": (DIM_MASS, 1e-6),
    "µg": (DIM_MASS, 1e-9),
    "ug": (DIM_MASS, 1e-9),
    "ng": (DIM_MASS, 1e-12),
    "pg": (DIM_MASS, 1e-15),
    "": (DIMENSIONLESS, 1.0),
}


def dim_of_unit(unit: str | None) -> Dimension:
    """Dimension of a named unit (``None``/unknown -> dimensionless)."""
    if unit is None:
        return DIMENSIONLESS
    entry = _NAMED_UNITS.get(unit)
    if entry is None:
        raise UnitError(f"unknown unit {unit!r}")
    return entry[0]


def factor_to_si(unit: str) -> float:
    """Conversion factor from ``unit`` to the SI base (same dimension)."""
    entry = _NAMED_UNITS.get(unit)
    if entry is None:
        raise UnitError(f"unknown unit {unit!r}")
    return entry[1]


def declare_unit(name: str, dim: Dimension, si_factor: float) -> None:
    """Register (or confirm) a named unit in the conversion table.

    Identical re-declaration is a no-op (idempotent, like the anchors in
    :mod:`helixlang.core.units`); a conflicting declaration is a hard
    :class:`UnitError` so plugins cannot silently skew a dimension.
    """
    if name in _NAMED_UNITS and _NAMED_UNITS[name] != (dim, si_factor):
        raise UnitError(
            f"conflicting declaration for unit {name!r}: "
            f"{_NAMED_UNITS[name]} vs {(dim, si_factor)}")
    _NAMED_UNITS[name] = (dim, si_factor)


def compatible(u1: str | Dimension, u2: str | Dimension) -> bool:
    """True when two units (or dimensions) share a dimension."""
    d1 = u1 if isinstance(u1, Dimension) else dim_of_unit(u1)
    d2 = u2 if isinstance(u2, Dimension) else dim_of_unit(u2)
    return d1 == d2


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` between two units (same dimension), exactly.

    ``convert(1, 'min', 's') == 60.0`` and ``convert(60, 's', 'min') == 1.0``.
    """
    if not compatible(from_unit, to_unit):
        raise UnitError(
            f"cannot convert {from_unit!r} ("
            f"{dim_of_unit(from_unit).tree()}) to {to_unit!r} "
            f"({dim_of_unit(to_unit).tree()})")
    return value * factor_to_si(from_unit) / factor_to_si(to_unit)


#: Tolerated exactness for equality comparisons (conversion factors are the
#: exact anchors in :mod:`units`, so strict equality already holds).
_EQ_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class Quantity:
    """A value with a dimension, expressed in a named unit when available.

    ``Quantity(5, 'min') + Quantity(7, 'min')`` is ``Quantity(12, 'min')``;
    adding a quantity of a different dimension raises :class:`UnitError`
    naming both dimension trees.
    """

    value: float
    unit: str | None = None
    dim: Dimension = DIMENSIONLESS

    def __post_init__(self) -> None:
        if self.unit is not None:
            object.__setattr__(self, "dim", dim_of_unit(self.unit))

    @property
    def base_value(self) -> float:
        """The quantity in DI/base units of its dimension."""
        if self.unit is None:
            return self.value
        return self.value * factor_to_si(self.unit)

    def convert_to(self, unit: str) -> Quantity:
        if self.unit is not None and not compatible(self.unit, unit):
            raise UnitError(
                f"cannot convert {self!r} to {unit!r} "
                f"({dim_of_unit(unit).tree()})")
        return Quantity(convert(self.value, self.unit or "", unit), unit)

    def _require_dim(self, other: Quantity, op: str) -> None:
        if self.dim != other.dim:
            raise UnitError(
                f"{op}: incompatible dimensions {self.dim.tree()} vs "
                f"{other.dim.tree()} ({self!r} {op} {other!r})")

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        self._require_dim(other, "+")
        unit = self.unit if self.unit is not None else other.unit
        rhs = other if unit is None else other.convert_to(unit)
        return Quantity(self.value + rhs.value, unit)

    def __radd__(self, other: Quantity) -> Quantity:
        return self.__add__(other)

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        self._require_dim(other, "-")
        unit = self.unit if self.unit is not None else other.unit
        rhs = other if unit is None else other.convert_to(unit)
        return Quantity(self.value - rhs.value, unit)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dim != other.dim:
            return False
        return abs(self.base_value - other.base_value) <= _EQ_TOL

    def __hash__(self) -> int:
        return hash((self.dim, round(self.base_value, 12)))

    def __repr__(self) -> str:
        unit = self.unit if self.unit is not None else self.dim.tree()
        return f"Quantity({self.value}, {unit})"


#: Named-unit suffix on a numeric literal, e.g. ``5 µM`` / ``40 min``.
_QUANTITY_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"([A-Za-zµμ³³]*)\s*$")


def parse_quantity(text: str) -> Quantity:
    """Parse ``"5 µM"`` / ``"40 min"`` / ``"3"`` into a :class:`Quantity`.

    A bare number is a dimensionless quantity; an unknown unit raises
    :class:`UnitError`.
    """
    m = _QUANTITY_RE.match(text)
    if m is None:
        raise UnitError(f"not a quantity literal: {text!r}")
    value_s, unit = m.groups()
    if unit and unit not in _NAMED_UNITS:
        raise UnitError(f"unknown unit {unit!r} in {text!r}")
    return Quantity(float(value_s), unit or None)


__all__ = [
    "AVOGADRO", "Dimension", "Quantity", "UnitError",
    "DIMENSIONLESS", "DIM_LENGTH", "DIM_MASS", "DIM_TIME", "DIM_AMOUNT",
    "DIM_VOLUME", "DIM_CONCENTRATION",
    "compatible", "convert", "declare_unit", "dim_of_unit", "factor_to_si",
    "parse_quantity",
]
