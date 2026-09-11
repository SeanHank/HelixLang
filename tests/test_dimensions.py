"""Dimensional analysis (doc/38 §8): Dimension, Quantity, conversions."""
import pytest

from helixlang.core.dimensions import (
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


def test_dimension_algebra():
    assert DIM_VOLUME == DIM_LENGTH.scale(3)
    assert DIM_CONCENTRATION == DIM_AMOUNT - DIM_VOLUME
    assert DIMENSIONLESS.dimensionless
    assert not DIM_VOLUME.dimensionless
    assert DIM_TIME.tree() == "time"
    assert DIM_CONCENTRATION.tree() == "amount * length^-3" or \
        "length^-3" in DIM_CONCENTRATION.tree()
    assert (DIM_LENGTH + DIM_TIME) == Dimension(1, 0, 1, 0, 0, 0, 0)


def test_dimension_ordered_components():
    assert DIM_MASS == Dimension(0, 1, 0, 0, 0, 0, 0)
    assert dim_of_unit("min") == DIM_TIME
    assert dim_of_unit("µm3") == DIM_VOLUME
    assert dim_of_unit("µM") == DIM_CONCENTRATION
    assert dim_of_unit(None) == DIMENSIONLESS


def test_quantity_add_same_unit():
    assert (Quantity(5, "min") + Quantity(7, "min")) == Quantity(12, "min")


def test_quantity_add_cross_unit_same_dim():
    assert (Quantity(5, "min") + Quantity(30, "s")) == \
        Quantity(5.5, "min")
    assert (Quantity(30, "s") + Quantity(1, "min")) == \
        Quantity(90, "s")


def test_quantity_add_incompatible_raises_with_dim_tree():
    with pytest.raises(UnitError, match="incompatible dimensions"):
        Quantity(5, "min") + Quantity(7, "µm3")
    with pytest.raises(UnitError, match="length"):
        Quantity(1, "µM") + Quantity(1, "µm3")
    with pytest.raises(UnitError):
        Quantity(3, "µM") - Quantity(1, "min")


def test_quantity_equality_via_base():
    assert Quantity(1, "min") == Quantity(60, "s")
    from helixlang.core.dimensions import AVOGADRO
    assert Quantity(AVOGADRO, "molecule") == Quantity(1, "mol")


def test_conversion_table_minutes_seconds_exact():
    assert convert(1, "min", "s") == 60.0
    assert convert(60, "s", "min") == 1.0
    assert convert(1, "tick", "s") == 60.0
    assert convert(1, "µm3", "µm3") == 1.0
    assert convert(1e-3, "µM", "M") == pytest.approx(1e-9)


def test_convert_incompatible_raises():
    with pytest.raises(UnitError, match="cannot convert"):
        convert(1, "min", "µm")


def test_convert_to_unit():
    q = Quantity(1, "min")
    assert q.convert_to("s") == Quantity(60, "s")


def test_quantity_base_value():
    assert Quantity(2, "min").base_value == 120.0
    assert Quantity(3, "µM").base_value == pytest.approx(3e-3)


def test_parse_quantity():
    assert parse_quantity("5 µM") == Quantity(5, "µM")
    assert parse_quantity("40 min") == Quantity(40, "min")
    assert parse_quantity("3") == Quantity(3, None)
    assert parse_quantity("2.5e-3 M").value == pytest.approx(2.5e-3)
    with pytest.raises(UnitError, match="unknown unit"):
        parse_quantity("5 furlongs")
    with pytest.raises(UnitError, match="not a quantity"):
        parse_quantity("soon")


def test_available_units_cover_runtime_anchors():
    # Every anchor in core/units.py must be expressible in the named table.
    assert compatible("tick", "min")
    assert compatible("s", "min")
    assert compatible("µm", "µm3") is False
    assert compatible("gDW", "g")
    assert compatible("molecule", "mol")  # same dimension (amount)
    assert compatible("mol", "µM") is False


def test_dimension_non_dimension_binary_ops_return_not_implemented():
    d = DIM_LENGTH
    assert d.__add__(5) is NotImplemented
    assert d.__sub__(5) is NotImplemented
    assert d.tree()  # smoke


def test_dimension_tree_variants():
    assert DIM_LENGTH.tree() == "length"
    assert DIM_VOLUME.tree() == "length^3"
    assert DIMENSIONLESS.tree() == "dimensionless"
    assert Dimension(2, 0, -1, 0, 0, 0, 0).tree() == "length^2 * time^-1"
    assert Dimension(1, 1, 0, 0, 0, 0, 0).tree() == "length * mass"


def test_dim_of_unit_unknown_raises():
    with pytest.raises(UnitError):
        dim_of_unit("furlong")
    with pytest.raises(UnitError):
        factor_to_si("furlong")


def test_declare_unit_conflict_and_idempotent():
    declare_unit("candela", DIM_LENGTH, 2.0)  # new registration
    assert factor_to_si("candela") == 2.0
    # identical re-declaration is a no-op (name present, same value)
    declare_unit("candela", DIM_LENGTH, 2.0)
    # conflicting declaration raises
    with pytest.raises(UnitError):
        declare_unit("candela", DIM_MASS, 2.0)
    # a different-but-conflicting factor also raises
    with pytest.raises(UnitError):
        declare_unit("candela", DIM_LENGTH, 3.0)


def test_quantity_convert_to_incompatible_raises():
    q = Quantity(5, "min")
    with pytest.raises(UnitError):
        q.convert_to("µM")


def test_quantity_add_non_quantity_not_implemented():
    q = Quantity(1, "min")
    assert q.__add__(5) is NotImplemented
    assert q.__radd__(5) is NotImplemented
    assert q.__sub__(5) is NotImplemented
    assert q.__eq__(5) is NotImplemented


def test_quantity_radd_and_sub_same_result():
    q1 = Quantity(10, "min")
    q2 = Quantity(15, "s")
    assert q1.__radd__(q1) == Quantity(20, "min")  # __radd__
    assert (q1 - q2).value == 9.75  # 10 min - 15 s = 9.75 min
    assert hash(q1)  # __hash__
    assert Quantity(1, "min") == Quantity(60, "s")  # equal via base
    assert not (Quantity(1, "min") == Quantity(1, "µM"))  # differing dims
