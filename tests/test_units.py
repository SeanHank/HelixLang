"""Units module unit tests."""
from __future__ import annotations

import pytest

from helixlang.core import units
from helixlang.core.dimensions import DIM_TIME, UnitError
from helixlang.core.units import (
    TIME_TICK_MIN,
    TIME_TICK_S,
    Q,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    declare_unit,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    ticks_to_min,
)


def test_ticks_to_min() -> None:
    assert ticks_to_min(2.0) == 2.0 * TIME_TICK_MIN
    assert ticks_to_min(0) == 0.0


def test_diffusion_to_lattice_basic() -> None:
    assert diffusion_to_lattice(100.0, 60.0, 10.0) == 60.0


def test_diffusion_to_lattice_zero_or_neg_dx_raises() -> None:
    with pytest.raises(ValueError):
        diffusion_to_lattice(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        diffusion_to_lattice(1.0, 1.0, -5.0)


def test_diffusion_to_lattice_negative_dt_raises() -> None:
    with pytest.raises(ValueError):
        diffusion_to_lattice(1.0, -1.0, 10.0)


def test_diffusion_lattice_to_dx() -> None:
    assert diffusion_lattice_to_dx(100.0, 60.0, 60.0) == pytest.approx(10.0)


def test_diffusion_lattice_to_dx_nonpositive_raises() -> None:
    with pytest.raises(ValueError):
        diffusion_lattice_to_dx(100.0, 60.0, 0.0)
    with pytest.raises(ValueError):
        diffusion_lattice_to_dx(100.0, 60.0, -1.0)


def test_decay_from_half_life_ticks() -> None:
    assert decay_from_half_life_ticks(110.0) == pytest.approx(0.994, abs=1e-3)


def test_decay_from_half_life_nonpositive_raises() -> None:
    with pytest.raises(ValueError):
        decay_from_half_life_ticks(0.0)
    with pytest.raises(ValueError):
        decay_from_half_life_ticks(-1.0)


def test_decay_to_half_life_ticks() -> None:
    assert decay_to_half_life_ticks(0.5) == pytest.approx(1.0)
    assert decay_from_half_life_ticks(
        decay_to_half_life_ticks(0.5)) == pytest.approx(0.5)


def test_decay_to_half_life_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(0.0)
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(1.0)
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(1.5)


def test_declare_unit_registers() -> None:
    declare_unit("fortnight", DIM_TIME, 1209600.0)
    assert Q("fortnight", 1.0).base_value == pytest.approx(1209600.0)


def test_declare_unit_conflict_raises() -> None:
    with pytest.raises(UnitError):
        declare_unit("min", DIM_TIME, 999.0)


def test_q_helper() -> None:
    q = Q("min", 5.0)
    assert q.base_value == pytest.approx(5.0 * TIME_TICK_S)
    assert q.convert_to("s").value == pytest.approx(5.0 * TIME_TICK_S)


def test_module_constants_exist() -> None:
    assert units.TIME_TICK_S == 60.0
    assert units.LATTICE_SPACING_UM == 10.0
    assert units.ATP_PER_GLUCOSE == 38
