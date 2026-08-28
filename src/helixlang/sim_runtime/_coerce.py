"""Config coercion / column-selection helpers for sim backends."""
from __future__ import annotations

__all__ = ["_TRUE", "_FALSE", "_coerce_float", "_coerce_int", "_coerce_bool", "_coerce_enum", "_opt_float", "_opt_int", "_opt_bool", "_opt_enum", "_opt_int_or_none", "_opt_float_or_none", "_opt_float_dict", "_opt_float_list", "_opt_str_list", "_opt_replicon_specs", "_select_columns", "_project"]


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from helixlang.core.ast_nodes import Program
from helixlang.core.errors import SimConfigError
from helixlang.plugins.runtime.virtual_cell import RepliconSpec

_TRUE = {"true", "1", "yes"}


_FALSE = {"false", "0", "no"}


def _coerce_float(key: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise SimConfigError(
            f"sim key {key!r}: expected a number, got {raw!r}") from None


def _coerce_int(key: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise SimConfigError(
            f"sim key {key!r}: expected an integer, got {raw!r}") from None


def _coerce_bool(key: str, raw: str) -> bool:
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise SimConfigError(
        f"sim key {key!r}: expected true/false/1/0/yes/no, got {raw!r}")


def _coerce_enum(key: str, raw: str, allowed: frozenset[str]) -> str:
    if raw in allowed:
        return raw
    raise SimConfigError(
        f"sim key {key!r}: expected one of {sorted(allowed)}, got {raw!r}")


def _opt_float(sim: dict[str, str], key: str, default: float) -> float:
    return _coerce_float(key, sim[key]) if key in sim else default


def _opt_int(sim: dict[str, str], key: str, default: int) -> int:
    return _coerce_int(key, sim[key]) if key in sim else default


def _opt_bool(sim: dict[str, str], key: str, default: bool) -> bool:
    return _coerce_bool(key, sim[key]) if key in sim else default


def _opt_enum(sim: dict[str, str], key: str, default: str,
              allowed: frozenset[str]) -> str:
    return _coerce_enum(key, sim[key], allowed) if key in sim else default


def _opt_int_or_none(sim: dict[str, str], key: str,
                     default: int | None) -> int | None:
    if key not in sim:
        return default
    if sim[key].strip() == "none":
        return None
    return _coerce_int(key, sim[key])


def _opt_float_or_none(sim: dict[str, str], key: str,
                       default: float | None) -> float | None:
    if key not in sim:
        return default
    if sim[key].strip() == "none":
        return None
    return _coerce_float(key, sim[key])


def _opt_float_dict(sim: dict[str, str], key: str,
                    default: dict[str, float]) -> dict[str, float]:
    """Coerce ``"a=1.0,b=2.5"`` -> ``{"a": 1.0, "b": 2.5}`` (§6.3 dict)."""
    if key not in sim:
        return dict(default)
    out: dict[str, float] = {}
    for pair in sim[key].split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            raise SimConfigError(
                f"sim key {key!r}: expected comma-separated k=v pairs, "
                f"got {sim[key]!r}")
        k, v = pair.split("=", 1)
        k = k.strip()
        if not k:
            raise SimConfigError(
                f"sim key {key!r}: empty name in {pair!r}")
        out[k] = _coerce_float(f"{key}.{k}", v.strip())
    return out


def _opt_float_list(sim: dict[str, str], key: str,
                    default: tuple[float, ...]) -> tuple[float, ...]:
    """Coerce ``"0.5,1.0,1.5"`` -> ``(0.5, 1.0, 1.5)``."""
    if key not in sim:
        return tuple(default)
    out: list[float] = []
    for part in sim[key].split(","):
        part = part.strip()
        if part:
            out.append(_coerce_float(key, part))
    return tuple(out)


def _opt_str_list(sim: dict[str, str], key: str,
                  default: tuple[str, ...]) -> tuple[str, ...]:
    """Coerce ``"goldman,fountain"`` -> ``("goldman", "fountain")``."""
    if key not in sim:
        return tuple(default)
    return tuple(p.strip() for p in sim[key].split(",") if p.strip())


def _opt_replicon_specs(sim: dict[str, str], key: str,
                        ) -> dict[str, RepliconSpec]:
    """Coerce ``"pBR322:20,pUC19:500"`` -> replicon specs (Phase-C C2).

    Replicons declared here are plasmids (constant base copy number);
    the chromosome replicon is implicit and fork-driven.
    """
    if key not in sim:
        return {}
    out: dict[str, RepliconSpec] = {}
    for pair in sim[key].split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise SimConfigError(
                f"sim key {key!r}: expected 'name:copy' pairs, got {pair!r}")
        name, _, copy_s = pair.partition(":")
        name = name.strip()
        copy = _coerce_int(f"{key}.{name}", copy_s.strip())
        if copy < 1:
            raise SimConfigError(
                f"sim key {key!r}: copy number must be >= 1, got {copy}")
        out[name] = RepliconSpec(kind="plasmid", copy_number=copy)
    return out


def _select_columns(program: Program, rows: list[dict[str, Any]],
                    default: list[str] | None = None) -> list[str]:
    """``#config output=`` / ``#sim output=`` column selection (§6.7);
    else ``default`` or the first-seen union of the row keys."""
    requested = program.config.output
    ext_output = program.sim_extensions.get("output")
    if ext_output:
        requested = [c.strip() for c in ext_output.split(",") if c.strip()]
    if requested and requested != ["stdout"]:
        return requested
    if default is not None:
        return default
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    return cols


def _project(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {k: row.get(k) for k in columns}

