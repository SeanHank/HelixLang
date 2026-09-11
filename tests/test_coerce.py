"""Unit tests for sim_runtime._coerce helpers."""
import pytest

from helixlang.core.ast_nodes import Program
from helixlang.core.errors import SimConfigError
from helixlang.sim_runtime._coerce import (
    _coerce_bool,
    _coerce_enum,
    _coerce_float,
    _coerce_int,
    _opt_bool,
    _opt_enum,
    _opt_float,
    _opt_float_dict,
    _opt_float_list,
    _opt_float_or_none,
    _opt_int,
    _opt_int_or_none,
    _opt_replicon_specs,
    _opt_str_list,
    _project,
    _select_columns,
)


def test_coerce_bool_invalid_raises():
    with pytest.raises(SimConfigError):
        _coerce_bool("k", "maybe")
    assert _coerce_bool("k", " TRUE ") is True
    assert _coerce_bool("k", "0") is False


def test_coerce_enum_allowed_and_error():
    assert _coerce_enum("k", "goldman", frozenset({"goldman", "fountain"})) == "goldman"
    with pytest.raises(SimConfigError):
        _coerce_enum("k", "nope", frozenset({"goldman", "fountain"}))


def test_coerce_float_int_errors():
    with pytest.raises(SimConfigError):
        _coerce_float("k", "abc")
    with pytest.raises(SimConfigError):
        _coerce_int("k", "xyz")
    assert _coerce_float("k", "2.5") == 2.5
    assert _coerce_int("k", "7") == 7


def test_opt_defaults_when_key_missing():
    assert _opt_float({}, "k", 1.0) == 1.0
    assert _opt_int({}, "k", 3) == 3
    assert _opt_bool({}, "k", True) is True
    assert _opt_enum({}, "k", "goldman", frozenset()) == "goldman"


def test_opt_int_or_none():
    assert _opt_int_or_none({}, "k", 5) == 5
    assert _opt_int_or_none({"k": "none"}, "k", 5) is None
    assert _opt_int_or_none({"k": "8"}, "k", 5) == 8


def test_opt_float_or_none():
    assert _opt_float_or_none({}, "k", 1.5) == 1.5
    assert _opt_float_or_none({"k": "none"}, "k", 1.5) is None
    assert _opt_float_or_none({"k": "2.5"}, "k", 1.5) == 2.5


def test_opt_float_dict_cases():
    assert _opt_float_dict({}, "k", {"a": 1.0}) == {"a": 1.0}
    assert _opt_float_dict({"k": "a=1.0,b=2.5"}, "k", {}) == {"a": 1.0, "b": 2.5}
    with pytest.raises(SimConfigError):
        _opt_float_dict({"k": "a=1,,b=2"}, "k", {})
    with pytest.raises(SimConfigError):
        _opt_float_dict({"k": "=1.0"}, "k", {})
    with pytest.raises(SimConfigError):
        _opt_float_dict({"k": "bad"}, "k", {})


def test_opt_float_list_skips_empty():
    assert _opt_float_list({}, "k", (0.5,)) == (0.5,)
    assert _opt_float_list({"k": "1.0,,2.0"}, "k", ()) == (1.0, 2.0)


def test_opt_str_list():
    assert _opt_str_list({}, "k", ("a",)) == ("a",)
    assert _opt_str_list({"k": "goldman,fountain"}, "k", ()) == ("goldman", "fountain")
    assert _opt_str_list({"k": "a, ,b"}, "k", ()) == ("a", "b")


def test_opt_replicon_specs():
    assert _opt_replicon_specs({}, "k") == {}
    specs = _opt_replicon_specs({"k": "pBR322:20, ,pUC19:500"}, "k")
    assert set(specs) == {"pBR322", "pUC19"}
    assert specs["pBR322"].copy_number == 20
    with pytest.raises(SimConfigError):
        _opt_replicon_specs({"k": "pBR322:0"}, "k")
    with pytest.raises(SimConfigError):
        _opt_replicon_specs({"k": "pBR322"}, "k")


def _make_program(ext_output=None, output=None):
    p = Program()
    if output is not None:
        p.config.output = output
    if ext_output is not None:
        p.sim_extensions["output"] = ext_output
    return p


def test_select_columns_requested():
    p = _make_program(output=["a", "b"], ext_output="x,y")
    assert _select_columns(p, [{"a": 1, "b": 2}]) == ["x", "y"]


def test_select_columns_requested_no_ext():
    p = _make_program(output=["a", "b"])
    assert _select_columns(p, [{"a": 1, "b": 2}]) == ["a", "b"]


def test_select_columns_stdout_uses_default():
    p = _make_program(output=["stdout"])
    assert _select_columns(p, [{"a": 1}], default=["d"]) == ["d"]


def test_select_columns_union_of_keys():
    p = _make_program()
    rows = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
    assert _select_columns(p, rows) == ["a", "b", "c"]


def test_project():
    assert _project({"a": 1, "b": 2}, ["b", "c"]) == {"b": 2, "c": None}
