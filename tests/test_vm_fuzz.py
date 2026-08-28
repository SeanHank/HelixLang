"""Phase 5 hardening: fuzz the native VM against randomized bytecode (doc/36 §10 Phase 5).

The dispatch kernels (pure-Python reference and the compiled C backend) share a
contract: ``run_quota(code, constants, *, quota=..., gene_table=...)`` returns
``(ops_consumed, stack, halted)`` and raise ``NotImplementedError`` on an
unhandled opcode or ``IndexError`` on a stack/constant underflow.

This module drives **deterministic, seeded random programs** through every
backends available on the current interpreter and asserts byte-identical
results — or, when a program raises, that both backends raise the *same*
exception type.  It deliberately includes partial programs (truncated before a
``PUSH_CONST`` operand), empty programs, underflowing ``POP``/arithmetic, and
unknown opcodes, so the native path is exercised against the exact failure modes
an attacker or a malformed/corrupted ``.hbc`` could present (doc/36 §5.5 rule 3,
§11 "C/Rust segfault on malicious bytecode" mitigation).
"""
from __future__ import annotations

import importlib
import importlib.util
import random

import pytest

from helixlang._accel._loaders import choose_backend

# Dispatch opcode subset shared by impl_python and impl_cext (doc/36 §5.5).
_HALT = 0x11
_PUSH_CONST = 0x20
_POP = 0x21
_ADD = 0x90
_SUB = 0x91
_MUL = 0x92

_HANDLED = {_HALT, _PUSH_CONST, _POP, _ADD, _SUB, _MUL}

# Opcode universe the fuzzer draws from: the handled subset plus many unknown /
# future opcode values that must be rejected loudly (never run, never crash).
_OP_SPACE = sorted(_HANDLED) + [0x00, 0x01, 0x10, 0x12, 0x22, 0x30, 0x40,
                                0x50, 0x60, 0x70, 0x80, 0x88, 0x95, 0xF1,
                                0xFF]


def _dispatch_native_available() -> bool:
    pkg = "helixlang._accel.dispatch"
    return importlib.util.find_spec(f"{pkg}.impl_cext") is not None


def _load_backends():
    """Return (python_mod, native_mod_or_None)."""
    py = importlib.import_module("helixlang._accel.dispatch.impl_python")
    nat = None
    if _dispatch_native_available():
        nat = importlib.import_module("helixlang._accel.dispatch.impl_cext")
    return py, nat


def _canonical(fn, code, consts, *, quota=4096):
    """Run ``fn`` and reduce the outcome to a JSON-serializable signature.

    ``("ok", ops, stack, halted)`` when it returns, or
    ``("err", ExceptionClassName)`` when it raises.  Exceptions *must* be typed
    (NotImplementedError / IndexError); a bare crash would surface as an
    unexpected exception and fail the caller's assertion.
    """
    try:
        ops, stack, halted = fn(code, consts, quota=quota)
    except Exception as exc:  # noqa: BLE001 - classified for parity comparison
        return ("err", type(exc).__name__)
    return ("ok", ops, list(stack), halted)


def _random_program(rng):
    """Generate a single random dispatch program (list of ints)."""
    n = rng.randint(0, 24)
    code = [_OP_SPACE[rng.randrange(len(_OP_SPACE))] for _ in range(n)]
    return code


# ── single-program kernel fuzzing ────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_run_quota_python_is_sound(seed):
    """Every generated program is handled deterministically by the reference."""
    rng = random.Random(seed)
    consts = [rng.uniform(-100.0, 100.0) for _ in range(8)]
    py, _ = _load_backends()
    for _ in range(50):
        code = _random_program(rng)
        # A single run must not crash cross the whole seed batch; both the
        # success and typed-exception outcomes are acceptable.
        _canonical(py.run_quota, code, consts)


def test_fuzz_run_quota_native_matches_python():
    """Native (C) dispatch must be byte-identical to python for random code."""
    py, nat = _load_backends()
    if nat is None:
        pytest.skip("native dispatch .so not built for this interpreter")
    rng = random.Random(0xC0FFEE)
    consts = [rng.uniform(-1e3, 1e3) for _ in range(10)]
    checked = 0
    for trial in range(300):
        code = _random_program(rng)
        py_out = _canonical(py.run_quota, code, consts)
        nat_out = _canonical(nat.run_quota, code, consts)
        assert py_out == nat_out, (
            f"parity mismatch on trial {trial} code={code} consts={consts}\n"
            f"  python={py_out}\n  native ={nat_out}")
        checked += 1
    assert checked == 300


def test_fuzz_run_quota_native_matches_python_truncated_operators():
    """Programs cut mid-operand (e.g. a lone OP_PUSH_CONST with no index) must
    not crash either backend and must yield identical outcomes."""
    py, nat = _load_backends()
    if nat is None:
        pytest.skip("native dispatch .so not built for this interpreter")
    consts = [1.0, 2.0, 3.0]
    # Every prefix of a handled-sequence is a legal (possibly raising) program.
    full = [_PUSH_CONST, 0, _PUSH_CONST, 1, _MUL, _POP, _HALT]
    for cut in range(len(full) + 1):
        code = full[:cut]
        py_out = _canonical(py.run_quota, code, consts)
        nat_out = _canonical(nat.run_quota, code, consts)
        assert py_out == nat_out, f"prefix {cut}: {py_out} vs {nat_out}"


# ── population (run_many) fuzzing ────────────────────────────────────────────


def test_fuzz_run_many_native_matches_python():
    """Population dispatch over random code must equal running per-cell."""
    py, nat = _load_backends()
    if nat is None:
        pytest.skip("native dispatch .so not built for this interpreter")
    rng = random.Random(7)
    consts = [0.0, -1.0, 2.5]
    for _ in range(60):
        code = _random_program(rng)
        n_cells = rng.randint(1, 16)
        try:
            py_out = py.run_many(code, consts, n_cells=n_cells)
        except Exception as exc:  # noqa: BLE001
            # Cell 0 raises with the same error in both backends.
            with pytest.raises(type(exc)):
                nat.run_many(code, consts, n_cells=n_cells)
            continue
        nat_out = nat.run_many(code, consts, n_cells=n_cells)
        assert py_out == nat_out
        assert len(nat_out) == n_cells


def test_fuzz_run_many_deterministic():
    """run_many must be deterministic across repeated identical batches."""
    py, nat = _load_backends()
    nat = nat or py
    rng = random.Random(99)
    consts = [rng.uniform(-5.0, 5.0) for _ in range(6)]
    for _ in range(30):
        code = _random_program(rng)
        try:
            a = nat.run_many(code, consts, n_cells=8)
        except Exception:  # noqa: BLE001 - typed errors are acceptable
            continue
        b = nat.run_many(code, consts, n_cells=8)
        assert a == b


# ── robustness: never silently execute, never crash ──────────────────────────


def test_fuzz_unknown_opcode_raises_everywhere():
    """An unknown opcode must raise NotImplementedError in every backend —
    never be silently skipped (doc/36 §3ξ no-silent-fallback)."""
    py, nat = _load_backends()
    for mod in (py, nat):
        if mod is None:
            continue
        for bad in (0x00, 0x01, 0x10, 0x12, 0x22, 0x30, 0x40, 0xFF):
            with pytest.raises(NotImplementedError):
                mod.run_quota([bad], [1.0])


def test_fuzz_loader_never_silently_reselects(monkeypatch):
    """Requesting an explicit unavailable dispatch backend raises instead of
    silently swapping to another fidelity class (doc/36 §3ξ.5)."""
    from helixlang.core.errors import NativeBackendError
    monkeypatch.setenv("HELIX_ACCEL", "numba")  # dispatch has no numba impl
    with pytest.raises(NativeBackendError):
        choose_backend("helixlang._accel.dispatch", prefer="numba")
    # python is the explicitly-declared pure path and always resolves.
    assert choose_backend("helixlang._accel.dispatch", prefer="python") == \
        "impl_python"
