"""Mocked ESM generation tests covering predict_structure_esm result handling."""
import numpy as np
import pytest

import helixlang.plugins.runtime.protein_structure_predictor as psp
from helixlang.plugins.runtime.protein_structure_predictor import (
    predict_structure_esm,
)


class _Tensor:
    """Minimal torch-like tensor with .detach().cpu().numpy()."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Result:
    def __init__(self, n, ptm=None):
        rng = np.random.default_rng(0)
        self.coordinates = _Tensor(rng.random((n, 4, 3)) * 10.0)
        self.plddt = _Tensor(rng.random(n))
        self.ptm = ptm


class _Model:
    def __init__(self, state):
        self._state = state

    def generate(self, protein, config):
        return self._state


def _patch(monkeypatch, state):
    monkeypatch.setattr(psp, "_get_model", lambda *a, **k: _Model(state))
    return state


def test_predict_structure_esm_mocked_coords_ndim3(monkeypatch):
    n = 12
    state = _Result(n, ptm=_ConstTensor(np.array([0.4])))
    _patch(monkeypatch, state)
    res = predict_structure_esm("MKWVTFISLLFLFSSAYS"[:n - 1] + "A")
    assert res.coords.shape == (n, 3)
    assert len(res.plddt) == n
    assert res.ptm_score == pytest.approx(0.4)


def test_predict_structure_esm_plddt_scaled(monkeypatch):
    n = 10
    state = _Result(n, ptm=None)
    _patch(monkeypatch, state)
    res = predict_structure_esm("ACDEFGHIKL")
    assert all(0.0 <= p <= 100.0 for p in res.plddt)


def test_predict_structure_esm_plain_arrays_ndim2(monkeypatch):
    n = 8
    rng = np.random.default_rng(1)
    coords = rng.random((n, 3)) * 10.0
    plddt = rng.random(n) * 0.9
    state = _Result(n, ptm=None)
    state.coordinates = coords.astype(float)
    state.plddt = plddt.astype(float)
    _patch(monkeypatch, state)
    res = predict_structure_esm("ACDEFGHI")
    assert res.coords.shape == (n, 3)
    assert all(0.0 <= p <= 100.0 for p in res.plddt)


def test_predict_structure_esm_plain_arrays_bad_ndim(monkeypatch):
    n = 6
    rng = np.random.default_rng(2)
    state = _Result(n, ptm=None)
    state.coordinates = rng.random((n,)) * 5.0
    state.plddt = rng.random(n) * 0.9
    _patch(monkeypatch, state)
    res = predict_structure_esm("ACDEFG")
    assert res.coords.shape == (n, 3)
    assert all(v == 0.0 for v in res.coords[0])


def test_predict_structure_esm_plain_ptm_scalar(monkeypatch):
    n = 8
    state = _Result(n, ptm=0.62)
    _patch(monkeypatch, state)
    res = predict_structure_esm("ACDEFGHI")
    assert res.ptm_score == pytest.approx(0.62)


def test_predict_structure_esm_cuda_device(monkeypatch):
    devices = []

    def fake_get_model(model_name, device):
        devices.append(device)
        return _Model(_Result(8, ptm=0.5))

    monkeypatch.setattr(psp, "_has_cuda", lambda: True)
    monkeypatch.setattr(psp, "_get_model", fake_get_model)
    res = predict_structure_esm("ACDEFGHI")
    assert devices == ["cuda"]
    assert res.sequence == "ACDEFGHI"


def test_predict_structure_esm_explicit_device(monkeypatch):
    devices = []
    monkeypatch.setattr(psp, "_has_cuda", lambda: True)

    def fake_get_model(model_name, device):
        devices.append(device)
        return _Model(_Result(8, ptm=0.5))

    monkeypatch.setattr(psp, "_get_model", fake_get_model)
    res = predict_structure_esm("ACDEFGHI", device="cpu")
    assert devices == ["cpu"]
    assert res.sequence == "ACDEFGHI"


class _ConstTensor:
    """A tensor-like whose .mean() returns a tensor-like with .item()."""

    def __init__(self, arr):
        self._arr = arr

    def mean(self):
        return _ScalarTensor(float(np.mean(self._arr)))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _ScalarTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value
