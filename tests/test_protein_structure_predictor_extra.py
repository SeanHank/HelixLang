"""Additional coverage for protein_structure_predictor uncovered branches."""
import numpy as np
import pytest

import helixlang.plugins.runtime.protein_structure_predictor as psp
from helixlang.plugins.runtime.protein_structure_predictor import (
    _derive_secondary_from_coords,
    _detect_tm_helices,
    _normalize_plddt,
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


def test_derive_secondary_from_tensor_coords():
    coords = _Tensor(np.zeros((4, 3)))
    ss = _derive_secondary_from_coords(coords)
    assert len(ss) == 4
    assert all(c in "HEC" for c in ss)


def test_derive_secondary_bad_shape():
    coords = np.zeros((4, 2))
    assert _derive_secondary_from_coords(coords) == "CCCC"
    coords3 = np.zeros(5)
    assert _derive_secondary_from_coords(coords3) == "CCCCC"


def test_normalize_plddt_tensor_and_scaling():
    arr = np.array([0.1, 0.8])
    normalized = _normalize_plddt(_Tensor(arr))
    np.testing.assert_allclose(normalized, [10.0, 80.0])
    already = np.array([90.0, 95.0])
    np.testing.assert_allclose(_normalize_plddt(already), [90.0, 95.0])


def test_detect_tm_helices_trailing():
    hydrophobic = "LIIIIIIIIIIIIIIIIIIIIL"
    seq = "ACDEFGHIKL" + hydrophobic
    plddt = np.full(len(seq), 0.9)
    helices = _detect_tm_helices(seq, plddt, min_len=10, max_len=30)
    assert any(h.end == len(seq) for h in helices)


def test_predict_structure_esm_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(psp, "_ESM_AVAILABLE", False)
    with pytest.raises(ImportError):
        predict_structure_esm("ACDE")


def test_predict_structure_batch_dispatches(monkeypatch):
    calls = []

    def fake_predict(seq, *a, **k):
        calls.append(seq)
        return "ok"

    monkeypatch.setattr(psp, "predict_structure_esm", fake_predict)
    from helixlang.plugins.runtime.protein_structure_predictor import (
        predict_structure_batch,
    )
    out = predict_structure_batch(["AAA", "BBB"])
    assert out == ["ok", "ok"]
    assert calls == ["AAA", "BBB"]


def test_has_cuda_no_torch(monkeypatch):
    import sys
    real = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        assert psp._has_cuda() is False
    finally:
        if real is not None:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)
