"""Tests for doc/34 P1: provenance schema.

Verifies that provenance dicts are correctly built and attached to results.
"""
from __future__ import annotations

from helixlang.provenance import attach_provenance, build_provenance


class TestBuildProvenance:
    """build_provenance returns a well-formed dict."""

    def test_has_required_keys(self) -> None:
        prov = build_provenance(seed=42, backend="fba")
        assert "helix_version" in prov
        assert "seed" in prov
        assert "backend" in prov
        assert "dependencies" in prov
        assert "timestamp" in prov

    def test_seed_stored(self) -> None:
        prov = build_provenance(seed=123)
        assert prov["seed"] == 123

    def test_seed_none(self) -> None:
        prov = build_provenance()
        assert prov["seed"] is None

    def test_backend_stored(self) -> None:
        prov = build_provenance(backend="whole_cell")
        assert prov["backend"] == "whole_cell"

    def test_parameters_stored(self) -> None:
        prov = build_provenance(parameters={"organism": "ecoli", "model": "iML1515"})
        assert prov["parameters"]["organism"] == "ecoli"
        assert prov["parameters"]["model"] == "iML1515"

    def test_source_hash_computed(self) -> None:
        prov = build_provenance(source="ATG TAA")
        assert prov["source_hash"].startswith("sha256:")
        assert len(prov["source_hash"]) == 71  # "sha256:" + 64 hex chars

    def test_source_path_stored(self) -> None:
        prov = build_provenance(source_path="examples/02_lac_operon.helix")
        assert prov["source_path"] == "examples/02_lac_operon.helix"

    def test_runtime_seconds_stored(self) -> None:
        prov = build_provenance(runtime_seconds=0.42)
        assert prov["runtime_seconds"] == 0.42

    def test_extra_fields_merged(self) -> None:
        prov = build_provenance(extra={"custom_key": "custom_value"})
        assert prov["custom_key"] == "custom_value"

    def test_dependency_versions_dict(self) -> None:
        prov = build_provenance()
        deps = prov["dependencies"]
        assert "python" in deps
        assert isinstance(deps["python"], str)

    def test_timestamp_format(self) -> None:
        prov = build_provenance()
        ts = prov["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts


class TestAttachProvenance:
    """attach_provenance adds provenance to a result dict."""

    def test_attaches_provenance(self) -> None:
        result = {"columns": ["tick", "biomass"], "rows": []}
        attach_provenance(result, seed=42, backend="fba")
        assert "provenance" in result
        assert result["provenance"]["seed"] == 42

    def test_returns_same_dict(self) -> None:
        result = {"data": 1}
        returned = attach_provenance(result, backend="test")
        assert returned is result

    def test_does_not_overwrite_existing_data(self) -> None:
        result = {"data": 1}
        attach_provenance(result, backend="test")
        assert result["data"] == 1
