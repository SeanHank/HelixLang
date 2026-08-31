"""Tests for doc/34 P1: provenance schema.

Verifies that provenance dicts are correctly built and attached to results.
"""
from __future__ import annotations

from helixlang.core.provenance import (
    PROVENANCE_CONTRACT_KEYS,
    attach_provenance,
    build_provenance,
    complete_provenance,
    provenance_from_registry,
)


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


class TestFidelityAndRegistryProvenance:
    """doc/36 §3ξ.6: provenance records backend + fidelity (no 'default' runs)."""

    def test_fidelity_absent_by_default(self) -> None:
        """Default provenance has no fidelity key (keeps the golden stable)."""
        prov = build_provenance(seed=42, backend="fba")
        assert "fidelity" not in prov

    def test_fidelity_stored_when_passed(self) -> None:
        prov = build_provenance(seed=42, backend="fba", fidelity="reduced")
        assert prov["fidelity"] == "reduced"

    def test_provenance_from_registry_full_fidelity(self) -> None:
        from helixlang.core.lexer import Lexer
        from helixlang.core.parser import Parser
        from helixlang.core.plugin_registry import Registry
        from helixlang.core.use_stmt import apply_use_directives

        r = Registry()
        r.discover("grn")
        prog = Parser(list(Lexer("#use grn\n").tokens())).parse()
        apply_use_directives(prog.use_directives, r)

        prov = provenance_from_registry(r, seed=7, parameters={"x": 1})
        assert "fidelity" in prov
        assert prov["fidelity"] == "full"
        assert "grn" in prov["backend"]
        assert "python" in prov["backend"]

    def test_provenance_from_registry_reduced_fidelity(self) -> None:
        from helixlang.core.plugin_registry import Registry

        r = Registry()
        r.discover("grn", "fba")
        r.declare_capability("--low-fidelity")
        prov = provenance_from_registry(r, seed=1)
        assert prov["fidelity"] == "reduced"
        # No active backends -> empty backend descriptor, but fidelity recorded.
        assert prov["backend"] == ""


class TestProvenanceContract:
    """doc/41 §7 (Item 6): the unified 8-field Model Provenance contract.

    Every provenance dict carries all eight canonical keys with defaults, so
    results are never "missing" a provenance field.
    """

    def test_eight_contract_keys_present_by_default(self) -> None:
        prov = build_provenance(seed=42, backend="fba")
        for key in PROVENANCE_CONTRACT_KEYS:
            assert key in prov, f"missing contract key {key!r}"

    def test_source_hash_conditional_only(self) -> None:
        prov = build_provenance(seed=1)
        assert prov["source_hash"] == ""
        prov = build_provenance(seed=1, source="ATG TAA")
        assert prov["source_hash"].startswith("sha256:")

    def test_model_version_key(self) -> None:
        prov = build_provenance(model_version="iML1515@1.0")
        assert prov["model_version"] == "iML1515@1.0"

    def test_parameter_set_has_fingerprint(self) -> None:
        prov = build_provenance(parameters={"ticks": 100, "react_steps": 1})
        ps = prov["parameter_set"]
        assert ps["fields"] == {"ticks": 100, "react_steps": 1}
        assert ps["fingerprint"].startswith("sha256:")
        assert len(ps["fingerprint"]) == 71

    def test_parameter_set_fingerprint_is_stable(self) -> None:
        a = build_provenance(parameters={"x": 1, "y": 2})["parameter_set"]["fingerprint"]
        b = build_provenance(parameters={"y": 2, "x": 1})["parameter_set"]["fingerprint"]
        assert a == b

    def test_literature_references_list(self) -> None:
        prov = build_provenance(references=["doi:10.1", "url:https://x"])
        assert prov["literature_references"] == ["doi:10.1", "url:https://x"]

    def test_backend_implementation_descriptor(self) -> None:
        prov = build_provenance(backend="human", backend_impl={"name": "human"})
        bi = prov["backend_implementation"]
        assert bi["name"] == "human"
        # native is a bool switch, never omitted (doc/41 §7.3)
        assert isinstance(bi["native"], bool)
        assert set(bi) <= {"name", "native", "module", "version"}

    def test_backend_implementation_native_defaults_to_bool(self) -> None:
        # native reflects wheel presence at build/run time; never omitted.
        assert isinstance(
            build_provenance(backend_impl={"name": "fba"})["backend_implementation"]["native"],
            bool,
        )

    def test_solver_key_default(self) -> None:
        prov = build_provenance()
        assert "solver" in prov
        prov = build_provenance(solver={"id": "qpsolver", "status": "optimal"})
        assert prov["solver"]["id"] == "qpsolver"
        assert prov["solver"]["status"] == "optimal"

    def test_random_seed_maps_roles(self) -> None:
        prov = build_provenance(seed=7, seeds={"fit_seed": 1, "noise_seed": 2})
        assert prov["random_seed"]["seed"] == 7
        assert prov["random_seed"]["fit_seed"] == 1
        assert prov["random_seed"]["noise_seed"] == 2

    def test_fidelity_mode_defaults_to_full(self) -> None:
        prov = build_provenance(seed=1)
        assert prov["fidelity_mode"] == "full"
        # legacy narrow key stays optional-only (golden stability)
        assert "fidelity" not in prov

    def test_fidelity_mode_overridable(self) -> None:
        prov = build_provenance(fidelity_mode="reduced")
        assert prov["fidelity_mode"] == "reduced"


class TestCompleteProvenance:
    """complete_provenance fills the contract onto existing results."""

    def test_fills_contract_on_empty_provenance(self) -> None:
        prov = complete_provenance({}, seed=42, backend_name="human")
        for key in PROVENANCE_CONTRACT_KEYS:
            assert key in prov
        assert prov["seed"] == 42
        assert prov["backend"] == "human"

    def test_preserves_existing_values(self) -> None:
        existing = {"backend": "hpc", "fidelity": "reduced"}
        prov = complete_provenance(existing, seed=1, backend_name="hpc")
        assert prov["backend"] == "hpc"
        assert prov["fidelity"] == "reduced"
        assert prov["backend_implementation"]["name"] == "hpc"

    def test_uses_backend_name_for_descriptor(self) -> None:
        prov = complete_provenance({}, backend_name="fba")
        assert prov["backend_implementation"]["name"] == "fba"

    def test_returns_merged_dict(self) -> None:
        prov = complete_provenance({"note": "kept"}, backend_name="fba")
        assert prov["note"] == "kept"
        assert "helix_version" in prov

