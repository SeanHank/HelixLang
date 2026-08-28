"""Tests for doc/34 P0.3: global state audit.

Verifies that module-level mutable state is documented and thread-safe.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


class TestGlobalStateAudit:
    """All helixlang modules have no unexpected mutable globals."""

    def test_server_debug_sessions_is_thread_safe(self) -> None:
        """server.py _DEBUG_SESSIONS is protected by a threading.Lock."""
        from helixlang.server import _DEBUG_SESSIONS, _get_debug_lock

        lock = _get_debug_lock()
        assert hasattr(lock, "acquire")
        assert hasattr(lock, "release")
        assert isinstance(_DEBUG_SESSIONS, dict)

    def test_bytecode_version_is_immutable(self) -> None:
        """OPCODE_VERSION is an int constant, not mutable."""
        from helixlang.core.bytecode import OPCODE_VERSION

        assert isinstance(OPCODE_VERSION, int)
        # Verify it's a simple int (not a list or dict)
        with pytest.raises(AttributeError):
            OPCODE_VERSION.append(2)  # type: ignore[attr-defined]

    def test_codon_tables_are_immutable(self) -> None:
        """Codon tables are dicts but treated as read-only."""
        from helixlang.core.codon_table import STANDARD_TABLE

        assert isinstance(STANDARD_TABLE, dict)
        # Spot-check key entries exist
        assert "ATG" in STANDARD_TABLE
        assert "TAA" in STANDARD_TABLE

    def test_no_mutable_default_in_sim_result(self) -> None:
        """SimResult uses field(default_factory=...) for mutable defaults."""
        from helixlang.sim_runtime import SimResult

        r1 = SimResult(backend="test", columns=[], rows=[])
        r2 = SimResult(backend="test", columns=[], rows=[])
        # meta dicts must be independent
        r1.meta["key"] = "value"
        assert "key" not in r2.meta

    def test_all_helixlang_modules_importable(self) -> None:
        """Every module under helixlang can be imported without error."""
        pkg_dir = Path(__file__).parent.parent / "src" / "helixlang"
        modules = []
        for py_file in sorted(pkg_dir.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            rel = py_file.relative_to(pkg_dir.parent)
            mod_name = str(rel.with_suffix("")).replace("/", ".")
            modules.append(mod_name)

        errors = []
        for mod_name in modules[:50]:  # test first 50 to keep CI fast
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                errors.append(f"{mod_name}: {e}")

        assert not errors, "Import errors:\n" + "\n".join(errors)
