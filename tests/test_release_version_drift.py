"""Version-drift gate (doc/38 §2.3) regression tests.

Every version-bearing source must carry the same release version, or
``release.py --check-versions`` aborts a release before it can overwrite a
partially-synced tree.  This keeps that invariant under the normal test suite.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def release_mod():
    spec = importlib.util.spec_from_file_location("release", _ROOT / "release.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_all_version_locations_agree(release_mod) -> None:
    found = release_mod.read_versions()
    assert len(found) == len(release_mod._VERSION_LOCATIONS), \
        f"missing version strings in: {found}"
    versions = set(found.values())
    assert len(versions) == 1, f"version drift across sources: {found}"


def test_sync_targets_are_covered_by_check(release_mod) -> None:
    """Every location sync_version writes must also be drift-checked."""
    synced = {
        "pyproject.toml",
        "core/version.py",
        "server/app.py",
        "core/bytecode.py",
    }
    checked = set(release_mod.read_versions().keys())
    assert synced <= checked
