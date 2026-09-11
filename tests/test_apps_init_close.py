"""Coverage closure for :mod:`helixlang.plugins.apps.__init__`.

Exercises the plugin-provider machinery (``_check`` probe, ``_load`` /
``_make_backend``) through the real :class:`Registry` path, including the
optional-dependency failure branch (temporarily making ``numpy`` import fail).
"""
import builtins

import pytest

import helixlang.plugins.apps as apps_pkg
from helixlang.core.errors import PluginDependencyError
from helixlang.core.plugin_registry import Registry


def _numpy_import_forcer(real, *args, **kwargs):
    if args and args[0] == "numpy":
        raise ImportError("no numpy")
    return real(*args, **kwargs)


class TestAppsPlugin:
    def test_provider_metadata(self):
        assert apps_pkg.PLUGIN.name == "apps"
        assert "evolve" in apps_pkg.PLUGIN.keywords
        assert apps_pkg.PLUGIN.native is None

    def test_check_probe_true_and_false(self):
        assert apps_pkg._check("os") is True
        assert apps_pkg._check("helixlang_no_such_module_xyz") is False

    def test_activate_returns_pipeline_config(self):
        reg = Registry()
        reg.discover("apps")
        prov = reg.provider("apps")
        assert prov.name == "apps"
        assert prov.check_dependencies() == []
        backend = reg.activate("apps")
        assert backend is apps_pkg._make_backend
        assert backend({}).__name__ == "PipelineConfig"

    def test_load_without_numpy_raises(self, monkeypatch):
        real = builtins.__import__
        blocker = lambda *a, **k: _numpy_import_forcer(real, *a, **k)  # noqa: E731
        monkeypatch.setattr(builtins, "__import__", blocker)
        assert apps_pkg._check("numpy") is False
        with pytest.raises(PluginDependencyError):
            apps_pkg._load()

    def test_activate_marks_active(self):
        reg = Registry()
        reg.discover("apps")
        reg.activate("apps")
        assert "apps" in reg.active()

    def test_make_backend_returns_pipeline_config(self):
        assert apps_pkg._make_backend({"organism_name": "x"}).__name__ == \
            "PipelineConfig"
