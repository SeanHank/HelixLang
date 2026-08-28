"""Bundled plugins (doc/36 §3.3 / §8).

Each scientific capability lives in its own subpackage under here and exposes a
module-level ``PLUGIN`` object (a :class:`helixlang.core.plugin_registry.PluginProvider`).
Nothing in this package is imported by :mod:`helixlang` at startup; the registry
:meth:`~helixlang.core.plugin_registry.Registry.discover` imports a plugin only
when a ``use`` statement or a ``#keyword`` first requires it (lazy, doc/36 §3.5).
"""
