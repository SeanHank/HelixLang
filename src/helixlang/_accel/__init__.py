"""Native acceleration namespace (doc/36 §4).

Hot-loop implementations for the language VM and scientific kernels.  Each hot
path lives in its own subpackage under here with one ``backend.py`` exposing the
shared callable API and one implementation per technology ``impl_*``.

Selection among implementations is **speed-only, same-fidelity** (doc/36 §4.2 /
§3ξ.5): it never silently crosses a fidelity boundary.  A missing chosen backend
raises :class:`~helixlang.core.errors.NativeBackendError` (or, if the caller
declared ``--pure-python``, the caller selects the pure-Python impl explicitly).

Nothing in this package is imported by default; ``import helixlang`` stays
dependency-light.
"""
