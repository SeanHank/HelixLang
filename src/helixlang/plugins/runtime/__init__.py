"""Biological runtime (doc/36 Layer 2) — heavy scientific modules.

Consolidated home for the core biological-runtime modules (cell, GRN,
metabolism, environment, population, central dogma, …) plus sequence/encoding
utilities.  These are the Layer-2 scientific modules that the Layer-1 language
core never imports directly; the CLI / VM / server (root) and the thin plugin
entry packages (:mod:`helixlang.plugins.grn` …, :mod:`helixlang.plugins.apps` …)
import from here on demand.
"""
from __future__ import annotations
