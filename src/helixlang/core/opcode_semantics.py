"""Runtime opcode semantics — physical-unit constant registry (doc/36 §2.3).

The values here tune what each Helix **opcode** does at run time (the
``OP_REGULATE`` edge weight, the ``OP_BIND`` expression boost, the
``OP_SIGNAL`` autoinducer emission, …).  They are pure-language semantics with
**zero** scientific-modeling dependency, so they live in the minimal core.

Both the Layer-1/2 VM (``helixlang.core.vm``) and the Layer-2 biological runtime
(``helixlang.plugins.runtime.population``) consume them from here, which keeps
those two modules decoupled at module-load time.

Quantities are in physical units (µM signals, molecule-count energies) unless
cited otherwise.  The behaviors these tune were wired in to replace documented
no-op stubs; sources are cited at each constant.
"""
from __future__ import annotations

#: regulatory-edge weight applied by the runtime ``OP_REGULATE`` opcode.
#: Grounding: the Jacob & Monod (1961) operon model and Ptashne (2004)
#: "Genetic Regulatory Networks" — a cell can dynamically rewire its
#: regulatory graph (e.g. lacI repression) rather than fixing it at
#: compile time. Sign is set by the operand's sign bit (0 = activate,
#: 1 = inhibit).
REGULATE_EDGE_WEIGHT = 1.0

#: expression-level boost applied by one ``OP_BIND`` protein-DNA binding
#: event (transcription-factor activator). Grounding: Berg & von Hippel
#: (1987 PNAS) binding specificity and McClure (1985) — a bound activator
#: raises the target promoter's effective output; binding is protein-limited
#: (no transcription factor available => no binding).
BIND_LEVEL_BOOST = 0.5

#: ``OP_EMIT_MORPHOGEN`` injects ``(id + 1) / EMIT_MORPHOGEN_SCALE`` into the
#: field's V channel (Turing 1952 reaction-diffusion morphogens; Pearson
#: 1993 measured presets). id=0 keeps a non-zero emission.
EMIT_MORPHOGEN_SCALE = 256

#: ``OP_SIGNAL`` releases ``SIGNAL_EMISSION_AMOUNT * (1 + ch)`` (capped at
#: 1.0) of the V channel at the cell position — the quorum-sensing
#: autoinducer pool read by ``#quorum``. Grounding: Miller & Bassler
#: (2001), Xavier & Bassler (2003): AI-2 is secreted and sensed at uM
#: concentrations. The V channel stores µM AI-2 (0.5 µM per event;
#: ~5 adjacent emitters cross the 10 µM quorum threshold).
SIGNAL_EMISSION_AMOUNT = 0.5

#: ribosome density (ribosomes/100 nt mRNA) used by the central-dogma
#: pipeline (P0-1.2). Grounding: Ingolia 2009 (in vivo ribosome
#: profiling) — E. coli ribosomes load at ~1 per 100 nt.
RIBO_SOME_DENSITY_PER_100NT = 0.1

#: protein-yield coupling gain: ``protein_amount = mrna_level * yield *
#: aa_count``.  Grounding: Bernstein 2002 — one mRNA makes ~10^2-10^3
#: proteins over its lifetime; 0.1 is the normalized yield.
PROTEIN_YIELD_PER_MRNA_AA = 0.1

#: GRN feedback gain: protein abundance raises the gene's level.
PROTEIN_TO_GRN_GAIN = 0.01

#: morphogen field V concentration -> pigment-gene activation gain.
MORPHOGEN_TO_GRN_GAIN = 0.1

#: constitutive promoter reference strength (no explicit promoter).
CONSTITUTIVE_PROMOTER_STRENGTH = 0.5

__all__ = [
    "REGULATE_EDGE_WEIGHT",
    "BIND_LEVEL_BOOST",
    "EMIT_MORPHOGEN_SCALE",
    "SIGNAL_EMISSION_AMOUNT",
    "RIBO_SOME_DENSITY_PER_100NT",
    "PROTEIN_YIELD_PER_MRNA_AA",
    "PROTEIN_TO_GRN_GAIN",
    "MORPHOGEN_TO_GRN_GAIN",
    "CONSTITUTIVE_PROMOTER_STRENGTH",
]
