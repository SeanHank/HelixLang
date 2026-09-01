"""ODE model authoring plugin (doc/42 Phase D, gap RT-1).

Lets users author original ODE models in the Helix language, not just
tune Python-hardcoded parameters.  Three grammar descriptors provide the
DSL surface:

- ``#model name="..." k1=0.1 k2=0.05 t_end=10 steps=100`` —
  model metadata + scalar parameters.
- ``#species name="..." initial=100 units="mol"`` — an ODE state
  variable (one per species).
- ``#reaction species="..." expr="-k1*A"`` — an ODE rate law for
  the named species (one per reaction).

The backend (``_run_ode_model`` in ``sim_runtime/backends/pipelines.py``)
reads the structured model from ``sim_extensions`` and integrates the
system, proving that biology can be **authored** in Helix, not just
parameterized.
"""
from __future__ import annotations

from helixlang.api.grammar import FieldSpec, GrammarDescriptor
from helixlang.api.registry import PluginProvider

ODE_MODEL_DESCRIPTOR = GrammarDescriptor(
    keyword="model",
    fields=(
        FieldSpec(key="name", type="str", required=True),
        FieldSpec(key="k1", type="float", required=True),
        FieldSpec(key="k2", type="float", required=True),
        FieldSpec(key="t_end", type="float", default="10"),
        FieldSpec(key="steps", type="int", default="100"),
    ),
    extension_key="ode_model",
)

SPECIES_DESCRIPTOR = GrammarDescriptor(
    keyword="ode_species",
    fields=(
        FieldSpec(key="name", type="str", required=True),
        FieldSpec(key="initial", type="float", required=True),
        FieldSpec(key="units", type="str", default=""),
    ),
    extension_key="ode_species",
)

REACTION_DESCRIPTOR = GrammarDescriptor(
    keyword="ode_reaction",
    fields=(
        FieldSpec(key="species", type="str", required=True),
        FieldSpec(key="expr", type="str", required=True),
    ),
    extension_key="ode_reaction",
)

PLUGIN = PluginProvider(
    name="ode_model",
    extra="ode_model",
    keywords=("model", "ode_species", "ode_reaction"),
    grammars=(ODE_MODEL_DESCRIPTOR, SPECIES_DESCRIPTOR, REACTION_DESCRIPTOR),
)

__all__ = [
    "ODE_MODEL_DESCRIPTOR",
    "PLUGIN",
    "REACTION_DESCRIPTOR",
    "SPECIES_DESCRIPTOR",
]
