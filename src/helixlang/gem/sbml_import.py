"""SBML / BiGG model import for genome-scale metabolic models (doc/24 Phase A).

Provides functions to load full genome-scale models from SBML files or the
BiGG database via cobrapy, converting them to HelixLang's internal
:class:`MetabolicModel` format with preserved GPR rules and gene registry.

Usage::

    from helixlang.gem.sbml_import import load_bigg_model, load_sbml_model

    # Load from BiGG (requires network)
    model = load_bigg_model("iML1515")

    # Load from local SBML file
    model = load_sbml_model("/path/to/model.xml")

The ``cobra`` package must be installed (``pip install cobra`` or
``pip install helixlang[bio]``).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from helixlang.errors import BioError
from helixlang.metabolism import MetabolicModel


@contextlib.contextmanager
def _stderr_for_cobra():
    """Temporarily redirect stdout to stderr.

    COBRApy's Rich progress bar writes to stdout, which contaminates
    CSV / structured output.  This wrapper keeps the progress bar
    visible on terminals (via stderr) while keeping stdout clean.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _require_cobra():  # type: ignore[no-untyped-def]
    try:
        import cobra  # noqa: F401
        return cobra
    except ImportError:
        raise BioError(
            "SBML/BiGG model loading requires the 'cobra' package. "
            "Install with: pip install cobra  (or  pip install helixlang[bio])"
        ) from None


def load_sbml_model(path: str | Path, preserve_gpr: bool = True) -> MetabolicModel:
    """Load a metabolic model from an SBML Level 2/3 XML file.

    Parameters
    ----------
    path : path to an ``.xml`` or ``.sbml`` file
    preserve_gpr : if True (default), extract gene-protein-reaction rules
        and build a gene registry on the model

    Returns
    -------
    MetabolicModel with reactions, bounds, GPR rules, and genes populated.
    """
    cobra = _require_cobra()
    p = Path(path)
    if not p.exists():
        raise BioError(f"SBML file not found: {p}")
    try:
        with _stderr_for_cobra():
            sbml_model = cobra.io.read_sbml_model(str(p))
    except Exception as exc:
        raise BioError(f"failed to parse SBML file {p.name}: {exc}") from exc
    from helixlang.metabolism import _from_cobra_model
    return _from_cobra_model(sbml_model, preserve_gpr=preserve_gpr)


def load_bigg_model(model_id: str, preserve_gpr: bool = True) -> MetabolicModel:
    """Load a genome-scale model from the BiGG database.

    Downloads the model via cobrapy's network interface.  Common models:
    - ``"iML1515"`` — E. coli K-12 MG1655 (Monk 2017, 2712 reactions)
    - ``"iSyn810"`` — Synechocystis PCC 6803 (Knoop 2013, 1948 reactions)
    - ``"iJO1366"`` — E. coli K-12 (Orth 2011, 2583 reactions)
    - ``"iMM904"`` — S. cerevisiae (Mo 2009, 1228 reactions)
    - ``"iBsu1103"`` — B. subtilis (Oh 2007, 2583 reactions)

    Parameters
    ----------
    model_id : BiGG model identifier (e.g. ``"iML1515"``)
    preserve_gpr : if True, preserve gene-protein-reaction rules

    Returns
    -------
    MetabolicModel ready for FBA.
    """
    cobra = _require_cobra()
    try:
        with _stderr_for_cobra():
            sbml_model = cobra.io.load_model(model_id)
    except Exception as exc:
        raise BioError(
            f"could not load BiGG model {model_id!r}: {exc}. "
            "Check network connectivity and model ID."
        ) from exc
    from helixlang.metabolism import _from_cobra_model
    return _from_cobra_model(sbml_model, preserve_gpr=preserve_gpr)


def download_bigg_model(model_id: str, output_path: str | Path) -> Path:
    """Download a BiGG model as SBML XML and save to a local file.

    Useful for offline use and caching.
    """
    cobra = _require_cobra()
    try:
        with _stderr_for_cobra():
            sbml_model = cobra.io.load_model(model_id)
    except Exception as exc:
        raise BioError(f"could not download BiGG model {model_id!r}: {exc}") from exc
    out = Path(output_path)
    cobra.io.write_sbml_model(sbml_model, str(out))
    return out


def detect_exchange_reactions(model: MetabolicModel) -> list[str]:
    """Auto-detect exchange reactions in a model.

    Exchange reactions are identified by:
    1. ID starts with ``EX_`` (BiGG convention)
    2. Subsystem contains ``"exchange"`` or ``"Exchange"``
    3. Single-metabolite reactions (boundary condition)

    Returns sorted list of exchange reaction IDs.
    """
    exchanges: list[str] = []
    for rid, rxn in model.reactions.items():
        if rid.startswith("EX_") or "exchange" in rxn.subsystem.lower():
            exchanges.append(rid)
        elif len(rxn.stoichiometry) == 1 and (rxn.lower_bound < 0 or rxn.upper_bound > 0):
            # single-metabolite reactions with non-zero bounds are likely exchanges
            if rid.startswith("EX_"):
                exchanges.append(rid)
    return sorted(set(exchanges))


def detect_compartments(model: MetabolicModel) -> dict[str, list[str]]:
    """Detect metabolite compartments from ID suffixes.

    BiGG convention: ``metabolite_suffix`` where suffix is ``_c`` (cytosol),
    ``_e`` (extracellular), ``_p`` (periplasm), ``_m`` (mitochondria), etc.

    Returns dict of compartment_suffix -> [metabolite_ids].
    """
    compartments: dict[str, list[str]] = {}
    for met in model.metabolites:
        if "_" in met:
            suffix = met.rsplit("_", 1)[-1]
            if len(suffix) <= 3 and suffix.isalnum():
                compartments.setdefault(suffix, []).append(met)
            else:
                compartments.setdefault("unknown", []).append(met)
        else:
            compartments.setdefault("bare", []).append(met)
    return {k: sorted(v) for k, v in sorted(compartments.items())}


def get_model_info(model: MetabolicModel) -> dict:
    """Get summary information about a metabolic model.

    Returns dict with keys: n_reactions, n_metabolites, n_genes,
    n_exchange, biomass_reaction, compartments, exchange_reactions.
    """
    exchanges = detect_exchange_reactions(model)
    compartments = detect_compartments(model)
    return {
        "n_reactions": len(model.reactions),
        "n_metabolites": len(model.metabolites),
        "n_genes": len(model.genes),
        "n_exchange": len(exchanges),
        "biomass_reaction": model.biomass_reaction,
        "compartments": {k: len(v) for k, v in compartments.items()},
        "exchange_reactions": exchanges[:20],  # first 20 for brevity
    }
