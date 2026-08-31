"""SBML / BiGG model import for genome-scale metabolic models (doc/24 Phase A).

Provides functions to load full genome-scale models from SBML files or the
BiGG database via cobrapy, converting them to HelixLang's internal
:class:`MetabolicModel` format with preserved GPR rules and gene registry.

Usage::

    from helixlang.plugins.gem.sbml_import import (
        load_bigg_model, load_bigg_cobra_model, load_sbml_model)

    # Load from BiGG (requires network)
    model = load_bigg_model("iML1515")

    # Load from a local vendored copy first, network only as fallback
    # (doc/41 — offline-first CI: vendored models live in
    # validation/references/models/; set HELIX_BENCHMARK_OFFLINE=1 to forbid
    # the network fallback entirely so the CI test job is deterministic).
    model = load_bigg_model("iML1515", model_dir="validation/references/models")

    # Load from local SBML file
    model = load_sbml_model("/path/to/model.xml")

The ``cobra`` package must be installed (``pip install cobra`` or
``pip install helixlang[bio]``).
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from helixlang.core.errors import BioError
from helixlang.plugins.runtime.metabolism import MetabolicModel


@contextlib.contextmanager
def _stderr_for_cobra() -> Iterator[None]:
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


def _offline_enabled() -> bool:
    """True when the caller must never use the network (``HELIX_BENCHMARK_OFFLINE=1``)."""
    return os.environ.get("HELIX_BENCHMARK_OFFLINE", "") == "1"


def _vendored_candidates(model_id: str, model_dir: Path) -> list[Path]:
    """Candidate vendored files for a BiGG model id (doc/41).

    Normalises separators/case so ``e_coli_core`` finds ``ecoli_core.json`` and
    ``iML1515`` finds ``iml1515.xml`` etc.  Tried in order (SBML before JSON);
    the first loadable candidate wins.
    """
    names = {
        model_id,
        model_id.lower(),
        model_id.replace("_", ""),
        model_id.lower().replace("_", ""),
    }
    paths: list[Path] = []
    for ext in (".xml", ".sbml", ".json"):
        for n in sorted(names):
            paths.append(model_dir / f"{n}{ext}")
    return paths


def load_bigg_cobra_model(
    model_id: str,
    model_dir: str | Path | None = None,
    offline: bool = False,
) -> object:
    """Load a BiGG model into a COBRApy model, preferring a vendored copy.

    Resolution order (doc/41 — offline-first CI):
    1. ``model_dir``: a vendored ``<id>.xml``/``<id>.sbml`` (SBML) or
       ``<id>.json`` (BiGG JSON export, e.g. ``validation/references/models/``).
    2. Network ``cobra.io.load_model(model_id)`` — unless ``offline=True`` or
       ``HELIX_BENCHMARK_OFFLINE=1``.
    3. Otherwise raise :class:`BioError` naming the model as unavailable;
       callers that cannot obtain it should SKIP their benchmark, never FAIL.

    Returns a COBRApy ``cobra.core.Model``.
    """
    cobra = _require_cobra()
    force_offline = offline or _offline_enabled()
    if model_dir is not None:
        for cand in _vendored_candidates(model_id, Path(model_dir)):
            if not cand.exists():
                continue
            try:
                with _stderr_for_cobra():
                    if cand.suffix.lower() in (".json",):
                        return cobra.io.load_json_model(str(cand))
                    return cobra.io.read_sbml_model(str(cand))
            except Exception:
                # Try the next candidate format/path; fall through to network.
                continue
    if force_offline:
        raise BioError(
            f"BiGG model {model_id!r} unavailable offline: no loadable vendored "
            f"copy in {model_dir or '<none>'}"
        )
    try:
        with _stderr_for_cobra():
            return cobra.io.load_model(model_id)
    except Exception as exc:
        raise BioError(
            f"could not load BiGG model {model_id!r}: {exc}. "
            "Check network connectivity and model ID."
        ) from exc


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
    from helixlang.plugins.runtime.metabolism import _from_cobra_model
    return _from_cobra_model(sbml_model, preserve_gpr=preserve_gpr)


def load_bigg_model(
    model_id: str,
    preserve_gpr: bool = True,
    model_dir: str | Path | None = None,
    offline: bool = False,
) -> MetabolicModel:
    """Load a genome-scale model from the BiGG database (or a vendored copy).

    Common models:
    - ``"iML1515"`` — E. coli K-12 MG1655 (Monk 2017, 2712 reactions)
    - ``"iSyn810"`` — Synechocystis PCC 6803 (Knoop 2013, 1948 reactions)
    - ``"iJO1366"`` — E. coli K-12 (Orth 2011, 2583 reactions)
    - ``"iMM904"`` — S. cerevisiae (Mo 2009, 1228 reactions)
    - ``"iBsu1103"`` — B. subtilis (Oh 2007, 2583 reactions)

    Parameters
    ----------
    model_id : BiGG model identifier (e.g. ``"iML1515"``)
    preserve_gpr : if True, preserve gene-protein-reaction rules
    model_dir : optional directory with vendored models (doc/41 offline-first).
        Files are looked up as ``<id>.xml`` / ``<id>.sbml`` / ``<id>.json``.
    offline : if True (or ``HELIX_BENCHMARK_OFFLINE=1``), never touch the
        network; a missing vendored copy raises :class:`BioError`.

    Returns
    -------
    MetabolicModel ready for FBA.
    """
    sbml_model = load_bigg_cobra_model(model_id, model_dir=model_dir, offline=offline)
    from helixlang.plugins.runtime.metabolism import _from_cobra_model
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
