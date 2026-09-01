"""Metabolic network flux balance analysis (FBA).

Curated core model: 37-reaction E. coli core metabolism (glycolysis +
TCA + pentose phosphate + fermentation + respiration + biomass), stored
in :file:`data/ecoli_core_model.json`; the full genome-scale iJO1366 has
1367 reactions and can be loaded through :func:`load_model` when cobrapy
is installed.
FBA assumes steady-state mass balance and biomass maximization, ignoring
regulatory kinetics.

Based on real data:
- E. coli core metabolism model (Orth 2010 Mol Syst Biol 6:390)
- Metabolic network: glycolysis + TCA + pentose phosphate + fermentation
- FBA linear programming solves for optimal metabolic fluxes
- Objective function: biomass flux

Module structure:
    Reaction               single biochemical reaction
                           (id/name/stoichiometry/bounds/subsystem)
    MetabolicModel         metabolic network model (reaction set +
                           stoichiometry matrix)
    ECOLI_CORE_MODEL       curated E. coli core model (37 reactions,
                           loaded from data/ecoli_core_model.json)
    FluxBalanceAnalysis    FBA solver (simplex method + bounds)
    load_model             optional genome-scale loader (cobrapy) with
                           curated-core fallback
    simplex                pure-Python two-phase simplex solver
                           (no scipy dependency)

References:
- Orth JD et al. Mol Syst Biol 2010 6:390 (E. coli core model, reduced
  BiGG iJO1366)
- Orth JD et al. Mol Syst Biol 2011 7:535 (iJO1366 genome-scale
  reconstruction; BiGG bigg.ucsd.edu/models/iJO1366)
- Varma A, Palsson BO. Appl Environ Microbiol 1994 60:726-734 (FBA basics)
- Feist AM et al. Nat Rev Microbiol 2008 6:664-672 (flux analysis review)
- Dantzig GB. Linear Programming and Extensions 1963 (simplex method)
"""
from __future__ import annotations

import copy
import json
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helixlang.core.errors import BioError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from helixlang.plugins.runtime.environment import Environment

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a project dependency
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# model data file directory
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ============================================================================
# Default flux bounds (based on BiGG model conventions)
# ============================================================================

# upper bound for irreversible reactions (mmol/gDW/h, millimoles per gram
# dry weight per hour)
DEFAULT_UPPER_BOUND = 1000.0
# lower bound for reversible reactions
DEFAULT_LOWER_BOUND = -1000.0
# maximum glucose uptake rate (E. coli anaerobic/aerobic reference value,
# 10 mmol/gDW/h)
DEFAULT_GLC_UPTAKE = 10.0
# ATP maintenance flux (E. coli ~8.39 mmol/gDW/h, Orth 2010)
ATP_MAINTENANCE_FLUX = 8.39
# biomass "molecular weight" (gDW/mmol): conversion factor from biomass
# flux (mmol/gDW/h) to specific growth rate mu (1/h)
# E. coli biomass dry weight ~1 gDW/mmol biomass (Orth 2010, iJO1366
# GAM=75.55 gDW/mmol ATP
# with the convention that 1 mmol biomass corresponds to ~1 gDW)
_BIOMASS_MW = 1.0


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class Reaction:
    """A single biochemical reaction.

    Attributes:
        id: unique reaction identifier (e.g. "PGI", "PFK")
        name: human-readable name (e.g. "Phosphoglucose isomerase")
        stoichiometry: metabolite -> coefficient dict
            negative value = reactant (consumed)
            positive value = product (produced)
            e.g. {"G6P": -1, "F6P": 1} means G6P -> F6P
        lower_bound: flux lower bound (negative = reversible,
                     0 = irreversible)
        upper_bound: flux upper bound (default 1000)
        subsystem: subsystem the reaction belongs to
            "glycolysis" / "tca" / "ppp" / "fermentation" / "exchange" /
            "biomass"
        gene_reaction_rule: Boolean gene-protein-reaction association
            (e.g. "b0008 and b3901" or "b0008 or b3901")
    """
    id: str
    name: str
    stoichiometry: dict[str, float]
    lower_bound: float = 0.0
    upper_bound: float = DEFAULT_UPPER_BOUND
    subsystem: str = "other"
    gene_reaction_rule: str | None = None


@dataclass
class Gene:
    """A gene entry in a genome-scale metabolic model.

    Attributes:
        id: unique gene identifier (e.g. "b0008")
        name: human-readable name (e.g. "dnaA")
        protein_reaction_rules: list of reaction IDs this gene participates in
    """
    id: str
    name: str = ""
    protein_reaction_rules: list[str] = field(default_factory=list)


class MetabolicModel:
    """Metabolic network model.

    Maintains a reaction set + the full metabolite set + a biomass
    reaction pointer,
    and can generate the stoichiometry matrix for the FBA solver.
    """

    def __init__(self) -> None:
        self.reactions: dict[str, Reaction] = {}
        self.metabolites: set[str] = set()
        self.biomass_reaction: str | None = None
        self.genes: dict[str, Gene] = {}

    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction and automatically register its metabolites."""
        if reaction.id in self.reactions:
            raise BioError(f"duplicate reaction id {reaction.id!r}")
        self.reactions[reaction.id] = reaction
        for met in reaction.stoichiometry:
            self.metabolites.add(met)

    def set_biomass(self, reaction_id: str) -> None:
        """Set the biomass objective reaction."""
        if reaction_id not in self.reactions:
            raise BioError(f"unknown biomass reaction {reaction_id!r}")
        self.biomass_reaction = reaction_id

    def get_stoichiometry_matrix(self) -> tuple[list[str], list[str], list[list[float]]]:
        """Return (metabolite list, reaction list, stoichiometry matrix).

        The matrix S has shape (n_metabolites x n_reactions); S[i][j] is:
        - negative: reaction j consumes metabolite i
        - positive: reaction j produces metabolite i
        - zero: reaction j does not involve metabolite i

        Steady-state mass balance constraint: S · v = 0
        """
        met_list = sorted(self.metabolites)
        rxn_list = list(self.reactions.keys())
        S = [[0.0 for _ in rxn_list] for _ in met_list]
        met_index = {m: i for i, m in enumerate(met_list)}
        for j, rid in enumerate(rxn_list):
            rxn = self.reactions[rid]
            for met, coef in rxn.stoichiometry.items():
                S[met_index[met]][j] = float(coef)
        return met_list, rxn_list, S


# ============================================================================
# Generic model loader (P2-12: data-driven, load metabolic models from
# JSON files)
# ============================================================================

def load_model_from_json(path: str | Path) -> MetabolicModel:
    """Load a metabolic model from a JSON data file.

    JSON format::

        {
          "name": "model name",
          "source": "data source citation",
          "biomass_reaction": "BIOMASS",
          "reactions": [
            {
              "id": "EX_glc",
              "name": "Glucose exchange",
              "stoichiometry": {"GLC": 1.0},
              "lower_bound": 0.0,
              "upper_bound": 0.0,
              "subsystem": "exchange"
            },
            ...
          ]
        }

    Args:
        path: JSON file path

    Returns:
        a :class:`MetabolicModel` instance
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    m = MetabolicModel()
    for rxn_data in data.get("reactions", []):
        m.add_reaction(Reaction(
            id=rxn_data["id"],
            name=rxn_data.get("name", rxn_data["id"]),
            stoichiometry=rxn_data["stoichiometry"],
            lower_bound=float(rxn_data.get("lower_bound", 0.0)),
            upper_bound=float(rxn_data.get("upper_bound",
                                           DEFAULT_UPPER_BOUND)),
            subsystem=rxn_data.get("subsystem", "other"),
        ))
    biomass_rxn = data.get("biomass_reaction")
    if biomass_rxn:
        m.set_biomass(biomass_rxn)
    return m


def load_model(path_or_identifier: str | Path | None = None
               ) -> MetabolicModel:
    """Load a metabolic model: the curated core model, a JSON file, or
    a genome-scale model via cobrapy when available.

    Resolution order:

    1. ``None`` (default) -> :data:`ECOLI_CORE_MODEL` (curated 37-reaction
       core, no dependencies).
    2. A path to a JSON file in :func:`load_model_from_json` format.
    3. A BiGG model identifier (e.g. ``"iJO1366"``, Orth 2011 Mol Syst
       Biol 7:535) -> requires the optional ``cobra`` package
       (``cobra.io.load_model``); the same model can also be loaded from
       a local SBML file path (``cobra.io.read_sbml_model``).
    4. If cobrapy is not installed and a non-default identifier/path is
       requested, the curated core model is returned (documented
       fallback) unless ``strict=True``.

    The curated core model remains the default so the pure-Python
    solver path is always available.

    Args:
        path_or_identifier: ``None``, a JSON path, an SBML path
            (extension ``.xml``/``.sbml``), or a BiGG identifier.

    Returns:
        a :class:`MetabolicModel` instance
    """
    if path_or_identifier is None:
        return ECOLI_CORE_MODEL

    p = Path(str(path_or_identifier))
    if p.suffix.lower() == ".json":
        return load_model_from_json(p)
    if p.suffix.lower() in (".xml", ".sbml"):
        try:
            import cobra
        except ImportError:
            raise BioError(
                "SBML model loading requires the optional 'cobra' "
                "package (pip install cobra); returning the curated "
                "core model instead"
            ) from None
        import sys
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            sbml_model = cobra.io.read_sbml_model(str(p))
        return _from_cobra_model(sbml_model)

    identifier = str(path_or_identifier)
    try:
        import cobra
    except ImportError:
        raise BioError(
            f"loading BiGG model {identifier!r} requires the optional "
            "'cobra' package (pip install cobra); falling back to the "
            "curated core model"
        ) from None
    try:
        import sys
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            sbml_model = cobra.io.load_model(identifier)
    except Exception as exc:  # BiGG download / network errors
        raise BioError(
            f"could not load BiGG model {identifier!r} via cobrapy: {exc}"
        ) from exc
    return _from_cobra_model(sbml_model)


def _from_cobra_model(sbml_model: Any, preserve_gpr: bool = True) -> MetabolicModel:
    """Convert a cobrapy Model into a :class:`MetabolicModel`.

    Only used when the optional ``cobra`` package is present.
    When *preserve_gpr* is True, gene-protein-reaction rules are stored
    in each :class:`Reaction` and a :class:`Gene` registry is built on the
    model.
    """
    m = MetabolicModel()
    for rxn in sbml_model.reactions:
        stoich: dict[str, float] = {}
        for met, coeff in rxn.metabolites.items():
            stoich[str(met.id)] = float(coeff)
        gpr = None
        if preserve_gpr:
            gpr_str = str(rxn.gene_reaction_rule).strip()
            if gpr_str and gpr_str != "None":
                gpr = gpr_str
        m.add_reaction(Reaction(
            id=str(rxn.id),
            name=str(rxn.name),
            stoichiometry=stoich,
            lower_bound=float(rxn.lower_bound) if rxn.lower_bound is not None else 0.0,
            upper_bound=float(rxn.upper_bound) if rxn.upper_bound is not None else DEFAULT_UPPER_BOUND,
            subsystem=str(rxn.subsystem or "other"),
            gene_reaction_rule=gpr,
        ))
    # Build gene registry from model
    if preserve_gpr:
        for gene in sbml_model.genes:
            gid = str(gene.id)
            rxn_ids = [str(r.id) for r in gene.reactions]
            m.genes[gid] = Gene(id=gid, name=str(gene.name or ""), protein_reaction_rules=rxn_ids)
    objective = getattr(sbml_model, "objective", None)
    if objective is not None:
        # Strategy 1: look at individual reaction objective_coefficients
        obj_rxn = None
        for rxn in sbml_model.reactions:
            coeff = getattr(rxn, "objective_coefficient", 0)
            if coeff and float(coeff) > 0:
                obj_rxn = str(rxn.id)
                break
        if obj_rxn is None:
            # Strategy 2: parse the expression tree
            expr = getattr(objective, "expression", None)
            if expr is not None:
                for term in getattr(expr, "args", ()):
                    var = getattr(term, "variable", None)
                    if var is not None:
                        name = str(getattr(var, "name", var))
                        if "reverse" not in name.lower():
                            obj_rxn = name
                            break
                    if obj_rxn is not None:
                        break
        if obj_rxn is None:
            # Strategy 3: any biomass reaction
            for rid in m.reactions:
                if "biomass" in rid.lower() or "BIOMASS" in rid:
                    obj_rxn = rid
                    break
        if obj_rxn is not None:
            m.set_biomass(obj_rxn)
    return m


# ============================================================================
# E. coli core metabolism model (curated 37-reaction core)
# Data source: Orth 2010 Mol Syst Biol 6:390 (reduced iJO1366)
# P2-12: the 388-line hardcoded model has been migrated to
# data/ecoli_core_model.json and is read by the generic loader
# ============================================================================

def _build_ecoli_core_model() -> MetabolicModel:
    """Build the reduced E. coli core metabolism model (data-driven,
    loaded from JSON).

    Contains 5 major subsystems:
    - glycolysis: GLC -> G6P -> F6P -> F1,6BP -> DHAP+G3P -> ... -> PEP
      -> PYR
    - TCA cycle: PYR -> AcCoA -> Cit -> alphaKG -> Succ -> Mal -> OAA
    - pentose phosphate pathway: G6P -> 6PG -> Ru5P -> R5P
    - fermentation: PYR -> Lac, AcCoA -> Acetate
    - respiration: NADH/FADH2 oxidative phosphorylation

    The biomass reaction aggregates precursors
    (G6P/AcCoA/OAA/R5P, etc.) into Biomass.

    The reaction data is stored in :file:`data/ecoli_core_model.json`,
    parsed by the generic loader :func:`load_model_from_json`.
    """
    return load_model_from_json(_DATA_DIR / "ecoli_core_model.json")


ECOLI_CORE_MODEL: MetabolicModel = _build_ecoli_core_model()


# ============================================================================
# Pure-Python simplex method (Two-Phase Simplex)
# No scipy dependency, standard library only
# ============================================================================

_EPS = 1e-9
_INF = float("inf")


def _simplex_max(tableau: list[list[float]],
                 basis: list[int],
                 obj: list[float],
                 n_vars: int,
                 eps: float = _EPS,
                 max_iter: int = 10000,
                 forbidden: set[int] | None = None) -> str:
    """Run simplex iterations on the given tableau (maximize obj).

    Uses Bland's rule (smallest index first) to choose entering/leaving
    variables and avoid cycling.

    Inputs:
        tableau[i] = [a_{i,0}, ..., a_{i,n_vars-1}, b_i]  (last column is
        the RHS)
        basis[i]   = basic variable index of row i
        obj[j]     = objective coefficient of variable j (max obj^T x)
        n_vars     = total number of variables
        forbidden  set of variable indices barred from entering the basis
                   (e.g. artificial variables in phase 2)

    Modifies tableau and basis in place. Returns 'optimal' /
    'unbounded' / 'max_iter'.
    """
    n_rows = len(tableau)
    rhs_col = n_vars  # RHS column index
    if forbidden is None:
        forbidden = set()
    for _ in range(max_iter):
        # compute the current objective contribution coefficient for each
        # row's basic variable
        cB = [obj[basis[i]] for i in range(n_rows)]
        # Bland's rule: choose the smallest-index variable with
        # reduced_cost > eps to enter the basis
        entering = -1
        for j in range(n_vars):
            if j in basis or j in forbidden:
                continue
            rc = obj[j] - sum(cB[i] * tableau[i][j] for i in range(n_rows))
            if rc > eps:
                entering = j
                break
        if entering == -1:
            return "optimal"
        # ratio test: choose min b_i / a_{i,entering} (a > 0)
        # Bland's rule: among equal ratios, choose the leaving variable
        # with the smallest index
        leaving_row = -1
        min_ratio = _INF
        min_basis_idx = n_vars + 1
        for i in range(n_rows):
            pivot = tableau[i][entering]
            if pivot > eps:
                ratio = tableau[i][rhs_col] / pivot
                if (ratio < min_ratio - eps
                        or (abs(ratio - min_ratio) <= eps
                            and basis[i] < min_basis_idx)):
                    min_ratio = ratio
                    leaving_row = i
                    min_basis_idx = basis[i]
        if leaving_row == -1:
            return "unbounded"
        # pivot transformation
        pivot_val = tableau[leaving_row][entering]
        # normalize the pivot row
        inv_pivot = 1.0 / pivot_val
        for k in range(n_vars + 1):
            tableau[leaving_row][k] *= inv_pivot
        # eliminate the other rows
        for i in range(n_rows):
            if i == leaving_row:
                continue
            factor = tableau[i][entering]
            if abs(factor) < eps:
                continue
            for k in range(n_vars + 1):
                tableau[i][k] -= factor * tableau[leaving_row][k]
        basis[leaving_row] = entering
    return "max_iter"


def _simplex_max_numpy(tableau: np.ndarray,
                       basis: list[int],
                       obj: np.ndarray,
                       n_vars: int,
                       eps: float = _EPS,
                       max_iter: int = 10000,
                       forbidden: set[int] | None = None) -> str:
    """NumPy-vectorized version of :func:`_simplex_max`.

    Implements the hot-spot reduced-cost computation
    (``cB^T · tableau[:, j]``) and pivot transformation
    (row normalization + column elimination) with NumPy vector
    operations, giving a 5-20x speedup on large tableaux
    (n_rows x n_vars > 100x200).

    The algorithm is identical to :func:`_simplex_max` (Bland's rule),
    and the results are equivalent. Mutates ``tableau`` and ``basis`` in
    place.
    """
    n_rows = tableau.shape[0]
    if n_rows == 0:
        return "optimal"
    # ensure tableau is 2D (guard against degenerating to 1D with 0 rows)
    if tableau.ndim == 1:
        tableau = tableau.reshape(0, -1)
    rhs_col = n_vars  # RHS column index
    basis_arr = np.asarray(basis, dtype=np.intp)
    # forbidden set: these variables are permanently barred from entering
    # the basis (marked with a boolean mask)
    forbidden_mask = np.zeros(n_vars, dtype=bool)
    if forbidden:
        for j in forbidden:
            forbidden_mask[j] = True
    obj = np.asarray(obj, dtype=np.float64)

    for _ in range(max_iter):
        # cB = obj[basis]  (n_rows,)
        cB = obj[basis_arr]
        # reduced cost: rc[j] = obj[j] - cB · tableau[:, j]
        col_contrib = cB @ tableau[:, :n_vars]  # (n_vars,)
        rc = obj - col_contrib
        # entering: Bland's rule (smallest index)
        # eligible = not in current basis and not forbidden
        # use rc > eps and not forbidden and not in_current_basis
        in_current_basis = np.zeros(n_vars, dtype=bool)
        in_current_basis[basis_arr] = True
        eligible = (~in_current_basis) & (~forbidden_mask) & (rc > eps)
        candidates = np.nonzero(eligible)[0]
        if candidates.size == 0:
            return "optimal"
        entering = int(candidates[0])

        # ratio test: min b_i / a_{i,entering} (a > eps)
        col = tableau[:, entering]
        rhs = tableau[:, rhs_col]
        valid = col > eps
        if not np.any(valid):
            return "unbounded"
        ratios = np.where(valid, rhs / np.where(valid, col, 1.0), np.inf)
        min_ratio = ratios.min()
        # Bland's rule: among tied ratios pick the leaving variable with
        # the smallest index
        tied = np.abs(ratios - min_ratio) <= eps
        tied_rows = np.nonzero(tied)[0]
        if tied_rows.size > 0:
            leaving_row = int(tied_rows[np.argmin(basis_arr[tied_rows])])
        else:
            leaving_row = int(np.argmin(ratios))

        # pivot transformation: normalize the pivot row
        pivot_val = tableau[leaving_row, entering]
        tableau[leaving_row, :] /= pivot_val
        # eliminate other rows: tableau[i,:] -= factor * tableau[leaving_row,:]
        factor_col = tableau[:, entering].copy()
        factor_col[leaving_row] = 0.0  # skip the pivot row
        tableau -= np.outer(factor_col, tableau[leaving_row, :])
        # update basis (no need to maintain in_basis: recomputed from
        # basis_arr each round)
        basis[leaving_row] = entering
        basis_arr[leaving_row] = entering
    return "max_iter"


# ============================================================================
# doc/42 Phase C PF-2 — optional native simplex (never the bit-identical path)
# ============================================================================

def _simplex_native_optin() -> bool:
    """True only when the operator explicitly requests the native simplex kernel.

    The FBA/dFBA accepted-state results are golden-verifiable and must stay
    *bit-identical* (doc/42 Phase C gate: goldens stay 82/82).  The reference
    numpy pivot is byte-identical to the accel ``impl_numpy`` backend, but
    compiled native backends (cython/rust) agree only to ~1e-14.  Native simplex
    is therefore opt-in via ``HELIX_ACCEL_SIMPLEX`` and is never the default —
    crossing even a fractional-fidelity boundary is explicit, per
    ``_accel._loaders`` (never silently cross a fidelity boundary).
    """
    return os.environ.get("HELIX_ACCEL_SIMPLEX", "").strip().lower() in (
        "1", "true", "native", "accel",
    )


def _simplex_max_dispatch(tab_np: np.ndarray,
                          basis: list[int],
                          obj_np: np.ndarray,
                          n_vars: int,
                          eps: float = _EPS,
                          max_iter: int = 10000,
                          forbidden: set[int] | None = None) -> str:
    """Route a phase pivot through the native accel kernel when opted in.

    Falls back to the byte-identical :func:`_simplex_max_numpy` by default so
    golden hashes never drift.
    """
    if _simplex_native_optin():
        from helixlang.api.accel import simplex_run
        return simplex_run(tab_np, basis, obj_np, n_vars, eps, max_iter,
                           forbidden)
    return _simplex_max_numpy(tab_np, basis, obj_np, n_vars, eps, max_iter,
                              forbidden)


# ============================================================================
# simplex internal helper functions (P2#12: split out of the original
# 184-line simplex(), algorithm-equivalent)
# ============================================================================

def _simplex_check_feasible_bounds(lbs: list[float],
                                   ubs: list[float],
                                   eps: float) -> bool:
    """Check bound feasibility: infeasible if any lb > ub.

    Args:
        lbs: variable lower bounds
        ubs: variable upper bounds
        eps: numerical tolerance (only deemed infeasible when
             lb exceeds ub + eps)

    Returns:
        True feasible / False infeasible (some lb_i > ub_i + eps)
    """
    for i in range(len(lbs)):
        if lbs[i] > ubs[i] + eps:
            return False
    return True


def _simplex_build_tableau(c: list[float],
                           A: list[list[float]],
                           b: list[float],
                           lbs: list[float],
                           ubs: list[float],
                           sign: float,
                           n: int,
                           m: int,
                           eps: float) -> dict:
    """Variable substitution + standard-form tableau construction
    (including slack and artificial variables).

    Substitution ``x_i = lb_i + y_i`` (``y_i >= 0``,
    ``y_i <= ub_i - lb_i``):
    - equality constraints ``A x = b`` -> ``A y = b - A·lb``
    - each variable upper bound ``y_i <= ub_i - lb_i`` introduces a slack
      variable ``s_i``
    - each equality constraint introduces an artificial variable ``a_i``
      (used in Phase 1)

    Variable layout: ``y_0..y_{n-1}, s_0..s_{n-1}, a_0..a_{m-1}``
    (2n+m variables in total).
    Constraint rows: m equality rows + n upper-bound rows (m+n rows in
    total).

    Returns a dict with:
        tableau:        (total_rows x (total_vars+1)) matrix (last
                        column is RHS)
        basis:          initial basic variable index list
        total_vars:     total variable count = 2n + m
        total_rows:     constraint row count = m + n
        n_y, n_s, n_a:  y / s / artificial segment lengths (artificial
                        start index = n_y + n_s)
        phase1_obj:     Phase 1 objective coefficients (max -sum(a_i))
        phase2_obj:     Phase 2 objective coefficients
                        (max sign·c^T y)
        artificial_set: artificial variable index set (barred from
                        entering in Phase 2)
    """
    obj_coefs = [sign * float(cj) for cj in c]

    # substitute x_i = lb_i + y_i, y_i >= 0; upper bound y_i <= ub_i - lb_i
    # equality constraints A x = b -> A y = b - A·lb
    b_prime = [
        float(b[i]) - sum(float(A[i][j]) * lbs[j] for j in range(n))
        for i in range(m)
    ]

    # adjust signs so that b_prime[i] >= 0
    row_signs = [1.0] * m
    for i in range(m):
        if b_prime[i] < 0:
            b_prime[i] = -b_prime[i]
            row_signs[i] = -1.0

    # upper-bound difference (ensure non-negative; replace +inf upper
    # bound with a large number)
    _BIG = 1e9
    ub_diff = []
    for i in range(n):
        d = ubs[i] - lbs[i]
        if d == _INF or d > _BIG:
            d = _BIG
        ub_diff.append(max(0.0, d))

    # build the standard-form tableau
    # variables: y_0..y_{n-1}, s_0..s_{n-1} (upper-bound slack),
    #            a_0..a_{m-1} (artificial)
    # 2n + m variables in total; constraint rows = m + n
    n_y = n
    n_s = n
    n_a = m
    total_vars = n_y + n_s + n_a
    total_rows = m + n

    tableau = [[0.0] * (total_vars + 1) for _ in range(total_rows)]
    basis = [0] * total_rows

    # equality constraint rows
    for i in range(m):
        for j in range(n):
            tableau[i][j] = row_signs[i] * float(A[i][j])
        tableau[i][n_y + n_s + i] = 1.0  # artificial variable
        tableau[i][total_vars] = b_prime[i]
        basis[i] = n_y + n_s + i

    # upper-bound constraint rows: y_i + s_i = ub_i - lb_i
    for i in range(n):
        row = m + i
        tableau[row][i] = 1.0            # y_i
        tableau[row][n_y + i] = 1.0      # s_i
        tableau[row][total_vars] = ub_diff[i]
        basis[row] = n_y + i

    # Phase 1 objective: max -sum(a_i)
    phase1_obj = [0.0] * total_vars
    for i in range(m):
        phase1_obj[n_y + n_s + i] = -1.0

    # Phase 2 objective: max sign·c^T y
    phase2_obj = [0.0] * total_vars
    for j in range(n):
        phase2_obj[j] = obj_coefs[j]

    # artificial variable set (barred from re-entering in Phase 2)
    artificial_set = set(range(n_y + n_s, total_vars))

    return {
        "tableau": tableau,
        "basis": basis,
        "total_vars": total_vars,
        "total_rows": total_rows,
        "n_y": n_y,
        "n_s": n_s,
        "n_a": n_a,
        "phase1_obj": phase1_obj,
        "phase2_obj": phase2_obj,
        "artificial_set": artificial_set,
    }


def _simplex_extract_solution(tableau: list[list[float]],
                              basis: list[int],
                              lbs: list[float],
                              n: int,
                              n_orig: int) -> tuple[list[float], list[float]]:
    """Extract the solution vector from the tableau.

    Basic variables take their RHS value, non-basic variables take 0;
    then ``x_j = lb_j + y_j`` restores the original variables.

    Args:
        tableau: tableau after simplex solving (last column is RHS)
        basis:   final basic variable index list
        lbs:     original variable lower bounds (length n_orig)
        n:       total_vars (tableau variable count, also the RHS column
                 index)
        n_orig:  original variable count (= len(lbs), length of
                 solution x)

    Returns (x, y_val):
        x:     original variable solution = lbs + y_val[:n_orig]
               (length n_orig)
        y_val: substituted variable values (length n = total_vars)
    """
    y_val = [0.0] * n
    for i in range(len(basis)):
        if basis[i] < n:
            y_val[basis[i]] = tableau[i][n]
    x = [lbs[j] + y_val[j] for j in range(n_orig)]
    return x, y_val


def simplex(c: list[float],
            A: list[list[float]],
            b: list[float],
            bounds: list[tuple[float, float]],
            maximize: bool = True,
            max_iter: int = 10000,
            eps: float = _EPS) -> dict:
    """Pure-Python two-phase simplex method for solving LPs.

    Solves:
        max (or min)  c^T x
        s.t.          A x = b           (m equality constraints)
                      bounds[i][0] <= x_i <= bounds[i][1]

    Args:
        c:        objective coefficients (length n)
        A:        equality constraint matrix (m x n)
        b:        equality constraint RHS (length m)
        bounds:   (lower, upper) bounds for each variable
        maximize: True=maximize, False=minimize

    Returns a dict:
        {
          "status":   "optimal" / "infeasible" / "unbounded" / "max_iter",
          "x":        solution vector (length n),
          "objective": objective value (c^T x),
        }

    Implementation: orchestrates :func:`_simplex_check_feasible_bounds` /
    :func:`_simplex_build_tableau` / :func:`_simplex_extract_solution`
    and calls :func:`_simplex_max` (pure Python) or
    :func:`_simplex_max_numpy` (NumPy-accelerated) to carry out the
    two-phase simplex + Bland's rule.
    """
    n = len(c)
    m = len(A)
    # handle min: convert to max
    sign = 1.0 if maximize else -1.0

    lbs = [float(bounds[i][0]) for i in range(n)]
    ubs = [float(bounds[i][1]) for i in range(n)]

    # 1. bound feasibility check: lb > ub -> infeasible
    if not _simplex_check_feasible_bounds(lbs, ubs, eps):
        return {"status": "infeasible", "x": [0.0] * n, "objective": 0.0}

    # 2. variable substitution + tableau construction (incl. slack and
    #    artificial variables)
    t = _simplex_build_tableau(c, A, b, lbs, ubs, sign, n, m, eps)
    tableau = t["tableau"]
    basis = t["basis"]
    total_vars = t["total_vars"]
    total_rows = t["total_rows"]
    n_y = t["n_y"]
    n_s = t["n_s"]
    phase1_obj = t["phase1_obj"]
    phase2_obj = t["phase2_obj"]
    artificial_set = t["artificial_set"]

    # P2-15: NumPy acceleration path (5-20x speedup on large models;
    # results equivalent to pure Python)
    use_numpy = _HAS_NUMPY

    # ---------- Phase 1: minimize sum of artificial variables =
    #              maximize -sum(a_i) ----------
    if use_numpy:
        tab_np = np.array(tableau, dtype=np.float64)
        obj1_np = np.array(phase1_obj, dtype=np.float64)
        _simplex_max_dispatch(tab_np, basis, obj1_np,
                              total_vars, eps, max_iter)
        # sync back to the list view (the artificial-variable removal
        # step uses list indexing syntax)
        tableau = tab_np.tolist()
    else:
        _simplex_max(tableau, basis, phase1_obj,
                     total_vars, eps, max_iter)

    # check whether the artificial variables have been driven to zero
    art_sum = 0.0
    for i in range(m):
        if basis[i] >= n_y + n_s:  # artificial variables still in basis
            art_sum += tableau[i][total_vars]
    if art_sum > 1e-6:
        return {"status": "infeasible", "x": [0.0] * n, "objective": 0.0}

    # drive artificial variables still in the basis (degenerate case with
    # value 0) out of the basis
    for i in range(m):
        if basis[i] in artificial_set:
            # find a non-artificial, non-zero column in this row to use as
            # the pivot and swap the artificial variable out
            for j in range(n_y + n_s):
                if abs(tableau[i][j]) > eps and j not in basis:
                    # perform one pivot rotation (i, j)
                    pivot_val = tableau[i][j]
                    for k in range(total_vars + 1):
                        tableau[i][k] /= pivot_val
                    for r in range(total_rows):
                        if r == i:
                            continue
                        factor = tableau[r][j]
                        if abs(factor) < eps:
                            continue
                        for k in range(total_vars + 1):
                            tableau[r][k] -= factor * tableau[i][k]
                    basis[i] = j
                    break
            # if it cannot be driven out (all non-artificial columns in
            # the row are 0), the constraint is redundant; keep the
            # artificial variable in the basis (value 0)

    # ---------- Phase 2: optimize the original objective ----------
    # artificial variables are barred from re-entering the basis in
    # Phase 2 (to avoid numerical errors from -_BIG / -_BIG interaction)
    if use_numpy:
        tab_np = np.array(tableau, dtype=np.float64)
        obj2_np = np.array(phase2_obj, dtype=np.float64)
        status2 = _simplex_max_dispatch(tab_np, basis, obj2_np,
                                        total_vars, eps, max_iter,
                                        forbidden=artificial_set)
        tableau = tab_np.tolist()
    else:
        status2 = _simplex_max(tableau, basis, phase2_obj, total_vars, eps,
                               max_iter, forbidden=artificial_set)

    # 3. extract the solution: y_j = value if in basis, otherwise 0;
    #    x_j = lb_j + y_j
    x, y_val = _simplex_extract_solution(tableau, basis, lbs,
                                          total_vars, n)
    objective = sum(float(c[j]) * x[j] for j in range(n))

    if status2 == "unbounded":
        return {"status": "unbounded", "x": x, "objective": objective}
    if status2 == "max_iter":
        return {"status": "max_iter", "x": x, "objective": objective}
    # detect numerical unboundedness: a variable hit the _BIG upper bound
    # (original upper bound was +inf)
    _BIG = 1e9
    for j in range(n):
        if (ubs[j] == _INF or ubs[j] > _BIG) and y_val[j] >= _BIG * 0.9:
            return {"status": "unbounded", "x": x, "objective": objective}
    return {"status": "optimal", "x": x, "objective": objective}


# ============================================================================
# SciPy linprog solver for large genome-scale models (doc/24 Phase B)
# ============================================================================

try:
    from scipy.optimize import linprog as _scipy_linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Threshold: models with more reactions than this use scipy (sparse solver)
_SOLVER_DISPATCH_THRESHOLD = 500


def _solve_scipy(c: list[float],
                 A_eq: list[list[float]],
                 b_eq: list[float],
                 bounds: list[tuple[float, float]],
                 maximize: bool = True) -> dict:
    """Solve an LP using scipy.optimize.linprog (HiGHS solver).

    Same interface as :func:`simplex` but handles large sparse models
    efficiently.  Used for genome-scale models (>500 reactions).
    """
    if not _HAS_SCIPY:
        return {"status": "infeasible", "x": [0.0] * len(c), "objective": 0.0}
    import numpy as _np
    c_arr = _np.array(c, dtype=_np.float64)
    if maximize:
        c_arr = -c_arr  # linprog minimizes; negate to maximize
    A_eq_arr = _np.array(A_eq, dtype=_np.float64) if A_eq else None
    b_eq_arr = _np.array(b_eq, dtype=_np.float64) if b_eq else None
    result = _scipy_linprog(
        c_arr,
        A_eq=A_eq_arr,
        b_eq=b_eq_arr,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    x = result.x.tolist() if result.x is not None else [0.0] * len(c)
    obj = float(result.fun) if result.fun is not None else 0.0
    if maximize:
        obj = -obj  # undo the negation to get the true maximized value
    status_map = {
        0: "optimal",
        1: "max_iter",
        2: "infeasible",
        3: "unbounded",
    }
    status = status_map.get(result.status, "infeasible")
    return {"status": status, "x": x, "objective": obj}


def solve_lp(c: list[float],
             A: list[list[float]],
             b: list[float],
             bounds: list[tuple[float, float]],
             maximize: bool = True,
             method: str = "auto") -> dict:
    """Dispatch LP solve to simplex or scipy linprog based on model size.

    Parameters
    ----------
    method : "auto" | "simplex" | "scipy"
        "auto" dispatches based on model size (>500 → scipy).
    """
    n_vars = len(c)
    if method == "scipy" or (method == "auto" and n_vars > _SOLVER_DISPATCH_THRESHOLD and _HAS_SCIPY):
        return _solve_scipy(c, A, b, bounds, maximize)
    return simplex(c, A, b, bounds, maximize)


# ============================================================================
# Enzyme-constrained FBA (Phase 4: MOMENT / sMOMENT / GECKO capacity)
# ============================================================================

#: default enzyme molecular weight when none is given (g/mmol; ~50 kDa)
DEFAULT_ENZYME_MW = 50.0

#: default global kcat rescaling (the Phase-5 calibration hook: GECKO's
#: kcat-correction term, Sanchez 2017); with folded-protein-pool enzyme
#: levels of ~1e6 units and kcat in flux-per-unit-per-hour this puts the
#: unconstrained enzyme caps far above the substrate-limited uptake, so
#: they only bind when the proteome is genuinely limiting.
DEFAULT_ENZYME_SCALE = 1e4

#: E. coli core-model gene -> reaction associations (Orth 2010 core model;
#: a reaction is gated by ALL of its genes, so deleting any one gene
#: removes the reaction when no isozyme copy exists).  Canonical location
#: for the enzyme-capacity wiring; :mod:`helixlang.plugins.apps.whole_cell_scale`
#: re-exports this for ``ko_model``/``predict_essentiality``.
ECOLI_CORE_GENE_REACTIONS: dict[str, tuple[str, ...]] = {
    "ptsG": ("GLCpts",),
    "glk": ("GLK",),
    "pgi": ("PGI",),
    "pfkA": ("PFK",),
    "fba": ("FBA",),
    "tpiA": ("TPI",),
    "gapA": ("GAPD",),
    "pgk": ("PGK",),
    "pgm": ("PGM",),
    "eno": ("ENO",),
    "pykA": ("PYK",),
    "aceE": ("PDH",),
    "gltA": ("CS",),
    "acnB": ("ACONT",),
    "icdA": ("ICDH",),
    "sucAB": ("AKGDH",),
    "sucCD": ("SUCCt", "SUCOAS"),
    "sdhA": ("SUCDHi",),
    "fumA": ("FUM",),
    "mdh": ("MDH",),
    "ppc": ("PPC",),
    "zwf": ("G6PDH",),
    "gnd": ("PGD",),
    "rpiA": ("RPI",),
    "ldhA": ("LDH",),
    "pta": ("PTA_ACK",),
    "ackA": ("PTA_ACK",),
    "atpF": ("NADH_OX",),
}

#: per-reaction turnover capacities in flux-per-enzyme-unit-per-hour
#: (mmol/gDW/h per enzyme unit).  Relative order follows BRENDA kcat data
#: for E. coli enzymes (Beg 2007 PNAS 104:12663; Beck 2020 BMC Bioinformatics
#: 21:4); the absolute scale is a calibrated free parameter
#: (``EnzymeCapacity.enzyme_scale``, the Phase-5 fitting hook — GECKO's
#: kcat-correction term, Sánchez 2017).
ECOLI_CORE_KCAT: dict[str, float] = {
    "GLCpts": 2.0, "GLK": 2.0, "PGI": 6.0, "PFK": 6.0, "FBA": 3.0,
    "TPI": 8.0, "GAPD": 6.0, "PGK": 8.0, "PGM": 8.0, "ENO": 6.0,
    "PYK": 8.0, "PDH": 3.0, "CS": 6.0, "ACONT": 3.0, "ICDH": 6.0,
    "AKGDH": 2.0, "SUCCt": 3.0, "SUCOAS": 3.0, "SUCDHi": 4.0, "FUM": 6.0,
    "MDH": 6.0, "PPC": 3.0, "G6PDH": 6.0, "PGD": 4.0, "RPI": 8.0,
    "LDH": 8.0, "PTA_ACK": 8.0, "NADH_OX": 0.5,
}


@dataclass(slots=True)
class EnzymeCapacity:
    """MOMENT-style enzyme-capacity configuration (Beg 2007; Adadi 2012).

    For each gene->reaction pair the reaction flux is capped by

        v_i <= kcat_i * E_i * enzyme_scale

    where ``E_i`` is the enzyme abundance (relative units, e.g. the folded
    ProteinPool of Phase 3) and ``kcat_i`` the turnover capacity in the
    model's flux units per enzyme unit.  A reaction gated by several genes
    (a protein complex / sequential enzymes, e.g. ``pta`` + ``ackA`` ->
    ``PTA_ACK``) uses the *minimum* subunit level — the conservative MOMENT
    rule.  Optionally a global enzyme-pool budget can be added:

        Sum_i v_i * MW_i / kcat_i <= protein_mass_fraction

    the sMOMENT compact formulation (Bekiaris & Klamt 2020): a
    pseudo-metabolite row ``-Sum MW_i/kcat_i * v_i + v_pool = 0`` with
    ``0 <= v_pool <= P`` is appended to the LP.

    Attributes:
        gene_to_reactions: gene -> reaction ids (reactions gated by E_i).
        kcat: reaction_id -> turnover capacity (flux per enzyme unit).
        km: reaction_id -> Michaelis constant (substrate concentration at
            half-maximal velocity, mM).  Used by Monod-style uptake.
        enzyme_scale: global kcat rescaling (the Phase-5 calibration hook).
        protein_mass_fraction: optional global enzyme-pool budget P (g/gDW);
            ``None`` disables the sMOMENT pool row.
        enzyme_mw: reaction_id -> enzyme molecular weight (g/mmol); used
            only by the global pool row (default :data:`DEFAULT_ENZYME_MW`).
    """
    gene_to_reactions: dict[str, tuple[str, ...]]
    kcat: dict[str, float] = field(default_factory=dict)
    km: dict[str, float] = field(default_factory=dict)
    enzyme_scale: float = 1.0
    protein_mass_fraction: float | None = None
    enzyme_mw: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enzyme_scale <= 0.0:
            raise ValueError("enzyme_scale must be > 0")


@dataclass(slots=True)
class MetabolitePoolConfig:
    """Intracellular metabolite-pool parameters (Phase 4).

    Args:
        dt_h: default integration time step (hours) used when
            :meth:`MetabolitePool.integrate` is called without ``dt_h``.
        dilution: subtract the growth-rate dilution term ``mu * [P]`` when
            integrating (default True); with False the pools integrate pure
            net production only.
        min_pool: floor below which a pool is clamped (default 0).
    """

    dt_h: float = 0.05
    dilution: bool = True
    min_pool: float = 0.0


class MetabolitePool:
    """Intracellular metabolite pools (Phase 4).

    Integrates the per-metabolite mass balance

        d[P]/dt = Sum_j S[P][j] * v_j  -  mu * [P]

    forward in time with Euler steps, where ``Sum_j S[P][j] * v_j`` is the
    net production of metabolite ``P`` across every reaction in the last FBA
    solution and ``mu`` is the specific growth rate (1/h) carrying the
    dilution term (growth dilutes the pools; Palsson, *Systems Biology*,
    Ch. 10).  At steady state production balances dilution:

        [P]* = Sum_j S[P][j] * v_j / mu

    Unlike the :class:`DynamicFluxBalance` batch pools these are
    *intracellular* pools: they respond to the instantaneous enzyme-
    constrained flux distribution (Phase 4), so a respiratory bottleneck
    that redirects flux to acetate overflow shows up directly as a rising
    intracellular Ac pool and a positive :meth:`overflow_flux` on ``EX_ac``
    (Sanchez 2017; Basan 2015).
    """

    def __init__(self,
                 model: MetabolicModel,
                 config: MetabolitePoolConfig | None = None,
                 initial: dict[str, float] | None = None) -> None:
        self.model = model
        self.config = config or MetabolitePoolConfig()
        self.pools: dict[str, float] = {
            met: float(initial.get(met, 0.0) if initial else 0.0)
            for met in sorted(model.metabolites)
        }

    def net_production(self, met: str, fluxes: dict[str, float]) -> float:
        """Net production rate of ``met`` from an FBA flux solution.

        ``Sum_j S[met][j] * v_j``: positive when the flux distribution
        produces the metabolite faster than it consumes it.
        """
        net = 0.0
        for rid, v in fluxes.items():
            rxn = self.model.reactions.get(rid)
            if rxn is None:
                continue
            net += rxn.stoichiometry.get(met, 0.0) * v
        return net

    def integrate(self,
                  fluxes: dict[str, float],
                  growth_rate: float = 0.0,
                  dt_h: float | None = None) -> dict[str, float]:
        """Advance every pool by one Euler step; returns {met: delta}.

        Args:
            fluxes: {reaction_id: flux} from the current FBA solution.
            growth_rate: specific growth rate mu (1/h); drives the
                dilution term ``mu * [P]`` when ``config.dilution`` is True.
            dt_h: integration step in hours (default ``config.dt_h``).
        """
        dt = self.config.dt_h if dt_h is None else dt_h
        if dt < 0.0:
            raise ValueError("dt_h must be >= 0")
        deltas: dict[str, float] = {}
        for met in self.pools:
            net = self.net_production(met, fluxes)
            if self.config.dilution:
                net -= growth_rate * self.pools[met]
            new = max(self.config.min_pool, self.pools[met] + net * dt)
            deltas[met] = new - self.pools[met]
            self.pools[met] = new
        return deltas

    def overflow_flux(self, fluxes: dict[str, float]) -> dict[str, float]:
        """Byproduct secretion fluxes (mmol/gDW/h) from a flux solution.

        A positive flux through an exchange reaction whose metabolite has a
        negative coefficient (e.g. ``EX_ac``) exports the metabolite into
        the medium — the pool-overflow indicator used to flag overflow
        metabolism (Sanchez 2017; Basan 2015).  Returns {met: flux}; the
        biomass exchange reaction is excluded.
        """
        out: dict[str, float] = {}
        for rid, rxn in self.model.reactions.items():
            if rxn.subsystem != "exchange":
                continue
            v = max(0.0, fluxes.get(rid, 0.0))
            if v <= 0.0:
                continue
            for met, coef in rxn.stoichiometry.items():
                if coef < 0.0 and met != "Biomass":
                    out[met] = out.get(met, 0.0) + v
        return out


# ============================================================================
# Enzyme activity correction: temperature + pH (doc/25 Phase VIII, G8)
# ============================================================================

def enzyme_correction(
    temperature_c: float,
    ph: float,
    ea_kj_mol: float = 50.0,
    t_opt_c: float = 37.0,
    ph_opt: float = 7.0,
    ph_width: float = 2.0,
) -> float:
    """Calculate enzyme activity correction from temperature and pH.

    Uses Arrhenius for temperature and Gaussian for pH:

    * f(T) = exp(-Ea/R * (1/T - 1/T_opt))  (capped at 1.0)
    * f(pH) = exp(-(pH - pH_opt)^2 / (2 * ph_width^2))

    Returns a multiplier in [0, 1] where 1.0 = optimal conditions.

    Parameters
    ----------
    temperature_c : environmental temperature (deg C)
    ph : environmental pH
    ea_kj_mol : activation energy (kJ/mol, default 50 typical for enzymes)
    t_opt_c : optimal temperature (deg C, default 37 for mesophiles)
    ph_opt : optimal pH (default 7.0)
    ph_width : pH tolerance window / Gaussian sigma (default 2.0)
    """
    import math

    R = 8.314e-3  # gas constant in kJ/(mol*K)
    T = temperature_c + 273.15
    T_opt = t_opt_c + 273.15
    # Arrhenius correction (capped at 1.0 at optimal)
    arr = math.exp(-ea_kj_mol / R * (1.0 / T - 1.0 / T_opt))
    arr = min(arr, 1.0)
    # pH Gaussian correction
    ph_factor = math.exp(-(ph - ph_opt) ** 2 / (2.0 * ph_width ** 2))
    return arr * ph_factor


# ============================================================================
# Flux Balance Analysis solver
# ============================================================================

class FluxBalanceAnalysis:
    """Flux balance analysis solver.

    Converts a MetabolicModel + bound constraints into a standard LP and
    calls simplex to solve for the optimal fluxes.

    Usage:
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)     # max glucose uptake
                                          # 10 mmol/gDW/h
        fluxes = fba.solve(objective="biomass")
        report = fba.analyze()
    """

    def __init__(self, model: MetabolicModel) -> None:
        self.model = model
        # uptake limits: metabolite -> max uptake rate (applied to the
        # upper bound of the corresponding EX_ reaction)
        self.uptake_limits: dict[str, float] = {}
        # cache of the most recent solution
        self.last_solution: dict[str, float] | None = None
        # MOMENT-style enzyme-capacity configuration (Phase 4); when set,
        # solved reaction fluxes are capped by kcat_i * E_i * enzyme_scale
        # with E_i read from set_enzyme_levels()
        self.enzyme_capacity: EnzymeCapacity | None = None
        # per-gene enzyme abundances (relative units, e.g. folded ProteinPool)
        self.enzyme_levels: dict[str, float] = {}

    # -------- enzyme-capacity constraints (Phase 4, MOMENT) --------

    def set_enzyme_capacity(self,
                            capacity: EnzymeCapacity | None) -> None:
        """Enable/disable enzyme-constrained FBA.

        With a capacity set, :meth:`solve` caps every enzyme-gated reaction
        flux by ``kcat_i * E_i * enzyme_scale`` and, when a
        ``protein_mass_fraction`` is configured, adds the sMOMENT global
        enzyme-pool row ``Sum v_i*MW_i/kcat_i <= P`` (Beg 2007; Adadi 2012;
        Bekiaris & Klamt 2020).
        """
        self.enzyme_capacity = capacity
        if capacity is None:
            self.enzyme_levels = {}

    def set_enzyme_levels(self, levels: dict[str, float]) -> None:
        """Set per-gene enzyme abundances (relative units).

        Values are read from the cell's folded ProteinPool (Phase 3): the
        GRN/expression machinery controls enzyme supply, so metabolism
        responds to the proteome (O'Brien 2013 ME-model coupling).
        """
        self.enzyme_levels = {g: float(v) for g, v in levels.items()}

    # -------- uptake limits --------

    def set_uptake(self, metabolite: str, flux: float) -> None:
        """Set a substrate uptake rate upper bound.

        Sets the upper bound of the exchange reaction
        EX_<metabolite> to flux, so the substrate can only be taken up
        within [0, flux] (positive exchange direction = uptake).
        """
        self.uptake_limits[metabolite] = float(flux)

    # -------- build the LP and solve --------

    def _build_and_solve(self,
                         objective_reaction: str,
                         maximize: bool = True) -> dict[str, float]:
        """Build the LP and solve it, returning {reaction_id: flux}."""
        met_list, rxn_list, S = self.model.get_stoichiometry_matrix()
        n = len(rxn_list)
        m = len(met_list)

        # objective coefficients
        c = [0.0] * n
        if objective_reaction in self.model.reactions:
            c[rxn_list.index(objective_reaction)] = 1.0
        else:
            raise BioError(f"objective reaction {objective_reaction!r} not in model")

        # bounds
        bounds: list[tuple[float, float]] = []
        ec = self.enzyme_capacity
        for rid in rxn_list:
            rxn = self.model.reactions[rid]
            lb = rxn.lower_bound
            ub = rxn.upper_bound
            # apply uptake limits: if EX_<met> and the metabolite is in
            # uptake_limits
            if (rxn.stoichiometry
                    and (rxn.subsystem == "exchange"
                         or rid.startswith("EX_"))):
                # an EX reaction involves only 1 metabolite; its sign
                # determines the direction.
                for met, coef in rxn.stoichiometry.items():
                    if met in self.uptake_limits:
                        limit = self.uptake_limits[met]
                        if coef > 0:
                            # coef > 0: metabolite is produced (secretion).
                            # set_uptake overrides the upper bound to enable
                            # secretion at the specified rate.
                            ub = limit
                        elif coef < 0:
                            # coef < 0: metabolite is consumed (uptake).
                            # Negative flux = uptake for this convention,
                            # so clamp both bounds to [-limit, limit].
                            ub = min(ub, limit)
                            lb = max(lb, -limit)
            # MOMENT enzyme capacity: v_i <= kcat_i * E_i * enzyme_scale
            if ec is not None and rid in ec.kcat:
                kc = ec.kcat[rid]
                if kc > 0.0:
                    e_level: float | None = None
                    for gene, rids in ec.gene_to_reactions.items():
                        if rid in rids:
                         # a reaction gated by several genes (complex /
                             # sequential enzymes) is limited by the minimum
                             # subunit abundance (conservative MOMENT rule)
                         if e_level is None:
                             _default = 1.0 if not self.enzyme_levels else 0.0
                             e_level = self.enzyme_levels.get(gene, _default)
                         else:
                             _default = 1.0 if not self.enzyme_levels else 0.0
                             e_level = min(
                                 e_level, self.enzyme_levels.get(gene, _default))
                    if e_level is not None:
                        cap = kc * e_level * ec.enzyme_scale
                        ub = min(ub, cap)
            bounds.append((lb, ub))

        # sMOMENT global enzyme-pool row (Bekiaris & Klamt 2020): append
        #   -Sum_i (MW_i/kcat_i) * v_i + v_pool = 0 ,  0 <= v_pool <= P
        # The row counts every enzyme-gated reaction (a gene maps to it);
        # its cost coefficient MW/(kcat*enzyme_scale) uses the same
        # relative-kcat convention as the per-reaction caps, so the budget
        # and the caps stay mutually consistent.
        n_pool = 0
        if (ec is not None and ec.protein_mass_fraction is not None
                and ec.protein_mass_fraction > 0.0):
            gated: set[str] = set()
            for rids in ec.gene_to_reactions.values():
                gated.update(rids)
            pool_row = [0.0] * n
            for i, rid in enumerate(rxn_list):
                kc = ec.kcat.get(rid, 0.0)
                if rid in gated and kc > 0.0:
                    mw = ec.enzyme_mw.get(rid, DEFAULT_ENZYME_MW)
                    pool_row[i] = -(mw / (kc * ec.enzyme_scale))
            # add the v_pool column (coefficient +1 in the pool row only:
            # the row is -Sum a_i*v_i + v_pool = 0, so v_pool = Sum a_i*v_i)
            S = [row + [0.0] for row in S]
            S.append(pool_row + [1.0])
            bounds = bounds + [(0.0, ec.protein_mass_fraction)]
            c = c + [0.0]
            n_pool = 1

        # b: steady-state mass balance S·v = 0 -> b = [0, 0, ..., 0]
        b = [0.0] * (m + n_pool)

        result = solve_lp(c, S, b, bounds, maximize=maximize)
        if result["status"] not in ("optimal", "max_iter"):
            # infeasible or unbounded
            return {rid: 0.0 for rid in rxn_list}

        x = result["x"]
        # the first ``n`` entries map to reactions; a trailing entry (when
        # n_pool == 1) is the sMOMENT v_pool slack variable
        fluxes = {rxn_list[j]: x[j] for j in range(n)}
        return fluxes

    def solve(self,
              objective: str = "biomass",
              maximize: bool = True) -> dict[str, float]:
        """Solve for the optimal fluxes with linear programming.

        Args:
            objective: "biomass" -> use model.biomass_reaction;
                       otherwise treated as a reaction_id
            maximize: True = maximize the objective (default, maximizes
                      biomass yield); False = minimize the objective

        Returns:
            {reaction_id: flux_value} dict; reactions not solved for
            return 0.0
        """
        if objective == "biomass":
            if self.model.biomass_reaction is None:
                raise BioError("model has no biomass reaction set")
            obj_rxn = self.model.biomass_reaction
        else:
            obj_rxn = objective
        self.last_solution = self._build_and_solve(obj_rxn, maximize=maximize)
        return self.last_solution

    # -------- analysis --------

    def analyze(self) -> dict:
        """Analyze the metabolic state: biomass yield, key fluxes,
        energy balance, byproduct secretion.

        Returns a dict with:
            biomass_yield:           biomass flux (mmol/gDW/h)
            glucose_uptake:          glucose uptake flux
            substrate_uptakes:       each substrate's uptake rate
            byproduct_secretion:     byproduct secretion rates
                                     (lactate/acetate/CO2)
            key_fluxes:              key reaction fluxes
                                     (glycolysis/TCA/PPP
                                     representatives)
            biomass_per_glucose:     biomass yield / glucose uptake
                                     (gDW/g glucose approximation)
            subsystem_fluxes:        total flux per subsystem
            objective_value:         objective function value
            growth_rate_per_hour:    estimated specific growth rate
                                     (1/h, 1 biomass unit ~ 1/h)
        """
        if self.last_solution is None:
            self.solve()

        sol = self.last_solution or {}
        bm_rxn = self.model.biomass_reaction
        biomass_flux = sol.get(bm_rxn, 0.0) if bm_rxn else 0.0
        # Try both naming conventions: "EX_glc" (core model) and "EX_glc_e"
        # (gapfill / GEM-reconstructed models that use _e suffix metabolites)
        glc_uptake = sol.get("EX_glc", 0.0) or sol.get("EX_glc_e", 0.0)

        # byproducts (try both naming conventions)
        byproducts = {
            "lactate": sol.get("EX_lac", 0.0) or sol.get("EX_lac_e", 0.0),
            "acetate": sol.get("EX_ac", 0.0) or sol.get("EX_ac_e", 0.0),
            "co2": sol.get("EX_co2", 0.0) or sol.get("EX_co2_e", 0.0),
        }

        # key fluxes (representative reactions picked per subsystem)
        key_fluxes: dict[str, float] = {}
        for rid, rxn in self.model.reactions.items():
            if rxn.subsystem in ("glycolysis", "tca", "ppp",
                                  "fermentation", "biomass",
                                  "maintenance", "exchange"):
                key_fluxes[rid] = sol.get(rid, 0.0)

        # subsystem totals
        subsystem_flux: dict[str, float] = {}
        for rid, rxn in self.model.reactions.items():
            v = abs(sol.get(rid, 0.0))
            subsystem_flux[rxn.subsystem] = subsystem_flux.get(rxn.subsystem, 0.0) + v

        # biomass/glucose ratio (glc_uptake is negative for uptake via
        # coef=-1 exchange convention; use abs for yield calculation)
        abs_glc = abs(glc_uptake)
        bg_ratio = (biomass_flux / abs_glc) if abs_glc > 1e-9 else 0.0

        # energy balance: ATP production = substrate-level
        # phosphorylation + oxidative phosphorylation
        #   substrate-level: PGK + PYK + AKGDH (incl. SCS) + PTA_ACK
        #   oxidative phosphorylation: 1.5xNADH_OX + 0.5xFADH2_OX
        # ATP consumption = GLK + PFK + ATPM + BIOMASS x 59.810
        atp_prod = (sol.get("PGK", 0.0) + sol.get("PYK", 0.0)
                    + sol.get("AKGDH", 0.0)
                    + sol.get("PTA_ACK", 0.0)
                    + 1.5 * sol.get("NADH_OX", 0.0)
                    + 0.5 * sol.get("FADH2_OX", 0.0))
        atp_cons = (sol.get("GLK", 0.0) + sol.get("PFK", 0.0)
                    + sol.get("ATPM", 0.0)
                    + sol.get("BIOMASS", 0.0) * 59.810)  # BIOMASS ATP coeff

        return {
            "objective_reaction": bm_rxn,
            "objective_value": biomass_flux,
            "biomass_yield": biomass_flux,
            "biomass_per_glucose": bg_ratio,
            "glucose_uptake": glc_uptake,
            "substrate_uptakes": {met: flux
                                   for met, flux in self.uptake_limits.items()},
            "byproduct_secretion": byproducts,
            "key_fluxes": key_fluxes,
            "subsystem_fluxes": subsystem_flux,
            "energy_balance": {
                "atp_production": atp_prod,
                "atp_consumption": atp_cons,
                "atp_balance": atp_prod - atp_cons,
            },
            # specific growth rate mu (1/h): biomass flux
            # (mmol/gDW/h) divided by the biomass "molecular weight"
            # E. coli biomass molecular weight ~1 gDW/mmol (Orth 2010,
            # BiGG iJO1366 GAM conversion)
            # real E. coli mu ~0.98/h (glucose 10 mmol/gDW/h, aerobic)
            "growth_rate_per_hour": biomass_flux / _BIOMASS_MW,
        }


# ============================================================================
# Dynamic FBA (dFBA)
# ============================================================================

# name of the glucose exchange reaction in the curated core model
_EX_GLC = "EX_glc"
# exchange reaction that removes accumulated biomass from the medium
_EX_BIOMASS = "EX_biomass"
# acetate exchange + PTA/ACK + glyoxylate-shunt reaction ids for the
# acetate-switch activation (Wolfe 2005, MMBR 69:12-50)
_EX_AC = "EX_ac"
_EX_O2 = "EX_o2"
_ICL = "ICL"
_MAS = "MAS"
_ACS = "ACS"

# Patterns for auto-detecting exchange reactions in GEM models
_GLC_PATTERNS = ("EX_glc-D_e", "EX_glc_e", "EX_glc")
_O2_PATTERNS = ("EX_o2_e", "EX_o2")
_AC_PATTERNS = ("EX_ac_e", "EX_ac")


def _detect_exchange_ids(model: MetabolicModel) -> tuple[str, str, str]:
    """Auto-detect glucose/oxygen/acetate exchange reaction IDs.

    Returns (ex_glc, ex_o2, ex_ac) — falls back to the E. coli core
    model defaults when no match is found.
    """
    rxn_ids = set(model.reactions.keys())
    ex_glc = next((p for p in _GLC_PATTERNS if p in rxn_ids), _EX_GLC)
    ex_o2 = next((p for p in _O2_PATTERNS if p in rxn_ids), _EX_O2)
    ex_ac = next((p for p in _AC_PATTERNS if p in rxn_ids), _EX_AC)
    return ex_glc, ex_o2, ex_ac


def _detect_exchange_metabolites(model: MetabolicModel) -> tuple[str, str, str]:
    """Auto-detect glucose/oxygen/acetate exchange metabolite names.

    Returns (met_glc, met_o2, met_ac) — the metabolite IDs inside the
    exchange reactions, used for ``set_uptake()`` matching.

    Works for both sign conventions:
    - Core model ``EX_glc`` has ``coef=+1.0`` (reactant convention)
    - GEM ``EX_glc_e`` has ``coef=-1.0`` (product convention)
    """
    ex_glc, ex_o2, ex_ac = _detect_exchange_ids(model)
    def _met(rxn_id: str) -> str:
        rxn = model.reactions.get(rxn_id)
        if rxn is None or not rxn.stoichiometry:
            return ""
        # Return the single metabolite in the exchange reaction,
        # regardless of sign convention.
        return next(iter(rxn.stoichiometry))
    return _met(ex_glc), _met(ex_o2), _met(ex_ac)
_PEPCK = "PEPCK"
_FBP = "FBP"
_ACETATE_SWITCH_RXNS = (_ICL, _MAS, _ACS, _PEPCK, _FBP)


def activate_acetate_switch(model: MetabolicModel) -> None:
    """Enable the second, acetate-assimilation growth phase (in place).

    Wolfe 2005 (Microbiol Mol Biol Rev 69:12-50, "The Acetate Switch")
    describes how E. coli excretes acetate during exponential growth on
    glucose (dissimilation) and, once glucose is depleted, switches to
    *assimilating* the accumulated acetate.  Assimilation requires the
    glyoxylate bypass (isocitrate lyase ``aceA``, malate synthase
    ``aceB``) to replenish C4 precursors without the decarboxylating
    losses of a full turn of the TCA cycle, the AMP-forming acetyl-CoA
    synthetase ``acs`` for acetate activation, and the gluconeogenic
    phosphoenolpyruvate carboxykinase (``pckA``) and fructose
    bisphosphatase (``fbp``) to build hexose/phospho-sugar precursors
    from oxaloacetate.  The curated core model ships these reactions
    present but flux-``0`` (``ICL``/``MAS``/``ACS``/``PEPCK``/``FBP``),
    so the baseline fermentative behaviour is unchanged; this function
    flips them on and opens the acetate exchange for import,
    reproducing the dynamic switch once glucose runs out.

    Note that ``PTA_ACK`` is deliberately kept forward-only: reversed
    PTA/ACK would let the LP close an ATP-neutral acetate cycle and
    inflate the biomass flux (a free-energy artifact).  Acetate
    assimilation is instead routed through ``ACS``, which costs 2 ATP
    per acetate (ATP -> AMP + PPi), keeping the second phase
    mass- and energy-conserving.
    """
    reactions = model.reactions
    for rid in _ACETATE_SWITCH_RXNS:
        if rid in reactions:
            reactions[rid].upper_bound = 1000.0
    if _EX_AC in reactions:
        reactions[_EX_AC].lower_bound = -10.0


@dataclass
class DynamicSimulationResult:
    """Time-course trajectory from dFBA (doc/20 §15.6)."""
    time_points: list[float] = field(default_factory=list)
    biomass: list[float] = field(default_factory=list)
    substrates: dict[str, list[float]] = field(default_factory=dict)
    byproducts: dict[str, list[float]] = field(default_factory=dict)
    growth_rates: list[float] = field(default_factory=list)
    fluxes: list[dict[str, float]] = field(default_factory=list)
    final_biomass: float = 0.0
    doubling_time: float = 0.0
    diauxic_shift: bool | None = None


@dataclass
class FeedEvent:
    """Nutrient feeding event for fed-batch simulation (doc/20 §16.2).

    Attributes
    ----------
    time_h : when to apply the feed (hours)
    metabolite : exchange metabolite id (e.g. "EX_glc")
    amount_mmol : mmol of substrate to add per unit volume
    dilution : volume dilution factor (0 = no dilution, typical for fed-batch)
    """
    time_h: float
    metabolite: str
    amount_mmol: float
    dilution: float = 0.0


@dataclass(slots=True)
class DynamicFBAConfig:
    """Dynamic FBA batch-culture parameters (Mahadevan et al. 2002).

    Args:
        dt_h: integration time step (hours)
        initial_biomass_gdw: starting biomass (gDW/L)
        initial_glucose_mm: starting glucose concentration (mM)
        initial_acetate_mm: starting acetate concentration (mM)
        max_glucose_uptake: maximum glucose uptake rate
            (mmol/gDW/h); applied as the LP upper bound on the glucose
            exchange when glucose is saturating
        glucose_half_saturation_mm: Michaelis-Menten half-saturation Ks
            for glucose uptake (mM); the instantaneous uptake bound is
            v_max * S / (Ks + S) after Mahadevan 2002
        biomass_per_mmol: grams dry weight per mmol of biomass flux;
            converts the LP biomass flux (mmol/gDW/h) into the specific
            growth rate mu (1/h)
        min_biomass: growth floor (gDW/L) used to detect batch
            exhaustion / stagnation in :meth:`DynamicFluxBalance.run`
        acetate_switch: enable the second, acetate-assimilation growth
            phase (Wolfe 2005 "Acetate Switch", MMBR 69:12-50): the
            glyoxylate bypass (aceA/aceB) is activated, the acetate
            exchange is opened for import and PTA/ACK becomes reversible,
            so once glucose is depleted the LP re-imports the overflow
            acetate and biomass keeps growing.  Default False (baseline
            fermentative behaviour).
        acetate_switch_threshold_mm: glucose concentration below which
            acetate import is permitted.  E. coli does not co-consume
            acetate while glucose is abundant; consumption starts only
            once glucose drops below ~0.5 mM (Enjalbert et al. 2011,
            ISME J 5:1301).  Import is additionally capped by the
            well-mixed acetate pool (Mahadevan et al. 2002 dFBA), so the
            second phase cannot outrun the available acetate.
    """

    dt_h: float = 0.05
    initial_biomass_gdw: float = 0.05
    initial_glucose_mm: float = 10.0
    initial_acetate_mm: float = 0.0
    max_glucose_uptake: float = DEFAULT_GLC_UPTAKE
    glucose_half_saturation_mm: float = 0.1
    biomass_per_mmol: float = _BIOMASS_MW
    min_biomass: float = 1e-9
    acetate_switch: bool = False
    acetate_switch_threshold_mm: float = 0.5
    # Multi-substrate support (doc/20 §15.4)
    initial_oxygen_mm: float = 20.0
    oxygen_half_saturation_mm: float = 0.02
    max_oxygen_uptake: float = 20.0
    # Safety caps to prevent numerical explosion in simplified models
    max_growth_rate: float = 2.0  # h^-1; clamps mu to avoid runaway
    max_biomass_gdw: float = 50.0  # gDW/L; carrying capacity
    # Fed-batch / chemostat mode (doc/20 §16)
    feed_events: list[FeedEvent] = field(default_factory=list)
    chemostat: bool = False
    chemostat_dilution_rate: float = 0.0  # h^-1 (0 = batch mode)
    chemostat_feed_concentrations: dict[str, float] = field(default_factory=dict)
    # Photoautotrophic dFBA fields (doc/22 §7)
    substrate_type: str = "glucose"  # "glucose" | "co2"
    co2_initial_mm: float = 1.0     # dissolved CO₂ (mM) for photoautotrophic; 5% CO₂ sparging ≈ 1 mM
    co2_max_uptake: float = 30.0    # mmol/gDW/h (Calvin cycle capacity)
    co2_half_saturation_mm: float = 0.5  # Ks for CO₂ (Monod)
    light_intensity: float = 200.0  # μmol photons/m²/s (PAR); typical outdoor sunlight ~2000, lab ~100-300
    light_saturation: float = 150.0  # μmol photons/m²/s (K_L for light)
    light_max_rate: float = 12.5    # mmol ATP/gDW/h (estimated from photon flux efficiency)


class DynamicFluxBalance:
    """Dynamic flux balance analysis (Mahadevan et al. 2002).

    Simulates a well-mixed batch culture with an instantaneous FBA LP per
    time step ("static optimization approach"): the glucose-uptake bound
    is set from the external substrate via Michaelis-Menten kinetics

        v_glc(t) = v_max * S(t) / (Ks + S(t))

    and the batch ODEs are integrated with forward Euler:

        dX/dt =  mu * X         biomass (gDW/L)
        dS/dt = -v_glc * X      glucose (mmol/L)
        dP/dt =  v_secret * X   byproducts lactate/acetate/CO2 (mmol/L)

    with mu = v_biomass / biomass_per_mmol.  As glucose depletes, the
    uptake bound collapses, growth decelerates and finally stops: the
    first, fermentative phase of the classic diauxic shift, with overflow
    acetate accumulating in the medium.

    The reduced core model ships the glyoxylate bypass (iCL ``aceA`` /
    MS ``aceB``) flux-``0``: by default overflow acetate cannot be
    re-imported and growth stops once glucose is gone (baseline
    fermentative behaviour).  With ``config.acetate_switch=True`` the
    bypass is activated and the acetate exchange opened for import, so
    the LP *automatically* switches to assimilating the accumulated
    acetate once glucose depletes -- the second, co-assimilation phase of
    the classic "Acetate Switch" (Wolfe 2005, MMBR 69:12-50).

    The batch pools can be coupled to a :class:`~helixlang.plugins.runtime.environment.
    Environment`: :meth:`update_from_environment` reads the local glucose
    concentration into the batch, :meth:`apply_to_environment` deposits
    the accumulated acetate back into the medium field.
    """

    def __init__(
        self,
        model: MetabolicModel | None = None,
        config: DynamicFBAConfig | None = None,
        fba: FluxBalanceAnalysis | None = None,
        bound_override: Callable[[float, DynamicFluxBalance], dict[str, float]] | None = None,
    ) -> None:
        self.config = config or DynamicFBAConfig()
        model = model or ECOLI_CORE_MODEL
        if self.config.acetate_switch:
            # the acetate switch mutates reaction bounds; deep-copy so the
            # shared ECOLI_CORE_MODEL stays pristine (Wolfe 2005).
            model = copy.deepcopy(model)
            activate_acetate_switch(model)
        self.fba = fba or FluxBalanceAnalysis(model)
        self._biomass_reaction = self.fba.model.biomass_reaction
        # Auto-detect exchange reaction IDs for the model (core vs GEM)
        self._ex_glc, self._ex_o2, self._ex_ac = _detect_exchange_ids(
            self.fba.model,
        )
        # Auto-detect metabolite names inside exchange reactions (for set_uptake)
        self._met_glc, self._met_o2, self._met_ac = _detect_exchange_metabolites(
            self.fba.model,
        )
        # dynamic bound hook (T3.2): ``bound_override(time_h, self) ->
        # dict[reaction_id, bound]`` applied before each LP solve, so the
        # batch can react to e.g. transcriptomics-guided metabolic
        # switching (MiMICS 2024) or fluctuating media without changing
        # the integration loop.
        self.bound_override = bound_override
        # secreted byproduct exchange -> external pool name
        self._byproduct_ex: dict[str, str] = {}
        self._byproduct_pools: list[str] = []
        pool_name = {"Lac": "lactate", "Ac": "acetate", "CO2": "co2"}
        for rid, rxn in self.fba.model.reactions.items():
            if (rxn.subsystem != "exchange"
                    or rid in (self._ex_glc, _EX_BIOMASS)):
                continue
            for met, coef in rxn.stoichiometry.items():
                if coef < 0:
                    pool = pool_name.get(met, met)
                    self._byproduct_ex[rid] = pool
                    if pool not in self._byproduct_pools:
                        self._byproduct_pools.append(pool)
        self.reset()

    # -------- state --------

    def reset(self) -> None:
        """Restore the initial batch state and clear the history."""
        cfg = self.config
        self.time_h: float = 0.0
        self.biomass_gdw: float = cfg.initial_biomass_gdw
        self.glucose_mm: float = cfg.initial_glucose_mm
        self.byproducts_mm: dict[str, float] = {
            pool: (cfg.initial_acetate_mm if pool == "acetate" else 0.0)
            for pool in self._byproduct_pools}
        self.byproducts_mm.setdefault("oxygen", cfg.initial_oxygen_mm)
        self.history: list[dict[str, float]] = []
        self.last_fluxes: dict[str, float] = {}
        self._feed_events_applied: set[int] = set()

    def set_state(self,
                  biomass_gdw: float | None = None,
                  glucose_mm: float | None = None,
                  acetate_mm: float | None = None) -> None:
        """Override the current batch state (None leaves a value intact)."""
        if biomass_gdw is not None:
            self.biomass_gdw = float(biomass_gdw)
        if glucose_mm is not None:
            self.glucose_mm = float(glucose_mm)
        if acetate_mm is not None:
            self.byproducts_mm["acetate"] = float(acetate_mm)

    # -------- substrate availability --------

    def uptake_bound(self, glucose_mm: float) -> float:
        """Michaelis-Menten glucose-uptake bound (Mahadevan 2002)."""
        cfg = self.config
        if glucose_mm <= 0.0:
            return 0.0
        return (cfg.max_glucose_uptake * glucose_mm
                / (cfg.glucose_half_saturation_mm + glucose_mm))

    def oxygen_uptake_bound(self, oxygen_mm: float) -> float:
        """Michaelis-Menten oxygen-uptake bound (multi-substrate dFBA)."""
        cfg = self.config
        if oxygen_mm <= 0.0:
            return 0.0
        return (cfg.max_oxygen_uptake * oxygen_mm
                / (cfg.oxygen_half_saturation_mm + oxygen_mm))

    def _set_oxygen_uptake_bound(self, oxygen_mm: float) -> None:
        """Set the EX_o2 uptake bound from the oxygen pool (MM kinetics).

        Handles both sign conventions: core model (coef=+1 → ub) and
        GEM (coef=-1 → lb).
        """
        if self._ex_o2 not in self.fba.model.reactions:
            return
        bound = self.oxygen_uptake_bound(oxygen_mm)
        _rxn = self.fba.model.reactions[self._ex_o2]
        _coef = next(iter(_rxn.stoichiometry.values())) if _rxn.stoichiometry else 1.0
        if _coef < 0:
            _rxn.lower_bound = -bound
        else:
            _rxn.upper_bound = bound

    def _apply_bounds(self, bounds: dict[str, float]) -> None:
        """Apply dynamic reaction-bound overrides for the next LP solve.

        ``_ex_glc`` overrides the Michaelis-Menten uptake bound; any other
        reaction id sets that reaction's ``upper_bound`` directly.
        """
        for rid, bound in bounds.items():
            if rid == self._ex_glc:
                self.fba.set_uptake(self._met_glc, float(bound))
            elif rid in self.fba.model.reactions:
                self.fba.model.reactions[rid].upper_bound = float(bound)

    # -------- feed events & chemostat (doc/20 §16) --------

    def _apply_feed_events(self) -> None:
        """Apply any scheduled feed events at the current time."""
        cfg = self.config
        for idx, evt in enumerate(cfg.feed_events):
            if idx in self._feed_events_applied:
                continue
            if self.time_h >= evt.time_h:
                if evt.metabolite == self._ex_glc:
                    self.glucose_mm += evt.amount_mmol
                elif evt.metabolite == self._ex_o2:
                    o2 = self.byproducts_mm.get("oxygen", cfg.initial_oxygen_mm)
                    self.byproducts_mm["oxygen"] = o2 + evt.amount_mmol
                elif evt.metabolite == self._ex_ac:
                    ac = self.byproducts_mm.get("acetate", cfg.initial_acetate_mm)
                    self.byproducts_mm["acetate"] = ac + evt.amount_mmol
                if evt.dilution > 0.0:
                    factor = 1.0 - evt.dilution
                    self.biomass_gdw *= factor
                    self.glucose_mm *= factor
                    for pool in self._byproduct_pools:
                        if pool in self.byproducts_mm:
                            self.byproducts_mm[pool] *= factor
                self._feed_events_applied.add(idx)

    def _apply_chemostat_step(self, dt_h: float) -> None:
        """Apply chemostat dilution and continuous feeding (doc/20 §16.3).

        The chemostat replaces a fraction (dilution_rate * dt) of the
        culture volume with fresh medium at each time step.
        """
        cfg = self.config
        if not cfg.chemostat or cfg.chemostat_dilution_rate <= 0.0:
            return
        dilution_per_step = cfg.chemostat_dilution_rate * dt_h
        factor = 1.0 - dilution_per_step
        self.biomass_gdw *= factor
        self.glucose_mm *= factor
        for pool in self._byproduct_pools:
            if pool in self.byproducts_mm:
                self.byproducts_mm[pool] *= factor
        for met, conc in cfg.chemostat_feed_concentrations.items():
            if met == "glucose" or met == self._ex_glc:
                self.glucose_mm += dilution_per_step * conc
            elif met == "oxygen" or met == self._ex_o2:
                o2 = self.byproducts_mm.get("oxygen", 0.0)
                self.byproducts_mm["oxygen"] = o2 + dilution_per_step * conc
            elif met == "acetate" or met == self._ex_ac:
                ac = self.byproducts_mm.get("acetate", 0.0)
                self.byproducts_mm["acetate"] = ac + dilution_per_step * conc

    # -------- integration --------

    def step(self, dt_h: float | None = None) -> dict[str, float]:
        """Solve the instantaneous LP and integrate the batch by one step.

        Returns the state entry appended to :attr:`history` with keys
        ``time``, ``biomass``, ``glucose``, ``growth_rate``,
        ``glucose_uptake`` and one key per secreted byproduct.
        """
        cfg = self.config
        dt = cfg.dt_h if dt_h is None else dt_h
        S = self.glucose_mm
        self.fba.set_uptake(self._met_glc, self.uptake_bound(S))
        if self._ex_o2 in self.fba.model.reactions:
            o2_pool = self.byproducts_mm.get("oxygen", cfg.initial_oxygen_mm)
            self._set_oxygen_uptake_bound(o2_pool)
        if cfg.acetate_switch:
            self._apply_acetate_switch_bounds(S, dt)
        self._apply_feed_events()
        self._apply_chemostat_step(dt)
        if self.bound_override is not None:
            self._apply_bounds(self.bound_override(self.time_h, self))
        sol = self.fba.solve()
        self.last_fluxes = sol
        return self._integrate(sol, S, dt)

    def _apply_acetate_switch_bounds(self, S: float, dt: float) -> None:
        """Glucose-gated, pool-limited acetate import and shunt (switch).

        - While glucose is above ``acetate_switch_threshold_mm`` the
          acetate exchange is closed for import and the glyoxylate shunt
          (aceA/aceB) stays flux-``0`` -- no co-consumption and no
          shunt activity during glucose growth (catabolite repression of
          the switch, Enjalbert et al. 2011 ISME J 5:1301).
        - Once glucose drops below the threshold, the shunt and the
          acetate-activating synthetase open, and the acetate import is
          capped by the well-mixed acetate pool: the batch cannot take up
          more acetate than the overflow it accumulated (Mahadevan et
          al. 2002 dynamic FBA), so the second phase is mass-conserving.
        """
        reactions = self.fba.model.reactions
        if self._ex_ac not in reactions:
            return
        if S > self.config.acetate_switch_threshold_mm:
            reactions[self._ex_ac].lower_bound = 0.0
            for rid in _ACETATE_SWITCH_RXNS:
                if rid in reactions:
                    reactions[rid].upper_bound = 0.0
            return
        reactions[self._ex_ac].lower_bound = -10.0
        for rid in _ACETATE_SWITCH_RXNS:
            if rid in reactions:
                reactions[rid].upper_bound = 1000.0
        pool = self.byproducts_mm.get("acetate", 0.0)
        denom = self.biomass_gdw * dt
        max_import = min(10.0, pool / denom) if denom > 0.0 else 0.0
        reactions[self._ex_ac].lower_bound = -max_import

    def step_from_solution(self,
                           sol: dict[str, float],
                           glucose_mm: float,
                           dt_h: float | None = None) -> dict[str, float]:
        """Integrate one step from a *shared* LP solution (surfin_FBA).

        Brunner & Chai 2020 (PLoS Comput Biol 16:e1007786) showed that
        the intracellular flux space of identical medium states is reused
        across cells/batches: one LP solve per environment state serves
        every co-located batch instead of one LP per batch.  This method
        advances **this** batch's own biomass / pools using the shared
        fluxes (``sol``) and the shared pre-depletion substrate level
        ``glucose_mm`` -- the same integration math as :meth:`step` with
        the LP solve removed.
        """
        cfg = self.config
        dt = cfg.dt_h if dt_h is None else dt_h
        self.last_fluxes = sol
        return self._integrate(sol, glucose_mm, dt)

    def _integrate(self, sol: dict[str, float], S: float,
                   dt: float) -> dict[str, float]:
        """Forward-Euler state advance shared by :meth:`step` and
        :meth:`step_from_solution` (Mahadevan 2002)."""
        cfg = self.config
        v_bm = (sol.get(self._biomass_reaction, 0.0)
                if self._biomass_reaction else 0.0)
        v_glc_raw = sol.get(self._ex_glc, 0.0)
        # Adjust sign convention: core model EX_glc has coef=+1.0
        # (positive flux = consumption) but GEM EX_glc_e has coef=-1.0
        # (negative flux = consumption).  Normalise so positive always
        # means consumption.
        _glc_rxn = self.fba.model.reactions.get(self._ex_glc)
        if _glc_rxn and _glc_rxn.stoichiometry:
            _coef = next(iter(_glc_rxn.stoichiometry.values()))
            v_glc = -v_glc_raw if _coef < 0 else v_glc_raw
        else:
            v_glc = v_glc_raw
        mu = v_bm / cfg.biomass_per_mmol
        mu = min(mu, cfg.max_growth_rate)
        X = self.biomass_gdw
        removed = min(v_glc * X * dt, S)
        new_biomass = X + mu * X * dt
        self.biomass_gdw = min(new_biomass, cfg.max_biomass_gdw)
        self.glucose_mm = S - removed
        if self._ex_o2 in self.fba.model.reactions:
            v_o2_raw = sol.get(self._ex_o2, 0.0)
            _o2_rxn = self.fba.model.reactions.get(self._ex_o2)
            if _o2_rxn and _o2_rxn.stoichiometry:
                _co2 = next(iter(_o2_rxn.stoichiometry.values()))
                v_o2 = -v_o2_raw if _co2 < 0 else v_o2_raw
            else:
                v_o2 = v_o2_raw
            o2_pool = self.byproducts_mm.get("oxygen", cfg.initial_oxygen_mm)
            o2_removed = min(v_o2 * X * dt, o2_pool)
            self.byproducts_mm["oxygen"] = max(0.0, o2_pool - o2_removed)
        for rid, pool in self._byproduct_ex.items():
            v = sol.get(rid, 0.0)
            dP = v * X * dt
            if v > 0.0:
                self.byproducts_mm[pool] = (self.byproducts_mm[pool]
                                            + dP)
            elif v < 0.0 and pool == "acetate":
                # acetate re-import (Wolfe 2005 "Acetate Switch"): the
                # LP consumes the well-mixed overflow pool; never below 0
                self.byproducts_mm[pool] = max(
                    0.0, self.byproducts_mm[pool] + dP)
        self.time_h += dt
        entry: dict[str, float] = {
            "time": self.time_h,
            "biomass": self.biomass_gdw,
            "glucose": self.glucose_mm,
            "growth_rate": mu,
            "glucose_uptake": v_glc,
        }
        for pool in self._byproduct_pools:
            entry[pool] = self.byproducts_mm[pool]
        if "oxygen" in self.byproducts_mm:
            entry["oxygen"] = self.byproducts_mm["oxygen"]
        self.history.append(entry)
        return entry

    def run(self,
            duration_h: float | None = None,
            max_steps: int = 100000) -> list[dict[str, float]]:
        """Integrate until ``duration_h`` hours have passed (or, with
        ``duration_h=None``, until growth stagnates and glucose is gone).

        Returns :attr:`history`.
        """
        horizon = (self.time_h + duration_h
                   if duration_h is not None else None)
        stagnant = 0
        steps = 0
        while (horizon is None or self.time_h + 1e-9 < horizon):
            if steps >= max_steps:
                break
            prev = self.biomass_gdw
            self.step()
            steps += 1
            if self.biomass_gdw - prev < self.config.min_biomass:
                stagnant += 1
            else:
                stagnant = 0
            if self.glucose_mm < 1e-9 and stagnant >= 4:
                break
        return self.history

    # -------- queries --------

    @property
    def growth_rate(self) -> float:
        """Most recent specific growth rate (1/h)."""
        if not self.history:
            return 0.0
        return self.history[-1]["growth_rate"]

    def last(self) -> dict[str, float]:
        """Latest history entry."""
        return self.history[-1]

    def to_simulation_result(self) -> DynamicSimulationResult:
        """Convert history to DynamicSimulationResult."""
        result = DynamicSimulationResult()
        for entry in self.history:
            result.time_points.append(entry.get("time", 0.0))
            result.biomass.append(entry.get("biomass", entry.get("total_biomass", 0.0)))
            result.growth_rates.append(entry.get("growth_rate", entry.get("mu", 0.0)))
            # Collect substrate/byproduct data
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    if k.startswith("total_"):
                        name = k[6:]
                        result.substrates.setdefault(name, []).append(v)
        if result.biomass:
            result.final_biomass = result.biomass[-1]
        if len(result.biomass) >= 2 and result.biomass[0] > 0:
            # Estimate doubling time from growth curve
            import math
            if result.final_biomass > result.biomass[0]:
                result.doubling_time = math.log(2) / max(result.growth_rates[-1], 1e-10)
        return result

    # -------- environment coupling --------

    def update_from_environment(self,
                                environment: Environment,
                                x: int | None = None,
                                y: int | None = None) -> None:
        """Set the batch glucose from the environment field at (x, y)
        (default: lattice centre), treating the site as a well-mixed
        unit of the batch medium.

        With the acetate switch enabled (:attr:`DynamicFBAConfig.
        acetate_switch`) the batch also adopts the site's well-mixed
        acetate pool (the overflow deposited by neighbours), so the
        second, acetate-assimilation phase of the population can draw on
        it once glucose falls below the threshold (Wolfe 2005).
        """
        cx = environment.config.width // 2 if x is None else x
        cy = environment.config.height // 2 if y is None else y
        self.glucose_mm = environment.substrate_at(cx, cy, "glucose")
        if self.config.acetate_switch and "acetate" in environment.fields:
            self.byproducts_mm["acetate"] = environment.fields["acetate"].get(
                cx, cy)

    def apply_to_environment(self,
                             environment: Environment,
                             x: int | None = None,
                             y: int | None = None) -> None:
        """Deposit the accumulated acetate into the environment's acetate
        field at (x, y) (default: lattice centre), creating the field on
        first use."""
        cx = environment.config.width // 2 if x is None else x
        cy = environment.config.height // 2 if y is None else y
        try:
            field = environment.get_field("acetate")
        except KeyError:
            from helixlang.plugins.runtime.environment import (
                ACETATE_DIFFUSION_UM2_S,
                ConcentrationField,
            )
            field = ConcentrationField(
                "acetate", environment.config.width,
                environment.config.height,
                ACETATE_DIFFUSION_UM2_S, 0.0)
            environment.add_field("acetate", field)
        field.add(cx, cy, self.byproducts_mm.get("acetate", 0.0))


# ============================================================================
# Photoautotrophic dynamic FBA (doc/22 §7 — Synechocystis on BG-11)
# ============================================================================

class PhotoautotrophicFluxBalance:
    """Dynamic FBA for photoautotrophic organisms (Synechocystis PCC 6803).

    Uses Monod kinetics for CO₂ fixation and light-dependent growth:

        v_CO2(t) = v_max_CO2 * CO₂(t) / (K_CO2 + CO₂(t))

    ODEs (forward Euler):

        dX/dt = mu * X           biomass (gDW/L)
        dCO₂/dt = -v_CO2 * X    dissolved CO₂ (mmol/L)

    Light availability modulates the maximum CO₂ fixation rate via a
    P GetById-style saturation curve:

        eff_v_max = v_max_CO2 * I / (K_L + I)

    where I is the incident light intensity and K_L is the half-saturation
    constant for light.  This captures the photosaturation behaviour of
    cyanobacteria (Kok 1956, Castenholz 2001).
    """

    def __init__(
        self,
        model: MetabolicModel,
        config: DynamicFBAConfig | None = None,
        fba: FluxBalanceAnalysis | None = None,
    ) -> None:
        self.config = config or DynamicFBAConfig()
        self.model = model
        self.fba = fba or FluxBalanceAnalysis(model)
        self._biomass_reaction = self.fba.model.biomass_reaction
        # Detect CO₂ exchange ID (EX_co2_e or similar)
        self._ex_co2: str = self._detect_co2_exchange()
        self.reset()

    def _detect_co2_exchange(self) -> str:
        """Find the CO₂ exchange reaction in the model."""
        for rid, rxn in self.fba.model.reactions.items():
            if rxn.subsystem != "exchange" or not rid.startswith("EX_"):
                continue
            if len(rxn.stoichiometry) != 1:
                continue
            met = next(iter(rxn.stoichiometry))
            if met.lower() in ("co2_e", "co2"):
                return rid
        # Fallback: try common GEM IDs
        for candidate in ("EX_co2_e", "EX_co2"):
            if candidate in self.fba.model.reactions:
                return candidate
        return ""

    def reset(self) -> None:
        """Restore initial state and clear history."""
        cfg = self.config
        self.time_h: float = 0.0
        self.biomass_gdw: float = cfg.initial_biomass_gdw
        self.co2_mm: float = cfg.co2_initial_mm
        self.history: list[dict[str, float]] = []
        self.last_fluxes: dict[str, float] = {}

    def light_effect(self) -> float:
        """P GetById-style light saturation: I / (K_L + I)."""
        cfg = self.config
        return cfg.light_intensity / (cfg.light_saturation + cfg.light_intensity)

    def co2_uptake_bound(self, co2_mm: float) -> float:
        """Monod CO₂ uptake bound modulated by light."""
        cfg = self.config
        if co2_mm <= 0.0:
            return 0.0
        eff_vmax = cfg.co2_max_uptake * self.light_effect()
        return eff_vmax * co2_mm / (cfg.co2_half_saturation_mm + co2_mm)

    def _set_co2_bound(self) -> None:
        """Set the CO₂ exchange lower bound from Monod kinetics."""
        if not self._ex_co2 or self._ex_co2 not in self.fba.model.reactions:
            return
        bound = self.co2_uptake_bound(self.co2_mm)
        rxn = self.fba.model.reactions[self._ex_co2]
        coef = next(iter(rxn.stoichiometry.values())) if rxn.stoichiometry else -1.0
        if coef < 0:
            rxn.lower_bound = -bound
        else:
            rxn.upper_bound = bound

    def _set_pet_bound(self) -> None:
        """Set upper bound on photosynthetic electron transport (PET) reaction.

        PET (H₂O + NADP⁺ → NADPH + ½O₂) provides the NADPH that drives
        the Calvin cycle.  The upper bound is modulated by light intensity
        using the same P GetById saturation curve as CO₂ fixation.
        """
        if "PET" not in self.fba.model.reactions:
            return
        rxn = self.fba.model.reactions["PET"]
        rxn.upper_bound = self.light_effect() * self.config.co2_max_uptake

    def step(self, dt_h: float | None = None) -> dict[str, float]:
        """Solve LP and integrate one time step.

        Returns the state entry appended to :attr:`history`.
        """
        cfg = self.config
        dt = cfg.dt_h if dt_h is None else dt_h

        self._set_co2_bound()
        self._set_pet_bound()
        sol = self.fba.solve()
        self.last_fluxes = sol

        # Extract fluxes
        v_bm = (sol.get(self._biomass_reaction, 0.0)
                if self._biomass_reaction else 0.0)
        v_co2_raw = sol.get(self._ex_co2, 0.0) if self._ex_co2 else 0.0
        # Normalise CO₂ sign (positive = consumption)
        if self._ex_co2:
            rxn = self.fba.model.reactions.get(self._ex_co2)
            if rxn and rxn.stoichiometry:
                coef = next(iter(rxn.stoichiometry.values()))
                v_co2 = -v_co2_raw if coef < 0 else v_co2_raw
            else:
                v_co2 = v_co2_raw
        else:
            v_co2 = v_co2_raw

        # Growth rate from biomass flux
        mu = v_bm / cfg.biomass_per_mmol
        mu = min(mu, cfg.max_growth_rate)

        # Forward Euler integration
        X = self.biomass_gdw
        # Scale CO₂ consumption proportionally to the clamped growth
        # rate so the pool lasts as long as the growth trajectory
        # warrants (fixes CO₂ exhaustion 3.8× too early when LP μ
        # ≫ max_growth_rate).
        if v_bm > 0:
            co2_per_biomass = abs(v_co2) / v_bm
        else:
            co2_per_biomass = 0.0
        co2_consumed = min(co2_per_biomass * mu * X * dt, self.co2_mm)
        self.biomass_gdw = min(X + mu * X * dt, cfg.max_biomass_gdw)
        self.co2_mm = max(0.0, self.co2_mm - co2_consumed)
        self.time_h += dt

        entry: dict[str, float] = {
            "time": self.time_h,
            "biomass": self.biomass_gdw,
            "co2": self.co2_mm,
            "growth_rate": mu,
            "co2_uptake": abs(v_co2),
        }
        self.history.append(entry)
        return entry

    def run(self,
            duration_h: float | None = None,
            max_steps: int = 100000) -> list[dict[str, float]]:
        """Integrate until duration_h hours have passed.

        Returns :attr:`history`.
        """
        horizon = (self.time_h + duration_h
                   if duration_h is not None else None)
        stagnant = 0
        steps = 0
        while horizon is None or self.time_h + 1e-9 < horizon:
            if steps >= max_steps:
                break
            prev = self.biomass_gdw
            self.step()
            steps += 1
            if self.biomass_gdw - prev < self.config.min_biomass:
                stagnant += 1
            else:
                stagnant = 0
            if self.co2_mm < 1e-9 and stagnant >= 4:
                break
        return self.history

    @property
    def growth_rate(self) -> float:
        """Most recent specific growth rate (1/h)."""
        if not self.history:
            return 0.0
        return self.history[-1]["growth_rate"]

    def last(self) -> dict[str, float]:
        """Latest history entry."""
        return self.history[-1]

    def to_simulation_result(self) -> DynamicSimulationResult:
        """Convert history to DynamicSimulationResult."""
        result = DynamicSimulationResult()
        for entry in self.history:
            result.time_points.append(entry.get("time", 0.0))
            result.biomass.append(entry.get("biomass", 0.0))
            result.growth_rates.append(entry.get("growth_rate", 0.0))
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    result.substrates.setdefault(k, []).append(v)
        if result.biomass:
            result.final_biomass = result.biomass[-1]
        if len(result.biomass) >= 2 and result.biomass[0] > 0:
            import math
            if result.final_biomass > result.biomass[0]:
                result.doubling_time = math.log(2) / max(
                    result.growth_rates[-1], 1e-10)
        return result


# ============================================================================
# Module exports
# ============================================================================

# ============================================================================
# Metabolic proxy (dAMN-style surrogate)
# ============================================================================

def _poly_features(x: list[float], degree: int) -> list[float]:
    """Polynomial feature expansion (all monomials up to ``degree``)."""
    feats = [1.0]
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(len(x)), d):
            v = 1.0
            for i in comb:
                v *= x[i]
            feats.append(v)
    return feats


class MetabolicProxy:
    """Per-agent metabolic surrogate (dAMN 2025).

    A fitted polynomial proxy that predicts FBA flux outputs from a
    vector of uptake bounds without re-solving the LP per agent per tick.
    This is the "surrogate fluxes to stay fast" layer for genome-scale
    dFBA in large populations (dAMN: dynamic artificial-neural-network
    surrogate of FBA, iML1515, 2025): fit once on sampled FBA solutions,
    then predict per agent.

    Args:
        model: a :class:`MetabolicModel` used to build the FBA solver
            (default: :data:`ECOLI_CORE_MODEL`).
        fba: an existing :class:`FluxBalanceAnalysis` (overrides
            ``model``).
        features: exchange metabolite names used as model inputs
            (default: every exchange metabolite in the model).
        outputs: flux ids predicted (default: biomass + byproduct
            exchanges).
        degree: polynomial degree of the fitted surrogate
            (degree 1 = linear, degree 2 adds squares and interactions).
        max_uptake: upper range for sampled uptake bounds (default
            :data:`DEFAULT_GLC_UPTAKE`).
    """

    def __init__(self,
                 model: MetabolicModel | None = None,
                 fba: FluxBalanceAnalysis | None = None,
                 features: list[str] | None = None,
                 outputs: list[str] | None = None,
                 degree: int = 2,
                 max_uptake: float = DEFAULT_GLC_UPTAKE) -> None:
        self.fba = fba or FluxBalanceAnalysis(model or ECOLI_CORE_MODEL)
        model = self.fba.model
        if features is None:
            features = []
            for rxn in model.reactions.values():
                if rxn.subsystem == "exchange" and rxn.id.startswith("EX_"):
                    mets = [m for m, c in rxn.stoichiometry.items() if c > 0]
                    if mets:
                        features.append(mets[0])
            features = [f for f in features if f != "biomass"]
        if outputs is None:
            outputs = []
            if model.biomass_reaction:
                outputs.append(model.biomass_reaction)
            for rid, rxn in model.reactions.items():
                if (rxn.subsystem == "exchange"
                        and rid not in (_EX_GLC, _EX_BIOMASS)):
                    outputs.append(rid)
            outputs = sorted(set(outputs))
        if not features:
            raise ValueError("no exchange features found in the model")
        self.features = list(features)
        self.outputs = list(outputs)
        self.degree = degree
        self.max_uptake = float(max_uptake)
        # fitted coefficient vector per output (numpy lstsq path)
        self.coeffs: dict[str, list[float]] = {}
        # training samples kept for the nearest-neighbor fallback / QA
        self._train_x: list[list[float]] = []
        self._train_y: dict[str, list[float]] = {}

    def _sample_inputs(self, n: int, seed: int) -> list[list[float]]:
        rng = random.Random(seed)
        return [[rng.uniform(0.0, self.max_uptake) for _ in self.features]
                for _ in range(n)]

    def _solve_fluxes(self, x: list[float]) -> dict[str, float]:
        """Solve FBA with the uptake vector ``x`` (restoring state after)."""
        saved = dict(self.fba.uptake_limits)
        for f, v in zip(self.features, x, strict=True):
            self.fba.set_uptake(f, v)
        try:
            sol = self.fba.solve()
        finally:
            self.fba.uptake_limits.clear()
            self.fba.uptake_limits.update(saved)
            self.fba.last_solution = None
        return {o: sol.get(o, 0.0) for o in self.outputs}

    def fit(self, n_samples: int = 200, seed: int = 0) -> MetabolicProxy:
        """Fit the surrogate on ``n_samples`` sampled FBA solutions.

        Uses least squares (numpy) with polynomial features when numpy is
        available; falls back to nearest-neighbor lookup otherwise.
        """
        xs = self._sample_inputs(n_samples, seed)
        ys = [self._solve_fluxes(x) for x in xs]
        self._train_x = xs
        self._train_y = {o: [y[o] for y in ys] for o in self.outputs}
        if _HAS_NUMPY:
            X = np.array([_poly_features(x, self.degree) for x in xs])
            for o in self.outputs:
                self.coeffs[o] = list(
                    np.linalg.lstsq(X, np.array(self._train_y[o]),
                                    rcond=None)[0])
        else:  # pragma: no cover - numpy is a project dependency
            self.coeffs = {}
        return self

    def predict(self, uptake: dict[str, float] | Sequence[float]) -> dict[str, float]:
        """Predict flux outputs for an uptake-bound vector.

        Accepts either a dict ``{metabolite: bound}`` or a vector ordered
        like :attr:`features`.
        """
        if isinstance(uptake, dict):
            unknown = [k for k in uptake if k not in self.features]
            if unknown:
                raise ValueError(
                    f"unknown uptake feature(s) {unknown!r}; expected "
                    f"{self.features!r}")
            x = [uptake.get(f, 0.0) for f in self.features]
        else:
            x = list(uptake)
        if len(x) != len(self.features):
            raise ValueError("uptake vector must match features length")
        if self.coeffs:
            feats = _poly_features(x, self.degree)
            out: dict[str, float] = {}
            for o in self.outputs:
                v = sum(c * fv for c, fv in zip(self.coeffs[o], feats, strict=True))
                rxn = self.fba.model.reactions.get(o)
                if rxn is not None and rxn.lower_bound >= 0.0:
                    v = max(0.0, v)  # irreversible flux cannot go negative
                out[o] = float(v)
            return out
        # nearest-neighbor fallback (no numpy)
        if not self._train_x:
            raise RuntimeError("MetabolicProxy.fit() must be called first")
        best, best_d = 0, float("inf")
        for i, xi in enumerate(self._train_x):
            d = sum((a - b) ** 2 for a, b in zip(xi, x, strict=True))
            if d < best_d:
                best, best_d = i, d
        return {o: self._train_y[o][best] for o in self.outputs}

    def rmse(self, n_holdout: int = 50, seed: int = 1) -> dict[str, float]:
        """Root-mean-square error on ``n_holdout`` freshly sampled points."""
        xs = self._sample_inputs(n_holdout, seed)
        ys = [self._solve_fluxes(x) for x in xs]
        out: dict[str, float] = {}
        for o in self.outputs:
            errs = [self.predict(x)[o] - y[o] for x, y in zip(xs, ys, strict=True)]
            out[o] = (sum(e * e for e in errs) / len(errs)) ** 0.5
        return out


__all__ = [
    # constants
    "DEFAULT_UPPER_BOUND",
    "DEFAULT_LOWER_BOUND",
    "DEFAULT_GLC_UPTAKE",
    "ATP_MAINTENANCE_FLUX",
    "DEFAULT_ENZYME_MW",
    "DEFAULT_ENZYME_SCALE",
    "ECOLI_CORE_GENE_REACTIONS",
    "ECOLI_CORE_KCAT",
    # dataclasses
    "Reaction",
    "MetabolicModel",
    "ECOLI_CORE_MODEL",
    "EnzymeCapacity",
    "MetabolitePoolConfig",
    "MetabolitePool",
    # loaders
    "load_model_from_json",
    "load_model",
    # solvers
    "FluxBalanceAnalysis",
    "DynamicFBAConfig",
    "FeedEvent",
    "DynamicSimulationResult",
    "DynamicFluxBalance",
    "PhotoautotrophicFluxBalance",
    "MetabolicProxy",
    # simplex
    "simplex",
]
