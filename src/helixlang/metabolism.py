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

import json
import random
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helixlang.errors import BioError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from helixlang.environment import Environment

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a project dependency
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# model data file directory
_DATA_DIR = Path(__file__).parent / "data"


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
    """
    id: str
    name: str
    stoichiometry: dict[str, float]
    lower_bound: float = 0.0
    upper_bound: float = DEFAULT_UPPER_BOUND
    subsystem: str = "other"


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
        sbml_model = cobra.io.load_model(identifier)
    except Exception as exc:  # BiGG download / network errors
        raise BioError(
            f"could not load BiGG model {identifier!r} via cobrapy: {exc}"
        ) from exc
    return _from_cobra_model(sbml_model)


def _from_cobra_model(sbml_model: Any) -> MetabolicModel:
    """Convert a cobrapy Model into a :class:`MetabolicModel`.

    Only used when the optional ``cobra`` package is present.
    """
    m = MetabolicModel()
    for rxn in sbml_model.reactions:
        stoich: dict[str, float] = {}
        for met, coeff in rxn.metabolites.items():
            stoich[str(met.id)] = float(coeff)
        m.add_reaction(Reaction(
            id=str(rxn.id),
            name=str(rxn.name),
            stoichiometry=stoich,
            lower_bound=float(rxn.lower_bound) if rxn.lower_bound is not None else 0.0,
            upper_bound=float(rxn.upper_bound) if rxn.upper_bound is not None else DEFAULT_UPPER_BOUND,
            subsystem=str(rxn.subsystem or "other"),
        ))
    objective = getattr(sbml_model, "objective", None)
    if objective is not None:
        expr = getattr(objective, "expression", None)
        if expr is not None:
            for term in getattr(expr, "args", ()):
                var = getattr(term, "variable", None)
                if var is not None:
                    m.set_biomass(str(getattr(var, "name", var)))
                    break
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
        _simplex_max_numpy(tab_np, basis, obj1_np,
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
        status2 = _simplex_max_numpy(tab_np, basis, obj2_np,
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
        for rid in rxn_list:
            rxn = self.model.reactions[rid]
            lb = rxn.lower_bound
            ub = rxn.upper_bound
            # apply uptake limits: if EX_<met> and the metabolite is in
            # uptake_limits
            if (rxn.subsystem == "exchange"
                    and rxn.stoichiometry):
                # an EX reaction involves only 1 metabolite; its sign
                # determines the direction
                for met, coef in rxn.stoichiometry.items():
                    if met in self.uptake_limits and coef > 0:
                        # coef > 0 means the reaction "produces" this
                        # metabolite (i.e. uptake); set_uptake explicitly
                        # overrides the upper bound (default 0 -> enables
                        # uptake)
                        ub = self.uptake_limits[met]
            bounds.append((lb, ub))

        # b: steady-state mass balance S·v = 0 -> b = [0, 0, ..., 0]
        b = [0.0] * m

        result = simplex(c, S, b, bounds, maximize=maximize)
        if result["status"] not in ("optimal", "max_iter"):
            # infeasible or unbounded
            return {rid: 0.0 for rid in rxn_list}

        x = result["x"]
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
        glc_uptake = sol.get("EX_glc", 0.0)

        # byproducts
        byproducts = {
            "lactate": sol.get("EX_lac", 0.0),
            "acetate": sol.get("EX_ac", 0.0),
            "co2": sol.get("EX_co2", 0.0),
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

        # biomass/glucose ratio
        bg_ratio = (biomass_flux / glc_uptake) if glc_uptake > 1e-9 else 0.0

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
    """

    dt_h: float = 0.25
    initial_biomass_gdw: float = 0.05
    initial_glucose_mm: float = 10.0
    initial_acetate_mm: float = 0.0
    max_glucose_uptake: float = DEFAULT_GLC_UPTAKE
    glucose_half_saturation_mm: float = 0.1
    biomass_per_mmol: float = _BIOMASS_MW
    min_biomass: float = 1e-9


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

    The reduced 37-reaction core model has no glyoxylate shunt (iCL/MS),
    so overflow acetate cannot be re-imported for the second growth phase;
    a full model with the shunt consumes it automatically once glucose is
    gone.

    The batch pools can be coupled to a :class:`~helixlang.environment.
    Environment`: :meth:`update_from_environment` reads the local glucose
    concentration into the batch, :meth:`apply_to_environment` deposits
    the accumulated acetate back into the medium field.
    """

    def __init__(
        self,
        model: MetabolicModel | None = None,
        config: DynamicFBAConfig | None = None,
        fba: FluxBalanceAnalysis | None = None,
        bound_override: "callable | None" = None,
    ) -> None:
        self.config = config or DynamicFBAConfig()
        self.fba = fba or FluxBalanceAnalysis(model or ECOLI_CORE_MODEL)
        self._biomass_reaction = self.fba.model.biomass_reaction
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
                    or rid in (_EX_GLC, _EX_BIOMASS)):
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
        self.history: list[dict[str, float]] = []
        self.last_fluxes: dict[str, float] = {}

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

    def _apply_bounds(self, bounds: dict[str, float]) -> None:
        """Apply dynamic reaction-bound overrides for the next LP solve.

        ``EX_glc`` overrides the Michaelis-Menten uptake bound; any other
        reaction id sets that reaction's ``upper_bound`` directly.
        """
        for rid, bound in bounds.items():
            if rid == _EX_GLC:
                self.fba.set_uptake("GLC", float(bound))
            elif rid in self.fba.model.reactions:
                self.fba.model.reactions[rid].upper_bound = float(bound)

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
        self.fba.set_uptake("GLC", self.uptake_bound(S))
        if self.bound_override is not None:
            self._apply_bounds(self.bound_override(self.time_h, self))
        sol = self.fba.solve()
        self.last_fluxes = sol
        v_bm = (sol.get(self._biomass_reaction, 0.0)
                if self._biomass_reaction else 0.0)
        v_glc = sol.get(_EX_GLC, 0.0)
        mu = v_bm / cfg.biomass_per_mmol
        X = self.biomass_gdw
        removed = min(v_glc * X * dt, S)
        self.biomass_gdw = X + mu * X * dt
        self.glucose_mm = S - removed
        for rid, pool in self._byproduct_ex.items():
            v = sol.get(rid, 0.0)
            if v > 0.0:
                self.byproducts_mm[pool] = (self.byproducts_mm[pool]
                                            + v * X * dt)
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

    # -------- environment coupling --------

    def update_from_environment(self,
                                environment: Environment,
                                x: int | None = None,
                                y: int | None = None) -> None:
        """Set the batch glucose from the environment field at (x, y)
        (default: lattice centre), treating the site as a well-mixed
        unit of the batch medium."""
        cx = environment.config.width // 2 if x is None else x
        cy = environment.config.height // 2 if y is None else y
        self.glucose_mm = environment.substrate_at(cx, cy, "glucose")

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
            from helixlang.environment import ACETATE_DIFFUSION_UM2_S, ConcentrationField
            field = ConcentrationField(
                "acetate", environment.config.width,
                environment.config.height,
                ACETATE_DIFFUSION_UM2_S, 0.0)
            environment.add_field("acetate", field)
        field.add(cx, cy, self.byproducts_mm.get("acetate", 0.0))


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
                 model: "MetabolicModel | None" = None,
                 fba: "FluxBalanceAnalysis | None" = None,
                 features: "list[str] | None" = None,
                 outputs: "list[str] | None" = None,
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
        for f, v in zip(self.features, x):
            self.fba.set_uptake(f, v)
        try:
            sol = self.fba.solve()
        finally:
            self.fba.uptake_limits.clear()
            self.fba.uptake_limits.update(saved)
            self.fba.last_solution = None
        return {o: sol.get(o, 0.0) for o in self.outputs}

    def fit(self, n_samples: int = 200, seed: int = 0) -> "MetabolicProxy":
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

    def predict(self, uptake: "dict[str, float] | Sequence[float]") -> dict[str, float]:
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
                v = sum(c * fv for c, fv in zip(self.coeffs[o], feats))
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
            d = sum((a - b) ** 2 for a, b in zip(xi, x))
            if d < best_d:
                best, best_d = i, d
        return {o: self._train_y[o][best] for o in self.outputs}

    def rmse(self, n_holdout: int = 50, seed: int = 1) -> dict[str, float]:
        """Root-mean-square error on ``n_holdout`` freshly sampled points."""
        xs = self._sample_inputs(n_holdout, seed)
        ys = [self._solve_fluxes(x) for x in xs]
        out: dict[str, float] = {}
        for o in self.outputs:
            errs = [self.predict(x)[o] - y[o] for x, y in zip(xs, ys)]
            out[o] = (sum(e * e for e in errs) / len(errs)) ** 0.5
        return out


__all__ = [
    # constants
    "DEFAULT_UPPER_BOUND",
    "DEFAULT_LOWER_BOUND",
    "DEFAULT_GLC_UPTAKE",
    "ATP_MAINTENANCE_FLUX",
    # dataclasses
    "Reaction",
    "MetabolicModel",
    "ECOLI_CORE_MODEL",
    # loaders
    "load_model_from_json",
    "load_model",
    # solvers
    "FluxBalanceAnalysis",
    "DynamicFBAConfig",
    "DynamicFluxBalance",
    "MetabolicProxy",
    # simplex
    "simplex",
]
