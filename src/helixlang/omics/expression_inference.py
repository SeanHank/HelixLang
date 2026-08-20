"""Expression inference from GRN (doc/20 §14).

Computes steady-state enzyme concentrations from a gene regulatory network,
promoter strength model, and mRNA/protein degradation rates.  The output
is a ``gene_id → relative_enzyme_level`` mapping that can be fed directly
into :class:`FluxBalanceAnalysis` via ``set_enzyme_levels()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helixlang.annotation import GeneAnnotation
    from helixlang.gem.grn_inference import GRNInferenceResult

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExpressionModel:
    """Simple gene expression model for prokaryotes (doc/20 §14.3).

    All values are *relative* (0.0–1.0) unless otherwise noted.
    Defaults are chosen to be biologically reasonable for E. coli under
    standard laboratory conditions.
    """

    promoter_strength: dict[str, float] = field(default_factory=dict)
    mrna_half_life: dict[str, float] = field(default_factory=dict)
    protein_half_life: dict[str, float] = field(default_factory=dict)
    rbs_strength: dict[str, float] = field(default_factory=dict)
    tf_effects: dict[str, list[tuple[str, float, float, str]]] = field(
        default_factory=dict
    )


@dataclass
class ExpressionState:
    """Snapshot of expression levels at a point in time."""

    gene_levels: dict[str, float] = field(default_factory=dict)
    enzyme_levels: dict[str, float] = field(default_factory=dict)
    mrna_levels: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Hill function
# ---------------------------------------------------------------------------


def hill_function(
    tf_level: float,
    kd: float,
    n: float,
    effect: str = "activation",
) -> float:
    """Compute TF modulation of target gene expression (doc/20 §14.3).

    Parameters
    ----------
    tf_level : float
        Relative concentration of the transcription factor (0.0–1.0).
    kd : float
        Dissociation constant (half-maximal level).
    n : float
        Hill coefficient (cooperativity).
    effect : str
        ``"activation"`` or ``"repression"``.

    Returns
    -------
    float
        Multiplicative factor in [0, 1].
    """
    if kd <= 0:
        return 1.0
    ratio: float = tf_level / kd
    ratio_n: float = ratio ** n
    if effect == "repression":
        return 1.0 / (1.0 + ratio_n)
    return ratio_n / (1.0 + ratio_n)


# ---------------------------------------------------------------------------
# Build expression model from GRN
# ---------------------------------------------------------------------------


def build_expression_model(
    grn_result: GRNInferenceResult,
    annotations: dict[str, GeneAnnotation] | None = None,
) -> ExpressionModel:
    """Build an :class:`ExpressionModel` from GRN inference results.

    For each TF → target edge in the GRN, a Hill-function entry is stored
    in ``tf_effects``.  Promoter / RBS strengths default to 1.0 for genes
    that lack annotation data.
    """
    model = ExpressionModel()

    # Collect all genes from annotations
    all_genes: list[str] = []
    if annotations:
        all_genes = list(annotations.keys())

    # Default promoter / RBS strengths
    for gid in all_genes:
        model.promoter_strength.setdefault(gid, 1.0)
        model.rbs_strength.setdefault(gid, 1.0)
        # mRNA half-life ~2–5 min for E. coli; default 3 min
        model.mrna_half_life.setdefault(gid, 3.0)
        # Protein half-life ~30–300 min; default 60 min
        model.protein_half_life.setdefault(gid, 60.0)

    # Populate tf_effects from GRN edges
    for edge in grn_result.regulatory_edges:
        target = edge.target_gene
        kd = 0.5  # default half-max
        n = 2.0   # default Hill coefficient
        effect = edge.regulation_type
        model.tf_effects.setdefault(target, []).append(
            (edge.tf_id, kd, n, effect)
        )

    return model


# ---------------------------------------------------------------------------
# Expression inference
# ---------------------------------------------------------------------------


def infer_expression(
    grn_result: GRNInferenceResult,
    annotations: dict[str, GeneAnnotation] | None = None,
    model: ExpressionModel | None = None,
    environment: dict[str, float] | None = None,
) -> dict[str, float]:
    """Infer steady-state enzyme concentrations from GRN + expression model.

    Parameters
    ----------
    grn_result : GRNInferenceResult
        Result of GRN inference (edges, TF candidates).
    annotations : dict
        Gene annotations keyed by gene_id.
    model : ExpressionModel, optional
        Pre-built expression model.  If ``None``, one is built from the GRN.
    environment : dict, optional
        Environmental TF levels (TF gene_id → relative concentration).
        Defaults to 0.5 for all TFs.

    Returns
    -------
    dict[str, float]
        ``gene_id → relative enzyme level`` (0.0–1.0).
    """
    if model is None:
        model = build_expression_model(grn_result, annotations)

    all_genes = set(model.promoter_strength.keys())
    # Also add targets from tf_effects
    for targets in model.tf_effects.values():
        for entry in targets:
            all_genes.add(entry[0])  # tf_id
            # target is the key of tf_effects

    # Default TF levels (assume all TFs at basal level)
    tf_levels: dict[str, float] = {}
    for candidate in grn_result.tf_candidates:
        tf_levels[candidate.gene_id] = 0.5
    if environment:
        tf_levels.update({
            k: v for k, v in environment.items()
            if k in tf_levels
        })

    # Compute expression for each gene
    result: dict[str, float] = {}
    for gene_id in all_genes:
        # Basal expression from promoter + RBS
        basal = model.promoter_strength.get(gene_id, 1.0)
        rbs = model.rbs_strength.get(gene_id, 1.0)
        expr = basal * rbs

        # Apply TF regulation (multiplicative Hill functions)
        tf_reg = model.tf_effects.get(gene_id, [])
        for entry in tf_reg:
            tf_id, kd, n, effect = entry
            tf_level = tf_levels.get(tf_id, 0.5)
            mod = hill_function(tf_level, kd, n, effect)
            expr *= mod

        result[gene_id] = max(0.0, min(1.0, expr))

    return result


def infer_expression_at_time(
    grn_result: GRNInferenceResult,
    annotations: dict[str, GeneAnnotation] | None = None,
    model: ExpressionModel | None = None,
    tf_levels: dict[str, float] | None = None,
    time_hours: float = 0.0,
) -> ExpressionState:
    """Compute expression state at a specific time point (for dFBA).

    At steady state the mRNA and protein levels are constant, so this
    function is equivalent to :func:`infer_expression` but returns the
    full :class:`ExpressionState` with mRNA and protein levels.
    """
    env = tf_levels if tf_levels else {}
    enzyme_levels = infer_expression(
        grn_result, annotations, model, environment=env,
    )

    # mRNA and protein levels are assumed proportional to enzyme levels
    # at steady state (proportionality absorbed into degradation rates)
    mrna = {g: v for g, v in enzyme_levels.items()}

    return ExpressionState(
        gene_levels=enzyme_levels,
        enzyme_levels=enzyme_levels,
        mrna_levels=mrna,
        timestamp=time_hours,
    )
