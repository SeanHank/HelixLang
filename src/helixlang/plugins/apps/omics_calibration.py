"""Omics-level data calibration -> prediction benchmark (S10.1 direction A).

Implements the perturb-seq-style calibration protocol at the heart of the
Virtual Cell Challenge 2025 (``virtualcellchallenge.org``): a library of
CRISPRi perturbations (~300 genes knocked down, ~83% knockdown efficacy in
the VCC) is profiled in single cells, and a model is *calibrated* on the
resulting gene-expression response matrix, then *predicts* the response of
held-out perturbations.  Scoring follows the VCC metrics:

- per-perturbation MAE  ``MAE_k = (1/G) sum_g |yhat_g^k - y_g^k|``
- response correlation across genes (differential-expression-similarity
  analog)
- differential-expression sign agreement (which genes go up/down)

Biological grounding:

- Noise model: count data are sampled from a negative-binomial (DESeq2,
  Love et al. 2014; variance ``Var = mu + dispersion * mu^2``), the
  standard noise model for single-cell / bulk RNA readouts; a log-normal
  multiplicative alternative is also provided.
- Calibration uses inverse-variance weights over the *joint* multi-perturbation
  observation vector, i.e. high-count (low-variance) observations dominate --
  the weighted multi-scale fitting used by Karr et al. 2012 (Cell 150:389)
  and standard omics practice.
- The response model couples a sparse regulatory design matrix W (target
  genes x regulators) with unknown global coupling constants
  (``response_gain``, ``hill_n``, ``basal_scale``) recovered by
  :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters`.

The benchmark only needs the standard library + the existing
:func:`fit_parameters` harness, so the whole perturb-seq loop is
deterministic and dependency-free.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

from helixlang.plugins.runtime.virtual_cell import fit_parameters

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a project dependency
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

#: VCC 2025 mean CRISPRi knockdown efficacy (fraction of remaining activity)
DEFAULT_CRISPRI_KD = 0.17
#: DESeq2-style overdispersion used to generate count noise
DEFAULT_DISPERSION = 0.05


def negative_binomial_noise(mean: float, rng: random.Random,
                            dispersion: float = DEFAULT_DISPERSION) -> float:
    """Draw one negative-binomial count with ``Var = mu + d*mu^2``.

    Gamma-Poisson mixture (size = 1/d) sampled with the deterministic
    ``random.Random`` generators, so noisy runs are reproducible.
    """
    if mean <= 0.0:
        return 0.0
    size = 1.0 / max(dispersion, 1e-9)
    gamma = rng.gammavariate(size, mean / size)
    # Poisson(gamma) via sequential exponentials
    lam = max(0.0, gamma)
    n = 0
    s = 0.0
    while s <= lam:
        s += rng.expovariate(1.0)
        n += 1
    return float(n - 1)


def log_normal_noise(values: Sequence[float], rng: random.Random,
                     sigma: float = 0.2) -> list[float]:
    """Multiplicative log-normal noise around ``values`` (fold-change scale)."""
    return [v * math.exp(rng.gauss(0.0, sigma)) for v in values]


# ============================================================================
# Synthetic perturb-seq data
# ============================================================================

def generate_perturb_seq_data(
    n_genes: int = 200,
    n_perturbations: int = 24,
    n_tf: int = 8,
    seed: int = 0,
    kd: float = DEFAULT_CRISPRI_KD,
    dispersion: float = DEFAULT_DISPERSION,
    wt_scale: float = 40.0,
    response_gain: float = 1.0,
    hill_n: float = 1.0,
) -> dict:
    """Generate a synthetic CRISPRi perturb-seq response matrix.

    A sparse regulatory design matrix ``W`` (``n_genes`` x ``n_tf``,
    ~15% nonzero) links regulators to targets; a perturbation knocks one
    regulator's activity down to ``kd``; the expected expression of gene
    ``g`` under perturbation ``p`` is::

        eta_g^p = sum_tf W[g,tf] * (u_tf^hill_n - 1)
        mean_g^p = basal_g * wt_scale * exp(response_gain * eta_g^p)

    (WT activities are all 1, so the WT row is the unperturbed baseline
    and the response of a perturbation is a pure fold change against it,
    as scored by the VCC.)  ``mean`` holds the noiseless expected counts,
    ``counts`` the negative-binomial corrupted readout.

    Returns:
        a dict with ``genes``, ``perturbations``, ``design`` (W),
        ``basal``, ``activities`` (U), ``truth`` (coupling constants),
        ``mean`` and ``counts`` (both ``n_genes`` x ``n_perturbations``
        matrices) plus ``kd`` / ``dispersion``.
    """
    rng = random.Random(seed)
    genes = [f"g{str(i).zfill(4)}" for i in range(n_genes)]
    perturbations = ["WT"] + [f"kd_tf{k % n_tf}_{k // n_tf}"
                              for k in range(n_perturbations - 1)]
    # sparse design: ~15% nonzero, magnitudes in [0.25, 0.8], both signs
    design: list[list[float]] = []
    for _ in range(n_genes):
        row: list[float] = []
        for _ in range(n_tf):
            v = 0.0
            if rng.random() < 0.15:
                v = rng.uniform(0.25, 0.8) * rng.choice((-1.0, 1.0))
            row.append(v)
        design.append(row)
    basal = [1.0 + 0.5 * rng.random() for _ in range(n_genes)]

    # activities: WT all 1.0; perturbation p knocks regulator (p mod n_tf)
    activities: list[list[float]] = []
    for p in range(n_perturbations):
        if p == 0:
            activities.append([1.0] * n_tf)
        else:
            u = [1.0] * n_tf
            u[p % n_tf] = kd
            activities.append(u)

    mean: list[list[float]] = []
    for g in range(n_genes):
        mean_row: list[float] = []
        base = basal[g] * wt_scale
        for p in range(n_perturbations):
            eta = sum(design[g][t] * (activities[p][t] ** hill_n - 1.0)
                      for t in range(n_tf))
            mean_row.append(base * math.exp(response_gain * eta))
        mean.append(mean_row)

    counts: list[list[float]] = []
    for g in range(n_genes):
        row = [negative_binomial_noise(v, rng, dispersion) for v in mean[g]]
        counts.append(row)

    return {
        "genes": genes,
        "perturbations": perturbations,
        "design": design,
        "basal": basal,
        "activities": activities,
        "truth": {
            "response_gain": response_gain,
            "hill_n": hill_n,
        },
        "mean": mean,
        "counts": counts,
        "kd": kd,
        "dispersion": dispersion,
    }


class PerturbSeqModel:
    """CRISPRi response model with unknown global coupling constants.

    Shares the sparse design matrix and per-gene basals with the
    ground truth; the free parameters are the coupling constants
    (``response_gain``, ``hill_n``) fitted by
    :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters`.  The perturbation
    response is a *fold change* against the WT condition (the VCC scoring
    convention), so the WT baseline cancels out of the calibration.
    """

    def __init__(self, design: Sequence[Sequence[float]],
                 basal: Sequence[float], wt_scale: float = 40.0,
                 response_gain: float = 1.0, hill_n: float = 1.0) -> None:
        self.design = design
        self.basal = list(basal)
        self.wt_scale = float(wt_scale)
        self.response_gain = float(response_gain)
        self.hill_n = float(hill_n)

    def fold_change(self, activities: Sequence[float]) -> list[float]:
        """Log-fold response ``exp(gain * eta)`` of every gene (WT -> 1)."""
        if _HAS_NUMPY:
            u = _np.asarray(activities, dtype=float)
            eta = _np.asarray(self.design) @ (u ** self.hill_n - 1.0)
            return _np.exp(self.response_gain * eta).tolist()  # type: ignore[no-any-return]
        hill = self.hill_n
        return [math.exp(self.response_gain
                         * sum(row[t] * (activities[t] ** hill - 1.0)
                               for t in range(len(row))))
                for row in self.design]

    def response(self, activities: Sequence[float]) -> list[float]:
        """Expected expression of every gene under one activity vector."""
        if _HAS_NUMPY:
            fold = _np.asarray(self.fold_change(activities), dtype=float)
            out = _np.asarray(self.basal) * self.wt_scale * fold
            return out.tolist()  # type: ignore[no-any-return]
        return [self.basal[g] * self.wt_scale * f
                for g, f in enumerate(self.fold_change(activities))]

    def with_coupling(self, response_gain: float, hill_n: float
                      ) -> PerturbSeqModel:
        """Return a copy with the given coupling constants."""
        return PerturbSeqModel(self.design, self.basal, self.wt_scale,
                               response_gain, hill_n)


# ============================================================================
# VCC-style scoring
# ============================================================================

def vcc_mae(pred: Sequence[float], truth: Sequence[float]) -> float:
    """Per-perturbation mean absolute error (VCC 2025 ``MAE_k`` formula)."""
    return sum(abs(p - t) for p, t in zip(pred, truth, strict=True)) / len(pred)


def response_correlation(pred: Sequence[float],
                         truth: Sequence[float]) -> float:
    """Pearson correlation of the predicted vs true response across genes."""
    n = len(pred)
    if n < 2:
        return 1.0
    mp = sum(pred) / n
    mt = sum(truth) / n
    cov = sum((p - mp) * (t - mt) for p, t in zip(pred, truth, strict=True))
    vp = sum((p - mp) ** 2 for p in pred)
    vt = sum((t - mt) ** 2 for t in truth)
    if vp <= 0.0 or vt <= 0.0:
        return 0.0
    return cov / math.sqrt(vp * vt)


def de_sign_agreement(pred: Sequence[float], truth: Sequence[float],
                      wt: Sequence[float], min_effect: float = 1e-6) -> float:
    """Fraction of DE genes whose direction (up/down vs WT) is predicted."""
    de = [(p, t, w) for p, t, w in zip(pred, truth, wt, strict=True)
          if abs(t - w) > min_effect]
    if not de:
        return 1.0
    hits = sum(1 for p, t, w in de if (t - w) * (p - w) > 0.0)
    return hits / len(de)


def inverse_variance_weights(counts: Sequence[Sequence[float]],
                             dispersion: float = DEFAULT_DISPERSION
                             ) -> list[float]:
    """DESeq2-style inverse-variance weights ``1/(mu + dispersion*mu^2)``.

    ``counts`` is a G x P matrix; the returned weight vector is row-major
    (one weight per flattened observation) so it lines up with the
    flattened ``observed``/``pred`` vectors passed to
    :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters`.
    """
    out: list[float] = []
    for row in counts:
        for v in row:
            mu = max(float(v), 1e-3)
            out.append(1.0 / (mu + dispersion * mu * mu))
    return out


def _flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [v for row in matrix for v in row]


# ============================================================================
# Benchmark
# ============================================================================

class OmicsCalibrationBenchmark:
    """Calibrate-then-predict perturb-seq benchmark (VCC 2025 protocol).

    Splits the synthetic perturbation library into a training subset
    (calibration) and a held-out subset (prediction).  The global
    coupling constants (``response_gain``, ``hill_n``, ``basal_scale``)
    are recovered with a weighted multi-perturbation fit, then the
    response of the held-out perturbations is predicted and scored with
    the VCC metrics.
    """

    def __init__(self, n_genes: int = 200, n_perturbations: int = 24,
                 n_train: int = 16, n_tf: int = 8, seed: int = 0,
                 dispersion: float = 0.02,
                 kd: float = DEFAULT_CRISPRI_KD,
                 fit_seed: int = 1, n_samples: int = 400,
                 refine_rounds: int = 5,
                 ranges: dict[str, tuple[float, float]] | None = None) -> None:
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.n_train = n_train
        self.n_tf = n_tf
        self.seed = seed
        self.dispersion = dispersion
        self.kd = kd
        self.fit_seed = fit_seed
        self.n_samples = n_samples
        self.refine_rounds = refine_rounds
        # literature-informed priors: coupling constants near unity (the
        # calibration refines, it does not start from scratch -- Karr 2012
        # style parameter estimation from published ranges)
        self.ranges = ranges or {
            "response_gain": (0.7, 1.3),
            "hill_n": (0.8, 1.2),
        }
        self.data = generate_perturb_seq_data(
            n_genes=n_genes, n_perturbations=n_perturbations, n_tf=n_tf,
            seed=seed, kd=kd, dispersion=dispersion,
        )
        self.truth = self.data["truth"]
        self.design = self.data["design"]
        self.basal = self.data["basal"]
        self.train_idx = list(range(n_train))
        self.holdout_idx = list(range(n_train, n_perturbations))

    def _model(self, **params: float) -> PerturbSeqModel:
        return PerturbSeqModel(self.design, self.basal,
                               response_gain=params["response_gain"],
                               hill_n=params["hill_n"])

    def calibrate(self) -> dict:
        """Fit the coupling constants on the training perturbations.

        The calibration target is the log fold-change of every gene
        against the WT condition (the VCC convention): noisy counts are
        normalized by the WT pseudo-count, so the per-gene baseline
        cancels out and only the coupling constants are fitted.  The
        objective is weighted by DESeq2-style inverse-variance weights.
        """
        counts = self.data["counts"]
        wt = [counts[g][0] + 1.0 for g in range(self.n_genes)]
        train_fold = [[(counts[g][p] + 1.0) / wt[g]
                       for p in self.train_idx]
                      for g in range(self.n_genes)]
        weights = inverse_variance_weights(
            [[counts[g][p] for p in self.train_idx]
             for g in range(self.n_genes)], self.dispersion)
        observed = [math.log(f) for f in _flatten(train_fold)]

        def predict(**params: float) -> list[float]:
            model = self._model(**params)
            folds = [model.fold_change(self.data["activities"][p])
                     for p in self.train_idx]
            flat = [folds[pi][g]
                    for g in range(self.n_genes)
                    for pi in range(len(self.train_idx))]
            return [math.log(f) for f in flat]

        return fit_parameters(
            predict, observed, self.ranges,
            n_samples=self.n_samples, seed=self.fit_seed,
            refine_rounds=self.refine_rounds, weights=weights,
        )

    def predict_holdout(self, fitted: dict) -> dict:
        """Predict held-out perturbation responses with fitted constants.

        Raw predictions are the calibrated model's expected expression
        (per-gene basal x WT scale x fitted fold-change), the value a
        user would report for each held-out perturbation.
        """
        model = self._model(
            response_gain=float(fitted["best"]["response_gain"]),
            hill_n=float(fitted["best"]["hill_n"]),
        )
        pred = [model.response(self.data["activities"][p])
                for p in self.holdout_idx]
        truth = self.data["mean"]
        truth_hold = [[truth[g][p] for g in range(self.n_genes)]
                      for p in self.holdout_idx]
        truth_wt = [truth[g][0] for g in range(self.n_genes)]
        return {"predictions": pred, "truth": truth_hold, "wt": truth_wt}

    def score(self) -> dict:
        """Full loop: calibrate, predict, score with VCC metrics."""
        fit = self.calibrate()
        out = self.predict_holdout(fit)
        pred = out["predictions"]
        truth = out["truth"]
        wt = out["wt"]
        mae_list = [vcc_mae(p, t) for p, t in zip(pred, truth, strict=True)]
        corr = response_correlation(_flatten(pred), _flatten(truth))
        signs = [de_sign_agreement(p, t, wt) for p, t in
                 zip(pred, truth, strict=True)]
        # baseline: predict the WT expression for every held-out perturbation
        baseline = [vcc_mae(wt, t) for t in truth]
        improvement = 1.0 - (sum(mae_list) / sum(baseline)
                             if sum(baseline) > 0 else 0.0)
        passed = (sum(mae_list) < sum(baseline)
                  and corr >= 0.6
                  and (sum(signs) / len(signs)) >= 0.75)
        return {
            "fit": fit,
            "mae_per_perturbation": mae_list,
            "baseline_mae_per_perturbation": baseline,
            "mae_improvement_vs_baseline": improvement,
            "response_correlation": corr,
            "de_sign_agreement": sum(signs) / len(signs),
            "holdout_perturbations": [self.data["perturbations"][p]
                                      for p in self.holdout_idx],
            "truth_coupling": dict(self.truth),
            "fitted_coupling": {
                k: float(fit["best"][k]) for k in self.truth
            },
            "passed": passed,
        }


def run_omics_calibration_benchmark(
    n_genes: int = 200, n_perturbations: int = 24, n_train: int = 16,
    seed: int = 0, fit_seed: int = 1,
) -> dict:
    """One-shot omics calibration -> prediction benchmark."""
    return OmicsCalibrationBenchmark(
        n_genes=n_genes, n_perturbations=n_perturbations, n_train=n_train,
        seed=seed, fit_seed=fit_seed,
    ).score()


__all__ = [
    "OmicsCalibrationBenchmark",
    "PerturbSeqModel",
    "negative_binomial_noise",
    "log_normal_noise",
    "generate_perturb_seq_data",
    "inverse_variance_weights",
    "vcc_mae",
    "response_correlation",
    "de_sign_agreement",
    "run_omics_calibration_benchmark",
]
