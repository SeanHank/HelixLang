"""L2 reference-equivalence calibration for ChEMBL binding (doc/32 §6.2, §7.1).

The layer-1 uncertainty (sigma_1) of the multi-scale calibration cascade is
calibrated against the vendored ChEMBL binding snapshot: a leave-one-out
cross-validation in which every snapshot drug is re-predicted from the
remaining reference set by the fingerprint kNN resolver
(:func:`~helixlang.plugins.human.proteome_binding.knn_resolve`) and its
predicted Kd compared to the ChEMBL reference value.

This is the doc/41 Level-2 (reference-implementation validation) case for the
binding layer: the vendored snapshot is the *named reference implementation*
with a golden hash; the kNN resolver plus cascade form the target
implementation.  The pass criterion enforces the doc/32 L2 magnitude band
(median fold error <= 2.0, i.e. "magnitude" accuracy, tightening toward the
shoreline +/-10% once cascade layers compound).

References:
- doc/32 §6.2 L2 (magnitude), §7.1 (sigma_1 vs ChEMBL)
- doc/41 §3 L0-L5 taxonomy, L2 = reference-implementation validation
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from helixlang.plugins.human.calibration_cascade import CalibrationCascade
from helixlang.plugins.human.data.chembl_binding import (
    CHEMBL_DRUG_BINDINGS,
    CHEMBL_DRUG_SMILES,
)
from helixlang.plugins.human.proteome_binding import (
    ProteomeBindingCascade,
)

#: doc/32 L2 magnitude band: median |fold error| <= 2.0 passes.
L2_MAX_MEDIAN_FOLD = 2.0
#: share of comparisons that must land within the L2 band.
L2_MIN_FRACTION = 0.5


@dataclass
class ChemBLObservation:
    """One predicted-vs-observed binding pairing for a snapshot drug/target."""

    drug: str
    target: str
    reference_kd_um: float
    predicted_kd_um: float
    fold_error: float  # predicted / reference
    log10_fold: float


@dataclass
class ChemBLL2Report:
    """Equivalence report of the kNN resolver against the ChEMBL snapshot."""

    observations: list[ChemBLObservation] = field(default_factory=list)
    missing_targets: list[tuple[str, str]] = field(default_factory=list)
    golden_hash: str = ""
    sigma_1: float = 0.0          # std of log10 fold error (layer-1, ChEMBL)
    median_fold_error: float = 0.0
    max_fold_error: float = 0.0
    fraction_within_band: float = 0.0
    passed: bool = False

    @property
    def n_reference_pairs(self) -> int:
        return sum(len(v) for v in CHEMBL_DRUG_BINDINGS.values())


@dataclass
class ChemBLCascadeResult:
    """Calibrated layer-1 prediction width for a Kd (log10 units)."""

    predicted_log10_kd: float
    sigma_1: float
    ci_90_lower: float
    ci_90_upper: float
    uncertainty_ratio: float


def snapshot_golden_hash() -> str:
    """Deterministic SHA-256 over the vendored ChEMBL snapshot.

    The hash is the doc/41 L2 ``golden_hash`` for the reference
    implementation: any edit to the snapshot changes it, so equivalence
    results stay reproducible.
    """
    payload = json.dumps(
        {"bindings": CHEMBL_DRUG_BINDINGS, "smiles": CHEMBL_DRUG_SMILES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ChemBLL2Calibration:
    """Leave-one-out L2 calibration of the binding layer against ChEMBL.

    ``sigma_1`` (log10-Kd units) is the calibrated layer-1 uncertainty for
    the multi-scale cascade (doc/32 §7.1); the same observation set feeds the
    ``CalibrationCascade`` layer 0 so downstream ``predict`` calls carry the
    ChEMBL-calibrated spread.
    """

    def __init__(
        self,
        cascade: ProteomeBindingCascade | None = None,
        knn_k: int = 3,
    ) -> None:
        self.knn_k = knn_k
        self.cascade = CalibrationCascade()
        self._cascade = cascade

    @property
    def _resolver(self) -> ProteomeBindingCascade:
        if self._cascade is None:
            self._cascade = ProteomeBindingCascade(knn_k=self.knn_k)
        return self._cascade

    def run_loo(self) -> ChemBLL2Report:
        """Leave-one-out equivalence run over the ChEMBL snapshot."""
        observations: list[ChemBLObservation] = []
        missing: list[tuple[str, str]] = []
        for drug in sorted(CHEMBL_DRUG_BINDINGS):
            resolver = ProteomeBindingCascade(knn_k=self.knn_k, exclude={drug})
            profile = resolver.screen_drug(
                drug, CHEMBL_DRUG_SMILES[drug], drug_conc_um=1.0)
            predicted = {b.target: b.kd_um for b in profile.bindings}
            for target, params in CHEMBL_DRUG_BINDINGS[drug].items():
                ref = params["kd_um"]
                pred = predicted.get(target)
                if pred is None or pred <= 0.0:
                    missing.append((drug, target))
                    continue
                if ref <= 0.0:
                    missing.append((drug, target))
                    continue
                fold = pred / ref
                observations.append(ChemBLObservation(
                    drug=drug,
                    target=target,
                    reference_kd_um=ref,
                    predicted_kd_um=pred,
                    fold_error=fold,
                    log10_fold=math.log10(fold),
                ))
        report = self._compile(observations, missing)
        self._fit_cascade(report.observations)
        return report

    def _compile(
        self,
        observations: list[ChemBLObservation],
        missing: list[tuple[str, str]],
    ) -> ChemBLL2Report:
        if not observations:
            return ChemBLL2Report(
                observations=[],
                missing_targets=missing,
                golden_hash=snapshot_golden_hash(),
                passed=False,
            )
        log10_folds = sorted(abs(o.log10_fold) for o in observations)
        median = log10_folds[len(log10_folds) // 2]
        if len(log10_folds) % 2 == 0:
            median = 0.5 * (log10_folds[len(log10_folds) // 2 - 1] + median)
        sigma1 = math.sqrt(
            sum(o.log10_fold ** 2 for o in observations) / len(observations))
        frac = sum(1 for o in observations
                   if abs(o.log10_fold) <= math.log10(L2_MAX_MEDIAN_FOLD)
                   ) / len(observations)
        return ChemBLL2Report(
            observations=observations,
            missing_targets=missing,
            golden_hash=snapshot_golden_hash(),
            sigma_1=sigma1,
            median_fold_error=10.0 ** median,
            max_fold_error=10.0 ** max(abs(o.log10_fold) for o in observations),
            fraction_within_band=frac,
            passed=(10.0 ** median <= L2_MAX_MEDIAN_FOLD
                    and frac >= L2_MIN_FRACTION),
        )

    def _fit_cascade(self, observations: list[ChemBLObservation]) -> None:
        """Calibrate cascade layer 0 (SMILES~>binding) on log10 scale."""
        for obs in observations:
            self.cascade.calibrate_layer(
                0, predicted=math.log10(obs.predicted_kd_um),
                observed=math.log10(obs.reference_kd_um))

    def predict_sigma1(self, predicted_kd_um: float) -> ChemBLCascadeResult:
        """Calibrated layer-1 prediction interval for a Kd (µM)."""
        log10_kd = math.log10(max(predicted_kd_um, 1e-12))
        result = self.cascade.predict(0, x=log10_kd)
        sigma1 = self.cascade.layers[0].sigma_posterior
        ci = 1.645 * sigma1
        return ChemBLCascadeResult(
            predicted_log10_kd=result.predicted_value,
            sigma_1=sigma1,
            ci_90_lower=10.0 ** (result.predicted_value - ci),
            ci_90_upper=10.0 ** (result.predicted_value + ci),
            uncertainty_ratio=10.0 ** ci,
        )


__all__ = [
    "ChemBLObservation",
    "ChemBLL2Report",
    "ChemBLCascadeResult",
    "ChemBLL2Calibration",
    "snapshot_golden_hash",
    "L2_MAX_MEDIAN_FOLD",
    "L2_MIN_FRACTION",
]
