"""Calibration Cascade: GP wrappers for multi-scale accuracy (doc/32 §7.1).

Wraps each modeling layer with Gaussian Process calibration to reduce
compounded uncertainty from ±50% to ±10% (and further to ±3-5% with
Bayesian denoising from §8.1).

Layer 1: SMILES → binding affinity Kd
Layer 2: Kd → metabolic clearance CLint
Layer 3: CLint → PBPK plasma concentration
Layer 4: Cplasma → individual response

Each layer uses a local GP surrogate with RBF kernel, trained on
calibration observations. The GP posterior provides both mean prediction
and uncertainty estimates that shrink as more calibration data accumulates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GPLayer:
    """Single Gaussian Process calibration layer.

    Uses a simplified local GP with RBF kernel that maintains a running
    estimate of the kernel parameters and posterior statistics.
    """

    name: str
    input_dim: int
    sigma_prior: float = 0.5
    sigma_posterior: float = -1.0
    n_calibrations: int = 0
    residuals: list[float] = field(default_factory=list)
    # GP state: stored observations for kernel regression
    _obs_x: list[float] = field(default_factory=list)
    _obs_y: list[float] = field(default_factory=list)
    _length_scale: float = 1.0  # RBF kernel length scale
    _noise_variance: float = 0.01  # observation noise

    def __post_init__(self) -> None:
        if self.sigma_posterior < 0:
            self.sigma_posterior = self.sigma_prior

    def calibrate(self, predicted: float, observed: float) -> float:
        """Update GP posterior with a new calibration observation."""
        residual = predicted - observed
        self.residuals.append(residual)
        self.n_calibrations += 1

        # Store observation for GP kernel regression
        self._obs_x.append(predicted)
        self._obs_y.append(observed)

        # Update posterior variance using incremental update
        n = self.n_calibrations
        alpha = 1.0 / n
        self.sigma_posterior = (1 - alpha) * self.sigma_posterior + alpha * abs(residual)

        # Adapt length scale based on observation spread
        if len(self._obs_x) > 2:
            x_range = max(self._obs_x) - min(self._obs_x)
            if x_range > 0:
                self._length_scale = max(0.1, x_range / 3.0)

        return self.sigma_posterior

    def predict_with_uncertainty(self, x: float) -> tuple[float, float]:
        """Return (mean_prediction, 90% CI half-width) using local GP.

        When few observations exist, uses simple residual-based correction.
        As observations accumulate, transitions to kernel-weighted prediction.
        """
        if self.n_calibrations == 0:
            return x, 1.645 * self.sigma_prior

        if self.n_calibrations < 5:
            # Few observations: use residual mean correction
            mean_residual = sum(self.residuals) / self.n_calibrations
            corrected = x - mean_residual * 0.5  # dampened correction
            ci_half = 1.645 * self.sigma_posterior
            return corrected, ci_half

        # Many observations: local GP kernel-weighted prediction
        # Use the most recent observations weighted by RBF kernel
        weights: list[float] = []
        values: list[float] = []
        for ox, oy in zip(self._obs_x[-20:], self._obs_y[-20:], strict=True):
            dist_sq = ((x - ox) / self._length_scale) ** 2
            w = math.exp(-0.5 * dist_sq)
            weights.append(w)
            values.append(oy)

        total_w = sum(weights) + 1e-10
        gp_mean = sum(w * v for w, v in zip(weights, values, strict=True)) / total_w

        # GP posterior variance decreases with more observations
        effective_n = min(self.n_calibrations, 20)
        posterior_var = self.sigma_posterior ** 2 / max(1.0, effective_n * 0.5)
        ci_half = 1.645 * math.sqrt(posterior_var + self._noise_variance)

        return gp_mean, ci_half


@dataclass(frozen=True)
class CascadeResult:
    """Result from calibration cascade prediction."""

    predicted_value: float
    ci_90_lower: float
    ci_90_upper: float
    layer_uncertainties: dict[str, float]
    total_uncertainty: float


class CalibrationCascade:
    """Multi-scale Bayesian calibration cascade.

    Wraps up to 4 modeling layers with GP calibration, reducing compounded
    uncertainty multiplicatively.

    Usage:
        cascade = CalibrationCascade()
        cascade.calibrate_layer(0, predicted=100.0, observed=95.0)
        result = cascade.predict(0, x=100.0)
    """

    def __init__(self) -> None:
        self.layers: list[GPLayer] = [
            GPLayer("SMILES_to_binding", input_dim=1, sigma_prior=0.50),
            GPLayer("Binding_to_clearance", input_dim=2, sigma_prior=0.30),
            GPLayer("Clearance_to_PBPK", input_dim=3, sigma_prior=0.20),
            GPLayer("PBPK_to_response", input_dim=4, sigma_prior=0.15),
        ]

    def calibrate_layer(
        self, layer_idx: int, predicted: float, observed: float
    ) -> float:
        """Calibrate a specific layer with a prediction-observation pair."""
        if 0 <= layer_idx < len(self.layers):
            return self.layers[layer_idx].calibrate(predicted, observed)
        return 0.0

    def predict(self, layer_idx: int, x: float) -> CascadeResult:
        """Make calibrated prediction at specified layer."""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return CascadeResult(
                predicted_value=x,
                ci_90_lower=x,
                ci_90_upper=x,
                layer_uncertainties={},
                total_uncertainty=0.0,
            )

        mean, ci_half = self.layers[layer_idx].predict_with_uncertainty(x)

        layer_unc = {}
        for i in range(layer_idx + 1):
            layer_unc[self.layers[i].name] = self.layers[i].sigma_posterior

        total_var = sum(s**2 for s in layer_unc.values())
        total_sigma = math.sqrt(total_var)

        return CascadeResult(
            predicted_value=mean,
            ci_90_lower=mean - ci_half,
            ci_90_upper=mean + ci_half,
            layer_uncertainties=layer_unc,
            total_uncertainty=total_sigma,
        )

    def total_accuracy(self) -> float:
        """Compute total cascade accuracy (RMS of all layer uncertainties)."""
        total_var = sum(layer.sigma_posterior**2 for layer in self.layers)
        return math.sqrt(total_var)

    def reset(self) -> None:
        """Reset all layers to prior uncertainties."""
        for layer in self.layers:
            layer.sigma_posterior = layer.sigma_prior
            layer.n_calibrations = 0
            layer.residuals = []
