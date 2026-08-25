"""Multi-Scale Reduced-Order Models (MS-ROM) for organ spatial effects (doc/32 §7.5).

Captures 80-90% of spatial heterogeneity (drug concentration gradients, perfusion
heterogeneity, lobular zonation) at ODE computational cost using Proper Orthogonal
Decomposition (POD) modes.

Key spatial modes:
  φ₁: Mean concentration (well-mixed — current model)
  φ₂: Portal-central gradient (liver zonation)
  φ₃: Cortex-medulla gradient (kidney)
  φ₄: Periportal-pericentral gradient (liver drug metabolism zones)

References:
- Multiscale Liver Virtual Twin, npj Digital Medicine 2025
- Hypoxia surrogate 0D-3D-1D coupling, PMC 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PODMode:
    """A single POD spatial mode.

    phi(x): spatial shape function (evaluated at N_grid points)
    amplitude: time-varying coefficient q(t)
    """

    index: int
    label: str
    spatial_values: list[float]  # φᵢ at each grid point
    energy_fraction: float  # fraction of total variance captured
    amplitude: float = 1.0  # qᵢ(t) — current amplitude


@dataclass
class ReducedOrderOrgan:
    """Reduced-order organ model with POD spatial modes.

    Replaces the lumped ODE with a modal expansion:
        C(x,t) ≈ Σᵢ φᵢ(x) · qᵢ(t)

    Each mode has its own ODE:
        dqᵢ/dt = -λᵢ·qᵢ + Rᵢ(Q) + fᵢ(t)

    where λᵢ is the mode decay rate and Rᵢ captures nonlinear reactions.
    """

    organ: str
    modes: list[PODMode]
    decay_rates: list[float]  # λᵢ for each mode
    perfusion_rate: float  # organ blood flow (L/h)
    volume: float  # organ volume (L)
    baseline_concentration: float = 0.0

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def total_energy_captured(self) -> float:
        return sum(m.energy_fraction for m in self.modes)

    def evaluate_spatial(self, grid_position: float) -> float:
        """Evaluate concentration at a spatial position: C(x,t) = Σ φᵢ(x)·qᵢ(t)."""
        result = 0.0
        for mode in self.modes:
            idx = min(int(grid_position * (len(mode.spatial_values) - 1)),
                      len(mode.spatial_values) - 1)
            result += mode.spatial_values[idx] * mode.amplitude
        return result

    def step(self, dt: float, drug_input: float = 0.0) -> None:
        """Advance all modal amplitudes by one time step.

        Args:
            dt: time step (hours)
            drug_input: drug input rate (µmol/h) — distributed across modes
        """
        for i, mode in enumerate(self.modes):
            decay = self.decay_rates[i]
            # reaction term: linear decay + input distribution
            input_share = drug_input * (1.0 / (i + 1))  # higher modes get less input
            dx_dt = -decay * mode.amplitude + input_share / self.volume
            mode.amplitude = max(0.0, mode.amplitude + dx_dt * dt)

    def get_mean_concentration(self) -> float:
        """Volume-averaged concentration (equivalent to lumped model)."""
        if not self.modes:
            return 0.0
        return self.modes[0].amplitude

    def get_gradient(self) -> float:
        """Portal-central concentration gradient (mode 2 amplitude)."""
        if len(self.modes) < 2:
            return 0.0
        return self.modes[1].amplitude


class PODModeGenerator:
    """Generates pre-computed POD modes for organ spatial models.

    In production, modes come from offline FEM simulation + SVD.
    Here we use analytical approximations based on known organ physiology.
    """

    N_GRID = 20  # spatial discretization points

    @staticmethod
    def generate_liver_modes(n_modes: int = 4) -> ReducedOrderOrgan:
        """Generate POD modes for liver (portal-central zonation).

        Mode 1: uniform (well-mixed)
        Mode 2: portal-central gradient (periportal > pericentral)
        Mode 3: metabolic zonation (CYP-rich periportal zone)
        Mode 4: pericentral glutamine synthetase zone
        """
        N = PODModeGenerator.N_GRID
        grid = [i / (N - 1) for i in range(N)]  # 0=portal, 1=central

        modes: list[PODMode] = []
        energies = [0.60, 0.20, 0.12, 0.08]

        for i in range(min(n_modes, 4)):
            if i == 0:
                # Mode 1: uniform
                spatial = [1.0] * N
                label = "mean_concentration"
            elif i == 1:
                # Mode 2: portal-central gradient (linear + noise)
                spatial = [1.0 - 0.4 * x + 0.05 * math.sin(2 * math.pi * x) for x in grid]
                label = "portal_central_gradient"
            elif i == 2:
                # Mode 3: periportal metabolic zone
                spatial = [math.exp(-3.0 * x ** 2) for x in grid]
                label = "periportal_metabolic"
            else:
                # Mode 4: pericentral zone
                spatial = [math.exp(-3.0 * (1 - x) ** 2) for x in grid]
                label = "pericentral_gs"

            modes.append(PODMode(
                index=i,
                label=label,
                spatial_values=spatial,
                energy_fraction=energies[i],
            ))

        return ReducedOrderOrgan(
            organ="liver",
            modes=modes,
            decay_rates=[0.15, 0.4, 0.8, 0.6],
            perfusion_rate=15.0,  # ~1.5 L/min
            volume=1.5,
        )

    @staticmethod
    def generate_kidney_modes(n_modes: int = 3) -> ReducedOrderOrgan:
        """Generate POD modes for kidney (cortex-medulla gradient).

        Mode 1: uniform
        Mode 2: cortex-medulla gradient (cortex high, medulla low for most solutes)
        Mode 3: countercurrent multiplication pattern
        """
        N = PODModeGenerator.N_GRID
        grid = [i / (N - 1) for i in range(N)]

        modes: list[PODMode] = []
        energies = [0.55, 0.25, 0.15]

        for i in range(min(n_modes, 3)):
            if i == 0:
                spatial = [1.0] * N
                label = "mean_concentration"
            elif i == 1:
                # Cortex-medulla gradient
                spatial = [1.0 - 0.6 * x for x in grid]
                label = "cortex_medulla_gradient"
            else:
                # Countercurrent multiplication (U-shaped)
                spatial = [1.0 - 0.3 * math.sin(math.pi * x) for x in grid]
                label = "countercurrent"

            modes.append(PODMode(
                index=i,
                label=label,
                spatial_values=spatial,
                energy_fraction=energies[i],
            ))

        return ReducedOrderOrgan(
            organ="kidney",
            modes=modes,
            decay_rates=[0.2, 0.5, 0.7],
            perfusion_rate=7.2,  # ~1.2 L/min
            volume=0.3,
        )

    @staticmethod
    def generate_brain_modes(n_modes: int = 3) -> ReducedOrderOrgan:
        """Generate POD modes for brain (BBB penetration gradient).

        Mode 1: uniform (well-mixed parenchyma)
        Mode 2: cortical-subcortical gradient
        Mode 3: BBB penetration front
        """
        N = PODModeGenerator.N_GRID
        grid = [i / (N - 1) for i in range(N)]

        modes: list[PODMode] = []
        energies = [0.65, 0.20, 0.10]

        for i in range(min(n_modes, 3)):
            if i == 0:
                spatial = [1.0] * N
                label = "mean_concentration"
            elif i == 1:
                spatial = [1.0 - 0.3 * x ** 1.5 for x in grid]
                label = "cortical_subcortical"
            else:
                spatial = [math.exp(-5.0 * x) for x in grid]
                label = "bbb_penetration"

            modes.append(PODMode(
                index=i,
                label=label,
                spatial_values=spatial,
                energy_fraction=energies[i],
            ))

        return ReducedOrderOrgan(
            organ="brain",
            modes=modes,
            decay_rates=[0.12, 0.35, 0.9],
            perfusion_rate=0.5,
            volume=1.2,
        )

    @staticmethod
    def generate_all() -> dict[str, ReducedOrderOrgan]:
        """Generate reduced-order models for all major organs."""
        return {
            "liver": PODModeGenerator.generate_liver_modes(),
            "kidney": PODModeGenerator.generate_kidney_modes(),
            "brain": PODModeGenerator.generate_brain_modes(),
        }
