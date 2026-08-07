"""Gray-Scott reaction-diffusion model.

U' = U + (Du*∇²U - U*V² + F*(1-U))
V' = V + (Dv*∇²V + U*V² - (F+k)*V)

The default parameters F=0.035, k=0.065 generate Pearson 1993 spots (mitosis) patterns.
"""
from __future__ import annotations

import random


class GrayScott:
    """Two-dimensional reaction-diffusion field."""

    def __init__(self, n: int = 32, F: float = 0.035, k: float = 0.065,
                 Du: float = 0.16, Dv: float = 0.08, seed: int = 42):
        self.n = n
        self.F = F
        self.k = k
        self.Du = Du
        self.Dv = Dv
        self.u: list[list[float]] = [[1.0] * n for _ in range(n)]
        self.v: list[list[float]] = [[0.0] * n for _ in range(n)]
        self._seed(seed)

    @classmethod
    def from_preset(cls, preset_name: str, n: int = 32,
                    seed: int = 42) -> GrayScott:
        """Construct with the empirically measured preset parameters from Pearson 1993.

        preset_name: see helixlang.bio_data.GRAY_SCOTT_PRESETS
        """
        from helixlang.bio_data import get_gray_scott_preset
        p = get_gray_scott_preset(preset_name)
        return cls(n=n, F=p.F, k=p.k, Du=p.Du, Dv=p.Dv, seed=seed)

    def _seed(self, seed: int) -> None:
        n = self.n
        # Central square perturbation
        for i in range(n // 2 - 3, n // 2 + 3):
            for j in range(n // 2 - 3, n // 2 + 3):
                if 0 <= i < n and 0 <= j < n:
                    self.u[i][j] = 0.5
                    self.v[i][j] = 0.25
        # Random perturbation points
        rng = random.Random(seed)
        for _ in range(20):
            i = rng.randrange(2, n - 2)
            j = rng.randrange(2, n - 2)
            self.v[i][j] = 1.0
            self.u[i][j] = 0.5

    @staticmethod
    def _lap(f: list[list[float]], i: int, j: int) -> float:
        """Standard 5-point Laplacian operator (average of the 4 neighbors minus the center).

        Uses the Karl Sims / Pearson discrete convention: ``(Σ_neighbors)/4 - center``,
        i.e. ``(sum - 4*center) * 0.25``. Du/Dv are the nominal diffusion coefficients;
        the effective diffusion = Du * 0.25 (CFL: Du*0.25*4 = Du < 1 is stable, Du=0.16 ✓).
        """
        return (f[i - 1][j] + f[i + 1][j] + f[i][j - 1] + f[i][j + 1]
                - 4.0 * f[i][j]) * 0.25

    def step(self) -> None:
        """Advance one step."""
        n = self.n
        nu = [row[:] for row in self.u]
        nv = [row[:] for row in self.v]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                lu = self._lap(self.u, i, j)
                lv = self._lap(self.v, i, j)
                uij = self.u[i][j]
                vij = self.v[i][j]
                uvv = uij * vij * vij
                nu[i][j] = uij + (self.Du * lu - uvv + self.F * (1 - uij))
                nv[i][j] = vij + (self.Dv * lv + uvv - (self.F + self.k) * vij)
        # Clamp to [0, 1] to prevent numerical blow-up
        for i in range(n):
            for j in range(n):
                if nu[i][j] < 0.0:
                    nu[i][j] = 0.0
                elif nu[i][j] > 1.0:
                    nu[i][j] = 1.0
                if nv[i][j] < 0.0:
                    nv[i][j] = 0.0
                elif nv[i][j] > 1.0:
                    nv[i][j] = 1.0
        self.u = nu
        self.v = nv

    def emit(self, i: int, j: int, amount: float = 1.0) -> None:
        """Inject morphogen V at (i, j)."""
        if 0 <= i < self.n and 0 <= j < self.n:
            self.v[i][j] = min(1.0, self.v[i][j] + amount)

    def total_v(self) -> float:
        return sum(sum(row) for row in self.v)
