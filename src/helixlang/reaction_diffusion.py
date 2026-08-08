"""Gray-Scott reaction-diffusion model.

U' = U + (Du*∇²U - U*V² + F*(1-U))
V' = V + (Dv*∇²V + U*V² - (F+k)*V)

The default parameters F=0.035, k=0.065 generate Pearson 1993 spots (mitosis) patterns.

Performance: ``step()`` previously allocated two full-field copies every tick and
looped over every cell in pure Python. It now uses a scratch double-buffer (border
cells only are copied, O(n) instead of O(n²)) and, when ``numpy`` is installed
(the optional ``fast`` extra), a fully vectorized backend. Both paths produce
bit-identical results.
"""
from __future__ import annotations

import random
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an optional dependency
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


class GrayScott:
    """Two-dimensional reaction-diffusion field."""

    def __init__(self, n: int = 32, F: float = 0.035, k: float = 0.065,
                 Du: float = 0.16, Dv: float = 0.08, seed: int = 42):
        self.n = n
        self.F = F
        self.k = k
        self.Du = Du
        self.Dv = Dv
        if _HAS_NUMPY:
            self.u: Any = np.full((n, n), 1.0)
            self.v: Any = np.zeros((n, n))
            self._scratch_u = None
            self._scratch_v = None
        else:
            self.u = [[1.0] * n for _ in range(n)]
            self.v = [[0.0] * n for _ in range(n)]
            # Scratch double-buffers (same shape); avoid the per-step O(n²) copy.
            self._scratch_u = [[0.0] * n for _ in range(n)]
            self._scratch_v = [[0.0] * n for _ in range(n)]
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
        """Advance one step.

        The backend (vectorized numpy vs pure-Python scratch double-buffer) is
        fixed at construction time by the field's storage type.
        """
        if _HAS_NUMPY and isinstance(self.u, np.ndarray):
            self._step_numpy()
        else:
            self._step_py()

    def _step_py(self) -> None:
        """Pure-Python step using scratch double-buffers (no O(n²) copies)."""
        n = self.n
        u, v = self.u, self.v
        if self._scratch_u is None or self._scratch_v is None:
            self._scratch_u = [[0.0] * n for _ in range(n)]
            self._scratch_v = [[0.0] * n for _ in range(n)]
        nu, nv = self._scratch_u, self._scratch_v
        F, k, Du, Dv = self.F, self.k, self.Du, self.Dv
        # Borders are untouched by the interior update; copy them into the
        # scratch buffers so the swap yields a complete field.
        nu[0] = u[0][:]
        nu[n - 1] = u[n - 1][:]
        nv[0] = v[0][:]
        nv[n - 1] = v[n - 1][:]
        for i in range(1, n - 1):
            nui = nu[i]
            nvi = nv[i]
            ui = u[i]
            vi = v[i]
            nui[0] = ui[0]
            nui[n - 1] = ui[n - 1]
            nvi[0] = vi[0]
            nvi[n - 1] = vi[n - 1]
            um = u[i - 1]
            up = u[i + 1]
            vm = v[i - 1]
            vp = v[i + 1]
            for j in range(1, n - 1):
                lu = (um[j] + up[j] + ui[j - 1] + ui[j + 1]
                      - 4.0 * ui[j]) * 0.25
                lv = (vm[j] + vp[j] + vi[j - 1] + vi[j + 1]
                      - 4.0 * vi[j]) * 0.25
                uij = ui[j]
                vij = vi[j]
                uvv = uij * vij * vij
                nui[j] = uij + (Du * lu - uvv + F * (1.0 - uij))
                nvi[j] = vij + (Dv * lv + uvv - (F + k) * vij)
                # Clamp to [0, 1] to prevent numerical blow-up
                if nui[j] < 0.0:
                    nui[j] = 0.0
                elif nui[j] > 1.0:
                    nui[j] = 1.0
                if nvi[j] < 0.0:
                    nvi[j] = 0.0
                elif nvi[j] > 1.0:
                    nvi[j] = 1.0
        self.u, self._scratch_u = nu, u
        self.v, self._scratch_v = nv, v

    def _step_numpy(self) -> None:
        """Vectorized step (``numpy`` backend). Identical results to ``_step_py``."""
        u = self.u
        v = self.v
        # Discrete 5-point Laplacian (interior only; borders stay zero here and
        # are restored below, matching the pure-Python "borders unchanged" rule).
        lap_u = np.zeros_like(u)
        lap_v = np.zeros_like(v)
        lap_u[1:-1, 1:-1] = (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
            - 4.0 * u[1:-1, 1:-1]) * 0.25
        lap_v[1:-1, 1:-1] = (
            v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:]
            - 4.0 * v[1:-1, 1:-1]) * 0.25
        uvv = u * v * v
        self.u = np.clip(
            u + (self.Du * lap_u - uvv + self.F * (1.0 - u)), 0.0, 1.0)
        self.v = np.clip(
            v + (self.Dv * lap_v + uvv - (self.F + self.k) * v), 0.0, 1.0)
        # Preserve border cells exactly (they are never updated by the Laplacian).
        self.u[0, :] = u[0, :]
        self.u[-1, :] = u[-1, :]
        self.u[:, 0] = u[:, 0]
        self.u[:, -1] = u[:, -1]
        self.v[0, :] = v[0, :]
        self.v[-1, :] = v[-1, :]
        self.v[:, 0] = v[:, 0]
        self.v[:, -1] = v[:, -1]

    def emit(self, i: int, j: int, amount: float = 1.0) -> None:
        """Inject morphogen V at (i, j)."""
        if 0 <= i < self.n and 0 <= j < self.n:
            self.v[i][j] = min(1.0, self.v[i][j] + amount)

    def total_v(self) -> float:
        return float(sum(sum(row) for row in self.v))
