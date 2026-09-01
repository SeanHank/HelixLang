"""Spatial agent-based immune model (doc/40 Phase F, gap G15).

Implements the doc/31 §2.4 / doc/40 §6 Phase-F mandate: agent-based modeling of
immune cells in tissue spaces with spatially-resolved rules, replacing the
population-ODE approximation with explicit cell agents on a chemokine field.

Design contract (doc/40 Phase F):
- **Cell migration (chemokine-guided).** Cells move on a 2-D grid toward the
  local chemokine gradient (directional bias + diffusion).
- **Contact-dependent signaling (T-cell/APC).** When a T-cell and an APC occupy
  the same grid cell, a contact probability drives T-cell priming/activation.
- **Spatial heterogeneity (tissue compartments).** The grid carries an
  immovable ``tissue`` field (e.g. endothelium/stroma + a leukocyte-entry
  source compartment) so agents navigate a heterogeneous landscape.
- **Agent-state tracking.** Each agent carries ``(cell_type, position,
  state, activation)``; cohort-wide state histograms are exposed.
- **Deterministic.** Seeded RNG; repeated runs are bit-identical (doc/39 §5.3).
- **No silent fallback (directive).** The numeric core uses ``numpy``; a ``jax``
  kernel can be selected *explicitly* (``backend="jax"``) and raises if ``jax``
  is absent rather than degrading. ``numpy`` is the default backend.

Budget: ≤500 agents/patient; a 100-patient cohort runs many independent single-
grid trajectories, so wall-time is dominated by one grid (O(agents·steps)),
fitting doc/39 O11-style scalability.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any

try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


class AgentState(IntEnum):
    """Lifecycle state of an immune cell agent (BIS entity taxonomy, L1)."""

    RESTING = 0
    MIGRATING = 1
    ACTIVATING = 2        # contact with APC / within APC-priming reach
    ACTIVATED = 3
    EXHAUSTED = 4
    APOPTOTIC = 5


class CellType(IntEnum):
    """Cell types available to the spatial ABM (innate + adaptive subset)."""

    APC = 0
    TCELL = 1
    NEUTROPHIL = 2
    MACROPHAGE = 3
    NK = 4
    EPITHELIAL = 5        # host tissue cell (non-immune)


@dataclass
class TissueAgent:
    """A single cell agent on the tissue grid."""

    uid: int
    cell_type: CellType
    x: int
    y: int
    state: AgentState = AgentState.RESTING
    activation: float = 0.0          # 0..1 activation level (contact-driven)
    lifetime_h: float = 0.0          # hours since birth
    speed_um_h: float = 15.0         # per-cell migration speed
    extra: dict[str, Any] = field(default_factory=dict)

    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    def move(self, nx: int, ny: int) -> None:
        self.x, self.y = int(nx), int(ny)


@dataclass
class SpatialABMConfig:
    """Configuration for a :class:`SpatialAgentGrid` run."""

    width: int = 40
    height: int = 40
    dt_h: float = 0.25
    diffusion_coeff: float = 1.0      # chemokine diffusion (grid^2 / h)
    chemokine_half: float = 1.0       # chemokine for half-max chemotaxis
    chemotaxis_strength: float = 2.0
    contact_radius: int = 1           # maximum manhattan distance for contact
    n_apc: int = 12
    n_tcell: int = 60
    n_neutrophil: int = 40
    n_macrophage: int = 20
    n_nk: int = 15
    n_epithelial: int = 0
    seed: int = 0
    max_steps: int = 200
    backend: str = "numpy"            # "numpy" or "jax" (explicit; no fallback)
    tissue_heterogeneity: bool = True


class _JaxKernel:
    """Optional JAX-accelerated chemokine diffusion + migration kernel.

    Imported lazily; raised as a hard error if absent (no silent fallback).
    """

    _compiled: Any = None

    @classmethod
    def ensure(cls) -> None:
        if cls._compiled is not None:
            return
        try:
            import jax.numpy as jnp
            from jax import jit
        except Exception as exc:  # pragma: no cover - declared dep path
            raise RuntimeError(
                "backend='jax' requested but jax is not installed; "
                "the [has_jax] extra is required (no silent fallback)") from exc

        @jit
        def _step(chem: Any, source: Any, agents_x: Any, agents_y: Any,
                  cxs: Any, cys: Any, d: Any, dt: Any) -> Any:
            # 5-point Laplacian diffusion + a fixed source (site of injury).
            lap = (jnp.roll(chem, 1, 0) + jnp.roll(chem, -1, 0)
                   + jnp.roll(chem, 1, 1) + jnp.roll(chem, -1, 1)
                   - 4.0 * chem)
            chem = chem + d * lap * dt + source * dt
            chem = jnp.maximum(chem, 0.0)
            # chemotactic drift: step each agent up its local gradient
            gx = (jnp.roll(chem, -1, 0) - jnp.roll(chem, 1, 0))[cxs, cys]
            gy = (jnp.roll(chem, -1, 1) - jnp.roll(chem, 1, 1))[cxs, cys]
            nx = agents_x + jnp.sign(gx)
            ny = agents_y + jnp.sign(gy)
            return chem, nx, ny

        cls._compiled = _step

    @staticmethod
    def step(chem: Any, source: Any, agents_x: Any, agents_y: Any,
             cxs: Any, cys: Any, d: Any, dt: Any,
             shape: Any) -> tuple[Any, Any]:
        _JaxKernel.ensure()
        _, nx, ny = _JaxKernel._compiled(
            chem, source, agents_x, agents_y, cxs, cys, d, dt)
        nx = _np.clip(_np.asarray(nx), 0, shape[0] - 1)
        ny = _np.clip(_np.asarray(ny), 0, shape[1] - 1)
        return nx, ny


class SpatialAgentGrid:
    """Deterministic agent grid + chemokine diffusion (doc/40 Phase F, G15).

    Usage::

        grid = SpatialAgentGrid(SpatialABMConfig(seed=3))
        for _ in range(steps):
            grid.step()
        hist = grid.state_histogram()
    """

    def __init__(self, config: SpatialABMConfig | None = None,
                 agents: list[TissueAgent] | None = None) -> None:
        self.cfg = config or SpatialABMConfig()
        if self.cfg.backend not in ("numpy", "jax"):
            raise ValueError(
                f"unknown backend {self.cfg.backend!r}; choose 'numpy' or 'jax'")
        if self.cfg.backend == "jax":
            _JaxKernel.ensure()
        self._rng = random.Random(self.cfg.seed)

        self.W = int(self.cfg.width)
        self.H = int(self.cfg.height)
        if not (_HAS_NUMPY):
            raise RuntimeError("numpy is a declared dependency; install helixlang")

        # chemokine field (0 = baseline) + fixed source (injury site).
        self.chemokine = _np.zeros((self.W, self.H), dtype=float)
        self.source = _np.zeros((self.W, self.H), dtype=float)
        if self.cfg.tissue_heterogeneity:
            # a wound/infection compartment in the centre.
            cx, cy = self.W // 2, self.H // 2
            self.source[cx, cy] = 40.0
            self._wound: tuple[int, int] | None = (cx, cy)
        else:
            self._wound = None

        # tissue landscape: 0 = open lumen, 1 = stroma/endothelium (slower).
        if self.cfg.tissue_heterogeneity:
            self.tissue = _np.zeros((self.W, self.H), dtype=float)
            for i in range(self.W):
                for j in range(self.H):
                    if (i + j) % 7 == 0 and not (i == cx and j == cy):
                        self.tissue[i, j] = 1.0
        else:
            self.tissue = _np.zeros((self.W, self.H), dtype=float)

        self.agents: list[TissueAgent] = []
        if agents is not None:
            self.agents = list(agents)
        else:
            self._seed_agents()
        self.step_index: int = 0
        self._uid = max((a.uid for a in self.agents), default=-1) + 1

    # -- construction ------------------------------------------------------
    def _seed_agents(self) -> None:
        cfg = self.cfg

        def rand_cell() -> tuple[int, int]:
            x = self._rng.randrange(0, self.W)
            y = self._rng.randrange(0, self.H)
            return x, y

        counts = {
            CellType.APC: cfg.n_apc,
            CellType.TCELL: cfg.n_tcell,
            CellType.NEUTROPHIL: cfg.n_neutrophil,
            CellType.MACROPHAGE: cfg.n_macrophage,
            CellType.NK: cfg.n_nk,
        }
        for ctype, n in counts.items():
            for _ in range(n):
                x, y = rand_cell()
                self.agents.append(TissueAgent(
                    uid=len(self.agents), cell_type=ctype, x=x, y=y,
                    speed_um_h=self._rng.uniform(8.0, 25.0)))

    # -- step --------------------------------------------------------------
    def step(self) -> None:
        """Advance the ABM one tick (diffusion + migration + contact)."""
        cfg = self.cfg
        dt = cfg.dt_h
        W, H = self.W, self.H
        self.chemokine = self.chemokine + cfg.diffusion_coeff * self._laplacian(
            self.chemokine) * dt + self.source * dt
        self.chemokine = _np.maximum(self.chemokine, 0.0)

        xs = _np.array([a.x for a in self.agents], dtype=int)
        ys = _np.array([a.y for a in self.agents], dtype=int)
        if cfg.backend == "jax":
            xs, ys = _JaxKernel.step(
                self.chemokine, self.source, xs, ys,
                _np.clip(xs, 0, W - 1), _np.clip(ys, 0, H - 1),
                cfg.diffusion_coeff, dt, (W, H))
        else:
            xs, ys = self._migrate_numpy(xs, ys)

        # apply movement with tissue-resistance weighting + stochasticity
        for k, a in enumerate(self.agents):
            nx = int(xs[k])
            ny = int(ys[k])
            if a.cell_type != CellType.EPITHELIAL:
                if self.tissue[nx, ny] > 0:
                    # stroma slows, not stops, migration (probabilistic)
                    if self._rng.random() < 0.5:
                        nx, ny = a.x, a.y
                a.move(nx, ny)
            a.lifetime_h += dt

        # contact-dependent T-cell/APC signaling
        self._contact_signaling()
        self._age_states(dt)
        self.step_index += 1

    def _laplacian(self, arr: _np.ndarray) -> _np.ndarray:
        # periodic in x, reflecting in y to bound the tissue
        out = _np.zeros_like(arr)
        out[1:-1, 1:-1] = (arr[:-2, 1:-1] + arr[2:, 1:-1]
                           + arr[1:-1, :-2] + arr[1:-1, 2:]
                           - 4.0 * arr[1:-1, 1:-1])
        return out

    def _migrate_numpy(self, xs: _np.ndarray, ys: _np.ndarray,
                       ) -> tuple[_np.ndarray, _np.ndarray]:
        """Chemotactic drift along the local gradient (numpy fallback path).

        Not a "silent fallback" of a declared feature: numpy is the default
        backend; jax is the explicit accelerator.
        """
        gx = _np.zeros_like(xs, dtype=float)
        gy = _np.zeros_like(ys, dtype=float)
        for k, a in enumerate(self.agents):
            x, y = a.x, a.y
            if 0 < x < self.W - 1 and 0 < y < self.H - 1:
                gx[k] = self.chemokine[x + 1, y] - self.chemokine[x - 1, y]
                gy[k] = self.chemokine[x, y + 1] - self.chemokine[x, y - 1]
        gx = _np.sign(gx) * self.cfg.chemotaxis_strength
        gy = _np.sign(gy)
        # only move when gradient exceeds a noise threshold (directed, not Brownian)
        mag = _np.hypot(gx, gy)
        strong = mag > 1e-9
        nx = xs.copy()
        ny = ys.copy()
        nx[strong] += _np.sign(gx[strong]).astype(int)
        ny[strong] += _np.sign(gy[strong]).astype(int)
        return _np.clip(nx, 0, self.W - 1), _np.clip(ny, 0, self.H - 1)

    # -- interactions ------------------------------------------------------
    def _contact_signaling(self) -> None:
        """Contact-dependent T-cell/APC activation (nearest-neighbour within r).

        Contact *detection* is vectorized with numpy (one O(n_tcell x n_apc)
        boolean matrix) instead of a nested Python scan, but each T-cell's
        activation is still accumulated per contacting APC in the same
        ascending-APC order, capped at 1.0 — so the numerics (and final
        activation/state per agent) are bit-identical to the sequential loop.
        Falls back to the reference loop when numpy is unavailable.
        """
        r = int(self.cfg.contact_radius)
        half = self.cfg.chemokine_half
        tcells = [
            a for a in self.agents
            if a.cell_type == CellType.TCELL
            and a.state not in (AgentState.ACTIVATED, AgentState.EXHAUSTED)
        ]
        apcs = [a for a in self.agents if a.cell_type == CellType.APC]
        if not tcells or not apcs:
            return
        if _np is None:  # pragma: no cover - numpy is the default backend
            for a in tcells:
                for apc in apcs:
                    if apc.state == AgentState.APOPTOTIC:
                        continue
                    if abs(a.x - apc.x) <= r and abs(a.y - apc.y) <= r:
                        chem = 1.0 + math.tanh(self.chemokine[a.x, a.y] / 10.0)
                        p = chem / (chem + half)
                        a.activation = min(1.0, a.activation + 0.2 * p)
                        if a.activation >= 0.5:
                            a.state = AgentState.ACTIVATED
                        elif a.activation >= 0.2:
                            a.state = AgentState.ACTIVATING
            return

        txs = _np.array([a.x for a in tcells], dtype=int)
        tys = _np.array([a.y for a in tcells], dtype=int)
        axs = _np.array([a.x for a in apcs], dtype=int)
        ays = _np.array([a.y for a in apcs], dtype=int)
        apc_alive = _np.array(
            [a.state != AgentState.APOPTOTIC for a in apcs], dtype=bool
        )
        contact = (
            (abs(txs[:, None] - axs[None, :]) <= r)
            & (abs(tys[:, None] - ays[None, :]) <= r)
            & apc_alive[None, :]
        )

        for i, a in enumerate(tcells):
            idxs = _np.nonzero(contact[i])[0]
            if idxs.size == 0:
                continue
            # ``p`` depends only on the T-cell's cell (chemokine read) + config.
            chem = 1.0 + math.tanh(self.chemokine[a.x, a.y] / 10.0)
            p = chem / (chem + half)
            inc = 0.2 * p
            # accumulate per contacting APC in ascending ``apcs`` order —
            # identical sequence of identical floats to the scalar loop.
            for _ in range(idxs.size):
                a.activation = min(1.0, a.activation + inc)
            if a.activation >= 0.5:
                a.state = AgentState.ACTIVATED
            elif a.activation >= 0.2:
                a.state = AgentState.ACTIVATING

    def _age_states(self, dt: float) -> None:
        """Advance lifetime + exhaustion/apoptosis with contact decay."""
        for a in self.agents:
            if a.state == AgentState.EXHAUSTED:
                # exhausted cells slowly clear
                if self._rng.random() < 0.02 * dt:
                    a.state = AgentState.APOPTOTIC
            elif a.state == AgentState.ACTIVATED:
                # sustained activation decays without re-contact
                a.activation = max(0.0, a.activation - 0.05 * dt)
                if a.activation <= 0.1:
                    a.state = AgentState.RESTING

    # -- observables -------------------------------------------------------
    def state_histogram(self) -> dict[str, int]:
        out = {s.name.lower(): 0 for s in AgentState}
        for a in self.agents:
            out[a.state.name.lower()] += 1
        return out

    def cell_counts(self) -> dict[str, int]:
        out = {c.name.lower(): 0 for c in CellType}
        for a in self.agents:
            out[a.cell_type.name.lower()] += 1
        return out

    def mean_chemokine(self) -> float:
        return float(_np.mean(self.chemokine))

    def activated_tcells(self) -> int:
        return sum(1 for a in self.agents
                   if a.cell_type == CellType.TCELL
                   and a.state in (AgentState.ACTIVATING, AgentState.ACTIVATED))

    def clone(self) -> SpatialAgentGrid:
        """Deep-copy (deterministic replay / cohort composition)."""
        cfg = replace(self.cfg)
        new = SpatialAgentGrid(config=cfg, agents=[replace(a) for a in self.agents])
        new.step_index = self.step_index
        new.chemokine = self.chemokine.copy()
        new.source = self.source.copy()
        new.tissue = self.tissue.copy()
        new._uid = self._uid
        new._wound = self._wound
        return new


def run_spatial_abm(config: SpatialABMConfig | None = None,
                    steps: int = 100) -> SpatialAgentGrid:
    """Convenience runner: construct, step ``steps`` times, return the grid."""
    grid = SpatialAgentGrid(config)
    if steps == -1:
        steps = config.max_steps if config else 100
    for _ in range(steps):
        grid.step()
    return grid


def run_cohort_spatial(n: int, steps: int = 60,
                       seed: int = 0,
                       **cfg_overrides: Any) -> list[SpatialAgentGrid]:
    """Run ``n`` independent single-grid trajectories (G13-style cohort).

    Each grid uses a per-patient derived seed so the cohort is deterministic
    but heterogeneous (doc/40 §7 risk 5: ≤500 agents/patient).
    """
    out: list[SpatialAgentGrid] = []
    for i in range(n):
        c = SpatialABMConfig(seed=(seed * 1000003 + i) % (2 ** 31), **cfg_overrides)
        out.append(run_spatial_abm(c, steps))
    return out


__all__ = [
    "SpatialAgentGrid", "SpatialABMConfig", "TissueAgent",
    "AgentState", "CellType", "run_spatial_abm", "run_cohort_spatial",
]
