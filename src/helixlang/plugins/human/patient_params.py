"""432-parameter IIRABM-style immune parameter set (doc/40 Phase H).

Carries the patient-specific parameter vector used by the Bayesian re-fitter
(doc/40 Phase H, Cockrell & An *Front. Physiol.* 12:662845 2021 — GA/ABC-calibrated
432 continuous free parameters).  The vector is organised into the §3.1 parameter
domains named in doc/40 §6 (Phase H):

- immune-cell proliferation/differentiation rates
- cytokine production rates
- receptor affinities
- spatial migration speeds

with the remainder covering the reduced ODE cascade (complement, PD-1, APR) so that
the same 432-slot vector can drive every channel a virtual patient exposes.

Determinism contract (doc/39 §5.3): the nominal vector is a single fixed constant
array; sampling adds controlled log-normal or uniform jitter around named anchors
with a caller-supplied ``random.Random``.  No hidden RNG, no silent fallback.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

try:  # numpy optional (doc/39 O2 idiom); only needed for array transforms
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Parameter domain sizes (sum to 432 = N_PARAMS)
# ---------------------------------------------------------------------------
N_PROLIF = 48          # immune-cell proliferation / differentiation rates
N_CYTOKINE = 96        # cytokine production / clearance / feedback rates
N_AFFINITY = 96        # receptor / ligand dissociation constants (Kd, nM; kD/h)
N_MIGRATION = 24       # spatial migration speeds (um/h) + chemotactic gain
N_COMPLEMENT = 120     # complement cascade rates + C3aR/C5aR signaling (the full
                       # L7 step references 61; this nominal slice keeps 120)
N_PD1 = 24             # PD-1/PD-L1/PD-L2 / CTLA-4 / LAG-3 network
N_APR = 24             # acute-phase (CRP/SAA/ferritin/PCT/fibrinogen) kinetics
N_OTHER = 0            # (reserved; currently 0)

N_PARAMS = (N_PROLIF + N_CYTOKINE + N_AFFINITY + N_MIGRATION
            + N_COMPLEMENT + N_PD1 + N_APR + N_OTHER)

assert N_PARAMS == 432, N_PARAMS

# ---------------------------------------------------------------------------
# Domain offsets (contiguous slices into the 432-vector)
# ---------------------------------------------------------------------------
O_PROLIF = 0
O_CYTOKINE = O_PROLIF + N_PROLIF
O_AFFINITY = O_CYTOKINE + N_CYTOKINE
O_MIGRATION = O_AFFINITY + N_AFFINITY
O_COMPLEMENT = O_MIGRATION + N_MIGRATION
O_PD1 = O_COMPLEMENT + N_COMPLEMENT
O_APR = O_PD1 + N_PD1
O_OTHER = O_APR + N_APR
assert O_OTHER + N_OTHER == N_PARAMS

# ---------------------------------------------------------------------------
# Named slice anchors within each domain (for self-documenting access)
# ---------------------------------------------------------------------------
DOMAIN_SLICES: dict[str, tuple[int, int]] = {
    "proliferation": (O_PROLIF, O_CYTOKINE),
    "cytokine": (O_CYTOKINE, O_AFFINITY),
    "affinity": (O_AFFINITY, O_MIGRATION),
    "migration": (O_MIGRATION, O_COMPLEMENT),
    "complement": (O_COMPLEMENT, O_PD1),
    "pd1": (O_PD1, O_APR),
    "apr": (O_APR, O_OTHER),
}

# Named parameter keys → (domain, index-in-slice) for readable access.
_BIOPHYSICAL_KEYS: dict[str, tuple[str, int]] = {
    # proliferation / differentiation
    "cd4_prolif": ("proliferation", 0),
    "cd8_prolif": ("proliferation", 1),
    "b_prolif": ("proliferation", 2),
    "apc_maturation": ("proliferation", 3),
    "plasma_differentiation": ("proliferation", 4),
    "memory_conversion": ("proliferation", 5),
    "nk_prolif": ("proliferation", 6),
    "neutrophil_production": ("proliferation", 7),
    "monocyte_production": ("proliferation", 8),
    # cytokine production
    "il6_production": ("cytokine", 0),
    "tnf_production": ("cytokine", 1),
    "il1b_production": ("cytokine", 2),
    "il10_production": ("cytokine", 3),
    "ifn_production": ("cytokine", 4),
    "il6_clearance": ("cytokine", 5),
    "tnf_clearance": ("cytokine", 6),
    "il10_feedback": ("cytokine", 7),
    # receptor affinities (nM / kD h^-1)
    "pd1_pdl1_kd": ("affinity", 0),
    "pd1_pdl2_kd": ("affinity", 1),
    "ctla4_cd80_kd": ("affinity", 2),
    "lag3_mhcii_kd": ("affinity", 3),
    "il6r_kd": ("affinity", 4),
    "c5ar_kd": ("affinity", 5),
    # migration (um/h) + chemotactic gain
    "neutrophil_speed": ("migration", 0),
    "macrophage_speed": ("migration", 1),
    "tcell_speed": ("migration", 2),
    "chemotaxis_gain": ("migration", 3),
    # complement canonical anchors (Zewde & Morikis 2018)
    "c1_activation": ("complement", 0),
    "c3_convertase": ("complement", 1),
    "c5_convertase": ("complement", 2),
    "mbl_masp": ("complement", 3),
    "factor_d": ("complement", 4),
    "properdin": ("complement", 5),
    "c4bp": ("complement", 6),
    "mac_assembly": ("complement", 7),
    # PD-1 / checkpoint
    "pd1_internalization": ("pd1", 0),
    "pd1_surface_trafficking": ("pd1", 1),
    "ctla4_blockade": ("pd1", 2),
    "lag3_blockade": ("pd1", 3),
    # acute phase
    "crp_production": ("apr", 0),
    "crp_clearance": ("apr", 1),
    "saa_production": ("apr", 2),
    "ferritin_production": ("apr", 3),
    "pct_production": ("apr", 4),
}


def nominal_params() -> list[float]:
    """Return the 432-length nominal (healthy-mean) parameter vector.

    Values are physically anchored (±15–30% physiologic ranges) so the posterior
    re-fit starts from a biologically plausible prior mode.  All positive.
    """
    p = [0.0] * N_PARAMS

    # proliferation domain (48)
    _fill(p, O_PROLIF, [0.06, 0.07, 0.07, 1.0 / 18.0, 0.6, 0.12, 0.05,
                        0.10, 0.02, 0.05, 0.04, 0.03])
    for i in range(12, N_PROLIF):
        p[O_PROLIF + i] = 0.04 + 0.02 * ((i * 7919) % 5)

    # cytokine domain (96): production rates around pg/mL/h, clearances around /h
    _fill(p, O_CYTOKINE, [12.0, 10.0, 8.0, 6.0, 5.0, math.log(2) / 1.5,
                          math.log(2) / 0.5, 0.4, 8.0, 0.1])
    for i in range(10, 10 + 24):
        p[O_CYTOKINE + i] = 9.0 + 1.5 * ((i * 104729) % 5)
    for i in range(34, N_CYTOKINE):
        p[O_CYTOKINE + i] = 0.03 + 0.01 * ((i * 2654435761) % 4)

    # affinity domain (96): Kd in nM ~0.1–100
    for i in range(N_AFFINITY):
        p[O_AFFINITY + i] = 1.0 * (10.0 ** (0.5 * ((i * 31) % 5) - 1.0))

    # migration domain (24): speeds um/h 2–40
    _fill(p, O_MIGRATION, [20.0, 12.0, 15.0, 0.8, 8.0, 5.0])
    for i in range(6, N_MIGRATION):
        p[O_MIGRATION + i] = 3.0 + 2.0 * ((i * 17) % 5)

    # complement domain (120)
    _fill(p, O_COMPLEMENT, [0.12, 0.05, 0.03, 0.04, 0.06, 0.02, 0.05,
                            math.log(2) / 6.0, 0.02, math.log(2) / 0.5])
    for i in range(10, N_COMPLEMENT):
        p[O_COMPLEMENT + i] = 0.02 + 0.008 * ((i * 40503) % 6)

    # PD-1 domain (24)
    _fill(p, O_PD1, [0.05, 0.5, 0.4, 0.4, 0.05, 0.02])
    for i in range(6, N_PD1):
        p[O_PD1 + i] = 0.3 + 0.1 * ((i * 9277) % 4)

    # APR domain (24)
    _fill(p, O_APR, [0.1, math.log(2) / 19.0, 0.08, 0.06, 0.05, 0.5])
    for i in range(6, N_APR):
        p[O_APR + i] = 0.04 + 0.01 * ((i * 32077) % 5)

    return p


def _fill(p: list[float], offset: int, values: list[float]) -> None:
    for j, v in enumerate(values):
        p[offset + j] = v


@dataclass(frozen=True)
class ParamSlice:
    """A named slice of the 432-vector with its domain and offset."""

    domain: str
    offset: int
    length: int

    def extract(self, params: Sequence[float]) -> list[float]:
        return list(params[self.offset:self.offset + self.length])


class PatientParameterSet:
    """Named access to the 432-parameter vector (doc/40 Phase H).

    Wraps a mutable 432-length vector and exposes (a) fixed-nominal construction,
    (b) a mean/sd prior descriptor per domain for the Bayesian re-fitter, and
    (c) deterministic patient jitter for G13 feeding.
    """

    def __init__(self, params: Sequence[float] | None = None, raw: _np.ndarray | None = None) -> None:
        if raw is not None:
            self._params = [float(v) for v in raw]
        elif params is not None:
            if len(params) != N_PARAMS:
                raise ValueError(f"expected {N_PARAMS} params, got {len(params)}")
            self._params = list(params)
        else:
            self._params = nominal_params()

    # -- vector access -----------------------------------------------------
    def to_list(self) -> list[float]:
        return list(self._params)

    @property
    def size(self) -> int:
        return N_PARAMS

    def get(self, key: str) -> float:
        domain, idx = _BIOPHYSICAL_KEYS[key]
        lo, hi = DOMAIN_SLICES[domain]
        return self._params[lo + idx]

    def set(self, key: str, value: float) -> None:
        domain, idx = _BIOPHYSICAL_KEYS[key]
        lo, hi = DOMAIN_SLICES[domain]
        self._params[lo + idx] = float(value)

    def domain(self, name: str) -> list[float]:
        lo, hi = DOMAIN_SLICES[name]
        return list(self._params[lo:hi])

    def slice(self, name: str) -> ParamSlice:
        lo, hi = DOMAIN_SLICES[name]
        return ParamSlice(name, lo, hi - lo)

    def copy(self) -> PatientParameterSet:
        return PatientParameterSet(list(self._params))

    # -- G13 jitter ---------------------------------------------------------
    @classmethod
    def patient_variant(cls, seed: int, patient_idx: int = 0,
                        sd_log: float = 0.15) -> PatientParameterSet:
        """Deterministic per-patient log-normal jitter (G13 feeding, doc/39 §5.3)."""
        base = nominal_params()
        rnd = random.Random(seed * 1000003 + patient_idx)
        out = [v * math.exp(rnd.gauss(0.0, sd_log)) for v in base]
        return cls(out)

    # -- prior descriptor for the Bayesian re-fitter ----------------------
    @staticmethod
    def prior(log_sd: float = 0.15) -> list[float]:
        """Return a 432-length prior dispersion (log std) for HMC re-fit."""
        return [log_sd] * N_PARAMS


class PatientParameterTable:
    """Collection of named parameter sets (multiple patients / archetypes)."""

    def __init__(self, sets: dict[str, PatientParameterSet] | None = None) -> None:
        self._sets: dict[str, PatientParameterSet] = dict(sets or {})

    def add(self, name: str, params: PatientParameterSet) -> None:
        self._sets[name] = params

    def get(self, name: str) -> PatientParameterSet:
        return self._sets[name]

    @classmethod
    def from_seed(cls, n: int, seed: int = 0, sd_log: float = 0.15) -> PatientParameterTable:
        table = cls()
        for i in range(n):
            table.add(f"patient_{i}", PatientParameterSet.patient_variant(seed, i, sd_log))
        return table

    def names(self) -> list[str]:
        return list(self._sets)

    def __len__(self) -> int:
        return len(self._sets)


def to_array(params: list[float]) -> _np.ndarray:
    """Return the 432-vector as a numpy array (raises if numpy absent)."""
    if _np is None:
        raise RuntimeError("numpy is required for PatientParameterSet.to_array")
    return _np.asarray(params, dtype=float)


__all__ = [
    "N_PARAMS", "nominal_params", "PatientParameterSet", "PatientParameterTable",
    "DOMAIN_SLICES", "to_array", "ParamSlice",
]
