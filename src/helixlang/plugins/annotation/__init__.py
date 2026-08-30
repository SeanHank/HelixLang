"""Genome functional annotation pipeline (doc/20 §5)."""
from __future__ import annotations

from collections.abc import Callable

from helixlang.api.registry import PluginProvider
from helixlang.plugins.annotation.blast import run_diamond
from helixlang.plugins.annotation.tf_detection import detect_transcription_factors
from helixlang.plugins.annotation.transporter import classify_transporters

__all__ = [
    "GeneAnnotation",
    "run_diamond",
    "detect_transcription_factors",
    "classify_transporters",
]


class GeneAnnotation:
    """Per-gene functional annotation result (doc/20 §5.3)."""

    __slots__ = (
        "gene_id",
        "protein_seq",
        "ec_numbers",
        "kegg_ko",
        "go_terms",
        "subsystem",
        "is_transporter",
        "transport_substrate",
        "is_transcription_factor",
        "tf_family",
        "confidence",
    )

    def __init__(
        self,
        gene_id: str,
        protein_seq: str = "",
        ec_numbers: list[str] | None = None,
        kegg_ko: list[str] | None = None,
        go_terms: list[str] | None = None,
        subsystem: str = "",
        is_transporter: bool = False,
        transport_substrate: str | None = None,
        is_transcription_factor: bool = False,
        tf_family: str | None = None,
        confidence: float = 0.0,
    ) -> None:
        self.gene_id = gene_id
        self.protein_seq = protein_seq
        self.ec_numbers = ec_numbers or []
        self.kegg_ko = kegg_ko or []
        self.go_terms = go_terms or []
        self.subsystem = subsystem
        self.is_transporter = is_transporter
        self.transport_substrate = transport_substrate
        self.is_transcription_factor = is_transcription_factor
        self.tf_family = tf_family
        self.confidence = confidence

    def to_dict(self) -> dict[str, object]:
        return {
            "gene_id": self.gene_id,
            "ec_numbers": self.ec_numbers,
            "kegg_ko": self.kegg_ko,
            "go_terms": self.go_terms,
            "subsystem": self.subsystem,
            "is_transporter": self.is_transporter,
            "is_transcription_factor": self.is_transcription_factor,
            "tf_family": self.tf_family,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Plugin contract (doc/36 §7: annotation/* -> plugins/annotation/)
# ---------------------------------------------------------------------------


def _check(pkg: str) -> bool:
    def _probe() -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False
    return _probe()


def _make_backend(cfg: dict | None = None) -> type:
    from helixlang.plugins.annotation import GeneAnnotation
    return GeneAnnotation


def _load() -> Callable[[dict | None], type]:
    if not _check("numpy"):
        from helixlang.core.errors import PluginDependencyError
        raise PluginDependencyError("annotation", "numpy", "annotation")
    return _make_backend


PLUGIN = PluginProvider(
    name="annotation",
    extra="annotation",
    keywords=("gene_id", "enzyme"),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": lambda: _check("numpy")},
    load=_load,
)
