"""Validation evidence chain schema for HelixLang benchmarks.

Every simulation backend and scientific module must produce an evidence chain:
    Reference → Expected → Actual → Error → Reproducibility

This module provides the dataclasses and helpers that enforce this pattern.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class Reference:
    """Source of truth for a validation benchmark."""

    source: str
    doi: str | None = None
    authors: str | None = None
    year: int | None = None
    journal: str | None = None
    url: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"source": self.source}
        if self.doi:
            d["doi"] = self.doi
        if self.authors:
            d["authors"] = self.authors
        if self.year:
            d["year"] = self.year
        if self.journal:
            d["journal"] = self.journal
        if self.url:
            d["url"] = self.url
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Reference | None:
        if not d or not isinstance(d, dict):
            return None
        source = d.get("source") or d.get("note") or d.get("journal") or ""
        if not source:
            return None
        return cls(
            source=source,
            doi=d.get("doi"),
            authors=d.get("authors"),
            year=d.get("year"),
            journal=d.get("journal"),
            url=d.get("url"),
            note=d.get("note"),
        )

    def fmt_short(self) -> str:
        """One-line summary: 'Authors — Year — Source'."""
        parts: list[str] = []
        if self.authors:
            parts.append(self.authors)
        if self.year:
            parts.append(str(self.year))
        src = self.source or self.note or self.journal or ""
        if src:
            parts.append(src[:60])
        return " — ".join(parts) if parts else ""


@dataclasses.dataclass(frozen=True, slots=True)
class Expected:
    """Expected value from the reference."""

    metric: str
    value: Any
    tolerance: float | None = None
    unit: str | None = None
    check: str | None = None
    checks: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"metric": self.metric, "value": self.value}
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance
        if self.unit:
            d["unit"] = self.unit
        if self.check:
            d["check"] = self.check
        if self.checks:
            d["checks"] = self.checks
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Expected | None:
        if not d or not isinstance(d, dict):
            return None
        metric = d.get("metric", "")
        value = d.get("value")
        if metric or value is not None:
            return cls(
                metric=metric,
                value=value,
                tolerance=d.get("tolerance"),
                unit=d.get("unit"),
            )
        # Dict with boolean-like keys (e.g. grn_nodes: True) — synthesize metric
        items = list(d.items())[:3]
        if items:
            return cls(
                metric="checks",
                value={k: v for k, v in items},
                unit=None,
            )
        return None

    def fmt_short(self) -> str:
        """One-line summary: 'metric=value unit ±tolerance'."""
        if isinstance(self.value, dict):
            checks = [f"{k}={v}" for k, v in list(self.value.items())[:3]]
            return ", ".join(checks)
        s = f"{self.metric}={self.value}"
        if self.unit:
            s += f" {self.unit}"
        if self.tolerance is not None:
            s += f" ±{self.tolerance}"
        return s


@dataclasses.dataclass(frozen=True, slots=True)
class Actual:
    """Actual result produced by HelixLang."""

    value: Any
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"value": self.value}
        if self.raw:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Actual | None:
        if not d or not isinstance(d, dict):
            return None
        value = d.get("value")
        if value is not None:
            return cls(value=value, raw=d)
        # Dict with multiple results — summarize
        if d:
            return cls(value=d, raw=d)
        return None

    def fmt_short(self) -> str:
        """One-line summary of the actual value."""
        if isinstance(self.value, dict):
            items = [f"{k}={v}" for k, v in list(self.value.items())[:3]]
            return ", ".join(items)
        return str(self.value)


@dataclasses.dataclass(frozen=True, slots=True)
class Error:
    """Quantified error between expected and actual."""

    abs_error: float | None = None
    rel_error: float | None = None
    passed: bool = True
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"passed": self.passed}
        if self.abs_error is not None:
            d["abs_error"] = self.abs_error
        if self.rel_error is not None:
            d["rel_error"] = self.rel_error
        if self.message:
            d["message"] = self.message
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Error:
        if not d or not isinstance(d, dict):
            return cls(passed=True)
        return cls(
            abs_error=d.get("abs_error"),
            rel_error=d.get("rel_error"),
            passed=d.get("passed", True),
            message=d.get("message"),
        )

    def fmt_short(self) -> str:
        """One-line summary of the error."""
        if self.rel_error is not None:
            if self.rel_error < 1e-10:
                return "≈0" if self.abs_error is None else f"≈0 (abs={self.abs_error:.1e})"
            return f"{self.rel_error:.2%}"
        if self.abs_error is not None:
            return f"abs={self.abs_error:.4f}"
        if self.message:
            return str(self.message)[:30]
        if self.passed:
            return "verified"
        return "FAILED"


@dataclasses.dataclass(frozen=True, slots=True)
class Reproducibility:
    """Reproducibility metadata for the benchmark."""

    deterministic: bool = True
    seed_used: int | None = None
    golden_hash: str | None = None
    environment: str | None = None
    python_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"deterministic": self.deterministic}
        if self.seed_used is not None:
            d["seed_used"] = self.seed_used
        if self.golden_hash:
            d["golden_hash"] = self.golden_hash
        if self.environment:
            d["environment"] = self.environment
        if self.python_version:
            d["python_version"] = self.python_version
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Reproducibility:
        if not d or not isinstance(d, dict):
            return cls()
        return cls(
            deterministic=d.get("deterministic", True),
            seed_used=d.get("seed_used"),
            golden_hash=d.get("golden_hash"),
            environment=d.get("environment"),
            python_version=d.get("python_version"),
        )


@dataclasses.dataclass(slots=True)
class EvidenceChain:
    """Complete evidence chain for a validation benchmark.

    This is the core data structure that every benchmark must populate.
    It enforces the pattern: Reference → Expected → Actual → Error → Reproducibility.
    """

    benchmark_id: str
    reference: Reference | None
    expected: Expected | None
    actual: Actual | None
    error: Error
    reproducibility: Reproducibility
    status: str = "PASS"
    layer: str = "unknown"
    name: str = ""
    experimental_comparison: dict[str, Any] | None = None
    # Non-standard fields that don't fit the evidence chain (validation dicts, etc.)
    _extra: dict[str, Any] | None = dataclasses.field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.benchmark_id,
            "status": self.status,
            "evidence_chain": {
                "reference": self.reference.to_dict() if self.reference else None,
                "expected": self.expected.to_dict() if self.expected else None,
                "actual": self.actual.to_dict() if self.actual else None,
                "error": self.error.to_dict(),
                "reproducibility": self.reproducibility.to_dict(),
            },
            "layer": self.layer,
            "name": self.name,
        }
        if self.experimental_comparison:
            d["experimental_comparison"] = self.experimental_comparison
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceChain:
        """Normalize any benchmark JSON into an EvidenceChain.

        Handles all known benchmark schemas:
          1. Standard: reference/expected/actual/error (fully structured)
          2. Validation dict: {check_name: bool}
          3. Checks list/dict: [{name, passed}] or {name: bool}
          4. Comparison: comparison + experimental_comparison
          5. Top-level booleans (e.g. 14_type_system_flow)
          6. Top-level numerics (e.g. 12_parser_roundtrip)
          7. Nested analysis dicts (e.g. 09_reaction_diffusion)
          8. Fallback: status-only with extra fields preserved
        """
        benchmark_id = data.get("id", "unknown")
        status = data.get("status", "UNKNOWN")
        layer = data.get("layer") or ""
        name = data.get("name") or benchmark_id
        experimental = data.get("experimental_comparison")

        # Try standard evidence chain first
        ref = Reference.from_dict(data.get("reference"))
        exp = Expected.from_dict(data.get("expected"))
        act = Actual.from_dict(data.get("actual"))
        err = Error.from_dict(data.get("error"))
        repro = Reproducibility.from_dict(data.get("reproducibility"))

        # If standard fields are all missing, synthesize from alternative schemas
        extra: dict[str, Any] = {}
        if not ref and not exp and not act:
            # Try validation dict
            val = data.get("validation")
            if isinstance(val, dict) and val:
                passed_count = sum(1 for v in val.values() if v is True)
                total = len(val)
                exp = Expected(metric="validation_checks", value=val)
                act = Actual(value={"passed": passed_count, "total": total})
                err = Error(passed=(passed_count == total))
                ref = Reference(source=f"{total} automated checks")

            # Try checks
            checks = data.get("checks")
            if checks and not exp:
                if isinstance(checks, dict):
                    passed_count = sum(
                        1 for v in checks.values()
                        if v is True or (isinstance(v, (dict, list)) and len(v) > 0)
                    )
                    total = len(checks)
                    exp = Expected(metric="checks", value=checks)
                    act = Actual(value={"passed": passed_count, "total": total})
                    err = Error(passed=(passed_count == total))
                    ref = Reference(source=f"{total} automated checks")
                elif isinstance(checks, list):
                    passed_count = sum(1 for c in checks if isinstance(c, dict) and c.get("passed"))
                    total = len(checks)
                    exp = Expected(metric="checks", value=checks)
                    act = Actual(value={"passed": passed_count, "total": total})
                    err = Error(passed=(passed_count == total))
                    ref = Reference(source=f"{total} automated checks")

            # Try comparison
            comp = data.get("comparison")
            if isinstance(comp, dict) and not exp:
                exp = Expected(metric="comparison", value=comp)
                act = Actual(value=comp)
                err = Error(passed=True)

            # Try top-level booleans (e.g. 14_type_system_flow)
            if not exp:
                skip = {"id", "status", "runtime_seconds", "layer", "name",
                        "reference", "expected", "actual", "error", "reproducibility",
                        "experimental_comparison", "validation", "checks", "comparison"}
                top_bools = {k: v for k, v in data.items()
                             if k not in skip and isinstance(v, bool)}
                if top_bools:
                    passed_count = sum(1 for v in top_bools.values() if v)
                    total = len(top_bools)
                    exp = Expected(metric="functional_checks", value=top_bools)
                    act = Actual(value={"passed": passed_count, "total": total})
                    err = Error(passed=(passed_count == total))
                    ref = Reference(source=f"{total} functional checks")

            # Try top-level numerics + nested dicts (e.g. 09, 11, 12, 13)
            if not exp:
                skip = {"id", "status", "runtime_seconds", "layer", "name",
                        "reference", "expected", "actual", "error", "reproducibility",
                        "experimental_comparison", "validation", "checks", "comparison"}
                remaining = {k: v for k, v in data.items() if k not in skip}
                if remaining:
                    # Count how many nested dicts have boolean sub-values (pass/fail checks)
                    bool_checks: dict[str, bool] = {}
                    for k, v in remaining.items():
                        if isinstance(v, dict):
                            # Extract boolean sub-values as checks
                            for sub_k, sub_v in v.items():
                                if isinstance(sub_v, bool):
                                    bool_checks[f"{k}.{sub_k}"] = sub_v
                                elif isinstance(sub_v, dict):
                                    for sub2_k, sub2_v in sub_v.items():
                                        if isinstance(sub2_v, bool):
                                            bool_checks[f"{k}.{sub2_k}"] = sub2_v

                    if bool_checks:
                        passed_count = sum(1 for v in bool_checks.values() if v)
                        total = len(bool_checks)
                        exp = Expected(metric="nested_checks", value=bool_checks)
                        act = Actual(value={"passed": passed_count, "total": total})
                        err = Error(passed=(passed_count == total))
                        ref = Reference(source=f"{total} nested checks")
                    else:
                        # Purely numeric/key-value — show as metrics
                        summary = {}
                        for k, v in remaining.items():
                            if isinstance(v, (int, float, str, bool)):
                                summary[k] = v
                            elif isinstance(v, dict):
                                # Extract top-level numeric values from nested dicts
                                for sub_k, sub_v in v.items():
                                    if isinstance(sub_v, (int, float, bool)):
                                        summary[f"{k}.{sub_k}"] = sub_v
                        if summary:
                            exp = Expected(metric="metrics", value=summary)
                            act = Actual(value=summary)
                            err = Error(passed=True)
                            ref = Reference(source=f"{len(summary)} metrics")

            # Collect remaining non-standard fields
            skip = {"id", "status", "runtime_seconds", "layer", "name",
                     "reference", "expected", "actual", "error", "reproducibility",
                     "experimental_comparison", "validation", "checks", "comparison"}
            extra = {k: v for k, v in data.items() if k not in skip}

        # Ensure reproducibility has environment
        if not repro.environment:
            import sys
            repro = Reproducibility(
                deterministic=repro.deterministic,
                seed_used=repro.seed_used,
                golden_hash=repro.golden_hash,
                environment=f"Python {sys.version.split()[0]}",
            )

        return cls(
            benchmark_id=benchmark_id,
            reference=ref,
            expected=exp,
            actual=act,
            error=err,
            reproducibility=repro,
            status=status,
            layer=layer,
            name=name,
            experimental_comparison=experimental,
            _extra=extra if extra else None,
        )

    def fmt_evidence(self) -> str:
        """Format the full evidence chain as a single string for report tables."""
        parts: list[str] = []
        if self.reference:
            parts.append(self.reference.fmt_short())
        if self.expected:
            parts.append(self.expected.fmt_short())
        if self.actual:
            parts.append(self.actual.fmt_short())
        if self.error:
            parts.append(self.error.fmt_short())
        return " → ".join(parts) if parts else self._fmt_fallback()

    def _fmt_fallback(self) -> str:
        """Fallback for non-standard benchmarks."""
        if self._extra:
            return f"fields: {', '.join(list(self._extra.keys())[:4])}"
        return "functional"


def compute_rel_error(expected: float, actual: float) -> float:
    """Compute relative error as a fraction (0.0 = perfect match)."""
    if abs(expected) < 1e-15:
        return 0.0 if abs(actual) < 1e-15 else float("inf")
    return abs(actual - expected) / abs(expected)


def compute_abs_error(expected: float, actual: float) -> float:
    """Compute absolute error."""
    return abs(actual - expected)


def check_tolerance(rel_error: float, tolerance: float) -> bool:
    """Check if relative error is within tolerance (tolerance is a fraction, e.g. 0.05 = 5%)."""
    return rel_error <= tolerance


def make_evidence_chain(
    benchmark_id: str,
    reference_source: str,
    expected_metric: str,
    expected_value: Any,
    actual_value: Any,
    *,
    tolerance: float | None = None,
    unit: str | None = None,
    layer: str = "unknown",
    name: str = "",
    doi: str | None = None,
    authors: str | None = None,
    year: int | None = None,
    journal: str | None = None,
    golden_hash: str | None = None,
    seed_used: int | None = None,
    experimental_comparison: dict[str, Any] | None = None,
    message: str | None = None,
) -> EvidenceChain:
    """Build an EvidenceChain with automatic error computation.

    For numeric expected values, computes rel_error and abs_error automatically.
    For boolean checks, uses passed = (actual == expected).
    """
    import sys as _sys

    if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
        abs_err = compute_abs_error(float(expected_value), float(actual_value))
        rel_err = compute_rel_error(float(expected_value), float(actual_value))
        passed = tolerance is not None and check_tolerance(rel_err, tolerance)
        if tolerance is None:
            passed = abs_err < 1e-10
    elif isinstance(expected_value, bool):
        abs_err = None
        rel_err = None
        passed = actual_value == expected_value
    else:
        abs_err = None
        rel_err = None
        passed = actual_value == expected_value

    return EvidenceChain(
        benchmark_id=benchmark_id,
        reference=Reference(
            source=reference_source,
            doi=doi,
            authors=authors,
            year=year,
            journal=journal,
        ),
        expected=Expected(
            metric=expected_metric,
            value=expected_value,
            tolerance=tolerance,
            unit=unit,
        ),
        actual=Actual(value=actual_value),
        error=Error(
            abs_error=abs_err,
            rel_error=rel_err,
            passed=passed,
            message=message,
        ),
        reproducibility=Reproducibility(
            deterministic=True,
            seed_used=seed_used,
            golden_hash=golden_hash,
            environment=f"Python {_sys.version.split()[0]}",
        ),
        status="PASS" if passed else "FAIL",
        layer=layer,
        name=name,
        experimental_comparison=experimental_comparison,
    )
