#!/usr/bin/env python3
"""Benchmark 32: Drug-drug interaction prediction."""
from __future__ import annotations

import json
import sys
import time

from helixlang.plugins.human.ddi import (
    DEFAULT_DDI_RULES,
    DDIModel,
    DDIRule,
    create_default_ddi_model,
)


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "32_ddi_prediction"}
    try:
        # 1. create_default_ddi_model() returns a DDIModel with rules
        model = create_default_ddi_model()
        assert isinstance(model, DDIModel), f"Expected DDIModel, got {type(model).__name__}"
        assert len(model.rules) > 0, "DDIModel should have rules"

        # 2. Warfarin+Fluconazole is a known DDI pair
        warfarin_fluconazole_found = False
        for rule in model.rules:
            if rule.substrate == "warfarin" and rule.interacting_drug == "fluconazole":
                warfarin_fluconazole_found = True
                break
        assert warfarin_fluconazole_found, (
            "Warfarin+Fluconazole DDI pair not found in rules"
        )

        # 3. The interaction flag is triggered
        cyp_profiles = {"CYP2C9": 1.0, "CYP2D6": 1.0, "CYP3A4": 1.0}
        alerts = model.get_clinical_alerts(
            ["warfarin", "fluconazole"], cyp_profiles
        )
        assert len(alerts) > 0, "Warfarin+Fluconazole should trigger DDI alerts"
        warfarin_alert = next(
            (a for a in alerts if "warfarin" in a["drugs"]),
            None,
        )
        assert warfarin_alert is not None, "No alert for warfarin"
        assert warfarin_alert["severity"] in ("moderate", "severe", "contraindicated"), (
            f"Expected moderate+ severity, got {warfarin_alert['severity']!r}"
        )

        # Verify clearance modifier
        mods = model.compute_clearance_modifiers(
            ["warfarin", "fluconazole"], cyp_profiles
        )
        assert mods["warfarin"] < 1.0, (
            f"Warfarin clearance should be reduced (<1.0), got {mods['warfarin']}"
        )

        # 4. At least 5 DDI rules exist
        assert len(DEFAULT_DDI_RULES) >= 5, (
            f"Expected >=5 DDI rules, got {len(DEFAULT_DDI_RULES)}"
        )

        # Verify DDIRule dataclass
        sample_rule = DEFAULT_DDI_RULES[0]
        assert isinstance(sample_rule, DDIRule), "Rule should be DDIRule"
        assert sample_rule.substrate, "Rule should have a substrate"
        assert sample_rule.enzyme, "Rule should have an enzyme"
        assert sample_rule.fold_change > 0, "fold_change should be positive"

        # Verify empty regimen produces no alerts
        empty_alerts = model.get_clinical_alerts([], cyp_profiles)
        assert len(empty_alerts) == 0, "Empty regimen should produce no alerts"

        # Verify single-drug regimen (no interaction possible)
        single_alerts = model.get_clinical_alerts(["warfarin"], cyp_profiles)
        # May have enzyme-state rules, but no co-administration DDI
        co_admin_alerts = [
            a for a in single_alerts
            if len(a["drugs"]) > 1
        ]
        assert len(co_admin_alerts) == 0, (
            "Single-drug regimen should have no co-administration DDIs"
        )

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "checks": {
                "create_default_ddi_model_has_rules": True,
                "warfarin_fluconazole_is_known_ddi": True,
                "interaction_flag_is_triggered": True,
                "at_least_5_ddi_rules_exist": True,
            },
            "details": {
                "total_rules": len(DEFAULT_DDI_RULES),
                "alert_count_warfarin_fluconazole": len(alerts),
                "warfarin_clearance_modifier": round(mods["warfarin"], 4),
                "warfarin_alert_severity": warfarin_alert["severity"],
                "warfarin_alert_effect": warfarin_alert["effect"][:80],
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
