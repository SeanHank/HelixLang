#!/usr/bin/env python3
"""Benchmark 63: Phenotype calculation and human simulation.

Validates phenotype.py and simulation.py modules:
  - ExternalTraits with age, sex, BMI fields
  - PhenotypeCalculator for genotype + trait scaling
  - HumanSimulationConfig and HumanSimulation

Reference: Ursino M et al. 2020, Front Physiol 11:556
           (whole-body simulation).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.human.phenotype import (
            PhenotypeCalculator,
            ExternalTraits,
            create_default_traits,
        )
        from helixlang.human.simulation import (
            HumanSimulation,
            HumanSimulationConfig,
        )

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. All required imports succeeded
        checks["import_all_classes"] = True

        # 2. create_default_traits() returns ExternalTraits with age, sex, BMI fields
        traits = create_default_traits()
        checks["default_traits_is_external_traits"] = isinstance(traits, ExternalTraits)
        checks["traits_has_age"] = hasattr(traits, "age_years")
        checks["traits_has_sex"] = hasattr(traits, "sex")
        checks["traits_has_bmi"] = hasattr(traits, "bmi")
        details["default_age"] = traits.age_years
        details["default_sex"] = traits.sex
        details["default_bmi"] = traits.bmi

        # 3. PhenotypeCalculator: create with default traits, verify instantiation
        from helixlang.human.genotype import create_default_genotype

        genotype = create_default_genotype()
        calc = PhenotypeCalculator(genotype=genotype, traits=traits)
        checks["phenotype_calculator_instantiates"] = isinstance(
            calc, PhenotypeCalculator
        )

        # 4. PhenotypeCalculator has compute methods
        checks["has_compute_cyp_activity"] = hasattr(calc, "compute_cyp_activity")
        checks["has_compute_physiology"] = hasattr(calc, "compute_physiology")
        cyp = calc.compute_cyp_activity()
        checks["cyp_activity_returns_dict"] = isinstance(cyp, dict)
        checks["cyp_activity_nonempty"] = len(cyp) > 0
        details["cyp_enzymes"] = sorted(cyp.keys())

        # 5. HumanSimulationConfig: create with default params
        config = HumanSimulationConfig()
        checks["sim_config_instantiates"] = isinstance(config, HumanSimulationConfig)
        checks["sim_config_has_duration"] = hasattr(config, "total_duration_days")
        checks["sim_config_has_dfa_dt"] = hasattr(config, "dfa_dt_h")
        details["sim_config_duration_days"] = config.total_duration_days

        # 6. HumanSimulation: create with config, verify it has run() method
        sim = HumanSimulation(config)
        checks["human_simulation_instantiates"] = isinstance(
            sim, HumanSimulation
        )
        checks["has_run_method"] = hasattr(sim, "run")
        checks["has_config"] = hasattr(sim, "config") and sim.config is config

        all_pass = all(checks.values())
        return {
            "id": "63_phenotype_simulation",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Ursino M et al. 2020",
                "authors": "Ursino M, Zingales M, Magosso E, et al.",
                "year": 2020,
                "journal": "Front Physiol",
                "volume": "11",
                "pages": "556",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "63_phenotype_simulation",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
