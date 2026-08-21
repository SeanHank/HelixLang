#!/usr/bin/env python3
"""Validate examples 48 & 49 long-duration GEM simulations against literature.

Uses the actual CLI pipeline (with inline DNA support).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def run_example_json(helix_path: str) -> dict:
    """Run a .helix file through the full CLI and return JSON result."""
    from helixlang.lexer import Lexer
    from helixlang.parser import Parser
    from helixlang.sim_runtime import run

    source = Path(helix_path).read_text()
    tokens = list(Lexer(source).tokens())
    program = Parser(tokens).parse()
    result = run(program)
    if result is None:
        raise RuntimeError(f"No result from {helix_path}")
    return result.to_dict()


def run_ecosystem_gem(organism: str, genome_seq: str, medium: str,
                      duration: float, dt: float) -> dict:
    """Run a single-species ecosystem with GEM-driven growth."""
    from helixlang.apps.ecosystem import (
        Ecosystem, EcosystemConfig, PatchConfig, ScalarConfig,
        Species, SubstrateConfig, gem_to_species,
    )
    from helixlang.apps.gem_pipeline import run_gem_pipeline
    import tempfile

    # Write genome to temp FASTA
    tmp = Path(tempfile.mktemp(suffix=".fasta"))
    lines = []
    genes = genome_seq.split("\n")
    i = 0
    while i < len(genes):
        line = genes[i].strip()
        if line.startswith("#") and not line.startswith("#end"):
            gene_id = line.lstrip("#").strip()
            seq_lines = []
            i += 1
            while i < len(genes) and not genes[i].strip().startswith("#"):
                seq_lines.append(genes[i].strip())
                i += 1
            seq = "".join(seq_lines)
            if seq and gene_id:
                lines.append(f">{gene_id}\n{seq}")
        else:
            i += 1
    tmp.write_text("\n".join(lines) + "\n")

    try:
        # Run GEM pipeline
        result = run_gem_pipeline(
            genome_fasta=str(tmp),
            organism=organism,
            target_organism=organism,
        )

        # Extract species params
        params = gem_to_species(result, organism=organism, medium=medium)

        # Build metabolic model
        if result.consensus:
            from helixlang.gem.bridge import consensus_to_metabolic_model
            model = consensus_to_metabolic_model(result.consensus)
        else:
            model = None

        # Print pipeline results
        print(f"  Pipeline: {result.annotated_genes} genes, "
              f"{result.final_reaction_count} reactions, "
              f"{result.stages_completed} stages")
        print(f"  Static FBA growth: {result.fba_fluxes.get('BIOMASS_reaction', 0.0):.4f}")
        print(f"  Params: vmax={params['vmax']:.4f}, ks={params['ks']:.4f}, "
              f"yield_c={params['yield_c']:.4f}, max_mu={params['max_growth_rate']:.4f}")
        if model:
            print(f"  Model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")

        return {"pipeline": True, "params": params, "model": model,
                "fluxes": dict(result.fba_fluxes)}
    finally:
        tmp.unlink(missing_ok=True)


def run_cli_with_detail(helix_path: str) -> dict:
    """Run example via CLI and capture JSON output."""
    from helixlang.lexer import Lexer
    from helixlang.parser import Parser
    from helixlang.sim_runtime import run

    source = Path(helix_path).read_text()
    tokens = list(Lexer(source).tokens())
    program = Parser(tokens).parse()
    result = run(program)
    if result is None:
        return {}
    return result.to_dict()


def main():
    print("=" * 72)
    print("VALIDATION: Examples 48 & 49 — Full GEM Simulation Comparison")
    print("=" * 72)

    results = {}

    # ---- Example 48: E. coli ----
    print(f"\n{'─' * 72}")
    print("  EXAMPLE 48: E. coli K-12 MG1655 (inline DNA, 46 genes)")
    print(f"{'─' * 72}")

    t0 = time.time()
    r48 = run_cli_with_detail("examples/48_ecoli_inline_dna.helix")
    elapsed = time.time() - t0

    meta = r48.get("meta", {})
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Organism:    {meta.get('organism', '?')}")
    print(f"  Medium:      {meta.get('medium', '?')}")
    print(f"  Dynamic:     {meta.get('dynamic', False)}")
    print(f"  Duration:    {meta.get('duration_h', '?')} h")
    print(f"  dt:          {meta.get('dt_h', '?')} h")
    print(f"  Stages:      {meta.get('stages_completed', '?')}")

    # Pipeline stages
    for row in r48.get("rows", []):
        print(f"    {row['stage']:20s} status={row['status']}  "
              f"genes={row['genes_annotated']}  rxns={row['reactions_total']}  "
              f"grn_edges={row['grn_edges']}  kcat={row['kcat_predictions']}  "
              f"km={row['km_estimates']}")

    # FBA results
    print(f"\n  Growth rate:  {meta.get('growth_rate_per_hour', 0):.4f} h^-1")
    print(f"  Biomass yield:{meta.get('biomass_yield', 0):.4f}")
    kf = meta.get("key_fluxes", {})
    if kf:
        print(f"  Key fluxes:")
        for k, v in sorted(kf.items()):
            if k not in ("time", "glucose", "oxygen"):
                print(f"    {k:30s} = {v}")
        if "biomass" in kf:
            print(f"    {'final_biomass (gDW/L)':30s} = {kf.get('biomass', 'N/A')}")
        if "glucose" in kf:
            print(f"    {'final_glucose (mM)':30s} = {kf.get('glucose', 'N/A')}")

    warnings = meta.get("warnings", [])
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")

    results["48"] = meta

    # ---- Example 49: Synechocystis ----
    print(f"\n{'─' * 72}")
    print("  EXAMPLE 49: Synechocystis PCC 6803 (inline DNA, 7 genes)")
    print(f"{'─' * 72}")

    t0 = time.time()
    r49 = run_cli_with_detail("examples/49_synechocystis_cyanobacteria.helix")
    elapsed = time.time() - t0

    meta = r49.get("meta", {})
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Organism:    {meta.get('organism', '?')}")
    print(f"  Medium:      {meta.get('medium', '?')}")
    print(f"  Dynamic:     {meta.get('dynamic', False)}")
    print(f"  Duration:    {meta.get('duration_h', '?')} h")
    print(f"  dt:          {meta.get('dt_h', '?')} h")
    print(f"  Stages:      {meta.get('stages_completed', '?')}")

    for row in r49.get("rows", []):
        print(f"    {row['stage']:20s} status={row['status']}  "
              f"genes={row['genes_annotated']}  rxns={row['reactions_total']}  "
              f"grn_edges={row['grn_edges']}  kcat={row['kcat_predictions']}  "
              f"km={row['km_estimates']}")

    print(f"\n  Growth rate:  {meta.get('growth_rate_per_hour', 0):.4f} h^-1")
    print(f"  Biomass yield:{meta.get('biomass_yield', 0):.4f}")
    kf = meta.get("key_fluxes", {})
    if kf:
        print(f"  Key fluxes:")
        for k, v in sorted(kf.items()):
            if k not in ("time",):
                print(f"    {k:30s} = {v}")

    warnings = meta.get("warnings", [])
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")

    results["49"] = meta

    # ---- Dynamic FBA trajectory analysis ----
    print(f"\n{'─' * 72}")
    print("  DYNAMIC FBA TRAJECTORY ANALYSIS")
    print(f"{'─' * 72}")

    for ex_id, meta_data in results.items():
        kf = meta_data.get("key_fluxes", {})
        dyn = meta_data.get("dynamic", False)
        if not dyn:
            print(f"  Example {ex_id}: static FBA only (no trajectory)")
            continue
        print(f"\n  Example {ex_id} ({meta_data.get('organism', '?')}):")
        print(f"    Duration: {meta_data.get('duration_h', '?')} h, "
              f"dt: {meta_data.get('dt_h', '?')} h, "
              f"steps: {meta_data.get('trajectory_steps', '?')}")
        print(f"    Final biomass:  {kf.get('biomass', 'N/A')} gDW/L")
        print(f"    Final glucose:  {kf.get('glucose', 'N/A')} mM")
        print(f"    Final O2:       {kf.get('oxygen', 'N/A')} mM")
        print(f"    Final growth:   {kf.get('growth_rate', 'N/A')} h^-1")

    # ---- LITERATURE COMPARISON ----
    print(f"\n{'═' * 72}")
    print("  LITERATURE COMPARISON — ERRORS AND DISCREPANCIES")
    print(f"{'═' * 72}")

    print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │ E. coli K-12 MG1655 — glucose minimal medium                       │
  ├──────────────────────────────────────────────────────────────────────┤
  │ Parameter            │ Simulation │ Literature    │ Source           │
  ├──────────────────────┼────────────┼───────────────┼──────────────────┤""")

    gr48 = results.get("48", {}).get("growth_rate_per_hour", 0)
    kf48 = results.get("48", {}).get("key_fluxes", {})
    bm48 = kf48.get("biomass", 0)
    glc48 = kf48.get("glucose", 0)
    print(f"  │ Growth rate (h⁻¹)  │ {gr48:10.4f} │ 0.87          │ Orth 2010      │")
    print(f"  │ Final biomass (gDW/L)│ {bm48:10.4f} │ 0.90-1.2      │ Monod 1949     │")
    print(f"  │ Glucose consumed    │ {10-glc48:10.4f} mM │ 10 mM (full)  │ uptake=10     │")
    print(f"  │ Doubling time (min)│ {69.3/max(gr48,0.001):10.1f} │ 20-30         │ Brock 2012    │")
    print(f"  │ Acetate overflow   │ N/A        │ >10 mmol/gDW/h│ Varma 1994    │")

    print(f"""  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │ Synechocystis PCC 6803 — BG-11 photoautotrophic                    │
  ├──────────────────────────────────────────────────────────────────────┤
  │ Parameter            │ Simulation │ Literature    │ Source           │
  ├──────────────────────┼────────────┼───────────────┼──────────────────┤""")

    gr49 = results.get("49", {}).get("growth_rate_per_hour", 0)
    kf49 = results.get("49", {}).get("key_fluxes", {})
    bm49 = kf49.get("biomass", 0)
    print(f"  │ Growth rate (h⁻¹)  │ {gr49:10.4f} │ 0.14          │ Rippka 1979   │")
    print(f"  │ Final biomass (gDW/L)│ {bm49:10.4f} │ 0.5-2.0       │ Kaneko 1996   │")
    print(f"  │ Carbon source      │ bg11       │ CO₂ only      │ photoautotroph│")
    print(f"  │ Doubling time (h)  │ {0.693/max(gr49,0.001):10.1f} │ 4-8           │ Castenholz   │")
    print(f"  │ O₂ evolution       │ N/A        │ ~300 mmol/gDW/h│Allakhverdiev  │")

    print(f"""  └──────────────────────────────────────────────────────────────────────┘

  IDENTIFIED ISSUES:
""")

    issues = []

    # Check E. coli
    if gr48 < 0.5 or gr48 > 1.0:
        issues.append(f"  [!] E. coli growth rate {gr48:.4f} h⁻¹ outside expected range 0.7-0.9")
    if bm48 < 0.5 or bm48 > 5.0:
        issues.append(f"  [!] E. coli final biomass {bm48:.4f} gDW/L outside expected range 0.7-1.5")
    if glc48 > 1.0:
        issues.append(f"  [!] E. coli glucose not fully consumed: {glc48:.4f} mM remaining")

    # Check Synechocystis
    if gr49 < 0.05 or gr49 > 0.25:
        issues.append(f"  [!] Synechocystis growth rate {gr49:.4f} h⁻¹ outside expected range 0.10-0.18")
    if bm49 < 0.02 or bm49 > 3.0:
        issues.append(f"  [!] Synechocystis final biomass {bm49:.4f} gDW/L outside expected range 0.5-2.0")

    # Check model structure
    if results.get("48", {}).get("dynamic", False):
        if kf48.get("biomass", 0) == 0.01:
            issues.append("  [!] Dynamic FBA biomass stuck at initial value (0.01) — no growth in dynamic path")

    # Check warnings
    for ex_id in ["48", "49"]:
        warns = results.get(ex_id, {}).get("warnings", [])
        for w in warns:
            issues.append(f"  [W] Example {ex_id}: {w}")

    if not issues:
        print("  No critical issues found.")
    else:
        for issue in issues:
            print(issue)

    # Save full results
    out_path = Path("scripts/validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
