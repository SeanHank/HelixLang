"""Full-chain custom organism pipeline (doc/26 Phase F).

Orchestrates the complete chain:
  FASTA -> translation -> ESM3 structure -> sequence kinetics -> ecGEM -> community FBA -> simulation
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_GENE_NAME_TO_EC: dict[str, str] = {
    "glucokinase": "2.7.1.2",
    "phosphofructokinase": "2.7.1.11",
    "phosphofructokinase_1": "2.7.1.11",
    "hexokinase": "2.7.1.1",
    "pyruvate_kinase": "2.7.1.40",
    "pyruvate kinase": "2.7.1.40",
    "citrate_synthase": "4.1.3.16",
    "citrate synthase": "4.1.3.16",
    "aconitase": "4.2.1.3",
    "isocitrate_dehydrogenase": "1.1.1.40",
    "isocitrate dehydrogenase": "1.1.1.40",
    "succinyl_coa_synthetase": "6.2.1.4",
    "succinate_dehydrogenase": "1.3.5.4",
    "fumarase": "4.2.1.2",
    "malate_dehydrogenase": "1.1.1.37",
    "malate dehydrogenase": "1.1.1.37",
    "pyruvate_dehydrogenase": "1.2.4.1",
    "pyruvate dehydrogenase": "1.2.4.1",
    "pep_carboxylase": "4.1.1.31",
    "phosphoglucose_isomerase": "5.3.1.9",
    "triosephosphate_isomerase": "5.3.1.1",
    "g3p_dehydrogenase": "1.2.1.12",
    "phosphoglycerate_kinase": "2.7.2.3",
    "phosphoglycerate_mutase": "4.1.1.32",
    "enolase": "4.2.1.11",
}


def _infer_ec_number(gene_id: str) -> str:
    key = gene_id.lower().strip()
    if key in _GENE_NAME_TO_EC:
        return _GENE_NAME_TO_EC[key]
    key_no_space = key.replace("-", "_").replace(" ", "_")
    if key_no_space in _GENE_NAME_TO_EC:
        return _GENE_NAME_TO_EC[key_no_space]
    return ""


@dataclass
class PipelineConfig:
    organism_name: str = "custom_organism"
    medium: str = "glucose_minimal"
    ecgem: bool = True
    community: bool = False
    ticks: int = 4320
    esm_model: str = "esmfold_v1"
    device: str | None = None
    enzyme_mass_fraction: float = 0.55
    dry_weight_conc: float = 0.3
    temperature_c: float = 37.0
    ph: float = 7.0
    num_esm_steps: int = 10
    max_residues: int = 700
    ec_map_path: str | None = None


@dataclass
class PipelineResult:
    proteins: list[Any] = field(default_factory=list)
    structures: dict[str, Any] = field(default_factory=dict)
    kcat_predictions: dict[str, Any] = field(default_factory=dict)
    km_predictions: dict[str, float] = field(default_factory=dict)
    ec_numbers: dict[str, str] = field(default_factory=dict)
    ecgem: Any | None = None
    community: Any | None = None
    simulation: dict[str, Any] = field(default_factory=dict)
    pipeline_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)


def run_full_pipeline(
    fasta_path: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    if config is None:
        config = PipelineConfig()

    t0 = time.time()
    result = PipelineResult()
    warnings: list[str] = []

    proteins = _stage_a_fasta(fasta_path)
    result.proteins = proteins
    result.stages_completed.append("A_fasta")
    if not proteins:
        warnings.append("no proteins found in FASTA")
        result.warnings = warnings
        result.pipeline_time = time.time() - t0
        return result

    ec_map = _load_ec_map(config.ec_map_path)
    ec_numbers: dict[str, str] = {}
    for p in proteins:
        gene_id = p.gene_id if hasattr(p, "gene_id") else f"protein_{len(ec_numbers)}"
        ec_numbers[gene_id] = ec_map.get(gene_id, _infer_ec_number(gene_id))
    result.ec_numbers = ec_numbers

    structures = _stage_b_structure(proteins, config)
    result.structures = structures
    result.stages_completed.append("B_structure")

    kcat_preds, km_preds = _stage_c_kinetics(proteins, ec_numbers, config)
    result.kcat_predictions = kcat_preds
    result.km_predictions = km_preds
    result.stages_completed.append("C_kinetics")

    ecgem_result = None
    if config.ecgem and kcat_preds:
        ecgem_result = _stage_d_ecgem(proteins, kcat_preds, ec_numbers, config)
        result.ecgem = ecgem_result
        result.stages_completed.append("D_ecgem")

    community_result = None
    if config.community and ecgem_result is not None:
        community_result = _stage_e_community(ecgem_result, config)
        result.community = community_result
        result.stages_completed.append("E_community")

    simulation = _stage_f_simulate(result, config)
    result.simulation = simulation
    result.stages_completed.append("F_simulation")

    result.warnings = warnings
    result.pipeline_time = time.time() - t0
    return result


def _load_ec_map(ec_map_path: str | None) -> dict[str, str]:
    if not ec_map_path:
        return {}
    path = Path(ec_map_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _is_nucleotide_sequence(seq: str) -> bool:
    """Check if a sequence is nucleotide (ATGC) rather than protein."""
    s = seq.upper().strip()
    if not s:
        return False
    nuc_chars = set("ATGC")
    nuc_count = sum(1 for c in s if c in nuc_chars)
    return nuc_count / len(s) > 0.85


def _translate_dna_to_protein(dna: str) -> str:
    """Translate DNA to protein using standard codon table (NCBI table 11)."""
    _CODON_TABLE: dict[str, str] = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i + 3].upper()
        aa = _CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def _stage_a_fasta(fasta_path: str) -> list[Any]:
    try:
        from helixlang.annotation.sequences import extract_protein_sequences
        proteins_dict = extract_protein_sequences(fasta_path)
        if not proteins_dict:
            return []

        @dataclass
        class _Protein:
            gene_id: str
            sequence: str

        result = []
        for k, v in proteins_dict.items():
            seq = v
            if _is_nucleotide_sequence(v):
                seq = _translate_dna_to_protein(v)
                if not seq:
                    continue
            result.append(_Protein(gene_id=k, sequence=seq))
        return result
    except Exception:
        return []


def _stage_b_structure(
    proteins: list[Any], config: PipelineConfig
) -> dict[str, Any]:
    from helixlang.protein_structure_predictor import is_available
    if not is_available():
        return {}
    from helixlang.protein_structure_predictor import predict_structure_esm
    structures: dict[str, Any] = {}
    for p in proteins:
        seq = p.sequence if hasattr(p, "sequence") else str(p)
        gene_id = p.gene_id if hasattr(p, "gene_id") else f"protein_{len(structures)}"
        try:
            struct = predict_structure_esm(
                seq,
                model_name=config.esm_model,
                device=config.device,
                num_steps=config.num_esm_steps,
                max_residues=config.max_residues,
            )
            structures[gene_id] = struct
        except Exception:
            continue
    return structures


def _stage_c_kinetics(
    proteins: list[Any],
    ec_numbers: dict[str, str],
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, float]]:
    from helixlang.kinetics.sequence_predictor import (
        SequenceKcatPredictor,
        SequenceKmEstimator,
    )
    kcat_pred = SequenceKcatPredictor()
    km_est = SequenceKmEstimator()
    kcat_preds: dict[str, Any] = {}
    km_preds: dict[str, float] = {}
    for p in proteins:
        seq = p.sequence if hasattr(p, "sequence") else str(p)
        gene_id = p.gene_id if hasattr(p, "gene_id") else f"protein_{len(kcat_preds)}"
        ec = ec_numbers.get(gene_id, "")
        kr = kcat_pred.predict(
            reaction_id=gene_id,
            sequence=seq,
            ec_number=ec,
        )
        kcat_preds[gene_id] = kr
        km_r = km_est.predict(seq, "glucose", ec_number=ec)
        km_preds[gene_id] = km_r.km_value
    return kcat_preds, km_preds


def _stage_d_ecgem(
    proteins: list[Any],
    kcat_preds: dict[str, Any],
    ec_numbers: dict[str, str],
    config: PipelineConfig,
) -> Any | None:
    try:
        from helixlang.gem.ecgem import ECGEMBuilder
        kcat_dict: dict[str, float] = {}
        sequences: dict[str, str] = {}
        for p in proteins:
            gene_id = p.gene_id if hasattr(p, "gene_id") else f"protein_{len(kcat_dict)}"
            pred = kcat_preds.get(gene_id)
            if pred is not None and hasattr(pred, "kcat_value"):
                kcat_dict[gene_id] = pred.kcat_value
            seq = p.sequence if hasattr(p, "sequence") else ""
            if seq:
                sequences[gene_id] = seq
        builder = ECGEMBuilder(
            kcat_predictions=kcat_dict,
            ec_numbers=ec_numbers,
            sequences=sequences,
            organism=config.organism_name,
            enzyme_mass_fraction=config.enzyme_mass_fraction,
            dry_weight_conc=config.dry_weight_conc,
        )
        return builder.build()
    except Exception:
        return None


def _stage_e_community(
    ecgem_result: Any,
    config: PipelineConfig,
) -> Any | None:
    try:
        from helixlang.gem.community import CommunityFBAExtended, OrganismModel
        from helixlang.metabolism import FluxBalanceAnalysis
        exchange_reactions = []
        production: dict[str, float] = {}
        consumption: dict[str, float] = {}
        try:
            fba = FluxBalanceAnalysis(ecgem_result.model)
            if ecgem_result.model.biomass_reaction:
                fluxes = fba.solve(
                    objective=ecgem_result.model.biomass_reaction, maximize=True
                )
                for rxn_id, rxn in ecgem_result.model.reactions.items():
                    if rxn.subsystem == "exchange" or rxn_id.startswith("EX_"):
                        exchange_reactions.append(rxn_id)
                        flux = fluxes.get(rxn_id, 0.0)
                        met = ""
                        for m, _c in rxn.stoichiometry.items():
                            met = m
                            break
                        if met:
                            if flux > 0:
                                production[met] = flux
                            elif flux < 0:
                                consumption[met] = abs(flux)
        except Exception:
            pass
        org = OrganismModel(
            organism_id=config.organism_name,
            model=ecgem_result.model,
            ecgem=ecgem_result,
            exchange_reactions=exchange_reactions,
            production=production,
            consumption=consumption,
        )
        community = CommunityFBAExtended(organisms=[org])
        return community.solve()
    except Exception:
        return None


def _stage_f_simulate(
    result: PipelineResult,
    config: PipelineConfig,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "organism": config.organism_name,
        "medium": config.medium,
        "stages_completed": result.stages_completed,
        "num_proteins": len(result.proteins),
        "num_structures": len(result.structures),
        "ec_numbers_resolved": len([v for v in result.ec_numbers.values() if v]),
    }
    if result.ecgem is not None:
        info["ecgem_growth_rate"] = result.ecgem.growth_rate
        info["ecgem_growth_unconstrained"] = result.ecgem.growth_rate_unconstrained
        info["ecgem_enzyme_constraints"] = len(result.ecgem.enzyme_constraints)
        info["ecgem_warnings"] = result.ecgem.warnings
    if result.community is not None:
        info["community_total_biomass"] = result.community.total_biomass
        info["community_converged"] = result.community.converged

    if result.ecgem is not None and result.ecgem.model is not None:
        sim = _run_dfba_simulation(result.ecgem.model, config)
        info["dfba"] = sim
    elif config.ticks > 0:
        info["dfba"] = {"status": "skipped_no_model", "ticks": config.ticks}

    info["pipeline_time_s"] = result.pipeline_time
    return info


def _run_dfba_simulation(model: Any, config: PipelineConfig) -> dict[str, Any]:
    """Run dFBA simulation using DynamicFluxBalance."""
    try:
        from helixlang.metabolism import DynamicFBAConfig, DynamicFluxBalance
        dfba_config = DynamicFBAConfig(
            dt_h=max(config.ticks / 1000.0, 0.01),
            initial_biomass_gdw=0.05,
            initial_glucose_mm=10.0,
            max_glucose_uptake=10.0,
            glucose_half_saturation_mm=0.1,
        )
        dfba = DynamicFluxBalance(model=model, config=dfba_config)
        history = dfba.run(duration_h=24.0, max_steps=config.ticks)
        if not history:
            return {"status": "no_history", "ticks": config.ticks}
        final = history[-1]
        return {
            "status": "completed",
            "ticks_ran": len(history),
            "final_time_h": final.get("time", 0.0),
            "final_biomass": final.get("biomass", 0.0),
            "final_glucose": final.get("glucose", 0.0),
            "peak_growth_rate": max(h.get("growth_rate", 0.0) for h in history),
            "duration_h": config.ticks * dfba_config.dt_h,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "ticks": config.ticks}


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "run_full_pipeline",
]
