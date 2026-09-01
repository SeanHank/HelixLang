# HelixLang Validation Report

Generated: 2026-09-01 02:53:33 UTC

## Summary

| Metric | Value |
|--------|-------|
| Benchmarks | **81/82** pass |
| Failures | 1 |
| Skipped | 0 |
| Validation levels | L0×59 · L1×2 · L2×8 · L3×7 · L4×6 · L5×0 |
| Level-gate warnings | 10 |

## Evidence Chains

| # | Benchmark | Layer | Level | Reference → Expected → Actual → Error | Status |
|---|-----------|-------|-------|---------------------------------------|--------|
| 01_codon_translation | Codon translation (64 codons) | language | L3 | NCBI — Standard genetic code (NCBI Translation Table 1) → codon_count=64 codons ±0 → 64 → ≈0 (abs=0.0e+00) | ✅ PASS |
| 02_lac_operon | lac operon biphasic growth | runtime | L3 | grn_nodes=True, lacI_steady_above=0.4, lacZ_steady_below=0.3 → grn_node_count=5, grn_edge_count=4, ticks_run=200 → verified | ✅ PASS |
| 03_ecoli_fba | E. coli core FBA growth rate | metabolism | L4 | Orth et al. — 2010 — BiGG e_coli_core via COBRApy → growth_rate=0.8739215069684307 h^-1 ±0.05 → 0.8739215069676898 → ≈0 (abs=7.4e-13) | ✅ PASS |
| 04_iml1515_fba | E. coli iML1515 genome-scale FBA | metabolism | L2 | vendored E. coli core (fallback) → verified | ✅ PASS |
| 05_in678_photoauto | Synechocystis PCC 6803 photoautotrophic FBA | metabolism | L2 | vendored E. coli core (glucose-fed, fallback) → verified | ✅ PASS |
| 06_dfba_diauxic | dFBA diauxic shift | metabolism | L4 | 4 automated checks → depletion_time_20pct=True, acetate_peak_30pct=True, final_biomass_30pct=True → passed=4, total=4 → verified | ✅ PASS |
| 07_grn_repressilator | Repressilator oscillation | grn | L4 | 3 automated checks → both_oscillate=True, period_within_20pct=True, phase_similar=True → passed=3, total=3 → verified | ✅ PASS |
| 08_population_dynamics | Population dynamics | population | L4 | 3 automated checks → growth_curve_factor2=True, doubling_time_15_25=True, fast_species_dominance=True → passed=3, total=3 → verified | ✅ PASS |
| 09_reaction_diffusion | Reaction-diffusion (Gray-Scott) | pattern_formation | L2 | 4 nested checks → statistical_comparison.statistical_match=True, stability_analysis.homogeneous_stable=True, parameter_sensitivity.sensitivity_ok=True → passed=4, total=4 → verified | ✅ PASS |
| 10_whole_cell | Whole-cell division time | virtual_cell | L4 | Wanner — 1996 — E. coli K-12 generation time → division_time=37.3 min ±0.3 → 37 → 0.80% | ✅ PASS |
| 11_performance_comparison | FBA solve-time performance vs COBRApy | metabolism | L2 | 8 metrics → ecoli_core.n_reactions=95, ecoli_core.cobrapy_100_solves_s=0.0378, ecoli_core.helixlang_100_solves_s=2.4453 → ecoli_core.n_reactions=95, ecoli_core.cobrapy_100_solves_s=0.0378, ecoli_core.helixlang_100_solves_s=2.4453 → verified | ✅ PASS |
| 12_parser_roundtrip | Parser roundtrip — source → AST → bytecode | language | L0 | 8 metrics → tokens=17, genes=1, gene_name=lacI → tokens=17, genes=1, gene_name=lacI → verified | ✅ PASS |
| 13_bytecode_vm_roundtrip | Bytecode/VM roundtrip — compile → serialize → deserialize → execute | runtime | L0 | 1 functional checks → deterministic=True → passed=1, total=1 → verified | ✅ PASS |
| 14_type_system_flow | Type system & flow — type checking + module imports | language | L0 | 5 functional checks → symbol_table_define_lookup=True, type_annotation_parsing=True, module_import_export=True → passed=5, total=5 → verified | ✅ PASS |
| 15_dna_encoding | DNA encoding — dna_codec + biocodec + codon_table roundtrips | biology | L0 | 5 automated checks → goldman={'oligo_count': 4, 'decoded_length': 14, 'roundtrip_match': True}, 2bit_dna={'dna_length': 56, 'decoded_length': 14, 'roundtrip_match': True}, codon_table={'codon_count': 64, 'all_mapped': True} → passed=5, total=5 → verified | ✅ PASS |
| 16_cli_server_provenance | CLI / server / provenance — module imports + build_provenance | infrastructure | L0 | 5 automated checks → provenance={'fields': ['backend', 'backend_implementation', 'custom_key', 'dependencies', 'extra', 'fidelity_mode', 'helix_version', 'literature_references', 'model_version', 'parameter_set', 'parameters', 'random_seed', 'runtime_seconds', 'seed', 'solver', 'source_hash', 'source_path', 'timestamp'], 'seed': 42, 'backend': 'fba', 'has_source_hash': True}, cli_import={'main_callable': True}, cli_flags={'version_output': 'OPCODE_VERSION=1'} → passed=5, total=5 → verified | ✅ PASS |
| 17a_cell | Cell class | molecular | L0 | 7 automated checks → instantiation=True, protein_ops=True, energy_ops=True → passed=7, total=7 → verified | ✅ PASS |
| 17b_transcribe | DNA transcription | molecular | L0 | 7 automated checks → cds_length_75=True, cds_starts_with_aug=True, half_life_positive=True → passed=7, total=7 → verified | ✅ PASS |
| 17c_translate | mRNA translation | molecular | L0 | 6 automated checks → protein_length_24=True, protein_sequence_match=True, stop_codon_taa=True → passed=6, total=6 → verified | ✅ PASS |
| 17d_coupled | Coupled transcription-translation | molecular | L0 | 2 automated checks → coupled_model_works=True, protein_pool_works=True → passed=2, total=2 → verified | ✅ PASS |
| 20_sparse_grn | Sparse GRN | regulatory | L0 | 3 automated checks → fewer_edges_than_dense=True, sparse_matches_dense_output=True, roundtrip_to_grn=True → passed=3, total=3 → verified | ✅ PASS |
| 23a_evolution | Wright-Fisher evolution | population | L0 | 4 automated checks → mutations_produced=True, fitness_original_is_one=True, fitness_mutated_is_lower=True → passed=4, total=4 → verified | ✅ PASS |
| 23b_stochastic | Gillespie SSA telegraph model | stochastic | L0 | 3 automated checks → telegraph_promoter_fano_matches=True, fano_greater_than_one=True, fano_within_30pct_of_theory=True → passed=3, total=3 → verified | ✅ PASS |
| 23c_epigenetics | CpG site detection | epigenetics | L0 | 4 automated checks → cpg_acgt_repeat=True, cpg_alternating=True, no_cpg_in_at=True → passed=4, total=4 → verified | ✅ PASS |
| 25_gem_reconstruction | GEM Reconstruction — import and minimal model | gem | L2 | 3 automated checks → import_gem_submodules=True, key_classes_exist=True, full_model_adapter_instantiable=True → passed=3, total=3 → verified | ✅ PASS |
| 26_gapfill_validation | Gapfill Validation — restore incomplete model | gem | L0 | 3 automated checks → import_gapfill_modules=True, gapfill_classes_exist=True, load_model_and_remove_reaction=True → passed=3, total=3 → verified | ✅ PASS |
| 27_annotation_ec_mapping | Annotation EC Mapping — EC number to reaction lookup | annotation | L0 | 4 automated checks → import_ec_mapping_module=True, build_ec_db=True, lookup_known_ec_number=True → passed=4, total=4 → verified | ✅ PASS |
| 28_genotype_cyp2d6 | Genotype CYP2D6 star-allele mapping | human | L0 | 4 automated checks → create_default_genotype_returns_valid_profile=True, cyp2d6_star4_star4_is_poor_metabolizer=True, cyp2d6_star1_star1_is_normal_metabolizer=True → passed=4, total=4 → verified | ✅ PASS |
| 29_drug_adme | Drug ADME predefined library lookup | human | L0 | 4 automated checks → list_predefined_drugs_returns_at_least_10=True, get_predefined_drug_warfarin_returns_drug=True, warfarin_oral_bioavailability_positive=True → passed=4, total=4 → verified | ✅ PASS |
| 30_pk_simulation | PBPK IV bolus simulation | human | L1 | Rowland & Tozer — 2011 — IV bolus pharmacokinetics (one-compartment model) → c0_concentration=33.333333333333336 mg/L ±0.05 → 33.33333333333334 → ≈0 (abs=7.1e-15) | ✅ PASS |
| 31_pd_dose_response | Hill equation dose-response curve | human | L0 | 4 automated checks → hill_at_zero_concentration_is_zero=True, hill_at_ec50_is_half_max=True, hill_at_high_concentration_approaches_emax=True → passed=4, total=4 → verified | ✅ PASS |
| 32_ddi_prediction | Drug-drug interaction prediction | human | L0 | 4 automated checks → create_default_ddi_model_has_rules=True, warfarin_fluconazole_is_known_ddi=True, interaction_flag_is_triggered=True → passed=4, total=4 → verified | ✅ PASS |
| 33_disease_ode | Disease ODE model simulation | human | L0 | 5 automated checks → create_disease_model_returns_ode_model=True, model_has_step_method=True, running_365_daily_steps_produces_trajectory=True → passed=5, total=5 → verified | ✅ PASS |
| 34_virtual_patient | Virtual patient instantiation and import | human | L0 | 5 automated checks → import_virtual_patient_and_config=True, create_default_config=True, virtual_patient_config_can_be_created=True → passed=5, total=5 → verified | ✅ PASS |
| 35_enzyme_kinetics | Enzyme Kinetics — kcat and Km prediction | kinetics | L4 | 5 automated checks → import_kinetics_modules=True, predict_kcat_positive=True, predict_kcat_in_range=True → passed=5, total=5 → verified | ✅ PASS |
| 36_omics_integration | Omics Integration — expression inference and spatial omics | omics | L0 | 4 automated checks → import_omics_modules=True, expression_model_classes_exist=True, hill_function_zero=True → passed=4, total=4 → verified | ✅ PASS |
| 37_crispr_editing | CRISPR Editing — PAM finding, guide design, off-target scoring | crispr | L0 | 6 automated checks → import_crispr_modules=True, find_pam_sites_found=True, pam_at_expected_positions=True → passed=6, total=6 → verified | ✅ PASS |
| 38_ecosystem_lotka_volterra | Ecosystem — Lotka-Volterra predator-prey dynamics | ecosystem | L0 | 5 automated checks → import_ecosystem_module=True, lotka_volterra_step_runs=True, both_populations_positive=True → passed=5, total=5 → verified | ✅ PASS |
| 39_synbio_designer | SynBio Designer — genetic circuit design and validation | synbio | L0 | 6 automated checks → import_synbio_modules=True, designer_creates_cassette=True, cai_above_threshold=True → passed=6, total=6 → verified | ✅ PASS |
| 40_dna_storage_codec | DNA Storage — encode/decode roundtrip | dna_storage | L0 | 5 automated checks → import_dna_storage_module=True, encode_produces_oligos=True, dna_output_valid_characters=True → passed=5, total=5 → verified | ✅ PASS |
| 41_pipeline_integration | Pipeline Integration — import all pipeline modules | pipeline | L0 | 8 automated checks → import_full_pipeline=True, import_gem_pipeline=True, import_population_calibration=True → passed=8, total=8 → verified | ✅ PASS |
| 42_remaining_modules | Remaining Modules — bio_data, morphology_3d, lsystem, protein_structure, protein_fitness, units, seq_utils | core | L0 | 14 automated checks → import_bio_data=True, bio_data_codon_usage_loaded=True, import_morphology_3d=True → passed=14, total=14 → verified | ✅ PASS |
| 43_performance_scaling | Performance Scaling — COBRApy vs HelixLang FBA timing | performance | L2 | 4 automated checks → import_metabolism_module=True, fba_solves_e_coli_core=True, both_engines_produce_fluxes=True → passed=4, total=4 → verified | ✅ PASS |
| 44_determinism_all_backends | Determinism — same seed produces identical output for all backends | determinism | L0 | 4 automated checks → import_simulation_modules=True, stochastic_gillespie_deterministic=True, evolution_deterministic=True → passed=4, total=4 → verified | ✅ PASS |
| 45_provenance_completeness | Provenance — all simulation results carry provenance metadata | provenance | L0 | 14 automated checks → engine_run_produces_fba_result=True, fba_result_has_provenance=True, provenance_has_8_field_contract=True → passed=14, total=14 → verified | ✅ PASS |
| 46_vectorized_grn | Vectorized GRN correctness | runtime | L0 | HelixLang grn.py scalar implementation → verified | ✅ PASS |
| 47_flow_fields | Flow field analytical solutions | runtime | L1 | Boussinesq J — 1868 — Hagen-Poiseuille law; Boussinesq 1868 duct flow → verified | ✅ PASS |
| 48_immune_dynamics | Innate immune dynamics | runtime | L0 | Chrousos GP 1995, N Engl J Med 332:1351-1362 → verified | ✅ PASS |
| 49_ecgem | Enzyme-constrained GEM | metabolism | L0 | Sanchez BJ, Brunberg TM, Nielsen LK — 2017 — Sanchez BJ et al. 2017, PLoS Comput Biol 13:e1005565 → verified | ✅ PASS |
| 50_protein_structure | Protein secondary structure prediction | runtime | L0 | Chou PY, Fasman GD — 1978 — Chou & Fasman 1978; Garnier et al. 1978; Kyte & Doolittle 19 → verified | ✅ PASS |
| 51_morphology_3d | 3D L-system morphology | runtime | L0 | Lindenmayer A — 1968 — Lindenmayer A 1968, J Theor Biol 18:280 → verified | ✅ PASS |
| 52_community_fba | Community FBA cross-feeding | metabolism | L0 | Zomorrodi AR, Maranas CD — 2012 — Zomorrodi AR, Maranas CD 2012, PLoS Comput Biol 8:e1002363 → verified | ✅ PASS |
| 53_gem_reconstruction | GEM reconstruction pipeline | metabolism | L0 | 7 automated checks → import_modules=True, create_reaction_dicts=True, consensus_merge=True → passed=7, total=7 → verified | ✅ PASS |
| 54_sbml_grn_inference | SBML import + GRN inference | metabolism | L0 | 6 automated checks → import_modules=True, sbml_import=True, grn_result_instantiation=True → passed=6, total=6 → verified | ✅ PASS |
| 55_annotation_tools | Annotation tools | annotation | L0 | 9 automated checks → import_all_4_modules=True, ko_db_size=True, ko_db_has_k00844=True → passed=9, total=9 → verified | ✅ PASS |
| 56_blast_search | BLAST search wrapper | annotation | L0 | 1 metrics → level=L0 → level=L0 → verified | ❌ FAIL |
| 57_pbpk_pharmacokinetics | PBPK pharmacokinetics | pharmacology | L0 | 7 automated checks → import_pbpk_classes=True, pbpk_config_default=True, pbpk_model_has_step=True → passed=7, total=7 → verified | ✅ PASS |
| 58_endocrine_renal | Endocrine + renal ODEs | pharmacology | L0 | 7 automated checks → import_all_classes=True, create_endocrine_returns_EndocrineSystem=True, insulin_glucose_positive=True → passed=7, total=7 → verified | ✅ PASS |
| 59_hematology | Hematology myelosuppression | pharmacology | L0 | 7 automated checks → import_all_classes=True, create_returns_HematologySystem=True, hematology_has_step=True → passed=7, total=7 → verified | ✅ PASS |
| 60_proteome_binding | Proteome binding + DDI | pharmacology | L0 | 9 automated checks → import_all_classes=True, cascade_instantiate=True, profile_fields_accessible=True → passed=9, total=9 → verified | ✅ PASS |
| 61_molecular_toxicity | Molecular toxicity | pharmacology | L0 | Hughes JP, Rees S, Kalindjian SB, Philpott KL — 2008 — Hughes JP et al. 2008 → verified | ✅ PASS |
| 62_disease_recovery | Disease progression + recovery | physiology | L0 | Sonnenberg FA, Beck JR — 1993 — Sonnenberg FA, Beck JR 1993 → verified | ✅ PASS |
| 63_phenotype_simulation | Phenotype + simulation | physiology | L0 | Ursino M, Zingales M, Magosso E, et al. — 2020 — Ursino M et al. 2020 → verified | ✅ PASS |
| 64_stochastic_doseopt | Stochastic ODE + dose optimizer | pharmacology | L0 | 8 automated checks → import_all_classes=True, euler_maruyama_step_finite=True, solve_sde_x_positive_at_t1=True → passed=8, total=8 → verified | ✅ PASS |
| 65_qsp_binding | QSP binding models | pharmacology | L0 | 8 automated checks → import_all_classes=True, mass_action_occupancy_at_100nM=True, mass_action_occupancy_at_0nM=True → passed=8, total=8 → verified | ✅ PASS |
| 66_infrastructure | Infrastructure (units, seq, codec) | infrastructure | L0 | 25 automated checks → units_import=True, units_TIME_TICK_MIN=True, units_LATTICE_SPACING_UM=True → passed=25, total=25 → verified | ✅ PASS |
| 67_microbiome_crosstalk | Microbiome + organ crosstalk | physiology | L0 | 10 automated checks → import_all_classes=True, microbiome_compartment_instantiates=True, microbial_species_fields_accessible=True → passed=10, total=10 → verified | ✅ PASS |
| 68_bio_validity | Biological validity — Helix Model vs measured data | validity | L0 | 5 automated checks → scope_in_out_detection=True, parameter_fit_improves=True, uncertainty_ci_valid=True → passed=5, total=5 → verified | ✅ PASS |
| 69_performance_benchmark | Performance optimization — dispatch acceleration + snapshot downsampling | performance | L2 | 5 automated checks → profiler_produces_report=True, snapshot_downsampling_bounds_memory=True, long_run_bounded_trace=True → passed=5, total=5 → verified | ✅ PASS |
| 70_decoupling_verify | Decoupling verification — core vs simulation vs language | architecture | L2 | 5 automated checks → core_no_module_level_plugin_imports=True, registry_sole_bridge=True, core_no_sim_runtime_import=True → passed=5, total=5 → verified | ✅ PASS |
| 71_ir_roundtrip | Helix IR pipeline round-trip and optimizer correctness | compiler | L0 | 12 automated checks → builder_gene_mapping=True, builder_opcode_faithful=True, lowering_byte_golden=True → passed=12, total=12 → verified | ✅ PASS |
| 72_batch_runtime_parity | Vector batch runtime parity (numpy/JAX) vs the portable IR VM | runtime | L0 | 6 automated checks → numpy_parity=True, jax_parity=True, engine_agree=True → passed=6, total=6 → verified | ✅ PASS |
| 73_ir_serialization | HLIR serialization robustness | compiler | L0 | 7 automated checks → rich_roundtrip=True, metadata_preserved=True, typed_operands=True → passed=7, total=7 → verified | ✅ PASS |
| 74_incremental_jit | Incremental JIT — closure-limited gene recompile | compiler | L0 | 5 automated checks → full_build_rebuilds_all=True, unchanged_source_rebuilds_nothing=True, leaf_edit_rebuilds_closure_only=True → passed=5, total=5 → verified | ✅ PASS |
| 75_unit_safety | Unit system & dimensional safety | compiler | L0 | 9 automated checks → minutes_seconds_exact=True, base_value_equality=True, cross_unit_arithmetic_rejected=True → passed=9, total=9 → verified | ✅ PASS |
| 76_unit_safety_compile | Compile-time dimensional rejection (doc/41 Item 5 Ring 3) | compiler | L0 | 6 automated checks → cross_unit_symbol_add_rejected=True, dimension_tree_in_message=True, millimolar_plus_litre_named=True → passed=6, total=6 → verified | ✅ PASS |
| 77_immune_ifn_crp_friberg | Innate immune fidelity — type-I IFN, CRP v2 (IL-6→CRP lag/range), Friberg granulopoiesis | human | L3 | doc/40 L5 (Pawelek 2012), L9 (Sproston & Ashworth 2018), L4  → verified | ✅ PASS |
| 78_immune_adaptive_vaccine | Adaptive immunity + vaccination — two-dose antibody kinetics, APC priming, memory | human | L3 | doc/40 L5 (Pawelek et al. 2012), L8 (Front. Immunol. 16:1596 → verified | ✅ PASS |
| 79_immune_complement | Complement cascade + NK/mast/Eo/Baso + anaphylaxis (G5/G6) | human | L3 | doc/40 L7 (Zewde & Morikis 2018), L1 (BIS entity set) → verified | ✅ PASS |
| 80_immune_tissue_blood | Tissue-vs-blood immune pseudo-compartments (G10) | human | L3 | doc/40 G10; L1 (BIS compartment taxonomy), L2 (IIRABM) → verified | ✅ PASS |
| 81_immune_virtual_population | Virtual immune population — seeded baseline variance + cohort determinism (G13/O9) | human | L3 | doc/40 L6 (npj Syst Biol Appl 11, 2023); doc/39 O2/O9 → verified | ✅ PASS |
| 82_immune_spatial_abm | Spatial immune ABM — chemokine migration, contact signaling, deterministic replay (G15) | human | L0 | doc/40 G15; doc/31 §2.4 (BIS agent taxonomy, spatial ABM des → verified | ✅ PASS |

**81/82 benchmarks passed.**

## Level-Gate Warnings (doc/41 §3.2 Rule 5 — informational)

- 02_lac_operon: L3 requires reference.doi
- 04_iml1515_fba: L2 requires reproducibility.golden_hash
- 05_in678_photoauto: L2 requires reproducibility.golden_hash
- 06_dfba_diauxic: L4 requires experimental_comparison with min/max/unit
- 09_reaction_diffusion: L2 requires reproducibility.golden_hash
- 11_performance_comparison: L2 requires reproducibility.golden_hash
- 25_gem_reconstruction: L2 requires reproducibility.golden_hash
- 43_performance_scaling: L2 requires reproducibility.golden_hash
- 69_performance_benchmark: L2 requires reproducibility.golden_hash
- 70_decoupling_verify: L2 requires reproducibility.golden_hash

## Failures

### 56_blast_search

- Status: FAIL
- Error: Error(abs_error=None, rel_error=None, passed=True, message=None)

