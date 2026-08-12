# Production-Grade Upgrade Plan

> **HISTORICAL.** This is the plan for a completed earlier upgrade round; the
> constant values and line numbers it cites are snapshots from that era. The
> current runtime uses physical units end-to-end (see `src/helixlang/units.py`,
> `doc/simulation-model.md` §6.3, and `doc/gameplay-units-upgrade.md`).

> Goal: replace every simplified / education-oriented implementation in `src/helixlang/` with engineering-grade equivalents backed by primary literature, while preserving the public API surface, the compiler/VM pipeline, and a fully green test suite.
>
> Baseline: 1382 tests passing under `/opt/anaconda3/envs/helix/bin/python`; `ruff check` and `mypy` clean.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Audit Ledger — Current Simplifications](#2-audit-ledger--current-simplifications)
3. [Primary Literature for the Upgrade](#3-primary-literature-for-the-upgrade)
4. [Tiered Implementation Plan](#4-tiered-implementation-plan)
   - [Tier 1: Paper-implementable, highest value](#tier-1)
   - [Tier 2: Replace uncited engine constants](#tier-2)
   - [Tier 3: Documentation-only](#tier-3)
5. [Compatibility and API Preservation](#5-compatibility-and-api-preservation)
6. [Verification Strategy](#6-verification-strategy)
7. [Implementation Batches](#7-implementation-batches)

---

## 1. Background and Motivation

HelixLang is a DSL whose biology modules were deliberately built "pure-Python, educational, dependency-light". The audit (Section 2) shows two distinct constant classes:

- **Real, cited constants** — trustworthy, do not touch: GAM/ATP maintenance (Orth 2010), mutation rates (Drake/Lee 2012), PCR/synthesis error rates (Saiki 1988 / Potapov 2017 / Filges 2021), Gray–Scott presets (Pearson 1993), Doench/Hsu scoring tables, CUTG/Kazusa codon tables, transcription/translation elongation rates (Proshkin 2010 / Ingolia 2009).
- **Made-up, uncited constants** — the upgrade targets: arbitrary energy/threshold units, home-grown scoring heuristics, single-value half-lives, and simplified algorithm variants that deviate from the published method.

The upgrade has three hard constraints:

1. **Do not change existing functionality** (the language, bytecode, VM semantics, and every documented API must keep working).
2. **Keep all tests green** — 1382 tests under the `helix` conda env, plus `ruff` + `mypy` gates.
3. **Stay dependency-light** — pure-Python and stdlib first; numpy/biopython/reedsolo only where already optional.

---

## 2. Audit Ledger — Current Simplifications

Full-file audit of all 31 modules in `src/helixlang/` (plus `protein_structure.py`, documented separately). Line numbers are exact as of this document.

### 2.1 Real + cited (NO ACTION)

| Module | Item | Citation |
|---|---|---|
| `central_dogma.py` | transcription 50 nt/s, translation 20 aa/s, 30 nt coupling offset | Proshkin 2010, Ingolia 2009, Miller 1972 |
| `metabolism.py` | ATP maintenance 8.39, GAM 75.55, exchange bounds ±1000 | Orth 2010, BiGG convention |
| `evolution.py` | substitution 2.2e-10, indel 2.2e-11, ts/tv=3, Ne=1.3e8 | Lee 2012, Drake 1991, Hartl & Clark 2007 |
| `dna_codec.py` | PCR sub 1.5e-4 / indel 4.5e-6 / 30 cycles; synthesis rates | Saiki 1988, Potapov 2017, Filges 2021 |
| `reaction_diffusion.py` | F=0.035/k=0.065/Du=0.16/Dv=0.08; 5 Pearson presets | Pearson 1993 |
| `crispr.py` | PAM definitions, Doench 2016 + Hsu 2013 coefficient tables | Jinek 2012, Ran 2015, Zetsche 2015, Doench 2016, Hsu 2013 |
| `bio_data.py` | E. coli/yeast/human codon usage (CUTG/Kazusa); tRNA abundances | CUTG, Kazusa, Dong 1996, Chan & Lowe 2009, Dittmar 2006 |
| `codon_table.py` | STANDARD / MITO_VERTEBRATE / CILIATE genetic-code tables | NCBI (real biology; the codon→opcode mapping is a language-design decision, not a biological claim) |

### 2.2 Simplified algorithm variants (UPGRADE)

| Module | Location | Current state | Production replacement |
|---|---|---|---|
| `bio_data.py` | L272 | **CAI computed as a proportion** (fraction of "preferred" codons), not the Sharp–Li geometric mean | True Sharp & Li 1987 CAI: `w_i = f_i / max(f_j)` per synonymous family, CAI = geometric mean over sequence |
| `bio_data.py` | L712, L794 | tRNA copy numbers "approximate" | Keep (sourced); optionally re-verify against GtRNAdb / updated Dong 1996 counts |
| `dna_codec.py` | L10, L129 | Goldman-style encoding **simplified to 6 trits/byte** (true code = 5.05 trits/byte base-3 Huffman) | Implement the published base-3 Huffman code table (US Patent 10387301; `View_huff3.cd.new.correct` on EBI goldman-srv) with 8-byte → 13-trit framing and 4-trit checksum |
| `dna_codec.py` | L430–431 | Erlich screening GC dev 0.05 / homopolymer 3 — already matches paper | Align remaining params to publication: `redundancy` 1.1 → `alpha=0.07` (5–10% surplus oligos), RS over GF(256), robust soliton `c=0.025, δ=0.001` (Z=1.033) |
| `protein_structure.py` | whole file | Chou-Fasman ~60%, GOR IV ~65–68%, disorder ~60–70%, TM ~75% vs 80–85% professional tools | See §4.4: real Chou-Fasman parameter tables, real GOR IV singlet/pair parameters (DSSP-trained), IUPred-style energy-matrix disorder, TMHMM-grade TM prediction |
| `protein_structure.py` | — | GOR IV parameters **derived from CHOU_FASMAN_TABLE** instead of real PDB/DSSP-trained values | Real GOR IV 20×20 singlet + 400-entry pair tables (Garnier, Gibrat & Robson 1996) |
| `protein_structure.py` | — | Disorder = simplified Dunker 2001 charge-hydropathy with 2001-era parameters | IUPred method (Dosztányi 2005): 20×20 energy predictor matrix, 2–100 residue window, 21-residue smoothing, score>0.5 = disordered |
| `protein_structure.py` | — | TMHMM replaced by Kyte-Doolittle hydropathy window + two-threshold heuristic | Two-threshold Kyte-Doolittle is a legitimate first-pass; add real TMHMM-style HMM or use the validated Krogh 2001 thresholds; document Q3/TM-score caveats |
| `central_dogma.py` | L468–470 | hardcoded rate floor `max(0.5, 20.0 * relative_rate)` — **made-up 0.5 aa/s clamp** | Codon-specific decoding rates (PLOS ONE 2015 tables; eLife 2015 ribosome residence times; mean 62.5 ms/codon) |
| `central_dogma.py` | L56 | single `MRNA_HALF_LIFE_MEDIAN_MIN = 5.0` applied to every transcript | Per-gene / per-sequence half-life model (2–20 min range; RNase-E dependence, Bernstein 2002) |
| `central_dogma.py` | L144, L259–261 | promoter strength as dimensionless linear multiplier; initiation = `strength * 10/min` | Optionally: Michaelis–Menten / thermodynamic initiation model with sigma-factor binding (future tier) |
| `crispr.py` | L393–486 | **home-grown "simplified" on-target scoring** — GC buckets 0.7/0.3, linear position ramp `1+0.5·i/(n-1)`, base scores A/C/G=1.0/T=0.8, `tt_penalty×0.5`, blend `gc·0.3+pos·0.5+tt·0.2` | Full Doench 2016 Rule Set 2 (30-mer context, position-specific nucleotide + dinucleotide weights, GC penalty curve, polyT penalty; the coefficient tables already in `_DOENCH_*` are the published ones — route the "simplified" path through them) |
| `crispr.py` | L892–896 | Paixão 2022 asymmetric NHEJ repair approximated | Implement the full asymmetric repair-length distribution (Paixão et al. 2022 Nat Commun) |
| `crispr.py` | L966 | editing = literal sequence swap at cut site, no NHEJ/HDR repair simulation | Optionally add repair-pathway simulation (HDR donor resolution; Heyer 2010 efficiency 1–10%) |
| `evolution.py` | L34, L623 | dN/dS = "Nei-Gojobori simplified" | Keep Nei–Gojobori as default (cheap, correct); optionally add codon-substitution (M-series) models as a `method=` option |
| `grn.py` | L37 | `DECAY = 0.7` — single universal, uncited decay constant | Per-gene decay from real protein half-lives; optional Hill-equation kinetics with real Kd |
| `type_system.py` | L10 | "A simplified Hindley-Milner type inference" | Algorithmic simplification, not data — document as intended scope; no action |

### 2.3 Made-up engine / gameplay constants (UPGRADE or DOCUMENT)

| Module | Location | Current state | Action |
|---|---|---|---|
| `cell.py` | L7, L17, L19 | 4-neighborhood; `energy: int = 100`; 256 protein slots; move cost 1; division halves energy — all arbitrary units | Tier 3: explicit "gameplay units" disclaimer + constant registry with citations where available |
| `population.py` | L35–41, L50 | `division_threshold=200`, `death_threshold=0`, `signal_diffusion=0.1`, `signal_threshold=5`, `metabolic_cost=1`, `energy_intake=5`, initial energy 100 | Tier 3: same treatment; keep the real Xavier 2003 10 µM AI-2 citation but mark the dimensionless energy axis |
| `epigenetics.py` | L376, L428, L432 | methylation/histone modifiers `acc -= 0.4·prob`, `+= 0.7·prob`, `+= 0.2·prob` — uncited linear heuristics | Tier 2: cite DNMT/TET/histone-reader kinetics or mark coefficients as heuristic with source; keep Gardiner-Garden CpG thresholds (real) |
| `central_dogma.py` | L192 | `stop_efficiency = 1.0` hardcoded | Tier 2: use literature release-factor efficiency (codon-dependent, <1) |
| `metabolism.py` | L3–4 (docstring) | ✅ **resolved** — docstring now says "37-reaction core" (matches the JSON model); dFBA layer (`DynamicFluxBalance`) added on top | — |

### 2.4 Honest docstrings (no change needed, keep as documentation)

`reaction_diffusion.py:6`, `grn.py:1`, `codon_table.py:4,6`, `dna_codec.py:10,129`, `crispr.py:393,444,892,966`, `evolution.py:34,623,936`, `bio_data.py:272,712`, `metabolism.py:3–4,19–20,207–208`, `central_dogma.py:5–13,468`. The codebase is self-aware; the ledger above is the actionable diff.

### 2.5 Notes

- `transcription.py` **does not exist** — transcription lives in `central_dogma.py` (`transcribe()`, ~L216). Any plan referencing a standalone transcription module is out of date.
- The Doench/Hsu **published** coefficient tables in `crispr.py` (`_DOENCH_*`, `_HSU_*`) are real; only the "simplified" scoring path is fabricated and should be rerouted.

---

## 3. Primary Literature for the Upgrade

New references researched for this plan (all verified accessible):

| Ref | Work | DOI / Access | Use |
|---|---|---|---|
| R1 | Doench et al. 2016, *Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9*, Nat Biotechnol 34:184–191 | 10.1038/nbt.3437 | Rule Set 2 on-target scoring — replace `crispr.py` simplified path |
| R2 | Hsu et al. 2013, *DNA targeting specificity of RNA-guided Cas9 nucleases*, Nat Biotechnol 31:827–832 | 10.1038/nbt.2647 | Off-target scoring (already present) |
| R3 | Goldman et al. 2013, *Towards practical, high-capacity, low-maintenance information storage in synthesized DNA*, Nature 494:77–80 | 10.1038/nature11875 | Base-3 Huffman code (5.05 trits/byte); full code table in US Patent 10387301 and EBI `goldman-srv/DNA-storage/View_huff3.cd.new.correct` |
| R4 | Erlich & Zielinski 2017, *DNA Fountain*, Science 355:950–954 | 10.1126/science.aaj2038 | Luby Transform + robust soliton (c=0.025, δ=0.001, Z=1.033), screening GC 45–55%/homopolymer≤3, RS over GF(256), α=0.07 surplus; reference code `github.com/TeamErlich/dna-fountain` (Python 3 port: `jdbrody/dna-fountain`) |
| R5 | Chou & Fasman 1978, *Empirical predictions of protein conformation*, Ann Rev Biochem 47:251–276 | 10.1146/annurev.bi.47.070178.001343; parameter tables in Chou & Fasman 1974 Biochemistry 13:211–222, 10.1021/bi00699a002 | Real Pα/Pβ/P_turn tables (table verified via csbsju mirror: e.g. Val Pβ=1.70, Glu Pα=1.51, Pro Pβ=0.55) |
| R6 | Garnier, Gibrat & Robson 1996, *GOR method for predicting protein secondary structure*, Methods Enzymol 266:540–553 | 10.1016/S0076-6879(96)66034-0 | GOR IV: 17-residue window, singlet + pair information parameters, accuracy 55→64.4% |
| R7 | Dosztányi et al. 2005, *The pairwise energy content ... discriminates between folded and intrinsically unstructured proteins*, J Mol Biol 347:827–839 | 10.1016/j.jmb.2005.01.071 | IUPred: 20×20 energy predictor matrix (Thomas–Dill statistical potential), 2–100 residue window, 21-residue smoothing, score>0.5 |
| R8 | Dosztányi et al. 2005, *IUPred: web server ...*, Bioinformatics 21:3433–3434 | 10.1093/bioinformatics/bti541 | IUPred server algorithm + long/short disorder parameter sets |
| R9 | Krogh, Larsson, von Heijne, Sonnhammer 2001, *Predicting transmembrane protein topology with a hidden Markov model*, J Mol Biol 305:567–580 | 10.1006/jmbi.2000.4315 | TMHMM thresholds / HMM rationale for TM prediction |
| R10 | Sharp & Li 1987, *The codon adaptation index*, Nucleic Acids Res 15:1281–1295 | PMC340524 | CAI: w_i = f_i/max(f_j) per synonymous family; geometric mean |
| R11 | Proshkin et al. 2010, *Cooperation between translating ribosomes and RNA polymerase in transcription elongation*, Science 328:504–508 | 10.1126/science.1184937 | transcription elongation rate (already cited in code) |
| R12 | Ingolia et al. 2009, *Genome-wide analysis in vivo of translation with nucleotide resolution*, Science 324:218–223 | 10.1126/science.1168978 | translation elongation 20 aa/s (already cited) |
| R13 | Riedel et al. 2015, *Measurement of average decoding rates of the 61 sense codons in vivo*, eLife 4:e03735 | 10.7554/eLife.03735 | Ribosome residence times per codon (fastest ACC 0.70×avg, slowest CTC 1.89×avg) |
| R14 | Rudorf & Lipowsky 2015, *Protein synthesis in E. coli: dependence of codon-specific elongation on tRNA concentration and codon usage*, PLoS ONE 10:e0134994 | 10.1371/journal.pone.0134994 | Codon-specific elongation rates ω_c,elo from tRNA concentrations (S2/S3 Tables) |
| R15 | Orth et al. 2011, *A comprehensive genome-scale reconstruction of E. coli metabolism — iJO1366*, Mol Syst Biol 7:535 | 10.1038/msb.2011.65 | Full metabolic model; download SBML/JSON from BiGG `bigg.ucsd.edu/models/iJO1366`; loadable via COBRApy `load_model("iJO1366")` |
| R16 | Dosztányi 2018, *Prediction of protein disorder based on IUPred*, Protein Sci 27:331–340 | 10.1002/pro.3334 | IUPred reference implementation details |
| R17 | Paixão et al. 2022, *Asymmetric CRISPR-Cas repair outcomes* (Nat Commun 13) | 10.1038/s41467-022-* | NHEJ indel spectrum / asymmetric repair (7 indel classes) |
| R18 | Heyer, Ehmsen, Liu 2010, *Regulation of homologous recombination in eukaryotes*, Annu Rev Genet 44:113–139 | 10.1146/annurev-genet-051710-150955 | HDR efficiency 1–10% |
| R19 | Bernstein et al. 2002, *Global analysis of mRNA decay and abundance in E. coli*, J Bacteriol 184:6477 | 10.1128/JB.184.23.6477-6488.2002 | mRNA half-life distribution (median ≈5 min, range 2–20+ min) |
| R20 | Thomas & Dill 1996, *Statistical potentials extracted from protein structures*, J Mol Biol 257:457–469 | 10.1006/jmbi.1996.0175 | Statistical pairwise potential underlying IUPred |

---

## 4. Tiered Implementation Plan

### Tier 1

Paper-implementable with existing data; highest authenticity gain per unit of work.

#### 4.1 `crispr.py` — Doench 2016 Rule Set 2 on-target scoring

- **Current**: `on_target_score()` defaults to the home-grown "simplified" scoring (L393–486) with fabricated blend weights; the real `_DOENCH_*` tables exist but are not the default path.
- **Change**:
  1. Make Rule Set 2 the default `on_target_score` path: score from the 30-mer context (spacer 20 nt + PAM-proximal +3 nt each side where available) using published position-specific nucleotide weights, dinucleotide weights, GC-content penalty, and polyT penalty per Doench 2016.
  2. Keep the existing `_DOENCH_*` tables as the source of truth; remove/relegate the home-grown branch behind a `method="legacy"` compatibility flag so old callers/tests still resolve.
  3. Update `design_guide()` to select the guide with max Rule Set 2 score over all PAM sites (already selects best; scoring function swap only).
- **API preservation**: `on_target_score(guide)` signature unchanged; add optional `method="doench_2016"` param. All existing tests must keep passing (legacy path retained).
- **Tests**: add unit tests for known Rule Set 2 example guides; property test `0 <= score <= 1`.

#### 4.2 `dna_codec.py` — true Goldman base-3 Huffman code + Erlich alignment

- **Current**: Goldman-style scheme simplified to 6 trits/byte (L10, L129); Erlich uses robust soliton but `redundancy=1.1`.
- **Change**:
  1. Add the **published base-3 Huffman code table** (US Patent 10387301 / EBI `View_huff3.cd.new.correct`) — 8 bytes → 13 trits, no DNA triplet is the complement of another (complement-read safety), and add a 4-trit checksum per oligo. Achieve the paper's 5.05 trits/byte.
  2. Keep the existing 6-trit scheme behind `scheme="goldman_legacy"`; new default `scheme="goldman"` = full code. **Note**: the rotation/addressing scheme (segment framing L13 + checksum L4) already exists — replace only the byte→trit mapping table.
  3. Erlich: accept `redundancy` (kept for back-compat) and add `alpha=0.07` as the recommended surplus; default the encoding path to α=0.07 (5–10% surplus per paper). Keep RS over GF(256) via `reedsolo` when available.
- **API preservation**: `helix_to_dna(..., scheme=...)`, `dna_to_helix(...)`, `encode`/`decode` signatures unchanged.
- **Tests**: round-trip all `examples/*.helix` under the new Goldman code; verify density ≈5.05 trits/byte on random bytes; verify no complement palindromes in emitted trits; keep 6-trit legacy round-trip test.

#### 4.3 `bio_data.py` — true Sharp–Li CAI

- **Current**: CAI is "the proportion of preferred codons" (L272) — a linear fraction, not the geometric mean.
- **Change**:
  1. Implement canonical CAI: for each amino acid's synonymous codon family, `w_c = f_c / max_{j in family} f_j`; `CAI = geometric_mean(w_c over the sequence)`.
  2. Expose `cai(sequence, codon_usage)`; keep any existing simplified function name with a `simplified=True` kwarg or a `_legacy` alias so no test/API breaks.
  3. Update `biocodec.py` CAI-filter (threshold `>= 0.3`, L28–29/616/662) to use the new metric (threshold semantics may need retuning — document the new distribution).
- **API preservation**: keep existing public names; add `cai()`.
- **Tests**: synthetic sequence with only the optimal codon → CAI ≈ 1.0; all-synonymous-family-worst → CAI = min w_c; verify geometric-mean property against Sharp & Li worked example.

#### 4.4 `protein_structure.py` — real GOR IV / Chou-Fasman / IUPred

- **Current**: GOR IV parameters derived from `CHOU_FASMAN_TABLE` (not DSSP); disorder = simplified Dunker charge-hydropathy; TM = Kyte-Doolittle two-threshold heuristic; Chou-Fasman uses a 20-aa table (values ~match literature).
- **Change**:
  1. **Chou-Fasman**: replace/normalize the 20-aa Pα/Pβ table to the published Chou & Fasman 1974/1978 values (verified examples: Val Pβ=1.70, Glu Pα=1.51, Pro Pβ=0.55, Ala Pα=1.42). Keep the existing algorithm (nucleation/extension/break rules) — it already matches the paper.
  2. **GOR IV**: add the real GOR IV singlet (20×20, 17-residue window) and pair information tables trained on a PDB/DSSP non-redundant set (Garnier 1996). Current `predict_gor4()` must switch from `CHOU_FASMAN_TABLE`-derived to the real tables. Accuracy target ≈64% (Q3) on a standard reference set.
  3. **Disorder**: add an **IUPred-style** predictor: 20×20 energy predictor matrix (Thomas–Dill statistical potential; the R7/R8/R16 papers give the construction), per-residue energy `E_p = Σ_j P_ij · n_jp` over the local composition window (2–100 residues), 21-residue smoothing, score∈[0,1], threshold 0.5. Keep the existing Dunker charge-hydropathy as `method="chou_dunker"` legacy.
  4. **TM**: keep Kyte-Doolittle two-threshold (Krogh 2001 rationale) but expose and document the window/threshold parameters; add an optional TMHMM-style scoring layer if a HMM parameter set can be embedded (license permitting); otherwise document the ~75%-vs-85% accuracy caveat in the report.
- **API preservation**: `predict_secondary`, `predict_structure`, `hydropathy_profile`, `gravy`, `predict_transmembrane`, `predict_disorder` all unchanged in signature; add `method=` kwarg and new function `predict_iupred(seq)`.
- **Tests**: cross-check disorder on a known IDP (e.g. a protein with an experimentally mapped IDR from the IUPred dataset); assert GOR IV does not use Chou-Fasman values (table provenance test); keep all existing accuracy-benchmark tests passing within tolerances.

#### 4.5 `central_dogma.py` — codon-specific elongation, per-gene half-lives

- **Current**: flat `20.0 aa/s` with hardcoded 0.5 aa/s floor (L468–470); single 5 min mRNA half-life (L56).
- **Change**:
  1. Add a codon-specific decoding-rate table for E. coli (R13 eLife RRT values, or R14 ω_c,elo S2/S3 tables, or the Ribo-seq-derived mean 62.5 ms/codon from PMC10542608). Replace the `max(0.5, 20*relative_rate)` clamp with the table-driven per-codon rate. Keep the overall 20 aa/s average for back-compat as the table's normalization target.
  2. Replace the single `MRNA_HALF_LIFE_MEDIAN_MIN = 5.0` with a per-sequence half-life estimate (RNase-E target bias; range 2–20 min per Bernstein 2002). Expose `half_life = f(sequence)` with default = median 5 min when no features present, so `calculate_mrna_level()` behavior stays continuous with today.
  3. Keep `stop_efficiency` as a parameter; Tier 2 will set a codon-dependent default.
- **API preservation**: `transcribe/translate/coupled_transcription_translation/calculate_mrna_level` unchanged. Add optional `tables="ecoli"` and `half_life_model=` kwargs.
- **Tests**: elongation time of a gene enriched in fast codons < same gene enriched in slow codons; total elongation rate on a 10 000-codon random CDS ≈ 20 aa/s; existing kinetic tests pass unchanged (defaults preserve current behavior).

### Tier 2

Replace uncited engine constants; improve data authenticity.

#### 4.6 `metabolism.py` — fix stale docs + optional full iJO1366

- **Current**: 37-reaction curated core model (docstring still says "24"); pure-Python simplex.
- **Change**:
  1. Fix the module docstring (L3–4): "37-reaction E. coli core model" (or "curated core model"; reference the JSON data file).
  2. Add an optional full-model loader: if `cobrapy` is installed, `load_model("iJO1366")` (R15) or from a local BiGG JSON/SBML file; fall back to the curated core model otherwise. Keep `FluxBalanceAnalysis` as the unified API.
  3. Keep the pure-Python simplex as the default solver; optionally add scipy `linprog` (HiGHS) when installed.
- **API preservation**: `ECOLI_CORE_MODEL`, `FluxBalanceAnalysis`, `set_uptake/solve/analyze` unchanged; new `load_model(path_or_identifier=None)`.
- **Tests**: existing FBA tests unchanged; if cobrapy available, `analyze()` on iJO1366 must produce feasible growth (not required in CI without the dep).

#### 4.7 `epigenetics.py` — cite or replace heuristic coefficients

- **Current**: `acc -= 0.4·meth_prob`, `+= 0.7·prob`, `+= 0.2·prob` (L376/428/432) uncited; CpG-island thresholds real (Gardiner-Garden).
- **Change**:
  1. Add module-level constants with citations (e.g. DNMT processivity, TET oxidation kinetics, H3K27me3/H3K4me3 occupancy-to-expression mapping from ChIP-seq literature) — or mark the coefficients explicitly as `# heuristic: not a measured kinetic constant` with a source suggestion.
  2. Keep Gardiner-Garden / Takai 2002 thresholds.
- **API preservation**: unchanged signatures; constants module-level for overridability.
- **Tests**: existing tests pass (coefficient values unchanged unless cited replacement lands).

#### 4.8 `grn.py` — real decay / optional Hill kinetics

- **Current**: `DECAY = 0.7` universal (L37).
- **Change**: per-gene decay from protein half-life data; optional Hill-equation binding with real Kd for the sigmoid threshold model. Keep discrete-tick semantics.
- **API preservation**: `GeneNode`, `GRN.add_gene/step` unchanged; new optional `decay=` and `hill_n=`, `kd=` per gene.
- **Tests**: existing GRN tests unchanged with defaults; new test that steady-state expression matches Hill prediction.

#### 4.9 `evolution.py` — optional codon-substitution models

- **Current**: Nei-Gojobori simplified dN/dS (L34, L623).
- **Change**: keep Nei–Gojobori default; add `method=` option for a codon-substitution model (M-series; e.g. M0/M1a/M2a log-likelihood) as optional. Low priority — Nei-Gojobori is already correct and cheap.
- **API preservation**: `dnds_ratio(ref, query)` unchanged; new `dnds_codeml()` optional.

### Tier 3

Documentation-only — no behavior change.

#### 4.10 `cell.py` / `population.py` — explicit unit disclaimer

- Add a module-level `UNITS` note: energy/signal thresholds are **gameplay units**, not physical units, except where cited (Xavier 2003 10 µM AI-2). Register all magic numbers as named constants so future calibration is a one-line edit.
- Preserve all defaults exactly; existing tests pass unchanged.

---

## 5. Compatibility and API Preservation

Hard rules for every batch:

1. **Public signatures never change** — every function/class in the audit's API lists (§2 + module docstrings) keeps its name, parameter order, and return shape.
2. **New behavior is opt-in** — new defaults that change numerical output must be introduced behind `method=`, `scheme=`, or `tables=` kwargs; the legacy behavior remains reachable so existing tests and documented examples resolve identically.
3. **No new hard dependencies** — optional deps (numpy/biopython/reedsolo/cobrapy) guarded by import try/except with a stdlib fallback, matching the existing pattern (see `dna_codec.py:81` placeholder classes).
4. **Wire format stability** — DNA codec output changes (Goldman) are a **feature**, not a break: the codec is a pure transform with no persisted state, so round-trip tests are updated to the new scheme while legacy scheme round-trips are retained.
5. **Docstrings stay honest** — each upgraded module updates its self-admitted simplification marker to describe the new production implementation.

---

## 6. Verification Strategy

Every batch, in the `helix` conda env (`/opt/anaconda3/envs/helix/bin/python`):

```bash
ruff check src tests                     # lint gate
mypy                                     # type gate
python -m pytest -q                      # full suite: 1382 passing baseline
python -m pytest -q --benchmark-only     # 4 benchmark tests actually run in this env
```

Per-batch additions:

- **Tier 1.1 (CRISPR)**: Rule Set 2 known-guide regression tests; legacy method equivalence tests.
- **Tier 1.2 (DNA codec)**: density ≈5.05 trits/byte assertion; complement-safety property test; legacy 6-trit round-trip retained.
- **Tier 1.3 (CAI)**: Sharp & Li worked-example check; geometric-mean properties.
- **Tier 1.4 (protein structure)**: table-provenance test (GOR IV must not read Chou-Fasman); IUPred known-IDP check; accuracy tolerances preserved.
- **Tier 1.5 (central dogma)**: fast-codon vs slow-codon elongation ordering; mean-rate normalization check.
- **Tiers 2–3**: existing suites must remain green; new tests for each added option.

Coverage gate: `pytest --cov=helixlang --cov-fail-under=80` must stay ≥80% (new tables add coverage, not reduce it).

---

## 7. Implementation Batches

Status ledger for the §4 tiers. Every batch is verified under `/opt/anaconda3/envs/helix/bin/python` with the gates of §6. Current totals after the batches below: **1467 tests passing** (baseline 1382; +85 new test methods across batches 1–11), **ruff clean**, **mypy clean** (36 source files), **coverage 90%+** (gate ≥80%), 4 benchmark tests pass.

### Batch 1 — §4.1 `crispr.py` (Doench 2016 Rule Set 2) — DONE

- **Code**: `on_target_score(guide, model="doench2016", method=None)`. The `_DOENCH_*` tables stay the source of truth and remain the default path (`model="doench2016"`). Added the doc-required `method=` kwarg: `"doench_2016"` (Rule Set 2, default) and `"legacy"` (alias of `model="simplified"`); when both are given `method` wins. Unknown model/method now raises `ValueError` instead of silently falling through to the full path. The `_DOENCH_POSITION_NT_WEIGHTS` / `_DOENCH_DINUC_WEIGHTS` / GC-quadratic / polyT tables are unchanged.
- **Honesty fix**: docstring now states the real Rule Set 2 is a gradient-boosted regression tree (30-mer context + dinucleotides + thermodynamic terms + intercept) and that this module implements the reduced linear version using the published direction/magnitude tables, per §4.1.
- **Verified constants**: `_DOENCH_INTERCEPT = 0.05`, GC optimal 0.50, penalty coef 3.0, polyT penalty 0.30; typical spacer (GC 50%, no polyT) lands in [0.4, 0.6].
- **Tests**: +6 (`method=`/`model=` equivalence, `method="legacy"` ≡ `model="simplified"`, unknown model raises, empty-spacer == 0.0 under every model, `mode="best"` score selection, unknown design mode raises). `tests/test_crispr.py` 58 → 64 passing.
- **§4.1 step 3 applied opt-in**: `design_guide(target_dna, cas_variant, position, mode="nearest"|"best")` gained a `mode` kwarg. `"best"` implements the spec verbatim — selects the PAM site with the maximum Rule Set 2 `on_target_score` over **all** PAM sites (ties resolve 5'-most), `position` ignored. The default `"nearest"` preserves the documented legacy contract (PAM closest to `position`), so `test_design_guide_finds_nearest_pam` and all other existing tests pass unchanged (§5 rule 2 opt-in). Verified: poly-A spacer scores 0.329 vs a GC-balanced spacer 0.546 on the same target; `mode="best"` returns the high-scorer.
- `doc/08-api-reference.md` signature updated to include `method`.

### Batch 2 — §4.2 `dna_codec.py` (Goldman base-3 Huffman) — PARTIAL

The Goldman half of §4.2 is implemented and verified; the Erlich/legacy half is still pending.

- **Code (done)**: `_GOLDMAN_HUFFMAN_CODE` — the published 256-codeword base-3 Huffman table taken verbatim from the corrected specification file (`View_huff3.cd.new.correct`, EBI `goldman-srv/DNA-storage/orig_files/`), 5–6 trits/byte, average 5.07 (paper quotes ~5.05). Encoding appends the published EXTRA symbol (codeword `222020`) and pads the trit stream to a multiple of 25 before the rotating-key DNA mapping (no homopolymers). `helix_to_dna(..., scheme="goldman")` is the default. `goldman_encode`/`goldman_decode` retain the 100 nt / 25 nt 4×-overlap segmentation and 17 nt index header with parity; adjacent segments alternate reverse complement. Density ~0.28 bit/nt (paper: 0.29).
- **Verified**: the Huffman table was validated by re-decoding the author-supplied `View_huff3.cd.new.dna` back to the spec file (byte-identical up to the two known corruption errors in the published data); round-trip and density tests pass.
- **Tests**: `tests/test_dna_codec.py` 65 passing (unchanged).
- **Remaining**: §4.2 step 2 (retain the old 6-trit scheme behind `scheme="goldman_legacy"`) and step 3 (`alpha=0.07` recommended surplus as the Erlich default; `erlich_encode` still defaults `redundancy=1.1`). These are a follow-up batch; see the pending list at the end of this section.

### Batch 3 — §4.3 `bio_data.py` (Sharp–Li CAI) — DONE

- **Code**: `cai(sequence, species="ecoli", simplified=False)` — canonical Sharp & Li 1987 geometric-mean CAI. Per family `w_c = f_c / max_{j in family} f_j` (family-max codons → w = 1.0); stops/unknowns skipped. `simplified=True` keeps the legacy arithmetic "proportion of preferred codons". `biocodec.codon_adaptation_index_full` delegates to `bio_data.cai`; `import math` removed from `biocodec.py`.
- **Verified**: `cai("CTGCTA") == sqrt(1 · 0.04/0.47)` against the E. coli Leu family (CTG 0.47 / CTA 0.04) from `TRNA_ABUNDANCE`.
- **Tests**: +7 `TestGeneLevelCAI` in `tests/test_species.py` (58 → 77; optimal family codon == 1.0, geometric vs arithmetic, species specificity, stops/unknowns skipped, empty/noncoding, range, unknown species raises).

### Batch 4 — §4.4 `protein_structure.py` (real GOR IV + IUPred) — DONE

- **GOR IV tables**: `_GOR_IV_DSSP_SINGLET` (1020 entries: 20 AAs × offsets −8..+8 × H/E/C) and `_GOR_IV_DSSP_PAIR` (4800 entries: 20×20 × offsets ±1/±2 × H/E/C) trained on real DSSP statistics (DECIPHER `HEC_MI1`/`HEC_MI2`, 15,201 ASTRAL non-redundant proteins; DSSP 8→3 states H={H,G,I}, E={E}, C={B,S,C,T}). Weights `GOR_IV_DSSP_SINGLET_WEIGHT = 15/17`, `GOR_IV_DSSP_PAIR_WEIGHT = 60/17` (mass-conserving scale-up of DECIPHER's 2/17 per-pair weight to the module's 4-offset pair geometry); backgrounds H=−0.12 / E=−0.25 / C=0.23; argmax over H/E/C (no coil threshold, as in the real method). `predict_secondary_gor(sequence, method="gor_iv_dssp" | "chou_fasman")` — the legacy Chou-Fasman-derived path stays reachable.
- **IUPred**: `iupred_scores(sequence, mode="long" | "short")` — verbatim IUPred2A long/short energy matrices (400 entries each) and histograms (long 1071 bins, min −1.145 / max 4.205; short 1070 bins, min −0.955 / max 4.39), local window uc=100/25, 21-residue smoothing (short adds the −1.26 terminus edge compensation), score ∈ [0,1] via the survival histogram with 2-bin clamping, threshold 0.5. `predict_disorder(sequence, method="iupred" | "chou_dunker")`; legacy Dunker path unchanged and still the default.
- **Verified**: GOR provenance — all 680 shared singlet and 3200 shared pair entries differ from the Chou-Fasman-derived legacy tables. IUPred port is **byte-identical** to the official `iupred2a_lib.iupred()` (max abs diff 0.0 on P53_HUMAN, both modes): long disorder fraction 0.509, short 0.438; hydrophobic control (I50+V50) 0% disordered. Target predictions: A24→all H, V24/I24→all E, MAEELKKLAA→all H, AE8→all H, AG8→alternating H/C (8H/8C), VI8→all E, A10+PP+A12→21H with PP break, ALEK6→all H, VIYF6→all E.
- **Tests**: +30 (`TestGorIvDsspTables` 5, `TestGorIvDsspPredictions` 13, `TestIupredScores` 8, `TestPredictDisorderIupred` 4) in `tests/test_protein_structure.py`; 74 → 104 passing. P53_HUMAN embedded as a module constant.

### Batch 5 — §4.5 `central_dogma.py` (codon-specific elongation, per-gene half-lives) — DONE

- **Code**: `CODON_ELONGATION_RATE_AA_PER_S` — per-codon rate = 20 × (codon tRNA abundance / 3500), floor `MIN_CODON_ELONGATION_RATE_AA_PER_S = 20/3500`, replacing the old `max(0.5, 20·relative_rate)` clamp. `translate(..., tables="ecoli")` validates the flavor (mypy-safe `rate_source: dict | None`). Per-sequence mRNA half-life `_estimate_mrna_half_life(cds_dna)` (5′ 60-nt AU/GC ratio, clamped [2, 20] min, Bernstein 2002) behind `transcribe(..., half_life_model="bernstein_2002" | "flat")`, with `MRNA_HALF_LIFE_MIN_MIN=2.0`, `MRNA_HALF_LIFE_MAX_MIN=20.0`, `MRNA_HALF_LIFE_FEATURE_WINDOW_NT=60`, `MRNA_HALF_LIFE_MIN_FEATURE_SEQUENCE_NT=30`, `MRNA_HALF_LIFE_FEATURE_BASE=0.5`.
- **Tests**: +8 in `tests/test_central_dogma.py` (50 → 58): codon-rate table, 300-codon mean-rate bound 10–20 aa/s (10 most-abundant codon battery + ATG prefix), custom tRNA floor, `tables=` validation, AU-rich < GC-rich half-life ordering, flat model, unknown-model ValueError.

### Batch 6 — §4.6 `metabolism.py` (37-reaction doc fix + optional full model) — DONE

- **Code**: module docstring corrected 24 → 37 reactions (documents `data/ecoli_core_model.json`). New `load_model(path_or_identifier=None)`: `None` → `ECOLI_CORE_MODEL`; `.json` → `load_model_from_json`; `.xml`/`.sbml` → `cobra.io.read_sbml_model`; otherwise a BiGG identifier via `cobra.io.load_model`; `BioError` when cobrapy is absent (optional dep, guarded). `_from_cobra_model` converter added with `from typing import Any`. `load_model` exported in `__all__`.
- **Tests**: +3 `TestLoadModel` in `tests/test_metabolism.py` (55 → 58).

### Batch 7 — §4.7 `epigenetics.py` (cite or label heuristic coefficients) — DONE

- **Code**: magic numbers promoted to named module-level constants: `METHYLATION_ACCESSIBILITY_WEIGHT=0.4`, `PROMOTER_METHYLATION_REPRESSION=0.7` (Bird 2002), `GENE_BODY_METHYLATION_REPRESSION=0.2`, `BASE_ACCESSIBILITY=0.5`. Registry comment names future calibration sources (Jeltsch 2001, Ito 2011, Young 2011, Watt 1988). `HISTONE_MARK_TYPES` documented as heuristic weights with literature-correct mark/effect assignments. Gardiner-Garden / Takai CpG-island thresholds untouched.
- **Tests**: `tests/test_epigenetics.py` passes unchanged (behavior preserved).

### Batch 8 — §4.8 `grn.py` (real decay, optional Hill kinetics) — DONE

- **Code**: `hill(x, n, kd)` (0 for ≤0 input; 0.5 at x=kd), `decay_from_half_life_ticks` (`0.5**(1/ticks)`; `ValueError` for ≤0; E. coli median ~110 min → decay ~0.994 vs the legacy universal 0.7), `GeneNode` optional `decay`/`hill_n`/`kd`, `add_gene(...)` extended kwargs, `step()` Hill-vs-sigmoid dispatch. `DECAY=0.7` relabeled as the legacy heuristic. Constitutive-node test pins `decay=1.0` for a flat steady state.
- **Tests**: +5 in `tests/test_grn.py` (7 → 12).

### Batch 9 — §4.9 `evolution.py` (codon-substitution models) — DONE

- **Code**: `dnds_ratio(..., method="nei_gojobori" | "codeml" | "m0")`; new `dnds_codeml(dna, ancestral, kappa=3.0)` — M0 one-ratio ML (Goldman & Yang 1994) with the first-hit Poisson approximation (no 61×61 matrix exp), `_TS_TRANSITIONS`, `_codon_mutation_table()`, module `_CODON_MUTATIONS`, multi-position diffs = one nonsynonymous transversion-equivalent step, bounded golden-section `_maximize` (tol 1e-7), nested fit (ω ∈ [0,5], t ∈ [0,10]) + one coordinate-ascent round, LRT vs ω=1 via `math.erfc`, no-substitution → ω=1.0/t=0.0. Returns a superset of the `dnds_ratio` keys.
- **Tests**: +6 `TestDndsCodeml` in `tests/test_evolution.py` (71 → 77).

### Batch 10 — §4.10 `cell.py` / `population.py` (unit disclaimer + named constants) — DONE

- **Code**: module-level `UNITS` disclaimer (energy/signal values are gameplay units, not physical units, except the cited Xavier 2003 10 µM AI-2). Registered magic numbers as named constants: `cell.py` — `INITIAL_CELL_ENERGY=100`, `CELL_PROTEIN_SLOT_COUNT=256`, `MOVE_ENERGY_COST=1`, `FEED_ENERGY_AMOUNT=10`, `MIN_DIVISION_ENERGY=2`, `DEFAULT_CELL_COLOR`, `UNITS`; `population.py` — `DEFAULT_MAX_POPULATION_SIZE=10000`, `DEFAULT_GRID_WIDTH/HEIGHT=100`, `DIVISION_ENERGY_THRESHOLD=200.0`, `DEATH_ENERGY_THRESHOLD=0.0`, `SIGNAL_DIFFUSION_COEFFICIENT=0.1`, `QUORUM_SIGNAL_THRESHOLD=5.0`, `METABOLIC_COST_PER_STEP=1.0`, `ENERGY_INTAKE_PER_STEP=5.0`, `POPULATION_CELL_INITIAL_ENERGY=100.0`. `PopulationConfig`/`PopulationCell` defaults preserved exactly.
- **Tests**: `tests/test_cell.py` (52) and `tests/test_population.py` (27) pass unchanged.

### Batch 11 — VM no-op opcodes wired in (`vm.py`, `cell.py`) — DONE

Eliminated the remaining documented no-op / unconsumed opcode behavior, replacing the prototype stubs with literature-grounded semantics.

- **`OP_REGULATE <mode>`** (was: no-op, "Regulation is driven by GRN data"): dynamic regulatory-edge (re)wiring — source = the currently-executing gene, target = gene selected by the operand's low nibble (`mode & 0x0F`, modulo node count, the same wobble-addressing convention as `OP_CALL_GENE`), sign = bit 7 (0 = activate `+REGULATE_EDGE_WEIGHT`, 1 = inhibit `-REGULATE_EDGE_WEIGHT`); adds the edge or updates its weight in place. Grounding: Jacob & Monod 1961 operon model; Ptashne 2004 "Genetic Regulatory Networks" (a cell rewires its regulatory graph at runtime, e.g. lacI derepression).
- **`OP_BIND <site>`** (was: no-op): protein-DNA binding — the current gene's transcription factor (or any available protein reservoir) is consumed (1 unit) and the target gene's expression level is boosted by `BIND_LEVEL_BOOST` (protein-limited: no TF → no binding). Grounding: Berg & von Hippel 1987 binding specificity; McClure 1985 activator binding raises the promoter's effective output.
- **`OP_SIGNAL <ch>`** (was: pushed an unconsumed `("signal", ch)` tuple): now releases a quorum-sensing autoinducer into the Gray-Scott field's V channel at the cell position — `min(1.0, SIGNAL_EMISSION_AMOUNT * (1 + ch))` — the pool read by `#quorum`, and every emission is counted (`_signal_emissions`). Grounding: Miller & Bassler 2001; Xavier & Bassler 2003 (AI-2, ~10 µM).
- **`OP_EMIT_MORPHOGEN <id>`** (was: fixed 1.0): the morphogen ID now scales the injected amount to `(id + 1) / EMIT_MORPHOGEN_SCALE` (id=0 keeps a non-zero legacy emission). Grounding: Turing 1952 reaction-diffusion morphogens; Pearson 1993 measured presets.
- **`OP_MODIFY_STATE <field>`** (was: fields 0/1 only): completed to a full 4-field map — 0 green color, 1 `age += 1`, 2 yellow color, 3 magenta color — so all four Pro codons have distinct effects.
- **Observability**: snapshot gained `signal_emissions`, `regulation_edges`, `binding_events`. New runtime event logs `_regulation_events` / `_binding_events`. `Cell.add_protein`/`consume_protein` widened to `int | str` (matches the `proteins: dict[int | str, float]` field already used by the central-dogma str-keyed path — a pre-existing mypy mismatch surfaced by the new binder code).
- **Tests**: `tests/test_vm.py` — signal tests (field release, channel scaling, no-field count), 8 `TestRegulateBind` tests (edge add / inhibit / in-place update / index wrap / empty-GRN-safe / protein-consumption boost / protein-limited no-op / empty-GRN-safe), `OP_MODIFY_STATE` fields 2/3, `OP_EMIT_MORPHOGEN` scaling; snapshot-structure test extended with the 3 new keys. `tests/test_cell.py` unchanged. Full suite **1467 passing** (baseline 1382, +85 cumulative), coverage **90.13%**, mypy + ruff clean.

---

### Pending follow-ups

- **§4.2 Erlich/legacy**: add `alpha=0.07` recommended surplus as the Erlich default (`erlich_encode` currently defaults `redundancy=1.1`) and retain the pre-Huffman 6-trit scheme behind `scheme="goldman_legacy"` for legacy round-trips (§5 rule 4).
- **§4.4 TM** (doc step 4): the Kyte-Doolittle two-threshold heuristic is kept and its window/threshold parameters are already exposed; an optional TMHMM-style scoring layer was deferred pending a license-clean HMM parameter set.
- **§4.5 stop_efficiency**: kept as a parameter; a codon-dependent default is Tier 2 work.
- **Remaining ignored operands** (deliberately left parameterized-but-functional): `OP_FEED <src>`, `OP_DIVIDE <mode>`, `OP_DIE <mode>`, `OP_DIFFUSE <dir>`, `OP_REACT <type>` — the core action runs; only the operand is unused. Wiring them (e.g. `OP_FEED` feeding `src` energy instead of the documented +10) would change documented semantics and break the wobble-derived operand convention, so they are left as-is.
