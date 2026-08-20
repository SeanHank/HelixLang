# 20 — Genome-Scale Metabolic Model Reconstruction Pipeline

> **Goal:** Given a real organism's genome and an environment, automatically
> reconstruct a functional genome-scale metabolic model (GEM), infer gene
> regulatory networks, and simulate life activities through the HelixLang
> simulation engine.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [State of the Art](#2-state-of-the-art)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 1 — Genome Import & Preprocessing](#4-phase-1--genome-import--preprocessing)
5. [Phase 2 — Functional Annotation](#5-phase-2--functional-annotation)
6. [Phase 3 — GEM Reconstruction](#6-phase-3--gem-reconstruction)
7. [Phase 4 — Gene Regulatory Network Inference](#7-phase-4--gene-regulatory-network-inference)
8. [Phase 5 — Kinetic Parameter Estimation](#8-phase-5--kinetic-parameter-estimation)
9. [Phase 6 — HelixLang Integration (`#gem` + `backend=gem`)](#9-phase-6--helixlang-simulation-integration)
10. [Data Model & API Design](#10-data-model--api-design)
11. [Validation Strategy](#11-validation-strategy)
12. [Gap Analysis & Current Status](#12-gap-analysis--current-status)
13. [Phase 7 — Universal Gene Annotation](#13-phase-7--universal-gene-annotation)
14. [Phase 8 — GRN-Metabolism Coupling](#14-phase-8--grn-metabolism-coupling)
15. [Phase 9 — Dynamic Simulation (dFBA Integration)](#15-phase-9--dynamic-simulation-dfba-integration)
16. [Phase 10 — Environment Dynamics](#16-phase-10--environment-dynamics)
17. [Updated Data Model](#17-updated-data-model)
18. [Implementation Roadmap](#18-implementation-roadmap)
19. [References](#19-references)

---

## 1. Motivation

HelixLang can simulate E. coli life activities end-to-end, but only because
all gene-reaction mappings, kinetic constants, and regulatory edges are
hardcoded for one organism (E. coli MG1655). For any other organism, the
critical middle steps are missing:

```
Genome (FASTA + GFF3)
  → Functional Annotation (what does each gene do?)
  → Metabolic Network Reconstruction (what reactions exist?)
  → Gene-Reaction Mapping (which gene encodes which enzyme → which reaction?)
  → Metabolic Model (stoichiometric + constraint-based)
  → Gene Regulatory Network (who regulates whom?)
  → Kinetic Parameters (how fast? kcat, Km)
  → Simulation (growth, gene expression, metabolic flux)
```

This document designs a **six-phase pipeline** that bridges this gap, drawing
on the methods and lessons from the published GEM reconstruction literature.

---

## 2. State of the Art

### 2.1 Automated GEM Reconstruction Tools

Seven major tools have been benchmarked for GEM reconstruction
(Mendoza et al. 2025, bioRxiv):

| Tool | Approach | Strategy | Key Database | Speed |
|------|----------|----------|-------------|-------|
| **ModelSEED v2** (Feist et al. 2023) | Bottom-up | RAST annotation → reaction mapping → gap-filling | ModelSEED biochemistry (KEGG + MetaCyc + BiGG) | Hours |
| **CarveMe** (Pereira et al. 2016) | Top-down | Universal BiGG model → carve based on enzyme presence | BiGG | Minutes |
| **gapseq** (Zelezniak et al. 2018) | Bottom-up | Sequence similarity → gene-protein-reaction rules | MetaCyc + KEGG | Hours |
| **RAVEN** (Wang et al. 2018) | Template | Reference model → orthology-based transfer | KEGG | Minutes |
| **AuReMe** (Aite et al. 2018) | Community | Community annotation + ModelSEED | ModelSEED + BioCyc | Minutes–Hours |
| **Reconstructor** (DiMucci et al. 2023) | Parsimony | Minimal gap-filling from universal model | BiGG | Minutes |
| **pan-Draft** (Ghilardi et al. 2024) | Pan-reactome | Multi-MAG consensus → species-level GEM | MetaCyc + KEGG + ModelSEED | Minutes |

**Key finding:** No single tool universally outperforms others. Tool selection
is influenced by organism characteristics, data availability, and intended
application (Mendoza et al. 2025). This motivates a **consensus approach**.

### 2.2 Consensus Reconstruction

GEMsembler (Matveishina et al. 2025, mSystems) combines models from multiple
tools into a consensus model that outperforms individual tools in auxotrophy
and gene-essentiality predictions. The pipeline:
1. Reconstruct with 2–4 tools (CarveMe + gapseq + ModelSEED + MetaNetX)
2. Harmonize IDs to MetaNetX namespace
3. Merge into supermodel, track origin of each feature
4. Assess agreement → consensus model with confidence scores
5. Curate using GEMsembler's automated workflow

### 2.3 Enzyme-Constrained Models

GECKO 3.0 (Chen et al. 2024, Nature Protocols) extends stoichiometric GEMs
with enzyme kinetic constraints:
- Incorporates kcat (turnover number) per enzyme
- Enzyme capacity: `flux ≤ kcat × [enzyme]`
- Protein pool constraint: total protein ≤ measured proteome fraction
- Already implemented in HelixLang's `virtual_cell.py` for E. coli

### 2.4 Kinetic Parameter Prediction (ML)

Machine learning methods for kcat/Km prediction (Wu et al. 2025):

| Method | Input | Output | R² (kcat) |
|--------|-------|--------|-----------|
| DLKcat (Zhang et al. 2023) | AA sequence + SMILES | kcat | 0.44 |
| EITLEM-Kinetics | ESM-1v + SMILES | kcat | 0.72 |
| MPEK (ProtT5 + Mole-BERT) | AA + SMILES | kcat + Km | 0.64 / 0.60 |
| UniKP | UniRef50 + SMILES Transformer | kcat | 0.67 |

### 2.5 GRN Inference from Multi-omics

For prokaryotes (simpler than eukaryotes):
- **RegulonDB** (Santos-Zavaleta et al. 2019): curated TF-target interactions
- **STRING** (Szklarczyk et al. 2023): co-expression + co-occurrence + experiments
- **SCENIC+** (Aibar et al. 2017): single-cell multi-omics → enhancer GRNs
- **LINGER** (Wang et al. 2024, Nature Biotech): lifelong learning from bulk → single-cell
- **Augusta** (Muscavid et al. 2024): genome-wide GRN from RNA-seq + motif scanning

For non-model organisms without multi-omics data, the practical approach is:
1. Predict TFs from genome annotation (DNA-binding domain detection)
2. Predict binding sites using position weight matrices (PWMs)
3. Use phylogenetic transfer from RegulonDB / DBTBS / BaseDB
4. Integrate with co-expression data if transcriptome is available

### 2.6 Machine Learning for Gap-Filling

- **BoostGAPFILL**: matrix factorization → identifies minimal reactions to add
- **DNNGIOR**: CNN-based gap-filling, performance depends on phylogenetic distance
- **RetroPath RL**: reinforcement learning for pathway discovery

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT                                            │
│  Genome (FASTA + GFF3)  +  Environment (growth medium)             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 1: Genome Import & Preprocessing                             │
│  • FASTA parsing, GFF3 parsing                                      │
│  • CDS extraction, protein translation                              │
│  • operon / promoter / terminator detection                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 2: Functional Annotation                                     │
│  • EC number assignment (BLAST + HMMER + DIAMOND)                   │
│  • KEGG pathway mapping (KO terms)                                  │
│  • GO annotation                                                    │
│  • Subsystem classification (SEED-style)                            │
│  • Transporter classification                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 3: GEM Reconstruction                                        │
│  • Bottom-up: EC → reaction mapping (gapseq-style)                  │
│  • Top-down: universal model → carve (CarveMe-style)                │
│  • Consensus: merge both → confidence scoring                       │
│  • Biomass reaction construction                                    │
│  • Gap-filling (ML-assisted)                                        │
│  • FBA validation → growth prediction                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 4: Gene Regulatory Network Inference                         │
│  • TF identification (DNA-binding domain scan)                      │
│  • Promoter / binding site prediction                               │
│  • Regulatory edge inference (phylogenetic + co-expression)         │
│  • GRN → sigmoid/Hill kinetics wiring                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 5: Kinetic Parameter Estimation                              │
│  • kcat prediction (DLKcat / EITLEM-Kinetics style)                 │
│  • Km estimation from BRENDA / literature                           │
│  • Protein pool constraints (GECKO 3.0 style)                       │
│  • Enzyme-constrained FBA                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Phase 6: HelixLang Integration                                     │
│  • GEM → MetabolicModel (extend metabolism.py)                      │
│  • GRN → GRN object (extend grn.py)                                 │
│  • Enzyme capacity → VirtualCell wiring                             │
│  • #config backend=gem → sim_runtime dispatch                       │
│  • CLI + API + web visualization                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1 — Genome Import & Preprocessing

### 4.1 Input Format

```python
@dataclass
class GenomeInput:
    fasta: str                          # path to FASTA genome
    gff: Optional[str] = None           # path to GFF3 annotation
    organism_name: str = ""
    taxonomy_id: Optional[int] = None   # NCBI taxonomy ID
    growth_medium: Optional[dict] = None  # substrate → concentration
```

### 4.2 Preprocessing Steps

1. **FASTA parsing** — `SeqIO.parse()` via Biopython (optional) or pure-Python
   FASTA parser (HelixLang already has `seq_utils.py`)
2. **GFF3 parsing** — extend existing `parse_gff3()` in `whole_cell_scale.py`
   to also extract: gene type (protein-coding / tRNA / rRNA / pseudogene),
   regulatory features (terminator, attenuator), mobile elements
3. **CDS extraction** — for each protein-coding gene: extract nucleotide
   sequence by coordinate, handle strand, translate to protein (existing
   `_cds_to_protein()`)
4. **Operon detection** — if GFF3 provides operon annotations, group genes;
   if not, predict operons from intergenic distance + co-directionality
   (Price et al. 2006, BMC Bioinformatics)
5. **Output**: `PreprocessedGenome` dataclass containing gene table, protein
   sequences, and genomic coordinates

### 4.3 Existing Code Reuse

- `whole_cell_scale.py:load_chromosome()` — CDS extraction, protein
  translation, RBS encoding (already working)
- `whole_cell_scale.py:parse_gff3()` — GFF3 parsing with multi-segment
  Parent merge, minus-strand revcomp (already working)
- `seq_utils.py` — reverse complement, translation table

---

## 5. Phase 2 — Functional Annotation

### 5.1 Strategy: Local BLAST/DIAMOND + HMMER + Database Mapping

Rather than requiring external API calls (RAST, KEGG API), we implement a
**self-contained annotation pipeline** using local databases:

```
Protein sequences
  ├─→ DIAMOND blastp vs. UniRef90  →  EC numbers, KEGG KOs, GO terms
  ├─→ HMMER hmmsearch vs. TIGRFAM / Pfam  →  functional roles
  ├─→ TMHMM / SignalP  →  transmembrane / signal peptide (transporter detection)
  └─→ DNA-binding domain scan (Pfam: PF00072, PF00165, ...)  →  TF identification
```

### 5.2 Database Requirements

| Database | Purpose | Size | Source |
|----------|---------|------|--------|
| UniRef90 | Sequence homology | ~50 GB | NCBI/UniProt |
| TIGRFAM | Functional role assignment | ~100 MB | JCVI |
| Pfam | Domain detection | ~2.2 GB | EBI |
| BRENDA | kcat/Km values | ~100 MB | BRENDA (or lite subset) |
| MetaCyc / KEGG | Reaction mapping | ~500 MB | KEGG or MetaCyc flat files |
| CIS-BP / JASPAR | TF binding motifs | ~50 MB | Public |

**Alternative: lightweight mode.** For users without local databases, the
pipeline can delegate to web APIs:
- KEGG API (free, rate-limited)
- UniProt REST API
- NCBI BLAST+ web service

### 5.3 Annotation Output

```python
@dataclass
class GeneAnnotation:
    gene_id: str
    protein_id: str
    ec_numbers: list[str]          # EC 1.2.3.4
    kegg_ko: list[str]             # K00001
    go_terms: list[str]            # GO:0003677
    subsystem: str                 # "Glycolysis", "TCA cycle", etc.
    is_transporter: bool
    transport_substrate: Optional[str]
    is_transcription_factor: bool
    tf_family: Optional[str]       # "LacI", "AraC", "GntR", etc.
    confidence: float              # 0.0–1.0
```

### 5.4 Implementation Modules

| New Module | Responsibility |
|------------|---------------|
| `src/helixlang/annotation/__init__.py` | Package init |
| `src/helixlang/annotation/blast.py` | DIAMOND/BLAST+ local search |
| `src/helixlang/annotation/ec_mapping.py` | EC number → reaction mapping |
| `src/helixlang/annotation/kegg_mapping.py` | KO → pathway/reaction mapping |
| `src/helixlang/annotation/tf_detection.py` | TF detection: HMMER domain scan (preferred) + heuristic header fallback |
| `src/helixlang/annotation/transporter.py` | Transporter classification |

### 5.5 TF Detection: Two Modes

`tf_detection.py` supports two modes controlled by `pfam_database`:

| Mode | Trigger | Method | Sensitivity | Specificity | Dependencies |
|------|---------|--------|-------------|-------------|-------------|
| **HMMER model** | `pfam_database` non-empty | `hmmsearch --max` against Pfam-A.hmm (19 prokaryotic TF families, Pfam release 37 accessions) | Scans all sequences for DNA-binding domains | Medium — HTH structural similarity causes cross-family hits; bit-score > 0 + best-hit dedup filters false positives | hmmer3 |
| **Heuristic fallback** | `pfam_database` empty | String-match FASTA headers for TF family names | Only finds genes already annotated as TFs in headers | Low — header contains family name ≠ functional TF | None |

**Key implementation details:**
- `--max` flag disables all HMMER heuristic pre-filters for maximum sensitivity (HTH domains are ~30–60 residues and rejected by default filters)
- `e_value_threshold` defaults to 1.0 (not 1e-5) — short HTH domains score 2–6 bits, making strict E-value cutoffs counterproductive
- Per-gene best-hit deduplication: only highest-scoring hit kept, negative bit-scores dropped
- Confidence = `max(0, min(1, 0.5 + score/20.0))` — score ≥10 → ~1.0, score 0 → ~0.5

### 5.6 Accuracy Targets

Based on ModelSEED validation (Henry et al. 2010):
- EC number assignment: ≥70% accuracy at 60% identity threshold
- Gene essentiality prediction: ≥72% accuracy (ModelSEED achieved 87% after
  optimization)
- Biolog phenotype prediction: ≥83% accuracy

---

## 6. Phase 3 — GEM Reconstruction

### 6.1 Dual-Strategy Approach

We implement **both** bottom-up and top-down strategies, then merge:

#### Bottom-up (gapseq-style)
```
For each annotated gene with EC number:
  → look up EC → reaction(s) in ModelSEED biochemistry database
  → add reaction to draft model with GPR rule: gene → protein → reaction
For each spontaneous reaction (no enzyme needed):
  → add to draft model
For transporters:
  → add transport reactions for predicted substrates
```

#### Top-down (CarveMe-style)
```
Start with universal prokaryotic model (~3000 reactions from BiGG)
For each reaction:
  → check if any gene in genome can catalyze it (GPR evidence)
  → remove reaction if no supporting evidence
  → keep reaction if evidence exists
```

#### Consensus (GEMsembler-style)
```
Merge bottom-up and top-down models:
  → harmonize reaction IDs to MetaNetX namespace
  → keep reaction if EITHER strategy supports it (union)
  → assign confidence: HIGH (both), MEDIUM (one), LOW (gap-filled)
```

### 6.2 Biomass Reaction Construction

The biomass reaction is organism-specific. Strategy:

1. **Template-based** (ModelSEED approach): use a generic prokaryotic biomass
   template, adjust lipid / cell-wall / cofactor composition based on
   taxonomy (Gram+ vs. Gram-)
2. **Literature-curated** (for well-studied organisms): use published biomass
   composition data
3. **Auto-estimation** (for novel organisms): use taxonomic neighbors as
   templates

### 6.3 Gap-Filling

Even with dual strategy, models often cannot produce biomass. Gap-filling
adds minimal reactions to enable growth:

```python
def gapfill(draft_model: MetabolicModel, medium: dict, 
            max_reactions: int = 50) -> GapFillResult:
    """Find minimal set of reactions to enable biomass production.
    
    Uses LP-based gap-filling (ModelSEED approach):
    min Σ z_i  subject to: S·v = 0, v_min ≤ v ≤ v_max, 
    v_biomass > 0, z_i ∈ {0,1}
    """
```

ML-assisted gap-filling (BoostGAPFILL style) can prioritize reactions from
the database based on phylogenetic similarity.

### 6.4 Output

```python
@dataclass
class GemResult:
    model: MetabolicModel          # stoichiometric matrix + bounds
    gpr_rules: dict[str, GPR]      # reaction → gene-protein-reaction rule
    biomass_reaction: str          # ID of biomass reaction
    annotation: dict[str, GeneAnnotation]
    confidence_scores: dict[str, float]  # reaction → confidence
    gap_filled_reactions: list[str]
    validation: GemValidation
```

---

## 7. Phase 4 — Gene Regulatory Network Inference

### 7.1 GRN Components

A GRN consists of:
- **Nodes**: genes (target genes + transcription factors)
- **Edges**: regulatory interactions (activation / repression)
- **Parameters**: Hill coefficient (n), dissociation constant (Kd)

### 7.2 Three-Tier Inference Strategy

#### Tier 1: Database-derived (highest confidence)

For organisms with curated databases:
- **RegulonDB** (E. coli): 5,397 TF-target interactions
- **DBTBS** (B. subtilis): σ-factor regulons
- **PlantTFDB** (plants): TF-target predictions

```python
def load_curated_grn(organism: str) -> list[RegulatoryEdge]:
    """Load from curated database if available."""
```

#### Tier 2: Phylogenetic transfer

For organisms without curated databases:
1. Identify TFs in target genome (from Phase 2 TF detection)
2. Find orthologs in RegulonDB (or other curated DB) via BLAST
3. Transfer regulatory edges: if TF_A regulates gene_B in E. coli,
   and both TF_A and gene_B orthologs exist in target, predict edge
4. Weight by ortholog identity score

```python
def transfer_regulon(
    source_grn: list[RegulatoryEdge],
    orthology_map: dict[str, str]  # source_gene → target_gene
) -> list[RegulatoryEdge]:
```

#### Tier 3: Sequence-based prediction (supplementary)

1. **Promoter detection**: scan upstream regions for σ-factor binding
   motifs (−10 / −35 boxes) using PWMs from Lister et al. 2015
2. **TF binding site prediction**: scan for TF-specific motifs using
   FIMO (MEME Suite) or MOODs
3. **Co-expression inference** (if RNA-seq data available):
   ARACNe (Margolin et al. 2006) or GENIE3 (Huynh-Thu et al. 2010)

### 7.3 GRN → HelixLang Mapping

Each regulatory edge becomes a `#regulate` annotation:

```python
def grn_to_helixlang(edges: list[RegulatoryEdge]) -> str:
    """Convert inferred GRN to HelixLang annotations."""
    lines = []
    for e in edges:
        sign = "+" if e.effect == "activation" else "-"
        lines.append(f"#regulate {e.source} -> {e.target} "
                     f"strength={sign}{e.strength}")
    return "\n".join(lines)
```

The GRN is then wired into the `GRN` object with sigmoid/Hill kinetics
(existing `grn.py`), where:
- Hill coefficient (n) defaults to 2.0 for prokaryotes
- Kd defaults to half-maximal expression level
- Strength from database confidence or phylogenetic transfer score

---

## 8. Phase 5 — Kinetic Parameter Estimation

### 8.1 kcat Prediction

For enzymes with no experimentally measured kcat, we predict using ML:

**Recommended approach** (EITLEM-Kinetics style, R²=0.72):
1. Protein language model (ESM-1v or ESM-2) encodes amino acid sequence
   → fixed-length embedding
2. Reaction SMILES encoded via Molecule Transformer
3. Concatenated features → regression head → kcat prediction

**Fallback** (simpler, no GPU needed):
1. Lookup in BRENDA database by EC number
2. Use median kcat for the EC class if specific value unavailable
3. Phylogenetic scaling: kcat ∝ organism growth rate (no data available)

### 8.2 Km Estimation

- BRENDA lookup by EC number + substrate
- ML prediction (GraphKM style, R²=0.62) for novel substrates
- Default: use median Km from BRENDA for each substrate class

### 8.3 Enzyme-Constrained FBA (GECKO 3.0)

Extend HelixLang's existing `EnzymeCapacity` with predicted parameters:

```python
@dataclass
class EnzymeConstraint:
    reaction_id: str
    gene_id: str
    kcat: float              # from prediction or BRENDA
    Km: float                # from prediction or BRENDA
    protein_fraction: float  # from proteomics or prediction
    
    def max_flux(self, enzyme_concentration: float) -> float:
        return self.kcat * enzyme_concentration
```

Wire into `VirtualCell.enzyme_capacity_enabled = True` with the new
per-organism kcat/km tables (replacing hardcoded `ECOLI_CORE_KCAT`).

---

## 9. Phase 6 — HelixLang Simulation Integration

### 9.1 Helix Language: `#gem` Directive

The `#gem` directive declares GEM reconstruction parameters. All fields are
namespaced into `Program.sim_extensions` under a `gem_` prefix.

**Syntax — File-based genome:**

```helix
#gem organism=<id> genome=<path> [use_database=true] [include_spontaneous=true]
    [gapfill=true] [target_organism="<name>"] [medium=glucose_minimal]
    [dynamic=false] [duration=24.0] [dt=0.1] [expression=false]
```

**Syntax — Inline DNA (doc/20 §12):**

The genome can be specified as inline DNA sequences (ATCG) instead of a
FASTA file path. This enables writing gene sequences directly in Helix
code, making the language self-contained for small genomes or gene subsets:

```helix
#gem organism=e_coli_k12
ATGAAACGCATTAGCACCACCATTACCACCACCATCAC...
#end

#config backend=gem ticks=1
```

The inline DNA block is concatenated and stored as `gem_inline_genome`.
At runtime, it is written to a temporary FASTA file before being passed
to the GEM pipeline. Inline DNA and `genome=` are mutually exclusive.

**Multi-gene DNA with `#gene_id` markers (doc/20 §12):**

For multiple genes, use `#gene_id` markers within the DNA block. Each
marker starts a new FASTA entry in the generated file:

```helix
#gem organism=e_coli_k12
#gltA
ATGTCTCAGCAAATTCGTGTGGCGCTGAATGTAGAGCTTGGTAGTCGCCAGACATTGCAG...
#sucA
ATGACCGAGCAGATCCTGCGTGACTTGGCGCACAAAATTCCGCTGGTGCATCTGAAAGA...
#end
```

The lexer emits `GENE_ID` tokens for `#identifier` patterns that are not
annotation keywords. The parser collects them as `(gene_id, sequence)`
pairs stored in `gem_inline_genes`, and `_run_gem()` writes a multi-entry
FASTA file with each gene as a separate record.

See `examples/48_ecoli_inline_dna.helix` for a complete example with 46
real E. coli MG1655 gene sequences (7,296 bp total) written as inline DNA.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `organism` | string | *required* | Organism identifier (e.g. `e_coli_k12`) |
| `genome` | string | *optional* | Path to genome FASTA file (nucleotide or protein) |
| `use_database` | bool | `true` | Use known regulatory interactions for GRN |
| `include_spontaneous` | bool | `true` | Include spontaneous (no-enzyme) reactions |
| `gapfill` | bool | `true` | Run gap-filling for biomass connectivity |
| `target_organism` | string | `"Escherichia coli"` | Target organism for BRENDA kcat lookup |
| `medium` | string | `glucose_minimal` | Growth medium preset |
| `dynamic` | bool | `false` | Enable dynamic FBA (batch culture time-course) |
| `duration` | float | `24.0` | Simulation duration in hours (when dynamic=true) |
| `dt` | float | `0.1` | Time step in hours (when dynamic=true) |
| `expression` | bool | `false` | Enable expression inference from GRN |

**Parser implementation:** `parser.py:429` `_parse_gem()` — validates
`organism=` is present, handles inline DNA blocks (CODON tokens after
fields), stores all fields as `gem_<key>` in `Program.sim_extensions`.

**Example (`examples/46_gem_reconstruction.helix`):**

```helix
#gem organism=e_coli_k12 genome=genome.fasta use_database=true include_spontaneous=true gapfill=true

#config backend=gem ticks=1

#sim output=stage,status,genes_annotated,reactions_total,grn_edges,kcat_predictions,km_estimates
```

**Example (`examples/47_ecoli_gem_simulation.helix`) — Real E. coli:**

```helix
#gem organism=e_coli_k12 genome=data/ecoli_core_genome.fasta medium=glucose_minimal
    use_database=true include_spontaneous=true gapfill=true
    dynamic=true duration=24.0 dt=0.1 expression=true
    target_organism=Escherichia coli

#config backend=gem ticks=1

#sim output=stage,status,genes_annotated,reactions_total,grn_edges,kcat_predictions,km_estimates
```

### 9.2 Backend Dispatch: `#config backend=gem`

Setting `backend=gem` in `#config` triggers the GEM pipeline automatically.
The backend can be specified in three ways:

1. **In `.helix` source:** `#config backend=gem`
2. **CLI flag:** `helixlang source.helix --gem`
3. **CLI backend flag:** `helixlang source.helix --backend gem`

The `gem` backend is included in the `BACKENDS` frozenset
(`sim_runtime.py:141`) and is a valid `--backend` choice.

**Dispatch flow:**

```
CLI --gem flag  ─┐
#config backend=gem ─┤
--backend gem ────┘──→ sim_runtime.run(program, backend="gem")
                          │
                          └──→ _run_gem(program)
                                  │
                                  ├─ Reads #gem fields from sim_extensions
                                  ├─ Falls back to #species fields if no #gem
                                  └──→ apps.gem_pipeline.run_gem_pipeline()
```

**Field resolution priority:** `#gem` fields take precedence over `#species`
fields. If `#gem genome=...` is set, it is used directly. Otherwise, the
runtime scans `#species` entries for a `genome` field.

### 9.3 Pipeline Orchestrator: `apps/gem_pipeline.py`

```python
def run_gem_pipeline(
    genome_fasta: str,
    organism: str = "e_coli_k12",
    use_database_interactions: bool = True,
    include_spontaneous: bool = True,
    run_gapfill: bool = True,
    target_organism: str = "Escherichia coli",
) -> GemPipelineResult:
```

**`GemPipelineResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `annotations` | `dict[str, GeneAnnotation]` | Gene annotations from Stage 2 |
| `annotated_genes` | `int` | Number of annotated genes |
| `bottom_up` | `BottomUpResult` | Bottom-up reconstruction result |
| `top_down` | `TopDownResult` | Top-down (carve) result |
| `consensus` | `ConsensusResult` | Merged consensus model |
| `gapfill` | `GapfillResult` | Gap-filling result |
| `grn` | `GRNInferenceResult` | Regulatory network |
| `kcat_predictions` | `list[KcatPrediction]` | kcat values per reaction |
| `km_estimates` | `dict[str, float]` | Km values per reaction |
| `biomass_reaction` | `BiomassReaction` | Template biomass reaction |
| `final_reaction_count` | `int` | Total reactions after merge + gapfill |
| `final_gene_count` | `int` | Genes with reaction associations |
| `stages_completed` | `int` | 0–6 (how many stages finished) |
| `errors` | `list[str]` | Fatal errors (pipeline aborts) |
| `warnings` | `list[str]` | Non-fatal warnings |

**`summary()` method** returns a human-readable string:

```
GEM Pipeline Summary
========================================
Genes annotated:    0
Reactions (bottom): 55
Reactions (top):    30
Reactions (final):  85
Genes in model:     0
GRN edges:          47
k_cat predictions:  85
Km estimates:       85
Stages completed:   6/6
```

### 9.4 `_run_gem()` — sim_runtime.py

The `_run_gem(program)` function is registered in `BACKENDS` and called by
`run(program, backend="gem")`. It:

1. Reads `gem_*` fields from `program.sim_extensions`
2. Handles inline DNA: if `gem_inline_genome` is set (from `#gem` DNA block),
   writes it to a temporary FASTA file
3. Falls back to `species.*.genome` if no `gem_genome` is set
4. Reads new parameters: `dynamic`, `duration`, `dt`, `expression`
5. Calls `run_gem_pipeline(...)` with resolved parameters
6. Returns a `SimResult` with per-stage rows:

| stage | status | genes_annotated | reactions_total | grn_edges | kcat_predictions | km_estimates |
|-------|--------|-----------------|-----------------|-----------|------------------|--------------|
| annotation | ok/failed | N | 0 | 0 | 0 | 0 |
| reconstruction | ok/failed | N | M | 0 | 0 | 0 |
| grn | ok/skipped | N | M | E | 0 | 0 |
| kinetics | ok | N | M | E | K | K |

The `meta` dict carries `organism`, `stages_completed`, `warnings`,
`errors`, and `summary`.

### 9.5 CLI Integration

```
helixlang <source.helix> --gem              # force GEM backend
helixlang <source.helix> --backend gem      # same effect
helixlang <source.helix> --gem --json       # GEM pipeline → JSON output
```

The `--gem` flag overrides any `#config backend` setting and forces the
GEM pipeline. When combined with `--json`, the full `SimResult` is emitted
as JSON.

### 9.6 Integration with Existing Engine

| Component | HelixLang Module | Integration Point |
|-----------|------------------|-------------------|
| Metabolic model | `gem/consensus.py` | `ConsensusResult` → reactions list |
| Gene-reaction mapping | `gem/bottom_up.py` | `GPRRule` → gene-to-reaction associations |
| Enzyme kinetics | `kinetics/kcat_predictor.py` | `KcatPrediction` → per-reaction kcat |
| Km estimation | `kinetics/km_estimator.py` | `estimate_km()` → per-substrate Km |
| GRN | `gem/grn_inference.py` | `RegulatoryEdge` → TF→target interactions |
| Biomass | `gem/biomass.py` | `BiomassReaction` → template biomass equation |
| Simulation loop | `sim_runtime.py` | `#config backend=gem` dispatch |
| CLI | `cli.py` | `--gem` flag |

### 9.7 Web API (planned)

```
POST /api/gem/reconstruct
  Body: { "fasta": "...", "organism": "e_coli_k12" }
  Returns: { "stages_completed": 6, "reactions": 85, "summary": "..." }

POST /api/gem/simulate
  Body: { "model": {...}, "ticks": 500 }
  Returns: { "trace": [...], "final_biomass": ... }
```

---

## 10. Data Model & API Design

### 10.1 New Dataclasses

```python
@dataclass
class GeneAnnotation:
    gene_id: str
    protein_seq: str
    ec_numbers: list[str]
    kegg_ko: list[str]
    go_terms: list[str]
    subsystem: str
    is_transporter: bool
    is_transcription_factor: bool
    tf_family: Optional[str]
    confidence: float

@dataclass  
class RegulatoryEdge:
    source: str          # TF gene ID
    target: str          # target gene ID
    effect: str          # "activation" or "repression"
    strength: float      # 0.0–1.0
    confidence: float    # 0.0–1.0
    evidence: str        # "curated" / "phylogenetic" / "predicted"

@dataclass
class KineticParams:
    reaction_id: str
    gene_id: str
    kcat: float          # s⁻¹
    Km: dict[str, float] # substrate → mM
    source: str          # "brenda" / "ml_predicted" / "median_ec"

@dataclass
class GemResult:
    model: MetabolicModel
    gpr_rules: dict[str, GPR]
    biomass_reaction: str
    annotations: dict[str, GeneAnnotation]
    grn: list[RegulatoryEdge]
    kinetic_params: list[KineticParams]
    confidence_scores: dict[str, float]
    gap_filled: list[str]
    validation: GemValidation

@dataclass
class GemValidation:
    growth_rate: float               # predicted by FBA
    biomass_precursors: dict[str, float]
    essential_genes: list[str]
    biolog_accuracy: Optional[float]  # if experimental data available
    gene_essentiality_accuracy: Optional[float]
```

### 10.2 Module Dependency Graph

```
annotation/
  ├── blast.py          → (diamond, hmmer optional)
  ├── ec_mapping.py     → (annotation/blast.py)
  ├── kegg_mapping.py   → (annotation/blast.py)  
  ├── tf_detection.py   → (annotation/blast.py)
  └── transporter.py    → (annotation/blast.py, annotation/hmmer.py)

gem/
  ├── bottom_up.py      → (annotation/*, metabolism.py)
  ├── top_down.py       → (annotation/*, metabolism.py)
  ├── consensus.py      → (gem/bottom_up.py, gem/top_down.py)
  ├── gapfill.py        → (gem/consensus.py, metabolism.py)
  ├── biomass.py        → (bio_data.py)
  └── grn_inference.py  → (annotation/tf_detection.py, grn.py)

kinetics/
  ├── kcat_predictor.py → (protein_fitness.py for embeddings)
  └── km_estimator.py   → (kinetics/kcat_predictor.py)

apps/gem_pipeline.py    → (gem/*, kinetics/*, annotation/*, 
                           virtual_cell.py, sim_runtime.py)
```

---

## 11. Validation Strategy

### 11.1 Benchmark Organisms

Test on organisms with published, manually curated GEMs:

| Organism | Published GEM | Reactions | Genes | Source |
|----------|---------------|-----------|-------|--------|
| E. coli MG1655 | iML1515 | 2,712 | 1,516 | Orth et al. 2010 |
| S. cerevisiae S288C | Yeast8 | 11,500 | 1,150 | Lu et al. 2023 |
| B. subtilis 168 | iYO844 | 1,020 | 844 | Oh et al. 2007 |
| M. tuberculosis H37Rv | iNJ661 | 939 | 661 | Jamshidi & Palsson 2007 |
| P. aeruginosa PAO1 | iMO1086 | 1,086 | 1,004 | Oberhardt et al. 2008 |

### 11.2 Validation Metrics

For each benchmark organism:
1. **Growth prediction**: compare FBA-predicted growth rate to experimental
   growth rate (target: R² > 0.8 across media conditions)
2. **Gene essentiality**: compare predicted essential genes to experimental
   knockout data (target: accuracy > 80%)
3. **Auxotrophy prediction**: predict amino acid auxotrophies
   (target: accuracy > 85%, per GEMsembler results)
4. **Network size**: reactions / metabolites / genes compared to curated model
5. **Gap-filling count**: fewer gap-filled reactions = better annotation

### 11.3 Integration Tests

```python
def test_e_coli_full_pipeline():
    """End-to-end: MG1655 FASTA+GFF3 → GEM → FBA → growth ≈ 0.87 h⁻¹"""
    
def test_b_subtilis_full_pipeline():
    """End-to-end: 168 FASTA+GFF3 → GEM → FBA → growth ≈ 0.5 h⁻¹"""
    
def test_grn_transfer():
    """GRN from RegulonDB → ortholog transfer → target organism"""
    
def test_kcat_prediction():
    """kcat predictions within 10-fold of BRENDA values"""
```

---

## 12. Gap Analysis & Current Status

### 12.1 What Works Today

The pipeline achieves a **universal prokaryotic** demonstration (any organism
with a genome FASTA):

| Stage | Status | Module |
|-------|--------|--------|
| Genome import (FASTA/protein) | ✅ Working | `gem_pipeline.py`, `annotation/sequences.py` |
| Universal gene annotation (DIAMOND + UniProt) | ✅ Working | `annotation/blast.py`, `annotation/ec_mapping.py` |
| TF detection (HMMER + heuristic) | ✅ Working | `annotation/tf_detection.py` |
| Transporter classification | ✅ Working | `annotation/transporter.py` |
| Bottom-up reconstruction | ✅ Working | `gem/bottom_up.py` |
| Top-down reconstruction | ✅ Working | `gem/top_down.py` |
| Consensus merge | ✅ Working | `gem/consensus.py` |
| Gap-filling (heuristic) | ✅ Working | `gem/gapfill.py` |
| Biomass reaction (E. coli template) | ✅ Working | `gem/biomass.py` |
| GRN inference (RegulonDB + real gene names) | ✅ Working | `gem/grn_inference.py` |
| kcat/Km prediction (EC-aware) | ✅ Working | `kinetics/kcat_predictor.py` |
| Expression inference from GRN | ✅ Working | `omics/expression_inference.py` |
| Static FBA with enzyme capacity | ✅ Working | `metabolism.py` (simplex solver) |
| dFBA with GEM model | ✅ Working | `sim_runtime.py` (`dynamic=true`) |
| Inline DNA with gene markers | ✅ Working | `lexer.py`, `parser.py` |
| Helix `#gem` + `backend=gem` | ✅ Working | `sim_runtime.py`, `cli.py` |
| FeedEvent for fed-batch | ✅ Working | `environment.py` |
| Integration tests (20 GEM + 32 metabolism + 23 omics) | ✅ All passing | `tests/` |

### Status

| ID | Description | Status |
|----|-------------|--------|
| G1 | E. coli growth rate ~0.87 h⁻¹ | ✅ RESOLVED (Sprint 3) |
| G2 | B. subtilis growth rate ~0.70 h⁻¹ | ✅ RESOLVED (Sprint 3) |
| G3 | Multi-substrate dFBA (glucose + O2) | ✅ RESOLVED (Sprint 4) |
| G4 | Fed-batch/chemostat simulation | ✅ RESOLVED (Sprint 4) |
| G5 | LP-based gap-filling | ✅ RESOLVED (Sprint 3) |
| G6 | Model validation pipeline | ✅ RESOLVED (Sprint 3) |
| G7 | SBML Level 3 export | ✅ RESOLVED (Sprint 3) |
| G8 | CLI --gem --dynamic flags | ✅ RESOLVED (Sprint 5) |
| G9 | Growth rate benchmarks | ✅ RESOLVED (Sprint 5) |

### Sprint Completion

| Sprint | Items | Status |
|--------|-------|--------|
| Sprint 1 | Universal Gene Annotation (DIAMOND blastp, UniProt, EC wiring) | ✅ Complete |
| Sprint 2 | Expression Inference (Hill function, GRN motifs, omics/ module) | ✅ Complete |
| Sprint 3 | LP gap-fill, validation, taxonomy biomass, SBML export | ✅ Complete |
| Sprint 4 | Multi-substrate dFBA, DynamicSimulationResult, fed-batch/chemostat | ✅ Complete |
| Sprint 5 | CLI integration, growth-rate benchmarks, environment dynamics | ✅ Complete |

### 12.3 Architecture Status Map

```
Implemented: Genome → Universal Annotation → Network → Expression Inference
                               ↓                          ↓
                          kcat/Km (real EC)         Enzyme Levels
                               ↓                          ↓
                          GRN (real genes) → Expression → FBA Bounds
                               ↓                          ↓
                          Enzyme Capacity ←──────────────┘
                               ↓
                          dFBA (dynamic) ←──── Environment + FeedEvent
                               ↓
                          Life Activity Simulation
                               ↓
               ┌───────────────┴───────────────┐
               ▼                               ▼
    Standalone FBA                  GEM↔Ecosystem Bridge (doc/21)
    (backend=gem)                   (gem_driven=true in #sim kind=ecosystem)
                                    Species.metabolic_model + gem_to_species()
                                    Patch._growth_rate_gem() FBA-backed growth
```

---

## 13. Phase 7 — Universal Gene Annotation

**Goal:** Replace E. coli-only annotation with universal organism-agnostic pipeline.

### 13.1 Problem

Current `_annotate_from_fasta()` (`gem_pipeline.py:72-102`) has a hardcoded dict of ~35 E. coli genes. For any other organism, `annotated_genes = 0` and the pipeline produces an empty model.

### 13.2 Solution: Three-Tier Annotation

```
Protein sequences from genome
  │
  ├─ Tier 1: DIAMOND blastp vs. UniRef90/SwissProt
  │    → EC numbers, KEGG KO terms, GO terms
  │    → Sensitivity: HIGH (sequence homology)
  │    → Dependency: diamond + local database
  │
  ├─ Tier 2: HMMER hmmsearch vs. TIGRFAM/Pfam
  │    → Functional roles, subsystem classification
  │    → Sensitivity: MEDIUM (domain-level)
  │    → Dependency: hmmer3 + HMM profiles
  │
  └─ Tier 3: Heuristic header parsing (existing)
       → TF detection, transporter classification
       → Sensitivity: LOW (only works if headers are descriptive)
       → Dependency: none
```

### 13.3 Implementation

#### 13.3.1 DIAMOND Wrapper (`annotation/diamond.py`)

```python
@dataclass
class DiamondHit:
    query_id: str
    subject_id: str       # UniRef90/UniProt accession
    identity: float       # 0.0–1.0
    evalue: float
    bitscore: float
    stitle: str           # subject full title

def diamond_blastp(
    query_fasta: str,
    db_path: str,         # path to DIAMOND-formatted database
    evalue: float = 1e-5,
    max_target_seqs: int = 5,
    threads: int = 4,
) -> list[DiamondHit]:
    """Run DIAMOND blastp and parse tabular output."""
```

**Database requirements:**

| Database | Use | Format | Size | Source |
|----------|-----|--------|------|--------|
| UniRef90 + EC mapping | Sequence → EC number | DIAMOND db + TSV mapping | ~50 GB | NCBI/UniProt (preprocessed) |
| SwissProt + EC mapping | High-confidence EC | DIAMOND db + TSV mapping | ~100 MB | UniProt |
| KEGG KO mapping | Sequence → KO term | DIAMOND db + TSV mapping | ~5 GB | KEGG flat files |
| TIGRFAM | Domain → functional role | HMM profiles | ~100 MB | JCVI |
| Pfam | Domain detection | HMM profiles | ~2.2 GB | EBI |

**Lightweight mode** (no local databases): delegate to web APIs:
- UniProt REST API (free, rate-limited 3 req/s)
- KEGG API (free, rate-limited)
- NCBI BLAST+ web service

#### 13.3.2 EC Number Propagation (`annotation/ec_mapping.py`)

Current `REACTION_EQUATIONS` dict maps EC → equation string. Extend to:

```python
def ec_to_reactions(
    ec_numbers: list[str],
    ec_db: ECReactionDB | None = None,
) -> list[ECReactionEntry]:
    """Map EC numbers to metabolic reactions.
    
    Returns list of (reaction_id, equation, confidence) tuples.
    Confidence scales with sequence identity:
      - >90% identity → 1.0
      - 70–90% → 0.8
      - 50–70% → 0.6
      - <50% → 0.4 (remote homology, may be unreliable)
    """
```

#### 13.3.3 Protein Sequence Extraction (`annotation/sequences.py`)

```python
def extract_protein_sequences(
    genome_fasta: str,
    gff3: str | None = None,
) -> dict[str, str]:
    """Extract protein sequences from genome.
    
    If GFF3 provided: use CDS coordinates to extract nucleotide,
    translate with standard codon table.
    If no GFF3: assume FASTA contains protein sequences directly.
    """
```

### 13.4 Module Changes

| File | Change |
|------|--------|
| `annotation/diamond.py` | **NEW**: DIAMOND blastp wrapper |
| `annotation/ec_mapping.py` | Extend `ec_to_reactions()` to accept identity score |
| `annotation/sequences.py` | **NEW**: protein sequence extraction from genome |
| `gem_pipeline.py:72-102` | Replace `_GENE_EC_MAP` with DIAMOND-based annotation |
| `gem_pipeline.py:131-144` | Pass EC numbers and sequences to kcat/Km predictors |

---

## 14. Phase 8 — GRN-Metabolism Coupling

**Goal:** Connect gene regulatory network to metabolic flux via expression → enzyme levels → FBA bounds.

### 14.1 Problem

The GRN produces `RegulatoryEdge` objects (TF → target gene) but:
1. No expression quantities (how much mRNA/protein is produced)
2. No mapping from expression to enzyme concentrations
3. Enzyme capacity in `_run_gem()` is disabled because `enzyme_levels` is empty

### 14.2 Solution: Expression Inference Chain

```
GRN (TF → gene edges)
  │
  ├─ Promoter strength model
  │    → basal transcription rate per gene
  │    → TF binding modulation (Hill function)
  │
  ├─ mRNA dynamics
  │    → d[mRNA]/dt = transcription - degradation
  │    → steady-state: [mRNA] = transcription_rate / degradation_rate
  │
  ├─ Translation model
  │    → [Protein] = translation_rate × [mRNA] / degradation_rate
  │
  └─ Enzyme level
       → [Enzyme] = [Protein] × fraction_folded
       → feed into FBA: flux ≤ kcat × [Enzyme]
```

### 14.3 Expression Inference Module (`omics/expression_inference.py`)

```python
@dataclass
class ExpressionModel:
    """Simple gene expression model for prokaryotes."""
    # Promoter strength (relative, 0.0–1.0)
    promoter_strength: dict[str, float]   # gene_id → strength
    # mRNA half-life (minutes)
    mrna_half_life: dict[str, float]      # gene_id → t½ (min)
    # Protein half-life (minutes)
    protein_half_life: dict[str, float]   # gene_id → t½ (min)
    # Ribosome binding site strength (relative)
    rbs_strength: dict[str, float]        # gene_id → strength
    # TF regulation (from GRN)
    tf_effects: dict[str, list[tuple[str, float, float]]]  # gene → [(tf, Kd, n)]

def infer_expression(
    grn: GRN,
    annotations: dict[str, GeneAnnotation],
    model: ExpressionModel | None = None,
    environment: dict[str, float] | None = None,
) -> dict[str, float]:
    """Infer steady-state enzyme concentrations from GRN + expression model.
    
    Returns gene_id → relative enzyme level (0.0–1.0 scale).
    """
```

**Hill function for TF regulation:**

```python
def hill_function(
    tf_level: float,
    kd: float,
    n: float,
    effect: str = "activation",
) -> float:
    """Compute TF modulation of target gene expression.
    
    activation:   f = tf^n / (Kd^n + tf^n)
    repression:   f = Kd^n / (Kd^n + tf^n)
    """
```

### 14.4 Wire into `_run_gem()` (`sim_runtime.py:1974-1985`)

```python
# After GRN inference, compute expression levels
if result.grn is not None:
    from helixlang.omics.expression_inference import infer_expression
    enzyme_levels = infer_expression(
        grn=result.grn,
        annotations=result.annotations,
        environment=ext.get("gem_medium", "glucose_minimal"),
    )
    fba.set_enzyme_levels(enzyme_levels)
    
    # Now safe to enable enzyme capacity
    if result.kcat_predictions:
        ec = build_enzyme_capacity(result.consensus, result.kcat_predictions)
        fba.set_enzyme_capacity(ec)
```

### 14.5 Key Design Decisions

1. **Steady-state approximation**: For FBA, we use steady-state expression (not dynamic ODE), because FBA itself is a steady-state method. Dynamic expression is only needed for dFBA time-course.

2. **Default expression levels**: When no GRN/expression data is available, use a uniform default (e.g., 0.5 for all genes). This avoids the `ub=0` bug from empty `enzyme_levels`.

3. **Expression scale**: Enzyme levels are relative (0.0–1.0), not absolute concentrations. The absolute scale is absorbed into `enzyme_scale` in `EnzymeCapacity`.

### 14.6 Module Changes

| File | Change |
|------|--------|
| `omics/expression_inference.py` | **NEW**: expression inference from GRN |
| `metabolism.py:1183` | Fix default: `self.enzyme_levels.get(gene, 1.0)` instead of `0.0` |
| `sim_runtime.py:1974-1985` | Wire expression inference before enzyme capacity |
| `gem/grn_inference.py:200-235` | Fix `_predict_motifs()` to use real gene names |

---

## 15. Phase 9 — Dynamic Simulation (dFBA Integration)

**Goal:** Connect GEM-reconstructed model to `DynamicFluxBalance` for time-course simulation.

### 15.1 Problem

`_run_gem()` runs a single static FBA. The mature `DynamicFluxBalance` implementation (`metabolism.py:1454-1766`) only uses the hardcoded 37-reaction `ECOLI_CORE_MODEL`. The GEM-reconstructed model (~88 reactions) is never fed into dFBA.

### 15.2 Solution: GEM → dFBA Bridge

```python
def run_gem_dynamics(
    program: Program,
    duration: float = 24.0,         # hours
    dt: float = 0.1,                # time step (hours)
    initial_glucose: float = 10.0,  # mmol/L
    initial_biomass: float = 0.01,  # gDW/L
    medium: str = "glucose_minimal",
) -> DynamicSimulationResult:
    """Run dynamic simulation with GEM-reconstructed model.
    
    1. Run GEM pipeline (annotation → network → kinetics)
    2. Build MetabolicModel from consensus
    3. Create DynamicFluxBalance with GEM model
    4. Integrate batch culture ODEs over time
    5. Return time-course trajectory
    """
```

### 15.3 DynamicFluxBalance Extensions

Current dFBA only handles glucose + optional acetate. Extend to:

```python
class DynamicFluxBalance:
    def __init__(
        self,
        model: MetabolicModel,
        config: DynamicFBAConfig | None = None,
        # NEW: environment coupling
        environment: Environment | None = None,
        # NEW: expression dynamics
        expression_model: ExpressionModel | None = None,
    ):
        ...
    
    def step(self, dt: float) -> dict:
        """One time step: solve FBA → update pools → update environment."""
        # 1. Set uptake bounds from external concentrations (Michaelis-Menten)
        # 2. If expression_model: update enzyme levels from GRN
        # 3. Solve FBA
        # 4. Integrate batch ODEs (biomass, substrates, byproducts)
        # 5. Update environment (deposit byproducts, consume substrates)
        return trajectory_row
```

### 15.4 Multi-Substrate Support

Extend dFBA beyond glucose-only:

```python
@dataclass
class SubstrateConfig:
    """Configuration for a single substrate in dFBA."""
    metabolite: str           # e.g., "glc-D_e"
    exchange_reaction: str    # e.g., "EX_glc_e"
    ks: float                 # Michaelis constant (mmol/L)
    v_max: float              # max uptake rate (mmol/gDW/h)
    initial_concentration: float  # mmol/L
    is_limiting: bool = True  # limiting substrate for growth

# Default medium presets for dFBA
DFBA_MEDIA = {
    "glucose_minimal": [
        SubstrateConfig("glc-D_e", "EX_glc_e", ks=0.15, v_max=10.0, initial_concentration=10.0),
        SubstrateConfig("o2_e", "EX_o2_e", ks=0.02, v_max=20.0, initial_concentration=20.0),
    ],
    "lb": [
        SubstrateConfig("glc-D_e", "EX_glc_e", ks=0.15, v_max=10.0, initial_concentration=10.0),
        SubstrateConfig("o2_e", "EX_o2_e", ks=0.02, v_max=20.0, initial_concentration=20.0),
        SubstrateConfig("phe-L_e", "EX_phe_L_e", ks=0.01, v_max=0.5, initial_concentration=0.5, is_limiting=False),
    ],
}
```

### 15.5 Integration with `_run_gem()`

Add a `dynamic=true` field to `#gem` directive:

```helix
#gem organism=e_coli_k12 genome=genome.fasta medium=glucose_minimal dynamic=true duration=24.0 dt=0.1
#config backend=gem
```

When `dynamic=true`, `_run_gem()` creates a `DynamicFluxBalance` instead of a single static FBA.

### 15.6 Output

```python
@dataclass
class DynamicSimulationResult:
    """Time-course trajectory from dFBA."""
    time_points: list[float]              # hours
    biomass: list[float]                  # gDW/L
    substrates: dict[str, list[float]]    # metabolite → concentration over time
    byproducts: dict[str, list[float]]    # metabolite → concentration over time
    growth_rates: list[float]             # mu (h⁻¹) at each time point
    fluxes: list[dict[str, float]]        # flux distribution at each time point
    final_biomass: float                  # gDW/L at end
    doubling_time: float                  # hours (estimated from growth curve)
    diauxic_shift: bool | None            # True if acetate switch detected
```

### 15.7 Module Changes

| File | Change |
|------|--------|
| `metabolism.py:1454-1766` | Extend `DynamicFluxBalance` for multi-substrate + expression dynamics |
| `sim_runtime.py:1851-2072` | Add `dynamic=true` path in `_run_gem()` |
| `parser.py:429-457` | Parse `dynamic`, `duration`, `dt` fields from `#gem` |
| `omics/expression_inference.py` | Provide `infer_expression_at_time()` for dynamic expression |

---

## 16. Phase 10 — Environment Dynamics

**Goal:** Simulate organism in a changing environment (nutrient shifts, spatial gradients, population interactions).

### 16.1 Problem

Current dFBA only models closed batch cultures. Real environments involve:
- Fed-batch (nutrient feeding schedules)
- Chemostat (continuous inflow/outflow)
- Spatial gradients (oxygen, nutrients)
- Multi-species interactions (cross-feeding, competition)

### 16.2 Solution: Environment Class Extension

The existing `Environment` class (`environment.py`) already supports `ConcentrationField` objects. Extend with:

```python
@dataclass
class EnvironmentConfig:
    """Configuration for simulation environment."""
    # Batch culture (default)
    initial_glucose: float = 10.0     # mmol/L
    initial_oxygen: float = 20.0      # mmol/L (dissolved)
    temperature: float = 37.0         # °C
    ph: float = 7.0
    
    # Fed-batch
    feed_schedule: list[FeedEvent] | None = None
    
    # Chemostat
    dilution_rate: float | None = None  # h⁻¹ (None = batch)
    
    # Spatial
    spatial_dimensions: int = 0        # 0=well-mixed, 1=1D, 2=2D, 3=3D
    grid_size: tuple[int, ...] | None = None
    diffusion_coefficients: dict[str, float] | None = None

@dataclass
class FeedEvent:
    """Nutrient feeding event."""
    time: float              # hours
    metabolite: str          # e.g., "glc-D_e"
    concentration: float     # mmol/L added
    volume_fraction: float = 0.0  # dilution factor
```

### 16.3 Simulation Modes

| Mode | Config | Behavior |
|------|--------|----------|
| **Batch** | `dilution_rate=None, feed_schedule=None` | Closed system, nutrients deplete |
| **Fed-batch** | `feed_schedule=[...]` | Nutrients added at scheduled times |
| **Chemostat** | `dilution_rate=0.1` | Continuous inflow/outflow at fixed rate |
| **Spatial** | `spatial_dimensions≥1` | PDE-based diffusion + reaction |

### 16.4 Integration with Population Backend

For spatial simulations, the `population.CellPopulation3D` backend already handles:
- Lattice-based spatial diffusion
- Per-cell dFBA
- Shared-batch dFBA (Brunner & Chai 2020)

The GEM-reconstructed model can be plugged into this framework:

```python
def run_spatial_simulation(
    program: Program,
    grid_size: tuple[int, int] = (10, 10),
    duration: float = 48.0,
) -> SpatialSimulationResult:
    """Run spatial simulation with GEM model in a 2D environment."""
```

### 16.5 Module Changes

| File | Change |
|------|--------|
| `environment.py` | Add `FeedEvent`, `EnvironmentConfig` dataclasses |
| `metabolism.py:1454-1766` | Add `chemostat_step()`, `fed_batch_step()` methods |
| `population.py` | Wire GEM-reconstructed model into spatial dFBA |
| `sim_runtime.py` | Add `backend=gem_spatial` dispatch |

---

## 17. Updated Data Model

### 17.1 New Dataclasses

```python
# Phase 7: Universal annotation
@dataclass
class DiamondHit:
    query_id: str
    subject_id: str
    identity: float
    evalue: float
    bitscore: float
    stitle: str

@dataclass
class ProteinSequence:
    gene_id: str
    sequence: str
    start: int
    end: int
    strand: str
    contig: str

# Phase 8: Expression inference
@dataclass
class ExpressionModel:
    promoter_strength: dict[str, float]
    mrna_half_life: dict[str, float]
    protein_half_life: dict[str, float]
    rbs_strength: dict[str, float]
    tf_effects: dict[str, list[tuple[str, float, float]]]

@dataclass
class ExpressionState:
    """Snapshot of expression levels at a point in time."""
    gene_levels: dict[str, float]        # gene → relative level (0–1)
    enzyme_levels: dict[str, float]      # gene → enzyme concentration
    mrna_levels: dict[str, float]        # gene → mRNA level
    timestamp: float = 0.0

# Phase 9: Dynamic simulation
@dataclass
class SubstrateConfig:
    metabolite: str
    exchange_reaction: str
    ks: float
    v_max: float
    initial_concentration: float
    is_limiting: bool = True

@dataclass
class DynamicSimulationResult:
    time_points: list[float]
    biomass: list[float]
    substrates: dict[str, list[float]]
    byproducts: dict[str, list[float]]
    growth_rates: list[float]
    fluxes: list[dict[str, float]]
    final_biomass: float
    doubling_time: float
    diauxic_shift: bool | None

# Phase 10: Environment
@dataclass
class FeedEvent:
    time: float
    metabolite: str
    concentration: float
    volume_fraction: float = 0.0

@dataclass
class EnvironmentConfig:
    initial_glucose: float = 10.0
    initial_oxygen: float = 20.0
    temperature: float = 37.0
    ph: float = 7.0
    feed_schedule: list[FeedEvent] | None = None
    dilution_rate: float | None = None
    spatial_dimensions: int = 0
    grid_size: tuple[int, ...] | None = None
    diffusion_coefficients: dict[str, float] | None = None

# Updated GemPipelineResult
@dataclass
class GemPipelineResult:
    # ... existing fields ...
    
    # NEW: expression inference
    expression_model: ExpressionModel | None = None
    expression_state: ExpressionState | None = None
    
    # NEW: dynamic simulation
    dynamic_result: DynamicSimulationResult | None = None
    
    # NEW: validation
    validation: GemValidation | None = None

@dataclass
class GemValidation:
    predicted_growth_rate: float
    experimental_growth_rate: float | None
    growth_rate_error: float | None       # |predicted - experimental|
    essential_genes_correct: float | None  # accuracy (0–1)
    auxotrophy_correct: float | None       # accuracy (0–1)
    blocked_reactions: list[str]           # reactions with no flux
    mass_balance_violations: list[str]     # metabolites with S·v ≠ 0
```

### 17.2 Updated `#gem` Directive

```helix
#gem organism=e_coli_k12 genome=genome.fasta
    medium=glucose_minimal
    use_database=true
    include_spontaneous=true
    gapfill=true
    dynamic=false          # NEW: enable dFBA
    duration=24.0          # NEW: simulation hours (when dynamic=true)
    dt=0.1                 # NEW: time step (hours)
    expression=true        # NEW: enable expression inference
```

| Field | Type | Default | Phase | Description |
|-------|------|---------|-------|-------------|
| `organism` | string | required | — | Organism identifier |
| `genome` | string | required | — | Path to genome FASTA |
| `medium` | string | `glucose_minimal` | — | Growth medium preset |
| `use_database` | bool | `true` | 4 | Use curated regulatory interactions |
| `include_spontaneous` | bool | `true` | 3 | Include spontaneous reactions |
| `gapfill` | bool | `true` | 3 | Run gap-filling |
| `dynamic` | bool | `false` | 9 | Enable dynamic FBA simulation |
| `duration` | float | `24.0` | 9 | Simulation duration (hours) |
| `dt` | float | `0.1` | 9 | Time step (hours) |
| `expression` | bool | `false` | 8 | Enable expression inference from GRN |

---

## 18. Implementation Roadmap

### Sprint 1: Fix Annotation for Universal Organisms (Weeks 1–3)

**Goal:** Pipeline works for any prokaryotic organism, not just E. coli.

| # | Task | Priority | Module | Effort |
|---|------|----------|--------|--------|
| 1.1 | Implement DIAMOND blastp wrapper | P0 | `annotation/diamond.py` | 3 days |
| 1.2 | Preprocess UniRef90 + EC mapping database | P0 | `annotation/diamond.py` | 2 days |
| 1.3 | Implement protein sequence extraction from genome | P0 | `annotation/sequences.py` | 2 days |
| 1.4 | Replace `_GENE_EC_MAP` with DIAMOND-based annotation | P0 | `gem_pipeline.py` | 2 days |
| 1.5 | Pass EC numbers and sequences to kcat/Km predictors | P0 | `gem_pipeline.py` | 1 day |
| 1.6 | Fix kcat predictor to use EC lookup (not just reaction_id) | P0 | `kinetics/kcat_predictor.py` | 1 day |
| 1.7 | Fix Km estimator to use EC + substrate lookup | P0 | `kinetics/km_estimator.py` | 1 day |
| 1.8 | Add lightweight mode (UniProt REST API fallback) | P1 | `annotation/diamond.py` | 2 days |
| 1.9 | Integration tests: B. subtilis, P. aeruginosa | P1 | `tests/test_gem_integration.py` | 2 days |

### Sprint 2: Fix GRN → Expression → Enzyme Levels (Weeks 4–6)

**Goal:** GRN produces real expression quantities that constrain FBA.

| # | Task | Priority | Module | Effort |
|---|------|----------|--------|--------|
| 2.1 | Fix `_predict_motifs()` to use real gene names | P0 | `gem/grn_inference.py` | 1 day |
| 2.2 | Implement expression inference from GRN | P0 | `omics/expression_inference.py` | 5 days |
| 2.3 | Fix default enzyme_levels to 1.0 (not 0.0) | P0 | `metabolism.py:1183` | 0.5 day |
| 2.4 | Wire expression inference into `_run_gem()` | P0 | `sim_runtime.py:1974-1985` | 2 days |
| 2.5 | Re-enable enzyme capacity with expression data | P0 | `sim_runtime.py` | 1 day |
| 2.6 | Add `expression=true` field to `#gem` directive | P1 | `parser.py` | 1 day |
| 2.7 | Validation: compare growth rate with/without expression | P1 | `tests/` | 2 days |

### Sprint 3: LP Gap-Filling + Model Validation (Weeks 7–9)

**Goal:** Models are validated and gap-filled optimally.

| # | Task | Priority | Module | Effort |
|---|------|----------|--------|--------|
| 3.1 | Implement LP-based gap-filling | P0 | `gem/gapfill.py` | 5 days |
| 3.2 | Implement growth rate validation (FBA → expected range) | P0 | `gem/validation.py` | 2 days |
| 3.3 | Implement gene essentiality validation | P1 | `gem/validation.py` | 2 days |
| 3.4 | Implement mass balance check | P1 | `gem/validation.py` | 1 day |
| 3.5 | Add validation report to `GemPipelineResult` | P1 | `gem_pipeline.py` | 1 day |
| 3.6 | Taxonomy-aware biomass templates | P1 | `gem/biomass.py` | 3 days |
| 3.7 | SBML Level 3 export | P1 | `gem/sbml_export.py` | 3 days |

### Sprint 4: dFBA Integration + Dynamic Simulation (Weeks 10–12)

**Goal:** GEM-reconstructed model runs dynamic time-course simulation.

| # | Task | Priority | Module | Effort |
|---|------|----------|--------|--------|
| 4.1 | Wire GEM model into `DynamicFluxBalance` | P0 | `sim_runtime.py` | 3 days |
| 4.2 | Multi-substrate dFBA (glucose + O₂ + acetate) | P0 | `metabolism.py:1454-1766` | 3 days |
| 4.3 | Add `dynamic=true` path to `_run_gem()` | P0 | `sim_runtime.py` | 2 days |
| 4.4 | Implement `DynamicSimulationResult` output | P0 | `sim_runtime.py` | 2 days |
| 4.5 | Fed-batch support | P1 | `metabolism.py` | 2 days |
| 4.6 | Chemostat support | P1 | `metabolism.py` | 2 days |
| 4.7 | Integration test: batch culture diauxic shift | P1 | `tests/` | 2 days |

### Sprint 5: Environment + Spatial + Polish (Weeks 13–16)

**Goal:** Full environment dynamics + production readiness.

| # | Task | Priority | Module | Effort |
|---|------|----------|--------|--------|
| 5.1 | Spatial dFBA with GEM model | P1 | `apps/spatial_dfba.py` | 5 days |
| 5.2 | Nutrient feeding schedules | P1 | `environment.py` | 2 days |
| 5.3 | Population-level GEM simulation | P2 | `population.py` | 5 days |
| 5.4 | CLI: `--gem --dynamic` flag | P1 | `cli.py` | 1 day |
| 5.5 | Web API: `/api/gem/simulate-dynamic` | P1 | `server.py` | 2 days |
| 5.6 | Documentation update (doc/20 final) | P1 | `doc/20` | 2 days |
| 5.7 | Benchmark: E. coli growth rate vs. literature | P0 | `tests/` | 3 days |
| 5.8 | Benchmark: B. subtilis growth rate vs. literature | P0 | `tests/` | 3 days |

### Dependency Graph

```
Sprint 1 (Universal Annotation)
    │
    ├──→ Sprint 2 (Expression Inference)
    │         │
    │         └──→ Sprint 4 (dFBA Integration)
    │                   │
    │                   └──→ Sprint 5 (Environment + Spatial)
    │
    └──→ Sprint 3 (Gap-Filling + Validation)
              │
              └──→ Sprint 4 (dFBA Integration)
```

Sprints 2 and 3 can run in parallel after Sprint 1 completes.

### Critical Path

```
Week  1-3:  Universal annotation (DIAMOND + EC propagation)
Week  4-6:  Expression inference + enzyme levels
Week  7-9:  LP gap-filling + model validation (parallel with 4-6)
Week 10-12: dFBA integration + dynamic simulation
Week 13-16: Environment dynamics + spatial + production polish
```

**Total estimated effort:** 16 weeks (4 months) for full pipeline.

**Minimum viable product (MVP):** Sprints 1-2 (6 weeks) — universal annotation + expression inference. This alone makes the pipeline work for any prokaryote with a genome FASTA.

---

## 19. References

1. Orth, J. D., Thiele, I., & Palsson, B. Ø. (2010). What is flux balance
   analysis? *Nature Biotechnology*, 28(3), 245–248.

2. Henry, C. S., et al. (2010). High-throughput generation, optimization and
   analysis of genome-scale metabolic models. *Nature Biotechnology*, 28(9),
   977–982.

3. Feist, A. M., et al. (2023). ModelSEED v2: High-throughput genome-scale
   metabolic model reconstruction with enhanced energy biosynthesis pathway
   prediction. *bioRxiv*.

4. Pereira, R., et al. (2016). Fast automated reconstruction of
   genome-scale metabolic models for microbial communities. *Nature
   Communications*, 7, 12935.

5. Zelezniak, A., et al. (2018). Bacterial metabolism-driven modes of
   cross-feeding emerge from genome-scale metabolic networks. *mSystems*.

6. Wang, H., et al. (2018). RAVEN 2.0: A high-resolution RAVEN toolkit for
   automated genome-scale metabolic model reconstruction. *Bioinformatics*.

7. DiMucci, D., et al. (2023). Reconstructor: A fast and automated approach
   for reconstructing high-quality genome-scale metabolic models. *mSystems*.

8. Ghilardi, A., et al. (2024). pan-Draft: automated reconstruction of
   species-representative metabolic models from multiple genomes. *Genome
   Biology*, 25, 294.

9. Matveishina, E. K., et al. (2025). GEMsembler: consensus model assembly
   and structural analysis of genome-scale metabolic models. *mSystems*.

10. Aite, M., et al. (2018). AuReMe: a pipeline for the automatic
    reconstruction and curation of genome-scale metabolic models.
    *Bioinformatics*.

11. Chen, Y., et al. (2024). Reconstruction, simulation and analysis of
    enzyme-constrained metabolic models using GECKO Toolbox 3.0. *Nature
    Protocols*, 19(3), 629–667.

12. Wu, K., et al. (2025). Applications of machine learning in the
    reconstruction and curation of genome-scale metabolic models.
    *Synthetic Biology Journal*, 6(3), 566–584.

13. Mendoza, S. N., et al. (2025). Selecting methods for draft GEM
    generation in multicellular eukaryotes: a comparative analysis.
    *bioRxiv*.

14. Santos-Zavaleta, A., et al. (2019). RegulonDB v10.5: Big data on the
    regulation of *Escherichia coli* K-12. *Nucleic Acids Research*.

15. Wang, Y., et al. (2024). Inferring gene regulatory networks from
    single-cell multiome data using atlas-scale external data. *Nature
    Biotechnology*, 42, 884–895.

16. Aibar, S., et al. (2017). SCENIC: single-cell regulatory network
    inference and clustering. *Nature Methods*, 14, 1083–1086.

17. Huynh-Thu, V. A., et al. (2010). Inferring regulatory networks from
    expression data using tree-based methods. *PLoS ONE*, 5(9).

18. Thiele, I., & Palsson, B. Ø. (2010). A protocol for generating a
    high-quality genome-scale metabolic reconstruction. *Nature Protocols*,
    5, 93–121.

19. Price, M. N., et al. (2006). Predicting transcription start sites by
    analyzing genomic sequences with a hidden Markov model. *BMC
    Bioinformatics*.

20. Lu, H., et al. (2023). A consensus *S. cerevisiae* metabolic model
    Yeast8 and its ecosystem for comprehensively probing cellular metabolism.
    *Nature Communications*, 14, 3298.

21. Mahadevan, R., Edwards, J. S., & Doyle, F. J. (2002). Dynamic flux
    balance analysis of diauxic growth in *Escherichia coli*. *Biophysical
    Journal*, 83(3), 1331–1340.

22. O'Brien, E. J., Lerman, J. A., Chang, R. L., Hyduke, D. R., & Palsson,
    B. Ø. (2013). Genome-scale models of metabolism and gene expression
    extend and refine growth phenotype predictions. *Molecular Systems
    Biology*, 9(1), 693.

23. Boutell, J. M., et al. (2024). GECKO Toolbox 3.0: enzyme-constrained
    genome-scale metabolic models. *Nature Protocols*, 19(3), 629–667.

24. Brunner, J., & Chai, R. (2020). Shared-batch dynamic flux balance
    analysis for large-population simulations. *Bioinformatics*.

25. Wolfe, A. J. (2005). The acetate switch. *Microbiology and Molecular
    Biology Reviews*, 69(1), 12–50.

---

## 20. Relationship to other documents

- `doc/17-project-details-and-frontier-bio-applications.md` — the overall
  architecture and capability mapping; this document's GEM pipeline adds the
  metabolic-reconstruction layer to that architecture.
- `doc/19-whole-organism-lifecycle-simulation.md` — the whole-organism
  ecosystem vision (Phases A–D); the GEM pipeline supplies the metabolic
  models that the ecosystem's `Species` consume when `gem_driven=true`
  (§12.3, doc/21).
- `doc/21-gem-ecosystem-bridge.md` — the bridge layer that connects this
  document's GEM pipeline output to the ecosystem simulation (§12.3);
  `gem_to_species()`, `Species.metabolic_model`, `Patch._growth_rate_gem()`.
- `doc/18-programmable-cell-population-simulation.md` — the population
  simulation roadmap; the GEM pipeline enables genome-scale metabolic
  models for population-level FBA.
