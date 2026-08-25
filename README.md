<div align="center">

# 🧬 HelixLang

_"We are moving from reading the genetic code to writing it. " — J. Craig Venter_

**DNA is source code. Codons are mnemonics. The ribosome is a VM. The cell is the runtime.**

A domain-specific language where biological genetic material is the source, binary bytecode is the intermediate representation, and the execution semantics simulate biological behavior and morphology.

[Quick Start](#quick-start) ·
[Examples](#language-examples) ·
[Architecture](#architecture) ·
[API](#api--web-visualization) ·
[IDE Plugin](#ide-plugin-pycharm) ·
[Documentation](#documentation) ·
[Contributing](#contributing) ·
[License](#license)

![img.png](img.png)

[![PyPI - Version](https://img.shields.io/pypi/v/helixlang)](https://pypi.org/project/helixlang/)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/helixlang)](https://pypi.org/project/helixlang/)

</div>

---

## Why HelixLang?

Traditional biological modeling tools treat DNA as **data** — a passive string to be analyzed. HelixLang treats DNA as **code**: every codon is an instruction, every gene is a function, every cell is a runtime. The result is a single language that spans the full biological hierarchy — from individual codons to whole-cell physiology to multi-species ecosystems — with a real compiler, a real bytecode VM, 22 quantitative simulation backends, and a genome-to-ecosystem pipeline that can load real genome-scale metabolic models and run them in ecological context.

```helix
#gene name=hello
ATG GCT GGT TAA          # START -> BUILD_PROTEIN -> BUILD_MEMBRANE -> HALT
#end
```

Feed that sequence to the HelixLang compiler and you get a real bytecode program you can run, debug, and disassemble — just like Python or Java bytecode.

### Project at a glance

| Metric | Value |
|--------|-------|
| Source modules | 103 (80 top-level + 14 subpackages) |
| Test cases | 2721 (all passing, 81% coverage) |
| `.helix` examples | 55 (complete source) |
| Documentation | 31 files, 22,000+ lines |
| Runtime dependencies | **zero** (all optional: numpy, biopython, flask, esm, torch) |

### Highlights

#### Compiler & Language

- ⚙️ **Full compiler pipeline** — Lexer → Parser → AST → Semantic → Compiler → Bytecode → VM, with disassembly, debugging, and `.helixc` binary serialization.
- 🧬 **Biologically isomorphic** — language primitives map one-to-one to biological entities: codons, amino acids, genes, promoters, operons, regulators, proteins.
- 🎲 **Degeneracy as aliasing** — 64 codons map to ~30 opcodes; the third wobble position acts as an operand modifier, mirroring real biological degeneracy.
- 🛠️ **Zero dependencies** — the core compiler/VM uses only the Python standard library; numpy / biopython / flask are optional enhancements.

#### Simulation Runtimes

- 🖥️ **22 simulation backends from one source** — `#config backend` or `#sim kind=` selects from whole-cell physiology, 3D cell colonies, static/dynamic FBA, calibration, GEM reconstruction, full genome-scale model import, multi-species ecosystem, and 16 specialized backends — all from `.helix` source with `seed=` determinism and a `POST /api/sim/run` web endpoint.
- 🧬 **Whole-cell realism** — `VirtualCell` is a physically complete virtual organism: cell-cycle phasing with scheduled **Cooper–Helmstetter chromosome replication**, true volume in µm³ with **adder** size control, chaperone-mediated protein maturation & QC, and **enzyme-constrained FBA** over intracellular pools.
- 🧫 **Colony-scale spatial simulation** — per-cell GRN + bytecode, CROMICS cell-crowding diffusion, shoving/force mechanics, expression-gated FBA, metabolic stratification; a shared **4338-gene** sparse regulatory network across thousands of cells.
- 🌊 **Microfluidics in the box** — lattice-Boltzmann D2Q9 (2D) and D3Q19 (3D) solvers drive medium through microfluidic channels around growing colonies: no-slip obstacles, Stokes-drag drift, Hertzian contacts, substrate advection.
- 🌍 **Evolution with a physical body** — DNA programs evolve in real 3D space: a biological mutation spectrum mutates the source, the compiler revalidates the reading frame and recompiles, and fitness is *spatial behavior* — colony radius × core survival in an oxygen-poor colony core.
- 🎯 **Inverse modeling** — recovers hidden biophysical parameters from noisy, lab-realistic mixed observables at both whole-cell and colony scale (Karr 2012 / Virtual Cell Challenge–style weighted fitting).

#### Genome-to-Ecosystem Pipeline

- 🌱 **Genome → GEM → Ecosystem end-to-end** — a single `.helix` source can specify real genomes and run multi-species, multi-population simulation starting from genes. The six-phase GEM reconstruction pipeline (`#gem`) turns genome FASTA into a functional genome-scale metabolic model, and the GEM↔Ecosystem bridge (`gem_driven=true`) feeds the reconstructed metabolic models directly into the ecosystem tick loop.
- 🔬 **GEM reconstruction** — UniProt 3-tier functional annotation (ID mapping → NCBI BLASTP → UniProt sequence search), ~200 EC→reaction mappings, ~90 universal prokaryotic reactions, consensus bottom-up + top-down reconstruction, LP gap-filling, biomass reaction assembly, kcat prediction (60+ EC entries, organism scaling), Km estimation, GRN inference (RegulonDB PWMs + ChIP-seq evidence), FBA validation.
- 📥 **Full genome-scale model import** — load real BiGG/BioModels SBML models directly (`#gem model=path/to/model.xml`), bypassing reconstruction entirely. Validated against E. coli iML1515 (2712 reactions, 1877 metabolites, 1516 genes; growth rate 0.821 h⁻¹) and Synechocystis iJN678 (863 reactions, 795 metabolites, 622 genes; photoautotrophic growth 0.292 h⁻¹). CobraPy import with SBML fallback, automatic biomass reaction detection, and BiGG metabolite ID normalization.
- 🧬 **DNA → Structure → Kinetics → ecGEM → Ecosystem** — ESM3 end-to-end protein structure prediction from sequence alone (no MSA required; Lin et al. 2023), sequence-based kcat/Km prediction using ESM-2 embeddings + BRENDA calibration, auto ecGEM construction, community FBA with per-organism ecGEMs and cross-feeding. The full pipeline runs from a single genome FASTA file via `run_full_pipeline()`. Example 55 demonstrates the complete chain with Xenobacter alienus — 24 genes, 5 pathways, 22 enzyme kinetic parameters, 42-reaction core model with 24 enzyme constraints.
- 🌿 **Photoautotrophic dFBA** — light-dependent FBA for cyanobacteria and photoautotrophs: CO₂ uptake via Monod kinetics modulated by PAR light saturation, oxygen evolution tracked as a field, light scalar diurnal forcing, and photo-vmax separate from heterotrophic vmax. Photoautotrophic examples (Synechocystis PCC 6803) run with real iJN678 model fluxes.
- 🔗 **GRN → FBA closed loop** — regulatory edges carry `target_reaction` to directly clamp FBA exchange bounds: transcription factors repress/activate specific metabolic reactions, closing the gene-to-metabolism feedback loop.
- 🌡️ **Enzyme correction** — Arrhenius temperature dependence + Gaussian pH profile scale internal FBA bounds at non-optimal conditions.
- 📊 **Density-dependent enzyme scaling** — logistic model: full metabolic capacity at low biomass, decreasing toward zero as population approaches carrying capacity.
- 🧬 **CRISPR → enzyme feedback** — CRISPR knockdown and evolve edits automatically update enzyme levels and kcat values; FBA re-solves with the modified enzyme pool, propagating genetic changes to metabolic phenotype.
- 🌍 **Multi-species ecosystem** — `#sim kind=ecosystem` with `#species`/`#patch`: Lotka-Volterra competition/predation, cross-feeding, CENTURY litter/SOM pools, C/N biogeochemistry, Q10/DAMM temperature dependence, diurnal/seasonal forcing, photoautotrophy, event-driven Scheduler, invasion-fitness evolution.

#### Biology & Data

- 🔬 **Real biological data** — mutation rates, transition:transversion ratios, codon usage tables, tRNA abundances, and CAI sourced from published measurements (Lee 2012, Drake 1991, Ikemura 1985, Dong 1996, Sharp & Li 1987).
- 💾 **DNA data storage** — built-in Goldman 2013 rotating-key encoding and Erlich-Zielinski 2017 DNA Fountain code, encoding arbitrary byte streams into synthesizable DNA.
- 🧪 **Frontier biology modules** — programmable cells, stochastic gene expression (telegraph-promoter Fano + Gillespie SSA), environment-coupled Monod metabolism, dynamic FBA batch/diauxic simulation, CRISPR-Cas9 guide design, epigenetics, protein structure prediction (ESM3), directed evolution, sequence-based enzyme kinetics (ESM-2 + BRENDA).

---

## 🚀 Quick Start

### Install

HelixLang is published on **PyPI** — install the released package:

```bash
# Core (compiler + VM + CLI)
pip install helixlang

# Recommended: install all optional extras
pip install "helixlang[fast,web,bio,ml]"
```

> The core installs with **zero runtime dependencies** — only the Python standard library.
> Contributors working from a source checkout instead use `pip install -e ".[dev,fast,web,bio]"` (see [Contributing](#contributing)).

| Extra | Capability | Dependencies |
|-------|-----------|--------------|
| `fast` | vectorized mutation / reaction-diffusion speedup | numpy |
| `web` | Flask visualization frontend | flask |
| `bio` | physical DNA codec + IUPAC validation + full GEM import | biopython, reedsolo, cobra |
| `ml` | ESM3 protein structure + ESM-2 sequence-based kinetics | esm, torch |
| `human` | SMILES parsing + molecular properties for drug simulation | rdkit (2026.03.5) |
| `dev` | tests + coverage (source checkouts) | pytest, pytest-cov, scipy |

### Up and running in 30 seconds

```bash
# Run an example
helixlang examples/01_hello_dna.helix

# Disassemble the bytecode
helixlang examples/01_hello_dna.helix --disassemble

# Run a GEM reconstruction from genome FASTA
helixlang examples/46_gem_reconstruction.helix --gem

# Run the full-chain custom organism pipeline (DNA → structure → kinetics → ecGEM → ecosystem)
helixlang examples/55_custom_organism_ecosystem.helix --full-pipeline

# Launch the web visualization (http://127.0.0.1:5000)
helixlang --serve
```

### As a Python library

```python
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.compiler import Compiler
from helixlang.vm import CellVM
from helixlang.codon_table import get_table

src = "#gene name=hello\nATG GCT GGT TAA\n#end\n#config ticks=10\n"

table = get_table("standard")
tokens = list(Lexer(src).tokens())
program = Parser(tokens).parse()
chunk = Compiler(table).compile(program)

vm = CellVM(chunk, program)
trace = vm.run(program.config.ticks)
print(f"Ran {len(trace)} ticks, final energy = {trace[-1]['energy']}")
```

---

## 🧬 IDE Plugin

Write, inspect, and debug `.helix` programs right inside **PyCharm 2022.2+** (Community or
Professional) with the sibling repository
**[SeanHank/HelixLang-LSP-Plugin](https://github.com/SeanHank/HelixLang-LSP-Plugin)**: live
diagnostics, hover docs, completion, navigation, semantic highlighting, inlay hints, a bytecode
disassembler, and a line debugger, all over Language Server Protocol.

Install the language server (once, per Python interpreter):

```bash
pip install helixlang-lsp
```

Then grab the plugin zip from that repo's [Releases](https://github.com/SeanHank/HelixLang-LSP-Plugin/releases)
page and install it via **Settings → Plugins → ⚙ → Install Plugin from Disk…**. Full install steps,
build-from-source instructions, and the design docs live in the
[plugin README](https://github.com/SeanHank/HelixLang-LSP-Plugin).

---

## 🌟 Language Examples

### 1. The lac Operon — Gene Regulatory Network (GRN)

p_lacI constitutively expresses lacI → represses p_lac → lacZ/lacY shut off. A real negative-feedback loop:

```helix
#promoter name=p_lac  strength=0.5
#promoter name=p_lacI strength=-0.5

#gene name=lacZ promoter=p_lac
ATG GCT GCT GCT TAA
#end

#gene name=lacI promoter=p_lacI
ATG GCT GCT TAA
#end

#regulate lacI  -> p_lac  strength=-0.9     # lacI represses p_lac
#regulate p_lac -> lacZ  strength=+0.9      # p_lac activates lacZ

#config ticks=20 output=csv
```

### 2. Turing Patterns — Gray-Scott Reaction-Diffusion

```helix
#promoter name=p_pigment strength=-0.4
#gene name=pigment promoter=p_pigment
ATG GAT GAT GAA TAA
#end

#field size=32 F=0.035 k=0.065 Du=0.16 Dv=0.08

#config ticks=100 react_steps=2
```

Render the output to a PPM image: `helixlang examples/04_turing_pattern.helix --png turing`

### 3. CRISPR-Cas9 Gene Editing

```helix
#gene name=target_gene
ATG GCT GCT GCT GCT GCT GCT GCT GCT GCT TAA
#end

#crispr target=target_gene position=50 new_sequence="GGGG" cas=SpCas9 repair=HDR
```

Equivalent Python API call:

```python
from helixlang.crispr import design_guide, cut_dna, on_target_score

guide = design_guide("ATCGATCGATCGATCGATCGGATC", cas_variant="SpCas9")
print(f"On-target score: {on_target_score(guide):.3f}")
edited = cut_dna("ATCGATCGATCGATCGATCGGATC", guide, repair="NHEJ")
```

### 4. Evolution Simulation — Mutation + Selection + Drift

```helix
#gene name=ancestor
ATG CTG CTG CTG CTG CTG CTG CTG CTG CTG TAA
#end

#evolve target=ancestor generations=100 mutation_rate=0.01 population_size=1000
```

Parameters come from real papers: E. coli substitution rate 2.2e-10 /nt/gen (Lee 2012), transition:transversion ≈ 3:1 (Stoltzfus 2009).

### 5. DNA Data Storage — Write source code into DNA

```bash
# .helix source -> DNA sequence (Goldman rotating-key encoding)
helixlang examples/01_hello_dna.helix --encode-dna goldman

# High-density fountain code (~1.57 bit/nt, near the Shannon limit)
helixlang examples/01_hello_dna.helix --encode-dna erlich

# Simulate 30 cycles of PCR error injection
helixlang examples/01_hello_dna.helix --encode-dna goldman --pcr-cycles 30

# DNA -> source (reverse decode)
helixlang decoded.fasta --decode-dna oligos.fasta
```

```python
from helixlang.dna_codec import helix_to_dna, dna_to_helix

enc = helix_to_dna("#gene name=hello\nATG TAA\n#end\n", scheme="goldman")
print(enc["oligos"][0]["full"])              # 117 nt DNA sequence
recovered = dna_to_helix(enc, scheme="goldman")
assert recovered == "#gene name=hello\nATG TAA\n#end\n"
```

> 💡 Note the two distinct pipelines: the **compiler** translates `ATG GCT` into VM opcode bytes; the **DNA codec** maps the entire `.helix` file as a byte stream onto ACGT strings for real wet-lab DNA data storage. The two are fully orthogonal.

### 6. Simulation Backends — quantitative runs from `.helix`

`#config backend` selects the quantitative simulator instead of the bytecode VM (see `doc/09-bio-instructions.md` §6.1). All 55 examples ship in `examples/`:

| Backend | Representative examples | What it does |
|---|---|---|
| `fba` | `10`, `20`, `35` | Static/dynamic FBA: biomass solve, diauxic batch curves, acetate switch |
| `whole_cell` | `30`, `31`, `34` | Cooper–Helmstetter replication, adder size control, enzyme-constrained FBA |
| `population` | `21`, `32`, `37`, `38`, `39` | Per-cell GRN+bytecode colonies, dFBA, 4338-gene GRN, LBM microfluidics |
| `ecosystem` | `41`, `43`, `50`, `53`, `54`, `55` | Multi-species Lotka-Volterra, cross-feeding, diurnal forcing, full GEM models, **full-chain custom organism pipeline (DNA → structure → kinetics → ecGEM → ecosystem)** |
| `gem` | `46`, `47`, `48`, `49` | Genome→GEM reconstruction, E. coli / Synechocystis / B. subtilis / S. cerevisiae |
| `calibration` | `34`, `36` | Inverse modeling: recover hidden parameters from mixed observables |
| `#sim kind=` | `11`, `14`, `15`, `23`, `25`–`29` | Protein structure, synbio design, 3D morphology, digital evolution, fate analysis, directed evolution |

```bash
helixlang examples/10_metabolism_fba.helix --json   # machine-readable
helixlang examples/31_whole_cell_adder.helix --csv
helixlang examples/53_ecoli_full_model.helix        # full iML1515 GEM
```

Deterministic with `seed=`; same source parses under the classic backend
(`backend=classic`) for a bit-identical bytecode run.

### Reference data for GEM pipeline

The GEM reconstruction pipeline (`#gem`) needs reference databases. Download them with the built-in script:

```bash
# Python (cross-platform: Windows / Linux / macOS)
python scripts/download_data.py            # download everything
python scripts/download_data.py pfam       # Pfam only
python scripts/download_data.py ecoli      # E. coli reference only

# Bash (Linux / macOS)
bash scripts/download_data.sh
```

This downloads **Pfam-A.hmm** (~2.1 GB, CC0 public domain) from EBI and **E. coli K-12 MG1655** genome + proteome from NCBI/UniProt into the `data/` directory (gitignored).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HelixLang source program (.helix)              │
│   #gene promoter=strong                                          │
│   ATG GCT GGT TAA    # DNA triplet stream + annotation blocks    │
│   #end                                                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Lexer (split codons by 3 bases + annotation tokens)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Token -> Parser (ORF grouping: START..STOP = one gene) -> AST    │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Semantic analysis (reading frame / pairing / symbol table)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Compiler: codon-table decoding                                   │
│    ATG->OP_START  GCT->OP_BUILD_PROTEIN  GGT->OP_BUILD_MEMBRANE   │
│    TAA->OP_HALT   GTA->OP_MOVE(arg=South)  ...                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Chunk (bytecode + constant pool + line table)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  CellVM: stack-based bytecode virtual machine = "the ribosome"    │
│    ┌──────────┬──────────┬──────────────┬────────────────────┐  │
│    │  GRN     │  L-system│ React-Diff   │  Cell state         │  │
│    │ regulation│ morphogen│ Turing field │  energy/proteins   │  │
│    └──────────┴──────────┴──────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Module map

| Module | Responsibility |
|--------|---------------|
| [lexer](src/helixlang/lexer.py) / [parser](src/helixlang/parser.py) / [semantic](src/helixlang/semantic.py) | Frontend: Token → AST → semantic checks |
| [codon_table](src/helixlang/codon_table.py) / [compiler](src/helixlang/compiler.py) | Codon table + bytecode generation |
| [bytecode](src/helixlang/bytecode.py) / [disassembler](src/helixlang/disassembler.py) | Chunk data structure + disassembler |
| [vm](src/helixlang/vm.py) / [cell](src/helixlang/cell.py) | Stack VM + cell runtime |
| [grn](src/helixlang/grn.py) | Gene regulatory network (sigmoid / Hill kinetics, half-life decay, optional telegraph-promoter intrinsic noise) |
| [stochastic](src/helixlang/stochastic.py) | Two-state (telegraph) promoter Fano factor + Gillespie SSA of bursty gene expression (Peccoud & Ycart 1995) |
| [environment](src/helixlang/environment.py) | Diffusing nutrient/O₂ fields (µm²/s Fick diffusion), Monod / Michaelis-Menten uptake, chemostat flow, CROMICS crowding factor |
| [lsystem](src/helixlang/lsystem.py) / [reaction_diffusion](src/helixlang/reaction_diffusion.py) | L-system morphology + Gray-Scott field |
| [central_dogma](src/helixlang/central_dogma.py) | Transcription / translation / coupling — codon-specific elongation rates, per-gene mRNA half-lives, `ProteinPool` chaperone maturation & QC, memoized rho-independent terminator scan |
| [evolution](src/helixlang/evolution.py) / [population](src/helixlang/population.py) | Wright-Fisher evolution + dN/dS codon-substitution models + programmable-cell population (per-cell GRN + bytecode, CROMICS diffusion, shoving/force mechanics, trace streaming), per-cell dFBA in a shared environment, colony observables & metabolic stratification |
| [crispr](src/helixlang/crispr.py) | Cas variants / sgRNA design (nearest-PAM or max-score) / Doench 2016 on-target scoring / off-target prediction |
| [epigenetics](src/helixlang/epigenetics.py) | CpG islands / methylation / histone modification |
| [metabolism](src/helixlang/metabolism.py) | FBA flux balance analysis (+ SBML / BiGG load_model), enzyme-constrained FBA (kcat capacity bound by protein pools, MOMENT/GECKO), `MetabolitePool` intracellular dynamics, dynamic FBA batch/diauxic simulation (Mahadevan 2002), full model support (`build_functional_model_full` for iML1515/iJN678-class models) |
| [protein_structure](src/helixlang/protein_structure.py) | Chou-Fasman / GOR IV secondary structure, IUPred disorder prediction |
| [protein_structure_predictor](src/helixlang/protein_structure_predictor.py) | ESM3 end-to-end protein structure prediction from sequence (no MSA); falls back to Chou-Fasman when esm/torch unavailable; `ProteinStructure3D` with coords, pLDDT, TM helix, disorder annotations |
| [protein_fitness](src/helixlang/protein_fitness.py) | Fitness oracles: BLOSUM62 conservation + ESM-2 pseudo-likelihood + variant ranking |
| [morphology_3d](src/helixlang/morphology_3d.py) | 3D population + 3D concentration-field diffusion (6/26-connectivity) + LSystem3D |
| [vectorized](src/helixlang/vectorized.py) | Across-cell numpy GRN step, stable cell sorting, snapshot iteration, optional jit |
| [omics](src/helixlang/omics/) | Spatial-omics package — expression matrices → GRN states / FBA bounds, spatial atlas, heterogeneity (ARI), omics-level calibration, expression inference (`expression_inference.py`) |
| [virtual_cell](src/helixlang/virtual_cell.py) | Whole-cell budget model — cell-cycle phasing + Cooper–Helmstetter chromosome replication, physical volume + adder size control with threshold noise, protein maturation/QC, enzyme-constrained FBA wiring, gene encoding, `fit_parameters` inverse-variance-weighted fitting |
| [apps/consortium](src/helixlang/apps/consortium.py) | Synthetic microbial consortium — quorum consensus vote + composition (ratio) control |
| [apps/ecosystem](src/helixlang/apps/ecosystem.py) | Multi-species ecosystem — Lotka-Volterra, cross-feeding, CENTURY SOM pools, C/N biogeochemistry, Q10/DAMM temperature dependence, diurnal/seasonal forcing, photoautotrophy, event-driven Scheduler, invasion-fitness evolution, `gem_to_species()` bridge, FBA-backed per-species growth (`_growth_rate_gem`), enzyme correction + density scaling, CRISPR→enzyme feedback, `#reaction` block metabolic model construction |
| [apps/morphogen_gradient](src/helixlang/apps/morphogen_gradient.py) | French-flag positional information — diffusing morphogen + cross-repression thresholds (Wolpert 1969) |
| [apps/digital_evolution](src/helixlang/apps/digital_evolution.py) | Digital organisms evolve a signal task — Wright-Fisher + Eigen error catastrophe (Avida paradigm) |
| [apps/synbio_automation](src/helixlang/apps/synbio_automation.py) | Cello-style closed-loop automation — truth table → netlist → gates → plasmid + GenBank + SBOL3 → predicted dynamics |
| [apps/dna_storage](src/helixlang/apps/dna_storage.py) | DNA-storage scenario decision tool — fountain / Reed-Solomon / Goldman codec benchmarks + per-GB cost |
| [apps/spatial_dfba](src/helixlang/apps/spatial_dfba.py) | Spatial dynamic-FBA biofilm — 1-D glucose gradient, diffusion-coupled dFBA batches, depletion fronts |
| [apps/fate_analysis](src/helixlang/apps/fate_analysis.py) | Cell-fate decision analysis — toggle-switch bistability scan + stochastic switching + critical slowing down |
| [apps/protein_evolution](src/helixlang/apps/protein_evolution.py) | ML-guided directed evolution of GB1 — ESM-2/BLOSUM oracle, top-K screening vs random baseline, Spearman alignment |
| [apps/virtual_cell_bench](src/helixlang/apps/virtual_cell_bench.py) | 4-gate whole-cell benchmark (`run_whole_cell_benchmark`) — essentiality accuracy, batch doubling-time fidelity, adder slope, colony radial density profile |
| [apps/omics_calibration](src/helixlang/apps/omics_calibration.py) | Omics-level parameter calibration — CRISPRi PerturbSeq with negative-binomial noise, VCC-style log fold-change vs WT, inverse-variance weighted `fit_parameters` |
| [apps/whole_cell_scale](src/helixlang/apps/whole_cell_scale.py) | Whole-cell scale — FASTA genome loader (RBS + bare-ORF fallback), KO→FBA gene-essentiality screening (Feist 2007 / EcoCyc) |
| [apps/whole_cell_calibration](src/helixlang/apps/whole_cell_calibration.py) | Whole-cell calibration closure — two-stage separable fit recovering adder / k_fold / enzyme-scale / maintenance from mixed observables; adder-noise robustness via population averaging (`n_cells`) |
| [apps/population_dbtl](src/helixlang/apps/population_dbtl.py) | Population DBTL data loop — Design-Build-Test-Learn cycle for colony-scale phenotype optimization |
| [annotation/](src/helixlang/annotation/) | Functional annotation package — DIAMOND/UniProt 3-tier search (ID mapping → NCBI BLASTP → UniProt sequence search), EC/KEGG mapping (`ec_mapping.py`, `kegg_mapping.py`), TF detection via HMMER domain scan + RegulonDB PWMs + heuristic fallback (`tf_detection.py`), transporter classification (`transporter.py`), sequence utilities (`sequences.py`) |
| [gem/](src/helixlang/gem/) | GEM reconstruction — bottom-up (`bottom_up.py`), top-down universal prokaryotic carve (~90 reactions, `top_down.py`), consensus merge (`consensus.py`), LP gap-filling (`gapfill.py`), biomass reaction (`biomass.py`), GRN inference (`grn_inference.py`), HelixLang bridge (`bridge.py`), SBML export (`sbml_export.py`), full model import (`full_model.py`, `sbml_import.py`), organism registry (`organism_registry.py`), validation (`validation.py`) |
| [gem/ecgem](src/helixlang/gem/ecgem.py) | Enzyme-constrained GEM builder (ECMpy 2.0 / sMOMENT-lite): kcat capacity constraints, enzyme pool budget (0.55 g protein/gDW), real molecular weights from sequences; `ECGEMBuilder.build()` → `ECGEMResult` with constrained/unconstrained growth comparison and validation |
| [gem/community](src/helixlang/gem/community.py) | Community FBA extension (OptCom multi-level): per-organism ecGEMs, dynamic metabolite exchange, cross-feeding network detection, iterative convergence; `CommunityFBAExtended.solve()` → `CommunityResult` with exchange network and mass balance |
| [kinetics/](src/helixlang/kinetics/) | Enzyme kinetics — kcat prediction (60+ EC entries, 18 organism scaling factors, `kcat_predictor.py`), Km estimation (`km_estimator.py`), `km=` DSL extension for direct Km specification in `#enzyme` blocks |
| [kinetics/sequence_predictor](src/helixlang/kinetics/sequence_predictor.py) | Sequence-based kcat/Km prediction: ESM-2 embeddings (facebook/esm2_t6_8M_UR50D via transformers) → BRENDA EC-class medians → physics fallback (Bar-Even 2011); `SequenceKcatPredictor`, `SequenceKmEstimator`, substrate charge table, catalytic residue detection |
| [apps/gem_pipeline](src/helixlang/apps/gem_pipeline.py) | Six-phase GEM reconstruction orchestrator — genome → annotation → GEM → GRN → kinetics → HelixLang integration; 3-tier UniProt annotation, consensus + FBA validation |
| [apps/full_pipeline](src/helixlang/apps/full_pipeline.py) | Full-chain custom organism pipeline (doc/26): FASTA → translation → ESM3 structure → ESM-2 kinetics → ecGEM → community FBA → dFBA simulation; `run_full_pipeline()` with `PipelineConfig`/`PipelineResult`, auto EC number inference, DNA auto-detect + translate |
| [human/](src/helixlang/human/) | Human physiology & drug simulation (doc/27–31): organ volumes, blood flows, Recon3D GEM, disease states (Gaucher/PKU/T2D/Warburg), drug molecules (SMILES/inorganic/biologic via RDKit 2026.3), PBPK, pharmacodynamics (Hill + QSP binding: mass-action/TMDD/competitive), endocrine axes (insulin-glucose/HPA/HPT), innate immune ABM (cytokines/CRP/WBC), organ crosstalk, per-disease ODE models (8 categories), genotype→CYP450, phenotype scaling, clinical labs (35+ analytes), vital signs, disease progression, DDI, recovery; `#sim kind=human` with `#person`/`#trait`/`#disease`/`#drug`/`#pd_effect`/`#qsp_binding`/`#endocrine_config`/`#immune_config` |
| [gem/full_model](src/helixlang/gem/full_model.py) | Full GEM model loading — SBML import via CobraPy (`_load_sbml`), automatic biomass reaction detection, `build_functional_model()` that returns a functional `MetabolicModel` ready for FBA |
| [gem/sbml_import](src/helixlang/gem/sbml_import.py) | CobraPy SBML loader with stderr capture (`_stderr_for_cobra`), SBML fallback for non-CobraPy environments, `BiGGModels` model caching |
| [gem/organism_registry](src/helixlang/gem/organism_registry.py) | Organism-specific GEM registry — maps organism IDs to SBML model paths, biomass reactions, and growth parameters |
| [gem/validation](src/helixlang/gem/validation.py) | GEM model validation — FBA feasibility checks, biomass flux sanity, reaction mass-balance verification |
| [apps/genome_scale](src/helixlang/apps/genome_scale.py) | Genome-scale simulation adapter — `#sim kind=genome_scale` dispatch, organism registry lookup, full-model FBA integration |
| [apps/sim_runtime](src/helixlang/sim_runtime.py) | `#config backend` adapter — classic/whole_cell/population/fba/calibration/benchmark/gem/ecosystem dispatch, GEM↔Ecosystem bridge (`_attach_gem_to_ecosystem_species()`), key→dataclass coercion, `SimResult` (`to_dict`/CSV columns), determinism via `seed=` |
| [server](src/helixlang/server.py) / [web/](src/helixlang/web/) | Flask REST API + visualization frontend |
| [cli](src/helixlang/cli.py) | Command-line entry point |

---

## 🔌 API & Web Visualization

Launch the web server:

```bash
helixlang --serve --port 5000
```

Main REST endpoints:

| Endpoint | Function |
|----------|----------|
| `POST /api/compile` | Compile source, return disassembly + AST summary |
| `POST /api/run` | Compile and run, return trace + GRN + morphology |
| `POST /api/sim/run` | Run a sim backend (classic/whole_cell/population/fba/calibration/benchmark/gem/ecosystem) from source; returns `SimResult` JSON |
| `POST /api/dna/encode` | DNA data-storage encoding (goldman / erlich) |
| `POST /api/dna/decode` | DNA data-storage decoding |
| `GET  /api/bio/codon-usage` | Codon usage frequency table (ecoli / yeast / human) |
| `GET  /api/bio/trna` | tRNA abundance table |
| `GET  /api/bio/gray-scott-presets` | 14 Pearson 1993 measured presets |
| `POST /api/central-dogma/transcribe` | DNA → mRNA transcription |
| `POST /api/central-dogma/translate` | mRNA → protein translation |
| `POST /api/central-dogma/coupled` | Coupled transcription-translation |
| `POST /api/evolution/run` | Evolution simulation |
| `POST /api/debug/*` | Bytecode debugger (breakpoints / stepping / state) |
| `POST /api/gem/reconstruct` | GEM reconstruction from genome FASTA (6-stage pipeline) |
| `POST /api/gem/simulate` | Simulate a reconstructed GEM model |
| `POST /api/gem/simulate_full` | Simulate a full genome-scale model (SBML import → FBA) |
| `POST /api/full-pipeline` | Full-chain custom organism pipeline: FASTA → ESM3 structure → ESM-2 kinetics → ecGEM → community FBA → dFBA simulation |

---

## 📚 Documentation

The full technical documentation lives in [`doc/`](doc/). Reference by reader — all files are kept in sync with the implementation (when docs and code conflict, the code prevails).

### Quick reference to `doc/`

| Document | Audience | What it covers |
|---|---|---|
| **Core language** | | |
| [`00-overview.md`](doc/00-overview.md) | Everyone | Motivation, vision, end-to-end compiler → VM → simulation pipeline |
| [`02-language-spec.md`](doc/02-language-spec.md) | Language users | **Authoritative** spec: alphabet, lexing, annotation syntax, codon table, bytecode format, runtime semantics |
| [`09-bio-instructions.md`](doc/09-bio-instructions.md) | `.helix` authors | Annotation syntax (`#gene`, `#promoter`, `#regulate`, `#field`, …) + bio module usage |
| [`08-api-reference.md`](doc/08-api-reference.md) | Python library users | Per-module reference: dataclasses, function signatures, key parameters |
| **Simulation & biology** | | |
| [`04-simulation-model.md`](doc/04-simulation-model.md) | Simulator users | GRN, L-system morphogenesis, Gray-Scott reaction-diffusion, unified tick loop |
| [`07-bio-modules.md`](doc/07-bio-modules.md) | Bio module users | Central dogma, metabolism, protein structure, CRISPR, epigenetics, evolution |
| [`12-helix-language-wiring.md`](doc/12-helix-language-wiring.md) | Language designers | `.helix` backend wiring: `#config backend`, `#media`/`#enzyme`/`#metabolite`, `sim_runtime` adapter |
| **GEM pipeline** | | |
| [`20-gem-reconstruction-pipeline.md`](doc/20-gem-reconstruction-pipeline.md) | Researchers | Six-phase genome→GEM: annotation → reconstruction → GRN → kinetics → integration |
| [`24-full-gem-import.md`](doc/24-full-gem-import.md) | Researchers | Full SBML model import: CobraPy loader, BiGG ID normalization, `#gem model=path.xml` |
| [`25-realistic-parameter-consistency-gaps.md`](doc/25-realistic-parameter-consistency-gaps.md) | Researchers | GRN→FBA closed loop, enzyme correction, density scaling, CRISPR→enzyme feedback |
| [`26-full-chain-custom-organism-pipeline.md`](doc/26-full-chain-custom-organism-pipeline.md) | Researchers | DNA → ESM3 structure → ESM-2 kinetics → ecGEM → community FBA → ecosystem: full-chain custom organism pipeline (Xenobacter alienus example) |
| [`27-human-pathology-drug-simulation.md`](doc/27-human-pathology-drug-simulation.md) | Researchers | Human physiology (Recon3D GEM) + disease states (Gaucher, PKU, T2D, Warburg) + drug molecules (SMILES) + PBPK + pharmacodynamics + long-term therapy simulation |
| [`28-virtual-patient-system.md`](doc/28-virtual-patient-system.md) | Researchers | Full virtual patient: genome → CYP450 phenotype → organ scaling → disease staging → drug DDIs → clinical labs → vital signs → PBPK/PD → post-treatment recovery → virtual patient simulation |
| [`29-virtual-patient-fixes.md`](doc/29-virtual-patient-fixes.md) | Researchers | Critical fixes: PBPK stateful engine, µM-normalized scalers, dynamic QTc/SpO₂/RR/electrolytes/lipids/coagulation, disease name propagation, full VirtualPatientResult channels |
| [`30-computational-models-major-disease-categories.md`](doc/30-computational-models-major-disease-categories.md) | Researchers | Literature survey: ODE models for 8 disease categories (CV, metabolic, cancer, autoimmune, neurological, renal, hepatic, hematological) + organ crosstalk |
| [`31-frontier-virtual-patient-design.md`](doc/31-frontier-virtual-patient-design.md) | Researchers | SOTA survey + 5-phase roadmap: endocrine axes, QSP binding, immune ABM, organ crosstalk, per-disease ODEs, validation frameworks |
| **Engineering** | | |
| [`03-compiler-design.md`](doc/03-compiler-design.md) | Compiler contributors | Compilation pipeline, AST, bytecode format, stack VM, disassembler |
| [`06-engineering-design.md`](doc/06-engineering-design.md) | Maintainers | Module interfaces, data flow, error matrix, performance budgets, CI, test pyramid |
| [`11-helixc-binary-format.md`](doc/11-helixc-binary-format.md) | Tooling users | `.helixc` binary artifact format: versioned container, round-trip tests |
| [`13-performance-report.md`](doc/13-performance-report.md) | Performance engineers | Bottleneck analysis + scaling behavior (compile / VM / GRN / reaction-diffusion) |

> Full index of all 31 documents: [`doc/`](doc/). Research-oriented docs (01, 05, 10, 14–19, 21–23, 26–31) cover frontier biology plans, whole-cell realism roadmaps, population simulation designs, GEM↔Ecosystem bridge specs, the full-chain custom organism pipeline, human pathology + drug simulation, virtual patient simulation, critical defect fixes + full-parameter coverage, computational models for major disease categories, and frontier virtual patient architecture design.

### Suggested reading order

1. **[`00-overview.md`](doc/00-overview.md)** — the big picture.
2. **[`02-language-spec.md`](doc/02-language-spec.md)** — how to write programs (codons, genes, annotations, config).
3. **[`04-simulation-model.md`](doc/04-simulation-model.md)** — what happens when a program *runs*.
4. **[`07-bio-modules.md`](doc/07-bio-modules.md)** — the biological machinery, per domain.
5. **[`08-api-reference.md`](doc/08-api-reference.md)** + **[`09-bio-instructions.md`](doc/09-bio-instructions.md)** — while you write code.
6. **[`03-compiler-design.md`](doc/03-compiler-design.md)** — if you want to extend the toolchain.
7. **[`06-engineering-design.md`](doc/06-engineering-design.md)** — before touching internals.

---

## 🧪 Testing & Quality

```bash
# Full test suite + coverage gate (80%)
pytest --cov=helixlang --cov-fail-under=80

# Lint
ruff check src tests

# Type checking
mypy
```

- **2644 test cases** (all passing, 81% coverage)
- CI matrix: Python 3.11
- Three quality gates: ruff + mypy + pytest --cov-fail-under=80
- All 55 `examples/*.helix` covered + Python API companions
- doc/26 full-chain pipeline: 95 tests across `test_protein_structure_predictor`, `test_sequence_kinetics`, `test_ecgem`, `test_community_fba`, `test_full_pipeline`

---

## Contributing

Contributions are welcomed! Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** first — it covers
the development setup, quality gates (pytest + coverage, ruff, mypy), coding conventions, the
citation rules for biological constants, and the documentation policy.

In short: fork the repo, create a branch off `main`, and open a pull request.

```bash
git clone https://github.com/SeanHank/HelixLang.git
cd HelixLang
pip install -e ".[dev,fast,web,bio,ml]"
pytest --cov=helixlang --cov-fail-under=80 && ruff check src tests && mypy
```

Before opening a PR, check the [open issues](https://github.com/SeanHank/HelixLang/issues) to
see if your idea is already being worked on, and make sure the docs are updated alongside any
behavior change.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPLv3).  

Copyright © 2026 Sean Hank.
