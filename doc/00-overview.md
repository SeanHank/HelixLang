# HelixLang Project Overview

> A domain-specific language (DSL) whose source is biological genetic material, whose intermediate representation is binary bytecode, and whose execution semantics simulate biological behavior/morphology/properties — along with its compiler.

---

## 1. Project Vision

HelixLang abstracts "biological genetic information processing" into a **programmable, compilable, executable** computational model:

- **Source code** is a DNA-style sequence (bases `A C G T`) accompanied by structured annotation blocks (promoters, genes, terminators, etc.).
- **Compiler** splits the DNA sequence into triplets (codons) and maps them through a codon table to binary bytecode (opcode + operand).
- **Virtual machine** is a "cell simulator": with a stack-based bytecode VM at its core, wired to a gene regulatory network (GRN), an L-system morphogenesis generator, and a reaction-diffusion morphogenesis field; it executes bytecode tick by tick and produces **observable biological behavior/morphology/properties**.

In one sentence: **DNA is the source, codons are the mnemonics, the ribosome is the VM, and the cell is the runtime.**

---

## 2. Design Goals

| Goal | Description |
|---|---|
| **Biological isomorphism** | Language primitives correspond one-to-one to biological entities: codons, amino acids, genes, promoters, operons, regulatory factors, proteins. |
| **Compilable** | Provides a complete Lex → Parse → AST → IR → Bytecode → VM pipeline, and the output can be disassembled. |
| **Executable** | Bytecode drives a cell simulator, outputting morphology (L-system/reaction-diffusion), behavior (move/signal), and properties (concentration/energy). |
| **Degeneracy as aliasing** | 64 codons map to ~30 logical opcodes; **the third-position degeneracy acts as an operand modifier / parameter gear**, isomorphic to biological degeneracy. |
| **Turing-complete** | Formally has jumps, loops, and procedure calls; theoretically guaranteed by Păun splicing systems + universal computation. |
| **Zero external dependencies** | The prototype is implemented with the pure Python standard library, making it easy to teach, embed, and extend. |

---

## 3. Overall Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                 HelixLang source program (.helix)                  │
│  #gene promoter=strong                                             │
│  ATG GCT GGT TAA    # DNA triplet stream + annotation blocks       │
│  #end                                                              │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Lexer (split codons by 3 bases + annotation tokens)
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                                Token stream                                │
│  CODON(ATG) CODON(GCT) CODON(GGT) STOP(TAA) ANNOT(gene, promoter=strong)   │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ Parser (group ORFs: START..STOP is one gene)
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                                AST                                 │
│  Program(                                                          │
│    Gene(name, promoter, ORF: [Codon, Codon, ...], terminator))     │
│  )                                                                 │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Semantic analysis (reading frame, pairing, symbol table)
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│          Compiler: codon table decoding (codon → opcode)           │
│   ATG→OP_START  GCT→OP_BUILD(protein, kind=structural)             │
│   GGT→OP_BUILD(membrane)    TAA→OP_HALT                            │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│       Bytecode Chunk (opcode + constant pool + line numbers)       │
│   0x10 0x01 0x02 0x00 ...   (OP_START, OP_BUILD, OP_HALT, ...)     │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Disassembler (debuggable)
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Cell Virtual Machine (tick-based simulator)                                         │
│   ┌────────────┐  ┌──────────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│   │  Ribosome  │  │       GRN        │  │  L-system  │  │  Reaction-Diffusion    │   │
│   │    (VM)    │←→│(regulatory graph)│→→│ (morphology)│  │ (Turing pattern field)│   │
│   └────────────┘  └──────────────────┘  └─────────────┘  └───────────────────────┘   │
│     bytecode exec  concentration→gene switch  growth iteration  pattern formation    │
└──────────────────────────────────────────────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Observable output                                                                   │
│   • Cell state trajectory (protein concentration, energy, position)                  │
│   • Morphology (L-system string / reaction-diffusion PNG)                            │
│   • Behavior log (move / signal / divide)                                            │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Innovations

1. **Codons as mnemonics**: The 64 codons serve directly as assembly-level mnemonics, with semantics defined by the codon table. This is absent from other ALife languages (Tierra, Avida, NetLogo), which use raw binary or hand-crafted mnemonics.
2. **Degeneracy as aliasing + modifier bits**: Synonymous codons (e.g., all `GCN` encode Ala) map to aliases of the same opcode, and the third base acts as a parameter gear (0/1/2/3) — preserving biological fact while exploiting the 64 = 4 × 16 instruction space.
3. **Variable translation table = ISA variant**: The standard table, mitochondrial table, etc. can be switched as compilation parameters, achieving "the same DNA expressed differently in different cellular environments".
4. **Multimodal execution**: A single VM simultaneously drives the GRN (regulatory logic), L-system (topological morphology), and reaction-diffusion (continuous patterning) — bringing the "genotype → phenotype" mapping down to the bytecode layer.
5. **Zero-dependency pure Python**: The prototype can run as a single file, making it easy to embed in Jupyter teaching and bioinformatics pipelines.

---

## 5. Document Navigation

| Document | Content |
|---|---|
| [00-overview.md](./00-overview.md) | This document: project vision, architecture, navigation |
| [01-references.md](./01-references.md) | Academic literature review (DNA computing / ALife / compilers) |
| [02-language-spec.md](./02-language-spec.md) | Language spec: alphabet, syntax, codon table, data types |
| [03-compiler-design.md](./03-compiler-design.md) | Compiler pipeline, AST, bytecode, VM design |
| [04-simulation-model.md](./04-simulation-model.md) | Cell simulation: GRN, L-system, reaction-diffusion, tick loop |
| [05-prototype-plan.md](./05-prototype-plan.md) | Prototype implementation plan, verification cases, future extensions |
| [06-engineering-design.md](./06-engineering-design.md) | Engineering design: module decoupling, error handling, extension points |
| [07-bio-modules.md](./07-bio-modules.md) | Bio modules in detail: central dogma, metabolic FBA, protein structure, CRISPR, epigenetics, evolution |
| [08-api-reference.md](./08-api-reference.md) | API reference: core data classes, function signatures, parameter docs |
| [09-bio-instructions.md](./09-bio-instructions.md) | Bio instruction guide: .helix annotation syntax, bio operation usage |
| [10-frontier-biology-analysis.md](./10-frontier-biology-analysis.md) | Frontier biology analysis + upgrade plan: capability mapping, SOTA benchmark, gap analysis, tiered roadmap |
| [11-helixc-binary-format.md](./11-helixc-binary-format.md) | Binary artifact design (.helixc): versioned container, compile/decompile/compare, round-trip testing |
| [12-helix-language-wiring.md](./12-helix-language-wiring.md) | Wires the simulation library into `.helix`: `#config backend`, `#media`/`#enzyme`/`#metabolite`, `sim_runtime` adapter, CLI/API, example coverage audit |
| [13-performance-report.md](./13-performance-report.md) | Measured performance report: bottleneck analysis + scaling behavior of the full pipeline |
| [14-production-upgrade.md](./14-production-upgrade.md) | Production-grade upgrade plan (historical): literature-backed replacements preserving the public API |
| [15-whole-cell-realism.md](./15-whole-cell-realism.md) | Five-phase roadmap to a physically complete virtual cell — implemented & gated |
| [16-gameplay-units-upgrade.md](./16-gameplay-units-upgrade.md) | Gameplay-unit calibration plan (superseded): toy-design → physical-biology unit system |
| [17-project-details-and-frontier-bio-applications.md](./17-project-details-and-frontier-bio-applications.md) | Project details & frontier bio-applications: compile/run walkthrough, software architecture, worked examples, problem→capability mapping, delivered designs |
| [18-programmable-cell-population-simulation.md](./18-programmable-cell-population-simulation.md) | Programmable cell-population simulation: tick model, 3D, evolution line, and the delivered population roadmap designs |
---

## 6. Positioning vs. Existing Systems

| System | Source form | Output | HelixLang difference |
|---|---|---|---|
| Tierra / Avida | Numeric instructions (0/1 strings) | Digital organism populations | HelixLang uses real codon semantics + a morphology layer |
| NetLogo | Logo-like | Multi-agent | HelixLang is compiled + cell-level semantics |
| L-system | String rewriting | Plant morphology | HelixLang emits L-system as a growth opcode |
| Grammatical Evolution | BNF + chromosome | Arbitrary programs | HelixLang internalizes the GE idea as codon→opcode |
| CPPN-NEAT | Neural network weights | Morphology/texture | HelixLang uses GRN rather than NN for development |

HelixLang's uniqueness lies in **being the first to make "codon table → bytecode opcode" the core of its compiler**, and in unifying multiple ALife output modalities (GRN + L-system + reaction-diffusion) under a single bytecode VM.

---

## 7. Project Structure

HelixLang adopts a **16-core-module** architecture in three layers: the compiler toolchain (8 modules), the simulation runtime (4 modules), and the bio function modules (4 modules). It also includes data/tool modules and a CLI entry point.

```
HelixLang/
├── doc/                              # Design docs
│   ├── 00-overview.md                # Project overview
│   ├── 01-references.md             # Literature review
│   ├── 02-language-spec.md           # Language spec
│   ├── 03-compiler-design.md         # Compiler design
│   ├── 04-simulation-model.md        # Simulation model
│   ├── 05-prototype-plan.md           # Prototype plan
│   ├── 06-engineering-design.md      # Engineering design
│   ├── 07-bio-modules.md             # Bio modules in detail
│   ├── 08-api-reference.md           # API reference
│   ├── 09-bio-instructions.md        # Bio instruction guide
│   ├── 10-frontier-biology-analysis.md  # Frontier biology analysis + upgrade plan
│   ├── 11-helixc-binary-format.md     # Binary artifact design (.helixc)
│   ├── 12-helix-language-wiring.md    # Language ↔ simulation wiring
│   ├── 13-performance-report.md       # Performance report
│   ├── 14-production-upgrade.md       # Production upgrade plan (historical)
│   ├── 15-whole-cell-realism.md       # Virtual cell roadmap
│   ├── 16-gameplay-units-upgrade.md   # Gameplay-unit calibration (superseded)
├── src/helixlang/                    # Compiler and runtime implementation
│   ├── __init__.py                   # Package exports
│   │
│   │  ===== Compiler toolchain (8 modules) =====
│   ├── codon_table.py                # (1) 64-codon → opcode mapping
│   ├── lexer.py                      # (2) DNA splitter + annotation tokens
│   ├── parser.py                     # (3) Recursive-descent parser → AST
│   ├── ast_nodes.py                  # (4) AST node definitions
│   ├── compiler.py                   # (5) AST → bytecode chunk
│   ├── bytecode.py                   # (6) Chunk / Op / constant pool
│   ├── disassembler.py               # (7) Bytecode disassembly
│   ├── vm.py                         # (8) Stack VM (ribosome)
│   │
│   │  ===== Simulation runtime =====
│   ├── cell.py                       # (9) Cell state + tick loop
│   ├── grn.py                        # (10) Gene regulatory network
│   ├── lsystem.py                    # (11) L-system morphogenesis
│   ├── reaction_diffusion.py         # (12) Gray-Scott reaction-diffusion
│   ├── population.py                 # Population dynamics (2D/3D, mechanics, environments)
│   ├── environment.py                # Nutrient/O₂/AI-2 fields, Monod uptake, CROMICS crowding
│   ├── morphology_3d.py              # 3D population + 3D diffusion + LSystem3D
│   ├── vectorized.py                 # Across-cell numpy GRN, sorting, snapshot iteration
│   ├── stochastic.py                 # Telegraph-promoter noise + Gillespie SSA
│   │
│   │  ===== Bio function modules =====
│   ├── central_dogma.py              # Central dogma: transcription + translation + degradation
│   ├── metabolism.py                 # Metabolic network FBA: flux balance analysis + dynamic FBA
│   ├── protein_structure.py          # Protein structure prediction: secondary structure + TM + disorder
│   ├── protein_fitness.py            # PLM fitness oracles (BLOSUM62, ESM-2) + variant ranking
│   ├── crispr.py                     # CRISPR-Cas gene editing model
│   ├── omics.py                      # Spatial-omics: expression matrices → GRN/FBA states, atlas, ARI
│   ├── virtual_cell.py               # Virtual-cell budget model + parameter fitting + benchmarks
│   ├── interop.py                    # SBML import + SBOL3 export/import
│   │
│   │  ===== Data / tools / entry points =====
│   ├── bio_data.py                   # Real biological data (codon frequencies, mutation rates, etc.)
│   ├── epigenetics.py                # Epigenetics: DNA methylation + histone modification
│   ├── evolution.py                  # Evolution engine: mutation + selection + drift
│   ├── dna_codec.py                  # DNA codec (storage/watermark)
│   ├── biocodec.py                   # Bio codec utilities
│   ├── type_system.py                # Type system
│   ├── semantic.py                   # Semantic analysis
│   ├── units.py                      # Physical units (min/µM/µm²/s/ATP)
│   ├── errors.py                     # Exception hierarchy
│   ├── debugger.py                   # Debugger
│   ├── server.py                     # HTTP API server
│   ├── cli.py                        # CLI entry point
│   ├── apps/                         # Application layer
│   │   ├── dna_storage.py            # DNA storage app
│   │   ├── synbio_designer.py        # Synthetic biology designer
│   │   └── synbio_automation.py      # Design automation: truth table → DNA → SBOL3
│   └── web/                          # Web frontend
│       ├── index.html
│       └── labs.html
├── examples/                         # Example programs (.helix)
│   ├── 01_hello_dna.helix            # Minimal smoke test
│   ├── 02_lac_operon.helix           # lac operon GRN
│   ├── 03_plant_growth.helix         # L-system plant growth
│   ├── 04_turing_pattern.helix       # Reaction-diffusion Turing pattern
│   ├── 05_table_switch.helix         # Switchable translation table
│   ├── 06_crispr_edit.helix          # CRISPR gene editing
│   ├── 07_evolution.helix            # Evolution simulation
│   └── 08_epigenetics.helix          # Epigenetic regulation
└── tests/                            # Verification tests (~30 test files)
    ├── test_codon_table.py
    ├── test_lexer.py
    ├── test_parser.py
    ├── test_compiler.py
    ├── test_vm.py
    ├── test_cell.py
    ├── test_grn.py
    ├── test_central_dogma.py
    ├── test_metabolism.py
    ├── test_protein_structure.py
    ├── test_crispr.py
    ├── test_epigenetics.py
    ├── test_evolution.py
    └── ...
```
