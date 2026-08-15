# 🧬 HelixLang

_"We are moving from reading the genetic code to writing it." — J. Craig Venter_

**DNA is source code. Codons are mnemonics. The ribosome is a VM. The cell is the runtime.**

A domain-specific language where biological genetic material is the source, binary bytecode is the intermediate representation, and the execution semantics simulate biological behavior and morphology.

| | |
|---|---|
| 🐍 Requires | Python ≥ 3.11 |
| 📦 Runtime deps | **none** (stdlib only) |
| 🧪 Tested on | Python 3.11 |
| 📜 License | AGPLv3 |

---

## Why HelixLang?

Traditional biological modeling tools treat DNA as **data** — a passive string to be analyzed. HelixLang treats DNA as **code**: every codon is an instruction, every gene is a function, every cell is a runtime.

```helix
#gene name=hello
ATG GCT GGT TAA          # START -> BUILD_PROTEIN -> BUILD_MEMBRANE -> HALT
#end
```

Feed that sequence to the HelixLang compiler and you get a real bytecode program you can run, debug, and disassemble — just like Python or Java bytecode.

### Highlights

- 🧬 **Biologically isomorphic** — language primitives map one-to-one to biological entities: codons, amino acids, genes, promoters, operons, regulators, proteins.
- ⚙️ **Full compiler pipeline** — Lexer → Parser → AST → Semantic → Compiler → Bytecode → VM, with disassembly and debugging support.
- 🎲 **Degeneracy as aliasing** — 64 codons map to ~30 opcodes; the third wobble position acts as an operand modifier, mirroring real biological degeneracy.
- 🌱 **Executable life** — bytecode drives a cell simulator that emits morphology (L-system / reaction-diffusion), behavior (move / signal), and state (concentration / energy).
- 🔬 **Real biological data** — mutation rates, transition:transversion ratios, codon usage tables, tRNA abundances, and CAI are sourced from published measurements (Lee 2012, Drake 1991, Ikemura 1985, Dong 1996, Sharp & Li 1987).
- 💾 **DNA data storage** — built-in Goldman 2013 rotating-key encoding (true base-3 Huffman, ~5.05 trits/byte) and Erlich-Zielinski 2017 DNA Fountain code, encoding arbitrary byte streams into synthesizable DNA.
- 🛠️ **No hard dependencies** — the core compiler/VM uses only the Python standard library; numpy / biopython / flask are optional enhancements.

---

## Installation

```bash
# Core (compiler + VM + CLI)
pip install helixlang

# Recommended: install all optional extras
pip install "helixlang[dev,fast,web,bio]"
```

| Extra | Capability | Dependencies |
|-------|-----------|--------------|
| `dev` | tests + coverage | pytest, pytest-cov |
| `fast` | vectorized mutation / reaction-diffusion speedup | numpy |
| `web` | Flask visualization frontend | flask |
| `bio` | physical DNA codec + IUPAC validation | biopython, reedsolo |

The core installs with **zero runtime dependencies** — only the Python standard library.

---

## Quick Start

### Run an example

```bash
# Run a simulation
helixlang examples/01_hello_dna.helix

# Disassemble the bytecode
helixlang examples/01_hello_dna.helix --disassemble

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

## Language Examples

### 1. The lac Operon — Gene Regulatory Network (GRN)

`p_lacI` constitutively expresses `lacI` → represses `p_lac` → `lacZ`/`lacY` shut off. A real negative-feedback loop:

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

---

## Architecture

```
                    HelixLang source program (.helix)
   #gene promoter=strong
   ATG GCT GGT TAA    # DNA triplet stream + annotation blocks
   #end
                            │  Lexer (split codons by 3 bases + annotation tokens)
                            ▼
   Token -> Parser (ORF grouping: START..STOP = one gene) -> AST
                            │  Semantic analysis (reading frame / pairing / symbol table)
                            ▼
   Compiler: codon-table decoding
     ATG->OP_START  GCT->OP_BUILD_PROTEIN  GGT->OP_BUILD_MEMBRANE
     TAA->OP_HALT   GTA->OP_MOVE(arg=South)  ...
                            │  Chunk (bytecode + constant pool + line table)
                            ▼
   CellVM: stack-based bytecode virtual machine = "the ribosome"
     ┌──────────┬──────────┬──────────────┬────────────────────┐
     │  GRN     │  L-system│ React-Diff   │  Cell state         │
     │ regulation│ morphogen│ Turing field │  energy/proteins   │
     └──────────┴──────────┴──────────────┴────────────────────┘
```

### Module map

| Module | Responsibility |
|--------|---------------|
| `lexer` / `parser` / `semantic` | Frontend: Token → AST → semantic checks |
| `codon_table` / `compiler` | Codon table + bytecode generation |
| `bytecode` / `disassembler` | Chunk data structure + disassembler |
| `vm` / `cell` | Stack VM + cell runtime |
| `grn` | Gene regulatory network (sigmoid / Hill kinetics, half-life decay) |
| `stochastic` | Two-state (telegraph) promoter noise + Gillespie SSA |
| `environment` | Diffusing nutrient/O₂ fields, Monod uptake, chemostat flow, CROMICS crowding |
| `lsystem` / `reaction_diffusion` | L-system morphology + Gray-Scott field |
| `central_dogma` | Transcription / translation / coupling — codon-specific elongation rates, per-gene mRNA half-lives |
| `evolution` / `population` | Wright-Fisher evolution + dN/dS codon-substitution models + cell population |
| `morphology_3d` | 3D population + 3D diffusion + LSystem3D |
| `vectorized` | Across-cell numpy GRN, stable sorting, snapshot iteration |
| `crispr` | Cas variants / sgRNA design (nearest-PAM or max-score) / Doench 2016 on-target scoring / off-target prediction |
| `epigenetics` | CpG islands / methylation / histone modification |
| `metabolism` | FBA flux balance analysis (+ SBML / BiGG `load_model`) |
| `protein_structure` | Chou-Fasman / GOR IV secondary structure, IUPred disorder prediction |
| `protein_fitness` | Fitness oracles: BLOSUM62 + ESM-2 + variant ranking |
| `omics` | Expression matrices → GRN/FBA states, spatial atlas, heterogeneity (ARI) |
| `virtual_cell` | Virtual-cell budget model, gene encoding, parameter fitting, benchmarks |
| `interop` | SBML L3V1 import + SBOL3 export/import round-trip |
| `dna_codec` | Goldman / Erlich DNA data-storage codec |
| `bio_data` | Real biological datasets (codon tables / tRNA / CAI / Gray-Scott presets) |
| `type_system` | Type checker + symbol table |
| `debugger` | Bytecode-level debugger (breakpoints / stepping / state inspection) |
| `server` / `web` | Flask REST API + visualization frontend |
| `cli` | Command-line entry point |

---

## CLI

```text
usage: helixlang <source.helix> [options]

  --table standard|mito_vertebrate|ciliate   translation table
  --disassemble                              print bytecode and exit
  --debug                                    trace VM execution
  --csv                                      emit CSV trace
  --png PREFIX                               render morphology/field to PPM
  --ticks N                                  override #config ticks
  --serve [--port 5000] [--host 127.0.0.1]   launch web visualization
  --encode-dna goldman|erlich                encode source -> DNA FASTA
  --decode-dna FILE                          decode DNA -> source
  --pcr-cycles N                             simulate PCR error injection
```

---

## IDE Plugin (PyCharm)

Write, inspect, and debug `.helix` programs inside **PyCharm 2022.2+** (Community or Professional)
with the sibling repository [**HelixLang-LSP-Plugin**](https://github.com/SeanHank/HelixLang-LSP-Plugin):
live diagnostics, hover docs, completion, navigation, semantic highlighting, inlay hints, a
bytecode disassembler, and a line debugger — all over Language Server Protocol.

```bash
pip install "helixlang[lsp]"
```

Then install the plugin zip from that repo's [Releases](https://github.com/SeanHank/HelixLang-LSP-Plugin/releases)
via **Settings → Plugins → ⚙ → Install Plugin from Disk…**.

---

## Documentation

The full technical documentation lives in the [repository `doc/` folder](https://github.com/SeanHank/HelixLang/tree/main/doc). Reference by reader — all files are kept in sync with the implementation (when docs and code conflict, the code prevails).

| Document | Audience | What it covers |
|---|---|---|
| [00-overview.md](https://github.com/SeanHank/HelixLang/blob/main/doc/00-overview.md) | Everyone | The DSL's motivation, vision, and end-to-end compiler → VM → simulation pipeline |
| [02-language-spec.md](https://github.com/SeanHank/HelixLang/blob/main/doc/02-language-spec.md) | Language users | The **authoritative** spec: alphabet, lexing, annotation syntax, codon table, bytecode format, runtime semantics, type system |
| [08-api-reference.md](https://github.com/SeanHank/HelixLang/blob/main/doc/08-api-reference.md) | Python library users | Per-module reference: dataclasses, function signatures, key parameters |
| [09-bio-instructions.md](https://github.com/SeanHank/HelixLang/blob/main/doc/09-bio-instructions.md) | `.helix` authors | The annotation syntax (`#gene`, `#promoter`, `#regulate`, `#field`, …) + how to call the bio modules |
| [07-bio-modules.md](https://github.com/SeanHank/HelixLang/blob/main/doc/07-bio-modules.md) | Bio module users | Deep dive on the biological modules — central dogma, metabolism, protein structure, CRISPR, epigenetics, evolution |
| [04-simulation-model.md](https://github.com/SeanHank/HelixLang/blob/main/doc/04-simulation-model.md) | Simulator users | The "cell simulator" layer: GRN, L-system morphogenesis, Gray-Scott reaction-diffusion, and the unified tick loop |
| [03-compiler-design.md](https://github.com/SeanHank/HelixLang/blob/main/doc/03-compiler-design.md) | Compiler contributors | The compilation pipeline, AST, bytecode format, stack VM, disassembler, and implementation strategy |
| [06-engineering-design.md](https://github.com/SeanHank/HelixLang/blob/main/doc/06-engineering-design.md) | Maintainers | Implementable contracts: module interfaces, data flow, error matrix, performance budgets, CI, test pyramid, invariants |
| [01-references.md](https://github.com/SeanHank/HelixLang/blob/main/doc/01-references.md) | Researchers | The academic literature underpinning the design — DNA computing, codon-binary mapping, information theory, formal grammars, artificial life, DSL compilers (with DOI/arXiv) |
| [11-helixc-binary-format.md](https://github.com/SeanHank/HelixLang/blob/main/doc/11-helixc-binary-format.md) | Compiler / tooling users | The `.helixc` binary artifact format: versioned container, write / read-run / debug, disassemble, round-trip tests |
| [15-whole-cell-realism.md](https://github.com/SeanHank/HelixLang/blob/main/doc/15-whole-cell-realism.md) | Researchers | Five-phase roadmap to a physically complete virtual cell — **implemented & gated**: design + landing modules + tests + per-phase implementation status |
| [16-gameplay-units-upgrade.md](https://github.com/SeanHank/HelixLang/blob/main/doc/16-gameplay-units-upgrade.md) | Historical | Superseded plan to calibrate gameplay units into physically grounded, literature-cited targets — kept for provenance |
| [17-project-details-and-frontier-bio-applications.md](https://github.com/SeanHank/HelixLang/blob/main/doc/17-project-details-and-frontier-bio-applications.md) | Researchers | Project details & frontier bio-applications: compile/run walkthrough, software architecture, worked examples (31–39), problem→capability mapping, delivered solution designs |
| [18-programmable-cell-population-simulation.md](https://github.com/SeanHank/HelixLang/blob/main/doc/18-programmable-cell-population-simulation.md) | Researchers | Programmable cell-population simulation: the tick model, 3D lattice, evolution line, and the delivered population-roadmap designs |

### Suggested reading order

1. **[00-overview.md](https://github.com/SeanHank/HelixLang/blob/main/doc/00-overview.md)** — the big picture.
2. **[02-language-spec.md](https://github.com/SeanHank/HelixLang/blob/main/doc/02-language-spec.md)** — how to write programs (codons, genes, annotations, config).
3. **[04-simulation-model.md](https://github.com/SeanHank/HelixLang/blob/main/doc/04-simulation-model.md)** — what happens when a program *runs*.
4. **[07-bio-modules.md](https://github.com/SeanHank/HelixLang/blob/main/doc/07-bio-modules.md)** — the biological machinery, per domain.
5. **[08-api-reference.md](https://github.com/SeanHank/HelixLang/blob/main/doc/08-api-reference.md)** + **[09-bio-instructions.md](https://github.com/SeanHank/HelixLang/blob/main/doc/09-bio-instructions.md)** — while you write code.
6. **[03-compiler-design.md](https://github.com/SeanHank/HelixLang/blob/main/doc/03-compiler-design.md)** — if you want to extend the toolchain.
7. **[06-engineering-design.md](https://github.com/SeanHank/HelixLang/blob/main/doc/06-engineering-design.md)** — before touching internals.

---

## Testing & Quality

```bash
# Full test suite + coverage gate (80%)
pytest --cov=helixlang --cov-fail-under=80

# Lint
ruff check src tests

# Type checking
mypy
```

- **2134 test cases** (2134 passing, 89% coverage)
- CI matrix: Python 3.11
- Three quality gates: ruff + mypy + pytest --cov-fail-under=80
- All 39 `examples/*.helix` covered + Python API companions

---

## Contributing

Contributions are welcome! Please read [**CONTRIBUTING.md**](https://github.com/SeanHank/HelixLang/blob/main/CONTRIBUTING.md)
first — it covers the development setup, quality gates (pytest + coverage, ruff, mypy), coding
conventions, the citation rules for biological constants, and the documentation policy.

```bash
git clone https://github.com/SeanHank/HelixLang.git
cd HelixLang
pip install -e ".[dev,fast,web,bio]"
pytest --cov=helixlang --cov-fail-under=80 && ruff check src tests && mypy
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPLv3).

Copyright © 2026 Sean Hank.
