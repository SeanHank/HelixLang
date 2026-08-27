<div align="center">

# 🧬 HelixLang

**DNA is source code. Codons are mnemonics. The ribosome is a VM. The cell is the runtime.**

A domain-specific language where biological genetic material is the source, binary bytecode is the intermediate representation, and the execution semantics simulate biological behavior.

[Quick Start](#quick-start) ·
[5-Minute Proof](#5-minute-proof-lac-operon) ·
[Examples](#language-examples) ·
[Architecture](#architecture) ·
[API](#api--web-visualization) ·
[Documentation](#documentation) ·
[Contributing](#contributing) ·
[License](#license)

[![PyPI - Version](https://img.shields.io/pypi/v/helixlang)](https://pypi.org/project/helixlang/)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/helixlang)](https://pypi.org/project/helixlang/)

</div>

![](img.png)

---

## 5-Minute Proof: lac Operon

The lac operon is a real gene regulatory circuit in *E. coli*. Here's how it works: LacI constitutively represses the lac promoter. When lactose is absent, LacZ/LacY are off. This is a negative feedback loop — the same pattern that controls billions of genes across nature.

Here's the complete HelixLang program:

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

Run it:

```bash
pip install helixlang
helixlang examples/02_lac_operon.helix
```

Output (CSV):

```
tick,lacI_protein,lacZ_protein,p_lac_activity
0,0.000,0.000,0.500
1,0.500,0.000,0.050
2,0.500,0.000,0.050
3,0.500,0.000,0.050
...
```

**What just happened?**
1. `p_lacI` is constitutively active (strength=-0.5), so LacI protein accumulates
2. LacI represses `p_lac` (strength=-0.9), so LacZ stays near zero
3. The system reaches steady state in ~3 ticks — exactly what the real operon does

This is not a toy. The same compiler produces bytecode that runs on a real VM, and the same language specifies genome-scale metabolic models with 2,712 reactions validated against COBRApy with error < 10⁻¹³.

---

## Quick Start

### Install

```bash
# Core (compiler + VM + CLI) — zero runtime dependencies
pip install helixlang

# Full stack
pip install "helixlang[fast,web,bio,ml,human]"
```

| Extra | Capability | Dependencies |
|-------|-----------|--------------|
| `fast` | vectorized mutation / reaction-diffusion speedup | numpy |
| `web` | Flask visualization frontend | flask |
| `bio` | physical DNA codec + full GEM import | biopython, reedsolo, cobra |
| `ml` | ESM3 protein structure + ESM-2 kinetics | esm, torch |
| `human` | SMILES parsing + drug simulation | rdkit |
| `dev` | tests + coverage (source checkouts) | pytest, scipy |

### 30-Second Demo

```bash
# Run the lac operon
helixlang examples/02_lac_operon.helix

# Disassemble the bytecode
helixlang examples/02_lac_operon.helix --disassemble

# Run a genome-scale metabolic model (E. coli iML1515)
helixlang examples/53_ecoli_full_model.helix

# Launch the web visualization
helixlang --serve
```

### As a Python Library

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

## What HelixLang Does

HelixLang is a compiler, bytecode VM, and 22 quantitative simulation backends for biology. One `.helix` source file can:

1. **Compile to bytecode** — a real compiler pipeline (Lexer → Parser → AST → Compiler → Bytecode → VM) with `.helixc` binary serialization
2. **Run quantitative simulations** — FBA metabolism, whole-cell physiology, population dynamics, ecosystem ecology, human pharmacology
3. **Load real data** — genome-scale metabolic models (2,712 reactions for E. coli), drug molecules, disease parameters

### At a Glance

| Metric | Value |
|--------|-------|
| Source modules | 135 |
| Test cases | 3,169 (81% coverage) |
| Validation benchmarks | 67 (67 pass) |
| `.helix` examples | 60 |
| Documentation | 36 files, 25,000+ lines |
| Runtime dependencies | **zero** (all optional) |

---

## Language Examples

### 1. Gene Regulatory Network

```helix
#promoter name=p_tetR strength=-0.5
#promoter name=p_lacI strength=-0.5

#gene name=tetR promoter=p_tetR
ATG GCT GCT GCT TAA
#end

#gene name=lacI promoter=p_lacI
ATG GCT GCT TAA
#end

#regulate lacI  -> p_tetR strength=-0.9
#regulate tetR -> p_lacI strength=-0.9

#config ticks=20
```

### 2. CRISPR-Cas9 Gene Editing

```helix
#gene name=target_gene
ATG GCT GCT GCT GCT GCT GCT GCT GCT GCT TAA
#end

#crispr target=target_gene position=50 new_sequence="GGGG" cas=SpCas9 repair=HDR
```

### 3. Evolution Simulation

```helix
#gene name=ancestor
ATG CTG CTG CTG CTG CTG CTG CTG CTG CTG TAA
#end

#evolve target=ancestor generations=100 mutation_rate=0.01 population_size=1000
```

### 4. Human Virtual Patient

```helix
#person name=patient age=55 weight=78
#genotype CYP2D6=*4/*4
#disease name=type2_diabetes severity=0.6
#drug name=metformin dose=500 route=oral
#config backend=human ticks=168
```

### 5. DNA Data Storage

```bash
# Encode .helix source into synthesizable DNA
helixlang examples/01_hello_dna.helix --encode-dna goldman

# Decode DNA back to source
helixlang decoded.fasta --decode-dna oligos.fasta
```

---

## Simulation Backends

| Backend | What it does | Examples |
|---------|-------------|----------|
| `classic` | Bytecode VM — GRN + L-system + reaction-diffusion | All |
| `fba` | Static/dynamic FBA — biomass optimization | 10, 20, 35 |
| `whole_cell` | Cooper–Helmstetter replication, adder size control | 30, 31, 34 |
| `population` | Per-cell GRN+bytecode colonies, CROMICS diffusion | 21, 32, 37-39 |
| `ecosystem` | Multi-species Lotka-Volterra, cross-feeding | 41, 43, 50, 53-55 |
| `gem` | Genome→GEM reconstruction | 46-49 |
| `human` | PBPK + PD + pharmacogenomics + disease ODEs | Virtual patient |

Deterministic with `seed=`; same source + same seed = same result (verified with SHA256 goldens).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    .helix source program                         │
│   #gene promoter=strong                                          │
│   ATG GCT GGT TAA    # DNA triplet stream + annotation blocks    │
│   #end                                                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Lexer → Parser → AST → Compiler
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bytecode (Chunk)                                                │
│    ATG->OP_START  GCT->OP_BUILD_PROTEIN  TAA->OP_HALT           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         ┌──────────┐  ┌──────────┐  ┌──────────────┐
         │  CellVM  │  │  sim     │  │  ecosystem   │
         │ (bytecode│  │ _runtime │  │  / human /   │
         │  GRN,    │  │ (22      │  │  population  │
         │  L-sys)  │  │ backends)│  │              │
         └──────────┘  └──────────┘  └──────────────┘
```

---

## Validation

45 reproducible benchmarks validating every subsystem — all with SHA256-verified golden outputs:

| # | Benchmark | Evidence |
|---|-----------|----------|
| 01 | Codon translation (64 codons) | Functional |
| 02 | lac operon (negative feedback) | Gold-standard + Experimental |
| 03 | E. coli core FBA (μ = 0.872 h⁻¹) | Gold-standard (COBRApy err < 10⁻¹²) |
| 04 | iML1515 FBA (2,712 reactions) | Gold-standard (COBRApy err < 10⁻¹³) |
| 05 | iJN678 photoauto FBA | Gold-standard (COBRApy err < 10⁻¹⁴) |
| 06 | dFBA diauxic shift | Gold + Experimental (Enjalbert 2015) |
| 07 | Repressilator oscillation | Gold + Experimental (Elowitz 2000) |
| 08 | Population doubling time | Analytical |
| 09 | Reaction-diffusion pattern | Reference + Robustness |
| 10 | Whole-cell division time | Analytical |
| 11-45 | Parser, bytecode, CRISPR, evolution, GEM, pharmacology, ecosystem, determinism | Functional + Performance |

### Scientific Validation Metrics

| Metric | Value |
|--------|-------|
| Benchmarks passing | **67/67** |
| Published references cited | **40+** |
| Non-deterministic failures | **0** |
| Median error (quantitative benchmarks) | **~3.0%** |
| Worst-case error | **16.7%** (population doubling time) |
| SHA256 golden verification | **44/44** |

Every benchmark records: **Reference → Expected range → Helix result → Error → Reproducibility**.

```bash
# Run all benchmarks
bash validation/run_all.sh

# Verify golden outputs
python validation/goldens/verify_goldens.py
```

---

## Documentation

Full technical documentation in [`doc/`](doc/) (37 files, 25,000+ lines):

| Document | What it covers |
|----------|---------------|
| [`00-overview.md`](doc/00-overview.md) | Motivation, vision, end-to-end pipeline |
| [`02-language-spec.md`](doc/02-language-spec.md) | Authoritative spec: alphabet, lexing, bytecode |
| [`09-bio-instructions.md`](doc/09-bio-instructions.md) | Annotation syntax for `.helix` authors |
| [`08-api-reference.md`](doc/08-api-reference.md) | Python API reference |
| [`27-human-pathology-drug-simulation.md`](doc/27-human-pathology-drug-simulation.md) | Human physiology + drug simulation |
| [`34-architectural-improvement-plan.md`](doc/34-architectural-improvement-plan.md) | Architecture plan + validation suite |

---

## API & Web Visualization

```bash
helixlang --serve --port 5000
```

| Endpoint | Function |
|----------|----------|
| `POST /api/compile` | Compile source → disassembly + AST |
| `POST /api/run` | Compile and run → trace + GRN |
| `POST /api/sim/run` | Run any backend → SimResult JSON |
| `POST /api/dna/encode` | DNA data-storage encoding |
| `POST /api/gem/reconstruct` | GEM reconstruction from FASTA |
| `POST /api/full-pipeline` | DNA → ESM3 → kinetics → ecGEM → ecosystem |

---

## Testing & Quality

```bash
# Full test suite
pytest --cov=helixlang --cov-fail-under=80

# Lint
ruff check src tests

# Determinism audit
python tests/test_determinism_audit.py
```

- **3191 test cases**(all passing, 81% coverage)
- [67/67 validation benchmarks](validation/report.md) with SHA256 goldens
- CI matrix: Python 3.11
- Three quality gates: ruff + mypy + pytest

---

## Release

One-command release via `release.py`:

```bash
python release.py 2026.8.5
```

What `release.py` does:

| Step | Action |
|------|--------|
| 1 | Sync version to `pyproject.toml`, `__init__.py`, `server.py` |
| 2 | Run quality gates in parallel (ruff, mypy, pytest -n auto, validation benchmarks, examples smoke test) |
| 2b | Generate `validation/report.md` from fresh results |
| 3 | Sync metrics (test count, validation pass rate, modules, examples) to README/CONTRIBUTING |
| 4 | Build sdist + wheel |

Version format: `YYYY.M.D` or `YYYY.M.D.N` (e.g. `2026.9.1`, `2026.9.1.2`).

---

## Contributing

Contributions welcomed! Read **[CONTRIBUTING.md](CONTRIBUTING.md)** first.

```bash
git clone https://github.com/SeanHank/HelixLang.git
cd HelixLang
pip install -e ".[dev,fast,web,bio,ml]"
pytest --cov=helixlang --cov-fail-under=80 && ruff check src tests
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPLv3).

See [DISCLAIMER.md](DISCLAIMER.md) for important legal notices and limitations.

Copyright © 2026 Sean Hank.
