# HelixLang Engineering Design Document

> This document refines the design documents (`00`–`05`) to an **implementable, verifiable** engineering level: precise module interface contracts, data flow and timing, bytecode encoding conventions, error handling matrix, performance budgets, build and release, CI, code conventions, test pyramid, observability, and engineering invariants.
>
> Everything here is checked item by item against the current implementation in `src/helixlang/`; if the implementation and the docs conflict, the implementation prevails and the discrepancy is recorded in the §17 rollout steps.

---

## 0. Document Purpose and Audience

| Item | Value |
|---|---|
| Document level | Engineering design (Detailed Design / L4) |
| Upstream documents | `00-overview.md` through `05-prototype-plan.md` |
| Downstream artifacts | `src/helixlang/*.py`, `tests/*.py`, `examples/*.helix` |
| Audience | compiler implementers, bio-simulation implementers, test engineers, CI maintainers |
| Acceptance criteria | any engineer can independently reproduce the environment, locate modules, and write spec-compliant code and tests following this document |

---

## 1. Runtime Environment and Dependency Management

### 1.1 Python Interpreter

| Item | Value |
|---|---|
| Interpreter absolute path | `/opt/anaconda3/envs/helix/bin/python` |
| Version | 3.11.15 |
| Virtual environment | conda env `helix` |
| Key dependency features | `match/case` (3.10+), `@dataclass(slots=True)` (3.10+), `Self` type hints (3.11+), `tomllib` stdlib (3.11+) |

> **Mandatory constraint**: all command-line examples, CI scripts, and IDE Run Configurations must use the absolute path above, to avoid accidentally using the system Python. `pyproject.toml` declares `requires-python = ">=3.11"`.

### 1.2 Reproducing the conda Environment

Create the environment from scratch (first rollout or a new machine):

```bash
# 1. create conda env
conda create -y -n helix python=3.11.15
conda activate helix

# 2. verify
/opt/anaconda3/envs/helix/bin/python --version
# expected output: Python 3.11.15
```

Reproduce the environment (when `requirements-dev.txt` already exists):

```bash
/opt/anaconda3/envs/helix/bin/python -m pip install \
  -r requirements-dev.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn

# install this package in development mode
/opt/anaconda3/envs/helix/bin/python -m pip install -e . \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 1.3 pip Mirror Configuration

A [`pip.conf`](file:///Users/admin/PycharmProjects/HelixLang/pip.conf) is already in place in the project root:

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
```

**Precedence of effective paths** (pip load order):

1. `PIP_INDEX_URL` / `PIP_TRUSTED_HOST` environment variables (highest)
2. `--index-url` / `--trusted-host` command-line arguments
3. project-root `pip.conf` (only effective when pip is run from the root directory)
4. `~/.config/pip/pip.conf` (user-level, macOS)
5. `/Library/Application Support/pip/pip.conf` (system-level)

**Recommended practice**: CI and local shells should always explicitly pass `-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn`, to avoid depending on the `pip.conf` load path.

Equivalent environment variables (write into `~/.zshrc` or CI secrets):

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export PIP_DEFAULT_TIMEOUT=60
```

### 1.4 Dependency Matrix and Locking

| Dependency | Version constraint | Purpose | Category |
|---|---|---|---|
| `pytest` | `>=7.4` | unit/end-to-end tests | dev |
| `pytest-cov` | `>=4.1` | coverage statistics | dev |
| `numpy` | `>=1.26` | vectorized reaction-diffusion acceleration (not enabled) | optional `[fast]` |
| `matplotlib` | `>=3.8` | PNG output (currently replaced by pure-Python PPM) | optional `[viz]` |

**Zero runtime dependencies**: the runtime uses only the standard library (`dataclasses` / `enum` / `array` / `math` / `pathlib` / `argparse` / `random` / `sys`).

Dependency declarations in [`pyproject.toml`](file:///Users/admin/PycharmProjects/HelixLang/pyproject.toml):

```toml
[project]
dependencies = []                       # zero runtime dependencies

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1"]
fast = ["numpy>=1.26"]
viz = ["matplotlib>=3.8"]
```

**Dependency locking strategy** (recommended, optional during prototype stage):

```bash
# generate the lock file
/opt/anaconda3/envs/helix/bin/python -m pip install pip-tools
/opt/anaconda3/envs/helix/bin/python -m piptools compile \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --output-file requirements-dev.txt \
  pyproject.toml --extra dev --extra fast --extra viz
```

### 1.5 IDE / Toolchain Configuration

| Tool | Configuration |
|---|---|
| PyCharm | Project SDK = `/opt/anaconda3/envs/helix/bin/python`; Package Manager = pip |
| VS Code | `python.defaultInterpreterPath = "/opt/anaconda3/envs/helix/bin/python"` |
| pytest | `python_files = ["test_*.py"]`, `testpaths = ["tests"]`, `addopts = "-ra --strict-markers"` |
| `.python-version` | content fixed to `3.11` (pyenv compatible) |

### 1.6 Project Skeleton

```
HelixLang/
├── pyproject.toml                # PEP 621 metadata + tool config
├── pip.conf                      # tsinghua mirror config
├── .python-version               # 3.11
├── requirements-dev.txt          # (optional) locked dev dependencies
├── doc/
│   ├── 00-overview.md
│   ├── 01-references.md
│   ├── 02-language-spec.md
│   ├── 03-compiler-design.md
│   ├── 04-simulation-model.md
│   ├── 05-prototype-plan.md
│   ├── 06-engineering-design.md  # this document
│   ├── 07-bio-modules.md
│   ├── 08-api-reference.md
│   ├── 09-bio-instructions.md
│   ├── 10-frontier-biology-analysis.md
│   ├── 11-helixc-binary-format.md
│   ├── 12-helix-language-wiring.md
│   ├── 13-performance-report.md
│   ├── 14-production-upgrade.md
│   ├── 15-whole-cell-realism.md
│   ├── 16-gameplay-units-upgrade.md
├── src/
│   └── helixlang/
│       ├── __init__.py           # public API exports
│       ├── __main__.py           # python -m helixlang entry point
│       ├── errors.py             # exception hierarchy
│       ├── codon_table.py        # 64-codon → opcode mapping + three translation tables
│       ├── lexer.py              # dual-mode scanner
│       ├── ast_nodes.py          # AST dataclasses
│       ├── parser.py             # recursive-descent Parser
│       ├── semantic.py           # semantic analysis
│       ├── bytecode.py           # Chunk container
│       ├── compiler.py           # AST → Chunk
│       ├── disassembler.py       # Chunk → human-readable
│       ├── grn.py                # gene regulatory network
│       ├── lsystem.py            # L-system + turtle
│       ├── reaction_diffusion.py # Gray-Scott
│       ├── cell.py               # Cell state
│       ├── vm.py                 # CellVM
│       └── cli.py                # command line
├── examples/
│   ├── 01_hello_dna.helix
│   ├── 02_lac_operon.helix
│   ├── 03_plant_growth.helix
│   ├── 04_turing_pattern.helix
│   └── 05_table_switch.helix
└── tests/
    ├── conftest.py
    ├── test_lexer.py
    ├── test_parser.py
    ├── test_codon_table.py
    ├── test_compiler.py
    ├── test_vm.py
    ├── test_grn.py
    ├── test_lsystem.py
    ├── test_reaction_diffusion.py
    └── test_end_to_end.py
```

---

## 2. Compilation Pipeline Overview

### 2.1 Stage Breakdown

```
.helix source
   │
   ▼
[Lexer]      ── Token stream ──▶ [Parser]  ── AST ──▶  [SemanticAnalyzer]
   (lexer.py)                  (parser.py)            (semantic.py)
                                                          │
                                                          ▼  Program (validated)
                                                      [Compiler]
                                                      (compiler.py)
                                                          │
                                                          ▼  Chunk
                                                   [CellVM.run]
                                                   (vm.py)
                                                          │
                                                          ▼
                                          trace / CSV / PPM / stdout
```

### 2.2 Stage Contracts

| Stage | Input | Output | Failure exception | Side effects |
|---|---|---|---|---|
| Lexer | `str` (source code) | `Iterator[Token]` | `LexError` | none |
| Parser | `list[Token]` + `stop_codons` | `Program` | `ParseError` | none |
| SemanticAnalyzer | `Program` | `None` (in-place validation + `warnings`) | `SemanticError` / `RegulationError` | none |
| Compiler | `Program` + `table` | `Chunk` | `CompileError` | none |
| CellVM | `Chunk` + `Program` | `list[dict]` (trace) | `RuntimeHelixError` (reserved) | mutates Cell/GRN/Field state |
| hxbc codec (`hxbc.py`) | `Program` (+ optional `Chunk`, `source`) | `.helixc` bytes / `LoadedArtifact` | `BinaryFormatError` / `BinaryVersionError` | none (pure codec; `.helixc` input never executes on load) |
| CLI | `argv` | process exit code | catches all `HelixError` subclasses | writes files (PPM/`.helixc`/`.helix`), writes stdout/stderr |

### 2.3 Call Sequence ([`cli.py:main`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/cli.py))

```python
table = get_table(args.table)                                  # 1. select translation table
stop_codons = _stop_codons_from_table(table)                   # 2. derive stop-codon set
tokens = list(Lexer(src).tokens())                             # 3. lexical analysis
program = Parser(tokens, stop_codons=stop_codons).parse()      # 4. syntactic analysis
SemanticAnalyzer(program).check()                              # 5. semantic analysis
chunk = Compiler(table).compile(program)                       # 6. compilation
vm = CellVM(chunk, program); trace = vm.run(program.config.ticks)  # 7. execution
```

**Key design**: `stop_codons` is derived from the translation table (the codons mapping to `OP_HALT`), ensuring that the Parser's ORF recognition and the Compiler's opcode mapping use the same table, avoiding a mismatch between ORFs and opcodes under `--table=mito_vertebrate`.

---

## 3. Module Interface Contracts

> Signatures match the current implementation in [`src/helixlang/`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang) exactly. Each module lists: public symbols, signatures, invariants, and typical exceptions.

### 3.1 [`errors.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/errors.py)

```python
class HelixError(Exception):
    def __init__(self, msg: str, *, line: int = 0, col: int = 0, codon_index: int = -1): ...

class LexError(HelixError): ...          # lexical error
class ParseError(HelixError): ...        # syntax error
class SemanticError(HelixError): ...     # semantic error
class CompileError(HelixError): ...      # compile-time error
class RegulationError(HelixError): ...   # regulatory-graph error
class RuntimeHelixError(HelixError): ... # runtime error (avoids the built-in RuntimeError)
```

**Invariants**:
- the `__str__` of every `HelixError` subclass is `[<ClassName> @ line N [codon #M]] msg`
- `line == 0` means the position is unknown (displayed as `<unknown>`)
- `codon_index < 0` means a non-codon-level error (the codon segment is not displayed)

### 3.2 [`codon_table.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/codon_table.py)

```python
class Op(IntEnum): ...                    # 31 opcodes (see §4.1)

OP_OPERAND_BYTES: dict[Op, int]           # operand byte count of each instruction
WOBBLE_BITS: dict[str, int]               # {"A":0,"C":1,"G":2,"T":3}
STANDARD_TABLE: dict[str, Op]             # 64 codons → Op (NCBI table 1)
MITO_VERTEBRATE_TABLE: dict[str, Op]      # mitochondrial table (NCBI table 2)
CILIATE_TABLE: dict[str, Op]              # ciliate table (NCBI table 6)
TABLES: dict[str, dict[str, Op]]          # name → table

def get_table(name: str) -> dict[str, Op]  # unknown table raises HelixError
def wobble(codon: str) -> int              # third-base wobble position 0..3
```

**Module-level self-check assertions** (run at import time):
- `len(STANDARD_TABLE) == 64`
- all 64 `{A,C,G,T}^3` triplets exist in `STANDARD_TABLE`

### 3.3 [`lexer.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/lexer.py)

```python
@dataclass(slots=True)
class Token:
    kind: str        # CODON | ANNOT_START | ANNOT_END | FIELD | ARROW | NEWLINE | EOF
    value: str
    line: int
    col: int
    codon_index: int = -1   # CODON only

class Lexer:
    BASES = set("ACGTacgt")
    def __init__(self, source: str): ...
    def tokens(self) -> Iterator[Token]: ...
```

**Lexical rules**:
- DNA bases are case-insensitive; consecutive ACGT is split into CODONs in groups of 3, and a non-multiple of 3 raises `LexError`
- lines starting with `#`: `#ident` is an annotation; `#` followed by whitespace/`#`/newline/EOF is treated as a line comment
- `#end` (case-insensitive) → `ANNOT_END`
- annotation names are lowercased (`#GENE` → `ANNOT_START value="gene"`)
- fields: `ident=value` → `FIELD`; `ident->ident` → `ARROW`; bare `ident` → `FIELD` with an empty value
- string values are wrapped in double quotes; the quotes are preserved in the value

### 3.4 [`ast_nodes.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/ast_nodes.py)

```python
@dataclass(slots=True) class Codon:       seq: str; index: int; line: int
@dataclass(slots=True) class Promoter:    name: str; strength: float; fields: dict[str,str]
@dataclass(slots=True) class Gene:        name: str; promoter: str|None; codons: list[Codon]; orf: list[Codon]; fields: dict[str,str]
@dataclass(slots=True) class Regulation:  source: str; target: str; strength: float
@dataclass(slots=True) class LSystemDecl: name: str; axiom: str; rules: dict[int, dict[str,str]]; angle: float; step: float = 1.0
@dataclass(slots=True) class FieldDecl:   size: int; F: float; k: float; Du: float; Dv: float
@dataclass(slots=True) class Config:      ticks: int = 100; output: list[str]; table: str = "standard"; ops_per_tick: int = 64; react_steps: int = 1
@dataclass(slots=True) class Program:     genes: list[Gene]; promoters: list[Promoter]; regulations: list[Regulation]; lsystems: dict[str, LSystemDecl]; field_decl: FieldDecl|None; config: Config
```

**Field naming conventions**:
- `Program.field_decl` (not `field`, to avoid clashing with the built-in)
- `LSystemDecl.rules` is a nested dict: `{k: {symbol: production}}`, where k is the ruleset number (VM defaults to 0)

### 3.5 [`parser.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/parser.py)

```python
class Parser:
    def __init__(self, tokens: list[Token], stop_codons: set[str] | None = None): ...
    def parse(self) -> Program: ...
```

**Key algorithms**:

- **NEWLINE filtering**: at construction time, `self.toks = [t for t in tokens if t.kind != "NEWLINE"]`
- **ORF recognition** ([`_extract_orf`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/parser.py)): from the first `ATG` to the first codon in `stop_codons` (inclusive); a missing START raises `ParseError`, and a missing STOP raises `ParseError`
- **Implicit termination**: an annotation block ends implicitly upon `ANNOT_START` / `CODON` / `EOF` (when `allow_no_end=True`)
- **Anonymous genes**: bare CODON streams are wrapped as `__anon_<n>`
- **L-system rule parsing**: `rules=0:F->F[+F]F[-F]F,1:F->FF` → `{0: {"F": "F[+F]F[-F]F"}, 1: {"F": "FF"}}`

### 3.6 [`semantic.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/semantic.py)

```python
class SemanticAnalyzer:
    def __init__(self, program: Program): ...
    def check(self) -> None                # raises SemanticError / RegulationError
    warnings: list[str]                    # regulation-cycle warnings
```

**Checks**:
1. `_collect_symbols`: promoter/gene names are unique
2. `_check_references`: the source/target of `#regulate` must exist in the symbol table; `#gene promoter=` must reference an existing promoter
3. `_check_orfs`: every gene's ORF is non-empty, starts with `ATG`, and ends with `TAA/TAG/TGA`
4. `_check_regulation_cycles`: DFS detects regulation cycles and **appends a warning** (does not error)
5. `_check_config`: `ticks > 0`, `ops_per_tick > 0`, `react_steps > 0`

### 3.7 [`bytecode.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/bytecode.py)

```python
@dataclass(slots=True)
class Chunk:
    code: bytearray
    constants: list[Any]
    lines: list[int]                       # source line number per byte
    codon_indices: list[int]               # codon index per byte
    gene_offsets: dict[str, int]           # gene name → starting byte offset

    def emit(self, op: Op, *operands: int, line: int = 0, codon_index: int = -1) -> int
    def emit_u16(self, op: Op, value: int, line: int = 0, codon_index: int = -1) -> int
    def add_constant(self, value: Any) -> int     # deduplicates, returns the index
    def read_u8(self, ip: int) -> tuple[int, int]
    def read_u16(self, ip: int) -> tuple[int, int]
    def __len__(self) -> int
```

**Invariants**:
- `len(code) == len(lines) == len(codon_indices)` (position info is attached to every byte)
- operands are big-endian encoded (`emit_u16` high byte first)
- `add_constant` deduplicates with `==` (note the degenerate `nan != nan` case)

### 3.8 [`compiler.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/compiler.py)

```python
class Compiler:
    def __init__(self, table: dict[str, Op] = STANDARD_TABLE): ...
    def compile(self, program: Program) -> Chunk: ...
```

**Two-pass compilation**:

1. **First pass** (emit ORF): for each gene, record `gene_offsets[name] = len(code)`, look up `op` per codon, and choose the emit form by `OP_OPERAND_BYTES[op]`:
   - `OP_CALL_GENE`: emit placeholder `0, 0`, record `(ip, wobble_arg)` into `_call_sites`
   - 1-byte-operand instructions: operand = `wobble(codon)`
   - 0-byte-operand instructions: no operand
   - if the end is not `OP_HALT`, append one automatically
2. **Second pass** (backfill CALL_GENE): use `wobble_arg % len(genes)` to select the target gene name and write its `gene_offsets` back into the placeholder bytes (big-endian)
3. **Constant pool**: L-system axiom/rules/angle/step, gene-name list

**`_ends_with_halt` implementation**: scan the whole chunk to find the opcode byte of the last instruction and check whether it is `OP_HALT`.

### 3.9 [`disassembler.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/disassembler.py)

```python
def disassemble(chunk: Chunk, name: str = "HelixLang Chunk") -> str: ...
```

**Output format**:

```
=== <name> ===
--- Gene Offsets ---
  gene_name             @ 0x0000 (0)
--- Code ---
  0000  OP_START                            ; codon #0 line 1
  0001  OP_BUILD_PROTEIN    00              ; codon #1 line 1
--- Constants ---
  [0] ('lsystem_axiom', 'default', 'F')
```

Unknown opcode bytes print `<unknown 0xNN>` and advance only 1 byte (conservative recovery).

### 3.10 [`grn.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/grn.py)

```python
def sigmoid(x: float) -> float              # numerically stable

@dataclass(slots=True) class GeneNode: name, threshold, level=0.0
@dataclass(slots=True) class Edge:          source, target, weight

class GRN:
    DECAY = 0.7                             # protein degradation coefficient
    def add_gene(self, name: str, threshold: float, initial_level: float = 0.0) -> None
    def add_edge(self, source: str, target: str, weight: float) -> None
    def set_level(self, name: str, level: float) -> None   # clamp [0,1]
    def step(self) -> list[str]              # returns gene names with level > 0.5
```

**Update formula** (per tick):

```
inputs(target) = Σ edge.weight * nodes[edge.source].level   (over edges where e.target == target)
raw = sigmoid(inputs - threshold)
new_level = clamp(DECAY * old_level + (1 - DECAY) * raw, 0, 1)
```

**Trigger condition**: genes with `new_level > 0.5` enter this tick's trigger list.

### 3.11 [`lsystem.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/lsystem.py)

```python
@dataclass(slots=True)
class TurtleState: x, y, heading, stack: list[tuple[float,float,float]]

class LSystem:
    def __init__(self, axiom: str, rules: dict[str,str], angle: float = 25.0, step: float = 1.0): ...
    def iterate(self) -> list[tuple[float, float]]   # parallel rewrite + turtle interpretation, returns new points
    def state_length(self) -> int
```

**Turtle commands**: `F` moves forward one step and draws, `+`/`-` turns by angle, `[`/`]` push/pop the stack. **Each `iterate` resets the turtle to the origin and re-interprets the entire state** (ensuring reproducible morphology). Initial heading is 90° (pointing up).

### 3.12 [`reaction_diffusion.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/reaction_diffusion.py)

```python
class GrayScott:
    def __init__(self, n: int = 32, F: float = 0.035, k: float = 0.065,
                 Du: float = 0.16, Dv: float = 0.08, seed: int = 42): ...
    def step(self) -> None
    def emit(self, i: int, j: int, amount: float = 1.0) -> None
    def total_v(self) -> float
```

**Equations**:

```
U' = U + (Du*∇²U - U*V² + F*(1-U))
V' = V + (Dv*∇²V + U*V² - (F+k)*V)
∇²f = (f[i-1][j] + f[i+1][j] + f[i][j-1] + f[i][j+1] - 4*f[i][j]) * 0.2
```

**Invariants**: after each step `U, V ∈ [0, 1]` (clamped). `seed=42` guarantees reproducible initialization.

### 3.13 [`cell.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/cell.py)

```python
DIRECTIONS = [(0,-1),(1,0),(0,1),(-1,0)]    # 0=N,1=E,2=S,3=W

@dataclass(slots=True)
class Cell:
    name: str = "cell-0"
    x: int = 0; y: int = 0
    energy: int = 100
    proteins: dict[int, float]
    slots: list                              # 256 memory slots
    alive: bool = True
    morphology_points: list[tuple[float,float]]
    color: tuple[int,int,int] = (255,255,255)
    age: int = 0; divisions: int = 0

    def add_protein(self, kind: int, amount: float = 1.0) -> None
    def consume_protein(self, kind: int, amount: float = 1.0) -> float
    def move(self, direction: int) -> None    # energy -1
    def consume_energy(self, n: int = 1) -> bool
    def feed(self, amount: int = 10) -> None
    def divide(self) -> bool                  # divides only if energy >=2, energy halved
    def die(self) -> None
    def dump(self) -> str
```

### 3.14 [`vm.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/vm.py)

```python
@dataclass(slots=True)
class Frame: return_ip: int; gene_name: str

class CellVM:
    def __init__(self, chunk: Chunk, program: Program): ...
    def run(self, max_ticks: int) -> list[dict]    # returns trace
    debug: bool                                     # enable per-instruction printing
    trace: list[dict]
```

**`_init_subsystems` behavior**:
- each promoter becomes a GRN node: `strength < 0` is treated as constitutive (`initial_level=1.0`)
- each gene becomes a GRN node: with a promoter, `threshold = promoter.strength, initial=0.0`; without a promoter, `threshold=-1.0, initial=1.0` (constitutive expression)
- L-systems are loaded into `self.lsystems` (using ruleset 0)
- if `field_decl` exists, create a `GrayScott`

**`_dispatch` key semantics**:
- both `OP_HALT` and `OP_RETURN` pop the current frame and restore `ip` to `return_ip`
- `OP_CALL_GENE` reads a u16 offset, pushes a new frame, and jumps
- `OP_GROW_LSYSTEM`: by default acts on the first L-system, iterating once
- `OP_REACT`: executes `config.react_steps` iterations of `field.step()`
- `OP_EMIT_MORPHOGEN`: injects V at `(cell.x % n, cell.y % n)`
- unknown opcode: skip its operand bytes (per `OP_OPERAND_BYTES`)
- stack underflow: silently ignored (no exception, preserving prototype robustness)

### 3.15 [`cli.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/cli.py)

```python
def main(argv: list[str] | None = None) -> int: ...

def _stop_codons_from_table(table: dict[str, Op]) -> set[str]: ...
def _emit_csv(trace: list[dict]) -> None: ...
def _emit_ppm(vm: CellVM, prefix: str) -> None: ...
```

**CLI arguments**:

| Argument | Type | Default | Description |
|---|---|---|---|
| `source` | Path | — | `.helix` source file |
| `--table` | choice | `standard` | `standard` / `mito_vertebrate` / `ciliate` |
| `--disassemble` | flag | False | print disassembly and exit |
| `--debug` | flag | False | per-instruction VM trace |
| `--csv` | flag | False | output CSV to stdout |
| `--png PREFIX` | str | None | output PPM file (`<PREFIX>.ppm`) |
| `--ticks N` | int | None | override `#config ticks` |

---

## 4. Bytecode Encoding Conventions

### 4.1 opcode Encoding Space

[`Op`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/codon_table.py) occupies 1 byte, categorized by the high 4 bits:

| High 4 bits | Category | opcode |
|---|---|---|
| `0x1_` | control flow | `OP_START 0x10` `OP_HALT 0x11` `OP_RETURN 0x12` `OP_NOP 0x13` |
| `0x2_` | stack | `OP_PUSH_CONST 0x20` `OP_POP 0x21` `OP_DUP 0x22` `OP_SWAP 0x23` |
| `0x3_` | synthesis | `OP_BUILD_PROTEIN 0x30` `OP_BUILD_MEMBRANE 0x31` `OP_BUILD_PIGMENT 0x32` |
| `0x4_` | behavior | `OP_MOVE 0x40` `OP_SIGNAL 0x41` `OP_DIVIDE 0x42` `OP_DIE 0x43` `OP_FEED 0x44` |
| `0x5_` | morphology | `OP_GROW_LSYSTEM 0x50` `OP_DIFFUSE 0x51` `OP_REACT 0x52` `OP_EMIT_MORPHOGEN 0x53` |
| `0x6_` | memory/regulation | `OP_READ_MEM 0x60` `OP_WRITE_MEM 0x61` `OP_MODIFY_STATE 0x62` `OP_REGULATE 0x63` `OP_BIND 0x64` |
| `0x7_` | call | `OP_CALL_GENE 0x70` |
| `0x8_` | synthetic control flow | `OP_JUMP 0x80` `OP_JUMP_IF_ZERO 0x81` |
| `0x9_` | arithmetic | `OP_ADD 0x90` `OP_SUB 0x91` `OP_MUL 0x92` `OP_LT 0x93` `OP_NOT 0x94` |
| `0xF_` | system | `OP_TICK 0xF0` `OP_DEBUG 0xFE` |

**Extension slots**: `0x00-0x0F`, `0x33-0x3F`, `0x45-0x4F`, `0x54-0x5F`, `0x65-0x6F`, `0x71-0x7F`, `0x82-0x8F`, `0x95-0xEF`, `0xFF` are reserved for future extensions.

### 4.2 Operand Width Table

See [`OP_OPERAND_BYTES`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/codon_table.py):

- **0 bytes**: all control flow (except CALL/JUMP), stack operations (except PUSH_CONST), arithmetic, `OP_BUILD_PIGMENT`, `OP_TICK`, `OP_DEBUG`
- **1 byte** (u8): synthesis, behavior, morphology, memory/regulation, `OP_PUSH_CONST`
- **2 bytes** (u16 big-endian): `OP_CALL_GENE`, `OP_JUMP`, `OP_JUMP_IF_ZERO`

**Fetch convention**: the VM and the disassembler must use the same `OP_OPERAND_BYTES` to determine the next instruction boundary, otherwise offsets misalign.

### 4.3 Endianness and Alignment

- multi-byte operands are **big-endian** encoded (high byte first)
- no alignment requirement (compact byte stream)
- `Chunk.read_u16`: `(code[ip] << 8) | code[ip+1]`

### 4.4 Constant Pool

`Chunk.constants` is a heterogeneous `list[Any]`, populated by the Compiler:

| Tuple shape | Meaning |
|---|---|
| `("lsystem_axiom", name, axiom)` | L-system axiom |
| `("lsystem_rules", name, rules_dict)` | L-system rulesets |
| `("lsystem_angle", name, angle)` | L-system angle |
| `("lsystem_step", name, step)` | L-system step |
| `("gene_name", gname)` | gene name (for debugging) |

The u8 operand of `OP_PUSH_CONST` is a constant index; when out of range, the VM falls back to pushing the index value itself (prototype tolerance).

### 4.5 gene_offsets and CALL_GENE Backfill

- `gene_offsets[name]` is written before the first-pass emit (value is the current `len(code)`)
- `OP_CALL_GENE` emits placeholder `0x00 0x00` in the first pass, recording `(placeholder start ip, wobble_arg)` into `_call_sites`
- the second pass uses `wobble_arg % len(genes)` to select the target gene name and writes `gene_offsets[target]` back into the placeholder big-endian
- if the target does not exist, the offset stays 0 (at runtime it jumps to the chunk head, harmlessly handled by `OP_START`)

---

## 5. Codon to opcode Mapping Strategy

### 5.1 Degeneracy Grouping (by Amino Acid Family)

| Amino acid | Codons | opcode | Notes |
|---|---|---|---|
| Met (M) | 1 (ATG) | `OP_START` | start codon |
| Stop | 3 (TAA/TAG/TGA) | `OP_HALT` | standard table |
| Phe (F) | 2 (TTT/TTC) | `OP_PUSH_CONST` | wobble distinguishes the constant value |
| Leu (L) | 6 (CTN+TTA/TTG) | `OP_GROW_LSYSTEM` | |
| Ile (I) | 3 (ATT/ATC/ATA) | `OP_READ_MEM` | |
| Val (V) | 4 (GTN) | `OP_MOVE` | |
| Ser (S) | 6 (TCN+AGT/AGC) | `OP_SIGNAL` | |
| Pro (P) | 4 (CCN) | `OP_MODIFY_STATE` | |
| Thr (T) | 4 (ACN) | `OP_DIFFUSE` | |
| Ala (A) | 4 (GCN) | `OP_BUILD_PROTEIN` | |
| Tyr (Y) | 2 (TAT/TAC) | `OP_WRITE_MEM` | |
| His (H) | 2 (CAT/CAC) | `OP_REGULATE` | |
| Gln (Q) | 2 (CAA/CAG) | `OP_EMIT_MORPHOGEN` | |
| Asn (N) | 2 (AAT/AAC) | `OP_DIVIDE` | |
| Lys (K) | 2 (AAA/AAG) | `OP_DIE` | |
| Asp (D) | 2 (GAT/GAC) | `OP_REACT` | |
| Glu (E) | 2 (GAA/GAG) | `OP_FEED` | |
| Cys (C) | 2 (TGT/TGC) | `OP_BIND` | |
| Trp (W) | 1 (TGG) | `OP_BUILD_PIGMENT` | |
| Arg (R) | 6 (CGN+AGA/AGG) | `OP_CALL_GENE` | wobble selects the target gene |
| Gly (G) | 4 (GGN) | `OP_BUILD_MEMBRANE` | |

### 5.2 Wobble Position Semantics

The third-base wobble position `WOBBLE_BITS = {"A":0, "C":1, "G":2, "T":3}` is used in two ways:

- **1-byte-operand instructions**: the wobble value is used directly as the operand (e.g., the protein kind of `OP_BUILD_PROTEIN`, the direction of `OP_MOVE`, the slot number of `OP_READ_MEM`)
- **`OP_CALL_GENE`**: the wobble value selects the target gene during second-pass compilation (`wobble_arg % len(genes)`)

### 5.3 Variable Translation Tables

| Table | NCBI | Differences from the standard table |
|---|---|---|
| `standard` | table 1 | baseline |
| `mito_vertebrate` | table 2 | `TGA→OP_BUILD_PIGMENT`, `ATA→OP_START`, `AGA/AGG→OP_HALT` |
| `ciliate` | table 6 | `TAA/TAG→OP_EMIT_MORPHOGEN` (no longer STOP) |

**Switching mechanism**: CLI `--table` → `get_table()` → passed to `Compiler(table)` and `Parser(stop_codons=...)`; `stop_codons` is derived by `_stop_codons_from_table` (all codons mapping to `OP_HALT`), ensuring ORF recognition and opcode mapping stay consistent.

### 5.4 Extension Slots

All 64 codons are already used. Expansion paths:
- introduce the `OP_NOP` family as a prefix byte (`0x13 <next-byte>`) to extend the instruction subspace
- use the constant pool to carry complex operands (e.g., target names for multi-gene calls)
- introduce a two-level lookup (codon + context modifier)

---

## 6. VM Execution Model

### 6.1 Tick Main Loop Timing

```
┌─────────────────────────────────────────────────────────────┐
│ while tick < max_ticks and cell.alive:                      │
│                                                             │
│   1. triggered = grn.step()           # update all gene levels │
│   2. for g in triggered:                                    │
│         _call_gene(g)                # push frame + jump to gene offset │
│   3. _execute_pending()              # execute bytecode until frames empty │
│                                      # or ops_per_tick exhausted │
│   4. _flush_morphology()             # (no-op, immediate morphology) │
│   5. _feedback()                     # field V → pigment gene │
│   6. _snapshot()                     # trace.append(snap)   │
│   7. tick += 1                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Frame Stack and Calling Conventions

- **GRN-triggered call**: `_call_gene(name)` pushes `Frame(return_ip=self.ip, gene_name=name)`, `ip = gene_offsets[name]`
- **`OP_CALL_GENE`**: pushes `Frame(return_ip=self.ip, gene_name="<call>")`, `ip = read_u16()`
- **`OP_HALT` / `OP_RETURN`**: pop the frame, `ip = frame.return_ip`; halt if there is no frame
- **quota exhausted but frames remain**: keep the frames and continue next tick (no forced pop)

### 6.3 ops_per_tick Quota

`_execute_pending` decrements `quota` by 1 per executed instruction and exits when `quota == 0`. `quota` is initialized to `program.config.ops_per_tick` (default 64). **Purpose**: prevent a single gene's infinite loop from freezing the whole tick.

### 6.4 GRN ↔ Bytecode ↔ Morphogen Field Feedback

```
        ┌──────────┐  triggered genes   ┌──────────┐
        │   GRN    │ ─────────────────▶ │ bytecode │
        │          │                    │ execution│
        └────▲─────┘                    └────┬─────┘
             │                               │ OP_EMIT_MORPHOGEN
             │ _feedback:                    ▼
             │ field.v[i][j] →      ┌──────────────┐
             │ pigment.level +=     │  Gray-Scott  │
             │ v * 0.1              │   field U,V  │
             └──────────────────────┤              │
                                    └──────────────┘
```

`_feedback` applies only when `field` exists and `"pigment" in grn.nodes`, adding the V concentration at the cell's grid cell times `v * 0.1` to the pigment gene's level (clamped).

---

## 7. Error Handling Matrix

| Exception type | Stage | Typical trigger | User-visible message | Propagation path |
|---|---|---|---|---|
| `LexError` | Lexer | DNA length not a multiple of 3; illegal characters; `#` not followed by an annotation name | `[LexError @ line N] msg` | caught by CLI → stderr → exit 1 |
| `ParseError` | Parser | ORF without ATG / without STOP; unclosed annotation block; unknown annotation; missing `name=` field; invalid strength value | `[ParseError @ line N] msg` | caught by CLI → stderr → exit 1 |
| `SemanticError` | Semantic | duplicate symbols; reference to an undefined promoter; empty ORF; `ticks<=0` | `[SemanticError] msg` | caught by CLI → stderr → exit 1 |
| `RegulationError` | Semantic | undefined `#regulate` source/target | `[RegulationError] msg` | caught by CLI → stderr → exit 1 |
| `CompileError` | Compiler | unknown codon (table/source mismatch) | `[CompileError @ line N codon #M] msg` | caught by CLI → stderr → exit 1 |
| `HelixError` | any | unknown table name in `get_table` | `[HelixError] msg` | caught by CLI → stderr → exit 1 |
| `RuntimeHelixError` | VM | (reserved; the current VM is fault-tolerant and does not raise) | — | — |
| `warnings` (not an exception) | Semantic | regulation-cycle detection | not printed (`SemanticAnalyzer.warnings` list) | readable by the caller |

**CLI error handling** ([`cli.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/cli.py)):

```python
try:
    table = get_table(args.table)
    tokens = list(Lexer(src).tokens())
    program = Parser(tokens, stop_codons=stop_codons).parse()
    SemanticAnalyzer(program).check()
    chunk = Compiler(table).compile(program)
except Exception as e:
    print(f"compile error: {e}", file=sys.stderr)
    return 1

try:
    trace = vm.run(program.config.ticks)
except Exception as e:
    print(f"runtime error: {e}", file=sys.stderr)
    return 1
```

> Note: during the prototype stage, a broad `except Exception` ensures that unexpected errors still produce friendly messages; the production version can narrow it to `except HelixError`.

---

## 8. Exit Code Conventions

| Exit code | Meaning |
|---|---|
| `0` | success |
| `1` | compile-time or runtime error (any `HelixError` subclass) |
| `2` | source file missing / CLI argument error (built into argparse) |

---

## 9. Performance Budget and Benchmarks

### 9.1 Module-Level Budget (single-core Python 3.11.15, 2026 hardware baseline)

| Workload | Target | Measurement method |
|---|---|---|
| Lex + Parse 1 KB `.helix` | < 30 ms | wrap `time.perf_counter` around `list(Lexer(src).tokens())` + `Parser(tokens).parse()` |
| Compile 100 genes | < 50 ms | build a 100-gene Program, time `Compiler(table).compile(prog)` |
| VM 1000 ticks single cell + GRN (5 nodes) | < 500 ms | time `vm.run(1000)` |
| 32×32 Gray-Scott 200 steps | < 3 s | time `for _ in range(200): field.step()` |
| L-system 7 iterations (≤10k chars) | < 100 ms | time `for _ in range(7): ls.iterate()` |

### 9.2 Performance Regression Gate

CI runs [`tests/test_end_to_end.py`](file:///Users/admin/PycharmProjects/HelixLang/tests/test_end_to_end.py) and asserts:

- total execution time of the 4 examples < 5 s
- 32×32 field, 200 steps < 3 s

CI fails on timeout (using `pytest --timeout=30` or `@pytest.mark.timeout`).

### 9.3 Optimization Paths (not enabled)

- port reaction-diffusion to numpy: `u, v = np.ndarray`, vectorized Laplacian (`fast` extra)
- replace VM dispatch `match/case` with `dict[Op, callable]`, micro-benchmark the comparison
- preallocate L-system turtle point sequences with numpy arrays

---

## 10. Build and Release

### 10.1 Development-Mode Installation

```bash
cd /Users/admin/PycharmProjects/HelixLang
/opt/anaconda3/envs/helix/bin/python -m pip install -e . \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn

# including dev dependencies
/opt/anaconda3/envs/helix/bin/python -m pip install -e '.[dev]' \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 10.2 Running

```bash
# entry script (available after pip install -e .)
helixlang examples/01_hello_dna.helix
helixlang examples/02_lac_operon.helix --csv > trace.csv
helixlang examples/03_plant_growth.helix --png plant
helixlang examples/04_turing_pattern.helix --png turing
helixlang examples/05_table_switch.helix --table=mito_vertebrate
helixlang examples/01_hello_dna.helix --disassemble

# python -m form (no installation required)
/opt/anaconda3/envs/helix/bin/python -m helixlang examples/01_hello_dna.helix
```

### 10.3 Testing

```bash
/opt/anaconda3/envs/helix/bin/python -m pytest -v
/opt/anaconda3/envs/helix/bin/python -m pytest --cov=helixlang --cov-report=term-missing

# no package installation required
PYTHONPATH=src /opt/anaconda3/envs/helix/bin/python -m pytest
```

### 10.4 Wheel Build

```bash
/opt/anaconda3/envs/helix/bin/python -m pip install build
/opt/anaconda3/envs/helix/bin/python -m build --wheel
# artifact: dist/helixlang-0.1.0-py3-none-any.whl
```

### 10.5 Versioning Strategy

- follow [SemVer](https://semver.org/): `MAJOR.MINOR.PATCH`
- single source of truth for the version: the `version` field in `pyproject.toml` + `__version__` in `__init__.py`
- sync the two before releasing, and tag `v0.1.0` in git

---

## 11. CI Pipeline (Reference)

`.github/workflows/ci.yml` (suggested):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11.15"
      - name: Configure pip mirror (tsinghua)
        run: |
          mkdir -p ~/.config/pip
          echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn" > ~/.config/pip/pip.conf
      - run: python -m pip install -e '.[dev]'
      - run: python -m pytest -v --cov=helixlang --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

**Gates**:
- pytest all green
- coverage ≥ 80% (prototype target; production ≥ 90%)
- 4 end-to-end example tests < 5 s

---

## 12. Code Conventions

### 12.1 File Header Template

At the top of every `.py`:

```python
"""<One-sentence description of the module's responsibility>.

<Optional: detailed description, algorithm references, invariants>
"""
from __future__ import annotations
```

### 12.2 Naming

| Category | Convention | Example |
|---|---|---|
| module | `snake_case` | `codon_table.py` |
| class | `PascalCase` | `CellVM`, `GrayScott` |
| function/method | `snake_case` | `_extract_orf`, `add_protein` |
| constant | `UPPER_SNAKE` | `STANDARD_TABLE`, `WOBBLE_BITS` |
| private | prefix `_` | `_dispatch`, `_call_sites` |
| Op enum | `OP_` prefix | `OP_BUILD_PROTEIN` |

### 12.3 Type Annotations

- all public APIs must have type annotations
- use `from __future__ import annotations` to enable PEP 563 lazy evaluation
- prefer built-in generics (`list[int]` over `List[int]`)
- express `Optional` as `X | None`

### 12.4 Dataclasses

- all AST nodes and state objects use `@dataclass(slots=True)` (less memory, faster attribute access)
- use `field(default_factory=...)` for mutable defaults

### 12.5 Raising Errors

- always raise `HelixError` subclasses carrying `line`/`col`/`codon_index`
- do not use `assert` for input validation (`assert` can be disabled by `-O`); `assert` is used only for module-level invariant self-checks (e.g., the 64-codon assertion at the end of `codon_table.py`)

### 12.6 Import Order

```python
# 1. standard library
import math
from dataclasses import dataclass, field

# 2. third-party (zero runtime deps; usually empty during prototype)

# 3. this package
from helixlang.errors import HelixError
```

### 12.7 Output Conventions

- **`print` is forbidden** except for CLI output and `OP_DEBUG`
- debug info goes to `stderr`, data goes to `stdout` (for pipelining)

---

## 13. Test Pyramid

### 13.1 Layers

| Layer | Share | Files | Description |
|---|---|---|---|
| unit tests | ~70% | `test_lexer.py` `test_parser.py` `test_codon_table.py` `test_compiler.py` `test_vm.py` `test_grn.py` `test_lsystem.py` `test_reaction_diffusion.py` | module isolation, no file IO dependency |
| end-to-end | ~30% | `test_end_to_end.py` | runs the full pipeline, asserts key invariants |

### 13.2 Fixture Conventions ([`tests/conftest.py`](file:///Users/admin/PycharmProjects/HelixLang/tests/conftest.py))

- `small_dna_source`: a small `.helix` snippet with 1–2 genes
- `sample_program`: an already-parsed Program
- `compiled_chunk`: an already-compiled Chunk

### 13.3 Test Conventions

- one `test_<name>.py` per module
- test function names `test_<behavior>_<condition>`, e.g., `test_orf_extraction_no_start_raises`
- assert invariants rather than specific output formats (avoid brittle tests)
- tests can run independently: `PYTHONPATH=src python -m pytest tests/test_lexer.py`
- do not depend on test execution order

### 13.4 Coverage Targets

| Module | Target |
|---|---|
| `codon_table.py` / `errors.py` / `bytecode.py` | 100% |
| `lexer.py` / `parser.py` / `compiler.py` | ≥ 95% |
| `vm.py` / `grn.py` / `lsystem.py` / `reaction_diffusion.py` | ≥ 85% |
| `cli.py` | ≥ 70% (argv parsing is easy to test; file IO uses tmp) |

---

## 14. Observability

### 14.1 Trace Schema (`CellVM._snapshot`)

Each tick appends one `dict`:

```python
{
    "tick": int,
    "x": int, "y": int,
    "energy": int,
    "alive": bool,
    "proteins": dict[int, float],          # kind → concentration
    "color": tuple[int, int, int],
    "gene_levels": dict[str, float],       # gene name → level
    "morphology_points_count": int,
    "field_total_v": float,                # total morphogen field V (0.0 if no field)
}
```

### 14.2 CSV Output Contract (`_emit_csv`)

```
tick,x,y,energy,alive,proteins,morphology_points,field_total_v
0,0,0,100,1,"0:1.00;1:1.00",1,0.0000
```

- `proteins` is `;`-separated `kind:concentration` pairs, wrapped as a whole in double quotes (against CSV injection)
- concentrations keep 2 decimal places
- `field_total_v` keeps 4 decimal places

### 14.3 PPM Output Contract (`_emit_ppm`)

- **with a morphogen field**: output an `n×n` V concentration image; blue channel = `int(v * 255)`, red and green channels are 0
- **no morphogen field but with L-system**: output a binary image, black=morphology points, white=background, size = `(maxx-minx+2) × (maxy-miny+2)`
- format: P3 (ASCII RGB)
- filename: `<prefix>.ppm`

### 14.4 Debug Mode

`--debug` sets `vm.debug = True`, printing before each instruction executes:

```
[tick=0 ip=1] OP_BUILD_PROTEIN stack=[]
```

### 14.5 Disassembly Output

`--disassemble` outputs gene offsets, per-instruction disassembly (including codon # and line), and the constant pool; see §3.9 for details.

---

## 15. Engineering Invariants and Regression Baseline

> Every invariant must have a corresponding test.

| # | Invariant | Test location |
|---|---|---|
| I1 | `len(STANDARD_TABLE) == 64` and covers all `{A,C,G,T}^3` | `test_codon_table.py` + module-level `assert` |
| I2 | `len(code) == len(lines) == len(codon_indices)` | `test_compiler.py` |
| I3 | after `_patch_calls`, `OP_CALL_GENE` operands are valid `gene_offsets` | `test_compiler.py` |
| I4 | `frames` is empty when the VM exits (normal halt) | `test_vm.py` |
| I5 | morphogen field `U, V ∈ [0, 1]` after any step | `test_reaction_diffusion.py` |
| I6 | GRN `level ∈ [0, 1]` after any step | `test_grn.py` |
| I7 | ORF starts with `ATG`, ends with STOP | `test_parser.py` + `_check_orfs` |
| I8 | `stop_codons` consistent with `--table` (under mito, TGA is not a STOP) | `test_end_to_end.py::test_table_switch_*` |
| I9 | `GrayScott` is reproducible with `seed=42` | `test_reaction_diffusion.py` |
| I10 | L-system resets the turtle every time; iterate is reproducible | `test_lsystem.py` |
| I11 | `#config ticks/ops_per_tick/react_steps > 0` | `test_parser.py` + `_check_config` |
| I12 | CLI exit codes: success 0 / error 1 / missing file 2 | `test_end_to_end.py` |

---

## 16. Risks and Mitigations

### 16.1 Engineering Risks

| Risk | Impact | Mitigation |
|---|---|---|
| accidentally using the system Python | inconsistent environment | docs mandate the absolute path `/opt/anaconda3/envs/helix/bin/python`; CI pins 3.11.15 |
| `pip.conf` not taking effect | downloads go to the default source and time out | CI explicitly passes `-i` + `--trusted-host`; shells set `PIP_INDEX_URL` |
| 64 codons are insufficient for all desired semantics | insufficient expressiveness | wobble 4× expansion + constant pool + `OP_NOP`-prefix extension slots (§5.4) |
| GRN dispatch deadlock/starvation | some genes never trigger | `ops_per_tick` quota + `DECAY=0.7` decay + constitutive genes with `initial=1.0` |
| reaction-diffusion numerical explosion | field values overflow | clamp `[0,1]` every step; `dt=1.0` sub-step |
| L-system string exponential growth | memory explosion | limit the maximum iterations (indirectly bounded by `#config ticks`) |
| `OP_CALL_GENE` offset backfill errors | jump to a wrong address | two-pass compilation + `_call_sites` recording; offset stays 0 when the target is missing (jumps to `OP_START`) |
| stack underflow | silent error | silently ignored during prototype; the production version should raise `RuntimeHelixError` |
| `nan` entering the constant pool | `add_constant` dedup fails | docs mandate not storing `nan`; tests cover the numerical cases |

### 16.2 Design Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| interpreter vs compiling to native | stack-based bytecode VM | portable, disassemblable, debugger-friendly |
| pure Python vs C extension | pure Python + optional numpy | zero runtime dependencies, easy install; the performance bottleneck is reaction-diffusion, which can be vectorized later |
| recursive descent vs Lark | hand-written recursive descent | the grammar is simple (annotations + DNA blocks), no LALR needed; keeps control |
| `match/case` vs dict dispatch | `match/case` | native to Python 3.10+, readable; the performance bottleneck is not here |
| GRN bistability implementation | simple sigmoid + decay | true bistability requires Hill cooperativity, beyond the prototype scope; tests use convergence validation instead |

---

## 17. Rollout Steps (Execution Order and Status)

| # | Step | Status | Artifact |
|---|---|---|---|
| 1 | create `pyproject.toml` + `pip.conf` + `.python-version` | ✅ | [`pyproject.toml`](file:///Users/admin/PycharmProjects/HelixLang/pyproject.toml) [`pip.conf`](file:///Users/admin/PycharmProjects/HelixLang/pip.conf) [`.python-version`](file:///Users/admin/PycharmProjects/HelixLang/.python-version) |
| 2 | create `src/helixlang/__init__.py` + `__main__.py` | ✅ | [`__init__.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/__init__.py) [`__main__.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/__main__.py) |
| 3 | implement `errors.py` + `codon_table.py` (64 codons + three tables + self-check assertions) | ✅ | [`errors.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/errors.py) [`codon_table.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/codon_table.py) |
| 4 | implement `bytecode.py` + `disassembler.py` | ✅ | [`bytecode.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/bytecode.py) [`disassembler.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/disassembler.py) |
| 5 | implement `lexer.py` + `ast_nodes.py` + `parser.py` + `semantic.py` | ✅ | [`lexer.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/lexer.py) [`ast_nodes.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/ast_nodes.py) [`parser.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/parser.py) [`semantic.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/semantic.py) |
| 6 | implement `compiler.py` (two-pass compilation + CALL_GENE backfill) | ✅ | [`compiler.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/compiler.py) |
| 7 | implement `grn.py` + `lsystem.py` + `reaction_diffusion.py` + `cell.py` | ✅ | [`grn.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/grn.py) [`lsystem.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/lsystem.py) [`reaction_diffusion.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/reaction_diffusion.py) [`cell.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/cell.py) |
| 8 | implement `vm.py` (tick main loop + dispatch + feedback) | ✅ | [`vm.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/vm.py) |
| 9 | implement `cli.py` | ✅ | [`cli.py`](file:///Users/admin/PycharmProjects/HelixLang/src/helixlang/cli.py) |
| 10 | write `examples/01-05.helix` | ✅ | [`examples/`](file:///Users/admin/PycharmProjects/HelixLang/examples) |
| 11 | write `tests/test_*.py` (unit + end-to-end) | ✅ | [`tests/`](file:///Users/admin/PycharmProjects/HelixLang/tests) |
| 12 | verify with `pip install -e .[dev]` + `pytest -v` | ✅ | 70 passed in 0.52s |
| 13 | run the 4 examples + translation table switching | ✅ | see [`05-prototype-plan.md`](file:///Users/admin/PycharmProjects/HelixLang/doc/05-prototype-plan.md) §8 |
| 14 | engineering design doc refinement (this document) | ✅ | [`06-engineering-design.md`](file:///Users/admin/PycharmProjects/HelixLang/doc/06-engineering-design.md) |

### 17.1 Verification Commands

```bash
# environment check
/opt/anaconda3/envs/helix/bin/python --version
# Python 3.11.15

# full test run
/opt/anaconda3/envs/helix/bin/python -m pytest -v
# 70 passed

# example smoke tests
/opt/anaconda3/envs/helix/bin/python -m helixlang examples/01_hello_dna.helix
/opt/anaconda3/envs/helix/bin/python -m helixlang examples/05_table_switch.helix --table=mito_vertebrate
/opt/anaconda3/envs/helix/bin/python -m helixlang examples/01_hello_dna.helix --disassemble
```

---

## 18. Subsequent Roadmap (Beyond Prototype Scope)

| Phase | Content | Trigger condition |
|---|---|---|
| short term | numpy-vectorized reaction-diffusion; VM dispatch micro-benchmark; L-system numpy point sequences | performance baseline not met |
| mid term | control-flow instructions (`OP_JUMP` family) synthesized by the compiler; real multicellular `OP_DIVIDE` division; evolution frontend (mutation/recombination/genetic algorithms) | expressiveness needs |
| long term | MLIR dialect (`helix.dna/gene/morph/sim`) lowered to LLVM; physical DNA output (Church/Goldman/Erlich encoding); CRISPR in-vivo writing | deployment on real biological hardware |
| toolchain | Lark migration (when the grammar grows complex); tree-sitter integration (IDE); Jupyter kernel (`%helix_run`) | user-experience needs |

---

## Appendix A: Module Dependency Graph

```
              ┌──────────┐
              │ errors   │ ◄──── (depended on by all modules)
              └──────────┘
              ┌──────────┐
              │codon_table│ ◄──── compiler, disassembler, vm, cli
              └──────────┘
              ┌──────────┐
              │ bytecode │ ◄──── compiler, disassembler, vm
              └──────────┘
              ┌──────────┐
              │ast_nodes │ ◄──── parser, semantic, compiler, vm
              └──────────┘

  lexer ──▶ parser ──▶ semantic       compiler ──▶ vm
    │         │          │               │          │
    └─────────┴──────────┴── ast_nodes ──┴──────────┘

  grn, lsystem, reaction_diffusion, cell ──▶ vm
  cli ──▶ (lexer, parser, semantic, compiler, vm, disassembler, codon_table)
```

No circular dependencies. `errors` and `codon_table` are leaf modules (they depend only on the standard library).

---

## Appendix B: Glossary

| Term | Meaning |
|---|---|
| ORF | Open Reading Frame, the codon sequence from ATG to STOP |
| wobble | the wobble pairing of the codon's third base, used here as the operand position |
| GRN | Gene Regulatory Network |
| Gray-Scott | a reaction-diffusion system; the parameters F/k determine the Turing pattern |
| L-system | Lindenmayer system, a parallel string rewriting system |
| Chunk | the HelixLang bytecode container |
| tick | one step of the VM main loop |
| constitutive expression | gene expression that is always active and not regulated |
| degeneracy | multiple codons encoding the same amino acid (here mapped to the same opcode) |
