# HelixLang Compiler Design

> This document describes HelixLang's compilation pipeline, AST, bytecode format, stack VM, disassembler, and implementation strategy. The prototype uses the pure Python standard library with no external dependencies.

---

## 1. Compilation Pipeline

```
.helix source
    │
    ▼  Lexer.dna_tokens() + Lexer.annot_tokens()
Token stream (CODON / ANNOT_START / KEY=VAL / ARROW / ANNOT_END)
    │
    ▼  Parser.parse()  (recursive descent)
AST (Program / Promoter / Gene / Regulation / Config)
    │
    ▼  SemanticAnalyzer.check()  (symbol table, reading frame, reference integrity)
Annotated AST
    │
    ▼  Compiler.compile()  (codon table → opcode)
BytecodeChunk (opcode[] + constant pool + line-number table + gene offset table)
    │
    ▼  Disassembler.disassemble()  (optional)
Human-readable disassembly
    │
    ▼  CellVM.run(chunk, ticks)
Execution trace + output snapshots
```

The design follows the "A Map of the Territory" chapter of Nystrom's *Crafting Interpreters* and the clox implementation, but uses a **VM first, then frontend** development order: get the bytecode format and VM working first, then write the Lexer/Parser, and finally wire up the Compiler.

---

## 2. Lexer

### 2.1 Dual-Mode Scanning

Because DNA blocks and annotation blocks have different lexical rules, the Lexer uses a **state machine**:

- **DNA mode**: skips all whitespace; aggregates every 3 bases into a CODON token; switches to annotation mode upon `#`.
- **Annotation mode**: scans line by line; recognizes `#ident`, `ident=value`, `ident->ident`, `#end`; switches back to DNA mode on a non-whitespace DNA character.

### 2.2 Token Types

```python
@dataclass
class Token:
    kind: str           # "CODON" | "ANNOT_START" | "ANNOT_END" | "KEY" | "ARROW" | "NEWLINE" | "EOF"
    value: str          # codon string or annotation field value
    line: int
    col: int
    codon_index: int    # CODON only: running codon number counted from the start of the source
```

### 2.3 Key Implementation Points

- The **codon index** runs through the whole pipeline to ease error location ("codon #42 (ATG) at line 5").
- DNA characters that appear inside annotation blocks (such as ATG in the example) are still treated as CODON tokens; the Parser decides their ownership.
- Case-insensitive: the source is normalized to uppercase before scanning.

---

## 3. Parser (Recursive Descent)

### 3.1 Grammar (EBNF)

```ebnf
program       := (promoter | gene | regulate | config | codon_stream)*

codon_stream  := CODON+                          (* bare DNA block, no annotations *)
                                              (* treated as an anonymous ORF, auto-wrapped in #gene name=__anon *)

promoter      := "#promoter" fields "#end"
gene          := "#gene" fields codon_stream? "#end"
regulate      := "#regulate" source "->" target fields
config        := "#config" fields
fields        := (IDENT "=" value)*
value         := IDENT | NUMBER | STRING | "[" value ("," value)* "]"
```

### 3.2 AST Nodes

```python
@dataclass
class Program:    genes: list; promoters: list; regulations: list; config: Config

@dataclass
class Promoter:   name: str; strength: float; genes: list[str]   # associated genes

@dataclass
class Gene:       name: str; promoter: str | None; codons: list[Codon]; orf: list[Codon]  # ORF = ATG...STOP

@dataclass
class Codon:      seq: str; index: int; line: int

@dataclass
class Regulation: source: str; target: str; strength: float

@dataclass
class Config:     ticks: int; output: list[str]; table: str
```

### 3.3 ORF Recognition

The Parser performs ORF recognition on the CODON stream inside each `#gene` block:

1. Begin recording the ORF at the first `ATG`.
2. Close the ORF upon encountering `TAA` / `TAG` / `TGA`.
3. Multiple ORFs are arranged sequentially within the same gene (polycistronic).
4. A missing `ATG` raises `FrameError`; a missing terminator raises `UnterminatedORF`.

---

## 4. SemanticAnalyzer

### 4.1 Checks

| Check | Description |
|---|---|
| **Symbol uniqueness** | Each gene/promoter name can be defined only once |
| **Reference integrity** | Names in `#regulate` and `#gene promoter=` must exist |
| **ORF validity** | Every gene must have at least one closed ORF |
| **Regulatory-loop detection** | Warns about (but does not forbid) loops in the regulatory graph |
| **Configuration integrity** | `ticks > 0`, and `table` is a known translation table |

### 4.2 Symbol Table

```python
symbols = {
    "promoters": {name: Promoter},
    "genes":     {name: Gene},
    "regulations": [(source, target, strength)],
}
```

---

## 5. Compiler

### 5.1 Codon Table (codon_table.py)

Core data structure:

```python
# Standard translation table
STANDARD_TABLE: dict[str, Opcode] = {
    "ATG": OP_START,
    "TAA": OP_HALT, "TAG": OP_HALT, "TGA": OP_HALT,
    "TTT": OP_PUSH_CONST, "TTC": OP_PUSH_CONST,  # Phe family
    # ... the remaining 58 codons
}

# Third-base gear
WobbleBits = {"A": 0, "C": 1, "G": 2, "T": 3}
```

When switching translation tables, a different `dict` is constructed (e.g., in the mito table, `"TGA": OP_BUILD_PIGMENT`).

### 5.2 Compilation Flow

```python
def compile_program(program, table=STANDARD_TABLE) -> Chunk:
    chunk = Chunk()
    gene_offsets = {}     # name -> bytecode start offset
    for gene in program.genes:
        gene_offsets[gene.name] = len(chunk.code)
        for codon in gene.orf:
            op = table[codon.seq]
            arg = wobble(codon.seq)   # third base
            chunk.emit(op, arg, codon.line, codon.index)
        # append one at the end if there is no HALT
        if not chunk.ends_with_halt():
            chunk.emit(OP_HALT, 0, ...)
    # second pass: replace OP_CALL_GENE operands from gene names to offsets
    patch_calls(chunk, gene_offsets)
    return chunk
```

### 5.3 Control-Flow Instructions

Although the codon table itself has no "jump" instruction, the compiler can **synthesize** control flow while generating bytecode:

- `OP_JUMP` / `OP_JUMP_IF_ZERO`: inserted by the compiler during optimization (e.g., unrolling a loop into a conditional jump).
- These instructions do not appear in the source codons; they are "synthetic instructions" at the bytecode layer.
- Advanced usage: extended codon tables use the `OP_NOP` family as placeholders, triggered via `OP_REGULATE mode=jump`.

In the prototype phase **control-flow synthesis is not implemented**; the focus is on the linear-ORF + GRN-scheduling execution model. The GRN is itself the control flow (data-driven scheduling).

### 5.4 Constant Pool

Stores:
- Protein name strings
- L-system rule sets
- Morphogen IDs
- Gene names (for disassembly annotations)

```python
class Chunk:
    code: bytearray       # opcode + operands
    constants: list       # constant pool
    lines: list[int]      # source line number per byte
    codon_indices: list[int]
    gene_offsets: dict[str, int]
```

---

## 6. Bytecode VM

### 6.1 Design Choices

**Stack VM** + a few "cell register" slots. Rationale:

1. Biological metaphor fit: the ribosome is itself a stack machine that takes things in and out in order.
2. The Python interpreter's own dispatch overhead is large, drowning out the difference between stack and register designs; choosing the simpler stack design yields more benefit.
3. The clox `run()` loop serves as the template.

### 6.2 Core Structures

```python
class CellVM:
    chunk: Chunk
    ip: int = 0                  # instruction pointer
    stack: list = []             # value stack
    frames: list[Frame] = []     # call frames (pushed by OP_CALL_GENE)
    cell: Cell                   # cell state
    grn: GRN                     # gene regulatory network
    tick: int = 0

class Frame:
    return_ip: int
    gene_name: str
    local_slots: dict            # memory slots
```

### 6.3 Dispatch Loop

```python
def run(self, max_ticks):
    while self.tick < max_ticks:
        self.grn_step()                    # 1. GRN update
        triggered = self.grn.triggered_genes()  # 2. genes above threshold
        for g in triggered:
            self.call_gene(g)              # 3. push a frame and execute
        self.execute_pending()             # 4. bytecode execution (incl. morphology/behavior)
        self.tick += 1
        self.snapshot()                    # 5. output snapshot
```

Bytecode dispatch uses `match`/`case` (Python 3.10+):

```python
def execute_one(self):
    op = self.chunk.code[self.ip]; self.ip += 1
    match op:
        case 0x10:  # OP_START
            pass    # merely marks an ORF entry
        case 0x11:  # OP_HALT
            self.frames.pop()
            return
        case 0x30:  # OP_BUILD_PROTEIN
            kind = self.read_byte()
            self.cell.add_protein(kind, 1)
        case 0x40:  # OP_MOVE
            d = self.read_byte()
            self.cell.move(d)
        case 0x50:  # OP_GROW_LSYSTEM
            rules = self.read_byte()
            self.cell.grow_lsystem(rules)
        case 0x70:  # OP_CALL_GENE
            gid = self.read_byte()
            self.call_gene(gid)
        case 0x80:  # OP_JUMP
            off = self.read_u16()
            self.ip += off
        ...
```

### 6.4 Debugging Support

- `DEBUG_TRACE_EXECUTION`: prints `[ip=N] OP_XXX arg=Y  stack=[...]` before each instruction executes.
- `disassemble_chunk(chunk)`: statically disassembles an entire chunk.
- `cell.dump()`: prints the cell state (protein pool, energy, position, morphology-field statistics).

---

## 7. Disassembler

Output format:

```
=== HelixLang Disassembly ===
Gene lacZ @ offset 0
  0000  OP_START                            ; ATG (#0, line 5)
  0001  OP_BUILD_PROTEIN arg=0              ; GCT (#1, line 5)
  0003  OP_BUILD_MEMBRANE arg=1             ; GGC (#2, line 5)
  0005  OP_MOVE arg=2                       ; GTA (#3, line 5)
  0007  OP_GROW_LSYSTEM arg=1               ; CTC (#4, line 5)
  0009  OP_HALT                             ; TAA (#5, line 5)
Gene lacY @ offset 11
  0011  OP_START                            ; ATG (#6, line 8)
  ...
Constants:
  [0] "lacZ_protein"
  [1] "axiom:F"
  [2] "rule:F->F[+F]F[-F]F"
```

The disassembly also shows both the **source codon** and the **target opcode**, letting users cross-check against the biological semantics.

---

## 8. Implementation Strategy

### 8.1 File Organization

```
src/helixlang/
├── __init__.py          # package entry + public exports
├── codon_table.py       # the three translation tables STANDARD/MITO/CILIATE + WobbleBits
├── lexer.py             # Token + Lexer
├── ast_nodes.py         # AST dataclasses
├── parser.py            # Parser (recursive descent)
├── semantic.py          # SemanticAnalyzer
├── bytecode.py          # Opcode constants + Chunk
├── compiler.py          # Program → Chunk
├── disassembler.py      # Chunk → str
├── grn.py               # GRN graph + sigmoid update
├── lsystem.py           # LSystem class
├── reaction_diffusion.py# Gray-Scott class
├── cell.py              # Cell state
├── vm.py                # CellVM
├── hxbc.py              # .helixc binary container codec (encode/decode Program,
│                        #   optional Chunk + source, checksums; see §8.4)
└── cli.py               # python -m helixlang <file>
```

### 8.2 Development Order (per Nystrom)

1. **VM First**: implement `bytecode.py` + `vm.py` + `cell.py` first, hand-build a chunk, and get a minimal example running.
2. **Compiler**: implement `codon_table.py` + `compiler.py`, able to generate a chunk from a list of codons.
3. **Frontend**: implement `lexer.py` + `parser.py` + `semantic.py`.
4. **Disassembler**: `disassembler.py`.
5. **GRN + morphology layer**: `grn.py` + `lsystem.py` + `reaction_diffusion.py`, wired into the VM.
6. **CLI + tests**: `cli.py` + `tests/`.

### 8.3 Performance Budget

- No compile-time optimization in the prototype phase (following CPython/Lua).
- VM dispatch uses `match/case` to avoid dict-lookup overhead.
- The reaction-diffusion field uses the `array` module or plain list-of-list (a grid smaller than 64×64 is sufficient for the prototype).
- Large-scale simulation can later migrate to numpy + MLIR/LLVM (see the extension roadmap in [05-prototype-plan.md](./05-prototype-plan.md)).

### 8.4 Binary Artifact (`.helixc`) Serialization

`--compile` serializes the compiled program into a `.helixc` container:
a versioned header + `PROG` (typed `Program` AST records) + optional `CHNK`
(precompiled `Chunk`) + optional `SRC` (original source) + a checksummed
trailer. The classic backend loads the `Chunk` directly; the sim backends and
the decompiler use the `Program`. Design decisions:

- **`Program` is authoritative**; `CHNK` is a derived cache verified and
  rebuilt from `PROG` on any mismatch (stale-cache self-healing).
- **Typed, bounds-checked encoding** (never `pickle`/`marshal`): 1-byte tags,
  length-prefixed strings, big-endian integers, IEEE-754 doubles, maps
  serialized in sorted key order for deterministic byte output.
- **Decompiler round-trip invariants**: `parse(decompile(p)) ≡ p`; canonical
  source round-trips byte-for-byte; an embedded `SRC` decompiles byte-for-byte
  regardless of source formatting.
- The container never executes code on load and rejects unknown versions with
  `BinaryVersionError`.

Full layout, record tags, CLI surface, debug/test behavior, and test matrix:
`doc/11-helixc-binary-format.md`.

---

## 9. Key Differences from clox

| Dimension | clox | HelixLang |
|---|---|---|
| Source language | lox, Lisp/JS-like | DNA + annotations |
| Lexical unit | Character tokens | Codon triplets |
| Instruction set source | Designer-defined | 64-codon table + degeneracy |
| Control flow | if/while/for compiled to jumps | GRN data-driven scheduling (no explicit jumps) |
| Functions | Explicit def / call | Gene ORF = function; promoter threshold = call condition |
| Global state | Global variable table | Cell state object (protein pool/energy/position/morphology field) |
| Output | print | Morphology PNG / behavior log / protein-concentration trajectories |

Core borrowings: chunk structure, value stack, frame stack, the `run()` dispatch loop, and the disassembler.
