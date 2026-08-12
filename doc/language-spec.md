# HelixLang Language Specification

> This document is the authoritative specification of the HelixLang programming language.
> It defines the alphabet, lexing rules, annotation syntax, the codon table (instruction
> set), bytecode format, runtime semantics, type system, configurable translation tables,
> command-line tooling, and the error model.
>
> **Grounding rule.** Everything in this document is derived from and verified against the
> reference implementation in `src/helixlang/` (lexer, parser, semantic analyzer, compiler,
> bytecode, VM, cell, GRN, reaction-diffusion, L-system). Where the document and the code
> ever disagree, the code is the source of truth.

---

## 1. Overview

A HelixLang program is **DNA that runs**. It mixes two kinds of top-level elements:

1. **DNA sequence blocks** — contiguous runs of `A`/`C`/`G`/`T`, split into triplets
   (**codons**). Codons are the machine instructions of the language: each codon maps to a
   bytecode opcode, and the *third base* (the wobble position) acts as the instruction's
   operand.
2. **Annotation blocks** — structured, line-oriented metadata introduced by `#`, describing
   genes, promoters, regulatory edges, L-systems, reaction-diffusion fields, runtime
   configuration, type annotations, and high-level "bio instructions" (CRISPR, evolution,
   epigenetics, transcription, translation, quorum sensing).

The compile pipeline is:

```
source (.helix) ──Lexer──> tokens ──Parser──> AST ──SemanticAnalyzer──> checked AST
   ──Compiler──> bytecode Chunk ──CellVM──> simulation trace (snapshots)
```

---

## 2. Source Form and Lexing

### 2.1 Lexical Tokens

| Token | Form | Meaning |
|---|---|---|
| `BASE` | a single `[ACGTacgt]` | DNA base; exists only transiently inside the lexer |
| `CODON` | exactly 3 bases, e.g. `ATG` | the minimal semantic unit (one instruction) |
| `ANNOT_START` | `#` + identifier, e.g. `#gene` | begins an annotation block |
| `ANNOT_END` | `#end` | ends an annotation block |
| `FIELD` | `ident=value` | a key/value annotation field |
| `ARROW` | `ident->ident` | a regulatory relationship (`#regulate`) |
| `NEWLINE` | `\n` | line separator inside annotation blocks |
| `EOF` | — | end of input |

**Case sensitivity.** Annotation identifiers, field names, and field values are
**case-sensitive** and passed through verbatim (annotation names are lower-cased, e.g.
`#GENE` lexes as `#gene`). DNA bases are **case-insensitive** and normalized to uppercase.

**Whitespace.** Inside DNA blocks, all whitespace and newlines are ignored, so a gene body
may be wrapped across lines for readability. Inside annotation blocks, line structure is
preserved (each line carries its own fields).

### 2.2 DNA Blocks

- A DNA block is a maximal run of `[ACGTacgt]` characters.
- Its length **must be a multiple of 3**; otherwise a `LexError` is raised
  ("DNA length not multiple of 3").
- Bases are grouped left-to-right into codons. Codons are numbered globally across the file
  (`codon_index`, 0-based); every diagnostic reports this index.
- A DNA block that appears *outside* any `#gene` block is wrapped as an **anonymous gene**
  named `__anon_<N>`.

### 2.3 Annotation Blocks and Comments

A `#` followed by whitespace, `#`, a newline, or end-of-file introduces a **line comment**
(the rest of the line is skipped).

A `#` followed by an identifier starts an annotation block:

```
#<kind> <field>=<value> <field2>=<value2> ...        (one or more fields, same line)
#end                                                   (optional terminator)
```

`#end` is matched case-insensitively. `#end` is *optional* for most annotation kinds — a
block implicitly ends at the next `#` annotation or at end-of-file. `#gene` is the
exception: it is the only block that consumes a trailing DNA body, and its `#end` is the
conventional terminator.

Field value forms:

| Form | Example | Meaning |
|---|---|---|
| number | `strength=0.8` | float/int literal |
| identifier | `cas=SpCas9` | bare identifier |
| string | `new_sequence="GGGG"` | double-quoted string (quotes stripped by the parser) |
| arrow | `lacI -> p_lac` | relationship (only in `#regulate`) |

### 2.4 Complete Grammar (EBNF)

```ebnf
program            := element*

element            := dna_block | annot_block

dna_block          := CODON+                    (* contiguous codon stream *)

annot_block        := "#" IDENT field* [ "#end" | NEWLINE ]

field              := IDENT "=" value
                    | IDENT "->" IDENT          (* regulatory relationship *)
                    | IDENT                      (* bare flag, value defaults to "" *)

value              := NUMBER | IDENT | STRING
```

---

## 3. Annotations

### 3.1 `#gene` — open reading frame definition

```
#gene name=<name> [promoter=<promoter_name>] [call_target=<gene_name>]
<CODON+>              (* DNA body, from ATG to a STOP codon, inclusive *)
#end
```

- `name` (required): the gene's symbol; must be unique.
- `promoter` (optional): the promoter that drives this gene. Without one, the gene is
  **constitutive** (active from tick 0).
- `call_target` (optional): overrides the target resolution of `OP_CALL_GENE` inside this
  gene's ORF. Because a codon operand is only a 2-bit wobble value (0–3), it can directly
  address at most the first 4 genes; `call_target` names an arbitrary gene and breaks that
  limit (see §6.4, `OP_CALL_GENE`).
- The **ORF** (open reading frame) is the subsequence from the *first* `ATG` to the *first*
  stop codon (`TAA`/`TAG`/`TGA` under the standard table), inclusive. It is validated by the
  semantic analyzer: the ORF must start with `ATG` and end with a stop codon, and must be
  non-empty.

### 3.2 `#promoter` — expression threshold

```
#promoter name=<name> strength=<float>
```

- `name` and `strength` are both required.
- `strength` is a float. **Negative strength marks the promoter constitutive** (its GRN node
  starts at level 1.0); positive strength is the activation threshold (§6.3).
- `strength` is also clamped to `[0, 1]` when used as the promoter-strength parameter of the
  central-dogma pipeline (§6.6).

### 3.3 `#regulate` — regulatory edge

```
#regulate <source> -> <target> [strength=<float>]
```

- Declares one directed regulatory edge in the gene regulatory network (GRN).
- `strength` defaults to `0.5` if omitted; `> 0` activates the target, `< 0` inhibits it.
- The source and target must be defined symbols (gene or promoter); otherwise a
  `RegulationError` is raised. A regulation **cycle** produces a warning, not an error.

### 3.4 `#lsystem` — L-system declaration

```
#lsystem name=<name> axiom=<string> rules=<spec> angle=<float> step=<float>
```

- `name` (default `"default"`), `axiom` (default `"F"`), `angle` (default `25`),
  `step` (default `1.0`).
- `rules` is a `;`-separated list of rule-set entries:
  `0:F->F[+F]F[-F]F;1:F->FF`, where each entry is `<set>:<sym>-><prod>,<sym>-><prod>,...`.
- Symbols are characters: `F` = draw a segment, `+`/`-` = turn by `angle`,
  `[`/`]` = push/pop turtle state.
- The VM executes rule set `0` (see `OP_GROW_LSYSTEM`).

### 3.5 `#field` — reaction-diffusion field

```
#field size=<int> F=<float> k=<float> Du=<float> Dv=<float>
```

- Declares the Gray-Scott reaction-diffusion field (morphogen grid).
- Defaults: `size=32`, `F=0.035`, `k=0.065`, `Du=0.16`, `Dv=0.08` (Pearson 1993 spots
  presets). Only one field may be declared per program.
- The field has two concentration channels, **U** (substrate) and **V** (morphogen/signal),
  seeded with a central perturbation plus 20 random points. `OP_SIGNAL` and
  `OP_EMIT_MORPHOGEN` inject into V; `OP_DIFFUSE` and `OP_REACT` advance it.

### 3.6 `#config` — runtime configuration

```
#config ticks=<n> output=<stdout|png|csv|none> table=<name> \
        ops_per_tick=<n> react_steps=<n> use_central_dogma=<bool> species=<name>
```

| Field | Default | Meaning |
|---|---|---|
| `ticks` | `100` | number of simulation ticks |
| `output` | `stdout` | comma-separated list of `stdout`, `png`, `csv`, `none` |
| `table` | `standard` | translation table / ISA (§8) |
| `ops_per_tick` | `64` | bytecode op budget per tick |
| `react_steps` | `1` | field steps performed by one `OP_REACT` |
| `use_central_dogma` | `false` | switch to the central-dogma pipeline (§6.6) |
| `species` | `ecoli` | species context (`ecoli` / `yeast` / `human`) |

### 3.6.1 Unit system (always on)

HelixLang runs on physical units end-to-end (no `#config units=` switch; the
legacy gameplay-unit catalog was removed). Energy counts are **ATP molecules**,
the signal field is in **µM**, diffusion is a physical **µm²/s** coefficient,
and one tick is **one minute** (`helixlang.units`; see `doc/simulation-model.md` §6.3).

| Quantity | Default | Physical meaning |
|---|---|---|
| 1 tick | — | 1 minute (Neidhardt 1996) |
| cell energy budget | `1e9` | ~10⁹ ATP molecules in a newborn cell (Orth 2010) |
| division threshold | `1.8e9` | reachable in ~20 rich-medium ticks at +4×10⁷ ATP/tick net intake |
| metabolic cost / intake | `1e7` / `5e7` | maintenance flux ~2.5×10⁷ ATP/min (Orth 2010) |
| quorum threshold | `10.0` | 10 µM AI-2 (Xavier & Bassler 2003) |
| diffusion | `100.0` | 100 µm²/s (Miller & Bassler 2001); converted to the on-lattice form (D ≈ 60) at the declared 10 µm lattice edge via stable sub-steps |
| GRN decay | `≈0.994` | median protein half-life 110 min ⇒ decay ≈ 0.994/tick (Mosteller 1980, Helbig 2011) |

### 3.7 `#type` — type annotation

```
#type <symbol>=<Type> [<symbol>=<Type> ...]
```

- Associates a biological type name with a defined symbol. Example:
  `#type target_protein=Protein`.
- When type checking is enabled, the parser verifies that every annotated symbol is defined.

### 3.8 Bio instructions

Bio instructions are single-line annotations routed at runtime to their corresponding
biological modules. Each requires a `target=<symbol>` field. They are processed every tick
by the VM (§6.5 / §6.6).

| Annotation | Fields | Runtime effect |
|---|---|---|
| `#crispr` | `target`, `position` (0), `new_sequence` (""), `cas` (`SpCas9`), `repair` | Performs CRISPR-Cas editing on the target gene's DNA (spacer design → cut → NHEJ/HDR repair); records the edit result |
| `#evolve` | `target`, `mutation_rate` (0.01), `indel_rate` (= mutation_rate×0.1) | Evolves the target gene's DNA by one generation of point/indel mutations; appends to evolution history |
| `#methylate` | `target`, `methylase` (`dam`) | Methylates the target DNA (Dam `GATC` / CpG sites); methylation represses expression by up to ~70% (Bird 2002) |
| `#histone` | `target`, `mark` (`H3K4me3`) | Applies a histone mark; positive-score marks activate, negative repress (e.g. `H3K4me3` +0.5, `H3K27me3` −0.7) |
| `#transcribe` | `target` | Forces transcription: sets the target gene's GRN level to 1.0 |
| `#translate` | `target` | Forces translation: adds 1.0 unit of the target gene's protein |
| `#quorum` | `target`, `threshold` (10.0 µM), `activate` (= `target`) | Quorum sensing: if the local V-channel signal ≥ `threshold`, sets `activate`'s GRN level to 1.0 |

---

## 4. Codon Table (Instruction Set)

### 4.1 Design Principles

- **64 codons = 6-bit instruction space**, folded into ~30 logical opcodes.
- **Degeneracy as aliasing**: synonymous codons map to the same opcode, mirroring the
  genetic code.
- **Wobble position = operand gear**: the third base of every codon encodes a 2-bit operand
  `0/1/2/3` (`A=0, C=1, G=2, T=3`) that parameterizes the instruction. In the compiled
  bytecode this value becomes the instruction's operand byte.
- **Start/stop**: `ATG` → `OP_START`; `TAA`/`TAG`/`TGA` → `OP_HALT`.
- **Switchable translation tables** (§8) change which opcode a codon decodes to — the same
  DNA behaves differently under a different "cellular environment".

### 4.2 Complete Standard Codon Table (NCBI table 1)

`wobble` is the third-base operand value. `aliases` lists every codon in the family.

| Amino acid | Opcode | Codons (aliases) | # | Operand (`wobble`) meaning |
|---|---|---|---|---|
| Met (start) | `OP_START` | `ATG` | 1 | — (function entry marker) |
| Stop | `OP_HALT` | `TAA` `TAG` `TGA` | 3 | — (terminate the ORF) |
| Phe | `OP_PUSH_CONST` | `TTT` `TTC` | 2 | constant-pool index gear |
| Leu | `OP_GROW_LSYSTEM` | `CTT` `CTC` `CTA` `CTG` `TTA` `TTG` | 6 | rule-set index (currently ignored) |
| Ile | `OP_READ_MEM` | `ATT` `ATC` `ATA` | 3 | memory slot to read |
| Val | `OP_MOVE` | `GTT` `GTC` `GTA` `GTG` | 4 | direction: 0=N, 1=E, 2=S, 3=W |
| Ser | `OP_SIGNAL` | `TCT` `TCC` `TCA` `TCG` `AGT` `AGC` | 6 | signal channel 0..3 |
| Pro | `OP_MODIFY_STATE` | `CCT` `CCC` `CCA` `CCG` | 4 | state field 0..3 |
| Thr | `OP_DIFFUSE` | `ACT` `ACC` `ACA` `ACG` | 4 | direction (ignored) |
| Ala | `OP_BUILD_PROTEIN` | `GCT` `GCC` `GCA` `GCG` | 4 | protein kind 0..3 |
| Tyr | `OP_WRITE_MEM` | `TAT` `TAC` | 2 | memory slot to write |
| His | `OP_REGULATE` | `CAT` `CAC` | 2 | mode: low nibble = target gene index, bit 7 = sign |
| Gln | `OP_EMIT_MORPHOGEN` | `CAA` `CAG` | 2 | morphogen ID (scales the injected amount) |
| Asn | `OP_DIVIDE` | `AAT` `AAC` | 2 | division mode (ignored) |
| Lys | `OP_DIE` | `AAA` `AAG` | 2 | death mode (ignored) |
| Asp | `OP_REACT` | `GAT` `GAC` | 2 | reaction type (ignored) |
| Glu | `OP_FEED` | `GAA` `GAG` | 2 | energy source (ignored) |
| Cys | `OP_BIND` | `TGT` `TGC` | 2 | binding site (target gene index) |
| Trp | `OP_BUILD_PIGMENT` | `TGG` | 1 | — |
| Arg | `OP_CALL_GENE` | `CGT` `CGC` `CGA` `CGG` `AGA` `AGG` | 6 | gene index 0..3 (see §6.4) |
| Gly | `OP_BUILD_MEMBRANE` | `GGT` `GGC` `GGA` `GGG` | 4 | target membrane permeability 0..3 |

**Note on Phe/Leu.** `TTA`/`TTG` belong to both families biologically ("dual identity").
HelixLang assigns them to **Leu** (`OP_GROW_LSYSTEM`), consistent with the NCBI standard
table; the Phe family therefore has only `TTT`/`TTC`. This assignment follows the selected
translation table and can be changed with a table switch.

---

## 5. Bytecode Format

### 5.1 Encoding

- Every instruction is **1 opcode byte**, optionally followed by **operand bytes**.
- Operand arity is fixed per opcode (see §5.2). All operands are unsigned bytes; 16-bit
  operands are big-endian (`hi`, `lo`).
- `OP_PUSH_CONST`'s operand is a **constant-pool index** (§5.3), not the wobble value: the
  compiler stores the wobble value as a constant and emits its index.
- `OP_CALL_GENE`'s operand is a **u16 gene offset**, back-patched by the compiler to the
  target gene's code address.
- `OP_JUMP` / `OP_JUMP_IF_ZERO` carry **relative** u16 offsets (added to the instruction
  pointer after the operand is read); the compiler emits inter-gene `OP_JUMP` barriers so an
  ORF can never fall through into the next gene once its op quota is exhausted.
- The chunk also carries `lines` and `codon_indices` metadata (per byte) for diagnostics,
  plus a `gene_offsets` table mapping gene name → code offset.

### 5.2 Complete Opcode Value Table

Groups follow the byte layout: control/flow, stack, synthesis, behavior, morphology,
memory/regulation, calls, VM-level control flow, arithmetic, and system.

| Opcode | Hex | Operand bytes | Encoding of the operand |
|---|---|---|---|
| `OP_START` | `0x10` | 0 | — (ORF entry marker) |
| `OP_HALT` | `0x11` | 0 | — (pop the current frame) |
| `OP_RETURN` | `0x12` | 0 | — (alias of `OP_HALT`) |
| `OP_NOP` | `0x13` | 0 | — |
| `OP_PUSH_CONST` | `0x20` | 1 | constant-pool index |
| `OP_POP` | `0x21` | 0 | — |
| `OP_DUP` | `0x22` | 0 | — |
| `OP_SWAP` | `0x23` | 0 | — |
| `OP_BUILD_PROTEIN` | `0x30` | 1 | protein kind |
| `OP_BUILD_MEMBRANE` | `0x31` | 1 | membrane permeability |
| `OP_BUILD_PIGMENT` | `0x32` | 0 | — |
| `OP_MOVE` | `0x40` | 1 | direction (0=N, 1=E, 2=S, 3=W) |
| `OP_SIGNAL` | `0x41` | 1 | signal channel |
| `OP_DIVIDE` | `0x42` | 1 | mode (currently ignored) |
| `OP_DIE` | `0x43` | 1 | mode (currently ignored) |
| `OP_FEED` | `0x44` | 1 | source (currently ignored) |
| `OP_GROW_LSYSTEM` | `0x50` | 1 | rule-set index (currently ignored) |
| `OP_DIFFUSE` | `0x51` | 1 | direction (currently ignored) |
| `OP_REACT` | `0x52` | 1 | reaction type (currently ignored) |
| `OP_EMIT_MORPHOGEN` | `0x53` | 1 | morphogen ID |
| `OP_READ_MEM` | `0x60` | 1 | slot index |
| `OP_WRITE_MEM` | `0x61` | 1 | slot index |
| `OP_MODIFY_STATE` | `0x62` | 1 | state field 0..3 |
| `OP_REGULATE` | `0x63` | 1 | mode (target nibble + sign bit) |
| `OP_BIND` | `0x64` | 1 | binding site |
| `OP_CALL_GENE` | `0x70` | 2 | u16 gene code offset (compiler back-patched) |
| `OP_JUMP` | `0x80` | 2 | u16 relative offset |
| `OP_JUMP_IF_ZERO` | `0x81` | 2 | u16 relative offset |
| `OP_ADD` | `0x90` | 0 | — |
| `OP_SUB` | `0x91` | 0 | — |
| `OP_MUL` | `0x92` | 0 | — |
| `OP_LT` | `0x93` | 0 | — |
| `OP_NOT` | `0x94` | 0 | — |
| `OP_TICK` | `0xF0` | 0 | — (tick boundary marker; no-op) |
| `OP_DEBUG` | `0xFE` | 0 | — (print the cell dump) |

The opcodes `OP_START`, `OP_HALT`, `OP_RETURN`, `OP_NOP`, `OP_TICK`, `OP_DEBUG` and the
arithmetic/jump opcodes are **not produced by codons**; they are compiler-generated markers,
system calls, or only reachable via direct bytecode construction.

### 5.3 Constant Pool

The chunk's constant pool stores:

- every `OP_PUSH_CONST` operand (the wobble value `0..3`) at its pool index;
- L-system declarations, as tuples: `("lsystem_axiom", name, axiom)`,
  `("lsystem_rules", name, rules)`, `("lsystem_angle", name, angle)`,
  `("lsystem_step", name, step)`;
- gene names for disassembly/debugging: `("gene_name", name)`.

Constants are deduplicated by value.

### 5.4 Disassembly

`--disassemble` prints the gene offset table, the code (one instruction per line with the
opcode name, hex operands, and source `codon #N line L` location), and the constant pool.
Unknown opcode bytes are printed as `<unknown 0xNN>`.

---

## 6. Runtime Semantics

### 6.1 Machine Model

The runtime (`CellVM`) is a stack-based bytecode machine combined with a cell simulator:

| Component | Type | Purpose |
|---|---|---|
| `chunk` | `Chunk` | the compiled bytecode, constants, gene offsets |
| `ip` | `int` | instruction pointer |
| `stack` | `list` | operand stack (dynamic values) |
| `frames` | `list[Frame]` | call-frame stack; a frame holds a return `ip` and a `gene_name` |
| `cell` | `Cell` | the simulated cell: position, energy, proteins, memory slots, color, age, membrane permeability, morphology points |
| `grn` | `GRN` | gene regulatory network: nodes + weighted edges |
| `lsystems` | `dict[str, LSystem]` | declared L-systems (rule set 0 active) |
| `field` | `GrayScott \| None` | the reaction-diffusion field, if declared |

**Cell state.** The `Cell` tracks `name`, position `(x, y)`, float `energy` (ATP
molecules, initial ~10⁹),
`proteins: dict[int|str, float]`, 256 memory `slots`, `alive`, `color` (RGB), `age`,
`divisions`, and `membrane_permeability` (0 = impermeable … 255 = fully permeable,
default 255). Energy is in physical ATP counts (see `cell.py` module note).

**Frame depth.** Frames are capped at 256 to prevent unbounded accumulation from the GRN
pushing frames across ticks; when the cap is exceeded pending frames are cleared and
execution restarts at the next tick.

### 6.2 Tick Loop (classic GRN path)

For each of the `#config ticks`:

1. **GRN update**: `grn.step()` recomputes every node's expression level (§6.3) and returns
   the genes whose level is `> 0.5`.
2. **Trigger expression**: each triggered gene pushes a frame and jumps to its ORF offset.
3. **Bytecode execution**: the VM executes instructions until the frames are empty or the
   per-tick op quota (`ops_per_tick`, default 64) is exhausted. Unconsumed quota may resume
   the next tick.
4. **Morphology**: L-system growth and reaction-diffusion effects have already been applied
   inline by their opcodes during step 3.
5. **Feedback**: the local V-channel concentration is fed into the `pigment` gene's level
   (`level += v × 0.1`, clamped to 1.0) when a `pigment` node exists.
6. **Snapshot**: one trace record is appended (§6.7).

### 6.3 GRN Activation Model

For each node, per tick:

```
inputs  = Σ over edges targeting the node of (edge.weight × source.level)
raw     = hill(inputs, n, kd)        if the node has Hill kinetics
        = sigmoid(inputs − threshold) otherwise            (legacy)
level'  = clamp(decay × level + (1 − decay) × raw)  to [0, 1]
trigger ⇔ level' > 0.5
```

- `sigmoid(x) = 1/(1+e^(−x))` (numerically stable), `hill(x,n,kd) = x^n/(kd^n+x^n)`
  (0 for non-positive input, half-max at `x = kd`).
- `decay` defaults to `GRN.DECAY = decay_from_half_life_ticks(110)` ≈ 0.994 (E. coli
  median protein half-life ≈ 110 min), or is derived from a measured protein
  half-life via `decay = 0.5^(1/half_life_ticks)`.
- A gene with no promoter, or whose promoter has **negative strength**, is constitutive
  (initial level 1.0, threshold −1). A promoted gene starts at level 0.0 with
  `threshold = promoter.strength`.

### 6.4 Opcode Semantics (reference behavior)

Operands below are read as `u8` (or `u16` where noted). "wobble" values are `0..3`.

| Opcode | Behavior |
|---|---|
| `OP_START` | No-op; marks an ORF entry. |
| `OP_NOP` | No-op. |
| `OP_TICK` | No-op; tick boundary marker. |
| `OP_HALT`, `OP_RETURN` | If the frame stack is non-empty, pop it and continue at the frame's return `ip`; otherwise the current execution pass ends. |
| `OP_PUSH_CONST <idx>` | Push `constants[idx]`; if `idx` is out of range, push `idx` itself. |
| `OP_POP` | Pop and discard the top of the stack (if any). |
| `OP_DUP` | Duplicate the top of the stack (if any). |
| `OP_SWAP` | Swap the top two stack entries (if present). |
| `OP_BUILD_PROTEIN <kind>` | Synthesize 1 unit of protein `kind`: `cell.add_protein(kind)`. |
| `OP_BUILD_MEMBRANE <perm>` | Set the cell's membrane permeability (clamped to 0–255). Lower permeability proportionally scales nutrient intake: `feed` gains `round(amount × perm/255)`. |
| `OP_BUILD_PIGMENT` | Set the cell color to pigment red `(200, 50, 50)` (mitochondrial-table Trp). |
| `OP_MOVE <dir>` | Move one step in direction `dir % 4` (0=N, 1=E, 2=S, 3=W, von Neumann grid), costing 1 energy when energy > 0. |
| `OP_SIGNAL <ch>` | Release a quorum autoinducer into the field's V channel at the cell position: `emit(min(1.0, 0.25 × (1+ch)))`. Every emission is counted (`signal_emissions`), even without a field. Grounding: AI-2 quorum signaling (Miller & Bassler 2001; Xavier & Bassler 2003). |
| `OP_DIVIDE <mode>` | Symmetric division: if `energy ≥ 2`, halve energy (`energy //= 2`) and increment `divisions`. The operand is reserved (mode). |
| `OP_DIE <mode>` | Cell death: `alive = False`. Operand reserved (mode). |
| `OP_FEED <src>` | Nutrient intake: `feed(10)` scaled by membrane permeability. Operand reserved (source). |
| `OP_GROW_LSYSTEM <rules>` | One L-system iteration on the first declared L-system; append the new morphology points to the cell. Operand reserved (rule-set index; rule set 0 is used). |
| `OP_DIFFUSE <dir>` | Advance the reaction-diffusion field by one step (if a field exists). Operand reserved (direction). |
| `OP_REACT <type>` | Advance the field by `react_steps` steps (if a field exists). Operand reserved (reaction type). |
| `OP_EMIT_MORPHOGEN <id>` | Inject `(id+1)/256` into the field's V channel at the cell position (if a field exists); `id=0` keeps a non-zero legacy emission. Grounding: Turing 1952 morphogens; Pearson 1993 presets. |
| `OP_READ_MEM <slot>` | Push `slots[slot]` onto the stack. |
| `OP_WRITE_MEM <slot>` | Pop the stack and store into `slots[slot]` (if the stack is non-empty). |
| `OP_MODIFY_STATE <f>` | `f=0` → color green `(100,200,50)`; `f=1` → `age += 1`; `f=2` → color yellow `(200,200,50)`; `f=3` → color magenta `(200,50,200)`. |
| `OP_REGULATE <mode>` | Dynamic regulatory rewiring: source = the currently executing gene (the top frame's `gene_name`; falls back to the first GRN node). Target = `genes[(mode & 0x0F) % len(genes)]`. Weight = `+1.0` if bit 7 is clear (activate), `−1.0` if set (inhibit). The edge is added, or its weight is updated in place. Each event is recorded in `_regulation_events`. Grounding: Jacob & Monod 1961; Ptashne 2004. |
| `OP_BIND <site>` | Protein–DNA binding: the executing gene's transcription factor (its name key, else the first available protein) is consumed (1 unit). If consumed, the target `genes[site % len(genes)]`'s expression level is boosted by `+0.5`. Protein-limited: no TF available ⇒ no binding. Recorded in `_binding_events`. Grounding: Berg & von Hippel 1987; McClure 1985. |
| `OP_CALL_GENE <off:u16>` | Push a frame (return `ip`, name `<call>`) and jump to the gene code offset. The compiler back-patches the offset from the wobble value (`names[wobble % len(names)]`), or from the owning gene's `call_target` field when present; an undefined target is a compile error. |
| `OP_JUMP <off:u16>` | `ip += off` (relative). |
| `OP_JUMP_IF_ZERO <off:u16>` | Pop a value (0 if the stack is empty); if it is falsy, `ip += off`. |
| `OP_ADD` / `OP_SUB` / `OP_MUL` | Pop `b`, pop `a`, push `a (op) b` (if the stack has ≥ 2 entries). |
| `OP_LT` | Pop `b`, pop `a`, push `1` if `a < b` else `0`. |
| `OP_NOT` | Pop `a`, push `0` if truthy else `1`. |
| `OP_DEBUG` | Print `cell.dump()` to stdout. |

**Unknown opcode bytes** are skipped one byte at a time (in debug mode a diagnostic line is
printed); the dispatcher falls back to skipping `OP_OPERAND_BYTES` bytes for known-but-
unimplemented opcodes, bounded by the chunk length.

### 6.5 Bio Instruction Processing

Bio instructions (§3.8) are dispatched **only in the central-dogma path** (see §6.6). In
the classic GRN path (`use_central_dogma=false`) they are parsed and validated but not
executed by the tick loop — expression is driven entirely by the GRN there.

### 6.6 Central Dogma Pipeline

When `use_central_dogma=true`, each tick runs:

1. Process all bio instructions (`#crispr`, `#evolve`, `#methylate`, `#histone`,
   `#transcribe`, `#translate`, `#quorum`).
2. **Transcription–translation**: `grn.step()`; then per gene — effective promoter strength
   `= clamp(promoter_strength × chromatin_modifier)`, transcription-factor fold changes from
   regulatory edges, transcription of the gene's DNA, translation with the species tRNA
   abundance table, mRNA level from a kinetic steady-state model, and protein accumulation
   (`proteins[gene] ∝ mrna_level × ribosome_density`), which feeds back into the GRN level.
3. Morphology update and field→GRN feedback.
4. Snapshot.

The `species` config selects the tRNA abundance and codon-usage context
(`ecoli` / `yeast` / `human`).

### 6.7 Event Observability

The VM keeps runtime event logs exposed via snapshots and the Python API:

| Snapshot key | Meaning |
|---|---|
| `tick` | current tick |
| `x`, `y` | cell position |
| `energy` | cell energy |
| `alive` | cell liveness |
| `proteins` | copy of the protein pool |
| `color` | RGB color |
| `gene_levels` | `{gene: level}` |
| `morphology_points_count` | number of L-system morphology points |
| `membrane_permeability` | 0–255 |
| `signal_emissions` | total `OP_SIGNAL` emissions |
| `regulation_edges` | current GRN edge count |
| `binding_events` | total `OP_BIND` events |
| `field_total_v` | sum of V-channel concentrations (0 if no field) |

---

## 7. Type System

HelixLang is a **dynamically typed** stack language at runtime, with an optional static
annotation layer.

### 7.1 Molecular types

| Type | Meaning | Source |
|---|---|---|
| `Protein` | protein with name + concentration | `HelixType.PROTEIN` |
| `Signal` | signal molecule with name + strength | `HelixType.SIGNAL` |
| `Float` / `Int` / `Bool` | numeric / logical values | `FLOAT` / `INT` / `BOOL` |
| `String` | text | `STRING` |
| `Gene` | gene reference | `GENE` |
| `Record` | record type with named fields | `RECORD` |
| `Any` | unknown / unconstrained | `ANY` |

### 7.2 Static checking

- `#type name=<Type>` annotates symbols (genes/promoters).
- When type checking is enabled (`Parser(enable_type_check=True)`), the analyzer verifies
  that: annotated symbols exist, `#regulate` source/target symbols exist, and bio-instruction
  targets exist.
- The module system (`Module` / `ModuleLoader`) treats each `.helix` file as a module
  exporting its promoters (typed `Protein`) and genes (typed `Gene`).

---

## 8. Translation Tables (ISA Variants)

The mapping codon → opcode is **switchable**, selected by `#config table=...` or the CLI
`--table` flag. The same DNA therefore expresses differently under different "cellular
environments".

| Table | NCBI code | Differences from the standard table |
|---|---|---|
| `standard` | table 1 | Default. `ATG`→`OP_START`, `TAA`/`TAG`/`TGA`→`OP_HALT`. |
| `mito_vertebrate` | table 2 | `TGA`→`OP_BUILD_PIGMENT` (Trp), `ATA`→`OP_START` (Met), `AGA`/`AGG`→`OP_HALT` (Stop). |
| `ciliate` | table 6 | `TAA`/`TAG`→`OP_EMIT_MORPHOGEN` (Gln). |

The stop-codon set used by the parser to delimit ORFs is derived from the selected table
(codons mapping to `OP_HALT`).

---

## 9. Command-Line Interface and Tooling

```
helixlang <source.helix> [--table=standard|mito_vertebrate|ciliate]
                         [--disassemble] [--debug] [--csv] [--png PREFIX]
                         [--ticks N]
helixlang --serve [--host 127.0.0.1] [--port 5000]
helixlang --encode-dna <goldman|erlich> <source.helix> [--pcr-cycles N]
helixlang --decode-dna <file.fasta>
```

| Flag | Effect |
|---|---|
| `--table NAME` | select the translation table (default `standard`) |
| `--disassemble` | print the bytecode disassembly and exit |
| `--debug` | trace every executed instruction with the stack contents |
| `--csv` | emit the trace as CSV to stdout |
| `--png PREFIX` | write the morphology/field image as a PPM file |
| `--ticks N` | override `#config ticks` |
| `--serve` | start the web visualization server (Flask) |
| `--host` / `--port` | web server bind address |
| `--encode-dna SCHEME` | encode the source file to DNA oligos (Goldman or Erlich codec), output FASTA |
| `--decode-dna FILE` | decode a FASTA DNA file back to `.helix` source |
| `--pcr-cycles N` | inject simulated PCR errors into encoded oligos (0 = none) |

---

## 10. Errors and Diagnostics

All errors derive from `HelixError` and carry a source position (`line`, `col`) and, for
codon-level errors, a global `codon_index`.

| Exception | Trigger example | Message shape |
|---|---|---|
| `LexError` | DNA block length not a multiple of 3; unexpected character | `[LexError @ line L] DNA length N not multiple of 3` |
| `ParseError` | missing `name=`/`strength=` field; ORF without `ATG`; ORF not terminated; unknown annotation | `[ParseError @ line L] #gene 'x' has no START codon (ATG)` |
| `SemanticError` | duplicate symbol; unknown promoter reference; empty ORF; invalid config value | `[SemanticError @ line L] duplicate symbol 'x'` |
| `RegulationError` | `#regulate` references an undefined source/target | `[RegulationError @ line L] #regulate source 'x' not defined` |
| `CompileError` | unknown codon; `OP_CALL_GENE` target not defined | `[CompileError @ line L] CALL_GENE target 'x' not defined` |
| `RuntimeHelixError` | runtime failure during execution | `[RuntimeHelixError @ line L codon #N] ...` |
| `BioError` | invalid biology-module input / solver failure | `[BioError @ line L] ...` |

Semantic warnings (non-fatal): regulation cycles are reported as warnings, not errors.

---

## 11. Example Program

The following program exercises genes, promoters, regulation, an L-system, a
reaction-diffusion field, and configuration:

```
#promoter name=p_lac  strength=0.5      # regulated promoter
#promoter name=p_lacI strength=-0.5     # negative strength => constitutive

#gene name=lacZ promoter=p_lac
ATG GCT GCT GCT TAA                      # build 3 proteins, halt
#end

#gene name=lacI promoter=p_lacI
ATG GCT GCT TAA                          # build protein, halt
#end

#regulate p_lacI -> lacI strength=+0.8
#regulate lacI  -> p_lac strength=-0.9   # lacI represses the lac promoter
#regulate p_lac  -> lacZ strength=+0.9

#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25 step=1.0
#field size=32 F=0.035 k=0.065 Du=0.16 Dv=0.08

#config ticks=20 output=stdout,png
```

---

## 12. Summary of Biological Correspondences

| Biological entity | HelixLang concept |
|---|---|
| DNA sequence | source code |
| Codon | mnemonic / opcode |
| Amino acid | opcode family |
| Degenerate codons | opcode aliases (wobble base = operand) |
| Start / stop codons | `OP_START` / `OP_HALT` |
| Gene / ORF | function / bytecode block |
| Promoter | expression threshold (GRN node) |
| Operon | several genes sharing one promoter |
| Transcription factor | GRN regulatory edge |
| Gene regulatory network | the GRN (data-driven call graph) |
| Ribosome | the VM bytecode dispatcher |
| Cytoplasm | value stack + protein pool |
| Cell membrane | 256 memory slots + permeability |
| Morphogenesis | L-system + Gray-Scott reaction-diffusion |
| Quorum sensing | `OP_SIGNAL` / `#quorum` (AI-2 autoinducer pool) |
| Epigenetics | `#methylate` / `#histone` expression modifiers |
| Evolution | `#evolve` codon-substitution mutations |
| Genome editing | `#crispr` spacer design + NHEJ/HDR repair |
| Cell division / apoptosis | `OP_DIVIDE` / `OP_DIE` |
| Translation table (NCBI) | `--table` ISA variant |

---

## Appendix A. Determinism

Simulation is deterministic for a fixed seed: the GRN update, Gray-Scott field seeding, and
all runtime RNG uses are seeded (`random.Random(0)` for the VM, seed `42` for the field).
Identical input produces an identical trace.
