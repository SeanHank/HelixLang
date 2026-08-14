# HelixLang Binary Artifact Design (`.helixc`)

Status: **Draft for review** (feature not yet implemented)
Owner: compiler / tooling track
Related docs: `language-spec.md` §5 (bytecode), §9 (CLI); `compiler-design.md` §5–§8;
`engineering-design.md` §2 (module contracts).

This document is the design contract for turning `.helix` source into a compiled
binary artifact (`.helixc`) and back. It covers all three capabilities requested:

1. **Write** — compile `.helix` source into a `.helixc` binary file;
2. **Read and run** — execute a program directly from a `.helixc` binary;
3. **Debug and test** — disassemble, trace, breakpoint, and round-trip-test
   programs whose only artifact is the binary file.

---

## 1. Goals and Non-Goals

### 1.1 Goals

- **G1 (Write).** `helixlang --compile foo.helix -o foo.helixc` produces a
  single, self-contained binary file containing everything needed to run the
  program later: the parsed program (annotations + ORFs), the compiled bytecode
  chunk, the codon table it was built with, and (optionally) the original
  source text.
- **G2 (Read + run).** `helixlang foo.helixc` runs the program. The classic
  backend loads the precompiled chunk without recompiling; the sim backends
  (`population`, `fba`, `calibration`, `benchmark`, …, §6.1 of the wiring doc)
  reconstruct the full `Program` from the binary. `--backend`, `--json`,
  `--csv`, `--ticks` behave exactly as they do for `.helix` input.
- **G3 (Debug).** `helixlang foo.helixc --disassemble`, `--debug`, and the
  interactive debugger work off the binary. Instruction→codon→source-line maps
  are stored in the container, so breakpoints can be set by gene name, codon
  index, or source line.
- **G4 (Test / round-trip).** `helixlang foo.helixc --decompile -o out.helix`
  regenerates valid source. The invariant `parse(decompile(program)) ≡ program`
  is enforced by tests; when the original source was embedded, `--decompile`
  reproduces it byte-for-byte. `--compare` runs the same binary through the
  source path and the binary path and diffs the traces.
- **G5 (Safety).** Loading never executes code from the artifact. The format is
  a typed, length-checked, versioned encoding — **not** `pickle`/`marshal`.
  Malformed or unknown files fail fast with actionable `BinaryFormatError`s.
- **G6 (Determinism).** Compiling the same source with the same flags yields a
  byte-identical artifact (sorted map keys, no timestamps, fixed encoding).

### 1.2 Non-Goals

- No cross-version bytecode stability guarantee beyond what the version table
  (see §8) implements; a `.helixc` is tied to the table it was built with.
- No obfuscation, licensing, or DRM. `.helixc` is a deterministic, inspectable
  artifact (`--decompile` is a first-class feature).
- No changes to the classic chunk wire encoding itself (`language-spec.md`
  §5.1 stays byte-for-byte compatible).
- No network/distribution format (that is `--encode-dna`'s job).

---

## 2. Design Overview

`.helixc` is a **versioned container of typed sections**. Three payload
sections plus a header and trailer:

```
┌───────────────────────────────────────────────────────────────┐
│ HLXC header  (magic, format version, flags, table id)          │
├───────────────────────────────────────────────────────────────┤
│ PROG section — serialized Program AST (§4)                    │  always
│   genes, promoters, regulations, lsystems, field_decl,         │
│   morphogen_feedback, config, bio_instructions,                │
│   type_annotations, media, enzymes, pools, sim_extensions      │
├───────────────────────────────────────────────────────────────┤
│ CHNK section — compiled Chunk (§5)                            │  flag bit 0
│   code bytes, constants, lines, codon_indices, gene_offsets,   │
│   dna_sequence (optional)                                      │
├───────────────────────────────────────────────────────────────┤
│ SRC section  — original source text (UTF-8)                   │  flag bit 1
├───────────────────────────────────────────────────────────────┤
│ trailer — section index + SHA-256 checksum                    │  always
└───────────────────────────────────────────────────────────────┘
```

The `Program` AST is the **authoritative** payload: it is sufficient to
recompile (classic), to run every sim backend, and to regenerate source.
The `Chunk` is a derived, optional cache that makes classic loads and
disassembly O(read) instead of O(compile). The `SRC` section is optional and
used only for byte-for-byte `--decompile`.

### 2.1 Why the AST and not just the chunk?

A chunk-only artifact would cover the classic backend and disassembly, but it
cannot:

- drive the sim backends, which read `program.config.backend/sim` and the
  full annotation set (`sim_runtime.run(program)`);
- regenerate source (`--decompile` needs gene/promoter/regulate/media/enzyme/
  metabolite/config/sim declarations and the `fields` dicts);
- validate ORFs or re-run the semantic analyzer.

The AST record encoding (§4) is small (a 20-gene program is a few KB) and is
the single source of truth, so both worlds are served by one container.

### 2.2 Why a custom typed encoding and not `pickle`/`marshal`/`json`?

| Option | Problem for `.helixc` |
|---|---|
| `pickle` | executes arbitrary code on `load`; opaque; no cross-version story |
| `marshal` | tied to the running Python version; unsafe for untrusted input |
| `json` | portable but text; loses the compact binary goal and needs a schema anyway; floats/dict ordering must be pinned |

The container uses a **self-describing typed byte stream** (1-byte type tags,
length-prefixed strings, big-endian fixed-width integers, IEEE-754 doubles)
that any future implementation (a later Python version, or a non-Python
port) can decode from this document alone. `hxbc.py` ships a strict
decoder that bounds-checks every length against the section size.

---

## 3. File Layout

### 3.1 Header (12 bytes)

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 4 | magic | ASCII `HLXC` (`0x484C5843`) |
| 4 | 1 | format version | `1` (see §8) |
| 5 | 1 | flags | bit0 = has `CHNK`, bit1 = has `SRC`, bits 2–7 = 0 |
| 6 | 1 | table id | 0=`standard`, 1=`mito_vertebrate`, 2=`ciliate` (§3.3) |
| 7 | 1 | reserved | `0` |
| 8 | 4 | payload length | u32 big-endian, total bytes of all sections |

### 3.2 Sections

Each section is:

| Size | Field |
|---|---|
| 4 | magic, ASCII `PROG` / `CHNK` / `SRC ` / `EOF ` |
| 4 | u32 length `L` (payload bytes that follow) |
| L | payload |
| (trailer only) 32 | SHA-256 over `magic + length + payload` of the whole section |

Sections appear in fixed order `PROG`, `CHNK?`, `SRC?`, `EOF`. The `EOF`
section has length 0 and carries the checksum. Decoders reject out-of-order
or duplicate sections.

### 3.3 Table id

The codon table is part of the runtime contract (`language-spec.md` §9
`--table`). The id maps onto the `codon_table.py` `STANDARD_TABLE`,
`MITO_TABLE`, `CILIATE_TABLE` constants; the loader resolves it at
read time so a `.helixc` remains self-describing across versions.

---

## 4. `PROG` Section — Serialized Program AST

### 4.1 Primitive encodings (little-endian, fixed width)

| Tag | Meaning | Encoding |
|---|---|---|
| `0x01` | `u8` | 1 byte unsigned |
| `0x02` | `u16` | 2 bytes big-endian (matches opcode operand convention §5.2) |
| `0x03` | `u32` | 4 bytes big-endian |
| `0x04` | `f64` | 8 bytes IEEE-754 double |
| `0x05` | `str` | u32 length + UTF-8 bytes; length must be ≤ 2²⁴ |
| `0x06` | `bool` | 1 byte, `0` or `1` |
| `0x07` | `opt<str>` | `0x00` = null, else `0x05` str |
| `0x08` | `field-map` | u16 count, then `(str key, str value)` pairs in **sorted key order** |
| `0x09` | `str-list` | u16 count, then `str`s in declaration order |
| `0x0A` | `record` | u8 record tag (§4.3) followed by its fields in fixed order |
| `0x0B` | `record-list` | u16 count, then `record`s |
| `0x0C` | `int-key-map` | u16 count, then `(u8 key, field-map value)` pairs in ascending key order |

Strings never contain NUL bytes; all numeric fields are bounds-checked against
the section payload length during decode (reject negative/oversized lengths).

### 4.2 Structure of the section

```
record-list<Program fields in the order below>:
  genes                 record-list<Gene>
  promoters             record-list<Promoter>
  regulations           record-list<Regulation>
  lsystems              str-key map<str, LSystemDecl> (sorted keys)
  field_decl            opt<FieldDecl>
  morphogen_feedback    record-list<MorphogenFeedback>
  config                Config
  bio_instructions      record-list<BioInstruction>
  type_annotations      str-key map<str, str> (sorted keys)
  media                 record-list<MediaDecl>
  enzymes               record-list<EnzymeDecl>
  pools                 record-list<PoolDecl>
  sim_extensions        str-key map<str, str> (sorted keys)
```

The field **order and cardinality mirror `ast_nodes.Program` exactly**; a
record is decoded positionally, so a `Program` round-trips 1:1.

### 4.3 Record tags and field layout (one row per field, in order)

**`Codon`** (tag `0x01`)
| Field | Type | Notes |
|---|---|---|
| seq | `str` | triplet, e.g. `ATG` |
| index | `u32` | global codon counter |
| line | `u32` | 1-based source line for diagnostics |

**`Gene`** (tag `0x02`)
| Field | Type | Notes |
|---|---|---|
| name | `str` | |
| promoter | `opt<str>` | |
| codons | `record-list<Codon>` | full stream |
| orf | `record-list<Codon>` | extracted ORF (§3.3 of compiler-design) |
| fields | `field-map` | `#gene` fields verbatim (sorted on write) |

**`Promoter`** (tag `0x03`)
| Field | Type | Notes |
|---|---|---|
| name | `str` | |
| strength | `f64` | |
| fields | `field-map` | verbatim |

**`Regulation`** (tag `0x04`)
| Field | Type |
|---|---|
| source | `str` |
| target | `str` |
| strength | `f64` |

**`LSystemDecl`** (tag `0x05`)
| Field | Type | Notes |
|---|---|---|
| name | `str` | |
| axiom | `str` | |
| rules | `int-key-map` | generation → symbol→production map, ascending generation |
| angle | `f64` | |
| step | `f64` | |

**`FieldDecl`** (tag `0x06`)
| Field | Type |
|---|---|
| size | `u32` |
| F | `f64` |
| k | `f64` |
| Du | `f64` |
| Dv | `f64` |

**`MorphogenFeedback`** (tag `0x07`)
| Field | Type |
|---|---|
| gene | `str` |
| channel | `str` |
| gain | `f64` |

**`MediaDecl`** (tag `0x08`)
| Field | Type | Notes |
|---|---|---|
| nutrient | `str` | |
| concentration | `f64` | |
| diffusion_um2_s | `opt<f64>` | |

**`EnzymeDecl`** (tag `0x09`)
| Field | Type | Notes |
|---|---|---|
| gene | `str` | |
| reaction | `str` | |
| kcat | `opt<f64>` | |

**`PoolDecl`** (tag `0x0A`)
| Field | Type |
|---|---|
| name | `str` |
| init | `f64` |

**`Config`** (tag `0x0B`)
| Field | Type | Default on decode |
|---|---|---|
| ticks | `u32` | 100 |
| output | `str-list` | `["stdout"]` |
| table | `str` | `"standard"` (ignored; §3.3 table id is authoritative) |
| ops_per_tick | `u32` | 64 |
| react_steps | `u32` | 1 |
| use_central_dogma | `bool` | false |
| species | `str` | `"ecoli"` |
| backend | `str` | `"classic"` |
| sim | `field-map` | {} (sorted keys) |

**`BioInstruction`** (tag `0x0C`)
| Field | Type | Notes |
|---|---|---|
| kind | `str` | `crispr` / `evolve` / `methylate` / `histone` / … |
| target | `str` | |
| params | `field-map` | verbatim (sorted) |
| line | `u32` | |

**`Program`** (tag `0x0D`) — 13 fields in §4.2 order.

### 4.4 Decode contract

- Decode is **purely structural**; no `Program` is executed on load.
- After decode, the loader runs the same field validation the parser performs
  (e.g. every gene must have a non-empty ORF, table id valid) so a corrupted
  file cannot produce a half-valid `Program`.
- Any unknown tag / count overflow / trailing bytes → `BinaryFormatError`
  naming the section and offset.

---

## 5. `CHNK` Section — Compiled Chunk

### 5.1 Layout

| Field | Type | Notes |
|---|---|---|
| `code` | `u8-list` | chunk bytecode (`bytecode.Chunk.code`) |
| `constants` | `record-list<Constant>` | §5.2 |
| `lines` | `u32-list` | per byte, 1-based source line |
| `codon_indices` | `u32-list` | per byte, global codon index (or `0xFFFFFFFF` for compiler-generated bytes) |
| `gene_offsets` | `str→u32 map` | gene name → code offset (sorted keys) |
| `dna_sequence` | `str-list` | optional, one triplet per codon (redundant with PROG; kept for disassembly speed) |

Constants are encoded as a tag byte selecting the pool entry type
(string / float / int / wobble) plus its primitive payload — mirroring
`compiler-design.md` §5.4.

### 5.2 Relation to PROG

`CHNK` is **derived** and must never be trusted over `PROG`: on load the
loader verifies `len(chunk.code) == len(chunk.lines) == len(chunk.codon_indices)`
and that every `gene_offsets` entry names a gene present in `PROG`. A mismatch
does not fail the whole file; the loader recompiles from `PROG` (a `warn` +
rebuild path) so a stale cache is self-healing.

### 5.3 When is CHNK emitted?

`--compile` always emits it. To keep the artifact maximally portable (and
small), `--compile --no-chunk` produces a PROG-only file that recompiles on
first classic run.

---

## 6. `SRC` Section

- UTF-8 bytes of the original source file, verbatim.
- Present iff header flag bit 1 is set. `--compile --no-source` omits it.
- Used for byte-for-byte `--decompile` (§7.3) and for source-line breakpoints
  in the interactive debugger (`debugger.py` breakpoints by `line=`).

---

## 7. CLI Surface

New forms (mirroring the existing `--encode-dna`/`--decode-dna` pair):

```
helixlang --compile <source.helix> -o <out.helixc> [--no-chunk] [--no-source]
helixlang <artifact.helixc> [--table T] [--disassemble] [--debug] [--csv]
                            [--png PREFIX] [--ticks N]
helixlang <artifact.helixc> [--backend NAME] [--json]
helixlang --decompile <artifact.helixc> -o <out.helix>
helixlang --compare <source.helix> <artifact.helixc> [--json]
```

### 7.1 Dispatch rule (`cli.py`)

Input is classified by extension, not by probe:

- `.helix` → existing pipeline (parse → compile → run).
- `.helixc` → `hxbc.load_program(path)` → `LoadedArtifact` (§9). If
  `--disassemble`: disassemble the chunk (rebuild from PROG if absent). If
  `--backend`/`#config backend` selects a sim backend: `sim_runtime.run(program)`.
  Otherwise: classic VM over the chunk, falling back to recompiling from PROG
  when no chunk exists.
- `.fasta` / `.txt` → unchanged (`--decode-dna`).

`--decompile` and `--compare` are recognized **before** extension dispatch and
never run the VM.

### 7.2 Read + run equivalence

The key contract: **`helixlang foo.helix` and `helixlang foo.helixc` produce
identical output** for the same program and flags. This is enforced by the
`--compare` mode and the test matrix (§11). `--compare` runs both paths and
diffs CSV traces (or `SimResult` JSON) tick by tick.

---

## 8. Decompiler

### 8.1 Canonical source regeneration

`decompile(program: Program) -> str` reconstructs `.helix` text in this order:

1. `#gene` / `#promoter` / `#regulate` / `#lsystem` / `#field` /
   `#morphogen` / `#media` / `#enzyme` / `#metabolite` declarations in
   `Program` declaration order;
2. each `#gene`'s codons as **12 whitespace-separated triplets per line**,
   matching the existing examples' layout;
3. `#config ticks=… output=… table=… ops_per_tick=… react_steps=… \
   use_central_dogma=… species=… backend=…` followed by `config.sim` keys in
   sorted order, then any `#sim key=value` extras from `sim_extensions` that
   are not part of `#config`;
4. `#type` annotations (sorted keys) and bio instructions in declaration order.

Field quoting: a value containing `=` or whitespace is emitted quoted
(`name="..."`), matching the parser's strip-quotes rule (`parser.py` §1.2).

### 8.2 Round-trip invariants (tested)

- **R1:** `parse(decompile(p)) ≡ p` for semantic equality (same chunk, same
  config, same sim_extensions).
- **R2:** `decompile(parse(text))` is byte-identical to `text` **when the
  source was already in canonical form** (12 codons/line, sorted config keys).
- **R3:** with an embedded `SRC`, `--decompile` reproduces the original file
  byte-for-byte regardless of canonical form.

### 8.3 Determinism

All map fields are serialized with sorted keys (§4.1), so R2/R3 hold on every
platform and every Python version. This mirrors the determinism policy already
used by the DNA codecs.

---

## 9. Debug and Test from the Binary

### 9.1 Disassembly

`--disassemble` on `.helixc` prints the existing format (`compiler-design.md`
§7). Because `lines` and `codon_indices` live in `CHNK`, the
`; ATG (#0, line 5)` annotations work identically from the binary. Without a
chunk, the loader recompiles from PROG first — output is identical either way
(equivalence enforced by tests).

### 9.2 Instruction tracing and the interactive debugger

- `--debug` uses the same `DEBUG_TRACE_EXECUTION` path.
- `HelixDebugger` (`debugger.py`) is constructed with `(vm, program)`; from a
  binary we build the VM from the chunk and hand it the PROG-derived `Program`.
  Breakpoints by `line=` resolve through `lines`/`codon_indices`; by `gene=`
  through `gene_offsets`. When `SRC` is present, source-line stepping is exact.

### 9.3 Integrity / self-test

`hxbc.verify(path)` re-decodes a `.helixc` and re-runs the checksum; a
one-byte corruption anywhere in a section fails with a section-offset error.
`tests/test_helixc.py` covers corrupt magic, version, flags, truncated
sections, and checksum mismatch.

---

## 10. New Module Contract — `src/helixlang/hxbc.py`

```
# ── Write ────────────────────────────────────────────────────────
save_program(program, path, *, chunk=None, source=None) -> None
dumps_program(program, *, chunk=None, source=None) -> bytes
compile_file(src_path, out_path, *, include_chunk=True,
             include_source=True) -> ArtifactInfo   # parse+compile+write
# ── Read / run ───────────────────────────────────────────────────
load_program(path) -> LoadedArtifact               # validate + decode
loads_program(data) -> LoadedArtifact
# ── Debug / test ─────────────────────────────────────────────────
decompile(program) -> str
decompile_to_file(program, path) -> None
verify(path) -> None                                # checksum + re-decode
```

`LoadedArtifact` dataclass:

```
program: Program
chunk: Chunk | None          # precompiled (may be recompiled lazily)
source: str | None           # original text when embedded
table: str                   # resolved codon table name
```

Error surface: `BinaryFormatError(BinaryError)` carrying `section`, `offset`,
and a human message; `BinaryVersionError(BinaryFormatError)` for unknown
format versions. Neither type is ever raised during ordinary parse; the CLI
catches them and exits with rc=2 plus a `!` diagnostic, matching the existing
`--decode-dna` failure style.

Registration: `hxbc` is added to `engineering-design.md` §2's module table and
imported by `cli.py` at the same level as `bytecode`/`sim_runtime` (already
present). `hxbc` imports only stdlib (`hashlib`, `struct`, `dataclasses`) and
`helixlang.ast_nodes` / `bytecode` / `codon_table` — no new third-party deps.

---

## 11. Test Matrix (new file `tests/test_helixc.py`, plus CLI cases)

| # | Test | Asserts |
|---|---|---|
| 1 | `dumps`→`loads` round-trip on a 20-gene program | equal `Program` fields, equal chunk bytes |
| 2 | R1 `parse(decompile(p)) ≡ p` for a program using every annotation kind | semantic equality via `recompile` |
| 3 | R2 canonical source round-trip byte-identical | `decompile(parse(canonical)) == canonical` |
| 4 | R3 byte-for-byte from embedded `SRC` | `--decompile` output `==` original bytes |
| 5 | determinism | two `dumps_program` calls byte-identical |
| 6 | classic run from binary | `main(["foo.helixc"])` == `main(["foo.helix"])` (CSV) |
| 7 | sim backend from binary | `main(["foo.helixc","--backend","population","--json"])` works |
| 8 | `--compare` | two CSV traces identical |
| 9 | `--disassemble` from binary | output == source-path output |
| 10 | `--debug` from binary | trace identical to source path |
| 11 | integrity | corrupt 1 byte in PROG/CHNK → `BinaryFormatError` with offset |
| 12 | version | bump version byte → `BinaryVersionError` |
| 13 | `--compile` on every `examples/*.helix` then run | all 34 examples: compile → run rc=0 |
| 14 | stale `CHNK` mismatch | loader recompiles from PROG, warns, still correct |

Gate: `mypy src`, `ruff check src tests`, `pytest --cov=helixlang` with
`fail_under=80` — unchanged project gate (`AGENTS.md`).

---

## 12. Versioning and Compatibility

- `HLXC_FORMAT_VERSION = 1` is the only format version in v1 of the feature.
- The decoder table maps version → reader function; unknown version →
  `BinaryVersionError("artifact uses .helixc format version N; this build
  supports 1 — recompile with --compile")`.
- New AST fields are introduced by **bumping the format version**, never by
  silently reinterpreting an old layout. The reader for v1 stays in the table,
  so old artifacts keep working.

---

## 13. Performance and Size

- Encoding is linear in `Program` size; a 20-gene example compiles to a few KB
  (`PROG` dominates; `CHNK` code bytes are ~2× source size, same as classic
  `Chunk`).
- Classic load from `CHNK` avoids the parser + compiler entirely
  (measured target: < 1 ms for the example suite's average file); sim backends
  still pay the same `Program` reconstruction cost as a parse, which is the
  inherent cost of their inputs.
- No allocations beyond the decoded structures; `struct`-based packing keeps
  encode/decode single-pass.

---

## 14. Implementation Plan

**M1 — codec core.** `hxbc.py` primitives + `PROG` encode/decode + `Chunk`/
`SRC`/header/trailer; tests 1, 5, 11, 12.
**M2 — CLI write/run.** `--compile`, extension dispatch, classic + sim run from
binary, `--compare`; tests 6, 7, 8, 13.
**M3 — debug + decompile.** `--decompile`, byte-for-byte mode, `--disassemble`
and `--debug` from binary, `verify`; tests 2, 3, 4, 9, 10, 14.
**M4 — docs.** §9 CLI table in `language-spec.md`, §8 additions in
`compiler-design.md`, this document marked Implemented.

Each milestone ends with a green gate (mypy/ruff/pytest≥80).

---

## 15. Risks and Open Questions

| Risk / question | Mitigation / decision |
|---|---|
| `CHNK` staleness after a parser change | CHNK is a cache; load verifies and rebuilds from PROG (§5.2) |
| Binary `Program` carries `line` numbers that drift from `SRC` | Only used for diagnostics; breakpoints prefer the embedded `SRC` when present |
| Field ordering in `fields` dicts is not semantically significant | Canonical sorted order on write; R2 only requires canonical input |
| Embedded source could be tampered independently | `SRC` sits inside the checksummed section; any byte change breaks the trailer |
| Do we need `.helixc` for the web server? | Out of scope for M1–M4; `dumps_program`/`loads_program` already provide the bytes API the server would need |

---

## 16. Summary

`.helixc` is a versioned, typed, self-contained container whose authoritative
payload is the serialized `Program` AST, optionally cached as a precompiled
`Chunk` and an embedded original `SRC`. It gives the CLI a first-class binary
surface for all three requested capabilities — **write** (`--compile`),
**read and run** (extension dispatch → classic or sim backends), and
**debug and test** (`--disassemble`/`--debug`/`--decompile`/`--compare`,
round-trip invariants R1–R3, and integrity self-tests) — without weakening the
safety or determinism guarantees the rest of the pipeline relies on.
