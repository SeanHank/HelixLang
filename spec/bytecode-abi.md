# HelixLang Bytecode ABI Specification v1

**Status**: Frozen as of HelixLang 2026.8.4  
**OPCODE_VERSION**: 1  
**Authoritative source**: `src/helixlang/bytecode.py`, `src/helixlang/codon_table.py`

---

## 1. Overview

HelixLang compiles `.helix` source into bytecode executed by the CellVM.
The bytecode is a sequence of variable-length instructions, each consisting
of an opcode byte followed by zero or more operand bytes.

## 2. Instruction format

```
┌──────────┬──────────────────────────┐
│ opcode   │ operands (0-2 bytes)     │
│ (1 byte) │ (big-endian u8 or u16)   │
└──────────┴──────────────────────────┘
```

- **opcode**: 1 byte, values `0x00`-`0xFF`. See §3.
- **operands**: 0, 1, or 2 bytes depending on opcode. See `OP_OPERAND_BYTES`.
- **operand encoding**: Big-endian unsigned. u16 = `(high << 8) | low`.

## 3. Opcode table

| Opcode | Value | Operand bytes | Description |
|--------|-------|---------------|-------------|
| `OP_START` | `0x10` | 0 | Gene start marker |
| `OP_HALT` | `0x11` | 0 | Halt execution |
| `OP_RETURN` | `0x12` | 0 | Return from gene |
| `OP_NOP` | `0x13` | 0 | No operation |
| `OP_PUSH_CONST` | `0x20` | 1 | Push constant pool entry |
| `OP_POP` | `0x21` | 0 | Pop top of stack |
| `OP_DUP` | `0x22` | 0 | Duplicate top |
| `OP_SWAP` | `0x23` | 0 | Swap top two |
| `OP_BUILD_PROTEIN` | `0x30` | 1 | Build protein from pool |
| `OP_BUILD_MEMBRANE` | `0x31` | 1 | Build membrane component |
| `OP_BUILD_PIGMENT` | `0x32` | 0 | Build pigment |
| `OP_MOVE` | `0x40` | 1 | Cell movement |
| `OP_SIGNAL` | `0x41` | 1 | Emit signal |
| `OP_DIVIDE` | `0x42` | 1 | Cell division |
| `OP_DIE` | `0x43` | 1 | Cell death |
| `OP_FEED` | `0x44` | 1 | Nutrient uptake |
| `OP_GROW_LSYSTEM` | `0x50` | 1 | L-system growth step |
| `OP_DIFFUSE` | `0x51` | 1 | Diffusion step |
| `OP_REACT` | `0x52` | 1 | Chemical reaction |
| `OP_EMIT_MORPHOGEN` | `0x53` | 1 | Emit morphogen |
| `OP_READ_MEM` | `0x60` | 1 | Read memory slot |
| `OP_WRITE_MEM` | `0x61` | 1 | Write memory slot |
| `OP_MODIFY_STATE` | `0x62` | 1 | Modify cell state |
| `OP_REGULATE` | `0x63` | 1 | Regulation action |
| `OP_BIND` | `0x64` | 1 | Binding action |
| `OP_CALL_GENE` | `0x70` | 2 | Call gene at u16 offset |
| `OP_JUMP` | `0x80` | 2 | Unconditional jump (u16 target) |
| `OP_JUMP_IF_ZERO` | `0x81` | 2 | Jump if top == 0 |
| `OP_ADD` | `0x90` | 0 | Add top two |
| `OP_SUB` | `0x91` | 0 | Subtract |
| `OP_MUL` | `0x92` | 0 | Multiply |
| `OP_LT` | `0x93` | 0 | Less-than comparison |
| `OP_NOT` | `0x94` | 0 | Logical NOT |
| `OP_TICK` | `0xF0` | 0 | Advance simulation tick |
| `OP_DEBUG` | `0xFE` | 0 | Debug breakpoint |

## 4. Chunk layout

A `Chunk` contains:

| Field | Type | Description |
|-------|------|-------------|
| `code` | `bytearray` | Instruction stream |
| `constants` | `list[Any]` | Constant pool (strings, numbers) |
| `lines` | `list[int]` | Source line per code byte |
| `codon_indices` | `list[int]` | Codon index per code byte |
| `gene_offsets` | `dict[str, int]` | Gene name → code offset |

## 5. Binary serialization (`.helixc`)

The `.helixc` format (implemented in `hxbc.py`) wraps a Chunk:

```
┌─────────────────────────────────────────────┐
│ Magic: "HLXC" (4 bytes)                     │
│ Format version: u8 (= 1)                    │
│ Sections:                                    │
│   ┌──────┬──────────┬────────────────┐      │
│   │ PROG │ length   │ serialized AST │      │
│   ├──────┼──────────┼────────────────┤      │
│   │ CHNK │ length   │ bytecode chunk │      │
│   ├──────┼──────────┼────────────────┤      │
│   │ SRC  │ length   │ source text    │      │
│   ├──────┼──────────┼────────────────┤      │
│   │ EOF  │ 0        │ SHA-256 hash   │      │
│   └──────┴──────────┴────────────────┘      │
└─────────────────────────────────────────────┘
```

## 6. Versioning policy

- **OPCODE_VERSION** is bumped when opcodes are added, removed, or reordered.
- **FORMAT_VERSION** (in `hxbc.py`) is bumped when the `.helixc` container changes.
- Both must be bumped together for breaking changes.
- Non-breaking additions (new opcodes at unused values) may bump only OPCODE_VERSION.

## 7. Stability guarantee

Once frozen, any HelixLang version ≥ 2026.8.4 shall:
1. Reject `.helixc` files with FORMAT_VERSION > its own.
2. Execute bytecode with OPCODE_VERSION ≤ its own.
3. Never change the semantics of an existing opcode without bumping OPCODE_VERSION.
