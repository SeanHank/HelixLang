# HelixLang VM Semantics Specification v1

**Status**: Frozen as of HelixLang 2026.8.4  
**Authoritative source**: `src/helixlang/vm.py`, `src/helixlang/cell.py`

---

## 1. Overview

The CellVM executes HelixLang bytecode against a virtual cell.
The VM is a stack machine with a single execution thread per cell.

## 2. Execution model

### 2.1 State

| Component | Description |
|-----------|-------------|
| **ip** | Instruction pointer (index into `Chunk.code`) |
| **stack** | Value stack (LIFO, max depth implementation-defined) |
| **constants** | Reference to `Chunk.constants` (read-only) |
| **cell** | Cell state (proteins, membrane, memory slots, position) |
| **grn** | Gene regulatory network state |
| **rng** | Seeded `random.Random` instance (if stochastic mode enabled) |

### 2.2 Tick semantics

Each call to `VM.tick()` or `VM.run_ticks(n)`:

1. Execute instructions until `OP_TICK` is encountered or instruction limit reached.
2. `OP_TICK` advances the simulation clock by 1 tick.
3. After each tick: update GRN, decay proteins, check division/death conditions.
4. The cell state after tick N is deterministic given the same bytecode and seed.

### 2.3 Instruction dispatch

```
while ip < len(code):
    op = code[ip]
    ip += 1
    match op:
        case OP_PUSH_CONST: push(constants[operand])
        case OP_POP: pop()
        case OP_BUILD_PROTEIN: cell.build_protein(operand)
        case OP_TICK: advance_clock(); break  # end of tick
        case OP_HALT: stop()
        ...
```

## 3. Determinism contract

### 3.1 RNG seeding

- If `seed` is provided, the VM creates `random.Random(seed)`.
- All stochastic operations (cell division probability, mutation, noise) use this RNG.
- If no seed is provided, the VM uses the global `random` module (non-deterministic).

### 3.2 Forbidden operations

The VM **shall not** use:
- `time.time()` or `datetime.now()` (wall-clock)
- `os.*` calls (file system)
- `socket.*` calls (network)
- Global `random.*` (must use seeded instance)

### 3.3 Floating-point

- All arithmetic uses Python `float` (IEEE 754 double).
- Same platform + same input → same output (bit-exact).
- Cross-platform bit-exactness is NOT guaranteed (ARM vs x87 FMA differences).

## 4. Error model

| Error class | When raised |
|-------------|------------|
| `RuntimeHelixError` | Invalid opcode, stack underflow, division by zero |
| `SimConfigError` | Invalid simulation configuration |
| `HelixError` | General runtime error |

Errors include the instruction pointer and source line (if available).

## 5. Memory model

### 5.1 Cell memory slots

- 256 memory slots (indexed by u8 operand).
- Each slot holds a float.
- Initialized to 0.0.
- `OP_READ_MEM i` pushes `memory[i]`.
- `OP_WRITE_MEM i` pops top and stores to `memory[i]`.

### 5.2 Protein pool

- Proteins are identified by constant pool index (u8).
- `OP_BUILD_PROTEIN i` increments protein count at index `i`.
- Proteins decay each tick according to `PROTEIN_HALF_LIFE_MEDIAN_TICKS`.

### 5.3 GRN state

- Gene expression levels are updated after each tick.
- Regulation edges apply: `OP_REGULATE` modifies target gene expression.
- Expression level range: 0.0 to 1.0 (clamped).

## 6. Initialization

### 6.1 Default cell state

```python
{
    "tick": 0,
    "alive": True,
    "volume": 1.0,  # µm³
    "proteins": {},  # index → count
    "membrane": {},  # component → count
    "memory": [0.0] * 256,
    "position": (0.0, 0.0, 0.0),
    "grn": {},  # gene → expression_level
}
```

### 6.2 Gene entry

When `OP_CALL_GENE offset` is executed:
1. Push current ip onto call stack.
2. Set ip to `offset`.
3. Execute until `OP_RETURN` or `OP_HALT`.

## 7. Backend integration

The VM interacts with the simulation runtime through:

- `sim_runtime.run()` — orchestrates VM + environment + GRN + FBA.
- `#config backend=<name>` — selects the simulation backend.
- `#sim kind=<name>` — selects a specialized backend.

The VM itself is backend-agnostic; backend-specific behavior is injected
through the `Cell` and `Environment` objects.
