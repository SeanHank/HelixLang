# HelixLang Simulation Execution Model

> This document defines the "cell simulator" layer of the HelixLang VM: the gene regulatory network (GRN), L-system morphogenesis, Gray-Scott reaction-diffusion, and the unified tick loop.

---

## 1. Cell State Structure

`Cell` is the VM's runtime context, updated every tick:

| Subsystem | Field | Type | Biological counterpart |
|---|---|---|---|
| **Genome** | `genes` | `dict[str, GeneBytecode]` | Chromosome |
| **Ribosome** | `ip`, `frames` | `int`, `list[Frame]` | Ribosome |
| **Cytoplasm** | `stack`, `proteins` | `list`, `dict[int, float]` | Cytoplasm (value stack + protein pool) |
| **Memory slots** | `slots[256]` | `list` | Cell membrane/nuclear memory |
| **GRN** | `grn` | `GRN` object | Gene regulatory network |
| **Energy** | `energy` | `int` | ATP/metabolic energy |
| **Position** | `x, y, heading` | `int, int, int` | Cell position and heading |
| **Morphogen field** | `field_u[N×N]`, `field_v[N×N]` | `list[list[float]]` | Morphogen U/V |
| **L-system** | `axiom`, `rules`, `state`, `turtle` | string + turtle state | Growing morphology |
| **Behavior log** | `log` | `list[Event]` | Behavior log (move/signal/divide/die) |

`proteins` uses `dict[int, float]`: the key is the protein kind (0..3, determined by the arg of `OP_BUILD_PROTEIN`), and the value is the concentration.

---

## 2. Gene Regulatory Network (GRN)

### 2.1 Model

It uses a hybrid model of **discrete ticks + sigmoid concentration thresholds**, between Kauffman Boolean networks and continuous ODEs:

- Each gene `g` has an expression level `level_g ∈ [0, 1]`.
- Each regulatory edge `(src, tgt, w)`: `w > 0` activates, `w < 0` represses.
- Promoter `p` has a threshold `θ_p` (i.e., the `strength` field).

### 2.2 Update Rules

Step 1 of each tick:

```
input(g) = Σ_{(s,g,w) in edges} w * active(s)
active(g) = level_g   # value from the previous tick
level_g_new = sigmoid(input(g) - θ_g)
```

where `sigmoid(x) = 1 / (1 + exp(-x))`.

Step 2 (triggered expression):

```
for g in genes:
    if level_g_new > 0.5 and not currently_running(g):
        vm.call_gene(g)
```

`level_g_new > 0.5` means the sigmoid input > 0, equivalent to "activating input exceeds repressing input".

### 2.3 Data Structures

```python
class GRN:
    nodes: dict[str, GeneNode]    # name -> {level, threshold}
    edges: list[Edge]             # (src, tgt, weight)

    def step(self) -> list[str]:  # return genes triggered this tick
        new_levels = {}
        for name, node in self.nodes.items():
            inp = sum(e.weight * self.nodes[e.src].level
                      for e in self.edges if e.tgt == name)
            new_levels[name] = sigmoid(inp - node.threshold)
        for name, lvl in new_levels.items():
            self.nodes[name].level = lvl
        return [n for n, lvl in new_levels.items() if lvl > 0.5]
```

### 2.4 Lambda Phage Toggle Switch Example

Two mutually repressing genes form a bistable switch:

```
#gene name=ci
ATG GCT GCT TAA
#end
#gene name=cro
ATG GGT GGT TAA
#end
#promoter name=p_ci strength=0.5
#promoter name=p_cro strength=0.5
#regulate cro -> p_ci strength=-1.0
#regulate ci -> p_cro strength=-1.0
```

With initial `ci.level=0.6, cro.level=0.4`, the system stabilizes at ci ON / cro OFF; after an external signal triggers a flip, it enters the other stable state. This is the HelixLang encoding of Ptashne's classic toggle switch.

### 2.5 Damping and Decay

To avoid repeated triggering from noise, each tick multiplies all levels by a decay factor `γ`:

```
level_g = γ * level_g + (1 - γ) * new_input
```

This models the balance of "protein degradation + synthesis".

Genes without an explicit `decay=` default to the E. coli median protein
half-life of ~110 min (Mosteller 1980, Helbig 2011). With one tick per minute,
`γ = decay_from_half_life_ticks(110)` ≈ 0.994 — a gene loses half its level
after ~110 ticks, and an explicit per-gene `decay=` always overrides the
universal default (`helixlang.core.units`).

---

## 3. L-System Morphogenesis

### 3.1 Model

Classic parametric L-system (Prusinkiewicz & Lindenmayer 1990):

- **Axiom**: the initial string, e.g., `F`.
- **Production rules**: `F -> F[+F]F[-F]F`, etc.
- **Turtle graphics interpretation**:
  - `F`: move forward one step and draw a line
  - `+` / `-`: turn left/right 25° (configurable)
  - `[` / `]`: push/pop the stack (branching)
  - Other characters: keep state, no drawing

### 3.2 `OP_GROW_LSYSTEM rules=k`

Each execution performs **one parallel rewrite** of the current L-system string (all characters replaced simultaneously), then uses the turtle interpretation to obtain a new point sequence.

```python
class LSystem:
    axiom: str
    rulesets: dict[int, dict[str, str]]   # k -> rules
    state: str                            # current string
    turtle: TurtleState                   # x, y, heading, stack

    def iterate(self, k):
        rules = self.rulesets[k]
        self.state = ''.join(rules.get(c, c) for c in self.state)
        return self.interpret()

    def interpret(self):
        points = []
        for c in self.state:
            if c == 'F':    self.turtle.forward(); points.append(self.turtle.pos())
            elif c == '+':  self.turtle.turn(+self.angle)
            elif c == '-':  self.turtle.turn(-self.angle)
            elif c == '[':  self.turtle.push()
            elif c == ']':  self.turtle.pop()
        return points
```

### 3.3 L-System Strings as Constants

Axioms and rule sets are stored in the Chunk's constant pool:

```
Constants:
  [0] ("axiom", "F")
  [1] ("rules", {0: {"F": "F[+F]F[-F]F"}, 1: {"F": "FF-[-F+F+F]+[+F-F-F]"}})
```

### 3.4 Morphology Output

Each `OP_GROW_LSYSTEM` appends the turtle point sequence to `cell.morphology_points`; the final output is a PNG (the prototype uses a pure-Python canvas or simple SVG).

---

## 4. Gray-Scott Reaction-Diffusion

### 4.1 Model

Classic Gray-Scott (Pearson 1993):

```
U' = U + (Du * ∇²U - U*V² + F*(1-U))
V' = V + (Dv * ∇²V + U*V² - (F+k)*V)
```

Parameters (default Pearson classic values):
- `Du = 0.16`, `Dv = 0.08`
- `F = 0.035`, `k = 0.065`  → produces spots (mitosis)
- `F = 0.012`, `k = 0.045`  → produces stripes (solitons)
- `F = 0.025`, `k = 0.06`   → coral-like

### 4.2 `OP_REACT type=t` / `OP_DIFFUSE dir=d` / `OP_EMIT_MORPHOGEN id=m`

- `OP_REACT`: advances N steps of Gray-Scott iteration (N defaults to 1, adjustable via `--config react_steps=N`).
- `OP_DIFFUSE`: performs a single diffusion step (Laplacian); direction `d` selects 4-neighborhood vs 8-neighborhood.
- `OP_EMIT_MORPHOGEN id=m`: injects morphogen V at concentration 1.0 into `field_v` at the cell's current position.

### 4.3 Implementation

```python
class GrayScott:
    def __init__(self, n=32):
        self.n = n
        self.u = [[1.0]*n for _ in range(n)]
        self.v = [[0.0]*n for _ in range(n)]
        # initial perturbation: a small block of V at the center
        for i in range(n//2-2, n//2+2):
            for j in range(n//2-2, n//2+2):
                self.v[i][j] = 1.0; self.u[i][j] = 0.5

    def step(self, F=0.035, k=0.065, Du=0.16, Dv=0.08):
        new_u = [row[:] for row in self.u]
        new_v = [row[:] for row in self.v]
        for i in range(1, self.n-1):
            for j in range(1, self.n-1):
                lu = self.laplace(self.u, i, j)
                lv = self.laplace(self.v, i, j)
                uvv = self.u[i][j] * self.v[i][j]**2
                new_u[i][j] = self.u[i][j] + (Du*lu - uvv + F*(1-self.u[i][j]))
                new_v[i][j] = self.v[i][j] + (Dv*lv + uvv - (F+k)*self.v[i][j])
        self.u, self.v = new_u, new_v

    @staticmethod
    def laplace(f, i, j):
        return (f[i-1][j] + f[i+1][j] + f[i][j-1] + f[i][j+1] - 4*f[i][j]) * 0.2
```

### 4.4 Morphology to Gene Feedback

At the end of each tick, the `V` concentration at the cell's current position is used as an additional activating input for a gene:

```
input for gene "pigment" += 2.0 * V[cell.x, cell.y]
```

This implements the developmental feedback loop where "the morphogen field in turn regulates gene expression".

### 4.5 Output

With `output=png`, `field_v` is rendered as a PNG every K ticks (the prototype writes PPM/PNG in pure Python, or merely dumps CSV for external plotting).

---

## 5. Behavior Primitives

### 5.1 `OP_MOVE dir`

`dir` is the wobble position of the third base (0..3) = N/E/S/W. The cell position `(x, y)` is updated; `cell.energy -= 1`.

### 5.2 `OP_SIGNAL channel`

Injects signal molecules (e.g., AI autoinducers) into the signal field at the current cell's position; other cells' GRNs can use them as inputs.

### 5.3 `OP_DIVIDE mode`

- `mode=0`: divide into the empty slot on the right; the daughter cell inherits the genome; energy is halved.
- `mode=1`: asymmetric division (the mother cell keeps more energy).

### 5.4 `OP_DIE mode`

- `mode=0`: apoptosis, removed from the field.
- `mode=1`: necrosis, releases all proteins and morphogens into the field.

### 5.5 `OP_FEED src`

Harvests energy from the environment: `src=0` photosynthesis (by position), `src=1` chemosynthesis (by morphogen field U).

---

## 6. Tick Loop (Overview)

```python
def run(self, max_ticks):
    while self.tick < max_ticks and self.cell.alive:
        # 1. update GRN
        triggered = self.grn.step()

        # 2. push triggered genes as frames
        for g in triggered:
            self.call_gene(g)

        # 3. bytecode execution (bounded by the tick quota)
        ops_this_tick = 0
        while self.frames and ops_this_tick < self.ops_quota:
            self.execute_one()
            ops_this_tick += 1

        # 4. morphology update (backlogged L-system / reaction-diffusion steps)
        self.cell.flush_morphology()

        # 5. feedback: morphogen field concentration → GRN input
        self.feedback_morph_to_grn()

        # 6. output snapshot
        self.snapshot()

        self.tick += 1
```

### 6.1 Tick Quota

`ops_quota` limits the number of bytecode instructions executed per tick, modeling that "ribosomal translation has a finite speed". Defaults to 64, configurable via `#config ops_per_tick=N`.

### 6.2 Multicellular Extension

The population runtime (`population.py`) runs *programmable* cells — not
just the single-celled prototype. Each `PopulationCell` carries a
`program` (bytecode chunk), a per-cell `GRN` (deep-copied from the
template so daughter state stays isolated), and an energy budget; the
`CellVM`-style dispatch runs per cell under a shared ops budget. The
shared tick loop is:

```
metabolism → field diffusion → per-cell GRN → per-cell dispatch
→ quorum → division → mechanics → environment
```

1. **Metabolism** — Monod uptake `vmax·S/(Ks+S)` scaled by 38 ATP/glucose
   depletes the local glucose field (optionally via `DynamicFluxBalance`
   coupling, documented in `07-bio-modules.md` §2); `OP_DIVIDE` spawns a real
   daughter (binary fission, energy split).
2. **Diffusion** — AI-2 (quorum), glucose, and O₂ fields advance one
   sub-stepped tick (`environment.py`, flux-conservative, D≤0.25/step;
   CROMICS crowding reduces D with occupied-neighbor fraction).
3. **Per-cell GRN + dispatch** — each cell steps its GRN, triggered genes
   push bytecode frames, and `execute_cell` runs up to `ops_per_tick`
   instructions (move/signal/feed/divide/die/build/bind/call/jump/stack).
4. **Quorum** — AI-2 field concentration triggers the quorum behavior
   (default 10 µM).
5. **Division** — daughter cells require an empty/neighbor site;
   mechanics (`shoving`/`force`) clears collisions.
6. **Trace** — optional streaming of per-cell snapshots (id/x/y/alive/
   energy/proteins/gene levels) for memory-bounded long runs.

The signal field, morphogen field, and nutrient fields are shared; the
GRN runs independently in each cell.

### 6.3 Unit System (always on)

The simulator runs on **physical units** end-to-end (no `units=` switch; the
legacy gameplay-unit catalog was removed). Energy is counted in **ATP
molecules**, the signal field is in **µM**, diffusion is a physical **µm²/s**
coefficient, and one tick is one minute (`helixlang.core.units`):

| Quantity | Default | Physical meaning |
|---|---|---|
| 1 tick | — | 1 minute (Neidhardt 1996) |
| cell energy budget | `1e9` | ~10⁹ ATP molecules (Orth 2010) |
| division threshold | `1.8e9` | ~20 rich-medium minutes at the +4×10⁷ ATP/tick net intake |
| quorum threshold | `10.0` | 10 µM AI-2 (Xavier & Bassler 2003) |
| signal diffusion | `100.0` | 100 µm²/s; converted to D≈60 on the 10 µm lattice edge, via stable sub-steps (§4.3) |
| glucose diffusion | `600.0` | glucose µm²/s in water (Stewart 2003; CRC Handbook) |
| O₂ diffusion | `2500.0` | O₂ µm²/s in water (CRC Handbook) |
| acetate diffusion | `1200.0` | acetate µm²/s (CRC Handbook) |
| glucose Ks (Monod) | `0.1` | mM, Kovárová-Kovar & Egli 1998 |
| GRN decay | `≈0.994` | 110-min half-life ⇒ ≈0.994/tick (§2.5) |

---

## 7. Output Formats

`#config output=stdout,csv,png` multiple options:

| Output | Content | File |
|---|---|---|
| `stdout` | one line per tick of cell state | terminal |
| `csv` | tick, x, y, energy, protein_0..3, level_lacZ, ... | `trace.csv` |
| `png` | morphogen field snapshot + L-system morphology | `morphology_tick_NN.png` |
| `none` | no output (for benchmarking) | — |

---

## 8. Correspondence Validation with Biology

| Biological process | HelixLang mechanism | Validation method |
|---|---|---|
| Translation (mRNA→protein) | bytecode dispatch execution | disassembly trace |
| Gene regulation | GRN sigmoid update | toggle switch bistability verification |
| Developmental morphology | L-system iteration | comparison of plant branching topology |
| Turing pattern formation | Gray-Scott reaction-diffusion | spot/stripe parameters vs. Pearson 1993 |
| Cell motility | OP_MOVE | trajectory log |
| Signal transduction | OP_SIGNAL + shared signal field | quorum sensing simulation |
| Cell division/death | OP_DIVIDE / OP_DIE | population curves |
| Codon degeneracy | third-base wobble position | synonymous codons yield the same opcode with different args |
| Translation table switching | `--table=mito_vertebrate` | TGA→pigment instead of stop |
