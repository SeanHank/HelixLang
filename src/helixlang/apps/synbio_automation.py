"""Synthetic design automation: Boolean -> DNA + SBOL3 (T2.3, problem P4).

A Cello-style workflow (Nielsen et al. 2016 Science 352:aac7341; Cello
2.0, Jones et al. 2022 Nat Protoc 17:1097; LOICA 2022 ACS Synth Biol)
compiled from a truth table:

    1. **Truth table**  ->  minimized sum-of-products Boolean netlist
       (Quine-McCluskey-style term merging, small input counts).
    2. **Gate assignment**: each netlist node is assigned a characterized
       library gate by scoring how well the gate's transfer function
       reproduces the required Boolean behavior given the *actual*
       upstream output levels (Cello's scoring principle).
    3. **DNA assembly**: assigned gates are assembled into one DNA
       sequence (promoter + RBS + CDS + terminator per gate, in
       topological order).
    4. **SBOL3 export**: the design is serialized to SBOL3 RDF/XML
       (:mod:`helixlang.interop`) for SynBioHub/Cello interop.
    5. **Predicted dynamics**: the assigned transfer functions are
       iterated to steady state and the observed truth table is checked
       against the target (the Cello "predicted dynamics" step).

The characterized gate library models multi-input Hill repression
(Cello NOR-gate logic; repressor transfer ``kd^n/(kd^n + sum L_i^n)``)
and Hill activation.  Repressor CDS fragments are representative
N-terminal sequences used for design validation (same convention as the
selection markers in :mod:`helixlang.apps.synbio_designer`).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from math import prod

from helixlang.apps.synbio_designer import (
    DEFAULT_MCS,
    MCS_SITES,
    ORIGIN_SEQUENCES,
    PROMOTER_SEQUENCES,
    SELECTION_MARKERS,
    TERMINATOR_SEQUENCES,
    genbank_format,
)
from helixlang.central_dogma import coupled_transcription_translation
from helixlang.interop import (
    SBOL_ROLE_GENE,
    SBOL_ROLE_PROMOTER,
    SBOL_ROLE_RBS,
    SBOL_ROLE_TERMINATOR,
    sbol3_dumps,
)

# ============================================================================
# Repressor coding sequences (representative N-terminal fragments)
# ============================================================================

_LACI_CDS = (
    "ATGGTAAATCCAGTTACCCTTTATGATGTAGCCGAATACGCCGGAGTTTCCTATCAGACAGTCTCTAGAGTTG"
    "TGAATCAGGCTTCTCACGTCAGTGCCAAAACTAGAGAAAAGGTGGAAGCAGCAATGGCAGAATTGAATTATATC"
    "CCTAACAGGGTCGCTCAACAACTGGCAGGAAAGCAAAGTCTGCTCATCGGTGTCGCA"
)
_TETR_CDS = (
    "ATGTCCAGATTAGATAAAAGTAAAGTGATTAACAGCGCATTAGAGCTGCTTAATGAGGTCGGAATCGAAGGTTT"
    "AACAACCCGTAAACTCGCCCAGAAGCTAGGTGTAGAGCAGCCTACATTGTATTGGCATGTAAAAAATAAGCGGG"
    "CTTTGCTCGACGCCTTAGCCATTGAGATGTTAGATAGGCACCATACTCACTTTTGCCCTTTAGAAGGGGAAAGC"
    "TGGCAAGATTTTTTACGTAATAACGCTAAAAGTTTTAGATGTGCTTTACTAAGTCATCGCGATGGAGCAAAAGT"
    "ACATTTAGGTACACGGCCTACAGAAAAACAGTATGAAACTCTCGAAAATCAATTAGCCTTTTTATGCCAACAAG"
    "GTTTTTCACTAGAGAATGCATTATATGCACTCAGCGCTGTGGGGCATTTTACTTTAGGTTGCGTATTGGAAGAT"
    "CAAGAGCATCAAGTCGCTAAAGAAGAGAAAGGGACACACTACTACTGATAGTATGCCGCCATTATTACGACAAG"
    "CTATCGAATTATTTGATCACCAAGGTGCAGAGCCAGCCTTCTTATTCGGCCTTGAATTGATCATATGCGGATTA"
    "GAAAAACAACTTAAATGTGAAAGTGGGTCCG"
)
_CI_CDS = (
    "ATGAATACTTTCAGCCAACCCGATGTAGAGCAGCGGATGAAGTCTTTACTTCAGGTCGTAACTACCGTAACCGT"
    "AGAGGCCATCGAAGACATCGCGGTAAAGGTTGCCGCAGCCAAACGCGCAAAAATTGTTGATAAACACGCCGCAGT"
    "TGCGTTTAATACGTCTGATTTTATCGAACTGGCGGCGATGGAAGTTATGCCCGATCAGGCGGATGCAGCGCAGTT"
    "TGTGAAAGTGATGGCGCATAAACCAACATTTGCCGCGTTTATCTTTGCAAAGAGCGATGCCGGCCTTTTCAGGGT"
    "CGCTGCTGGCG"
)
_ARAC_CDS = (
    "ATGGCGAAATTGTTTGATTTTCGATACCGTCGTTTTTCAGGATTCGATGATACCGAGCTGATCCGTATTTCCGA"
    "ACAATACAGCGTTCAGCCTGTTGCAAAATCAACAGCAACCATTGATTTTAGCGCTTACAGCGTCGGCATTCAAA"
    "GCGCCTATGATGTTGCCGCCGAGCTTGTTGAACTGGAAGGGATGATGATTAATCTTCGTAACGGTATCGCCATT"
    "TTTAATTTACAGCAGCAATATCAGGCTGGTGGCCATTTTCATTACTTTCTTTTTTACCGGCACATCAGCGATTC"
    "CTGGACAGATTATC"
)

REPRESSOR_CDS: dict[str, str] = {
    "lacI": _LACI_CDS,
    "tetR": _TETR_CDS,
    "cI": _CI_CDS,
    "araC": _ARAC_CDS,
}

#: Shine-Dalgarno ribosome binding site (representative)
RBS_SEQ = "AGGAGG"
#: default promoter driving gate expression
DEFAULT_GATE_PROMOTER = "lac"
DEFAULT_TERMINATOR = "rrnB_T1"


def _default_parts(protein: str) -> dict[str, str]:
    return {
        "promoter": PROMOTER_SEQUENCES[DEFAULT_GATE_PROMOTER],
        "rbs": RBS_SEQ,
        "cds": REPRESSOR_CDS[protein],
        "terminator": TERMINATOR_SEQUENCES[DEFAULT_TERMINATOR],
    }


# ============================================================================
# Boolean logic helpers
# ============================================================================

_AND = "AND"
_NAND = "NAND"
_OR = "OR"
_NOR = "NOR"
_NOT = "NOT"
_BUFFER = "BUFFER"
_CONST = "CONST"


def boolean_apply(logic: str, values: list[bool]) -> bool:
    """Evaluate a Boolean function over input truth values."""
    if logic == _CONST:
        return bool(values[-1])
    if logic == _BUFFER:
        return bool(values[0])
    if logic == _NOT:
        return not bool(values[0])
    if logic == _AND:
        return all(values)
    if logic == _NAND:
        return not all(values)
    if logic == _OR:
        return any(values)
    if logic == _NOR:
        return not any(values)
    raise ValueError(f"unknown gate logic {logic!r}")


def _merged(a: dict[str, bool], b: dict[str, bool]) -> dict[str, bool] | None:
    """Merge two product terms differing in exactly one literal (or None)."""
    diff: list[str] = []
    for var in set(a) | set(b):
        if var in a and var in b and a[var] != b[var]:
            diff.append(var)
        elif var in a and var not in b:
            diff.append(var)
        elif var not in a and var in b:
            diff.append(var)
    if len(diff) != 1:
        return None
    merged = {var: val for var, val in a.items() if var != diff[0]}
    return merged or None


def minimize_expression(inputs: list[str],
                        minterms: list[dict[str, bool]]) -> list[dict[str, bool]]:
    """Quine-McCluskey-style 2-level minimization of a DNF.

    Merges minterms differing in exactly one literal until no further
    merging is possible (exact for small input counts).  Returns the
    minimized product terms for the function that is true on ``minterms``.
    """
    terms = [dict(t) for t in minterms]
    changed = True
    while changed:
        changed = False
        next_terms: list[dict[str, bool]] = []
        used: set[int] = set()
        for i, a in enumerate(terms):
            for j in range(i + 1, len(terms)):
                b = terms[j]
                m = _merged(a, b)
                if m is not None:
                    if m not in next_terms:
                        next_terms.append(m)
                    used.add(i)
                    used.add(j)
                    changed = True
        for i, t in enumerate(terms):
            if i not in used and t not in next_terms:
                next_terms.append(t)
        terms = next_terms
    return terms


def _decompose_xor(a: str, b: str) -> str:
    """XOR(a, b) = (a AND NOT b) OR (NOT a AND b) built from primitives."""
    return f"({a} AND NOT {b}) OR (NOT {a} AND {b})"


# ============================================================================
# Truth table
# ============================================================================

@dataclass(slots=True)
class TruthTable:
    """A Boolean truth table over named inputs/outputs.

    ``rows`` is a list of ``(input_values, output_values)`` tuples where
    each element is a tuple of ``bool`` aligned with :attr:`inputs` /
    :attr:`outputs`.
    """

    inputs: list[str]
    outputs: list[str]
    rows: list[tuple[tuple[bool, ...], tuple[bool, ...]]]

    @classmethod
    def from_function(cls, inputs: list[str], outputs: list[str],
                      func: Callable[[tuple[bool, ...]], bool | tuple[bool, ...]]) -> TruthTable:
        """Enumerate all 2^n input combinations of ``func``.

        ``func`` maps a tuple of input bools to a tuple of output bools
        (or a single bool when there is one output).
        """
        rows: list[tuple[tuple[bool, ...], tuple[bool, ...]]] = []
        for combo in itertools.product((False, True), repeat=len(inputs)):
            result = func(combo)
            if not isinstance(result, tuple):
                result = (result,)
            if len(result) != len(outputs):
                raise ValueError(
                    f"func returned {len(result)} outputs for "
                    f"{len(outputs)} declared outputs")
            rows.append((combo, tuple(bool(r) for r in result)))
        return cls(inputs=inputs, outputs=outputs, rows=rows)

    def to_truth_value(self, input_values: tuple[bool, ...],
                       output_index: int = 0) -> bool:
        for inv, outv in self.rows:
            if inv == input_values:
                return bool(outv[output_index])
        raise ValueError(f"missing row for {input_values}")

    def minterms(self, output_index: int = 0) -> list[dict[str, bool]]:
        """Product terms (literals over :attr:`inputs`) where the output is 1."""
        terms: list[dict[str, bool]] = []
        for inv, outv in self.rows:
            if not outv[output_index]:
                continue
            terms.append({name: val for name, val in zip(self.inputs, inv, strict=True)})
        return terms


# ============================================================================
# Characterized gate library
# ============================================================================

@dataclass(slots=True)
class CharacterizedGate:
    """A characterized genetic gate (Cello-style part).

    Attributes:
        id: library identifier (e.g. ``"NOT_lacI"``)
        logic: one of BUFFER/NOT/AND/NAND/OR/NOR
        output_signal: the repressor protein the gate produces
        parts: DNA part sequences (promoter/rbs/cds/terminator)
        n: Hill coefficient (cooperativity)
        kd: half-maximal effector level for the multi-input transfer
        k: maximal output level (default 1.0)
        leak: basal (unrepressed) output leak in [0, 1]
    """

    id: str
    logic: str
    output_signal: str
    parts: dict[str, str]
    n: float = 2.0
    kd: float = 0.25
    k: float = 1.0
    leak: float = 0.02

    @property
    def num_inputs(self) -> int:
        return {"NOT": 1, "BUFFER": 1, "AND": 2, "NAND": 2,
                "OR": 2, "NOR": 2}[self.logic]

    @property
    def dna(self) -> str:
        """Full gate DNA: promoter + RBS + CDS + terminator."""
        return (self.parts["promoter"] + self.parts["rbs"]
                + self.parts["cds"] + self.parts["terminator"])

    def transfer(self, input_levels: list[float]) -> float:
        """Steady-state output level for input levels in [0, 1].

        Repressor logic (NOT/NOR/NAND) uses multi-input repression
        ``kd^n/(kd^n + sum L_i^n)``; activator logic (BUFFER/AND/OR)
        uses Hill activation.  NAND/AND combine inputs multiplicatively
        (both must be high to saturate); NOT/OR/NOR/BUFFER combine
        additively.
        """
        kd = self.kd
        n = self.n
        act: float
        if self.logic in ("AND", "NAND"):
            base = prod(max(l, 1e-9) for l in input_levels) ** n
            act = base / (kd ** n + base)
        else:
            s = sum(l ** n for l in input_levels)
            act = s / (kd ** n + s)
        if self.logic in ("NOT", "NOR", "NAND"):
            act = 1.0 - act
        return self.leak + (1.0 - self.leak) * self.k * act

    def output_levels(self) -> tuple[float, float]:
        """Characterized (low, high) steady-state output levels."""
        on = self.transfer([1.0] * self.num_inputs)
        off = self.transfer([0.0] * self.num_inputs)
        return (min(on, off), max(on, off))


def _library() -> list[CharacterizedGate]:
    gates: list[CharacterizedGate] = []
    specs: list[tuple[str, str]] = [
        # (logic, output repressor)
        ("BUFFER", "araC"),
        ("NOT", "lacI"), ("NOT", "tetR"), ("NOT", "cI"),
        ("AND", "lacI"), ("AND", "tetR"),
        ("NAND", "lacI"), ("NAND", "tetR"), ("NAND", "cI"),
        ("OR", "lacI"), ("OR", "tetR"),
        ("NOR", "lacI"), ("NOR", "tetR"), ("NOR", "cI"), ("NOR", "araC"),
    ]
    for logic, protein in specs:
        # NOT gates need strong repression (low kd) so their LOW output
        # sits far below the kd of downstream OR/NAND gates; wide-range
        # gates keep kd=0.25 (Cello separation-margin requirement).
        kd = 0.1 if logic == "NOT" else 0.25
        gates.append(CharacterizedGate(
            id=f"{logic}_{protein}",
            logic=logic,
            output_signal=protein,
            parts=_default_parts(protein),
            kd=kd,
        ))
    return gates


#: default characterized gate library
GATE_LIBRARY: list[CharacterizedGate] = _library()


# ============================================================================
# Netlist
# ============================================================================

@dataclass(slots=True)
class NetlistNode:
    """One gate of the synthesized circuit.

    Attributes:
        id: node identifier (its output signal name)
        logic: gate logic
        inputs: list of upstream signal names (node ids or circuit inputs)
        truth: optional (logic, num_inputs) override used to evaluate the
            node's Boolean function when it is a synthetic structure
            rather than a library gate (e.g. a constant).
    """

    id: str
    logic: str
    inputs: list[str]
    value: bool | None = None   # CONST nodes only

    def evaluate(self, input_values: dict[str, bool]) -> bool:
        if self.logic == _CONST:
            return bool(self.value)
        return boolean_apply(self.logic, [input_values[i] for i in self.inputs])


@dataclass(slots=True)
class Netlist:
    """Ordered circuit netlist (nodes sorted in topological order)."""

    inputs: list[str]
    outputs: list[str]
    nodes: list[NetlistNode]

    def evaluate(self, input_values: tuple[bool, ...]) -> tuple[bool, ...]:
        """Evaluate the netlist for one input combination."""
        values: dict[str, bool] = dict(zip(self.inputs, input_values, strict=True))
        for node in self.nodes:
            values[node.id] = node.evaluate(values)
        return tuple(values[o] for o in self.outputs)

    def topo_order(self) -> list[NetlistNode]:
        return list(self.nodes)


def _binary_reduce(nodes: list[NetlistNode], fresh: Callable[[str], str],
                   base: str, signals: list[str], logic: str) -> str | None:
    """Reduce a fan-in of signals into a balanced binary tree of gates.

    The characterized library only provides 2-input AND/OR/NAND/NOR, so
    higher fan-ins are decomposed into a balanced tree (Cello compiles
    from 2-input gates).  Returns the root signal id (or None for an
    empty fan-in).
    """
    current = list(signals)
    while len(current) > 1:
        nxt: list[str] = []
        for i in range(0, len(current), 2):
            pair = current[i:i + 2]
            if len(pair) == 1:
                nxt.append(pair[0])
                continue
            node_id = fresh(base)
            nodes.append(NetlistNode(node_id, logic, pair))
            nxt.append(node_id)
        current = nxt
    return current[0] if current else None


def synthesize_netlist(table: TruthTable) -> Netlist:
    """Build a Boolean netlist implementing ``table`` (minimized DNF)."""
    nodes: list[NetlistNode] = []
    counter: dict[str, int] = {}

    def fresh(base: str) -> str:
        counter[base] = counter.get(base, 0) + 1
        return f"{base}_{counter[base]}"

    for out_idx, output in enumerate(table.outputs):
        terms = minimize_expression(table.inputs, table.minterms(out_idx))
        always_off = (not terms and all(
            not rv[out_idx] for _, rv in table.rows))
        always_on = (not terms and all(
            rv[out_idx] for _, rv in table.rows))
        if always_on or always_off:
            const_id = fresh("const")
            nodes.append(NetlistNode(const_id, _CONST, [],
                                     value=always_on))
            nodes.append(NetlistNode(output, _BUFFER, [const_id]))
            continue
        # per-term AND
        and_ids: list[str] = []
        for term in terms:
            if not term:
                continue
            term_inputs: list[str] = []
            for var, positive in term.items():
                if positive:
                    term_inputs.append(var)
                else:
                    not_id = fresh(f"not_{var}")
                    nodes.append(NetlistNode(not_id, _NOT, [var]))
                    term_inputs.append(not_id)
            if len(term_inputs) == 1:
                and_ids.append(term_inputs[0])
                continue
            and_id = _binary_reduce(nodes, fresh, f"and_{output}",
                                    term_inputs, _AND)
            if and_id is not None:
                and_ids.append(and_id)
        if len(and_ids) == 0:
            const_id = fresh("const")
            nodes.append(NetlistNode(const_id, _CONST, [], value=False))
            nodes.append(NetlistNode(output, _BUFFER, [const_id]))
            continue
        if len(and_ids) == 1:
            nodes.append(NetlistNode(output, _BUFFER, [and_ids[0]]))
        else:
            or_id = _binary_reduce(nodes, fresh, f"or_{output}",
                                   and_ids, _OR)
            if or_id is not None:
                nodes.append(NetlistNode(output, _BUFFER, [or_id]))
    return Netlist(inputs=list(table.inputs), outputs=list(table.outputs),
                   nodes=nodes)


# ============================================================================
# Gate assignment (Cello-style scoring)
# ============================================================================

def _gate_high_low(gate: CharacterizedGate) -> tuple[float, float]:
    return gate.output_levels()


def _node_boolean_function(node: NetlistNode,
                           input_combo: tuple[bool, ...]) -> bool:
    values: dict[str, bool] = {}
    for name, val in zip(node.inputs, input_combo, strict=True):
        values[name] = val
    return node.evaluate(values)


def score_gate(gate: CharacterizedGate, node: NetlistNode,
               upstream_levels: list[tuple[float, float]]) -> float:
    """Cello-style assignment score for ``gate`` at ``node``.

    Evaluates the candidate gate's transfer function at every upstream
    on/off combination (using the *characterized* low/high output levels
    of the upstream gates) and sums the separation margin of the output
    from the 0.5 threshold, penalizing wrong polarity.  Mirrors Cello's
    principle of assigning gates whose transfer reproduces the required
    logic given the actual input signal levels.
    """
    if gate.num_inputs != len(upstream_levels):
        return -float("inf")
    total = 0.0
    for combo in itertools.product((False, True), repeat=len(upstream_levels)):
        levels = [high if b else low
                  for (low, high), b in zip(upstream_levels, combo, strict=True)]
        out = gate.transfer(levels)
        target = _node_boolean_function(node, combo)
        margin = (out - 0.5) if target else (0.5 - out)
        total += max(0.0, margin)
    return total


def assign_gates(netlist: Netlist,
                 library: list[CharacterizedGate] | None = None,
                 ) -> dict[str, CharacterizedGate]:
    """Assign a characterized gate to every netlist node (greedy, topo).

    Returns ``{node_id: gate}``.  CONST and BUFFER glue nodes that have
    no library match are assigned a library BUFFER (or, for CONST nodes,
    left unassigned -- they are constant sources consumed by BUFFERs).
    """
    lib = library if library is not None else GATE_LIBRARY
    by_logic: dict[str, list[CharacterizedGate]] = {}
    for gate in lib:
        by_logic.setdefault(gate.logic, []).append(gate)
    assignment: dict[str, CharacterizedGate] = {}
    # upstream output levels per node id
    levels: dict[str, tuple[float, float]] = {
        name: (0.0, 1.0) for name in netlist.inputs
    }
    for node in netlist.nodes:
        if node.logic == _CONST:
            buf = next((g for g in by_logic.get("BUFFER", [])), None)
            if buf is not None:
                assignment[node.id] = buf
            continue
        upstream = [levels[i] for i in node.inputs]
        candidates = by_logic.get(node.logic) or by_logic.get("BUFFER", [])
        best: CharacterizedGate | None = None
        best_score = -float("inf")
        for cand in candidates:
            s = score_gate(cand, node, upstream)
            if s > best_score:
                best_score = s
                best = cand
        if best is None:
            raise ValueError(f"no gate in library for logic {node.logic!r}")
        assignment[node.id] = best
        levels[node.id] = _gate_high_low(best)
    return assignment


# ============================================================================
# DNA assembly + predicted dynamics
# ============================================================================

def assemble_dna(netlist: Netlist,
                 assignment: dict[str, CharacterizedGate]) -> tuple[str, list[str]]:
    """Assemble the circuit DNA (topological order) and list gate parts.

    Returns ``(dna, gate_order)`` where ``gate_order`` lists the gate
    ids in assembly order (skipping CONST nodes).
    """
    parts: list[str] = []
    order: list[str] = []
    for node in netlist.topo_order():
        gate = assignment.get(node.id)
        if gate is None or node.logic == _CONST:
            continue
        parts.append(gate.dna)
        order.append(node.id)
    return "".join(parts), order


def simulate_netlist(netlist: Netlist,
                     assignment: dict[str, CharacterizedGate],
                     input_levels: tuple[float, ...],
                     max_iter: int = 2000, tol: float = 1e-9,
                     ) -> dict[str, float]:
    """Iterate the assigned transfer functions to steady state (Cello
    predicted-dynamics step).

    Returns a dict of ``{signal: level}`` for every netlist node and
    circuit input.
    """
    levels: dict[str, float] = {name: float(l)
                                for name, l in zip(netlist.inputs, input_levels, strict=True)}
    for node in netlist.nodes:
        gate = assignment.get(node.id)
        if node.logic == _CONST:
            levels[node.id] = 1.0 if node.value else 0.0
        elif gate is None:
            levels[node.id] = 0.0
        else:
            levels[node.id] = 0.0
    for _ in range(max_iter):
        delta = 0.0
        for node in netlist.nodes:
            gate = assignment.get(node.id)
            if node.logic == _CONST:
                new = 1.0 if node.value else 0.0
            elif gate is None:
                new = 0.0
            else:
                new = gate.transfer([levels[i] for i in node.inputs])
            d = abs(new - levels[node.id])
            if d > delta:
                delta = d
            levels[node.id] = new
        if delta < tol:
            break
    return levels


def simulate_truth_table(netlist: Netlist,
                         assignment: dict[str, CharacterizedGate],
                         ) -> TruthTable:
    """Predict the observed truth table from the simulated dynamics."""
    rows: list[tuple[tuple[bool, ...], tuple[bool, ...]]] = []
    for combo in itertools.product((False, True), repeat=len(netlist.inputs)):
        levels = simulate_netlist(
            netlist, assignment,
            tuple(1.0 if b else 0.0 for b in combo))
        outv = tuple(levels[o] > 0.5 for o in netlist.outputs)
        rows.append((combo, outv))
    return TruthTable(netlist.inputs, netlist.outputs, rows)


# ============================================================================
# Top-level design workflow
# ============================================================================

@dataclass(slots=True)
class BooleanCircuitDesign:
    """A completed Cello-style design (truth table -> DNA + SBOL3)."""

    truth_table: TruthTable
    netlist: Netlist
    assignment: dict[str, CharacterizedGate]
    dna: str
    gate_order: list[str]
    sbol3_xml: str
    predicted: TruthTable

    @property
    def matches_target(self) -> bool:
        """Whether the predicted dynamics reproduce the target truth table."""
        return self.predicted.rows == self.truth_table.rows

    def sbol3_component_definitions(self) -> list[dict]:
        """Structured SBOL3 payload (exported via :func:`sbol3_dumps`)."""
        features: list[dict] = []
        for node_id in self.gate_order:
            gate = self.assignment[node_id]
            features.append({
                "display_id": node_id,
                "role": SBOL_ROLE_GENE,
                "sequence": gate.dna,
            })
            features.append({
                "display_id": f"{node_id}_promoter",
                "role": SBOL_ROLE_PROMOTER,
                "sequence": gate.parts["promoter"],
            })
            features.append({
                "display_id": f"{node_id}_rbs",
                "role": SBOL_ROLE_RBS,
                "sequence": gate.parts["rbs"],
            })
            features.append({
                "display_id": f"{node_id}_terminator",
                "role": SBOL_ROLE_TERMINATOR,
                "sequence": gate.parts["terminator"],
            })
        return [{
            "display_id": "BooleanCircuit",
            "name": f"circuit_{'_'.join(self.truth_table.outputs)}",
            "role": "http://sbols.org/v3#EngineeredRegion",
            "components": features,
        }]


def compile_boolean_circuit(table: TruthTable,
                            library: list[CharacterizedGate] | None = None,
                            ) -> BooleanCircuitDesign:
    """Compile a truth table into a full DNA + SBOL3 design (T2.3).

    Runs synthesis, gate assignment, DNA assembly, SBOL3 export and the
    predicted-dynamics check end to end.

    Args:
        table: the target truth table.
        library: characterized gate library (defaults to
            :data:`GATE_LIBRARY`).

    Returns:
        a :class:`BooleanCircuitDesign`; check ``.matches_target`` and
        compare ``.predicted`` against ``.truth_table``.
    """
    netlist = synthesize_netlist(table)
    assignment = assign_gates(netlist, library)
    dna, order = assemble_dna(netlist, assignment)
    predicted = simulate_truth_table(netlist, assignment)
    design = BooleanCircuitDesign(
        truth_table=table,
        netlist=netlist,
        assignment=assignment,
        dna=dna,
        gate_order=order,
        sbol3_xml="",
        predicted=predicted,
    )
    cds = design.sbol3_component_definitions()
    design.sbol3_xml = sbol3_dumps(cds)
    return design


#: canonical convenience circuits
def not_gate() -> BooleanCircuitDesign:
    """Compile a single NOT gate (one input, one output)."""
    return compile_boolean_circuit(TruthTable.from_function(
        ["a"], ["y"], lambda v: (not v[0],)))


def nand_gate() -> BooleanCircuitDesign:
    return compile_boolean_circuit(TruthTable.from_function(
        ["a", "b"], ["y"], lambda v: (not (v[0] and v[1]),)))


def xor_gate() -> BooleanCircuitDesign:
    return compile_boolean_circuit(TruthTable.from_function(
        ["a", "b"], ["y"], lambda v: (v[0] != v[1],)))


# ============================================================================
# Closed-loop workflow: logic -> plasmid + time curves + validation
# ============================================================================

def build_plasmid(design: BooleanCircuitDesign,
                  backbone: str = "pUC19",
                  marker: str = "AmpR",
                  include_mcs: bool = True) -> tuple[str, int, list[str]]:
    """Assemble the circuit into a complete plasmid (Cello -> wet lab).

    The gate DNAs are cloned into a synthetic vector backbone built from
    :mod:`helixlang.apps.synbio_designer` parts: replicon + resistance
    marker + multiple cloning site, then the circuit gates in topological
    order (``synbio_designer.py`` "expression box / vector" layer).

    Returns ``(plasmid_dna, length, gate_order)``.
    """
    parts: list[str] = [ORIGIN_SEQUENCES[backbone], SELECTION_MARKERS[marker]]
    if include_mcs:
        parts.append("".join(MCS_SITES[s] for s in DEFAULT_MCS))
    gate_dna, order = assemble_dna(design.netlist, design.assignment)
    parts.append(gate_dna)
    full = "".join(parts)
    return full, len(full), order


def _plasmid_features(design: BooleanCircuitDesign,
                      backbone: str, marker: str,
                      gate_order: list[str]) -> list[dict]:
    """1-based GenBank feature annotations for the assembled plasmid."""
    features: list[dict] = []
    pos = 1

    def add(label: str, seq: str, ftype: str) -> None:
        nonlocal pos
        features.append({"type": ftype, "start": pos,
                         "end": pos + len(seq) - 1, "strand": 1,
                         "label": label})
        pos += len(seq)

    add(f"ori_{backbone}", ORIGIN_SEQUENCES[backbone], "rep_origin")
    add(marker, SELECTION_MARKERS[marker], "CDS")
    add("MCS", "".join(MCS_SITES[s] for s in DEFAULT_MCS), "misc_feature")
    for node_id in gate_order:
        gate = design.assignment[node_id]
        add(node_id, gate.dna, "gene")
    return features


def simulate_expression_curves(netlist: Netlist,
                               assignment: dict[str, CharacterizedGate],
                               time_course_min: float = 60.0,
                               time_step_min: float = 5.0,
                               ) -> dict[str, dict]:
    """Simulated expression time course of every gate's output protein.

    Uses the E. coli coupled transcription-translation model
    (:func:`helixlang.central_dogma.coupled_transcription_translation`)
    on each assigned gate's CDS, driven by the gate's characterized
    steady-state on level (Hill transfer) as promoter strength.  Returns
    ``{node_id: result_dict}`` where each result carries ``time_course``
    (list of TimeCoursePoint) and ``protein``.
    """
    curves: dict[str, dict] = {}
    for node in netlist.nodes:
        gate = assignment.get(node.id)
        if gate is None or node.logic == _CONST:
            continue
        promoter_strength = gate.transfer([1.0] * gate.num_inputs)
        curves[node.id] = coupled_transcription_translation(
            gate.parts["cds"],
            promoter_strength=promoter_strength,
            time_course_min=time_course_min,
            time_step_min=time_step_min,
        )
    return curves


@dataclass(slots=True)
class CelloWorkflowReport:
    """One-stop report: logic table -> DNA -> plasmid -> SBOL3 -> dynamics.

    Wraps a :class:`BooleanCircuitDesign` with the assembled plasmid, the
    GenBank export, the simulated expression time curves and a validation
    summary (the Cello "predicted dynamics" closed loop).
    """

    design: BooleanCircuitDesign
    plasmid_dna: str
    plasmid_length: int
    genbank: str
    time_curves: dict[str, dict]
    validation: dict

    @property
    def matches_target(self) -> bool:
        return bool(self.validation.get("matches_target"))

    @property
    def truth_table(self) -> TruthTable:
        return self.design.truth_table

    @property
    def netlist(self) -> Netlist:
        return self.design.netlist

    @property
    def assignment(self) -> dict[str, CharacterizedGate]:
        return self.design.assignment

    @property
    def gate_order(self) -> list[str]:
        return self.design.gate_order

    @property
    def sbol3_xml(self) -> str:
        return self.design.sbol3_xml


def run_cello_workflow(table: TruthTable,
                       library: list[CharacterizedGate] | None = None,
                       backbone: str = "pUC19",
                       marker: str = "AmpR",
                       include_mcs: bool = True,
                       time_course_min: float = 60.0,
                       time_step_min: float = 5.0,
                       ) -> CelloWorkflowReport:
    """Run the full Cello-style closed loop in one call.

    truth table -> netlist -> gate assignment -> DNA -> full plasmid +
    GenBank -> SBOL3 export -> predicted dynamics -> expression time
    curves -> validation summary (the "design-to-report" pipeline of the
    SDA plan; Nielsen et al. 2016, Jones et al. 2022).
    """
    design = compile_boolean_circuit(table, library)
    plasmid_dna, plasmid_length, order = build_plasmid(
        design, backbone, marker, include_mcs)
    curves = simulate_expression_curves(
        design.netlist, design.assignment,
        time_course_min=time_course_min, time_step_min=time_step_min)
    features = _plasmid_features(design, backbone, marker, order)
    genbank = genbank_format(
        plasmid_dna, f"circuit_{'_'.join(table.outputs)}", features)
    validation = {
        "matches_target": design.matches_target,
        "predicted_matches_target": design.predicted.rows == table.rows,
        "gate_count": len(order),
        "plasmid_length": plasmid_length,
        "sbol3_component_count": len(design.sbol3_component_definitions()),
        "time_curve_count": len(curves),
    }
    return CelloWorkflowReport(
        design=design,
        plasmid_dna=plasmid_dna,
        plasmid_length=plasmid_length,
        genbank=genbank,
        time_curves=curves,
        validation=validation,
    )


__all__ = [
    "REPRESSOR_CDS", "RBS_SEQ",
    "TruthTable", "NetlistNode", "Netlist",
    "CharacterizedGate", "GATE_LIBRARY",
    "minimize_expression", "synthesize_netlist", "assign_gates",
    "assemble_dna", "simulate_netlist", "simulate_truth_table",
    "BooleanCircuitDesign", "compile_boolean_circuit",
    "build_plasmid", "simulate_expression_curves",
    "CelloWorkflowReport", "run_cello_workflow",
    "not_gate", "nand_gate", "xor_gate",
]
