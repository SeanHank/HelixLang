"""Synthetic design automation tests: Boolean -> DNA + SBOL3 (T2.3, P4).

Verification goals:
- Truth tables enumerate correctly and produce correct minterms.
- Quine-McCluskey-style minimization merges terms exactly for small
  input counts.
- Boolean netlists evaluate to the target function on all inputs.
- Cello-style gate assignment picks characterized library gates whose
  transfer functions reproduce the required logic (scoring margin).
- Assembled DNA is a concatenation of promoter+RBS+CDS+terminator.
- Predicted dynamics (iterated transfer functions) reproduce the target
  truth table for NOT/NAND/XOR/OR/NOR/AND.
- The design round-trips through SBOL3 export (:mod:`helixlang.interop`).

References:
- Nielsen AA et al. Science 2016 352:aac7341 (Cello: Boolean -> DNA)
- Jones TS et al. Nat Protoc 2022 17:1097-1113 (Cello 2.0)
- LOICA 2022 ACS Synth Biol 11:4049-4063 (genetic circuit compiler)
"""
from __future__ import annotations

from helixlang.apps.synbio_automation import (
    GATE_LIBRARY,
    RBS_SEQ,
    REPRESSOR_CDS,
    BooleanCircuitDesign,
    CharacterizedGate,
    Netlist,
    NetlistNode,
    TruthTable,
    assemble_dna,
    assign_gates,
    boolean_apply,
    compile_boolean_circuit,
    minimize_expression,
    nand_gate,
    not_gate,
    simulate_netlist,
    simulate_truth_table,
    synthesize_netlist,
    xor_gate,
)
from helixlang.interop import (
    SBOL_ROLE_GENE,
    SBOL_ROLE_PROMOTER,
    SBOL_ROLE_RBS,
    SBOL_ROLE_TERMINATOR,
    sbol3_loads,
)

# ============================================================================
# Boolean helpers + truth tables
# ============================================================================

def test_boolean_apply_primitives() -> None:
    assert boolean_apply("NOT", [True]) is False
    assert boolean_apply("BUFFER", [False]) is False
    assert boolean_apply("AND", [True, True]) is True
    assert boolean_apply("AND", [True, False]) is False
    assert boolean_apply("NAND", [True, True]) is False
    assert boolean_apply("NAND", [False, True]) is True
    assert boolean_apply("OR", [False, True]) is True
    assert boolean_apply("OR", [False, False]) is False
    assert boolean_apply("NOR", [False, False]) is True
    assert boolean_apply("CONST", [True]) is True


def test_truth_table_enumerates_all_rows() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] and v[1],))
    assert len(tt.rows) == 4
    assert tt.to_truth_value((True, False)) is False
    assert tt.to_truth_value((True, True)) is True


def test_minimize_expression_merges() -> None:
    # (a AND b) OR (a AND NOT b) -> a
    terms = minimize_expression(
        ["a", "b"],
        [{"a": True, "b": True}, {"a": True, "b": False}])
    assert terms == [{"a": True}]


def test_minimize_expression_xor_not_mergeable() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] != v[1],))
    terms = minimize_expression(tt.inputs, tt.minterms())
    # XOR has no adjacent minterms: stays as two 2-literal terms
    assert len(terms) == 2
    assert all(len(t) == 2 for t in terms)


# ============================================================================
# Netlist synthesis + evaluation
# ============================================================================

def test_netlist_synthesis_not() -> None:
    tt = TruthTable.from_function(["a"], ["y"], lambda v: (not v[0],))
    net = synthesize_netlist(tt)
    assert net.evaluate((False,)) == (True,)
    assert net.evaluate((True,)) == (False,)


def test_netlist_synthesis_nand() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (not (v[0] and v[1]),))
    net = synthesize_netlist(tt)
    for combo in ((False, False), (False, True), (True, False), (True, True)):
        assert net.evaluate(combo) == (not (combo[0] and combo[1]),)


def test_netlist_synthesis_xor() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] != v[1],))
    net = synthesize_netlist(tt)
    for combo in ((False, False), (False, True), (True, False), (True, True)):
        assert net.evaluate(combo) == (combo[0] != combo[1],)


def test_netlist_constant_functions() -> None:
    always = TruthTable.from_function(["a"], ["y"], lambda v: (True,))
    net = synthesize_netlist(always)
    assert net.evaluate((False,)) == (True,)
    assert net.evaluate((True,)) == (True,)


def test_netlist_topo_order() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] != v[1],))
    net = synthesize_netlist(tt)
    order = net.topo_order()
    ids = [n.id for n in order]
    assert len(ids) == len(set(ids))


# ============================================================================
# Characterized gate library
# ============================================================================

def test_gate_library_covers_logics() -> None:
    logics = {g.logic for g in GATE_LIBRARY}
    assert {"NOT", "NAND", "NOR", "OR", "AND", "BUFFER"} <= logics
    assert len(GATE_LIBRARY) == 15


def test_gate_transfer_separation() -> None:
    for gate in GATE_LIBRARY:
        low, high = gate.output_levels()
        assert low < high
        assert high - low > 0.4
        assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_not_gate_low_output_is_deep() -> None:
    not_laci = next(g for g in GATE_LIBRARY if g.id == "NOT_lacI")
    low, high = not_laci.output_levels()
    assert low < 0.1  # strong repression keeps the LOW level near leak


def test_gate_dna_assembly() -> None:
    gate = next(g for g in GATE_LIBRARY if g.id == "NOT_lacI")
    assert gate.dna.startswith(gate.parts["promoter"])
    assert gate.parts["rbs"] == RBS_SEQ
    assert gate.parts["cds"] == REPRESSOR_CDS["lacI"]
    assert gate.dna.endswith(gate.parts["terminator"])


def test_gate_num_inputs() -> None:
    not_g = next(g for g in GATE_LIBRARY if g.logic == "NOT")
    nor_g = next(g for g in GATE_LIBRARY if g.logic == "NOR")
    assert not_g.num_inputs == 1
    assert nor_g.num_inputs == 2


# ============================================================================
# Gate assignment
# ============================================================================

def test_assignment_uses_matching_logic() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (not (v[0] and v[1]),))
    net = synthesize_netlist(tt)
    assignment = assign_gates(net)
    for node in net.nodes:
        if node.logic == "CONST":
            continue
        assert assignment[node.id].logic in (node.logic, "BUFFER")


def test_assignment_reproducible() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] != v[1],))
    net = synthesize_netlist(tt)
    a1 = assign_gates(net)
    a2 = assign_gates(net)
    assert [g.id for g in a1.values()] == [g.id for g in a2.values()]


def test_score_gate_polarity() -> None:
    # a gate with wrong polarity scores worse than one with the right one
    not_g = next(g for g in GATE_LIBRARY if g.id == "NOT_lacI")
    buf_g = next(g for g in GATE_LIBRARY if g.id == "BUFFER_araC")
    assert not_g.transfer([0.0]) > not_g.transfer([1.0])
    assert buf_g.transfer([1.0]) > 0.5  # BUFFER would output HIGH for HIGH in


# ============================================================================
# Assembly + predicted dynamics
# ============================================================================

def test_assemble_dna_concatenates_gates() -> None:
    tt = TruthTable.from_function(["a"], ["y"], lambda v: (not v[0],))
    net = synthesize_netlist(tt)
    assignment = assign_gates(net)
    dna, order = assemble_dna(net, assignment)
    assert dna == "".join(assignment[nid].dna for nid in order)
    assert all(nid in assignment for nid in order)


def test_simulate_netlist_steady_state() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] != v[1],))
    net = synthesize_netlist(tt)
    assignment = assign_gates(net)
    levels = simulate_netlist(net, assignment, (1.0, 1.0))
    assert set(net.inputs) <= set(levels)


def test_simulate_truth_table_shape() -> None:
    tt = TruthTable.from_function(["a", "b"], ["y"],
                                  lambda v: (v[0] != v[1],))
    net = synthesize_netlist(tt)
    assignment = assign_gates(net)
    pred = simulate_truth_table(net, assignment)
    assert len(pred.rows) == 4


# ============================================================================
# End-to-end convenience circuits
# ============================================================================

def test_not_gate_design_matches() -> None:
    d = not_gate()
    assert d.matches_target
    assert d.predicted.rows == d.truth_table.rows


def test_nand_gate_design_matches() -> None:
    d = nand_gate()
    assert d.matches_target
    assert d.predicted.rows == d.truth_table.rows


def test_xor_gate_design_matches() -> None:
    d = xor_gate()
    assert d.matches_target
    assert d.predicted.rows == d.truth_table.rows


def test_extra_logics_match() -> None:
    circuits = {
        "OR": TruthTable.from_function(["a", "b"], ["y"],
                                       lambda v: (v[0] or v[1],)),
        "NOR": TruthTable.from_function(["a", "b"], ["y"],
                                        lambda v: (not (v[0] or v[1]),)),
        "AND": TruthTable.from_function(["a", "b"], ["y"],
                                        lambda v: (v[0] and v[1],)),
        "NAND3": TruthTable.from_function(["a", "b", "c"], ["y"],
                                          lambda v: (not (v[0] and v[1] and v[2]),)),
    }
    for name, tt in circuits.items():
        design = compile_boolean_circuit(tt)
        assert design.matches_target, f"{name} mismatch: {design.predicted.rows}"


def test_design_dna_has_gates() -> None:
    d = nand_gate()
    assert len(d.dna) > 0
    assert len(d.gate_order) >= 2  # NAND -> NOT a OR NOT b needs >= 2 gates


def test_design_sbol3_roundtrip() -> None:
    d = xor_gate()
    parsed = sbol3_loads(d.sbol3_xml)
    assert len(parsed) == 1
    top = parsed[0]
    assert top["display_id"] == "BooleanCircuit"
    roles = {f["role"] for f in top["components"]}
    assert SBOL_ROLE_GENE in roles
    assert SBOL_ROLE_PROMOTER in roles
    assert SBOL_ROLE_RBS in roles
    assert SBOL_ROLE_TERMINATOR in roles
    for f in top["components"]:
        assert f["sequence"]


def test_design_matches_target_property_uses_rows() -> None:
    d = not_gate()
    assert d.matches_target is True
    d.truth_table.rows = [((False,), (False,)), ((True,), (True,))]
    assert d.matches_target is False


def test_custom_library_overrides() -> None:
    custom = [g for g in GATE_LIBRARY if g.id in ("NOT_lacI", "BUFFER_araC")]
    tt = TruthTable.from_function(["a"], ["y"], lambda v: (not v[0],))
    design = compile_boolean_circuit(tt, library=custom)
    assert design.matches_target


# ============================================================================
# Misc / data sanity
# ============================================================================

def test_repressor_cds_are_dna() -> None:
    import re

    for protein, seq in REPRESSOR_CDS.items():
        assert protein in {"lacI", "tetR", "cI", "araC"}
        assert re.match(r"^[ACGT]+$", seq)
        assert seq.startswith("ATG")  # start codon


def test_design_is_dataclass() -> None:
    d = not_gate()
    assert isinstance(d, BooleanCircuitDesign)
    assert isinstance(d.netlist, Netlist)
    assert isinstance(d.assignment, dict)
    for node in d.netlist.nodes:
        assert isinstance(node, NetlistNode)
        assert isinstance(d.assignment[node.id], CharacterizedGate)
