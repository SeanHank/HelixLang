"""Unit tests for web.serializers (L-system parsing + object serialization)."""
from helixlang.web.serializers import (
    _parse_lsystem_rules,
    _serialize_cell,
    _serialize_field,
    _serialize_grn,
    _serialize_morphology,
    _serialize_program_summary,
    _serialize_trace,
)


def test_parse_lsystem_rules_empty():
    assert _parse_lsystem_rules("") == {}
    assert _parse_lsystem_rules("   ") == {}


def test_parse_lsystem_rules_skips_empty_and_bad_segments():
    rules = _parse_lsystem_rules("F->F[+F]F;;X->FX")
    assert rules == {"F": "F[+F]F", "X": "FX"}


def test_parse_lsystem_rules_skips_no_separator():
    rules = _parse_lsystem_rules("F->F;NODLIM;X:FX")
    assert rules == {"F": "F", "X": "FX"}


def test_parse_lsystem_rules_skips_multi_char_symbol():
    rules = _parse_lsystem_rules("F->F;AB->F;X:FX")
    assert rules == {"F": "F", "X": "FX"}


def test_parse_lsystem_rules_colon_separator():
    assert _parse_lsystem_rules("F:F[-F]F") == {"F": "F[-F]F"}


class _Gene:
    def __init__(self, name, promoter, seqs):
        self.name = name
        self.promoter = promoter
        self.orf = [_Codon(seq) for seq in seqs]


class _Codon:
    def __init__(self, seq):
        self.seq = seq


class _Promoter:
    def __init__(self, name, strength):
        self.name = name
        self.strength = strength


class _Regulation:
    def __init__(self, source, target, strength):
        self.source = source
        self.target = target
        self.strength = strength


class _LSystemDecl:
    def __init__(self, axiom, angle, step):
        self.axiom = axiom
        self.angle = angle
        self.step = step


class _FieldDecl:
    def __init__(self, size, F, k, Du, Dv):
        self.size = size
        self.F = F
        self.k = k
        self.Du = Du
        self.Dv = Dv


class _Config:
    def __init__(self, ticks, table, ops, react):
        self.ticks = ticks
        self.table = table
        self.ops_per_tick = ops
        self.react_steps = react


class _Program:
    def __init__(self, field_decl=None, config=None):
        self.genes = [_Gene("gfp", "P1", ["ATG"])]
        self.promoters = [_Promoter("P1", 1.0)]
        self.regulations = [_Regulation("a", "b", -1.5), _Regulation("c", "d", 2.0)]
        self.lsystems = {"plant": _LSystemDecl("F", 25.0, 1.0)}
        self.field_decl = field_decl
        self.config = config or _Config(10, "standard", 64, 1)


def test_serialize_program_summary_with_field():
    p = _Program(field_decl=_FieldDecl(5, 0.035, 0.065, 0.16, 0.08))
    s = _serialize_program_summary(p)
    assert len(s["genes"]) == 1
    assert s["has_field"] is True
    assert s["field"]["size"] == 5
    assert len(s["regulations"]) == 2
    assert s["regulations"][0]["strength"] == -1.5


def test_serialize_program_summary_without_field():
    p = _Program()
    s = _serialize_program_summary(p)
    assert s["has_field"] is False
    assert s["field"] is None


class _GrnNode:
    def __init__(self, level, threshold):
        self.level = level
        self.threshold = threshold


class _GrnEdge:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight


class _Grn:
    def __init__(self):
        self.nodes = {"a": _GrnNode(0.5, 0.2), "b": _GrnNode(0.1, 0.5)}
        self.edges = [_GrnEdge("a", "b", 1.0), _GrnEdge("b", "a", -0.5)]


class _VM:
    def __init__(self, grn=None, field=None, cell=None):
        self.grn = grn or _Grn()
        self.field = field
        self.cell = cell


def test_serialize_grn():
    s = _serialize_grn(_VM())
    assert len(s["nodes"]) == 2
    assert s["edges"][0]["sign"] == "positive"
    assert s["edges"][1]["sign"] == "negative"


def test_serialize_trace():
    trace = [
        {"tick": 0, "energy": 100.0, "morphology_points_count": 2,
         "field_total_v": 0.1, "gene_levels": {"a": 0.5},
         "proteins": {1: 2.0}},
        {"tick": 1, "energy": 90.0, "morphology_points_count": 3,
         "field_total_v": 0.2, "gene_levels": {"a": 0.8, "b": 1.0},
         "proteins": {1: 3.0, 2: 1.0}},
    ]
    s = _serialize_trace(trace)
    assert s["ticks"] == [0, 1]
    assert set(s["gene_levels"]) == {"a", "b"}
    assert s["proteins"]["1"] == [2.0, 3.0]


class _Cell:
    def __init__(self):
        self.name = "c1"
        self.x = 1
        self.y = 2
        self.energy = 50.0
        self.alive = True
        self.color = (1, 0, 0)
        self.age = 3
        self.divisions = 2
        self.proteins = {1: 2.5, 2: 0.0}
        self.slots = [object(), None, object()]
        self.morphology_points = [(1.0, 2.0), (3.0, 4.5)]


def test_serialize_morphology():
    vm = _VM(cell=_Cell())
    s = _serialize_morphology(vm)
    assert s["count"] == 2
    assert s["points"][1] == [3.0, 4.5]


def test_serialize_field():
    class _Field:
        n = 2
        u = [[0.5, 1.0], [0.0, 0.5]]
        v = [[0.1, 0.2], [0.3, 0.4]]

    vm = _VM(field=_Field())
    s = _serialize_field(vm)
    assert s["n"] == 2
    assert len(s["u"]) == 2


def test_serialize_field_none():
    assert _serialize_field(_VM(field=None)) is None


def test_serialize_cell():
    vm = _VM(cell=_Cell())
    s = _serialize_cell(vm)
    assert s["name"] == "c1"
    assert s["color"] == [1, 0, 0]
    assert s["proteins"] == {"1": 2.5, "2": 0.0}
    assert s["slots_used"] == 2
