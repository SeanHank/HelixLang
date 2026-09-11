"""IR serialization unit tests for the file-variant helpers and error edges.

Covers:
- dump() writing to a text file.
- load() reading back a full IR program.
- dumps()/loads() round-trip with an lsystem + use_directive + typed inst.
- from_dict rejecting a wrong-format payload.
- load refusing a too-new version and a bad DIM tag.
"""
from __future__ import annotations

import io

import pytest

from helixlang.core.codon_table import Op
from helixlang.core.dimensions import Dimension
from helixlang.core.ir import IRFunction, IRInst, IRProgram, IRType
from helixlang.core.ir_serialize import (
    IRFormatError,
    dump,
    dumps,
    from_dict,
    idim_to_dim,
    load,
    loads,
)


def _ir() -> IRProgram:
    inst = IRInst(opcode=Op.OP_PUSH_CONST, operand=42,
                  value_type=IRType.NUM, dim=Dimension(2, 1, 0, 0, 0, 0, 0),
                  line=3, codon_index=1)
    return IRProgram(
        name="p", table="standard",
        functions=[IRFunction(name="g", instrs=[inst])],
        call_targets={"g": "h"},
        use_directives=[("plug", ("f", "g"))],
        lsystems={"l": ("F", [], 25.0, 2.0)},
        config={"ticks": 5}, version=1,
    )


def test_dump_lt_file_roundtrip():
    buf = io.StringIO()
    dump(_ir(), buf)
    text = buf.getvalue()
    assert isinstance(text, str)
    # Reload from the serialized string and verify the function survives.
    loaded = loads(text)
    assert loaded.functions[0].name == "g"


def test_load_from_textio():
    text = dumps(_ir())
    loaded = load(io.StringIO(text))
    assert loaded.functions[0].instrs[0].operand == 42


def test_loads_roundtrip_metadata():
    loaded = loads(dumps(_ir()))
    assert loaded.call_targets == {"g": "h"}
    assert loaded.use_directives == [("plug", ("f", "g"))]
    assert "l" in loaded.lsystems


def test_from_dict_wrong_format():
    with pytest.raises(IRFormatError):
        from_dict({"fmt": "nope"})


def test_from_dict_too_new_version():
    with pytest.raises(IRFormatError):
        from_dict({"fmt": "hlir", "version": 99999})


def test_idim_to_dim_bad_length():
    with pytest.raises(IRFormatError):
        idim_to_dim([1, 2, 3])


def test_fn_from_dict_unprefixed_op_name():
    # A hand-authored payload may write "NOP" instead of "OP_NOP"; the reader
    # must prefix it.
    payload = {
        "fmt": "hlir", "version": 1,
        "functions": [{"name": "g", "instrs": [{"op": "NOP", "line": 1}]}],
    }
    loaded = from_dict(payload)
    assert loaded.functions[0].instrs[0].opcode is Op.OP_NOP

def test_inst_without_dim_serializes_without_dim_key():
    from helixlang.core.ir_serialize import _fn_from_dict, _inst_to_dict, to_dict
    inst = IRInst(opcode=Op.OP_NOP, line=0, codon_index=0)  # dim is None
    d = _inst_to_dict(inst)
    assert "dim" not in d

    # round-trips: to_dict -> from_dict keeps dim None
    prog = IRProgram(
        name="p", table="standard",
        functions=[IRFunction(name="g", instrs=[inst])],
        call_targets={}, use_directives=[], lsystems={}, config={}, version=1,
    )
    loaded = from_dict(to_dict(prog))
    assert loaded.functions[0].instrs[0].dim is None

    # _fn_from_dict with an unprefixed op-name path (no dim) too
    fn = _fn_from_dict({"name": "x",
                        "instrs": [{"op": "NOP"}]})
    assert fn.instrs[0].opcode is Op.OP_NOP
