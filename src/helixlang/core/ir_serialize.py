"""IR serialization: typed IR <-> HLIR (JSON).

Makes the IR a storable, portable artifact (doc/37 §4.2): a compiled program
can be dumped, diffed, versioned and rebuilt without re-parsing the DSL, and
the bytecode/runtime can be *derived* from it at any time.

``IR_VERSION`` (ir.VERSION) is embedded in the payload; ``load`` refuses
payloads with a newer version so the readers stay sound.
"""
from __future__ import annotations

import json
from typing import Any, TextIO

from helixlang.core.codon_table import Op
from helixlang.core.dimensions import Dimension
from helixlang.core.ir import (
    IR_VERSION,
    IRFunction,  # imported here for _fn_from_dict
    IRInst,
    IRProgram,
    IRType,
)

FMT_NAME = "hlir"


class IRFormatError(ValueError):
    """Raised when an HLIR payload is malformed or too-new."""


def idim_to_dim(data: list[int] | tuple[int, ...]) -> Dimension:
    """Debase a serialized DIM tag (7 SI exponents) into a Dimension."""
    if len(data) != 7:
        raise IRFormatError(f"DIM tag must have 7 exponents, got {data!r}")
    return Dimension(*data)


def dumps(ir: IRProgram, *, indent: int | None = 2) -> str:
    return json.dumps(to_dict(ir), indent=indent, sort_keys=True)


def dump(ir: IRProgram, file: TextIO) -> None:
    file.write(dumps(ir))


def loads(text: str) -> IRProgram:
    return from_dict(json.loads(text))


def load(file: TextIO) -> IRProgram:
    return loads(file.read())


def to_dict(ir: IRProgram) -> dict[str, Any]:
    return {
        "fmt": FMT_NAME,
        "version": ir.version,
        "name": ir.name,
        "table": ir.table,
        "functions": [_fn_to_dict(fn) for fn in ir.functions],
        "call_targets": dict(ir.call_targets),
        "use_directives": [[p, list(flags)] for p, flags in ir.use_directives],
        "lsystems": {name: list(decl) for name, decl in ir.lsystems.items()},
        "config": dict(ir.config),
    }


def _fn_to_dict(fn: IRFunction) -> dict[str, Any]:
    return {
        "name": fn.name,
        "line": fn.line,
        "instrs": [_inst_to_dict(i) for i in fn.instrs],
    }


def _inst_to_dict(inst: IRInst) -> dict[str, Any]:
    d: dict[str, Any] = {
        "op": inst.opcode.name,
        "operand": inst.operand,
        "value_type": inst.value_type.value if inst.value_type else None,
        "line": inst.line,
        "codon_index": inst.codon_index,
    }
    # doc/38 §8.2: IR dimensional metadata round-trips as a DIM tag; a reader
    # that predates the tag ignores it (payload version gate still applies).
    if inst.dim is not None:
        d["dim"] = list(inst.dim)
    return d


def from_dict(payload: dict[str, Any]) -> IRProgram:
    if payload.get("fmt") != FMT_NAME:
        raise IRFormatError(f"not an HLIR payload: fmt={payload.get('fmt')!r}")
    version = int(payload.get("version", 0))
    if version > IR_VERSION:
        raise IRFormatError(
            f"HLIR payload version {version} exceeds reader {IR_VERSION}")
    ir = IRProgram(
        name=payload.get("name", ""),
        table=payload.get("table", "standard"),
        call_targets={str(k): str(v)
                      for k, v in payload.get("call_targets", {}).items()},
        use_directives=[(str(p), tuple(flags))
                        for p, flags in payload.get("use_directives", [])],
        lsystems={str(k): tuple(v)
                  for k, v in payload.get("lsystems", {}).items()},
        config=payload.get("config", {}),
    )
    for fn in payload.get("functions", []):
        ir.functions.append(_fn_from_dict(fn))
    return ir


def _fn_from_dict(payload: dict[str, Any]) -> IRFunction:
    fn = IRFunction(name=str(payload.get("name", "")),
                    line=int(payload.get("line", 1)))
    for inst in payload.get("instrs", []):
        name = str(inst["op"])
        if not name.startswith("OP_"):
            name = "OP_" + name
        opcode = Op[name]
        vtype = inst.get("value_type")
        value_type = IRType.from_string(vtype) if vtype else None
        dim_data = inst.get("dim")
        dim = idim_to_dim(dim_data) if dim_data is not None else None
        fn.instrs.append(IRInst(
            opcode=opcode,
            operand=inst.get("operand"),
            value_type=value_type,
            dim=dim,
            line=int(inst.get("line", 0)),
            codon_index=int(inst.get("codon_index", -1)),
        ))
    return fn
