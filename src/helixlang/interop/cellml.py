"""CellML interoperability (doc/42 Phase E, gap RT-6).

Parses a minimal CellML 1.x/2.x core document into a structured
user-authored ODE model that the Helix language can adopt as a model body
(Phase D).  Network/ODE-level constructs only:

- ``<component>`` holds ``<variable>`` elements; a variable with an
  ``initial_value`` (or ``value``) is a state or parameter,
- a MathML ``<apply><eq/><apply><diff/>...`` rate law is translated into an
  infix expression string (the same surface the ``#model`` DSL uses),
- a variable that is the target of a ``<diff>`` ODE is a **species**
  (ODE state); every other variable is a **parameter**.

The expression translator supports the core MathML content tokens (``cn``,
``ci``, ``plus``, ``minus``, ``times``, ``divide``, ``power`` and unary
negation) and emits ``pow(a,b)`` for ``power``.

Stdlib-only (``xml.etree``).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any

from helixlang.core.errors import BioError

_MATH_NS = "{http://www.w3.org/1998/Math/MathML}"


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(node: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in node if _localname(c.tag) == name]


def _child(node: ET.Element | None, name: str) -> ET.Element | None:
    if node is None:
        return None
    return next((c for c in node if _localname(c.tag) == name), None)


def _text(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


# ============================================================================
# MathML -> infix expression translation
# ============================================================================

_BINARY = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divide": "/",
}


def _mathml_to_infix(node: ET.Element) -> str:
    """Translate a MathML content subtree into an infix expression string."""
    tag = _localname(node.tag)
    if tag == "cn":
        return _text(node) or "0"
    if tag in ("ci", "csymbol"):
        return _text(node) or ""
    # <apply><fun/> args...</apply>
    children = list(node)
    if tag == "apply" and children:
        op = _localname(children[0].tag)
        args = [_mathml_to_infix(a) for a in children[1:]]
        if op in _BINARY:
            if not args:
                return "0"
            if op == "minus" and len(args) == 1:
                return f"-({args[0]})"
            return "(" + f" {_BINARY[op]} ".join(args) + ")"
        if op == "power":
            return f"pow({', '.join(args)})"
        if op == "eq":
            # not a numeric operation; fall through to first argument
            return args[0] if args else "0"
        if op == "diff":
            # a rate target: this node names the differentiated variable
            return args[1] if len(args) > 1 and args[1] else (args[0] if args else "")
        # unhandled operator: represent as a function call but keep it simple
        return args[0] if len(args) == 1 else f"({', '.join(args)})"
    return _text(node) or "0"


# ============================================================================
# CellML model parsing
# ============================================================================

def cellml_to_model(xml_text: str) -> dict[str, Any]:
    """Parse a minimal CellML document into a user-authored ODE model dict.

    Returns a dict with::

        {
          "model_id": str,
          "species":   {name: {"initial": float, "units": str}},
          "parameters":{name: float},
          "rates":     {species_name: "infix-expression-string"},
          "constants": {name: float},   # species with rate '' (no ODE)
        }

    A variable with an ``initial_value`` is a state; a variable with only a
    ``value`` is a parameter.  Rate laws come from ``<apply><eq/><diff/>...``
    MathML equations keyed by the differentiated variable.

    Args:
        xml_text: CellML document (model-level XML) as a string.

    Returns:
        the structured ODE model dict.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise BioError(f"malformed CellML XML: {exc}") from exc
    model = _child(root, "model")
    if model is None:
        model = root
    if root is not model and _localname(root.tag) != "cellml":
        # tolerate plain <model> roots too
        model = root

    model_id = model.get("name") or model.get("id") or "model"
    species: dict[str, dict[str, Any]] = {}
    parameters: dict[str, float] = {}
    rates: dict[str, str] = {}
    constants: dict[str, float] = {}

    # Collect variable declarations first (with a record of whether they are
    # ODE targets, filled in below while scanning math).
    targets: set[str] = set()
    declared: dict[str, dict[str, Any]] = {}
    for comp in _children(model, "component"):
        for var in _children(comp, "variable"):
            vname = var.get("name") or _text(_child(var, "name"))
            if not vname:
                continue
            units = var.get("units") or ""
            init = var.get("initial_value")
            value = var.get("value")
            declared[vname] = {
                "initial": float(init) if init is not None else None,
                "value": float(value) if value is not None else None,
                "units": units,
            }

    # Scan MathML rate laws.
    for eq in _walk(root):
        target = _rate_target(eq)
        if target is None:
            continue
        expr = _rate_expression(eq)
        targets.add(target)
        if expr:
            rates[target] = expr

    # Classify declared variables.
    for vname, info in declared.items():
        if vname in targets:
            species[vname] = {
                "initial": info["initial"] if info["initial"] is not None else 0.0,
                "units": info["units"],
            }
        elif info["value"] is not None:
            parameters[vname] = info["value"]
        elif info["initial"] is not None:
            constants[vname] = info["initial"]

    if not species and not rates:
        raise BioError("CellML document declares no ODE species/rate equations")

    return {
        "model_id": model_id,
        "species": species,
        "parameters": parameters,
        "rates": rates,
        "constants": constants,
    }


def _walk(node: ET.Element) -> Iterator[ET.Element]:
    """Depth-first iterate over a subtree."""
    yield node
    for child in node:
        yield from _walk(child)


def _rate_target(eq: ET.Element) -> str | None:
    """If ``eq`` is a rate law, return its differentiated target variable."""
    if _localname(eq.tag) != "apply" or not len(eq):
        return None
    op = _localname(eq[0].tag)
    if op != "eq":
        return None
    lhs = eq[1] if len(eq) >= 2 else None
    if lhs is None or _localname(lhs.tag) != "apply" or not len(lhs):
        return None
    inner = _localname(lhs[0].tag)
    if inner != "diff":
        return None
    bvar = _child(lhs, "bvar")
    cvar = _child(lhs, "ci")
    name = _text(cvar) if cvar is not None else (_text(bvar) if bvar is not None else None)
    return name


def _rate_expression(eq: ET.Element) -> str:
    """Return the RHS infix expression of a rate law equation."""
    if len(eq) < 3:
        return ""
    return _mathml_to_infix(eq[2])


# ============================================================================
# File helpers
# ============================================================================

def load_cellml(path: str) -> dict[str, Any]:
    """Load a CellML model from a file on disk."""
    try:
        with open(path, encoding="utf-8") as fh:
            return cellml_to_model(fh.read())
    except OSError as exc:
        raise BioError(f"could not read CellML file {path!r}: {exc}") from exc


__all__ = [
    "cellml_to_model",
    "load_cellml",
]
