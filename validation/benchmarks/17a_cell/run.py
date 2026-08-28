#!/usr/bin/env python3
"""Benchmark 17a: Cell class — import, instantiate, basic operations."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "17a_cell"}
    try:
        from helixlang.plugins.runtime.cell import (
            CELL_PROTEIN_SLOT_COUNT,
            DEFAULT_CELL_COLOR,
            FEED_ENERGY_AMOUNT,
            INITIAL_CELL_ENERGY,
            MIN_DIVISION_ENERGY,
            MOVE_ENERGY_COST,
            Cell,
        )

        # Instantiation
        cell = Cell(name="bench-cell", x=5, y=10)
        assert cell.alive is True
        assert cell.energy == INITIAL_CELL_ENERGY
        assert cell.x == 5 and cell.y == 10

        # Protein operations
        cell.add_protein("LacZ", 5.0)
        assert cell.proteins["LacZ"] == 5.0
        consumed = cell.consume_protein("LacZ", 2.0)
        assert consumed == 2.0
        assert cell.proteins["LacZ"] == 3.0

        # Energy operations
        fed = cell.consume_energy(MOVE_ENERGY_COST)
        assert fed is True
        assert cell.energy == INITIAL_CELL_ENERGY - MOVE_ENERGY_COST

        # Feed
        cell.feed(FEED_ENERGY_AMOUNT)
        assert cell.energy > INITIAL_CELL_ENERGY - MOVE_ENERGY_COST

        # Movement
        cell.move(0)  # North
        assert cell.y == 9

        # Division
        cell2 = Cell(name="div-cell", energy=MIN_DIVISION_ENERGY * 2)
        divided = cell2.divide()
        assert divided is True
        assert cell2.energy == MIN_DIVISION_ENERGY

        # Membrane permeability
        cell.set_membrane_permeability(128)
        assert cell.membrane_permeability == 128
        cell.set_membrane_permeability(-10)
        assert cell.membrane_permeability == 0

        # Constants sanity
        assert INITIAL_CELL_ENERGY > 0
        assert CELL_PROTEIN_SLOT_COUNT == 256
        assert DEFAULT_CELL_COLOR == (255, 255, 255)

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "validation": {
                "instantiation": True,
                "protein_ops": True,
                "energy_ops": True,
                "feed": True,
                "movement": True,
                "division": True,
                "membrane": True,
            },
            "constants": {
                "INITIAL_CELL_ENERGY": INITIAL_CELL_ENERGY,
                "CELL_PROTEIN_SLOT_COUNT": CELL_PROTEIN_SLOT_COUNT,
                "MIN_DIVISION_ENERGY": MIN_DIVISION_ENERGY,
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
