#!/usr/bin/env python3
"""Benchmark 16: CLI / server / provenance — module imports + build_provenance."""
from __future__ import annotations

import json
import sys
import time


def _test_provenance() -> tuple[bool, dict]:
    from helixlang.core.provenance import build_provenance

    prov = build_provenance(
        seed=42,
        backend="fba",
        parameters={"organism": "ecoli"},
        source="#gene test\nATG TAA\n#end",
        source_path="/tmp/test.helix",
        runtime_seconds=1.23,
        extra={"custom_key": "custom_value"},
    )

    required_fields = {"helix_version", "seed", "backend", "parameters", "dependencies", "timestamp"}
    assert required_fields.issubset(prov.keys()), (
        f"missing fields: {required_fields - prov.keys()}"
    )
    assert prov["seed"] == 42
    assert prov["backend"] == "fba"
    assert prov["parameters"] == {"organism": "ecoli"}
    assert "source_hash" in prov
    assert prov["source_path"] == "/tmp/test.helix"
    assert prov["runtime_seconds"] == 1.23
    assert prov["custom_key"] == "custom_value"
    return True, {
        "fields": sorted(prov.keys()),
        "seed": prov["seed"],
        "backend": prov["backend"],
        "has_source_hash": "source_hash" in prov,
    }


def _test_cli_import() -> tuple[bool, dict]:
    from helixlang.cli import main
    assert callable(main)
    return True, {"main_callable": True}


def _test_cli_flags() -> tuple[bool, dict]:
    import contextlib
    import io

    from helixlang.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            main(["--check-bytecode-version"])
        except SystemExit:
            pass
    output = buf.getvalue()
    assert "OPCODE_VERSION=" in output, f"--check-bytecode-version did not print version, got: {output!r}"
    return True, {"version_output": output.strip()}


def _test_server_import() -> tuple[bool, dict]:
    from helixlang.server import run_server
    assert callable(run_server)
    return True, {"run_server_callable": True}


def _test_bytecode_version() -> tuple[bool, dict]:
    from helixlang.core.bytecode import OPCODE_VERSION
    assert isinstance(OPCODE_VERSION, int)
    assert OPCODE_VERSION >= 1
    return True, {"OPCODE_VERSION": OPCODE_VERSION}


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "16_cli_server_provenance"}
    try:
        checks = {}

        ok_p, info_p = _test_provenance()
        checks["provenance"] = info_p
        assert ok_p, f"provenance check failed: {info_p}"

        ok_c, info_c = _test_cli_import()
        checks["cli_import"] = info_c
        assert ok_c, f"CLI import failed: {info_c}"

        ok_f, info_f = _test_cli_flags()
        checks["cli_flags"] = info_f
        assert ok_f, f"CLI flags check failed: {info_f}"

        ok_s, info_s = _test_server_import()
        checks["server_import"] = info_s
        assert ok_s, f"server import failed: {info_s}"

        ok_b, info_b = _test_bytecode_version()
        checks["bytecode_version"] = info_b
        assert ok_b, f"bytecode version check failed: {info_b}"

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "checks": checks,
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
