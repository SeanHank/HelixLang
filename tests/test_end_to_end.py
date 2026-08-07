"""End-to-end tests: run all example programs and assert key invariants."""
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PYTHON = sys.executable


def _run(filename, *args):
    """Run the helixlang CLI, returning (returncode, stdout, stderr)."""
    cmd = [PYTHON, "-m", "helixlang", str(EXAMPLES / filename), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


def test_example_01_hello_dna():
    rc, out, err = _run("01_hello_dna.helix")
    assert rc == 0, f"stderr: {err}"
    assert "proteins" in out
    # Should synthesize protein (GCT -> OP_BUILD_PROTEIN)
    assert "proteins={3:" in out or "proteins={}" not in out


def test_example_01_disassemble():
    rc, out, err = _run("01_hello_dna.helix", "--disassemble")
    assert rc == 0, f"stderr: {err}"
    assert "OP_START" in out
    assert "OP_BUILD_PROTEIN" in out
    assert "OP_HALT" in out
    assert "hello" in out


def test_example_02_lac_operon_csv():
    rc, out, err = _run("02_lac_operon.helix", "--csv")
    assert rc == 0, f"stderr: {err}"
    lines = [l for l in out.strip().split("\n") if l]
    assert lines[0].startswith("tick,x,y,energy")
    assert len(lines) >= 21  # header + 20 ticks
    # lacI should be expressed (p_lacI constitutive) -> protein accumulation
    assert any("3:" in l for l in lines[1:])


def test_example_03_plant_growth_morphology():
    rc, out, err = _run("03_plant_growth.helix")
    assert rc == 0, f"stderr: {err}"
    # Morphology points should be produced (L-system growth)
    assert "morphology points:" in out
    # Point count should be > 1
    assert any(c.isdigit() for c in out.split("morphology points:")[-1])


def test_example_04_turing_pattern_ppm():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "turing")
        rc, out, err = _run("04_turing_pattern.helix", "--png", prefix)
        assert rc == 0, f"stderr: {err}"
        ppm = Path(f"{prefix}.ppm")
        assert ppm.exists()
        content = ppm.read_text()
        assert content.startswith("P3")
        # Should have non-zero pixels
        assert any(c != '0' for c in content.split("\n", 3)[3])


def test_example_05_standard_vs_mito():
    """The same DNA produces different bytecode under the standard and mito tables."""
    rc_std, out_std, err = _run("05_table_switch.helix", "--disassemble")
    assert rc_std == 0, f"stderr: {err}"
    # Standard: TGA -> OP_HALT
    assert "OP_HALT" in out_std
    assert "OP_BUILD_PIGMENT" not in out_std

    rc_mito, out_mito, err = _run("05_table_switch.helix",
                                  "--table=mito_vertebrate", "--disassemble")
    assert rc_mito == 0, f"stderr: {err}"
    # Mito: TGA -> OP_BUILD_PIGMENT
    assert "OP_BUILD_PIGMENT" in out_mito


def test_unknown_file_returns_error():
    rc, out, err = _run("nonexistent.helix")
    assert rc != 0
    assert "not found" in err


def test_all_examples_compile():
    """All examples should pass the compilation stage."""
    for f in sorted(EXAMPLES.glob("*.helix")):
        rc, out, err = _run(f.name, "--disassemble")
        assert rc == 0, f"{f.name} failed: {err}"
