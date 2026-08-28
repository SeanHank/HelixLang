# Doc 35 — Release Checklist

## Pre-release gates

All gates are automated by `release.py <version>`. Running the script
executes steps 1–6 and reports pass/fail for each gate.

### Code quality (automated by release.py)
- [x] Zero ruff errors (`ruff check src tests`)
- [x] Zero mypy errors (`mypy src/helixlang/`)
- [x] All tests pass (`pytest tests/ -n auto -q`) — parallelized via pytest-xdist

### Bytecode ABI (manual pre-checks)
- [x] OPCODE_VERSION = 1 frozen in bytecode.py
- [x] spec/bytecode-abi.md documents byte-level format
- [x] spec/vm-semantics.md documents execution model
- [x] `--check-bytecode-version` CLI flag works
- [x] bytecode roundtrip test passes (test_helixc.py)

### Validation (automated by release.py)
- [x] 67/67 benchmarks pass in validation/
- [x] validation/run_all.py runs without errors
- [x] validation/report.md generated with results
- [x] provenance attached to all SimResult outputs

### Documentation (manual pre-checks)
- [x] README.md rewritten with 5-minute proof
- [x] doc/00-overview.md has Layer 1/2/3 framing
- [x] All 36 docs cross-referenced
- [x] No orphaned documentation

### Product (manual pre-checks)
- [x] `pip install helixlang` works
- [x] `helixlang examples/02_lac_operon.helix` runs
- [x] `helix --check-bytecode-version` prints version
- [x] Python API: `from helixlang.sim_runtime import run` works

## Release steps (automated)

```bash
# Local release (gates + build)
python release.py <version>
```

### What the script does
1. **Syncs version** to `pyproject.toml`, `__init__.py`, `server/app.py`, `core/bytecode.py` comment
2. **Runs quality gates in parallel**: ruff, mypy, pytest (`-n auto`), validation, examples
3. **Syncs metrics** (test count, coverage, validation results) to README.md, README_PYPI.md, CONTRIBUTING.md
4. **Builds** sdist + wheel via `python -m build`

## Post-release

- [x] Monitor GitHub issues for 72 hours
- [x] Create GitHub release with changelog (automated by CI or `--gh-release`)
- [x] Update doc/34 success metrics table
