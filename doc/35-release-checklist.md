# Doc 35 — Release Checklist (v0.1)

## Pre-release gates

### Code quality
- [ ] Zero ruff errors (`ruff check src/helixlang/ tests/ validation/`)
- [ ] Zero mypy errors (`mypy src/helixlang/`)
- [ ] All tests pass (`pytest tests/ -n8 -q`)
- [ ] No `# type: ignore` without corresponding `warn_unused_ignores = False` in mypy.ini

### Bytecode ABI
- [ ] OPCODE_VERSION = 1 frozen in bytecode.py
- [ ] spec/bytecode-abi.md documents byte-level format
- [ ] spec/vm-semantics.md documents execution model
- [ ] `--check-bytecode-version` CLI flag works
- [ ] bytecode roundtrip test passes (test_helixc.py)

### Validation
- [ ] 10+ benchmarks pass in validation/
- [ ] validation/run_all.sh runs without errors
- [ ] validation/report.md generated with results
- [ ] provenance attached to all SimResult outputs

### Documentation
- [ ] README.md rewritten with 5-minute proof
- [ ] doc/00-overview.md has Layer 1/2/3 framing
- [ ] All 35 docs cross-referenced
- [ ] No orphaned documentation

### Product
- [ ] `pip install helixlang` works
- [ ] `helix examples/02_lac_operon.helix` runs
- [ ] `helix --check-bytecode-version` prints version
- [ ] Python API: `from helixlang.sim_runtime import run` works

## Release steps

1. Bump version in pyproject.toml, __init__.py, server.py
2. Update validation/report.md with fresh results
3. Run full test suite: `pytest tests/ -n8 -q`
4. Create git tag: `git tag v0.1`
5. Build: `python -m build`
6. Publish: `twine upload dist/*`

## Post-release

- [ ] Monitor GitHub issues for 72 hours
- [ ] Create GitHub release with changelog
- [ ] Update doc/34 success metrics table
