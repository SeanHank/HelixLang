# Contributing to HelixLang

Thanks for your interest in HelixLang! This project treats **DNA as source code**
and ships a full compiler → bytecode → VM → simulator pipeline with real
biological data. Whether you are fixing a typo in the docs, adding a codon-table
opcode, or contributing a new biological module, your help is welcome.

Please take a moment to read this guide before opening an issue or pull request.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ground rules](#ground-rules)
- [Getting started](#getting-started)
  - [Environment](#environment)
  - [Installation](#installation)
- [Where things live](#where-things-live)
- [Finding something to work on](#finding-something-to-work-on)
- [Development workflow](#development-workflow)
  - [Branches](#branches)
  - [Commit messages](#commit-messages)
  - [Open a pull request](#open-a-pull-request)
- [Quality gates](#quality-gates)
  - [Tests & coverage](#tests--coverage)
  - [Lint](#lint)
  - [Type checking](#type-checking)
  - [Examples smoke test](#examples-smoke-test)
  - [Continuous integration](#continuous-integration)
- [Coding conventions](#coding-conventions)
- [Biological constants & citations](#biological-constants--citations)
- [Documentation policy](#documentation-policy)
- [Reviewing and merging](#reviewing-and-merging)
- [License & contribution terms](#license--contribution-terms)

---

## Code of Conduct

Be respectful and constructive. This project welcomes contributors of all
levels — from first-time PRs to compiler veterans. Reviewers give actionable,
kind feedback; authors treat it as a learning opportunity. Harassment or
abusive behavior is not tolerated.

## Ground rules

- **Preserve the public API.** External behavior (the `#config`/annotation
  syntax, module-level function signatures, CLI flags, REST endpoints) is
  stable. Behavior changes must be opt-in or additive.
- **Never silently change defaults.** Legacy behavior is the compatibility
  contract. New semantics belong behind new flags or clearly documented
  opt-ins. The runtime now runs on **physical units** end-to-end (no
  `calibrated=`/`units=` switch; see `doc/04-simulation-model.md` §6.3).
- **Data is sourced, not invented.** Every magic number that stands for a
  biological quantity must carry a citation — see
  [Biological constants & citations](#biological-constants--citations).
- **Docs travel with code.** If you change behavior, update the affected
  `doc/*.md` files in the same pull request.

## Getting started

### Environment

- Python **3.11** (exactly; `>=3.11,<3.12`).
- A venv or conda environment — anything works, but keep it isolated.

### Installation

```bash
git clone https://github.com/SeanHank/HelixLang.git
cd HelixLang

# Editable install with every optional extra
pip install -e ".[dev,grn,fba,pk,disease,annotation,gem,human,apps,web,ml,viz,bio,native]"
```

| Extra | What you get |
|-------|--------------|
| `dev` | pytest + pytest-cov + scipy (tests & coverage) |
| `grn` | numpy GRN kernels |
| `fba` | flux-balance analysis + ODE (numpy, scipy) |
| `pk` | pharmacokinetics (numpy, scipy) |
| `disease` | disease models (numpy, scipy) |
| `annotation` | sequence annotation (numpy, biopython, reedsolo) |
| `gem` | genome-scale metabolic models (numpy, cobra) |
| `human` | rdkit (SMILES parsing, drug simulation) |
| `apps` | visualization + pipelines (numpy, matplotlib) |
| `web` | Flask visualization frontend (`helixlang --serve`) |
| `ml` | ESM3 protein structure + ESM-2 kinetics |
| `viz` | matplotlib (plotting) |
| `bio` | legacy alias: biopython + reedsolo + cobra (DNA codec, GEM import) |
| `native` | compiled C/Cython/PyO3 accel (cython, setuptools, build) |

The **core compiler/VM has zero hard dependencies** — the standard library
only. Optional extras are genuinely optional; keep it that way.

## Where things live

```
src/helixlang/     The package: three layers —
                     • core/        minimal dependency-free compiler core
                       (lexer → parser → semantic → compiler → bytecode → vm)
                     • plugins/     biological runtime + scientific apps (lazy)
                     • sim_runtime/ server/ debugger/ interop/ web/ _accel/
                    169 source modules across core, plugins, and support packages
tests/             pytest suite (131 files, 3,315 tests) + shared conftest fixtures
examples/          runnable .helix programs (must always compile & run)
doc/               All technical documentation (37 files, kept in sync with code)
validation/        75 reproducible benchmarks with SHA256-verified golden outputs
.github/workflows/ CI: lint / typecheck / test / examples-smoke
```

Key entry points for contributors:

- `doc/03-compiler-design.md` — the compilation pipeline, AST, bytecode format.
- `doc/06-engineering-design.md` — module interfaces, invariants, error matrix.
- `doc/02-language-spec.md` — the authoritative language spec.
- `doc/08-api-reference.md` — per-module Python API reference.
- `doc/34-architectural-improvement-plan.md` — architecture plan + validation suite.
- `validation/` — 75 reproducible benchmarks with SHA256-verified golden outputs.
- `tests/conftest.py` — shared fixtures (Flask client, example sources, paths).

## Finding something to work on

- **Issues**: look for `good first issue` / `help wanted` labels.
- **TODO markers**: `grep -rn "TODO" src tests doc`.
- **Coverage gaps**: `pytest --cov=helixlang --cov-report=term-missing` and
  pick an uncovered branch.
- **Docs drift**: the project prides itself on docs that track the code; a PR
  that fixes stale docstrings or `doc/*.md` tables is always appreciated.

Not sure where to start? Open an issue describing what you'd like to do before
writing code — maintainers can point you at the right module and expected
design.

## Development workflow

### Branches

Fork the repository, then create a focused branch off `main`:

```bash
git checkout -b fix/quorum-threshold-doc     # fixes
git checkout -b feat/hill-cooperativity      # features
git checkout -b docs/update-overview         # documentation
```

Keep each PR to **one logical change**. Small PRs review faster and are less
likely to bit-rot.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) — it's the
style already used in this repo's history:

```
feat: add Hill-cooperativity regulation mode
fix: correct diffusion variance in sub-stepped diffusion
doc: document the physical unit system
test: cover population doubling time under physical units
```

Include *what* and *why* in the body when it isn't obvious from the subject.

### Open a pull request

1. Push your branch to your fork.
2. Open a PR against `main`.
3. Fill in the PR template (what changed, why, how it was tested).
4. Make sure all [quality gates](#quality-gates) pass locally before requesting
   review.

## Quality gates

Every PR must pass all four gates. They are exactly what CI runs, so running
them locally catches issues early.

### Tests & coverage

```bash
pytest --cov=helixlang --cov-report=term-missing --cov-fail-under=80
```

- Coverage gate is **80%** (the config lives in `pyproject.toml`; current suite
  measures ~90%+).
- Run a single file: `pytest tests/test_grn.py`
- Run a single test: `pytest tests/test_grn.py::test_decay_default_halves_at_110_ticks`

**When you add behavior, add tests.** New opcodes, bio modules, or physical-unit
constants must ship with a validation suite that asserts the *physical meaning*
of the numbers, not just "it doesn't crash".

### Lint

```bash
ruff check src tests
```

Rules: `E, F, W, I, UP, B` (see `ruff.toml`), line length 100, imports sorted.
`E501`/`E741`/`B008` are intentionally disabled — don't re-enable them.

### Type checking

```bash
mypy
```

Strict: `disallow_untyped_defs`, `check_untyped_defs`, `warn_return_any`
(all in `setup.cfg`). Annotate every function you touch. The Flask view module
(`server/app.py`) is the single exempted module — keep it that way.

### Examples smoke test

All `examples/*.helix` must compile and run:

```bash
for f in examples/*.helix; do
  python -m helixlang "$f" --disassemble >/dev/null || echo "FAIL: $f"
done
python -m helixlang examples/01_hello_dna.helix
```

### Continuous integration

`.github/workflows/ci.yml` runs, on every PR:

| Job | Python | Runs |
|-----|--------|------|
| `lint` | 3.11 | `ruff check src tests` |
| `typecheck` | 3.11 | `mypy` |
| `test` | 3.11 | `pytest --cov=helixlang --cov-fail-under=80` |
| `examples-smoke` | 3.11 | compile + run all examples |

Green CI is required before merge. If you can't reproduce a CI-only failure,
mention it in the PR — the matrix runs both supported Python versions.

### Releasing

Releases are handled by `release.py` at the repo root. It runs all gates in
parallel, syncs version strings, and builds sdist + wheel.

```bash
python release.py <version>
```

Version format: `YYYY.M.D` or `YYYY.M.D.N` (e.g. `2026.9.1`, `2026.9.1.2`).
- `D` — iteration release of this month, **starting from 0** (the month's first release is `YYYY.M.0`, the second is `YYYY.M.1`, …).
- `N` (optional) — patch version of that iteration release (`2026.9.1.2` = 2nd patch of the month's 2nd release).

**What `release.py` does internally:**

| Step | Action |
|------|--------|
| 1 | Validate version format, sync to `pyproject.toml`, `core/version.py`, `server/app.py` |
| 2 | Run gates in parallel: ruff, mypy, pytest (pytest-xdist), validation benchmarks, examples smoke test |
| 2b | Regenerate `validation/report.md` from fresh benchmark results |
| 3 | Sync metrics (test count, pass rate, modules, docs, examples) to README.md, README_PYPI.md, CONTRIBUTING.md |
| 4 | Build sdist + wheel into `dist/` |

All gates must pass before the build proceeds. If any gate fails the script
exits with a non-zero code and prints the failing gate's last 15 lines.

## Coding conventions

- **Follow the surrounding style.** Match the file you're editing: imports,
  `"""docstrings"""`, section banners (`# ===...`), and data-driven table
  layouts.
- **Type annotations are mandatory** in `src/` (enforced by mypy).
- **Named constants over magic literals.** Every hard-coded value that has a
  meaning (energy costs, decay rates, diffusion coefficients, quorum
  thresholds, coupling gains) belongs in a named module-level constant. Values
  are **physical** (ATP molecules, µM, µm²/s); constants live in
  `src/helixlang/core/units.py` or their owning module with a `#:` citation.
- **Stdlib-first core.** `src/helixlang` core must import only the standard
  library. numpy / biopython / flask are optional extras — gate imports
  (`try: import numpy as np`), never a hard dependency.
- **No silent fallbacks (doc/36 §3ξ).** A high-fidelity path (FBA model, cobra,
  rdkit, esm, native accel) must never silently degrade to a lower-fidelity
  proxy. When the high-fidelity path is absent, raise the typed error
  (`PluginDependencyError`/`ModelMissingError`/`NativeBackendError`) naming the
  component + the `pip install helixlang[<extra>]` fix, and gate any genuine
  reduced-fidelity mode behind an explicit capability flag (`--low-fidelity` /
  `--approx-euler` / `--pure-python`) via `core.fidelity`. The
  `find_silent_fallbacks` lint enforces this over `core/` + `plugins/`.
- **Performance matters.** The suite includes benchmarks (`tests/test_benchmark.py`)
  and a documented performance report. Avoid O(n²) hot loops; vectorize the
  big-field paths (`OP_REACT`/`OP_DIFFUSE`, population metabolism).
- **No dead code.** Delete the code you're replacing; don't leave commented-out
  alternatives.

## Biological constants & citations

This is the most important convention in HelixLang. The codebase models real
biology, and every parameter that stands for a physical quantity must be
**traceable to a published measurement**:

```python
#: energy required for one cell division, ATP molecules (Orth 2010:
#: 1.8e9 ~= 20 min of maintenance flux ~2.5e7 ATP/min)
DIVISION_ENERGY_THRESHOLD = 1.8e9
```

Rules of thumb:

- Cite the *primary* source (first author + year) at the point of definition,
  and add the full reference to `doc/01-references.md`.
- If you can't find a published value, say so explicitly ("gameplay units,
  not experimentally measured") rather than inventing one.
- Constants are **directly physical** — no conversion functions or calibration
  registry. To add one, define the named constant in its owning module and add
  the unit + citation to the docstring (see `src/helixlang/core/units.py` for the
  canonical example). The former `CALIBRATED` registry / `calibrated=` mode was
  removed; `doc/16-gameplay-units-upgrade.md` documents that history.
- When you change a physical value, update its tests together (e.g. a
  half-life change must be reflected in the
  `test_default_decay_halves_at_110_ticks`-style assertions).

## Documentation policy

The project states: *"docs and code are kept in sync; when they conflict, the
code prevails."* Practically, that means:

- **Docstrings** describe what a function does and why, with citations for
  biological constants.
- **`doc/*.md`** is user- and contributor-facing documentation. Behavior
  changes must update the affected documents in the same PR:
  - Language syntax → `doc/02-language-spec.md`
  - Bio annotations / `.helix` authoring → `doc/09-bio-instructions.md`
  - Simulation semantics → `doc/04-simulation-model.md`
  - Python API signatures → `doc/08-api-reference.md`
- If you notice stale docs while working on something else, fixing them in the
  same PR is appreciated — but call it out in the description.

## Scientific validation

Every simulation backend and scientific module must have an evidence chain:
**Reference → Expected range → Helix result → Error → Reproducibility**.

```bash
# Run the full validation suite
python validation/run_all.py

# Verify golden outputs
python validation/goldens/verify_goldens.py

# Regenerate goldens after code changes
python validation/goldens/generate_goldens.py
```

Current metrics (2026-08-27):
- **71/75** benchmarks PASS
- **40+** published references cited
- **0** non-deterministic failures
- **Median error**: ~3.0% (quantitative benchmarks vs published/analytical references)
- **Worst error**: 16.7% (population doubling time ratio deviation)

When adding a new backend or module, you must add a corresponding benchmark in
`validation/benchmarks/` with:
1. A `benchmark.yaml` defining the reference source, expected value, and tolerance
2. A `run.py` that exercises the code and outputs JSON with PASS/FAIL
3. A golden output in `validation/goldens/`

## Reviewing and merging

- PRs need **one approving review** from a maintainer.
- The reviewer checks: gates green, public API preserved (or intentionally
  changed with a release note), citations present for new constants, docs
  updated, tests meaningful.
- Maintainers follow the same standards as contributors — no rubber-stamping.

## License & contribution terms

HelixLang is licensed under the **GNU Affero General Public License v3.0**
(`LICENSE`, full text of AGPL-3.0). Copyright © 2026 Sean Hank.

By opening a pull request, you agree that your contribution is offered under
the project's license (inbound = outbound). If your contribution incorporates
third-party code or data, ensure its license is compatible with AGPL-3.0 and
note the attribution in the PR.

Questions about licensing, contributing, or anything else? Open an issue — we
prefer public discussion so the whole community benefits.
