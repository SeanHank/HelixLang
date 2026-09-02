"""HelixLang web visualization server (Flask).

Usage:
    python -m helixlang --serve [--port 5000]
    helixlang --serve

Provides REST API:
    GET  /                       frontend single page
    GET  /api/health             health check
    GET  /api/examples           list example files
    GET  /api/examples/<name>    read example source
    POST /api/compile            compile source, return disassemble + AST summary
    POST /api/run                compile and run, return trace + GRN + morphology data
"""
from __future__ import annotations

import binascii
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flask import Flask, jsonify, request, send_from_directory

if TYPE_CHECKING:
    from helixlang.core.ast_nodes import Program
    from helixlang.core.bytecode import Chunk

from helixlang.core.compiler import Compiler
from helixlang.core.disassembler import disassemble
from helixlang.core.errors import HelixError, SimConfigError
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.vm import CellVM
from helixlang.sim_runtime import run
from helixlang.web.serializers import (
    _parse_lsystem_rules,
    _serialize_cell,
    _serialize_field,
    _serialize_grn,
    _serialize_morphology,
    _serialize_program_summary,
    _serialize_trace,
)

# Static assets directory: src/helixlang/web/ (sibling of server/, doc/36)
_WEB_DIR = Path(__file__).parent.parent / "web"
# examples/ directory: src/helixlang/server/app.py → server → helixlang → src → HelixLang/
_EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples"

# Debug session storage (in-process; used by the single-process dev server)
# Thread-safe: protected by _DEBUG_LOCK for concurrent request handling.
_DEBUG_LOCK: Any = None  # STATE: global (lazily initialized)
_DEBUG_SESSIONS: dict[str, Any] = {}  # STATE: global (thread-safe via _DEBUG_LOCK)

# LRU memoization of the lex→parse→semantic→compile pipeline (doc/39 O8).
#
# Repeated cohort sweeps / sensitivity runs submit near-identical sources; the
# single-process dev server re-runs the whole pipeline on every request.  We
# cache (source, table) -> pipeline result, evicting least-recently-used entries
# to keep memory bounded.  Compile for a given source+table is deterministic
# (seeded RNG, SHA256-verified goldens), so the cache never changes results —
# it only skips redundant work.  Consumers that mutate the returned ``Program``
# (``/api/run``, ``/api/sim/run`` set ``config.ticks`` and drive the VM) receive
# a :func:`copy.deepcopy`, so cached objects are never corrupted.
#
# A per-process lock protects the cache map from concurrent Flask workers.
_UNIT = 1024 * 1024  # 1 MiB
_CACHE_MEM_SOFT_LIMIT = 64    # MiB: bounded LRU; hit -> move to MRU; else evict
_PIPELINE_CACHE: OrderedDict[tuple[str, str], tuple[Any, Any, Any]] = (
    OrderedDict())
_PIPELINE_LOCK: Any = None  # STATE: lazily initialized threading.Lock


def _get_pipeline_lock() -> Any:
    global _PIPELINE_LOCK  # noqa: PLW0603
    if _PIPELINE_LOCK is None:
        import threading
        _PIPELINE_LOCK = threading.Lock()
    return _PIPELINE_LOCK


def _estimate_bytes(program: Any, chunk: Any) -> int:
    """Rough memory bound for a cached compile result (for LRU eviction)."""
    n = 0
    try:
        n += len(chunk.code)
    except TypeError:
        n += 0  # chunk may not support len (native-backed or stripped)
    try:
        n += len(chunk.constants)
    except (TypeError, AttributeError):
        n += 0  # chunk may expose constants differently depending on backend
    try:
        n += len(program.genes) * 3
    except (TypeError, AttributeError):
        n += 0  # program may not be a full Program object in all call paths
    return max(16, n * 16)


def _pipeline_cached(source: str, table_name: str) -> tuple[Any, Program, Chunk]:
    """Compile pipeline with a process-wide bounded LRU memo (doc/39 O8)."""
    key = (source, table_name)
    lock = _get_pipeline_lock()
    with lock:
        hit = _PIPELINE_CACHE.pop(key, None)
        if hit is not None:
            _PIPELINE_CACHE[key] = hit          # move to most-recently-used
            return hit
    result = _pipeline(source, table_name)
    with lock:
        _PIPELINE_CACHE[key] = result
        # bounded LRU: drop least-recently-used until under the memory budget
        used = sum(_estimate_bytes(p, c) for _, (_, p, c) in
                   _PIPELINE_CACHE.items())
        while used > _CACHE_MEM_SOFT_LIMIT * _UNIT and len(_PIPELINE_CACHE) > 1:
            _PIPELINE_CACHE.popitem(last=False)
            used = sum(_estimate_bytes(p, c) for _, (_, p, c) in
                       _PIPELINE_CACHE.items())
    return result


def _pipeline_fresh(source: str, table_name: str) -> tuple[Any, Program, Chunk]:
    """Memoized compile pipeline yielding a *fresh* mutable ``Program``.

    Compile results are immutable-derived (deterministic); callers that will
    mutate ``program`` (ticks, VM execution) get a deep copy so the cache entry
    stays pristine.  Read-only callers should use :func:`_pipeline_cached`.
    """
    import copy

    config, program, chunk = _pipeline_cached(source, table_name)
    return config, copy.deepcopy(program), chunk


def _get_debug_lock() -> Any:
    """Return the threading lock, initializing lazily."""
    global _DEBUG_LOCK  # noqa: PLW0603
    if _DEBUG_LOCK is None:
        import threading
        _DEBUG_LOCK = threading.Lock()
    return _DEBUG_LOCK


def _pipeline(source: str, table_name: str
              ) -> tuple[Any, Program, Chunk]:
    """Run the compile pipeline via one LanguageConfig (doc/38 §4).

    Returns ``(config, program, chunk)``; raises on failure."""
    config = LanguageConfig.for_table(table_name)
    tokens = list(Lexer(source).tokens())
    program = Parser(tokens, config=config).parse()
    SemanticAnalyzer(program).check()
    chunk = Compiler(config=config).compile(program)
    return config, program, chunk


def create_app() -> Flask:
    """Create the Flask app."""
    app = Flask(
        __name__,
        static_folder=str(_WEB_DIR / "static"),
        template_folder=str(_WEB_DIR / "templates"),
    )

    # ---------- Static home page ----------
    @app.route("/")
    def index():
        return send_from_directory(str(_WEB_DIR), "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(str(_WEB_DIR), filename)

    # ---------- API ----------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "version": "2026.9.0"})

    @app.get("/api/examples")
    def list_examples():
        if not _EXAMPLES_DIR.exists():
            return jsonify({"examples": []})
        examples = sorted(
            p.name for p in _EXAMPLES_DIR.glob("*.helix")
        )
        return jsonify({"examples": examples})

    @app.get("/api/examples/<name>")
    def get_example(name):
        # Prevent path traversal
        if "/" in name or ".." in name or not name.endswith(".helix"):
            return jsonify({"error": "invalid example name"}), 400
        f = _EXAMPLES_DIR / name
        if not f.exists():
            return jsonify({"error": f"example {name!r} not found"}), 404
        return jsonify({"name": name, "source": f.read_text()})

    @app.post("/api/compile")
    def api_compile():
        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "")
        table_name = body.get("table", "standard")
        if table_name not in ("standard", "mito_vertebrate", "ciliate"):
            return jsonify({"error": f"unknown table {table_name!r}"}), 400
        _config, program, chunk = _pipeline_cached(source, table_name)
        disasm = disassemble(chunk, "preview")
        return jsonify({
            "ok": True,
            "disassemble": disasm,
            "program": _serialize_program_summary(program),
            "chunk_bytes": len(chunk.code),
            "constants_count": len(chunk.constants),
        })

    @app.post("/api/run")
    def api_run():
        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "")
        table_name = body.get("table", "standard")
        ticks = body.get("ticks")
        if table_name not in ("standard", "mito_vertebrate", "ciliate"):
            return jsonify({"error": f"unknown table {table_name!r}"}), 400
        _config, program, chunk = _pipeline_fresh(source, table_name)
        if ticks is not None:
            program.config.ticks = int(ticks)
        vm = CellVM(chunk, program)
        trace = vm.run(program.config.ticks)
        return jsonify({
            "ok": True,
            "trace": trace,
            "trace_series": _serialize_trace(trace),
            "grn": _serialize_grn(vm),
            "morphology": _serialize_morphology(vm),
            "field": _serialize_field(vm),
            "cell": _serialize_cell(vm),
            "program": _serialize_program_summary(program),
            "disassemble": disassemble(chunk, "preview"),
            "ticks_run": len(trace),
        })

    @app.post("/api/sim/run")
    def api_sim_run():
        """Run any ``#config backend`` and return the SimResult payload
        (wiring.md §9).  ``body.backend`` overrides the source's choice."""
        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "")
        table_name = body.get("table", "standard")
        backend = body.get("backend")
        if table_name not in ("standard", "mito_vertebrate", "ciliate"):
            return jsonify({"error": f"unknown table {table_name!r}"}), 400
        try:
            _config, program, chunk = _pipeline_fresh(source, table_name)
        except HelixError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        ticks = body.get("ticks")
        if ticks is not None:
            program.config.ticks = int(ticks)
        try:
            result = run(program, backend=backend)
        except (SimConfigError, ValueError, KeyError, IndexError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if result is None:
            return jsonify({"ok": True, "backend": "classic", "sim": None})
        return jsonify({"ok": True, **result.to_dict()})

    # ---------- DNA physical encode/decode API ----------
    @app.post("/api/dna/encode")
    def api_dna_encode():
        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "")
        scheme = body.get("scheme", "goldman")
        pcr_cycles = int(body.get("pcr_cycles", 0))
        if scheme not in ("goldman", "erlich"):
            return jsonify({"error": f"unknown scheme {scheme!r}"}), 400
        import random

        from helixlang.plugins.runtime.dna_codec import gc_stats, helix_to_dna, pcr_amplify
        enc = helix_to_dna(source, scheme=scheme)
        # PCR error injection
        if pcr_cycles > 0:
            rng = random.Random(42)
            for o in enc["oligos"]:
                seq_field = "full" if "full" in o else "payload"
                o[seq_field] = pcr_amplify(o[seq_field],
                                           cycles=pcr_cycles, rng=rng)
        # GC statistics
        for o in enc["oligos"]:
            seq = o.get("full") or o.get("payload", "")
            o["stats"] = gc_stats(seq)
        return jsonify({"ok": True, **enc})

    @app.post("/api/dna/decode")
    def api_dna_decode():
        body = request.get_json(force=True, silent=True) or {}
        oligos_data = body.get("oligos_data", {})
        scheme = body.get("scheme", "goldman")
        if scheme not in ("goldman", "erlich"):
            return jsonify({"error": f"unknown scheme {scheme!r}"}), 400
        from helixlang.plugins.runtime.dna_codec import dna_to_helix
        result = dna_to_helix(oligos_data, scheme=scheme)
        return jsonify({"ok": True, "source": result})

    @app.get("/api/bio/codon-usage")
    def api_codon_usage():
        """Codon usage frequency table (supports multiple species: ecoli / yeast / human)."""
        from helixlang.plugins.runtime.bio_data import SPECIES_CODON_USAGE, get_codon_usage
        species = (request.args.get("species") or "ecoli").strip().lower()
        try:
            table = get_codon_usage(species)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({
            "species": species,
            "available": list(SPECIES_CODON_USAGE.keys()),
            "codon_usage": [
                {"codon": k, "aa": v[0], "per_thousand": v[1], "fraction": v[2]}
                for k, v in table.items()
            ]
        })

    @app.get("/api/bio/species")
    def api_species():
        """List supported species and their display names."""
        from helixlang.plugins.runtime.bio_data import (
            SPECIES_CODON_USAGE,
            SPECIES_DISPLAY_NAMES,
            SPECIES_TRNA_ABUNDANCE,
        )
        return jsonify({
            "species": [
                {
                    "id": sid,
                    "name": SPECIES_DISPLAY_NAMES.get(sid, sid),
                    "n_codons": len(SPECIES_CODON_USAGE.get(sid, {})),
                    "has_trna": sid in SPECIES_TRNA_ABUNDANCE,
                }
                for sid in SPECIES_CODON_USAGE
            ]
        })

    @app.get("/api/bio/trna")
    def api_trna():
        """tRNA abundance table (per species, Dong 1996 / Chan 2016 GtRNAdb)."""
        from helixlang.plugins.runtime.bio_data import SPECIES_TRNA_ABUNDANCE, get_species_trna
        species = (request.args.get("species") or "ecoli").strip().lower()
        try:
            trna = get_species_trna(species)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        max_ab = max(trna.values()) if trna else 1
        return jsonify({
            "species": species,
            "available": list(SPECIES_TRNA_ABUNDANCE.keys()),
            "trna_abundance": [
                {"codon": codon, "abundance": ab,
                 "fraction": ab / max_ab if max_ab else 0.0}
                for codon, ab in sorted(trna.items(), key=lambda x: -x[1])
            ],
        })

    @app.get("/api/bio/gray-scott-presets")
    def api_gs_presets():
        """Gray-Scott 14 experimentally measured parameter presets from Pearson 1993."""
        from helixlang.plugins.runtime.bio_data import GRAY_SCOTT_PRESETS
        return jsonify({
            "presets": [
                {"name": p.name, "F": p.F, "k": p.k,
                 "Du": p.Du, "Dv": p.Dv, "description": p.description}
                for p in GRAY_SCOTT_PRESETS
            ]
        })

    # ---------- Central dogma API ----------
    @app.post("/api/central-dogma/transcribe")
    def api_transcribe():
        """DNA → mRNA transcription."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        promoter_strength = float(body.get("promoter_strength", 1.0))
        transcription_factors = body.get("transcription_factors")
        from dataclasses import asdict

        from helixlang.plugins.runtime.central_dogma import transcribe
        transcript = transcribe(
            dna, promoter_strength=promoter_strength,
            transcription_factors=transcription_factors,
        )
        return jsonify({
            "ok": True,
            # Expose the mRNA sequence at top level (canonical field name mrna)
            "mrna": transcript.sequence,
            "transcript": asdict(transcript),
        })

    @app.post("/api/central-dogma/translate")
    def api_translate():
        """mRNA → protein translation.

        Compatible with the ``dna`` field (DNA form, contains T) and the ``mrna`` field
        (mRNA form, contains U) sent by the frontend; if the input is DNA, transcribe first, then translate.
        """
        body = request.get_json(force=True, silent=True) or {}
        # Compatible with both the mrna and dna field names
        mrna = body.get("mrna") or body.get("dna") or ""
        promoter_strength = float(body.get("promoter_strength", 1.0))
        trna_abundance = body.get("trna_abundance")
        ribosome_density = float(body.get("ribosome_density", 1.0))
        from dataclasses import asdict

        from helixlang.plugins.runtime.central_dogma import transcribe, translate
        # Normalize to DNA (T form); use transcribe to handle ORF splitting and UTR detection
        dna_form = mrna.upper().replace("U", "T")
        transcript = transcribe(dna_form, promoter_strength=promoter_strength)
        result = translate(
            transcript, trna_abundance=trna_abundance,
            ribosome_density=ribosome_density,
        )
        return jsonify({
            "ok": True,
            # Top-level canonical fields: mrna / protein
            "mrna": transcript.sequence,
            "protein": result.protein,
            "result": asdict(result),
            "transcript": asdict(transcript),
        })

    @app.post("/api/central-dogma/coupled")
    def api_coupled():
        """Coupled transcription and translation."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        promoter_strength = float(body.get("promoter_strength", 1.0))
        from dataclasses import asdict

        from helixlang.plugins.runtime.central_dogma import coupled_transcription_translation
        result = coupled_transcription_translation(
            dna, promoter_strength=promoter_strength)
        # Canonical fields: time / mrna_level / protein_level
        time_series = {
            "time": [tc.time_min for tc in result["time_course"]],
            "mrna_level": [tc.mrna_level for tc in result["time_course"]],
            "protein_level": [tc.protein_accumulated
                              for tc in result["time_course"]],
        }
        return jsonify({
            "ok": True,
            "mrna": result["transcript"].sequence,
            "protein": result["protein"],
            "transcript": asdict(result["transcript"]),
            "mrna_steady_state": result["mrna_steady_state"],
            "time_course": [asdict(tc) for tc in result["time_course"]],
            "time_series": time_series,
            "transcription_time_s": result["transcription_time_s"],
            "translation_time_s": result["translation_time_s"],
            "coupling_offset_s": result["coupling_offset_s"],
            "translation_result": asdict(result["translation_result"]),
        })

    # ---------- Evolution API ----------
    @app.post("/api/evolution/run")
    def api_evolution_run():
        """Run an evolution simulation."""
        body = request.get_json(force=True, silent=True) or {}
        initial_dna = body.get("initial_dna", "ACGT" * 10)
        generations = int(body.get("generations", 50))
        population_size = int(body.get("population_size", 100))
        mutation_rate = float(body.get("mutation_rate", 0.01))
        fitness_method = body.get("fitness_method", "hamming")
        target_dna = body.get("target_dna")
        from helixlang.plugins.runtime.evolution import EvolutionaryPopulation, EvolutionConfig
        cfg = EvolutionConfig(
            mutation_rate=mutation_rate,
            population_size=population_size,
            generations=generations,
        )
        pop = EvolutionaryPopulation(initial_dna, config=cfg, target_dna=target_dna,
                         fitness_method=fitness_method)
        pop.evolve()
        stats = pop.get_generation_stats()
        best = pop.best_individual()
        return jsonify({
            "ok": True,
            "generations": stats,
            "final_dna": best.dna if best else initial_dna,
            "final_fitness": best.fitness if best else 0.0,
            "stats": {
                "final_generation": pop.generation,
                "final_diversity": pop.get_diversity(),
                "population_size": len(pop.individuals),
            },
        })

    @app.get("/api/evolution/params")
    def api_evolution_params():
        """Return evolution parameter defaults and descriptions."""
        import dataclasses

        from helixlang.plugins.runtime.evolution import EvolutionConfig
        cfg = EvolutionConfig()
        defaults = {
            f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
        }
        return jsonify({
            "ok": True,
            "defaults": defaults,
            "fitness_methods": ["hamming", "cai", "gc", "custom"],
            "descriptions": {
                "mutation_rate": "per-nt substitution rate per generation (E. coli default 2.2e-10)",
                "indel_rate": "per-nt indel rate",
                "population_size": "Wright-Fisher population size N",
                "generations": "number of generations",
                "selection_coefficient": "selection coefficient s (>0 positive, =0 neutral, <0 negative)",
                "recombination_rate": "recombination rate (0=asexual, 1=recombination every generation)",
                "transition_transversion_ratio": "transition:transversion ratio",
            },
        })

    # ---------- Multicellular population API ----------
    @app.post("/api/population/simulate")
    def api_population_simulate():
        """Multicellular population simulation."""
        body = request.get_json(force=True, silent=True) or {}
        initial_count = int(body.get("initial_count", 1))
        generations = int(body.get("generations", 10))
        config_body = body.get("config", {}) or {}
        from helixlang.plugins.runtime.population import (
            CellPopulation,
            PopulationCell,
            PopulationConfig,
        )
        cfg = PopulationConfig(
            grid_width=int(config_body.get("grid_width", 100)),
            grid_height=int(config_body.get("grid_height", 100)),
            max_size=int(config_body.get("max_size", 10000)),
            division_threshold=float(
                config_body.get("division_threshold", 200.0)),
            death_threshold=float(
                config_body.get("death_threshold", 0.0)),
            signaling_enabled=bool(
                config_body.get("signaling_enabled", True)),
            signal_diffusion=float(
                config_body.get("signal_diffusion", 0.1)),
            signal_threshold=float(
                config_body.get("signal_threshold", 5.0)),
            metabolic_cost=float(
                config_body.get("metabolic_cost", 1.0)),
            energy_intake=float(
                config_body.get("energy_intake", 5.0)),
        )
        cx = cfg.grid_width // 2
        cy = cfg.grid_height // 2
        initial_cells = [
            PopulationCell(id=i, energy=100.0, x=cx, y=cy)
            for i in range(max(0, initial_count))
        ]
        pop = CellPopulation(initial_cells, config=cfg)
        history = pop.evolve(generations)
        return jsonify({
            "ok": True,
            "history": history,
            "final_grid": pop.get_grid(),
            "signal_field": pop.get_signal_field(),
        })

    # ---------- CRISPR API ----------
    @app.post("/api/crispr/find-pam")
    def api_crispr_find_pam():
        """Search for PAM sites."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        cas_variant = body.get("cas_variant", "SpCas9")
        both_strands = bool(body.get("both_strands", True))
        from helixlang.plugins.runtime.crispr import find_pam_sites
        sites = find_pam_sites(dna, cas_variant=cas_variant,
                               both_strands=both_strands)
        return jsonify({"ok": True, "sites": sites})

    @app.post("/api/crispr/design-guide")
    def api_crispr_design_guide():
        """Design an sgRNA."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        cas_variant = body.get("cas_variant", "SpCas9")
        position = int(body.get("position", 0))
        from dataclasses import asdict

        from helixlang.plugins.runtime.crispr import design_guide, on_target_score
        guide = design_guide(dna, cas_variant=cas_variant,
                             position=position)
        score = on_target_score(guide)
        return jsonify({
            "ok": True,
            "guide": asdict(guide),
            "on_target_score": score,
        })

    @app.post("/api/crispr/edit")
    def api_crispr_edit():
        """Gene editing."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        target_position = int(body.get("target_position", 0))
        new_sequence = body.get("new_sequence", "")
        cas_variant = body.get("cas_variant", "SpCas9")
        from dataclasses import asdict

        from helixlang.plugins.runtime.crispr import edit_gene
        result = edit_gene(dna, target_position=target_position,
                           new_sequence=new_sequence,
                           cas_variant=cas_variant)
        return jsonify({
            "ok": True,
            "edited_dna": result.edited_dna,
            "edit_type": result.edit_type,
            "success": result.success,
            "off_targets": [asdict(ot) for ot in result.off_targets],
            "guide": asdict(result.guide),
            "edit_position": result.edit_position,
            "edit_length": result.edit_length,
            "repair": result.repair,
        })

    # ---------- Epigenetics API ----------
    @app.post("/api/epigenetics/methylate")
    def api_epigenetics_methylate():
        """DNA methylation."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        cell_type = body.get("cell_type", "ecoli")
        methylase = body.get("methylase", "dam")
        from dataclasses import asdict

        from helixlang.plugins.runtime.epigenetics import methylate_dna
        state = methylate_dna(dna, cell_type=cell_type,
                              methylase=methylase)
        return jsonify({"ok": True, "methylation": asdict(state)})

    @app.post("/api/epigenetics/histone")
    def api_epigenetics_histone():
        """Histone modification."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        gene_positions = body.get("gene_positions", [])
        cell_type = body.get("cell_type", "eukaryote")
        from dataclasses import asdict

        from helixlang.plugins.runtime.epigenetics import (
            ChromatinState,
            add_histone_marks,
            calculate_accessibility,
            calculate_expression_modifier,
            methylate_dna,
        )
        # gene_positions supports both list[dict] and dict[name -> dict]
        if isinstance(gene_positions, dict):
            gene_list = []
            for name, info in gene_positions.items():
                g = dict(info) if isinstance(info, dict) else {}
                g.setdefault("name", name)
                gene_list.append(g)
        else:
            gene_list = list(gene_positions)
        marks = add_histone_marks(dna, gene_positions=gene_list,
                                  cell_type=cell_type)
        # Build the chromatin state to compute accessibility and expression modification
        if cell_type == "ecoli":
            meth = methylate_dna(dna, cell_type="ecoli",
                                 methylase="dam")
        else:
            meth = methylate_dna(dna, cell_type=cell_type,
                                 methylase="cpg")
        chromatin = ChromatinState(methylation=meth,
                                   histone_marks=marks)
        accessibility = calculate_accessibility(chromatin)
        expression_modifier = calculate_expression_modifier(
            chromatin, gene_list)
        return jsonify({
            "ok": True,
            "histone_marks": [asdict(m) for m in marks],
            "accessibility": accessibility,
            "expression_modifier": expression_modifier,
        })

    @app.post("/api/epigenetics/cpg-islands")
    def api_cpg_islands():
        """CpG island detection."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        from helixlang.plugins.runtime.epigenetics import find_cpg_islands
        islands = find_cpg_islands(dna)
        return jsonify({"ok": True, "islands": islands})

    # ---------- 3D morphogenesis API ----------
    @app.post("/api/morphology3d/generate")
    def api_morphology_3d():
        """3D L-system morphology generation."""
        body = request.get_json(force=True, silent=True) or {}
        preset = body.get("preset")
        axiom = body.get("axiom", "F")
        rules = body.get("rules", {}) or {}
        # Frontend text input may pass "F:F[+F]F[-F]F;X:FX" format; parse it uniformly into a dict
        if isinstance(rules, str):
            rules = _parse_lsystem_rules(rules)
        angle = float(body.get("angle", 22.5))
        iterations = int(body.get("iterations", 3))
        from helixlang.plugins.runtime.morphology_3d import PLANT_PRESETS, LSystem3D
        if preset and preset in PLANT_PRESETS:
            p = PLANT_PRESETS[preset]
            axiom = p["axiom"]
            rules = p["rules"]
            angle = p["angle"]
        lsys = LSystem3D(axiom=axiom, rules=rules, angle=angle)
        lines = lsys.draw(iterations)
        points = lsys.get_points(iterations)
        bounds = lsys.get_bounds(iterations)
        return jsonify({
            "ok": True,
            "lines": [
                {
                    "start": [ln.start.x, ln.start.y, ln.start.z],
                    "end": [ln.end.x, ln.end.y, ln.end.z],
                    "width": ln.width,
                }
                for ln in lines
            ],
            "points": [[pt.x, pt.y, pt.z] for pt in points],
            "bounds": {
                "min": [bounds["min"].x, bounds["min"].y,
                        bounds["min"].z],
                "max": [bounds["max"].x, bounds["max"].y,
                        bounds["max"].z],
                "center": [bounds["center"].x, bounds["center"].y,
                           bounds["center"].z],
                "size": [bounds["size"].x, bounds["size"].y,
                         bounds["size"].z],
            },
        })

    # ---------- DNA storage API ----------
    @app.post("/api/dna-storage/store")
    def api_dna_storage_store():
        """DNA storage encoding.

        Input fields: ``text`` (data, base64-decoded first, falls back to UTF-8 text);
        ``redundancy`` accepts int or float; ``scheme`` is case-insensitive
        (frontend dropdown uses ``Goldman``/``Erlich``).
        """
        body = request.get_json(force=True, silent=True) or {}
        # Canonical input field name text
        data_str = body.get("text", "")
        scheme = (body.get("scheme", "erlich") or "erlich").lower()
        redundancy = float(body.get("redundancy", 0.15))
        import base64

        from helixlang.plugins.apps.dna_storage import DNAStorage
        from helixlang.plugins.runtime.dna_codec import gc_stats
        # Prefer base64 decoding; on failure treat as UTF-8 text (narrowed to specific exceptions)
        try:
            data = base64.b64decode(data_str, validate=True)
        except (ValueError, binascii.Error):
            data = data_str.encode("utf-8")
        storage = DNAStorage(scheme=scheme)
        report = storage.store(data, redundancy=redundancy)
        # Serialize the oligo list (picking fields by scheme); canonical fields: sequence +
        # gc_content + max_homopolymer (read directly by frontend table columns)
        oligos_ser = []
        for o in report.oligos:
            seq = o.full if scheme == "goldman" else o.payload
            stats = gc_stats(seq)
            entry = {
                "index": o.index, "payload": o.payload,
                "sequence": seq,
                "gc_content": stats["gc_content"],
                "max_homopolymer": stats["max_homopolymer"],
            }
            if scheme == "goldman":
                entry["overhang"] = o.overhang
                entry["full"] = o.full
            else:
                entry["seed"] = o.seed
            oligos_ser.append(entry)
        report_dict = {
            "scheme": report.scheme,
            "oligos": oligos_ser,
            "total_bp": report.total_bp,
            "density_bit_per_nt": report.density_bit_per_nt,
            "num_oligos": report.num_oligos,
            "avg_oligo_length": report.avg_oligo_length,
            "encoding_time": report.encoding_time,
            "data_len": report.data_len,
            "K": report.K,
        }
        return jsonify({
            "ok": True,
            # Canonical field name oligos
            "oligos": oligos_ser,
            "report": report_dict,
        })

    @app.post("/api/dna-storage/lifecycle")
    def api_dna_storage_lifecycle():
        """DNA storage lifecycle simulation."""
        body = request.get_json(force=True, silent=True) or {}
        data_str = body.get("text", "")
        scheme = (body.get("scheme", "erlich") or "erlich").lower()
        synthesis_quality = body.get("synthesis_quality", "typical")
        pcr_cycles = int(body.get("pcr_cycles", 10))
        polymerase = body.get("polymerase", "taq")
        sequencing_platform = body.get(
            "sequencing_platform", "illumina_hiseq_novaseq")
        storage_years = float(body.get("storage_years", 0))
        import base64

        from helixlang.plugins.apps.dna_storage import DNAStorage
        try:
            data = base64.b64decode(data_str, validate=True)
        except (ValueError, binascii.Error):
            data = data_str.encode("utf-8")
        storage = DNAStorage(scheme=scheme)
        report = storage.simulate_lifecycle(
            data, synthesis_quality=synthesis_quality,
            pcr_cycles=pcr_cycles, polymerase=polymerase,
            sequencing_platform=sequencing_platform,
            storage_years=storage_years,
        )
        recovered = report.recovered_data.decode("utf-8", errors="replace")
        report_dict = {
            "recovered_data": recovered,
            "integrity": report.integrity,
            "error_rate": report.error_rate,
            "synthesis_errors": report.synthesis_errors,
            "pcr_errors": report.pcr_errors,
            "sequencing_errors": report.sequencing_errors,
            "decay_damage": report.decay_damage,
            "success": report.success,
            "original_data_len": len(report.original_data),
        }
        return jsonify({
            "ok": True,
            "report": report_dict,
            # Canonical field names integrity / error_rate (array form; the last element is the final value)
            "integrity": [report.integrity],
            "error_rate": [report.error_rate],
            "recovered_data": recovered,
        })

    @app.post("/api/dna-storage/analyze")
    def api_dna_storage_analyze():
        """DNA storage analysis report.

        Fields aligned with the frontend: top level exposes ``density`` / ``bits_per_nt`` /
        ``total_oligos`` / ``total_bytes`` / ``cost`` / ``durability``.
        """
        body = request.get_json(force=True, silent=True) or {}
        oligos_data = body.get("oligos", []) or []
        data_len = int(body.get("data_len", 0))
        scheme = (body.get("scheme", "erlich") or "erlich").lower()
        from helixlang.plugins.apps.dna_storage import DNAStorage
        from helixlang.plugins.runtime.dna_codec import ErlichOligo, GoldmanOligo
        # Rebuild oligo objects from JSON
        oligos: list[GoldmanOligo | ErlichOligo] = []
        for o in oligos_data:
            if scheme == "goldman":
                oligos.append(GoldmanOligo(
                    index=int(o.get("index", 0)),
                    payload=o.get("payload", ""),
                    overhang=o.get("overhang", ""),
                    full=o.get("full", ""),
                ))
            else:
                oligos.append(ErlichOligo(
                    index=int(o.get("index", 0)),
                    seed=int(o.get("seed", 0)),
                    payload=o.get("payload", ""),
                    rs_oligo=b"",
                ))
        storage = DNAStorage(scheme=scheme)
        report = storage.analyze(oligos, data_len=data_len)
        report_dict = {
            "density": report.density_bit_per_nt,
            "shannon_efficiency": report.shannon_efficiency,
            "cost": report.estimated_cost_usd,
            "durability": report.durability_years,
            "oligo_count": report.oligo_count,
            "total_bp": report.total_bp,
            "gc_content": report.gc_content,
            "max_homopolymer": report.max_homopolymer,
            "comparison": report.comparison,
        }
        return jsonify({
            "ok": True,
            "report": report_dict,
            # Canonical top-level fields (aligned with the frontend renderAnalyze contract)
            "density": {"bits_per_nt": report.density_bit_per_nt},
            "total_oligos": report.oligo_count,
            "total_bytes": data_len,
            "cost": {"per_mb": report.estimated_cost_usd},
            "durability": {"years": report.durability_years},
        })

    @app.post("/api/dna-storage/retrieve")
    def api_dna_storage_retrieve():
        """DNA storage decoding: restore the original text from an oligo list.

        Request body: ``{oligos: [...], scheme: "erlich"|"goldman",
        data_len: int}``. The ``oligos`` structure matches the oligos returned by ``/store``.
        """
        body = request.get_json(force=True, silent=True) or {}
        oligos_data = body.get("oligos", []) or []
        scheme = (body.get("scheme", "erlich") or "erlich").lower()
        total_len = int(body.get("data_len", body.get("total_len", 0)))
        from helixlang.plugins.apps.dna_storage import DNAStorage
        from helixlang.plugins.runtime.dna_codec import ErlichOligo, GoldmanOligo
        oligos: list[GoldmanOligo | ErlichOligo] = []
        for o in oligos_data:
            if scheme == "goldman":
                oligos.append(GoldmanOligo(
                    index=int(o.get("index", 0)),
                    payload=o.get("payload", ""),
                    overhang=o.get("overhang", ""),
                    full=o.get("full", ""),
                ))
            else:
                oligos.append(ErlichOligo(
                    index=int(o.get("index", 0)),
                    seed=int(o.get("seed", 0)),
                    payload=o.get("payload", ""),
                    rs_oligo=b"",
                ))
        storage = DNAStorage(scheme=scheme)
        decoded = storage.retrieve(oligos, total_len=total_len)
        return jsonify({
            "ok": True,
            "text": decoded.decode("utf-8", errors="replace"),
            "data_len": len(decoded),
        })

    # ---------- Synthetic biology design API ----------
    @app.post("/api/synbio/design-cassette")
    def api_synbio_cassette():
        """Design an expression cassette."""
        body = request.get_json(force=True, silent=True) or {}
        protein = body.get("protein", "")
        promoter = body.get("promoter", "lac")
        terminator = body.get("terminator", "rrnB_T1")
        optimize_codons = bool(body.get("optimize_codons", True))
        add_histidine_tag = bool(body.get("add_histidine_tag", False))
        from helixlang.plugins.apps.synbio_designer import (
            CassetteConfig,
            SynBioDesigner,
        )
        designer = SynBioDesigner()
        config = CassetteConfig(
            promoter=promoter, terminator=terminator,
            optimize_codons=optimize_codons,
            add_histidine_tag=add_histidine_tag,
        )
        cassette = designer.design_cassette(protein, config=config)
        return jsonify({
            "ok": True,
            "cassette": {
                "full_sequence": cassette.full_sequence,
                "promoter_seq": cassette.promoter_seq,
                "rbs_seq": cassette.rbs_seq,
                "orf_seq": cassette.orf_seq,
                "terminator_seq": cassette.terminator_seq,
                "protein": cassette.protein,
                "cai": cassette.cai,
                "gc_content": cassette.gc_content,
                "restriction_sites_found":
                    cassette.restriction_sites_found,
                "validation_report": cassette.validation_report,
            },
        })

    @app.post("/api/synbio/design-vector")
    def api_synbio_vector():
        """Design a complete vector."""
        body = request.get_json(force=True, silent=True) or {}
        protein = body.get("protein", "")
        origin = body.get("origin", "pUC19")
        selection_marker = body.get("selection_marker", "AmpR")
        from helixlang.plugins.apps.synbio_designer import (
            CassetteConfig,
            SynBioDesigner,
            VectorConfig,
        )
        designer = SynBioDesigner()
        vcfg = VectorConfig(
            cassette=CassetteConfig(),
            origin_of_replication=origin,
            selection_marker=selection_marker,
        )
        vector = designer.design_vector(protein, vector_config=vcfg)
        return jsonify({
            "ok": True,
            "vector": {
                "full_sequence": vector.full_sequence,
                "total_length": vector.total_length,
                "features": vector.features,
                "origin_seq": vector.origin_seq,
                "marker_seq": vector.marker_seq,
                "mcs_seq": vector.mcs_seq,
                "cassette": {
                    "full_sequence": vector.cassette.full_sequence,
                    "cai": vector.cassette.cai,
                    "gc_content": vector.cassette.gc_content,
                },
            },
        })

    @app.post("/api/synbio/validate")
    def api_synbio_validate():
        """Validate a DNA sequence."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        from helixlang.plugins.apps.synbio_designer import SynBioDesigner
        designer = SynBioDesigner()
        validation = designer.validate(dna)
        return jsonify({
            "ok": True,
            "validation": {
                **validation,
                "has_start": bool(validation.get("orf_found", False)),
                "has_stop": bool(validation.get("stop_codon")),
                "cai": validation.get("orf_cai", 0.0),
            },
        })

    @app.post("/api/synbio/export-genbank")
    def api_synbio_genbank():
        """Export GenBank format."""
        body = request.get_json(force=True, silent=True) or {}
        dna = body.get("dna", "")
        name = body.get("name", "sequence")
        from helixlang.plugins.apps.synbio_designer import SynBioDesigner
        designer = SynBioDesigner()
        genbank_text = designer.export_genbank(dna, name)
        return jsonify({"ok": True, "genbank": genbank_text})

    @app.post("/api/synbio/export-fasta")
    def api_synbio_export_fasta():
        """Export FASTA format.

        Reuses the design parameters of ``/api/synbio/design-cassette``; if ``dna`` is provided directly
        it is used as the sequence; otherwise a cassette is designed from the parameters first and its full sequence is exported.
        ``name`` is the FASTA header (default ``helixlang_cassette``).
        """
        body = request.get_json(force=True, silent=True) or {}
        name = body.get("name", "helixlang_cassette")
        from helixlang.plugins.apps.synbio_designer import (
            CassetteConfig,
            SynBioDesigner,
        )
        designer = SynBioDesigner()
        dna = body.get("dna")
        if not dna:
            # Reuse the design-cassette fields; design once, then export
            protein = body.get("protein", "")
            promoter = body.get("promoter", "lac")
            terminator = body.get("terminator", "rrnB_T1")
            optimize_codons = bool(body.get("optimize_codons", True))
            add_histidine_tag = bool(body.get("add_histidine_tag", False))
            config = CassetteConfig(
                promoter=promoter, terminator=terminator,
                optimize_codons=optimize_codons,
                add_histidine_tag=add_histidine_tag,
            )
            cassette = designer.design_cassette(protein, config=config)
            dna = cassette.full_sequence
        fasta_text = designer.export_fasta(dna, name)
        return jsonify({"ok": True, "fasta": fasta_text, "name": name})

    # ---------- Debugger API ----------
    @app.post("/api/debug/session")
    def api_debug_session():
        """Create a debug session."""
        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "")
        table_name = body.get("table", "standard")
        if table_name not in ("standard", "mito_vertebrate", "ciliate"):
            return jsonify({
                "ok": False,
                "error": f"unknown table {table_name!r}",
            }), 400
        import uuid

        from helixlang.debugger import HelixDebugger
        _config, program, chunk = _pipeline_fresh(source, table_name)
        vm = CellVM(chunk, program)
        debugger = HelixDebugger(vm, program)
        debugger.start()
        state = debugger.get_state()
        session_id = str(uuid.uuid4())
        _DEBUG_SESSIONS[session_id] = debugger
        return jsonify({
            "ok": True,
            "session_id": session_id,
            "initial_state": {
                "ip": state.ip,
                "op": state.op,
                "stack": list(state.stack),
                "cell_state": state.cell_state,
                "grn_state": state.grn_state,
                "gene": state.gene,
                "line": state.line,
                "codon_index": state.codon_index,
            },
        })

    # ---------- Metabolism FBA API ----------
    @app.get("/api/metabolism/model")
    def api_metabolism_model():
        """E. coli core metabolism model structure (reactions/metabolites/subsystems)."""
        from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL
        m = ECOLI_CORE_MODEL
        reactions = []
        subsystems: dict[str, int] = {}
        for rid, rxn in m.reactions.items():
            reactions.append({
                "id": rid, "name": rxn.name,
                "stoichiometry": rxn.stoichiometry,
                "lower_bound": rxn.lower_bound,
                "upper_bound": rxn.upper_bound,
                "subsystem": rxn.subsystem,
            })
            subsystems[rxn.subsystem] = subsystems.get(rxn.subsystem, 0) + 1
        return jsonify({
            "n_reactions": len(m.reactions),
            "n_metabolites": len(m.metabolites),
            "biomass_reaction": m.biomass_reaction,
            "metabolites": sorted(m.metabolites),
            "subsystems": subsystems,
            "reactions": reactions,
        })

    def _make_fba(body):
        from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis
        objective = body.get("objective", "biomass")
        glc_uptake = float(body.get("glc_uptake", 10.0))
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        if glc_uptake > 0:
            fba.set_uptake("GLC", glc_uptake)
        fba.solve(objective=objective)
        return fba

    @app.post("/api/metabolism/fba")
    def api_metabolism_fba():
        """Solve the optimal flux distribution with FBA."""
        body = request.get_json(force=True, silent=True) or {}
        fba = _make_fba(body)
        fluxes = fba.last_solution or {}
        return jsonify({
            "ok": True,
            "objective": body.get("objective", "biomass"),
            "objective_value": fluxes.get(fba.model.biomass_reaction, 0.0),
            "fluxes": fluxes,
        })

    @app.post("/api/metabolism/analyze")
    def api_metabolism_analyze():
        """Metabolism state analysis (biomass yield/energy balance/byproducts/subsystem fluxes)."""
        body = request.get_json(force=True, silent=True) or {}
        fba = _make_fba(body)
        report = fba.analyze()
        return jsonify({"ok": True, "analysis": report})

    # ---------- Protein structure API ----------
    @app.post("/api/protein/secondary")
    def api_protein_secondary():
        """Secondary structure prediction (Chou-Fasman / GOR IV)."""
        body = request.get_json(force=True, silent=True) or {}
        sequence = body.get("sequence", "")
        method = body.get("method", "chou-fasman")
        from helixlang.plugins.runtime.protein_structure import (
            predict_secondary,
            predict_secondary_gor,
        )
        if method == "gor":
            ss, segments = predict_secondary_gor(sequence)
        else:
            ss, segments = predict_secondary(sequence)
        return jsonify({
            "ok": True,
            "method": method,
            "sequence": sequence,
            "secondary_structure": ss,
            "segments": [
                {"start": s.start, "end": s.end, "ss_type": s.ss_type,
                 "score": s.score, "sequence": s.sequence}
                for s in segments
            ],
        })

    @app.post("/api/protein/transmembrane")
    def api_protein_transmembrane():
        """Transmembrane helix prediction (simplified TMHMM)."""
        body = request.get_json(force=True, silent=True) or {}
        sequence = body.get("sequence", "")
        from helixlang.plugins.runtime.protein_structure import (
            gravy,
            hydropathy_profile,
            predict_transmembrane,
        )
        tms = predict_transmembrane(sequence)
        return jsonify({
            "ok": True,
            "sequence": sequence,
            "n_helices": len(tms),
            "is_membrane_protein": len(tms) > 0,
            "gravy": gravy(sequence),
            "hydropathy_profile": hydropathy_profile(sequence),
            "helices": [
                {"start": t.start, "end": t.end, "length": t.length,
                 "mean_hydropathy": t.mean_hydropathy, "sequence": t.sequence}
                for t in tms
            ],
        })

    @app.post("/api/protein/disorder")
    def api_protein_disorder():
        """Intrinsically disordered region prediction (simplified Dunker 2001)."""
        body = request.get_json(force=True, silent=True) or {}
        sequence = body.get("sequence", "")
        from helixlang.plugins.runtime.protein_structure import predict_disorder
        regions = predict_disorder(sequence)
        dis_len = sum(r.length for r in regions)
        total = max(len(sequence), 1)
        return jsonify({
            "ok": True,
            "sequence": sequence,
            "n_regions": len(regions),
            "disorder_fraction": dis_len / total,
            "regions": [
                {"start": r.start, "end": r.end, "length": r.length,
                 "mean_hydropathy": r.mean_hydropathy, "sequence": r.sequence}
                for r in regions
            ],
        })

    # ---------- Debugger API (step / continue / breakpoint / state) ----------
    def _get_debugger():
        sid = (request.args.get("session_id")
               or (request.get_json(silent=True) or {}).get("session_id"))
        debugger = _DEBUG_SESSIONS.get(sid) if sid else None
        if debugger is None:
            return None, (jsonify({"ok": False, "error": "unknown session_id"}),
                          400)
        return debugger, None

    def _serialize_state(state):
        return {
            "ip": state.ip, "op": state.op, "stack": list(state.stack),
            "cell_state": state.cell_state, "grn_state": state.grn_state,
            "gene": state.gene, "line": state.line,
            "codon_index": state.codon_index,
        }

    @app.get("/api/debug/state")
    def api_debug_state():
        """Get the current debug state snapshot."""
        debugger, err = _get_debugger()
        if err is not None:
            return err
        state = debugger.get_state()
        return jsonify({
            "ok": True,
            "state": _serialize_state(state),
            "call_stack": debugger.get_call_stack(),
            "breakpoints": [
                {"offset": b.offset, "line": b.line,
                 "codon_index": b.codon_index, "condition": b.condition,
                 "enabled": b.enabled, "hit_count": b.hit_count}
                for b in debugger.list_breakpoints()
            ],
            "halted": not debugger.vm.frames,
        })

    @app.post("/api/debug/step")
    def api_debug_step():
        """Execute a single instruction (step into)."""
        _body = request.get_json(force=True, silent=True) or {}
        debugger, err = _get_debugger()
        if err is not None:
            return err
        if not debugger.vm.frames:
            return jsonify({"ok": True, "state": None, "halted": True})
        state = debugger.step()
        return jsonify({
            "ok": True,
            "state": _serialize_state(state),
            "halted": not debugger.vm.frames,
        })

    @app.post("/api/debug/step-over")
    def api_debug_step_over():
        """Step over: execute one instruction; if CALL_GENE is entered, keep executing until returning to the original frame depth."""
        _body = request.get_json(force=True, silent=True) or {}
        debugger, err = _get_debugger()
        if err is not None:
            return err
        if not debugger.vm.frames:
            return jsonify({"ok": True, "state": None, "halted": True})
        state = debugger.step_over()
        return jsonify({
            "ok": True,
            "state": _serialize_state(state),
            "halted": not debugger.vm.frames,
        })

    @app.post("/api/debug/step-out")
    def api_debug_step_out():
        """Step out: keep executing until returning from the current frame (frame depth decreases or frames are empty)."""
        _body = request.get_json(force=True, silent=True) or {}
        debugger, err = _get_debugger()
        if err is not None:
            return err
        if not debugger.vm.frames:
            return jsonify({"ok": True, "state": None, "halted": True})
        state = debugger.step_out()
        return jsonify({
            "ok": True,
            "state": _serialize_state(state),
            "halted": not debugger.vm.frames,
        })

    @app.post("/api/debug/continue")
    def api_debug_continue():
        """Continue executing until hitting a breakpoint or HALT."""
        debugger, err = _get_debugger()
        if err is not None:
            return err
        state = debugger.continue_run()
        return jsonify({
            "ok": True,
            "state": _serialize_state(state) if state else None,
            "halted": state is None,
        })

    @app.post("/api/debug/breakpoint")
    def api_debug_breakpoint():
        """Set/remove breakpoints. action: set | remove | clear | list."""
        body = request.get_json(force=True, silent=True) or {}
        debugger, err = _get_debugger()
        if err is not None:
            return err
        action = body.get("action", "set")
        if action == "clear":
            for bp in list(debugger.list_breakpoints()):
                debugger.remove_breakpoint(bp)
            return jsonify({"ok": True, "breakpoints": []})
        if action == "list":
            return jsonify({"ok": True, "breakpoints": [
                {"offset": b.offset, "line": b.line,
                 "codon_index": b.codon_index, "condition": b.condition,
                 "enabled": b.enabled, "hit_count": b.hit_count}
                for b in debugger.list_breakpoints()]})
        if action == "remove":
            offset = body.get("offset")
            for bp in debugger.list_breakpoints():
                if bp.offset == offset:
                    debugger.remove_breakpoint(bp)
            return jsonify({"ok": True, "breakpoints": [
                {"offset": b.offset, "line": b.line,
                 "codon_index": b.codon_index, "condition": b.condition,
                 "enabled": b.enabled, "hit_count": b.hit_count}
                for b in debugger.list_breakpoints()]})
        # default: set
        bp = debugger.set_breakpoint(
            offset=body.get("offset"),
            line=body.get("line"),
            codon_index=body.get("codon_index"),
            condition=body.get("condition"))
        return jsonify({
            "ok": True,
            "breakpoint": {"offset": bp.offset, "line": bp.line,
                           "codon_index": bp.codon_index,
                           "condition": bp.condition, "enabled": bp.enabled},
            "breakpoints": [
                {"offset": b.offset, "line": b.line,
                 "codon_index": b.codon_index, "condition": b.condition,
                 "enabled": b.enabled, "hit_count": b.hit_count}
                for b in debugger.list_breakpoints()],
        })

    # ------------------------------------------------------------------
    # GEM reconstruction API (doc/20 §9.7)
    # ------------------------------------------------------------------
    @app.post("/api/gem/reconstruct")
    def api_gem_reconstruct():
        """Run GEM reconstruction pipeline from genome FASTA.

        Body: { "fasta": "<path or inline FASTA>", "organism": "e_coli_k12",
                "use_database": true, "gapfill": true }
        Returns: { "ok": true, "stages_completed": 6, "reactions": N, ... }
        """
        from helixlang.plugins.apps.gem_pipeline import run_gem_pipeline

        body = request.get_json(force=True, silent=True) or {}
        fasta = body.get("fasta", "")
        organism = body.get("organism", "e_coli_k12")
        if not fasta:
            return jsonify({"ok": False, "error": "fasta field required"}), 400

        try:
            result = run_gem_pipeline(
                genome_fasta=fasta,
                organism=organism,
                use_database_interactions=body.get("use_database", True),
                include_spontaneous=body.get("include_spontaneous", True),
                run_gapfill=body.get("gapfill", True),
                target_organism=body.get("target_organism", "Escherichia coli"),
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify({
            "ok": True,
            "stages_completed": result.stages_completed,
            "annotated_genes": result.annotated_genes,
            "reactions_total": result.final_reaction_count,
            "grn_edges": result.grn.total_edges if result.grn else 0,
            "kcat_predictions": len(result.kcat_predictions),
            "km_estimates": len(result.km_estimates),
            "warnings": result.warnings,
            "errors": result.errors,
            "summary": result.summary(),
        })

    @app.post("/api/gem/simulate")
    def api_gem_simulate():
        """Run a GEM-reconstructed model simulation.

        Body: { "fasta": "<path>", "organism": "e_coli_k12",
                "ticks": 100, "output": "csv" }
        Returns: SimResult payload (columns + rows + meta).
        """
        from helixlang.plugins.apps.gem_pipeline import run_gem_pipeline

        body = request.get_json(force=True, silent=True) or {}
        fasta = body.get("fasta", "")
        organism = body.get("organism", "e_coli_k12")
        if not fasta:
            return jsonify({"ok": False, "error": "fasta field required"}), 400

        try:
            result = run_gem_pipeline(
                genome_fasta=fasta,
                organism=organism,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        # Build SimResult-like output
        columns = [
            "stage", "status", "genes_annotated", "reactions_total",
            "grn_edges", "kcat_predictions", "km_estimates",
        ]
        rows = [
            {"stage": "annotation", "status": "ok",
             "genes_annotated": result.annotated_genes,
             "reactions_total": 0, "grn_edges": 0,
             "kcat_predictions": 0, "km_estimates": 0},
            {"stage": "reconstruction", "status": "ok",
             "genes_annotated": result.annotated_genes,
             "reactions_total": result.final_reaction_count,
             "grn_edges": 0, "kcat_predictions": 0, "km_estimates": 0},
            {"stage": "kinetics", "status": "ok",
             "genes_annotated": result.annotated_genes,
             "reactions_total": result.final_reaction_count,
             "grn_edges": result.grn.total_edges if result.grn else 0,
             "kcat_predictions": len(result.kcat_predictions),
             "km_estimates": len(result.km_estimates)},
        ]
        return jsonify({
            "ok": True, "backend": "gem",
            "columns": columns, "rows": rows,
            "meta": {"summary": result.summary()},
        })

    @app.post("/api/full-pipeline")
    def api_full_pipeline():
        """Run full-chain custom organism pipeline (doc/26).

        Body: { "fasta": "<path>", "organism": "custom_organism",
                "ecgem": true, "community": false }
        Returns: pipeline result with structures, kinetics, ecGEM, simulation.
        """
        from helixlang.plugins.apps.full_pipeline import PipelineConfig, run_full_pipeline

        body = request.get_json(force=True, silent=True) or {}
        fasta = body.get("fasta", "")
        if not fasta:
            return jsonify({"ok": False, "error": "fasta field required"}), 400

        config = PipelineConfig(
            organism_name=body.get("organism", "custom_organism"),
            medium=body.get("medium", "glucose_minimal"),
            ecgem=body.get("ecgem", True),
            community=body.get("community", False),
        )

        try:
            result = run_full_pipeline(fasta, config)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify({
            "ok": True,
            "stages_completed": result.stages_completed,
            "num_proteins": len(result.proteins),
            "num_structures": len(result.structures),
            "kcat_predictions": len(result.kcat_predictions),
            "ecgem_growth_rate": result.ecgem.growth_rate if result.ecgem else None,
            "community_biomass": result.community.total_biomass if result.community else None,
            "pipeline_time": result.pipeline_time,
            "warnings": result.warnings,
        })

    @app.errorhandler(HelixError)
    def handle_helix_error(e):
        # User/input errors (lexical/syntactic/semantic/compile/runtime/bio): 4xx
        return jsonify({
            "ok": False,
            "error": str(e),
            "type": type(e).__name__,
        }), 400

    @app.errorhandler(Exception)
    def handle_unexpected(e):
        # Unexpected errors (implementation bugs, AttributeError, etc.): 5xx; avoid exposing the stack to the frontend
        return jsonify({
            "ok": False,
            "error": f"internal error: {e}",
            "type": type(e).__name__,
        }), 500

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not found", "path": request.path}), 404

    return app


def run_server(host: str = "127.0.0.1", port: int = 5000,
               debug: bool = False) -> int:
    """Start the development server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)
    return 0


# For `python -c "from helixlang.server import app; app.run()"`
app = create_app()


if __name__ == "__main__":
    run_server()
