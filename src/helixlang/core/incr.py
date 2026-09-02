"""Incremental JIT: edit -> closure -> patch IR -> splice bytecode (doc/38 §3).

The single-source-of-truth loop, incremental:

    DNA source (only truth) -> Parser -> Program
      -> GeneDependencyGraph (regulations + OP_CALL_GENE edges)
      -> per-gene ORF/header hashing (IncrementalCache, stamped at parse time)
      -> edit detected -> invalidation closure
      -> re-derive only closure IR functions -> IRProgram.patch_gene
      -> splice closure code regions into the previous chunk (IRLowerer
         emit_gene_region), leaving untouched genes' bytes verbatim

The splice reproduces a chunk *byte-identical* to a from-scratch compile of
the same source, so the incremental result is guaranteed equal to a full
rebuild — the acceptance's differential test.  A gene edit is splice-safe when
it preserves that gene's region length (so every ``OP_CALL_GENE`` offset and
inter-gene barrier keeps its value) and introduces no constant-pool change
(so every ``OP_PUSH_CONST`` index keeps its value); any edit that violates
those two structural invariants (a gene that grows/shrinks, or a brand-new
pool literal) falls back to a whole-program re-lower of the patched IR.  The
savings come from hashing ORFs once at parse time (never re-hashed on a
recompile), re-deriving only the closure, and splicing code instead of
re-lowering the program.

Down-callers: this module imports from :mod:`ir_builder`, :mod:`ir_lower`
and :mod:`language` only — it participates in the Layer-2 plugin model (no
reverse imports from the runtime side).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field

from helixlang.core.ast_nodes import Gene, Program, gene_block_digest
from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import Op
from helixlang.core.ir import IRFunction, IRProgram
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_lower import IRLowerer
from helixlang.core.language import LanguageConfig


def hash_gene_block(gene: Gene) -> str:
    """SHA-256 over a gene's ORF codon block and ``#gene`` header fields.

    This is the per-gene edit discriminator (doc/38 §3 step 2): changing any
    codon or header field (including a ``pure=1`` purity declaration, which
    effect typing consumes) changes the hash; untouched genes do not.
    """
    return gene_block_digest(gene)


def hash_annotations(program: Program) -> str:
    """SHA-256 over the program-global ``#type`` annotations (doc/38 §8.2).

    ``#type`` bindings seed unit/dim metadata on IR instructions, so an edit
    to any annotation is a whole-program input change: the incremental cache
    treats it as a shape change (full rebuild) rather than a per-gene edit.
    """
    h = hashlib.sha256()
    for name in sorted(program.type_annotations):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(program.type_annotations[name].encode("utf-8"))
        h.update(b"\x01")
    return "sha256:" + h.hexdigest()


@dataclass(slots=True)
class IncrementalCache:
    """Sidecar state: gene name -> ORF/header block hash (doc/38 §3 step 2).

    Reproducible purely from source text (the ``SRC`` section of an artifact
    stores the source), so it needs no separate file format.  Two additions
    make the fast path closure-proportional:

    - block hashes are *stamped on the AST at parse time* (``Gene.block_hash``
      via :func:`helixlang.core.ast_nodes.gene_block_digest`), so recompiling
      a program never re-hashes untouched ORFs — the digest is produced once,
      during the unavoidable parse pass; and
    - ``graph`` carries the last program's dependency graph so the caller
      edges of untouched IR are never rescanned.
    """

    version: int = 3
    block_hashes: dict[str, str] = field(default_factory=dict)
    annotations: str = ""
    graph: GeneDependencyGraph | None = field(default=None, repr=False)

    @classmethod
    def compute(cls, program: Program,
                graph: GeneDependencyGraph | None = None) -> IncrementalCache:
        return cls(
            block_hashes={g.name: (g.block_hash or hash_gene_block(g))
                          for g in program.genes},
            annotations=hash_annotations(program),
            graph=graph)

    def diff(self, previous: IncrementalCache) -> tuple[bool, set[str]]:
        """Return ``(shape_stable, edited_gene_names)`` vs an older cache.

        A shape change (gene added / removed / renamed, or a program-global
        ``#type`` annotation edited) forces a full rebuild even when no ORF
        text changed, because ``OP_CALL_GENE`` targets that fall back to
        ``wobble % n`` resolution inherit the previous gene count and order,
        and annotation edits can retype IR values.
        """
        if self.annotations != previous.annotations:
            return False, set()
        if self.block_hashes.keys() != previous.block_hashes.keys():
            return False, set()
        changed = {name for name, value in self.block_hashes.items()
                   if previous.block_hashes.get(name) != value}
        return True, changed


@dataclass(slots=True)
class GeneDependencyGraph:
    """Program-level gene dependency DAG (doc/38 §3 step 1).

    Nodes are genes, keyed by name (the ORF-hash cache key).  Edges:

    - regulation edges ``source -> target`` for gene-sourced regulations, and
    - call edges ``target -> caller`` from ``OP_CALL_GENE`` operands.

    ``closure`` returns the invalidation set of an edit exactly as the doc/38
    §3 acceptance defines it: for each edited gene ``g``,

        {g} U {regulation sources targetting g} U {g's callers}.

    The set is deliberately *not* transitive: re-deriving more than this is
    never needed for correctness (the re-lower step recomputes every offset),
    and a transitive caller walk would rebuild an entire call chain for every
    leaf edit — the opposite of "proportional to the closure, not the program".
    """

    genes: set[str]
    regulators: dict[str, set[str]] = field(default_factory=dict)
    callers: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, program: Program,
              ir: IRProgram | None = None) -> GeneDependencyGraph:
        """Derive the graph from a program (and its previous IR, if any)."""
        graph = cls(genes={g.name for g in program.genes})
        for reg in program.regulations:
            if reg.source in graph.genes:
                graph.regulators.setdefault(reg.target, set()).add(reg.source)
        if ir is not None:
            for fn in ir.functions:
                for inst in fn.instrs:
                    if inst.opcode is Op.OP_CALL_GENE and inst.operand is not None:
                        target = str(inst.operand)
                        if target in graph.genes:
                            graph.callers.setdefault(target, set()).add(fn.name)
        return graph

    def closure(self, edited: Iterable[str]) -> set[str]:
        """Invaldation set for an edit set (doc/38 §3 acceptance, exact)."""
        closure: set[str] = set()
        for gene in edited:
            if gene not in self.genes:
                continue
            closure.add(gene)
            closure |= self.regulators.get(gene, set())
            closure |= self.callers.get(gene, set())
        return closure & self.genes


@dataclass(slots=True)
class IncrementalStats:
    """Observed work of one incremental compile (doc/38 §2 counters)."""

    full_build: bool = False
    rebuilt: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    splice: bool = False  # byte-region splice fast path vs whole-program re-lower


@dataclass(slots=True)
class CompileResult:
    """Output of :meth:`IncrementalCompiler.compile`."""

    ir: IRProgram
    chunk: Chunk
    cache: IncrementalCache
    stats: IncrementalStats


def _find_fn(ir: IRProgram, name: str) -> IRFunction:
    for fn in ir.functions:
        if fn.name == name:
            return fn
    raise KeyError(f"_find_fn: no gene {name!r} in IR program")


def _patched_ir(previous_ir: IRProgram,
                fns_by_name: dict[str, IRFunction]) -> IRProgram:
    """A copy of ``previous_ir`` with the closure functions replaced.

    Untouched ``IRFunction`` objects are shared by reference (no re-derivation,
    no whole-program copy of instruction lists); the caller's ``previous_ir``
    is left untouched so it stays a valid baseline for independent compiles.
    """
    functions = [fns_by_name.get(fn.name, fn) for fn in previous_ir.functions]
    return IRProgram(
        name=previous_ir.name,
        table=previous_ir.table,
        functions=functions,
        call_targets=dict(previous_ir.call_targets),
        use_directives=list(previous_ir.use_directives),
        lsystems=dict(previous_ir.lsystems),
        config=dict(previous_ir.config),
    )


def _call_targets(fn: IRFunction) -> list[str]:
    """Gene names this function's ``OP_CALL_GENE`` instructions target."""
    return [str(inst.operand) for inst in fn.instrs
            if inst.opcode is Op.OP_CALL_GENE and inst.operand is not None]


def _patch_graph(graph: GeneDependencyGraph, program: Program,
                 old_fns: dict[str, IRFunction],
                 new_fns: dict[str, IRFunction]) -> GeneDependencyGraph:
    """Update a cached dependency graph after a closure patch (proportional).

    Regulation edges are recomputed from ``program.regulations`` (cheap and
    program-level) and ``OP_CALL_GENE`` caller edges are updated only for the
    re-derived closure functions — never a whole-program IR scan.  ``old_fns``
    / ``new_fns`` hold each closure gene's pre-/post-patch IR function.
    """
    genes = set(graph.genes)
    regulators: dict[str, set[str]] = {}
    for reg in program.regulations:
        if reg.source in genes and reg.target in genes:
            regulators.setdefault(reg.target, set()).add(reg.source)

    callers: dict[str, set[str]] = {
        target: set(with_callers for with_callers in with_callers_set)
        for target, with_callers_set in graph.callers.items()
    }
    for name in old_fns:
        for old_target in _call_targets(old_fns[name]):
            bucket = callers.get(old_target)
            if bucket is not None:
                bucket.discard(name)
                if not bucket:
                    del callers[old_target]
        for new_target in _call_targets(new_fns[name]):
            if new_target in genes:
                callers.setdefault(new_target, set()).add(name)
    callers = {t: s for t, s in callers.items() if t in genes}
    return GeneDependencyGraph(genes=genes, regulators=regulators,
                               callers=callers)


def _splice_chunk(previous_chunk: Chunk, previous_ir: IRProgram,
                  ir: IRProgram, closure: set[str]) -> Chunk | None:
    """Byte-identical, closure-proportional chunk for a splice-safe edit (or None).

    A gene edit is *splice-safe* when every affected region keeps its previous
    byte count (so all ``OP_CALL_GENE`` targets, gene offsets and inter-gene
    barriers keep their values) and the edited genes introduce no constant-pool
    change (every ``OP_PUSH_CONST`` literal already exists in the previous
    pool, so no pool entry moves and no copied index goes stale).  Under those
    two conditions, re-emitting only the closure regions via
    :meth:`IRLowerer.emit_gene_region` and copying every untouched region
    verbatim reproduces a from-scratch lower exactly.  Any edit that violates
    them returns ``None`` and the caller falls back to a whole-program re-lower
    (which stays correct regardless).
    """
    lowerer = IRLowerer()
    if (ir.use_directives != previous_ir.use_directives
            or ir.lsystems != previous_ir.lsystems):
        return None
    offsets = previous_chunk.gene_offsets
    end = len(previous_chunk.code)
    pool = previous_chunk.constants
    chunk = Chunk()
    chunk.constants = list(pool)

    # Plugin opt-ins (doc/36 §3.2), in source order — mirrors IRLowerer.lower.
    for plugin, flags in ir.use_directives:
        const_idx = chunk.add_constant(("use_plugin", plugin, flags))
        chunk.emit(Op.OP_USE_PLUGIN, const_idx, line=0, codon_index=-1)

    def _offset_of(target: str) -> int:
        off = offsets.get(target)
        if off is None:
            raise LookupError(
                f"CALL_GENE target {target!r} not present in IR offsets; "
                f"the IR program is inconsistent")
        return off

    names = [fn.name for fn in ir.functions]
    n = len(names)
    for i, fn in enumerate(ir.functions):
        prev_start = offsets.get(fn.name)
        prev_end = offsets.get(names[i + 1]) if i + 1 < n else end
        if prev_start is None or prev_end is None:
            return None  # shape mismatch — caller's stable-check should prevent
        if fn.name in closure:
            start = len(chunk.code)
            _, region_end = lowerer.emit_gene_region(
                chunk, fn, _offset_of, is_last=(i == n - 1), end=end)
            if region_end - start != prev_end - prev_start:
                return None  # size change — full re-lower
            if any(inst.opcode is Op.OP_PUSH_CONST
                   and inst.operand not in pool for inst in fn.instrs):
                return None  # new pool entry — full re-lower
        else:
            chunk.code += previous_chunk.code[prev_start:prev_end]
            chunk.lines += previous_chunk.lines[prev_start:prev_end]
            chunk.codon_indices += (
                previous_chunk.codon_indices[prev_start:prev_end])
    chunk.gene_offsets = dict(offsets)
    return chunk


class IncrementalCompiler:
    """Edit-to-chunk pipeline: closure-limited recompiles (doc/38 §3).

    The first compile is a full build.  Subsequent compiles diff per-gene
    block hashes against the previous cache and:

    - shape change (gene added / removed / renamed, or a program-global
      ``#type`` annotation edited) -> full rebuild;
    - otherwise re-derive only ``closure(edited)`` IR functions, patch them
      into the previous IR in place, and — when a previous chunk is supplied
      and the edit is splice-safe — splice the closure's code regions into it
      (byte-identical to a from-scratch compile, proportional to the closure);
      edits that are not splice-safe still patch the IR but fall back to a
      whole-program re-lower.
    """

    def __init__(self, config: LanguageConfig | None = None):
        self.config = config or LanguageConfig.for_table("standard")

    def compile(self, program: Program, *, previous_ir: IRProgram | None = None,
                previous_cache: IncrementalCache | None = None,
                previous_chunk: Chunk | None = None) -> CompileResult:
        cache = IncrementalCache.compute(program)
        builder = IRBuilder(dict(self.config.codon_to_op))

        def _full() -> CompileResult:
            ir = builder.build(program, table_name=self.config.table_name)
            graph = GeneDependencyGraph.build(program, ir)
            next_cache = IncrementalCache.compute(program, graph=graph)
            return CompileResult(
                ir=ir,
                chunk=IRLowerer().lower(ir),
                cache=next_cache,
                stats=IncrementalStats(
                    full_build=True,
                    rebuilt=[g.name for g in program.genes],
                    reused=[]),
            )

        if previous_ir is None or previous_cache is None:
            return _full()

        stable, edited = cache.diff(previous_cache)
        if not stable:
            return _full()

        graph = previous_cache.graph
        if graph is None:
            graph = GeneDependencyGraph.build(program, previous_ir)
        closure = graph.closure(edited)

        old_fns = {name: _find_fn(previous_ir, name) for name in closure}
        by_name = {g.name: g for g in program.genes}
        new_fns = {name: builder.build_function(
            program, by_name[name], table_name=self.config.table_name)
            for name in closure}
        ir = _patched_ir(previous_ir, new_fns)

        rebuilt = sorted(closure)
        reused = [fn.name for fn in ir.functions if fn.name not in closure]
        next_graph = _patch_graph(graph, program, old_fns, new_fns)
        next_cache = IncrementalCache.compute(program, graph=next_graph)

        splice = False
        chunk: Chunk
        if previous_chunk is not None:
            spliced = _splice_chunk(previous_chunk, previous_ir, ir, closure)
            if spliced is not None:
                chunk, splice = spliced, True
            else:
                chunk = IRLowerer().lower(ir)
        else:
            chunk = IRLowerer().lower(ir)
        return CompileResult(
            ir=ir,
            chunk=chunk,
            cache=next_cache,
            stats=IncrementalStats(
                full_build=False,
                rebuilt=rebuilt,
                reused=reused,
                splice=splice),
        )
