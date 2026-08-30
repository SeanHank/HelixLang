"""Incremental JIT: edit -> closure -> patch IR -> re-lower (doc/38 §3).

The single-source-of-truth loop, incremental:

    DNA source (only truth) -> Parser -> Program
      -> GeneDependencyGraph (regulations + OP_CALL_GENE edges)
      -> per-gene ORF/header hashing (IncrementalCache)
      -> edit detected -> invalidation closure
      -> re-derive only closure IR functions -> IRProgram.patch_gene
      -> re-lower (article chunk, byte-identical to a from-scratch compile)

The last step makes the pipeline *faithfully differential*: the patched IR
lowers to a byte-identical Chunk to ``Compiler.compile`` of the same source,
so the incremental result is guaranteed equal to a full rebuild — the
acceptance's differential test.  The savings come from not re-running the
solver/optimiser over untouched genes and from never re-hashing untouched
ORFs.

Down-callers: this module imports from :mod:`ir_builder`, :mod:`ir_lower`
and :mod:`language` only — it participates in the Layer-2 plugin model (no
reverse imports from the runtime side).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field

from helixlang.core.ast_nodes import Gene, Program
from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import Op
from helixlang.core.ir import IRProgram
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_lower import IRLowerer
from helixlang.core.language import LanguageConfig


def hash_gene_block(gene: Gene) -> str:
    """SHA-256 over a gene's ORF codon block and ``#gene`` header fields.

    This is the per-gene edit discriminator (doc/38 §3 step 2): changing any
    codon or header field (including a ``pure=1`` purity declaration, which
    effect typing consumes) changes the hash; untouched genes do not.
    """
    h = hashlib.sha256()
    h.update(gene.name.encode("utf-8"))
    for key in sorted(gene.fields):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(gene.fields[key].encode("utf-8"))
        h.update(b"\x01")
    for codon in gene.orf:
        h.update(codon.seq.encode("utf-8"))
    return "sha256:" + h.hexdigest()


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
    stores the source), so it needs no separate file format.
    """

    version: int = 2
    block_hashes: dict[str, str] = field(default_factory=dict)
    annotations: str = ""

    @classmethod
    def compute(cls, program: Program) -> IncrementalCache:
        return cls(
            block_hashes={g.name: hash_gene_block(g) for g in program.genes},
            annotations=hash_annotations(program))

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


@dataclass(slots=True)
class CompileResult:
    """Output of :meth:`IncrementalCompiler.compile`."""

    ir: IRProgram
    chunk: Chunk
    cache: IncrementalCache
    stats: IncrementalStats


class IncrementalCompiler:
    """Edit-to-patch pipeline: closure-limited gene recompiles (doc/38 §3).

    The first compile is a full build.  Subsequent compiles hash each gene
    block, diff against the previous cache, and:

    - shape change (gene added / removed / renamed) -> full rebuild;
    - otherwise re-derive only ``closure(edited)`` IR functions, patch them
      into the previous IR in place, and re-lower the whole program so the
      chunk is byte-identical to a from-scratch compile.
    """

    def __init__(self, config: LanguageConfig | None = None):
        self.config = config or LanguageConfig.for_table("standard")

    def compile(self, program: Program, *, previous_ir: IRProgram | None = None,
                previous_cache: IncrementalCache | None = None) -> CompileResult:
        cache = IncrementalCache.compute(program)
        builder = IRBuilder(dict(self.config.codon_to_op))

        def _full() -> CompileResult:
            ir = builder.build(program, table_name=self.config.table_name)
            return CompileResult(
                ir=ir,
                chunk=IRLowerer().lower(ir),
                cache=cache,
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

        closure = GeneDependencyGraph.build(
            program, previous_ir).closure(edited)
        by_name = {g.name: g for g in program.genes}
        ir = previous_ir
        for name in sorted(closure):
            ir.patch_gene(name, builder.build_function(
                program, by_name[name], table_name=self.config.table_name))
        rebuilt = sorted(closure)
        reused = [fn.name for fn in ir.functions if fn.name not in closure]
        return CompileResult(
            ir=ir,
            chunk=IRLowerer().lower(ir),
            cache=cache,
            stats=IncrementalStats(
                full_build=False,
                rebuilt=rebuilt,
                reused=reused),
        )
