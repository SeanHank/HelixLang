"""AST node definitions. All nodes use slots=True to reduce memory usage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Codon:
    """A codon in the source program."""
    seq: str
    index: int
    line: int


@dataclass(slots=True)
class Promoter:
    """Promoter: determines the gene expression threshold."""
    name: str
    strength: float
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Gene:
    """Gene: a bytecode block containing an ORF (open reading frame)."""
    name: str
    promoter: str | None
    codons: list[Codon]
    orf: list[Codon]
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Regulation:
    """Regulation edge: source activates/suppresses target."""
    source: str
    target: str
    strength: float


@dataclass(slots=True)
class LSystemDecl:
    """L-system declaration."""
    name: str
    axiom: str
    rules: dict[int, dict[str, str]]
    angle: float
    step: float = 1.0


@dataclass(slots=True)
class FieldDecl:
    """Reaction-diffusion morphogen field declaration."""
    size: int
    F: float
    k: float
    Du: float
    Dv: float


@dataclass(slots=True)
class MorphogenFeedback:
    """Declarative morphogen→gene feedback wiring (G9).

    Binds a morphogen channel concentration at the cell position to a
    gene's GRN expression level: ``level += concentration * gain``
    (clamped to 1.0).  Replaces the legacy hard-coded ``"pigment"``
    feedback so any gene can read any channel.
    """
    gene: str
    channel: str = "V"     # "U" (substrate) or "V" (morphogen/signal)
    gain: float = 0.1


@dataclass(slots=True)
class MediaDecl:
    """Growth medium declaration (#media): a nutrient concentration (mM).

    The backend adapter turns these into FBA uptake bounds
    (``VirtualCellConfig.uptake`` / ``DynamicFBAConfig``) or shared
    ``Environment`` field inits for the population backend.
    """
    nutrient: str
    concentration: float
    diffusion_um2_s: float | None = None


@dataclass(slots=True)
class EnzymeDecl:
    """Enzyme--reaction binding (#enzyme) for enzyme-constrained FBA."""
    gene: str
    reaction: str
    kcat: float | None = None
    km: float | None = None


@dataclass(slots=True)
class ReactionDecl:
    """Direct reaction definition (#reaction) for DSL-authored metabolic networks."""
    id: str
    name: str = ""
    substrate: str = ""
    substrate_coeff: float = -1.0
    product: str = ""
    product_coeff: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = 1000.0
    subsystem: str = "other"
    reversible: bool = False


@dataclass(slots=True)
class PoolDecl:
    """Intracellular metabolite pool initialisation (#metabolite)."""
    name: str
    init: float = 0.0


@dataclass(slots=True)
class Config:
    """Runtime config."""
    ticks: int = 100
    output: list[str] = field(default_factory=lambda: ["stdout"])
    table: str = "standard"
    ops_per_tick: int = 64
    react_steps: int = 1
    # Central dogma pipeline switch: when enabled, the VM performs transcription-translation in the tick loop
    use_central_dogma: bool = False
    # Species (affects codon usage, tRNA abundance)
    species: str = "ecoli"
    # Backend selector: classic (bytecode VM) | whole_cell | population |
    # fba | calibration | benchmark (see 12-helix-language-wiring.md)
    backend: str = "classic"
    # doc/38 §2.2: allow the tick loop to hand arithmetic segments to the
    # optional C-dispatch accelerator. The VM observes what actually ran
    # (accel_used / accel_ops), never the request.
    use_accel: bool = True
    # Simulation parameters not consumed by the classic pipeline, preserved
    # verbatim as strings; coerced by the backend adapter (sim_runtime.py).
    sim: dict[str, str] = field(default_factory=dict)
    # doc/41 Item 5 Ring 2: unit-tagged ``#config`` numeric values resolved to
    # a Quantity (value preserved verbatim in ``sim`` for source round-trip;
    # this parallel map carries the parsed physical quantity). Keys mirror
    # ``sim``; populated only for values with a recognized unit suffix.
    quantities: dict[str, Any] = field(default_factory=dict)

    def quantity(self, key: str, unit: str) -> Any:
        """Return the ``#config`` value for ``key`` as a Quantity in ``unit``.

        Raises :class:`~helixlang.core.dimensions.UnitError` when the value is
        not a unit-carrying number of a compatible dimension, or
        :class:`KeyError` when ``key`` is not a unit-tagged config value.
        """
        q = self.quantities[key]
        return q.convert_to(unit)


@dataclass(slots=True)
class BioInstruction:
    """Bio instruction: advanced biological operations such as CRISPR/evolution/epigenetics.

    Declared in .helix source as #crispr / #evolve / #methylate annotations.
    At VM execution time it is dispatched to the corresponding module by kind.
    """
    kind: str            # "crispr" | "evolve" | "methylate" | "histone" |
                         # "transcribe" | "translate" | "quorum"
    target: str          # target gene name
    params: dict[str, str] = field(default_factory=dict)
    line: int = 0


@dataclass(slots=True)
class UseDecl:
    """A ``#use`` plugin opt-in statement (doc/36 §4).

    Declared in .helix source as ``use <plugin> [--flag ...]``.  Canonicalized
    via :func:`helixlang.core.use_stmt.parse_use_line`; recorded on the Program
    so the semantic analyzer / VM can resolve the plugin through the registry.
    """
    plugin: str
    flags: frozenset[str] = frozenset()
    model: str | None = None
    line: int = 0
    col: int = 0


@dataclass(slots=True)
class Program:
    """Top-level AST for a HelixLang program."""
    genes: list[Gene] = field(default_factory=list)
    promoters: list[Promoter] = field(default_factory=list)
    regulations: list[Regulation] = field(default_factory=list)
    lsystems: dict[str, LSystemDecl] = field(default_factory=dict)
    field_decl: FieldDecl | None = None
    morphogen_feedback: list[MorphogenFeedback] = field(default_factory=list)
    config: Config = field(default_factory=Config)
    # Bio instruction list (P0-1.1 language extension)
    bio_instructions: list[BioInstruction] = field(default_factory=list)
    # Type annotations (P0-1.3 type system)
    type_annotations: dict[str, str] = field(default_factory=dict)
    # Simulation-library declarations (consumed by sim_runtime, inert under classic)
    media: list[MediaDecl] = field(default_factory=list)
    enzymes: list[EnzymeDecl] = field(default_factory=list)
    reactions: list[ReactionDecl] = field(default_factory=list)
    pools: list[PoolDecl] = field(default_factory=list)
    # Open #sim key=value extension point (forward-compatible long-tail hook)
    sim_extensions: dict[str, Any] = field(default_factory=dict)
    # Typed extension namespace (doc/38 §6.3); a view over sim_extensions.
    _extensions_cached: Any = field(default=None, init=False, repr=False,
                                    compare=False)
    # Plugin opt-in statements (`use <plugin> [--flag ...]`, doc/36 §4)
    use_directives: list[UseDecl] = field(default_factory=list)

    @property
    def extensions(self) -> Any:
        """Typed, namespaced extension sections (doc/38 §6.3 E2).

        ``program.extensions.human.drugs`` reads the ``human`` section's
        ``drugs`` field; ``program.extensions.extension_for(k, v)`` routes a
        write to its owner.  A read-only-type view over ``sim_extensions`` for
        the migration window (single store, byte-identical legacy codecs).
        """
        if self._extensions_cached is None:
            from helixlang.core.extensions import ProgramExtensions

            self._extensions_cached = ProgramExtensions(self)
        return self._extensions_cached
