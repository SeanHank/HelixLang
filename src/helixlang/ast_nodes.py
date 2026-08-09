"""AST node definitions. All nodes use slots=True to reduce memory usage."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    # Unit mode: "gameplay" (default, legacy dimensionless budget) or
    # "real" (opt-in physical calibration via helixlang.units — energy in
    # ATP, signal in uM, GRN decay from the 110 min protein half-life).
    units: str = "gameplay"


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
class Program:
    """Top-level AST for a HelixLang program."""
    genes: list[Gene] = field(default_factory=list)
    promoters: list[Promoter] = field(default_factory=list)
    regulations: list[Regulation] = field(default_factory=list)
    lsystems: dict[str, LSystemDecl] = field(default_factory=dict)
    field_decl: FieldDecl | None = None
    config: Config = field(default_factory=Config)
    # Bio instruction list (P0-1.1 language extension)
    bio_instructions: list[BioInstruction] = field(default_factory=list)
    # Type annotations (P0-1.3 type system)
    type_annotations: dict[str, str] = field(default_factory=dict)
