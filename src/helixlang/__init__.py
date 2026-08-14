"""HelixLang: DNA codons → bytecode → biological simulation."""

__version__ = "2026.8.1"

from helixlang.codon_table import (
    CILIATE_TABLE,
    MITO_VERTEBRATE_TABLE,
    OP_OPERAND_BYTES,
    STANDARD_TABLE,
    TABLES,
    Op,
    get_table,
    wobble,
)
from helixlang.environment import (
    ACETATE_DIFFUSION_UM2_S,
    BULK_GLUCOSE_MM,
    BULK_OXYGEN_MM,
    CROMICS_CRITICAL_VOLUME_FRACTION,
    GLUCOSE_DIFFUSION_UM2_S,
    GLUCOSE_HALF_SATURATION_MM,
    OXYGEN_DIFFUSION_UM2_S,
    OXYGEN_HALF_SATURATION_MM,
    SITE_VOLUME_L,
    ConcentrationField,
    ConcentrationField3D,
    Environment,
    EnvironmentConfig,
    atp_yield,
    crowding_diffusion_factor,
    michaelis_menten_rate,
    molecules_per_site,
    monod_uptake,
)
from helixlang.errors import (
    CompileError,
    HelixError,
    LexError,
    ParseError,
    RegulationError,
    RuntimeHelixError,
    SemanticError,
)
from helixlang.grn import GRN
from helixlang.metabolism import (
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
    FluxBalanceAnalysis,
    MetabolicModel,
    Reaction,
)
from helixlang.morphology_3d import LSystem3D, Point3D
from helixlang.population import CellPopulation, CellPopulation3D, PopulationConfig
from helixlang.stochastic import (
    TelegraphPromoter,
    fano_to_noise_std,
    gillespie_telegraph,
    telegraph_fano_factor,
)
from helixlang.units import (
    AI2_DIFFUSION_UM2_S,
    ATP_PER_GLUCOSE,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    TIME_TICK_MIN,
    TIME_TICK_S,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    ticks_to_min,
)
from helixlang.vm import CellVM, Program

__all__ = [
    "Op", "STANDARD_TABLE", "MITO_VERTEBRATE_TABLE", "CILIATE_TABLE",
    "TABLES", "get_table", "wobble", "OP_OPERAND_BYTES",
    "HelixError", "LexError", "ParseError", "SemanticError",
    "CompileError", "RegulationError", "RuntimeHelixError",
    "TIME_TICK_MIN", "TIME_TICK_S", "LATTICE_SPACING_UM",
    "AI2_DIFFUSION_UM2_S", "DIFFUSION_DT_S", "ATP_PER_GLUCOSE",
    "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "ticks_to_min", "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
    # environment
    "Environment", "EnvironmentConfig", "ConcentrationField",
    "ConcentrationField3D",
    "GLUCOSE_DIFFUSION_UM2_S", "OXYGEN_DIFFUSION_UM2_S",
    "ACETATE_DIFFUSION_UM2_S",
    "GLUCOSE_HALF_SATURATION_MM", "OXYGEN_HALF_SATURATION_MM",
    "BULK_GLUCOSE_MM", "BULK_OXYGEN_MM", "SITE_VOLUME_L",
    "CROMICS_CRITICAL_VOLUME_FRACTION",
    "monod_uptake", "michaelis_menten_rate",
    "molecules_per_site", "atp_yield", "crowding_diffusion_factor",
    # stochastic gene expression
    "TelegraphPromoter", "telegraph_fano_factor",
    "fano_to_noise_std", "gillespie_telegraph",
    # GRN + VM + population
    "GRN", "CellVM", "Program", "CellPopulation", "CellPopulation3D",
    "PopulationConfig",
    # 3D morphology
    "LSystem3D", "Point3D",
    # flux balance analysis
    "MetabolicModel", "Reaction", "ECOLI_CORE_MODEL",
    "FluxBalanceAnalysis", "DynamicFBAConfig", "DynamicFluxBalance",
    "__version__",
]
