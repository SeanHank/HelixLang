"""Virtual-cell integration and validation benchmarks (T3.4, gap G8).

Ties the four modeling layers into one cell-cycle budget model and
publishes standardized benchmark cases:

- :class:`VirtualCell`: a Karr-2012-style cell that couples
  :mod:`helixlang.plugins.runtime.central_dogma` (transcription/translation -> protein),
  :mod:`helixlang.plugins.runtime.grn` (regulation decides which genes are expressed),
  :mod:`helixlang.plugins.runtime.metabolism` (FBA biomass flux -> energy budget) and a
  cell-cycle energy budget (maintenance + division gate).
- :class:`CellCyclePhase`: the temporal cell-cycle state machine
  (B_GAP -> C_PERIOD -> D_PERIOD -> DIVISION) that schedules chromosome
  replication so DNA copy number (gene dosage) rises during the
  replication period and doubles just before division — the minimal
  Cooper-Helmstetter pattern (``replication_mode="cooper_helmstetter"``).
- protein maturation / QC (``protein_maturation_mode="chaperone"``):
  newly translated protein is folded or misfolded at first order
  (Balchin 2016); the mature folded pool (with ~110 min turnover) is
  what the GRN sees, so expression is delayed and damped.
- :func:`fit_parameters`: a parameter-estimation harness (randomized
  search + coordinate refinement) that fits model parameters to observed
  data (e.g. doubling time, protein levels).
- :func:`run_biofilm_benchmark`: BM3-style uniform-biofilm growth metrics
  over a :class:`~helixlang.plugins.runtime.population.CellPopulation`.
- :func:`perturbation_response`: perturbation-response benchmark for a
  GRN (gene knockout, continuous-time response and settling metrics).

References:
- Karr et al. 2012. A whole-cell computational model of M. genitalium.
  Cell 150:389-401.
- Cooper & Helmstetter 1968. Chromosome replication and the division
  cycle of Escherichia coli B/r. J Mol Biol 31:519-540.
- Balchin et al. 2016. In vivo aspects of protein folding and quality
  control. Science 353:aac4354.
- Virtual Cell Challenge 2025 (integrated whole-cell benchmarks).
- iDynoMiCS 2.0 / BM3 biofilm benchmark (biofilm growth metrics).
"""
from __future__ import annotations

import enum
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from helixlang.api.units import (
    PROTEIN_AGGREGATION_RATE_PER_MIN,
    PROTEIN_DEGRADED_RATE_PER_MIN,
    PROTEIN_FOLD_RATE_PER_MIN,
    PROTEIN_FOLDING_ATP_PER_PROTEIN,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    PROTEIN_MISFOLD_RATE_PER_MIN,
    UNITS_ADDER_VOLUME_UM3,
    UNITS_CELL_C_PERIOD_MIN,
    UNITS_CELL_D_PERIOD_MIN,
    UNITS_CELL_DENSITY_DRY_PG_UM3,
    UNITS_CELL_DOUBLING_TIME_RICH_MIN,
    UNITS_CELL_SURFACE_EXPONENT,
    UNITS_CELL_VOLUME_NEWBORN_UM3,
    decay_from_half_life_ticks,
)
from helixlang.plugins.runtime.bio_data import ECOLI_CODON_USAGE
from helixlang.plugins.runtime.cell import INITIAL_CELL_ENERGY
from helixlang.plugins.runtime.central_dogma import (
    PROTEINS_PER_MRNA_LIFETIME,
    ProteinPool,
    advance_protein_pool,
    transcribe,
    translate,
)
from helixlang.plugins.runtime.grn import GRN, integrate_grn
from helixlang.plugins.runtime.metabolism import (
    DEFAULT_ENZYME_SCALE,
    ECOLI_CORE_GENE_REACTIONS,
    ECOLI_CORE_KCAT,
    ECOLI_CORE_MODEL,
    EnzymeCapacity,
    FluxBalanceAnalysis,
    MetabolitePool,
)
from helixlang.plugins.runtime.population import CellPopulation

#: translation cost per amino acid (ATP; Karr et al. 2012)
VIRTUAL_TRANSLATION_ATP_PER_AA = 4.0
#: transcription cost per nucleotide (ATP; Karr et al. 2012)
VIRTUAL_TRANSCRIPTION_ATP_PER_NT = 1.0
#: basal maintenance cost (ATP/min; ~2.5e7 for a newborn E. coli,
#: Orth 2010 + Alberts dry mass)
VIRTUAL_MAINTENANCE_ATP_PER_MIN = 2.5e7
#: energy budget required to divide (ATP)
VIRTUAL_DIVISION_ENERGY = 2.0e9
#: ATP gained per unit biomass flux per minute (coupling constant
#: between FBA biomass flux and the whole-cell energy budget)
VIRTUAL_BIOMASS_TO_ATP = 1.0e6
#: dry-mass gained per unit biomass flux per minute (pg dry weight; the
#: Phase-2 volume model grows ``volume_um3`` from biomass flux through the
#: dry-mass density rho, Milo & Phillips 2015).  Chosen so that at the
#: rich-medium biomass flux (~1.28 flux units/min) a newborn cell adds its
#: own volume (~1.6 um3, UNITS_ADDER_VOLUME_UM3) in one doubling time.
VIRTUAL_BIOMASS_TO_VOLUME_PG_PER_MIN = 0.009375
#: hard ceiling on per-gene DNA copy number (2^5) so multifork
#: replication cannot blow up the copy-count state (the tau = 20 min,
#: C = 40 min steady-state wave peaks at 8 origin copies / 2 terminus
#: copies at birth, so 2^5 leaves ample headroom)
MAX_DNA_COPY_NUMBER = 32


# ============================================================================
# Replicon model (Phase-C C2: chromosome oriC/terC + plasmids)
# ============================================================================

@dataclass(slots=True, frozen=True)
class RepliconSpec:
    """One replicon: the chromosome or a plasmid (doc/19 §5.5 C2).

    Attributes:
        kind: ``"chromosome"`` (fork-driven copy number, Cooper &
            Helmstetter) or ``"plasmid"`` (constant base copy number).
        copy_number: base DNA copy number.  For ``"chromosome"`` this is
            ignored (copy number is driven by replication forks); for a
            ``"plasmid"`` it is the constant dosage each gene of that
            replicon carries (pBR322 ~20, pUC ~500).
    """

    kind: str = "chromosome"
    copy_number: int = 1


# ============================================================================
# Cell-cycle phase state machine (Phase 1: Cooper-Helmstetter timing)
# ============================================================================

class CellCyclePhase(enum.Enum):
    """Temporal cell-cycle phase (Cooper & Helmstetter 1968).

    Values (in cycle order):

    - ``B_GAP``: post-division gap, before the origin fires (single-copy
      DNA, no active replication fork).
    - ``C_PERIOD``: chromosome replication is in progress; a replication
      fork is traversing the chromosome (DNA copy number steps 1 -> 2 ->
      4 as the fork passes each gene's coordinate).
    - ``D_PERIOD``: replication terminated; the cell waits the D-period
      before division (all loci at their doubled dosage).
    - ``DIVISION``: the division event itself (energy-gated in
      :class:`VirtualCell`).
    """
    B_GAP = "B_GAP"
    C_PERIOD = "C_PERIOD"
    D_PERIOD = "D_PERIOD"
    DIVISION = "DIVISION"


def _next_scheduled_division(age: float, doubling_time: float) -> float:
    """Smallest scheduled division time (multiple of tau) >= ``age``."""
    return ((age // doubling_time) + 1) * doubling_time


# ============================================================================
# Gene encoding
# ============================================================================

#: preferred E. coli codon per amino acid (most frequent, ECOLI_CODON_USAGE)
_PREFERRED_CODON: dict[str, str] = {}
for _codon, _info in ECOLI_CODON_USAGE.items():
    _PREFERRED_CODON.setdefault(_info[0], _codon)


def encode_gene(protein: str, rbs: str = "AGGAGG") -> str:
    """Build a translatable CDS DNA sequence from a protein sequence.

    ``RBS + spacer + ATG(start) + codons(rest) + TAA`` using the most
    frequent E. coli codon for each amino acid (ECOLI_CODON_USAGE); the
    RBS sits within 15 nt of the start codon so :func:`translate`
    detects it.  The start codon supplies the N-terminal methionine, so
    ``encode_gene("MA...")`` and ``encode_gene("A...")`` both translate
    to a protein beginning with ``M``.
    """
    if not protein:
        raise ValueError("protein sequence must be non-empty")
    # the leading residue is encoded by the ATG start codon when it is
    # already a methionine; otherwise add the start codon explicitly
    if protein[0] == "M":
        codons = "".join(_PREFERRED_CODON.get(aa, "GCT")
                         for aa in protein[1:])
        return rbs + "GACC" + "ATG" + codons + "TAA"
    codons = "".join(_PREFERRED_CODON.get(aa, "GCT") for aa in protein)
    return rbs + "GACC" + "ATG" + codons + "TAA"


# ============================================================================
# Integrated virtual cell
# ============================================================================

@dataclass(slots=True)
class VirtualCellConfig:
    """Whole-cell budget parameters (Karr 2012-style).

    Args:
        energy_init: starting energy budget (ATP molecules).
        division_energy: energy at which the cell divides (halving).
        maintenance_atp_per_min: basal maintenance burn (ATP/min).
        biomass_to_atp: energy gain per unit FBA biomass flux per minute.
        translation_atp_per_aa: translation cost (ATP/amino acid).
        transcription_atp_per_nt: transcription cost (ATP/nucleotide).
        protein_yield_per_mrna: protein molecules per transcribed mRNA.
        minutes_per_step: physical time of one ``step()``.
        uptake: initial FBA exchange uptake bounds (metabolite -> rate),
            applied to the solver at construction.
        replication_mode: chromosome replication model.  ``"flat"``
            (default) keeps every gene at single copy for the whole life
            (today's behaviour, bit-for-bit); ``"cooper_helmstetter"``
            schedules replication forks so DNA copy number rises from
            origin to terminus during the C-period and doubles before
            division (Cooper & Helmstetter 1968; Karr 2012).
        chromosome_map: gene -> chromosome coordinate in [0, 1]
            (0 = origin-proximal, 1 = terminus-proximal).  Only used in
            ``"cooper_helmstetter"`` mode; genes absent from the map are
            treated as origin-proximal (coordinate 0).
        c_period_min: C period, minutes for one round of chromosome
            replication (default 40, Cooper & Helmstetter 1968).
        d_period_min: D period, minutes between replication termination
            and division (default 20).
        doubling_time_min: scheduled doubling time tau, minutes between
            origin-firing events (default 20, rich medium); with
            ``tau < c_period_min + d_period_min`` replication overlaps
            (multifork).
        division_rule: what triggers division.  ``"energy"`` (default)
            divides when the ATP budget crosses ``division_energy``
            (today's behaviour); ``"adder"`` implements the adder size-
            control law and divides when the cell has added
            ``adder_volume_um3`` since birth (Taheri-Araghi 2015).
        volume_init_um3: newborn cell volume (um3; default 1.6,
            Taheri-Araghi 2015).
        adder_volume_um3: fixed added-volume threshold Delta for the
            adder rule (default 1.6; Delta ~ V_birth in rich medium).
        cell_density_dry_pg_um3: dry-mass density rho used to convert
            biomass flux into volume (0.15 pg/um3, Milo & Phillips 2015).
        biomass_to_volume_pg_per_min: dry mass (pg) accrued per unit
            biomass flux per minute (volume model).
        surface_scaling: scale FBA uptake bounds with
            ``(volume/volume_init)^surface_exponent`` so uptake follows
            surface area instead of being constant (default False,
            preserving today's constant-flux behaviour bit-for-bit).
        surface_exponent: surface-to-volume exponent (2/3 sphere;
            UNITS_CELL_SURFACE_EXPONENT).
        adder_noise_std: relative Gaussian noise on the adder threshold
            (default 0 = deterministic); drives the newborn-size
            distribution CV toward the ~0.1 observed by Taheri-Araghi.
        seed: RNG seed for adder noise (default None).
        protein_maturation_mode: how translated protein becomes functional.
            ``"instant"`` (default) credits protein immediately (today's
            behaviour, bit-for-bit); ``"chaperone"`` routes protein through
            the folding/QC pools of :func:`advance_protein_pool` so the
            folded pool lags and damps expression (Balchin 2016).
        frac_cotranslational_fold: fraction of newly translated protein
            that folds immediately (co-translational folding, ~30-40% of
            the cytosol; Balchin 2016); the rest enter the unfolded pool.
        folding_atp_per_protein: ATP cost per chaperone-assisted fold.
        fold_rate_per_min / misfold_rate_per_min: first-order folding vs
            misfolding rates; folded equilibrium = k_fold/(k_fold+k_misfold).
        aggregation_rate_per_min / degraded_rate_per_min: misfolded protein
            is aggregated (inert) or proteolyzed (Lon/Clp).
        protein_half_life_min: folded-pool turnover half-life (median
            ~110 min, Mosteller 1980 / Helbig 2011).
        enzyme_capacity_enabled: wire the FBA to the proteome (Phase 4,
            MOMENT/sMOMENT).  When True every reaction gated by
            :data:`ECOLI_CORE_GENE_REACTIONS` is capped by
            ``kcat * E * enzyme_scale`` with E the gene's folded
            ProteinPool (Phase 3), so metabolism responds to enzyme
            abundance (O'Brien 2013).  Default False (unchanged solve).
        enzyme_scale: global kcat rescaling for the enzyme caps (the
            Phase-5 calibration hook; GECKO's kcat-correction, Sanchez
            2017).
        protein_mass_fraction: optional sMOMENT global enzyme-pool budget
            P (g/gDW); when set, the FBA adds the pool row
            ``Sum v_i * MW_i / kcat_i <= P`` (Bekiaris & Klamt 2020).
            ``None`` disables the pool row.
        metabolite_pools_enabled: integrate the intracellular metabolite
            pools (Phase 4, :class:`~helixlang.plugins.runtime.metabolism.MetabolitePool`)
            one Euler step per cell tick from the current flux solution;
            history exposes ``metabolite_pools`` and
            ``overflow_secretion``.  Default False (unchanged behaviour).
        replicons: named replicon specs (Phase-C C2).  ``"chromosome"``
            is implicit; any other name is a plasmid whose genes carry a
            constant base copy number instead of fork-driven dosage.
        gene_replicons: gene -> replicon name.  Genes absent from the map
            belong to the chromosome.  A gene on a plasmid takes its DNA
            copy number (and therefore its expression dosage) from the
            replicon, not from replication forks.
    """

    energy_init: float = INITIAL_CELL_ENERGY
    division_energy: float = VIRTUAL_DIVISION_ENERGY
    maintenance_atp_per_min: float = VIRTUAL_MAINTENANCE_ATP_PER_MIN
    biomass_to_atp: float = VIRTUAL_BIOMASS_TO_ATP
    translation_atp_per_aa: float = VIRTUAL_TRANSLATION_ATP_PER_AA
    transcription_atp_per_nt: float = VIRTUAL_TRANSCRIPTION_ATP_PER_NT
    protein_yield_per_mrna: float = PROTEINS_PER_MRNA_LIFETIME
    minutes_per_step: float = 1.0
    uptake: dict[str, float] = field(default_factory=dict)
    replication_mode: str = "flat"
    chromosome_map: dict[str, float] | None = None
    c_period_min: float = UNITS_CELL_C_PERIOD_MIN
    d_period_min: float = UNITS_CELL_D_PERIOD_MIN
    doubling_time_min: float = UNITS_CELL_DOUBLING_TIME_RICH_MIN
    division_rule: str = "energy"
    volume_init_um3: float = UNITS_CELL_VOLUME_NEWBORN_UM3
    adder_volume_um3: float = UNITS_ADDER_VOLUME_UM3
    cell_density_dry_pg_um3: float = UNITS_CELL_DENSITY_DRY_PG_UM3
    biomass_to_volume_pg_per_min: float = VIRTUAL_BIOMASS_TO_VOLUME_PG_PER_MIN
    surface_scaling: bool = False
    surface_exponent: float = UNITS_CELL_SURFACE_EXPONENT
    adder_noise_std: float = 0.0
    seed: int | None = None
    protein_maturation_mode: str = "instant"
    frac_cotranslational_fold: float = 0.3
    folding_atp_per_protein: float = PROTEIN_FOLDING_ATP_PER_PROTEIN
    fold_rate_per_min: float = PROTEIN_FOLD_RATE_PER_MIN
    misfold_rate_per_min: float = PROTEIN_MISFOLD_RATE_PER_MIN
    aggregation_rate_per_min: float = PROTEIN_AGGREGATION_RATE_PER_MIN
    degraded_rate_per_min: float = PROTEIN_DEGRADED_RATE_PER_MIN
    protein_half_life_min: float = PROTEIN_HALF_LIFE_MEDIAN_TICKS
    enzyme_capacity_enabled: bool = False
    enzyme_scale: float = DEFAULT_ENZYME_SCALE
    protein_mass_fraction: float | None = None
    metabolite_pools_enabled: bool = False
    replicons: dict[str, RepliconSpec] = field(default_factory=dict)
    gene_replicons: dict[str, str] = field(default_factory=dict)


class VirtualCell:
    """Integrated cell: central dogma + GRN + metabolism + cell budget.

    Each :meth:`step` (one minute):
    1. advances the cell-cycle / chromosome-replication state (only in
       ``replication_mode="cooper_helmstetter"``; ``"flat"`` keeps single
       copy, matching the pre-Phase-1 behaviour bit-for-bit);
    2. advances the GRN (which genes cross the activation threshold);
    3. transcribes + translates each triggered gene, crediting protein
       and debiting transcription/translation ATP (scaled by the current
       DNA copy number -> gene-dosage wave);
    4. solves the FBA for the biomass flux and credits the energy budget
       (``flux * biomass_to_atp``);
    5. pays basal maintenance; divides when the budget allows (halving
       energy and DNA copy numbers); dies at ``energy <= 0``.

    The model is deliberately simple but couples the four layers so a
    calibration harness (e.g. :func:`fit_parameters` on doubling time)
    can tune the energy-coupling constants against data.
    """

    def __init__(self, genome: dict[str, str], grn: GRN,
                 fba: FluxBalanceAnalysis | None = None,
                 config: VirtualCellConfig | None = None,
                 name: str = "virtual-cell") -> None:
        self.genome = dict(genome)
        self.grn = grn
        self.config = config or VirtualCellConfig()
        self.name = name
        self.fba = fba or FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        for met, rate in self.config.uptake.items():
            self.fba.set_uptake(met, rate)
        self.energy: float = self.config.energy_init
        self.age: int = 0
        self.divisions: int = 0
        self.alive: bool = True
        self.proteins: dict[str, float] = {}
        self.mrna: dict[str, float] = {}
        self.mass: float = 1.0  # relative cell mass (biomass flux units)
        self.history: list[dict] = []
        # ---- protein maturation / folding / QC pools (Phase 3) ----
        self.protein_pools: dict[str, ProteinPool] = {}
        self._folded_decay_per_min: float = decay_from_half_life_ticks(
            self.config.protein_half_life_min
        )
        # ---- intracellular metabolite pools + enzyme-constrained FBA
        #       (Phase 4) ----
        self._metabolite_pool: MetabolitePool | None = None
        self._metabolism_deltas: dict = {}
        # ---- volume + size control (Phase 2, adder rule) ----
        self.volume_um3: float = self.config.volume_init_um3
        self.volume_birth_um3: float = self.config.volume_init_um3
        self._base_uptake: dict[str, float] = dict(self.config.uptake)
        self._rng = random.Random(self.config.seed)
        self._adder_threshold: float = self._draw_adder_threshold()
        # ---- cell-cycle / chromosome-replication state (Phase 1) ----
        self.phase: CellCyclePhase = CellCyclePhase.B_GAP
        self.phase_progress: float = 0.0
        self.cell_cycle_age: float = 0.0
        cmap = self.config.chromosome_map or {}
        self._coords: dict[str, float] = {
            gene: cmap.get(gene, 0.0) for gene in self.genome
        }
        # ---- replicon assignment (Phase-C C2) ----
        # genes not listed in ``gene_replicons`` live on the chromosome;
        # plasmid genes carry the constant base copy number of their
        # replicon and are excluded from fork-driven dosage.
        self._replicons: dict[str, str] = {
            gene: self.config.gene_replicons.get(gene, "chromosome")
            for gene in self.genome
        }
        self._chromosome_genes: list[str] = [
            gene for gene in self.genome
            if self._replicons[gene] == "chromosome"
        ]
        self.dna_copy_number: dict[str, int] = {}
        for gene in self.genome:
            replicon = self._replicons[gene]
            if replicon == "chromosome":
                self.dna_copy_number[gene] = 1
                continue
            spec = self.config.replicons.get(replicon)
            if spec is None:
                raise ValueError(
                    f"unknown replicon {replicon!r} for gene {gene!r}; "
                    "add it to VirtualCellConfig.replicons")
            self.dna_copy_number[gene] = spec.copy_number
        #: active replication forks as ``(initiation_cell_cycle_age,
        #: progress)``; ``progress`` is the fraction (0..1) of the
        #: chromosome the fork has traversed.
        self.replication_forks: list[tuple[float, float]] = []
        self._next_fork_age: float = 0.0
        if self.config.replication_mode == "cooper_helmstetter":
            self._seed_replication_state()

    # -------- replication (Phase 1, Cooper-Helmstetter timing) --------

    @property
    def replication_fork(self) -> float:
        """Progress (0..1) of the furthest replication fork, or 1.0 when no fork is active."""
        if not self.replication_forks:
            return 1.0
        return max(p for _, p in self.replication_forks)

    def _seed_replication_state(self) -> None:
        """Place the cell in Cooper-Helmstetter steady state at birth.

        A round that fires at cell-cycle age ``t_i`` terminates at
        ``t_i + C`` and serves the scheduled division at ``t_i + C + D``
        (exactly ``D`` minutes after termination).  Origins fire every
        ``doubling_time_min`` minutes; the round serving the next
        scheduled division fires at age ``tau - C - D``, so at birth the
        cell inherits every round that fired within the last C-period
        (the multifork regime when ``tau < C + D``) and, in slow growth,
        none (the first fork fires mid-cycle).
        """
        cfg = self.config
        first_fire = (cfg.doubling_time_min - cfg.c_period_min
                      - cfg.d_period_min)
        t_i = first_fire
        while t_i <= 0.0:
            if t_i >= -cfg.c_period_min:
                # progress of this round at birth: (0 - t_i) / C
                self.replication_forks.append(
                    (t_i, (0.0 - t_i) / cfg.c_period_min))
            t_i += cfg.doubling_time_min
        self._next_fork_age = first_fire
        while self._next_fork_age <= 0.0:
            self._next_fork_age += cfg.doubling_time_min
        # credit every locus that an inherited fork has already crossed
        # (origin-proximal genes carry the higher dosage at birth)
        for _, p in self.replication_forks:
            for gene in self._chromosome_genes:
                x = self._coords[gene]
                if x <= p:
                    self.dna_copy_number[gene] = min(
                        MAX_DNA_COPY_NUMBER, self.dna_copy_number[gene] * 2)
        # drop forks that already completed (their doubling is credited)
        self.replication_forks = [
            (t_i, p) for t_i, p in self.replication_forks if p < 1.0
        ]
        self._update_phase()

    def _advance_replication(self, dt: float) -> None:
        """Advance replication forks by ``dt`` minutes and fire new forks.

        New forks fire every ``doubling_time_min`` minutes (Cooper-
        Helmstetter: each origin fires once per scheduled doubling time).
        As a fork crosses a gene's chromosome coordinate, that gene's DNA
        copy number doubles (1 -> 2 -> 4).  Completed forks (progress 1)
        are removed: their doubling has already been credited.
        """
        cfg = self.config
        self.cell_cycle_age += dt
        a = self.cell_cycle_age
        c = cfg.c_period_min
        while self._next_fork_age <= a:
            t_f = self._next_fork_age
            # a fork that fires late (energy-gated divisions faster than
            # tau) has already traversed the region behind its origin
            p0 = min(1.0, (a - t_f) / c)
            self.replication_forks.append((t_f, p0))
            for gene in self._chromosome_genes:
                x = self._coords[gene]
                if x <= p0:
                    self.dna_copy_number[gene] = min(
                        MAX_DNA_COPY_NUMBER, self.dna_copy_number[gene] * 2)
            self._next_fork_age += cfg.doubling_time_min
        updated: list[tuple[float, float]] = []
        for init, p in self.replication_forks:
            new_p = min(1.0, p + dt / c)
            for gene in self._chromosome_genes:
                x = self._coords[gene]
                if p < x <= new_p:
                    self.dna_copy_number[gene] = min(
                        MAX_DNA_COPY_NUMBER, self.dna_copy_number[gene] * 2)
            if new_p < 1.0:
                updated.append((init, new_p))
        self.replication_forks = updated
        self._update_phase()

    def _divide_replication(self) -> None:
        """Halve DNA copy numbers and restart the cell-cycle clock.

        Each daughter inherits half of every locus's copies; ongoing
        (multifork) forks continue unchanged into the new cycle while the
        next origin-firing event is shifted back by one doubling time.
        """
        cfg = self.config
        # plasmid genes keep their constant base copy number through
        # division; only chromosome genes halve (each daughter inherits
        # half of every locus's copies).
        for gene in self._chromosome_genes:
            self.dna_copy_number[gene] = max(
                1, self.dna_copy_number[gene] // 2)
        self.cell_cycle_age -= cfg.doubling_time_min
        self._next_fork_age -= cfg.doubling_time_min
        self.replication_forks = [
            (init - cfg.doubling_time_min, p)
            for init, p in self.replication_forks
        ]
        self.phase = CellCyclePhase.DIVISION
        self.phase_progress = 0.0

    def _update_phase(self) -> None:
        """Derive the current :class:`CellCyclePhase` from replication state."""
        cfg = self.config
        a = self.cell_cycle_age
        replicating = any(p < 1.0 for _, p in self.replication_forks)
        if replicating:
            phase = CellCyclePhase.C_PERIOD
        elif _next_scheduled_division(a, cfg.doubling_time_min) - a \
                <= cfg.d_period_min:
            phase = CellCyclePhase.D_PERIOD
        else:
            phase = CellCyclePhase.B_GAP
        if phase is not self.phase:
            self.phase = phase
            self.phase_progress = 0.0
        else:
            self.phase_progress += 1.0

    # -------- internals --------

    def _express(self, gene: str) -> None:
        """Transcribe + translate one gene and pay the energy cost.

        Transcription output and its ATP cost are scaled by the gene's
        current DNA copy number, so a replicating chromosome produces a
        gene-dosage wave that peaks for origin-proximal genes (Cooper &
        Helmstetter 1968; Karr et al. 2012).

        In ``"chaperone"`` maturation mode the translated protein enters
        the unfolded pool instead of being credited directly; the folded
        pool (what GRN triggers read) then lags and damps expression
        (Balchin 2016).
        """
        if gene not in self.genome:
            return
        dna = self.genome[gene]
        node = self.grn.nodes.get(gene)
        strength = max(0.0, min(1.0, node.level if node else 1.0))
        copy = self.dna_copy_number.get(gene, 1)
        transcript = transcribe(dna, promoter_strength=strength,
                                copy_number=float(copy))
        result = translate(transcript)
        cost = (len(dna) * self.config.transcription_atp_per_nt
                + len(result.protein) * self.config.translation_atp_per_aa) * copy
        self.energy -= cost
        self.mrna[gene] = self.mrna.get(gene, 0.0) + copy
        made = self.config.protein_yield_per_mrna * copy
        if self.config.protein_maturation_mode == "chaperone":
            pool = self.protein_pools.setdefault(gene, ProteinPool())
            pool.unfolded += made * (1.0 - self.config.frac_cotranslational_fold)
            pool.folded += made * self.config.frac_cotranslational_fold
        else:
            self.proteins[gene] = self.proteins.get(gene, 0.0) + made

    def _mature_proteins(self) -> None:
        """Advance folding/QC kinetics for every gene (Phase 3).

        Only the folded pool is reported in ``self.proteins`` so the GRN
        and expression triggers see mature, functional protein.  The
        returned deltas update the running degradation/aggregation
        counters used by the history record.
        """
        if self.config.protein_maturation_mode != "chaperone":
            return
        cfg = self.config
        folded, misfolded, degraded = 0.0, 0.0, 0.0
        aggregated = 0.0
        atp = 0.0
        for gene, pool in self.protein_pools.items():
            delta = advance_protein_pool(
                pool,
                fold_rate_per_min=cfg.fold_rate_per_min,
                misfold_rate_per_min=cfg.misfold_rate_per_min,
                aggregation_rate_per_min=cfg.aggregation_rate_per_min,
                degraded_rate_per_min=cfg.degraded_rate_per_min,
                fold_atp_per_protein=cfg.folding_atp_per_protein,
                folded_decay_per_min=self._folded_decay_per_min,
                dt=float(self.config.minutes_per_step),
            )
            folded += delta["folded"]
            misfolded += delta["misfolded"]
            aggregated += delta["aggregated"]
            degraded += delta["degraded"]
            atp += delta["atp_cost"]
            self.proteins[gene] = pool.folded
        self.energy -= atp
        self._maturation_deltas = {
            "folded": folded,
            "misfolded": misfolded,
            "aggregated": aggregated,
            "degraded": degraded,
            "atp_cost": atp,
        }

    def _metabolism(self) -> float:
        """Return the FBA biomass flux (mmol/gDW/h).

        Phase 4 wiring: when ``enzyme_capacity_enabled`` the FBA is given
        MOMENT-style capacity caps with enzyme levels read from the folded
        ProteinPool (or, in instant mode, the credited proteins) of every
        gene mapped in :data:`ECOLI_CORE_GENE_REACTIONS`, so metabolism
        responds to the proteome (O'Brien 2013).  When
        ``metabolite_pools_enabled`` the intracellular pools are advanced
        one Euler step and the overflow secretion fluxes are recorded in
        :attr:`_metabolism_deltas`.
        """
        cfg = self.config
        if cfg.enzyme_capacity_enabled:
            self.fba.set_enzyme_capacity(EnzymeCapacity(
                dict(ECOLI_CORE_GENE_REACTIONS),
                kcat=dict(ECOLI_CORE_KCAT),
                enzyme_scale=cfg.enzyme_scale,
                protein_mass_fraction=cfg.protein_mass_fraction,
            ))
            if self.protein_pools:
                levels = {g: p.folded for g, p in self.protein_pools.items()}
            else:
                levels = dict(self.proteins)
            self.fba.set_enzyme_levels(levels)
        sol = self.fba.solve()
        bm = self.fba.model.biomass_reaction
        flux = sol.get(bm, 0.0) if bm is not None else 0.0
        meta: dict = {}
        if cfg.metabolite_pools_enabled:
            if self._metabolite_pool is None:
                self._metabolite_pool = MetabolitePool(self.fba.model)
            dt_h = cfg.minutes_per_step / 60.0
            mu = max(0.0, flux)  # biomass flux == specific growth rate 1/h
            deltas = self._metabolite_pool.integrate(sol, mu, dt_h=dt_h)
            meta = {
                "pools": dict(self._metabolite_pool.pools),
                "overflow": self._metabolite_pool.overflow_flux(sol),
                "net_production": deltas,
            }
        if cfg.enzyme_capacity_enabled:
            meta["enzyme_levels"] = dict(self.fba.enzyme_levels)
        self._metabolism_deltas = meta
        return flux

    # -------- public API --------

    def _draw_adder_threshold(self) -> float:
        """Draw the adder added-volume threshold for a generation.

        The noise draw is made once per generation (at birth), so a cell
        divides when it has added a *fixed* Delta (with a per-cell noise
        term) since birth — the textbook adder (Taheri-Araghi 2015; the
        per-cell variance drives the newborn-size CV toward ~0.1).
        """
        cfg = self.config
        if cfg.adder_noise_std > 0.0:
            return cfg.adder_volume_um3 * (
                1.0 + self._rng.gauss(0.0, cfg.adder_noise_std))
        return cfg.adder_volume_um3

    def _wants_division(self) -> bool:
        """True when the configured division rule is satisfied."""
        cfg = self.config
        if cfg.division_rule == "adder":
            added = self.volume_um3 - self.volume_birth_um3
            return added >= self._adder_threshold
        return self.energy >= cfg.division_energy

    def _divide(self) -> None:
        """Energy- and volume-halving division event."""
        self.energy /= 2.0
        self.volume_um3 /= 2.0
        self.volume_birth_um3 = self.volume_um3
        self._adder_threshold = self._draw_adder_threshold()
        self.divisions += 1
        if self.config.protein_maturation_mode == "chaperone":
            for gene, pool in self.protein_pools.items():
                pool.unfolded /= 2.0
                pool.folded /= 2.0
                pool.misfolded /= 2.0
                self.proteins[gene] = pool.folded
        if self._metabolite_pool is not None:
            for met in self._metabolite_pool.pools:
                self._metabolite_pool.pools[met] /= 2.0
        if self.config.replication_mode == "cooper_helmstetter":
            self._divide_replication()

    def step(self) -> dict:
        """Advance one minute; returns the history entry appended."""
        cfg = self.config
        if cfg.replication_mode == "cooper_helmstetter":
            self._advance_replication(cfg.minutes_per_step)
        triggered = self.grn.step()
        for gene in triggered:
            self._express(gene)
        self._mature_proteins()
        if cfg.surface_scaling:
            scale = (self.volume_um3 / cfg.volume_init_um3
                     ) ** cfg.surface_exponent
            for met, base in self._base_uptake.items():
                self.fba.set_uptake(met, base * scale)
        flux = self._metabolism()
        self.energy += flux * cfg.biomass_to_atp * cfg.minutes_per_step
        self.energy -= cfg.maintenance_atp_per_min * cfg.minutes_per_step
        self.mass += max(0.0, flux) * 0.01
        self.volume_um3 += (
            max(0.0, flux) * cfg.biomass_to_volume_pg_per_min
            * cfg.minutes_per_step / cfg.cell_density_dry_pg_um3)
        self.age += 1
        if self.energy <= 0.0:
            self.alive = False
        elif self._wants_division():
            self._divide()
        mat = getattr(self, "_maturation_deltas", None)
        entry = {
            "age": self.age,
            "energy": self.energy,
            "alive": self.alive,
            "divisions": self.divisions,
            "mass": self.mass,
            "volume_um3": self.volume_um3,
            "volume_birth_um3": self.volume_birth_um3,
            "added_volume_um3": self.volume_um3 - self.volume_birth_um3,
            "biomass_flux": flux,
            "proteins": dict(self.proteins),
            "triggered": triggered,
            "phase": self.phase.value,
            "dna_copy_number": dict(self.dna_copy_number),
            "replication_fork": self.replication_fork,
        }
        if mat is not None:
            entry["proteins_unfolded"] = {
                gene: pool.unfolded
                for gene, pool in self.protein_pools.items()
            }
            entry["proteins_misfolded"] = {
                gene: pool.misfolded
                for gene, pool in self.protein_pools.items()
            }
            entry["proteins_degraded"] = {
                gene: pool.degraded
                for gene, pool in self.protein_pools.items()
            }
            entry["proteins_aggregated"] = {
                gene: pool.aggregated
                for gene, pool in self.protein_pools.items()
            }
            entry["folding_atp_cost"] = mat["atp_cost"]
            entry["maturation"] = mat
        mdeltas = getattr(self, "_metabolism_deltas", None)
        if mdeltas:
            if "pools" in mdeltas:
                entry["metabolite_pools"] = mdeltas["pools"]
                entry["overflow_secretion"] = mdeltas["overflow"]
                entry["metabolite_net_production"] = mdeltas["net_production"]
            if "enzyme_levels" in mdeltas:
                entry["enzyme_levels"] = mdeltas["enzyme_levels"]
        self.history.append(entry)
        return entry

    def run(self, n_steps: int) -> list[dict]:
        """Run ``n_steps`` minutes; returns :attr:`history`."""
        for _ in range(n_steps):
            if not self.alive:
                break
            self.step()
        return self.history


# ============================================================================
# Parameter estimation harness
# ============================================================================

def fit_parameters(predict: Callable[..., list[float]], observed: list[float],
                   ranges: dict[str, tuple[float, float]],
                 n_samples: int = 500, seed: int = 0,
                 refine_rounds: int = 5, n_grid: int = 50,
                 weights: Sequence[float] | None = None,
                 polish_passes: int = 64) -> dict:
    """Fit model parameters to observed data.

    ``predict(**params) -> list[float]`` is evaluated at randomized
    parameter points inside ``ranges``; the best point is then refined
    in two stages: a coordinate-wise pattern search (full-box scan per
    axis with exponentially doubling resolution, which locates the
    valley) followed by parabolic-interpolation polish (which slides
    along narrow ridges such as ``a + b*x``).

    Args:
        predict: callable mapping parameters to a predicted vector.
        observed: target vector (same length as the prediction).
        ranges: parameter name -> (lower, upper) box.
        n_samples: random samples for the global search stage.
        seed: RNG seed (deterministic runs).
        refine_rounds: pattern-search passes after the random search;
            each pass scans every axis over its full range at
            ``2**(round+2)+1`` grid points.
        n_grid: kept for compatibility (resolution doubling is fixed);
            ignored by the current implementation.
        weights: optional per-observation weights (same length as
            ``observed``) giving the objective ``sum(w_i (p_i - o_i)^2)``.
            Supports multi-scale omics calibration: observations from
            heterogeneous readouts (mRNA vs protein, high- vs low-count
            perturb-seq conditions) are jointly fitted with inverse-
            variance weights (DESeq2 2014 variance structure
            ``Var = mu + dispersion*mu^2``; Karr et al. 2012 DREAM8
            weighted fitting). ``None`` = unit weights.
        polish_passes: maximum coordinate-wise parabolic-polish passes
            after the pattern search. 64 (the default) is enough for the
            unit-test benchmark objectives; expensive simulators should
            pass a small value (e.g. 4-8), since each pass costs three
            ``predict`` calls per fitted parameter.

    Returns:
        ``{"best": {param: value}, "sse": float, "n_samples": int}``.
    """
    if not ranges:
        raise ValueError("ranges must be non-empty")
    if not observed:
        raise ValueError("observed must be non-empty")
    if weights is not None and len(weights) != len(observed):
        raise ValueError("weights must have the same length as observed")
    rng = random.Random(seed)
    names = list(ranges)

    def sse(params: dict) -> float:
        try:
            pred = predict(**params)
        except TypeError as exc:
            raise ValueError(
                f"predict must accept the fitted parameters {names!r} "
                f"as keyword arguments: {exc}") from exc
        if len(pred) != len(observed):
            raise ValueError(
                "prediction length must match observed length")
        if weights is not None:
            return sum(w * (p - o) ** 2
                       for w, p, o in zip(weights, pred, observed,
                                          strict=True))
        return sum((p - o) ** 2 for p, o in zip(pred, observed, strict=True))

    best = {n: rng.uniform(*ranges[n]) for n in names}
    best_sse = sse(best)
    total = 0
    for _ in range(n_samples):
        params = {n: rng.uniform(*ranges[n]) for n in names}
        total += 1
        s = sse(params)
        if s < best_sse:
            best, best_sse = params, s
    # stage 1: coordinate-wise pattern search over the full box with
    # exponentially doubling resolution per round. Finds the valley of a
    # correlated objective without window-halving stalls.
    for rnd in range(refine_rounds):
        for n in names:
            lo, hi = ranges[n]
            steps = 2 ** (rnd + 2)
            cand, cand_s = None, best_sse
            for i in range(steps + 1):
                v = lo + (hi - lo) * i / steps
                params = dict(best)
                params[n] = v
                total += 1
                s = sse(params)
                if s < cand_s:
                    cand_s, cand = s, v
            if cand is not None:
                best[n] = cand
                best_sse = cand_s
    # stage 2: parabolic-interpolation polish on each axis. The discrete
    # grid cannot slide along a narrow ridge (e.g. a + b*x), so fit a
    # parabola through three samples and jump to its vertex.
    for _ in range(polish_passes):
        improved = False
        for n in names:
            lo, hi = ranges[n]
            delta = max((hi - lo) / 100.0, 1e-9)
            params = dict(best)
            f0 = best_sse
            params[n] = min(hi, max(lo, best[n] - delta))
            f1 = sse(params)
            params[n] = min(hi, max(lo, best[n] + delta))
            f2 = sse(params)
            denom = f1 - 2.0 * f0 + f2
            if abs(denom) < 1e-30:
                continue
            vertex = best[n] - delta * (f2 - f1) / (2.0 * denom)
            vertex = min(hi, max(lo, vertex))
            params = dict(best)
            params[n] = vertex
            total += 1
            fv = sse(params)
            if fv < best_sse:
                best, best_sse = params, fv
                improved = True
        if not improved:
            break
    return {"best": best, "sse": best_sse, "n_samples": total}


# ============================================================================
# Standardized benchmarks
# ============================================================================

def run_biofilm_benchmark(population: CellPopulation, n_steps: int = 120,
                          interval: int = 10) -> dict:
    """BM3-style uniform-biofilm growth benchmark.

    Runs a :class:`~helixlang.plugins.runtime.population.CellPopulation` (or
    ``CellPopulation3D``) for ``n_steps`` and reports standardized
    biofilm metrics: biomass (alive-cell) time series, final biomass,
    spatial extent and an estimated doubling interval.

    Returns:
        a dict with ``biomass`` (timeseries), ``final_biomass``,
        ``max_extent``, ``doubling_ticks`` and ``growth_rate_per_tick``.
    """
    biomass: list[int] = []
    extent: list[float] = []
    for s in range(n_steps):
        population.step()
        if s % interval == 0 or s == n_steps - 1:
            alive = [c for c in population.cells if c.alive]
            biomass.append(len(alive))
            if alive:
                xs = [c.x for c in alive]
                ys = [c.y for c in alive]
                extent.append(max(max(xs) - min(xs),
                                  max(ys) - min(ys)))
            else:
                extent.append(0.0)
    growth: float = 0.0
    if len(biomass) >= 2 and biomass[0] > 0:
        growth = (biomass[-1] - biomass[0]) / biomass[0] / max(1, n_steps)
    doubling_ticks: float | None = None
    for i in range(1, len(biomass)):
        if biomass[i] >= 2 * biomass[0] and biomass[0] > 0:
            doubling_ticks = i * interval
            break
    return {
        "biomass": biomass,
        "final_biomass": biomass[-1] if biomass else 0,
        "max_extent": max(extent) if extent else 0.0,
        "doubling_ticks": doubling_ticks,
        "growth_rate_per_tick": growth,
    }


def _clone_grn(grn: GRN) -> GRN:
    out = GRN(noise_enabled=grn.noise_enabled)
    for name, node in grn.nodes.items():
        out.add_gene(name, node.threshold, initial_level=node.level,
                     decay=node.decay, hill_n=node.hill_n, kd=node.kd,
                     noise=node.noise)
    for e in grn.edges:
        out.add_edge(e.source, e.target, e.weight)
    return out


def perturbation_response(grn: GRN, target: str,
                          knockout: str | None = None,
                          t_span: tuple[float, float] = (0.0, 600.0),
                          n_points: int = 300) -> dict:
    """Perturbation-response benchmark (continuous-time GRN).

    Integrates the GRN ODE (T2.2 DOPRI5) once unperturbed and once with
    ``knockout``'s outgoing edges set to zero, then reports the response
    of ``target``: final fold change, settling time (first time the
    trajectory is within 5% of its final value) and the response curves.

    Returns:
        a dict with ``control_final``, ``perturbed_final``,
        ``fold_change``, ``settling_time`` (minutes), ``times`` and
        ``response`` (perturbed trajectory of ``target``).
    """
    control = integrate_grn(grn, t_span, n_points=n_points,
                            method="rk45").trajectory(target)
    if knockout is not None:
        grn_p = _clone_grn(grn)
        for e in grn_p.edges:
            if e.source == knockout:
                e.weight = 0.0
        grn_p._rebuild_incoming()
    else:
        grn_p = grn
    result = integrate_grn(grn_p, t_span, n_points=n_points, method="rk45")
    perturbed = result.trajectory(target)
    control_final = control[-1]
    perturbed_final = perturbed[-1]
    fold = (perturbed_final / control_final if control_final > 0
            else float("inf"))
    settling = None
    tol = 0.05 * abs(perturbed_final)
    for t, v in zip(result.times, perturbed, strict=True):
        if abs(v - perturbed_final) <= tol:
            settling = t
            break
    return {
        "control_final": control_final,
        "perturbed_final": perturbed_final,
        "fold_change": fold,
        "settling_time": settling,
        "times": result.times,
        "response": perturbed,
    }


__all__ = [
    "VirtualCell",
    "VirtualCellConfig",
    "CellCyclePhase",
    "encode_gene",
    "fit_parameters",
    "perturbation_response",
    "run_biofilm_benchmark",
    "VIRTUAL_BIOMASS_TO_ATP",
    "VIRTUAL_DIVISION_ENERGY",
    "VIRTUAL_MAINTENANCE_ATP_PER_MIN",
    "VIRTUAL_TRANSCRIPTION_ATP_PER_NT",
    "VIRTUAL_TRANSLATION_ATP_PER_AA",
]
