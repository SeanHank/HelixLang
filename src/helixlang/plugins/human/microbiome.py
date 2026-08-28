"""Microbiome-Drug Interaction Modeling.

Models the gut microbiota as a metabolic compartment that:
1. Activates prodrugs (sulfasalazine → 5-ASA via bacterial azoreductase)
2. Reactivates detoxified drugs (bacterial β-glucuronidase cleaves glucuronide
   conjugates → reactivates SN-38, NSAIDs → GI toxicity)
3. Reduces drugs (Eubacterium lentum reduces digoxin → inactive metabolite)
4. Produces metabolites that affect liver function (SCFA, bile acids, TMAO)
5. Modulates immune tone via gut-liver axis

The microbiome compartment exchanges metabolites with the liver via portal
vein, creating a liver→bile→gut→microbiome→portal→liver feedback loop.

References:
- Guthrie & Bhatt, Nat Rev Microbiol 2023: gut microbiota drug metabolism
- Klaassen & Cui, Pharmacol Rev 2015: microbiome-drug interactions
- Maier et al., Nature 2018: 240+ drug-microbiome interactions
- Zip et al., Gut 2024: gut-liver axis in drug metabolism
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ============================================================================
# Known Microbial Drug-Metabolizing Reactions
# ============================================================================

def _canonical_name(name: str) -> str:
    """Normalize a compound name to match virtual_patient drug keys.

    Drug names elsewhere in the pipeline (e.g. virtual_patient.py) are
    lowercased with spaces/hyphens replaced by underscores
    ("SN-38" -> "sn_38"). Reaction substrates/products must use the same
    convention or lookups in MicrobiomeCompartment.step() silently fail.
    """
    return name.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class MicrobialReaction:
    """A single drug metabolism reaction performed by gut microbiota."""

    reaction_id: str
    substrate: str
    product: str
    enzyme_name: str  # bacterial enzyme
    organism: str     # primary bacterial species
    km_um: float      # Michaelis constant
    vmax_relative: float  # relative Vmax (arbitrary units)
    is_activation: bool   # True = prodrug activation, False = inactivation or reactivation
    description: str = ""

    def __post_init__(self) -> None:
        self.reaction_id = _canonical_name(self.reaction_id)
        self.substrate = _canonical_name(self.substrate)
        self.product = _canonical_name(self.product)
        self.enzyme_name = _canonical_name(self.enzyme_name)


# Curated from Maier et al. 2018, Guthrie & Bhatt 2023, PharmGKB microbiome section
MICROBIAL_REACTIONS: list[MicrobialReaction] = [
    # --- Prodrug activation ---
    MicrobialReaction(
        reaction_id="sulfasalazine_azo",
        substrate="sulfasalazine",
        product="5_aminosalicylic_acid",
        enzyme_name="bacterial_azoreductase",
        organism="Clostridium_sp._AVC5",
        km_um=50.0, vmax_relative=1.0,
        is_activation=True,
        description="Azo bond reduction → active 5-ASA (IBD therapy)",
    ),
    MicrobialReaction(
        reaction_id="irinotecan_reactivation",
        substrate="sn_38_glucuronide",
        product="sn_38",
        enzyme_name="beta_glucuronidase",
        organism="E._coli",
        km_um=30.0, vmax_relative=0.8,
        is_activation=False,  # reactivation of toxic metabolite
        description="Bacterial β-glucuronidase reactivates SN-38 → GI toxicity",
    ),
    MicrobialReaction(
        reaction_id="digoxin_reduction",
        substrate="digoxin",
        product="dihydrodigoxin",
        enzyme_name="digoxin_reductase",
        organism="Eubacterium_lentum",
        km_um=20.0, vmax_relative=0.6,
        is_activation=False,
        description="Inactivation of digoxin by gut bacteria",
    ),
    MicrobialReaction(
        reaction_id="nsaid_reactivation",
        substrate="ibuprofen_glucuronide",
        product="ibuprofen",
        enzyme_name="beta_glucuronidase",
        organism="Clostridium_perfringens",
        km_um=40.0, vmax_relative=0.5,
        is_activation=False,
        description="Bacterial β-glucuronidase reactivates NSAIDs → GI erosion",
    ),
    MicrobialReaction(
        reaction_id="codeine_3dmg",
        substrate="codeine",
        product="morphine",
        enzyme_name="CYP2D6_bacterial_homolog",
        organism="Pseudomonas_sp.",
        km_um=100.0, vmax_relative=0.2,
        is_activation=True,
        description="Bacterial O-demethylation of codeine → morphine (minor pathway)",
    ),
    MicrobialReaction(
        reaction_id="levodopa_decarboxylation",
        substrate="levodopa",
        product="dopamine",
        enzyme_name="aromatic_L_amino_acid_decarboxylase",
        organism="Enterococcus_faecium",
        km_um=200.0, vmax_relative=0.3,
        is_activation=False,
        description="Bacterial decarboxylation reduces levodopa bioavailability",
    ),
    # --- Bile acid metabolism ---
    MicrobialReaction(
        reaction_id="bile_salt_deconjugation",
        substrate="glycocholate",
        product="cholate",
        enzyme_name="bile_salt_hydrolase",
        organism="Lactobacillus_sp.",
        km_um=100.0, vmax_relative=1.0,
        is_activation=False,
        description="Deconjugation of bile salts → altered enterohepatic circulation",
    ),
    MicrobialReaction(
        reaction_id="bile_acid_7alpha_dehydroxylation",
        substrate="cholate",
        product="deoxycholate",
        enzyme_name="bile_acid_7alpha_dehydroxylase",
        organism="Clostridium_sp.",
        km_um=80.0, vmax_relative=0.4,
        is_activation=False,
        description="Primary→secondary bile acid conversion",
    ),
    # --- TMAO pathway ---
    MicrobialReaction(
        reaction_id="tma_formation",
        substrate="choline",
        product="trimethylamine",
        enzyme_name="TMA_lyase",
        organism="Clostridium_sp._AVC5",
        km_um=500.0, vmax_relative=0.7,
        is_activation=False,
        description="Choline → TMA → hepatic FMO3 → TMAO (CV risk marker)",
    ),
    # --- Drug inactivation ---
    MicrobialReaction(
        reaction_id="metformin_inactivation",
        substrate="metformin",
        product="guanylurea",
        enzyme_name="bacterial_guanidine_deaminase",
        organism="Shigella_sp.",
        km_um=1000.0, vmax_relative=0.3,
        is_activation=False,
        description="Bacterial degradation of metformin → reduced bioavailability",
    ),
    MicrobialReaction(
        reaction_id="mycophenolate_reactivation",
        substrate="mpag",
        product="mycophenolic_acid",
        enzyme_name="beta_glucuronidase",
        organism="E._coli",
        km_um=60.0, vmax_relative=0.5,
        is_activation=False,
        description="Bacterial β-glucuronidase reactivates MPAG → hepatotoxicity",
    ),
    # --- Phase 5: Additional reactions ---
    MicrobialReaction(
        reaction_id="sulfapyridine_hydroxylation",
        substrate="sulfapyridine",
        product="hydroxy_sulfapyridine",
        enzyme_name="bacterial_hydroxylase",
        organism="Citrobacter_freundii",
        km_um=80.0, vmax_relative=0.6,
        is_activation=False,
        description="Bacterial hydroxylation of sulfapyridine metabolite",
    ),
    MicrobialReaction(
        reaction_id="digoxin_inactivation2",
        substrate="digoxin",
        product="dihydrodigoxin",
        enzyme_name="nitroreductase",
        organism="Klebsiella_pneumoniae",
        km_um=25.0, vmax_relative=0.4,
        is_activation=False,
        description="Bacterial nitroreduction of digoxin",
    ),
    MicrobialReaction(
        reaction_id="enalapril_hydrolysis",
        substrate="enalaprilat",
        product="enalapril",
        enzyme_name="bacterial_esterase",
        organism="Lactobacillus_sp.",
        km_um=120.0, vmax_relative=0.3,
        is_activation=False,
        description="Bacterial esterase hydrolysis of enalaprilat",
    ),
    MicrobialReaction(
        reaction_id="ranitidine_n_reduction",
        substrate="ranitidine",
        product="amino_ranitidine",
        enzyme_name="nitroreductase",
        organism="Clostridium_sp.",
        km_um=200.0, vmax_relative=0.4,
        is_activation=False,
        description="Bacterial nitroreduction of ranitidine",
    ),
    MicrobialReaction(
        reaction_id="prednisone_reactivation",
        substrate="prednisone_glucuronide",
        product="prednisone",
        enzyme_name="beta_glucuronidase",
        organism="Bacteroides_sp.",
        km_um=70.0, vmax_relative=0.5,
        is_activation=False,
        description="Bacterial β-glucuronidase reactivates prednisone glucuronide",
    ),
    MicrobialReaction(
        reaction_id="warfarin_reductions",
        substrate="warfarin_alcohol",
        product="warfarin",
        enzyme_name="bacterial_keto_reductase",
        organism="Clostridium_sp.",
        km_um=90.0, vmax_relative=0.3,
        is_activation=False,
        description="Bacterial reduction of warfarin alcohol metabolite",
    ),
    MicrobialReaction(
        reaction_id="fluorouracil_degradation",
        substrate="5_fu",
        product="fluoroacetate",
        enzyme_name="bacterial_decarboxylase",
        organism="Pseudomonas_sp.",
        km_um=150.0, vmax_relative=0.2,
        is_activation=False,
        description="Bacterial degradation of 5-FU (protective)",
    ),
    MicrobialReaction(
        reaction_id="tamoxifen_biotransformation",
        substrate="tamoxifen_n_oxide",
        product="tamoxifen",
        enzyme_name="bacterial_reductase",
        organism="Eubacterium_lentum",
        km_um=80.0, vmax_relative=0.3,
        is_activation=True,
        description="Bacterial reduction of tamoxifen N-oxide → active metabolite",
    ),
]


# ============================================================================
# Microbiome Community Model
# ============================================================================

@dataclass
class MicrobialSpecies:
    """A key bacterial species in the drug-metabolizing gut community."""

    name: str
    abundance: float  # relative abundance 0-1
    reactions: list[str]  # reaction_ids this species performs
    growth_rate_h: float = 0.1  # doubling time ~7h
    scfa_production: float = 0.0  # short-chain fatty acid output (mM/h)
    lactate_production: float = 0.0
    ammonia_production: float = 0.0  # toxic if high


@dataclass
class MicrobiomeState:
    """Current state of the gut microbiome compartment."""

    total_biomass: float = 1.0          # relative to baseline
    diversity_index: float = 1.0        # Shannon index (1=normal, 0=depleted)
    scfa_total_mM: float = 80.0         # total short-chain fatty acids
    lactate_mM: float = 2.0
    ammonia_mM: float = 0.5
    tma_mM: float = 0.1
    ph_gut: float = 6.8                 # colonic pH
    beta_glucuronidase_activity: float = 1.0  # relative activity
    bile_salt_hydrolase_activity: float = 1.0
    inflammation_score: float = 0.0     # 0=none, 1=severe
    permeability: float = 0.02          # gut permeability (0=impermeable, 1=fully leaky)


@dataclass
class MicrobiomeDrugEffect:
    """Effect of microbiome on a specific drug."""

    drug_name: str
    bioavailability_modifier: float = 1.0  # multiplicative
    toxicity_modifier: float = 1.0         # multiplicative on GI toxicity
    active_metabolite_generated: str = ""
    amount_activated_umol: float = 0.0
    amount_inactivated_umol: float = 0.0
    reaction_ids: list[str] = field(default_factory=list)


# ============================================================================
# Gut Microbiome GEM (Genome-Scale Metabolic Model)
# ============================================================================

class MicrobiomeCompartment:
    """Gut microbiome metabolic compartment.

    Models the gut lumen as a metabolic reactor with:
    - Bacterial biomass (total + species composition)
    - Drug metabolism by microbial enzymes
    - Metabolite production (SCFA, ammonia, TMA, lactate)
    - Exchange with liver via portal vein
    - Immune modulation (gut permeability → bacterial translocation)
    """

    def __init__(self, healthy_composition: bool = True) -> None:
        self.state = MicrobiomeState()
        self._reactions = {r.reaction_id: r for r in MICROBIAL_REACTIONS}
        self._drug_concs: dict[str, float] = {}
        self._drug_effects: dict[str, MicrobiomeDrugEffect] = {}

        # Default healthy species composition
        self._species: dict[str, MicrobialSpecies] = {
            "E._coli": MicrobialSpecies(
                name="E._coli", abundance=0.15,
                reactions=["irinotecan_reactivation", "levodopa_decarboxylation",
                           "mycophenolate_reactivation"],
                growth_rate_h=0.15, scfa_production=0.5,
            ),
            "Clostridium_sp.": MicrobialSpecies(
                name="Clostridium_sp.", abundance=0.20,
                reactions=["sulfasalazine_azo", "nsaid_reactivation",
                           "bile_acid_7alpha_dehydroxylation", "tma_formation",
                           "ranitidine_n_reduction", "warfarin_reductions",
                           "prednisone_reactivation"],
                growth_rate_h=0.08, scfa_production=1.0,
            ),
            "Lactobacillus_sp.": MicrobialSpecies(
                name="Lactobacillus_sp.", abundance=0.25,
                reactions=["bile_salt_deconjugation", "enalapril_hydrolysis"],
                growth_rate_h=0.12, scfa_production=1.5, lactate_production=0.8,
            ),
            "Bacteroides_sp.": MicrobialSpecies(
                name="Bacteroides_sp.", abundance=0.20,
                reactions=["prednisone_reactivation"],
                growth_rate_h=0.06, scfa_production=1.2,
            ),
            "Bifidobacterium_sp.": MicrobialSpecies(
                name="Bifidobacterium_sp.", abundance=0.10,
                reactions=[],
                growth_rate_h=0.08, scfa_production=0.8,
            ),
            "Eubacterium_lentum": MicrobialSpecies(
                name="Eubacterium_lentum", abundance=0.05,
                reactions=["digoxin_reduction", "tamoxifen_biotransformation"],
                growth_rate_h=0.05, scfa_production=0.3,
            ),
            "Enterococcus_faecium": MicrobialSpecies(
                name="Enterococcus_faecium", abundance=0.05,
                reactions=["levodopa_decarboxylation"],
                growth_rate_h=0.10, lactate_production=0.5,
            ),
            # Phase 5: Additional species
            "Citrobacter_freundii": MicrobialSpecies(
                name="Citrobacter_freundii", abundance=0.02,
                reactions=["sulfapyridine_hydroxylation"],
                growth_rate_h=0.10, scfa_production=0.3,
            ),
            "Klebsiella_pneumoniae": MicrobialSpecies(
                name="Klebsiella_pneumoniae", abundance=0.02,
                reactions=["digoxin_inactivation2"],
                growth_rate_h=0.12, scfa_production=0.2,
            ),
            "Pseudomonas_sp.": MicrobialSpecies(
                name="Pseudomonas_sp.", abundance=0.01,
                reactions=["codeine_3dmg", "fluorouracil_degradation"],
                growth_rate_h=0.15, scfa_production=0.1,
            ),
        }

        if not healthy_composition:
            # Dysbiotic: low diversity, high proteobacteria
            self._species["E._coli"].abundance = 0.35
            self._species["Clostridium_sp."].abundance = 0.30
            self._species["Lactobacillus_sp."].abundance = 0.10
            self._species["Bacteroides_sp."].abundance = 0.10
            self._species["Bifidobacterium_sp."].abundance = 0.05
            self.state.diversity_index = 0.6
            self.state.beta_glucuronidase_activity = 1.8

    def apply_antibiotic(self, antibiotic_name: str, dose_mg: float = 500.0) -> None:
        """Apply antibiotic effect on microbiome composition.

        Broad-spectrum antibiotics reduce diversity and shift composition;
        narrow-spectrum antibiotics target specific species.
        """
        name_lower = antibiotic_name.lower()
        if any(k in name_lower for k in ["ampicillin", "amoxicillin", "penicillin"]):
            # Beta-lactams: kill gram-positive
            for sp_name in ["Lactobacillus_sp.", "Bifidobacterium_sp.",
                            "Enterococcus_faecium", "Eubacterium_lentum"]:
                if sp_name in self._species:
                    kill_frac = min(0.8, dose_mg / 1000.0)
                    self._species[sp_name].abundance *= (1.0 - kill_frac)
            self._species["E._coli"].abundance *= 1.3  # gram-negative expand
        elif any(k in name_lower for k in ["ciprofloxacin", "levofloxacin", "moxifloxacin"]):
            # Fluoroquinolones: broad-spectrum, kills aerobic
            for sp_name in ["E._coli", "Klebsiella_pneumoniae", "Pseudomonas_sp.",
                            "Citrobacter_freundii"]:
                if sp_name in self._species:
                    kill_frac = min(0.7, dose_mg / 500.0)
                    self._species[sp_name].abundance *= (1.0 - kill_frac)
        elif any(k in name_lower for k in ["metronidazole"]):
            # Nitroimidazoles: anaerobic bacteria
            for sp_name in ["Clostridium_sp.", "Bacteroides_sp.", "Eubacterium_lentum"]:
                if sp_name in self._species:
                    kill_frac = min(0.8, dose_mg / 500.0)
                    self._species[sp_name].abundance *= (1.0 - kill_frac)
        elif any(k in name_lower for k in ["vancomycin"]):
            # Vancomycin: gram-positive, including enterococci
            for sp_name in ["Enterococcus_faecium", "Lactobacillus_sp.",
                            "Bifidobacterium_sp."]:
                if sp_name in self._species:
                    kill_frac = min(0.7, dose_mg / 1000.0)
                    self._species[sp_name].abundance *= (1.0 - kill_frac)
        else:
            # Generic broad-spectrum: reduce all by 30%
            for sp in self._species.values():
                kill_frac = min(0.5, dose_mg / 2000.0)
                sp.abundance *= (1.0 - kill_frac)

        # Update diversity after antibiotic perturbation
        abundances = [sp.abundance for sp in self._species.values()]
        total = sum(abundances) or 1.0
        self.state.diversity_index = min(1.0, -sum(
            (a / total) * max(1e-10, (a / total))
            for a in abundances if a > 0
        ) / max(1e-10, self.state.diversity_index))

    def induce_dysbiosis(self, severity: float = 0.5) -> None:
        """Induce dysbiosis by shifting species composition.

        Args:
            severity: 0.0 (healthy) to 1.0 (severe dysbiosis)
        """
        # Reduce beneficial anaerobes
        for sp_name in ["Lactobacillus_sp.", "Bifidobacterium_sp.", "Bacteroides_sp."]:
            if sp_name in self._species:
                self._species[sp_name].abundance *= (1.0 - 0.5 * severity)

        # Expand proteobacteria
        for sp_name in ["E._coli", "Klebsiella_pneumoniae", "Pseudomonas_sp."]:
            if sp_name in self._species:
                self._species[sp_name].abundance *= (1.0 + 0.8 * severity)

        # Update state
        self.state.diversity_index = max(0.1, 1.0 - 0.7 * severity)
        self.state.beta_glucuronidase_activity = 0.5 + 2.5 * severity
        self.state.permeability = 0.02 + 0.15 * severity
        self.state.inflammation_score = 0.3 * severity

    def restore_microbiome(self, probiotic: str = "Lactobacillus") -> None:
        """Restore microbiome after antibiotic/dysbiosis with probiotics."""
        name_lower = probiotic.lower()
        if "lacto" in name_lower:
            if "Lactobacillus_sp." in self._species:
                self._species["Lactobacillus_sp."].abundance *= 1.5
        elif "bifido" in name_lower:
            if "Bifidobacterium_sp." in self._species:
                self._species["Bifidobacterium_sp."].abundance *= 1.5
        elif "sacc" in name_lower:
            # Saccharomyces boulardii: yeast probiotic
            pass  # doesn't affect bacterial composition directly

        # Recalculate diversity
        self.state.diversity_index = min(1.0, self.state.diversity_index + 0.2)

    def set_drug_concentration(self, drug_name: str, conc_um: float) -> None:
        """Set luminal drug concentration for microbial metabolism."""
        self._drug_concs[_canonical_name(drug_name)] = conc_um

    def step(self, dt_h: float) -> dict[str, MicrobiomeDrugEffect]:
        """Advance microbiome state by one time step.

        Returns drug effects for each drug present in the gut lumen.
        """
        self._drug_effects.clear()

        # Process each drug
        for drug_name, drug_conc in list(self._drug_concs.items()):
            if drug_conc <= 0:
                continue

            effect = MicrobiomeDrugEffect(drug_name=drug_name)

            for rxn_id, rxn in self._reactions.items():
                if rxn.substrate == drug_name or rxn.product == drug_name:
                    # Get relevant species activity
                    total_activity = 0.0
                    for sp in self._species.values():
                        if rxn_id in sp.reactions:
                            total_activity += sp.abundance * sp.growth_rate_h

                    # Michaelis-Menten kinetics
                    substrate_conc = drug_conc if rxn.substrate == drug_name else 0.0
                    rxn_rate = rxn.vmax_relative * total_activity * substrate_conc / (
                        rxn.km_um + substrate_conc + 1e-10
                    )
                    rxn_rate *= self.state.beta_glucuronidase_activity if "glucuronidase" in rxn.enzyme_name else 1.0

                    amount = rxn_rate * dt_h  # µmol produced/metabolized

                    if rxn.substrate == drug_name:
                        # Drug consumed by this reaction
                        self._drug_concs[drug_name] = max(0.0, drug_conc - amount)
                        effect.amount_inactivated_umol += amount if not rxn.is_activation else 0.0
                        effect.reaction_ids.append(rxn_id)

                    if rxn.product != drug_name:
                        # Drug is substrate → product is new metabolite
                        if rxn.is_activation:
                            effect.amount_activated_umol += amount
                            effect.active_metabolite_generated = rxn.product
                            effect.bioavailability_modifier *= (1.0 + amount * 0.1)
                        else:
                            # Reactivation of toxic metabolite → increase GI toxicity
                            effect.toxicity_modifier *= (1.0 + amount * 0.2)
                            effect.active_metabolite_generated = rxn.product
                            effect.reaction_ids.append(rxn_id)

            self._drug_effects[drug_name] = effect

        # Update microbiome state
        self._update_state(dt_h)

        return dict(self._drug_effects)

    def _update_state(self, dt_h: float) -> None:
        """Update microbiome community state."""
        # SCFA production (proportional to total biomass and anaerobic fermentation)
        total_scfa_rate = sum(
            sp.abundance * sp.scfa_production for sp in self._species.values()
        )
        self.state.scfa_total_mM += (total_scfa_rate - 0.5 * self.state.scfa_total_mM) * dt_h
        self.state.scfa_total_mM = max(20.0, min(150.0, self.state.scfa_total_mM))

        # Lactate
        total_lactate = sum(
            sp.abundance * sp.lactate_production for sp in self._species.values()
        )
        self.state.lactate_mM += (total_lactate - 0.3 * self.state.lactate_mM) * dt_h
        self.state.lactate_mM = max(0.5, min(10.0, self.state.lactate_mM))

        # Ammonia (toxic at high levels)
        total_ammonia = sum(
            sp.abundance * sp.ammonia_production for sp in self._species.values()
        )
        self.state.ammonia_mM += (total_ammonia - 0.2 * self.state.ammonia_mM) * dt_h
        self.state.ammonia_mM = max(0.1, min(5.0, self.state.ammonia_mM))

        # TMA (from choline)
        self.state.tma_mM = 0.1 * self.state.scfa_total_mM / 80.0

        # Gut pH (inversely related to SCFA)
        self.state.ph_gut = 7.0 - 0.003 * (self.state.scfa_total_mM - 80.0)
        self.state.ph_gut = max(5.5, min(7.5, self.state.ph_gut))

        # β-glucuronidase activity scales with E. coli and Clostridium abundance
        ecoli_frac = self._species.get("E._coli", MicrobialSpecies("", 0, [])).abundance
        clost_frac = self._species.get("Clostridium_sp.", MicrobialSpecies("", 0, [])).abundance
        self.state.beta_glucuronidase_activity = 0.5 + 2.0 * (ecoli_frac + clost_frac)

        # Bile salt hydrolase scales with Lactobacillus
        lacto_frac = self._species.get("Lactobacillus_sp.", MicrobialSpecies("", 0, [])).abundance
        self.state.bile_salt_hydrolase_activity = 0.3 + 3.0 * lacto_frac

    def get_portal_fluxes(self) -> dict[str, float]:
        """Compute metabolite fluxes from gut to liver via portal vein.

        Returns fluxes in µmol/h for portal vein delivery.
        """
        return {
            "scfa": self.state.scfa_total_mM * 0.5,      # portal SCFA delivery
            "ammonia": self.state.ammonia_mM * 0.3,       # hepatic ammonia load
            "tma": self.state.tma_mM * 0.8,               # TMA → hepatic FMO3 → TMAO
            "bile_acids": self.state.bile_salt_hydrolase_activity * 2.0,
            "lactate": self.state.lactate_mM * 0.2,
            "gut_permeability": self.state.permeability,  # endotoxin translocation
            "inflammation": self.state.inflammation_score,
        }

    def get_overall_drug_effect(self, drug_name: str) -> MicrobiomeDrugEffect:
        """Get the aggregate microbiome effect on a drug."""
        return self._drug_effects.get(
            _canonical_name(drug_name),
            MicrobiomeDrugEffect(drug_name=drug_name),
        )
