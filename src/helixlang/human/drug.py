"""Drug molecular specification for human pathology simulation (doc/27).

Defines the chemical identity of a therapeutic agent (:class:`DrugMolecule`)
and its complete dosing regimen with ADME parameters (:class:`Drug`), plus
SMILES-based property computation and a library of pre-defined drug profiles
anchored to published pharmacology.

Supported drug classes (doc/27 section 6.2): ``small_molecule``, ``metal_complex``,
``biologic``, and ``oligonucleotide``.

SMILES parsing uses RDKit when the optional ``human`` extra is installed
(``pip install helixlang[human]``); without RDKit the module degrades
gracefully to literature values only.  PK/PD simulation does not require
RDKit — only structure-based property prediction is affected.

References:
- Rowland M & Tozer TN. Clinical Pharmacokinetics and Pharmacodynamics, 5th ed. 2020.
- Genzyme. Cerezyme (imiglucerase) prescribing information, 2022.
- Rang HP et al. Rang & Dale's Pharmacology, 9th ed. 2019.
- Graham GG et al. Clinical pharmacokinetics of metformin. Clin Pharmacokinet 2011;50:81-98.
- Kelland L. The resurgence of platinum-based cancer chemotherapy. Nat Rev Cancer 2007;7:573-584.
- Jordan VC. Tamoxifen: a most unlikely pioneering medicine. Nat Rev Drug Discov 2003;2:205-213.
- Druker BJ et al. Five-year follow-up of patients receiving imatinib. NEJM 2006;355:2408-2417.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

SMALL_MOLECULE = "small_molecule"
METAL_COMPLEX = "metal_complex"
BIOLOGIC = "biologic"
OLIGONUCLEOTIDE = "oligonucleotide"

VALID_DRUG_TYPES = (SMALL_MOLECULE, METAL_COMPLEX, BIOLOGIC, OLIGONUCLEOTIDE)

ORAL = "oral"
IV = "iv"
IV_INFUSION = "iv_infusion"
SUBCUTANEOUS = "subcutaneous"
INTRAMUSCULAR = "intramuscular"
INTRATHECAL = "intrathecal"

VALID_ROUTES = (ORAL, IV, IV_INFUSION, SUBCUTANEOUS, INTRAMUSCULAR, INTRATHECAL)

LN_2 = math.log(2.0)

REFERENCE_BODY_WEIGHT_KG = 70.0
REFERENCE_BSA_M2 = 1.7


@dataclass
class DrugMolecule:
    """Chemical identity of a drug (doc/27 section 6.3)."""

    name: str
    drug_type: str = SMALL_MOLECULE
    smiles: str = ""
    molecular_weight_da: float = 0.0
    formula: str = ""
    amino_acid_sequence: str = ""
    metal_ion: str = ""
    coordination_geometry: str = ""
    target_protein: str = ""
    binding_affinity_kd_um: float = 0.0
    selectivity_index: float = 1.0
    protein_binding_fraction: float = 0.0
    log_p: float = 0.0
    solubility_mg_per_ml: float = 0.0


@dataclass
class Drug:
    """Complete drug specification with dosing regimen and ADME parameters.

    Concentration-derived methods use one-compartment first-order absorption
    and elimination approximations with linear kinetics (Rowland & Tozer 2020).
    """

    molecule: DrugMolecule
    dose_mg: float = 0.0
    dosing_interval_h: float = 24.0
    route: str = ORAL
    duration_days: float = 30.0
    bioavailability: float = 1.0
    absorption_rate_h: float = 1.0
    volume_distribution_l: float = 50.0
    clearance_ml_per_min: float = 100.0
    half_life_h: float = 6.0
    hepatic_extraction_ratio: float = 0.0
    renal_fraction: float = 0.0
    cyp_metabolism: dict[str, float] = field(default_factory=dict)
    #: Transporter substrates → fraction of uptake/efflux affected (0–1)
    transporter_affected: dict[str, float] = field(default_factory=dict)
    #: Non-CYP phase-II metabolism fractions (UGT1A1, TPMT, DPYD, NAT2, etc.)
    non_cyp_metabolism: dict[str, float] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of specification problems (empty when valid)."""
        problems: list[str] = []
        if self.route not in VALID_ROUTES:
            problems.append(f"unknown route: {self.route!r}")
        if not 0.0 <= self.bioavailability <= 1.0:
            problems.append(f"bioavailability out of [0,1]: {self.bioavailability}")
        if not 0.0 <= self.renal_fraction <= 1.0:
            problems.append(f"renal_fraction out of [0,1]: {self.renal_fraction}")
        eh = self.hepatic_extraction_ratio
        if not 0.0 <= eh <= 1.0:
            problems.append(f"hepatic_extraction_ratio out of [0,1]: {eh}")
        cyp_total = sum(self.cyp_metabolism.values())
        if self.cyp_metabolism and not math.isclose(cyp_total, 1.0, abs_tol=0.05):
            problems.append(f"cyp_metabolism fractions sum to {cyp_total:.2f}, expected ~1.0")
        if self.half_life_h <= 0.0:
            problems.append(f"half_life_h must be positive: {self.half_life_h}")
        return problems

    def elimination_rate_constant(self) -> float:
        """First-order elimination rate constant k (h^-1) = ln(2)/t1/2."""
        if self.half_life_h <= 0.0:
            return 0.0
        return LN_2 / self.half_life_h

    def total_administered_mg(self) -> float:
        """Cumulative dose over the treatment duration (mg)."""
        if self.dosing_interval_h <= 0.0:
            return 0.0
        n_doses = max(1, int(round(self.duration_days * 24.0 / self.dosing_interval_h)))
        return n_doses * self.dose_mg

    def accumulation_ratio(self) -> float:
        """Accumulation ratio at steady state for repeated dosing."""
        k = self.elimination_rate_constant()
        tau = self.dosing_interval_h
        if tau <= 0.0 or k <= 0.0:
            return 1.0
        return 1.0 / (1.0 - math.exp(-k * tau))

    def steady_state_average_mg_per_l(self) -> float:
        """Average concentration at steady state, Cavg = F*Dose/(CL*tau)."""
        cl_l_per_h = self.clearance_ml_per_min * 0.06
        tau = self.dosing_interval_h
        if cl_l_per_h <= 0.0 or tau <= 0.0:
            return 0.0
        return self.bioavailability * self.dose_mg / (cl_l_per_h * tau)

    def steady_state_peak_mg_per_l(self) -> float:
        """Approximate peak concentration at steady state (mg/L)."""
        vd = max(self.volume_distribution_l, 1e-9)
        k = self.elimination_rate_constant()
        ka = max(self.absorption_rate_h, 1e-9)
        t_peak = math.log(ka / k) / (ka - k) if abs(ka - k) > 1e-12 else 1.0 / ka
        acc = self.accumulation_ratio()
        if self.route in (IV, IV_INFUSION):
            single = self.bioavailability * self.dose_mg / vd * math.exp(-k * t_peak)
        else:
            base = self.bioavailability * self.dose_mg * ka / (vd * (ka - k))
            single = base * (math.exp(-k * t_peak) - math.exp(-ka * t_peak))
        return single * acc


def _formula_counts_from_mol(mol: Any) -> dict[str, int]:
    """Count element occurrences across all atoms of an RDKit mol.

    Uses ``CalcMolFormula`` when available to include implicit hydrogens;
    falls back to explicit-atom iteration.
    """
    try:
        from rdkit.Chem import rdMolDescriptors
        formula = rdMolDescriptors.CalcMolFormula(mol)
        counts: dict[str, int] = {}
        import re as _re
        for match in _re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
            elem = match.group(1)
            n = int(match.group(2)) if match.group(2) else 1
            counts[elem] = counts.get(elem, 0) + n
        return counts
    except Exception:
        counts_fallback: dict[str, int] = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts_fallback[symbol] = counts_fallback.get(symbol, 0) + 1
        return counts_fallback


def _hill_formula(counts: dict[str, int]) -> str:
    """Format element counts as a molecular formula in Hill order."""
    parts: list[str] = []

    def emit(symbol: str) -> None:
        n = counts[symbol]
        parts.append(symbol + (str(n) if n > 1 else ""))

    for leading in ("C", "H"):
        if counts.get(leading):
            emit(leading)
    for symbol in sorted(k for k in counts if k not in ("C", "H")):
        emit(symbol)
    return "".join(parts)


def _refresh_computed_properties(molecule: DrugMolecule) -> DrugMolecule:
    """Overwrite MW/formula/LogP with RDKit-computed values when possible.

    Literature values are kept as-is when RDKit is unavailable or the SMILES
    cannot be parsed (graceful degradation, doc/27 section 6.4).
    """
    if not _HAS_RDKIT or not molecule.smiles:
        return molecule
    mol = Chem.MolFromSmiles(molecule.smiles)
    if mol is None:
        return molecule
    molecule.molecular_weight_da = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    molecule.log_p = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    molecule.formula = _hill_formula(_formula_counts_from_mol(mol))
    return molecule


def parse_drug_smiles(
    smiles: str,
    name: str = "",
    drug_type: str = SMALL_MOLECULE,
) -> DrugMolecule:
    """Parse a SMILES string into a :class:`DrugMolecule`.

    With RDKit available, computes molecular weight (Descriptors.MolWt),
    lipophilicity (Descriptors.MolLogP), and the Hill-order formula from the
    parsed structure.  Without RDKit — or when the SMILES fails to parse —
    returns a molecule carrying only the supplied metadata and zero-valued
    computed properties (doc/27 section 6.4 graceful degradation).
    """
    if not smiles:
        return DrugMolecule(name=name, drug_type=drug_type)
    if _HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return DrugMolecule(
                name=name,
                drug_type=drug_type,
                smiles=smiles,
                molecular_weight_da=Descriptors.MolWt(mol),  # type: ignore[attr-defined]
                formula=_hill_formula(_formula_counts_from_mol(mol)),
                log_p=Descriptors.MolLogP(mol),  # type: ignore[attr-defined]
            )
    return DrugMolecule(name=name, drug_type=drug_type, smiles=smiles)


@dataclass(frozen=True)
class _Spec:
    """Internal template: (lookup key, molecule kwargs, regimen kwargs)."""

    key: str
    molecule: dict
    regimen: dict


_PREDEFINED_SPECS: tuple[_Spec, ...] = (
    _Spec(
        key="IMIGLUCERASE",
        molecule=dict(
            name="imiglucerase",
            drug_type=BIOLOGIC,
            molecular_weight_da=60000.0,
            target_protein="GBA1",
            binding_affinity_kd_um=0.01,
            solubility_mg_per_ml=10.0,
        ),
        regimen=dict(
            dose_mg=60.0 * REFERENCE_BODY_WEIGHT_KG / 40.0,
            dosing_interval_h=336.0,
            route=IV_INFUSION,
            duration_days=90.0,
            bioavailability=1.0,
            absorption_rate_h=1.0,
            volume_distribution_l=8.0,
            clearance_ml_per_min=9.2,
            half_life_h=10.0,
            hepatic_extraction_ratio=0.9,
            renal_fraction=0.0,
        ),
    ),
    _Spec(
        key="IBUPROFEN",
        molecule=dict(
            name="ibuprofen",
            drug_type=SMALL_MOLECULE,
            smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            molecular_weight_da=206.3,
            formula="C13H18O2",
            target_protein="COX1/COX2",
            binding_affinity_kd_um=13.0,
            protein_binding_fraction=0.99,
            log_p=3.5,
            solubility_mg_per_ml=0.021,
        ),
        regimen=dict(
            dose_mg=400.0,
            dosing_interval_h=8.0,
            route=ORAL,
            duration_days=7.0,
            bioavailability=1.0,
            absorption_rate_h=2.5,
            volume_distribution_l=10.0,
            clearance_ml_per_min=58.0,
            half_life_h=2.0,
            hepatic_extraction_ratio=0.8,
            renal_fraction=0.9,
            cyp_metabolism={"CYP2C9": 0.85, "CYP2C8": 0.10, "CYP3A4": 0.05},
        ),
    ),
    _Spec(
        key="METFORMIN",
        molecule=dict(
            name="metformin",
            drug_type=SMALL_MOLECULE,
            smiles="CN(C)C(=N)NC(=N)N",
            molecular_weight_da=165.6,
            formula="C4H11N5",
            target_protein="complex_I",
            binding_affinity_kd_um=20.0,
            log_p=-1.2,
            solubility_mg_per_ml=300.0,
        ),
        regimen=dict(
            dose_mg=500.0,
            dosing_interval_h=12.0,
            route=ORAL,
            duration_days=90.0,
            bioavailability=0.55,
            absorption_rate_h=0.7,
            volume_distribution_l=160.0,
            clearance_ml_per_min=463.0,
            half_life_h=4.0,
            renal_fraction=1.0,
        ),
    ),
    _Spec(
        key="CISPLATIN",
        molecule=dict(
            name="cisplatin",
            drug_type=METAL_COMPLEX,
            smiles="[Pt+2].([Cl-].[Cl-]).([NH3].[NH3])",
            molecular_weight_da=300.1,
            formula="Cl2H6N2Pt",
            metal_ion="Pt",
            coordination_geometry="square_planar",
            target_protein="DNA",
            protein_binding_fraction=0.9,
            log_p=-2.3,
            solubility_mg_per_ml=1.0,
        ),
        regimen=dict(
            dose_mg=50.0 * REFERENCE_BSA_M2,
            dosing_interval_h=504.0,
            route=IV_INFUSION,
            duration_days=126.0,
            bioavailability=1.0,
            absorption_rate_h=1.0,
            volume_distribution_l=280.0,
            clearance_ml_per_min=90.0,
            half_life_h=36.0,
            hepatic_extraction_ratio=0.1,
            renal_fraction=0.9,
        ),
    ),
    _Spec(
        key="TAMOXIFEN",
        molecule=dict(
            name="tamoxifen",
            drug_type=SMALL_MOLECULE,
            smiles="CCC(c1ccc2c(c1)OCO2)",
            molecular_weight_da=371.5,
            formula="C26H29NO",
            target_protein="ESR1",
            binding_affinity_kd_um=0.01,
            selectivity_index=50.0,
            protein_binding_fraction=0.99,
            log_p=6.3,
            solubility_mg_per_ml=0.001,
        ),
        regimen=dict(
            dose_mg=20.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.9,
            absorption_rate_h=0.5,
            volume_distribution_l=3500.0,
            clearance_ml_per_min=281.0,
            half_life_h=144.0,
            hepatic_extraction_ratio=0.3,
            renal_fraction=0.01,
            cyp_metabolism={"CYP3A4": 0.65, "CYP2D6": 0.30, "CYP2C9": 0.05},
        ),
    ),
    _Spec(
        key="IMATINIB",
        molecule=dict(
            name="imatinib",
            drug_type=SMALL_MOLECULE,
            smiles=("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"),
            molecular_weight_da=493.6,
            formula="C29H31N7O",
            target_protein="BCR_ABL",
            binding_affinity_kd_um=0.025,
            selectivity_index=100.0,
            protein_binding_fraction=0.95,
            log_p=3.5,
            solubility_mg_per_ml=0.05,
        ),
        regimen=dict(
            dose_mg=400.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.98,
            absorption_rate_h=0.8,
            volume_distribution_l=230.0,
            clearance_ml_per_min=148.0,
            half_life_h=18.0,
            hepatic_extraction_ratio=0.25,
            renal_fraction=0.05,
            cyp_metabolism={"CYP3A4": 0.75, "CYP2C8": 0.15, "CYP2D6": 0.10},
        ),
    ),
    # ---- Expanded drug library (Phase 3) ----
    _Spec(
        key="WARFARIN",
        molecule=dict(
            name="warfarin",
            drug_type=SMALL_MOLECULE,
            smiles="CC(=O)Cc1ccccc1C(=O)O",
            molecular_weight_da=308.3,
            formula="C19H16O4",
            target_protein="VKORC1",
            binding_affinity_kd_um=0.001,
            selectivity_index=500.0,
            protein_binding_fraction=0.99,
            log_p=3.1,
            solubility_mg_per_ml=0.017,
        ),
        regimen=dict(
            dose_mg=5.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.95,
            absorption_rate_h=1.5,
            volume_distribution_l=8.0,
            clearance_ml_per_min=0.045,
            half_life_h=40.0,
            hepatic_extraction_ratio=0.0,
            renal_fraction=0.0,
            cyp_metabolism={"CYP2C9": 0.80, "CYP3A4": 0.10, "CYP1A2": 0.05, "CYP2C19": 0.05},
        ),
    ),
    _Spec(
        key="CLOPIDOGREL",
        molecule=dict(
            name="clopidogrel",
            drug_type=SMALL_MOLECULE,
            smiles="COC(=O)C(c1ccccc1Cl)C1CS[C@@H](N1)Cc1ccccc1Cl",
            molecular_weight_da=321.8,
            formula="C16H16ClNO2S",
            target_protein="P2RY12",
            binding_affinity_kd_um=0.01,
            selectivity_index=100.0,
            protein_binding_fraction=0.98,
            log_p=3.5,
            solubility_mg_per_ml=0.005,
        ),
        regimen=dict(
            dose_mg=75.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.50,
            absorption_rate_h=1.0,
            volume_distribution_l=300.0,
            clearance_ml_per_min=116.0,
            half_life_h=6.0,
            hepatic_extraction_ratio=0.5,
            renal_fraction=0.5,
            cyp_metabolism={"CYP2C19": 0.45, "CYP3A4": 0.30, "CYP2B6": 0.15, "CYP1A2": 0.10},
        ),
    ),
    _Spec(
        key="ATORVASTATIN",
        molecule=dict(
            name="atorvastatin",
            drug_type=SMALL_MOLECULE,
            smiles="CC(C)c1c(C(=O)Nc2ccccc2)c(c(c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O)c1ccccc1",
            molecular_weight_da=558.6,
            formula="C33H35FN2O5",
            target_protein="HMGCR",
            binding_affinity_kd_um=0.008,
            selectivity_index=200.0,
            protein_binding_fraction=0.98,
            log_p=6.4,
            solubility_mg_per_ml=0.001,
        ),
        regimen=dict(
            dose_mg=20.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.14,
            absorption_rate_h=0.8,
            volume_distribution_l=381.0,
            clearance_ml_per_min=625.0,
            half_life_h=14.0,
            hepatic_extraction_ratio=0.5,
            renal_fraction=0.02,
            cyp_metabolism={"CYP3A4": 0.90, "CYP3A5": 0.05, "CYP2C8": 0.05},
            transporter_affected={"SLCO1B1": 0.8, "ABCB1": 0.2},
        ),
    ),
    _Spec(
        key="AMLODIPINE",
        molecule=dict(
            name="amlodipine",
            drug_type=SMALL_MOLECULE,
            smiles="CCOC(=O)C1=CNC(=C(C1c1ccc(Cl)cc1Cl)C(=O)OC)C",
            molecular_weight_da=408.9,
            formula="C20H25ClN2O5",
            target_protein="CACNA1C",
            binding_affinity_kd_um=0.02,
            selectivity_index=100.0,
            protein_binding_fraction=0.93,
            log_p=3.0,
            solubility_mg_per_ml=0.02,
        ),
        regimen=dict(
            dose_mg=5.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.65,
            absorption_rate_h=0.6,
            volume_distribution_l=21.0,
            clearance_ml_per_min=7.0,
            half_life_h=40.0,
            hepatic_extraction_ratio=0.6,
            renal_fraction=0.1,
            cyp_metabolism={"CYP3A4": 0.90, "CYP3A5": 0.10},
        ),
    ),
    _Spec(
        key="LOSARTAN",
        molecule=dict(
            name="losartan",
            drug_type=SMALL_MOLECULE,
            smiles="CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)CO",
            molecular_weight_da=422.9,
            formula="C22H23ClN6O",
            target_protein="AGTR1",
            binding_affinity_kd_um=0.02,
            selectivity_index=1000.0,
            protein_binding_fraction=0.98,
            log_p=4.0,
            solubility_mg_per_ml=0.1,
        ),
        regimen=dict(
            dose_mg=50.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.33,
            absorption_rate_h=1.0,
            volume_distribution_l=34.0,
            clearance_ml_per_min=57.0,
            half_life_h=6.0,
            hepatic_extraction_ratio=0.15,
            renal_fraction=0.6,
            cyp_metabolism={"CYP2C9": 0.60, "CYP3A4": 0.25, "CYP2C19": 0.15},
        ),
    ),
    _Spec(
        key="PANTOPRAZOLE",
        molecule=dict(
            name="pantoprazole",
            drug_type=SMALL_MOLECULE,
            smiles="COc1ccc2nc(CS(=O)c3ncc(CF)cc3C)cc2n1",
            molecular_weight_da=383.4,
            formula="C16H15F2N3O4S",
            target_protein="ATP4A",
            binding_affinity_kd_um=0.01,
            selectivity_index=500.0,
            protein_binding_fraction=0.98,
            log_p=2.1,
            solubility_mg_per_ml=0.5,
        ),
        regimen=dict(
            dose_mg=40.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=14.0,
            bioavailability=0.77,
            absorption_rate_h=1.0,
            volume_distribution_l=11.0,
            clearance_ml_per_min=7.6,
            half_life_h=1.0,
            hepatic_extraction_ratio=0.2,
            renal_fraction=0.8,
            cyp_metabolism={"CYP2C19": 0.80, "CYP3A4": 0.20},
        ),
    ),
    _Spec(
        key="SERTRALINE",
        molecule=dict(
            name="sertraline",
            drug_type=SMALL_MOLECULE,
            smiles="CC1=CC[C@@H]2CC[C@@H](CC2=C1)NCC1=C(Cl)C=CC=C1Cl",
            molecular_weight_da=306.2,
            formula="C17H17Cl2N",
            target_protein="SLC6A4",
            binding_affinity_kd_um=0.002,
            selectivity_index=100.0,
            protein_binding_fraction=0.98,
            log_p=5.1,
            solubility_mg_per_ml=0.005,
        ),
        regimen=dict(
            dose_mg=50.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=60.0,
            bioavailability=0.88,
            absorption_rate_h=0.5,
            volume_distribution_l=20.0,
            clearance_ml_per_min=1.5,
            half_life_h=26.0,
            hepatic_extraction_ratio=0.15,
            renal_fraction=0.4,
            cyp_metabolism={"CYP2B6": 0.40, "CYP2C19": 0.25, "CYP3A4": 0.20, "CYP2C9": 0.15},
        ),
    ),
    _Spec(
        key="FLUOXETINE",
        molecule=dict(
            name="fluoxetine",
            drug_type=SMALL_MOLECULE,
            smiles="CNCCC(c1ccccc1)OCc1ccc(C(F)(F)F)cc1",
            molecular_weight_da=309.3,
            formula="C17H18F3NO",
            target_protein="SLC6A4",
            binding_affinity_kd_um=0.001,
            selectivity_index=200.0,
            protein_binding_fraction=0.945,
            log_p=4.05,
            solubility_mg_per_ml=0.01,
        ),
        regimen=dict(
            dose_mg=20.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=60.0,
            bioavailability=0.72,
            absorption_rate_h=0.5,
            volume_distribution_l=45.0,
            clearance_ml_per_min=0.63,
            half_life_h=72.0,
            hepatic_extraction_ratio=0.1,
            renal_fraction=0.8,
            cyp_metabolism={"CYP2D6": 0.70, "CYP2C9": 0.15, "CYP3A4": 0.10, "CYP2C19": 0.05},
        ),
    ),
    _Spec(
        key="CITALOPRAM",
        molecule=dict(
            name="citalopram",
            drug_type=SMALL_MOLECULE,
            smiles="N#CCc1ccc(-c2c(c(CCN(C)C)c(OC)c2)C(F)(F)F)cc1",
            molecular_weight_da=324.4,
            formula="C20H21FN2O",
            target_protein="SLC6A4",
            binding_affinity_kd_um=0.001,
            selectivity_index=500.0,
            protein_binding_fraction=0.80,
            log_p=3.5,
            solubility_mg_per_ml=0.1,
        ),
        regimen=dict(
            dose_mg=20.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=60.0,
            bioavailability=0.80,
            absorption_rate_h=0.5,
            volume_distribution_l=12.0,
            clearance_ml_per_min=0.37,
            half_life_h=35.0,
            hepatic_extraction_ratio=0.1,
            renal_fraction=0.6,
            cyp_metabolism={"CYP2C19": 0.40, "CYP3A4": 0.30, "CYP2D6": 0.20, "CYP2B6": 0.10},
        ),
    ),
    _Spec(
        key="TRAMADOL",
        molecule=dict(
            name="tramadol",
            drug_type=SMALL_MOLECULE,
            smiles="COc1ccc2c(c1OC)CC[C@@H](C2)N(C)C",
            molecular_weight_da=263.4,
            formula="C16H25NO2",
            target_protein="OPRM1",
            binding_affinity_kd_um=1.0,
            selectivity_index=10.0,
            protein_binding_fraction=0.20,
            log_p=1.35,
            solubility_mg_per_ml=75.0,
        ),
        regimen=dict(
            dose_mg=50.0,
            dosing_interval_h=6.0,
            route=ORAL,
            duration_days=7.0,
            bioavailability=0.68,
            absorption_rate_h=1.5,
            volume_distribution_l=224.0,
            clearance_ml_per_min=440.0,
            half_life_h=6.0,
            hepatic_extraction_ratio=0.3,
            renal_fraction=0.9,
            cyp_metabolism={"CYP2D6": 0.60, "CYP3A4": 0.30, "CYP2B6": 0.10},
        ),
    ),
    _Spec(
        key="ALLOPURINOL",
        molecule=dict(
            name="allopurinol",
            drug_type=SMALL_MOLECULE,
            smiles="O=c1[nH]cnc2nc[nH]c12",
            molecular_weight_da=136.1,
            formula="C5H4N4O",
            target_protein="XO",
            binding_affinity_kd_um=0.05,
            selectivity_index=50.0,
            protein_binding_fraction=0.0,
            log_p=-0.55,
            solubility_mg_per_ml=1.4,
        ),
        regimen=dict(
            dose_mg=300.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.90,
            absorption_rate_h=1.0,
            volume_distribution_l=30.0,
            clearance_ml_per_min=18.0,
            half_life_h=1.5,
            hepatic_extraction_ratio=0.1,
            renal_fraction=0.9,
            cyp_metabolism={"CYP1A2": 0.10, "CYP2E1": 0.50, "CYP2C19": 0.40},
        ),
    ),
    _Spec(
        key="PREDNISONE",
        molecule=dict(
            name="prednisone",
            drug_type=SMALL_MOLECULE,
            smiles="CC12CCC3C(CCC4CC(=O)CCC34C)C1CC(O)=O2",
            molecular_weight_da=358.4,
            formula="C21H26O5",
            target_protein="NR3C1",
            binding_affinity_kd_um=0.5,
            selectivity_index=10.0,
            protein_binding_fraction=0.70,
            log_p=1.46,
            solubility_mg_per_ml=0.2,
        ),
        regimen=dict(
            dose_mg=10.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=14.0,
            bioavailability=0.80,
            absorption_rate_h=1.0,
            volume_distribution_l=45.0,
            clearance_ml_per_min=70.0,
            half_life_h=3.0,
            hepatic_extraction_ratio=0.0,
            renal_fraction=0.5,
            cyp_metabolism={"CYP3A4": 0.80, "CYP2C9": 0.10, "CYP1A2": 0.10},
        ),
    ),
    _Spec(
        key="OMEPRAZOLE",
        molecule=dict(
            name="omeprazole",
            drug_type=SMALL_MOLECULE,
            smiles="COc1ccc2nc(CS(=O)c3ncc(C)cc3C)cc2n1",
            molecular_weight_da=345.4,
            formula="C17H19N3O3S",
            target_protein="ATP4A",
            binding_affinity_kd_um=0.01,
            selectivity_index=500.0,
            protein_binding_fraction=0.95,
            log_p=2.23,
            solubility_mg_per_ml=0.06,
        ),
        regimen=dict(
            dose_mg=20.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=14.0,
            bioavailability=0.35,
            absorption_rate_h=0.8,
            volume_distribution_l=21.0,
            clearance_ml_per_min=500.0,
            half_life_h=1.0,
            hepatic_extraction_ratio=0.2,
            renal_fraction=0.8,
            cyp_metabolism={"CYP2C19": 0.75, "CYP3A4": 0.25},
        ),
    ),
    _Spec(
        key="SIMVASTATIN",
        molecule=dict(
            name="simvastatin",
            drug_type=SMALL_MOLECULE,
            smiles="CCC(C)(C)C(=O)OC[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](O)[C@@H]2C1",
            molecular_weight_da=418.6,
            formula="C25H38O5",
            target_protein="HMGCR",
            binding_affinity_kd_um=0.01,
            selectivity_index=200.0,
            protein_binding_fraction=0.98,
            log_p=4.7,
            solubility_mg_per_ml=0.003,
        ),
        regimen=dict(
            dose_mg=20.0,
            dosing_interval_h=24.0,
            route=ORAL,
            duration_days=365.0,
            bioavailability=0.05,
            absorption_rate_h=1.0,
            volume_distribution_l=356.0,
            clearance_ml_per_min=1290.0,
            half_life_h=3.0,
            hepatic_extraction_ratio=0.7,
            renal_fraction=0.13,
            cyp_metabolism={"CYP3A4": 0.90, "CYP2C8": 0.05, "CYP2D6": 0.05},
            transporter_affected={"SLCO1B1": 0.9, "ABCB1": 0.1},
        ),
    ),
    _Spec(
        key="DIPHENHYDRAMINE",
        molecule=dict(
            name="diphenhydramine",
            drug_type=SMALL_MOLECULE,
            smiles="CN(CCOc1ccccc1)c1ccccc1",
            molecular_weight_da=255.4,
            formula="C17H21NO",
            target_protein="HRH1",
            binding_affinity_kd_um=0.02,
            selectivity_index=50.0,
            protein_binding_fraction=0.80,
            log_p=3.27,
            solubility_mg_per_ml=2.7,
        ),
        regimen=dict(
            dose_mg=25.0,
            dosing_interval_h=8.0,
            route=ORAL,
            duration_days=3.0,
            bioavailability=0.50,
            absorption_rate_h=1.5,
            volume_distribution_l=17.0,
            clearance_ml_per_min=900.0,
            half_life_h=4.0,
            hepatic_extraction_ratio=0.6,
            renal_fraction=0.5,
            cyp_metabolism={"CYP2D6": 0.80, "CYP3A4": 0.15, "CYP2C9": 0.05},
        ),
    ),
)


def _build_predefined_drugs() -> dict[str, Drug]:
    """Instantiate every pre-defined profile from its spec template.

    Sources (doc/27 section 6.5): imiglucerase — Genzyme 2022 label, Weinreb
    2004 (60 IU/kg biweekly, 70 kg reference weight, ~40 IU/mg specific
    activity); ibuprofen — Rang & Dale 2019; metformin — Graham 2011;
    cisplatin — Kelland 2007 (50 mg/m2 q21d at reference BSA 1.7 m2);
    tamoxifen — Jordan 2003 (half-life of active N-desmethyl metabolite);
    imatinib — Druker 2006, Peng 2004.
    """
    drugs: dict[str, Drug] = {}
    for spec in _PREDEFINED_SPECS:
        molecule = _refresh_computed_properties(DrugMolecule(**spec.molecule))
        drugs[spec.key] = Drug(molecule=molecule, **spec.regimen)
    return drugs


PREDEFINED_DRUGS: dict[str, Drug] = _build_predefined_drugs()


def get_predefined_drug(name: str) -> Drug | None:
    """Return a deep copy of a pre-defined profile by case-insensitive name."""
    key = name.strip().upper()
    if key not in PREDEFINED_DRUGS:
        return None
    return deepcopy(PREDEFINED_DRUGS[key])


def list_predefined_drugs() -> tuple[str, ...]:
    """Names of all pre-defined drug profiles."""
    return tuple(PREDEFINED_DRUGS)


# ============================================================================
# SMILES → ADME Inference (Phase 3)
# ============================================================================

def biologics_adme(mw_da: float) -> dict[str, float]:
    """ADME parameter estimation for biologics (mAbs, enzymes, peptides).

    Biologics (MW > 5000 Da) follow fundamentally different pharmacokinetics
    than small molecules: IV-only absorption, FcRn-mediated recycling giving
    long half-lives (~21 days for IgG1-class mAbs), renal catabolism for
    fragments < 60 kDa, hepatic asialoglycoprotein receptor clearance for
    larger species, low volume of distribution (plasma-restricted), and near-
    complete protein binding.

    Parameters
    ----------
    mw_da : float
        Molecular weight in Daltons (e.g. 150 000 for a full IgG mAb).

    Returns
    -------
    dict[str, float]
        Same 10-key schema as ``smiles_to_adme()`` for seamless integration.
    """
    mw_da = max(mw_da, 5000.0)

    # Bioavailability: near zero for SC (1–5% depending on route) → default IV
    bioavailability = 0.01 if mw_da > 100_000 else 0.05

    # Protein binding: essentially 1.0 for intact mAbs (FcRn binding)
    protein_binding = 0.99

    # Half-life: FcRn recycling scales with IgG subclass; MW > 150 kDa → long
    # Small fragments (< 60 kDa) are renally cleared faster
    if mw_da < 60_000:
        half_life_h = 24.0 + (mw_da - 5_000) / 55_000.0 * 72.0  # 24–96 h
    elif mw_da < 150_000:
        half_life_h = 96.0 + (mw_da - 60_000) / 90_000.0 * 408.0  # 4–21 days
    else:
        half_life_h = 504.0  # ~21 days (FcRn-saturated IgG1-class)

    # Volume of distribution: restricted to plasma + interstitial (~3–5 L)
    volume_distribution_l = 3.0 + 2.0 * min(1.0, mw_da / 150_000.0)

    # Clearance: dominated by catabolic degradation (renal for < 60 kDa,
    # hepatic ASGPR + reticuloendothelial for larger species)
    if mw_da < 60_000:
        cl_base = 20.0 * (60_000.0 / max(mw_da, 1.0))  # mL/min, renal scaling
    else:
        cl_base = 0.5 + 1.5 * (1.0 - min(1.0, mw_da / 500_000.0))  # mL/min
    clearance_ml_per_min = max(0.1, min(cl_base, 50.0))

    # Absorption rate: IV bolus → effectively infinite; SC → very slow
    absorption_rate_h = 0.1  # placeholder for IV (not used if route is IV)

    # Hepatic extraction: low for most mAbs (FcRn recycling rescues them)
    hepatic_extraction_ratio = 0.05

    # Renal fraction: MW-gated; small fragments cleared renally
    renal_fraction = max(0.0, min(0.8, 0.8 * (1.0 - mw_da / 150_000.0)))

    # LogP: not meaningful for biologics; return 0.0
    log_p = 0.0

    return {
        "bioavailability": round(bioavailability, 3),
        "protein_binding": round(protein_binding, 3),
        "half_life_h": round(half_life_h, 1),
        "volume_distribution_l": round(volume_distribution_l, 1),
        "clearance_ml_per_min": round(clearance_ml_per_min, 2),
        "absorption_rate_h": round(absorption_rate_h, 1),
        "hepatic_extraction_ratio": round(hepatic_extraction_ratio, 2),
        "renal_fraction": round(renal_fraction, 2),
        "log_p": round(log_p, 2),
        "molecular_weight_da": round(mw_da, 1),
    }


def smiles_to_adme(
    smiles: str,
    *,
    drug_type: str = SMALL_MOLECULE,
    mw_da: float = 0.0,
) -> dict[str, float]:
    """Infer ADME parameters from a SMILES string using molecular descriptors.

    Uses RDKit when available for accurate descriptor computation; falls back
    to heuristic rules based on SMILES length and element counts when RDKit is
    unavailable.  This is for novel drugs not in the predefined library.

    Parameters
    ----------
    smiles : str
        SMILES representation (may be empty for biologics without SMILES).
    drug_type : str, optional
        One of ``VALID_DRUG_TYPES``.  When ``BIOLOGIC`` (or MW > 30 000 Da
        when ``mw_da`` is given), biologic-specific ADME heuristics are used.
    mw_da : float, optional
        Known molecular weight in Daltons.  Overrides Lipinski-derived MW for
        biologics or when SMILES is unavailable.

    Returns a dict with keys: ``bioavailability``, ``protein_binding``,
    ``half_life_h``, ``volume_distribution_l``, ``clearance_ml_per_min``,
    ``absorption_rate_h``, ``hepatic_extraction_ratio``, ``renal_fraction``,
    ``log_p``, ``molecular_weight_da``.
    """
    effective_mw = mw_da
    is_biologic = (drug_type == BIOLOGIC) or (effective_mw > 30_000.0)
    if is_biologic and effective_mw > 0.0:
        return biologics_adme(effective_mw)
    if _HAS_RDKIT:
        return _smiles_to_adme_rdkit(smiles)
    return _smiles_to_adme_heuristic(smiles)


def _smiles_to_adme_rdkit(smiles: str) -> dict[str, float]:
    """RDKit-based ADME inference using Lipinski/Veber descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _smiles_to_adme_heuristic(smiles)

    mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
    hbd = Descriptors.NumHDonors(mol)  # type: ignore[attr-defined]
    hba = Descriptors.NumHAcceptors(mol)  # type: ignore[attr-defined]
    rotbonds = Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]

    # Lipinski rule of 5 based bioavailability
    ro5_violations = 0
    if mw > 500:
        ro5_violations += 1
    if logp > 5:
        ro5_violations += 1
    if hbd > 5:
        ro5_violations += 1
    if hba > 10:
        ro5_violations += 1
    bioavailability = max(0.05, min(0.95, 1.0 - 0.2 * ro5_violations))

    # Protein binding: LogP and MW dependent
    protein_binding = min(0.99, max(0.0, 0.8 + logp * 0.03 - mw * 0.0001))

    # Half-life: Lipinski-based estimation
    base_t12 = 4.0
    if logp > 3:
        base_t12 += (logp - 3) * 1.5
    if mw > 400:
        base_t12 += (mw - 400) / 200.0
    half_life_h = max(1.0, min(base_t12, 48.0))

    # Volume of distribution
    vd = 30.0 + logp * 10.0 - (15.0 if mw > 500 else 0.0)
    volume_distribution_l = max(5.0, min(vd, 200.0))

    # Clearance: MW and tPSA dependent
    base_cl = 100.0 * (mw / 300.0) ** 0.5 * (1.0 - protein_binding)
    if tpsa > 140:
        base_cl *= 0.5
    clearance_ml_per_min = max(1.0, min(base_cl, 1000.0))

    # Absorption rate
    absorption_rate_h = max(0.3, min(3.0, 1.0 + rotbonds * 0.1 - logp * 0.1))

    # Hepatic extraction
    hepatic_extraction_ratio = min(0.9, max(0.0, 0.1 + logp * 0.05))

    # Renal fraction: MW < 500 and low LogP → more renal
    renal_fraction = max(0.0, min(1.0, 0.5 - logp * 0.08 + (1.0 if mw < 500 else 0.0)))

    return {
        "bioavailability": round(bioavailability, 3),
        "protein_binding": round(protein_binding, 3),
        "half_life_h": round(half_life_h, 1),
        "volume_distribution_l": round(volume_distribution_l, 1),
        "clearance_ml_per_min": round(clearance_ml_per_min, 1),
        "absorption_rate_h": round(absorption_rate_h, 1),
        "hepatic_extraction_ratio": round(hepatic_extraction_ratio, 2),
        "renal_fraction": round(renal_fraction, 2),
        "log_p": round(logp, 2),
        "molecular_weight_da": round(mw, 1),
    }


def _smiles_to_adme_heuristic(smiles: str) -> dict[str, float]:
    """Heuristic ADME inference when RDKit is unavailable."""
    # Simple heuristic based on SMILES string characteristics
    length = len(smiles)
    n_nitrogen = smiles.count("N") + smiles.count("n")
    n_halogen = smiles.count("F") + smiles.count("Cl") + smiles.count("Br")

    # Estimate MW proxy from SMILES length
    estimated_mw = 200.0 + length * 3.0
    estimated_logp = 1.0 + (length - 50) * 0.05 + n_halogen * 0.5 - n_nitrogen * 0.3

    # Bioavailability
    ro5_violations = 0
    if estimated_mw > 500:
        ro5_violations += 1
    if estimated_logp > 5:
        ro5_violations += 1
    bioavailability = max(0.05, min(0.95, 0.8 - 0.15 * ro5_violations))

    # Protein binding
    protein_binding = min(0.99, max(0.0, 0.7 + estimated_logp * 0.04))

    # Half-life
    half_life_h = max(1.0, min(48.0, 4.0 + estimated_logp * 1.0 + (estimated_mw - 300) / 200.0))

    # Volume of distribution
    vd = 30.0 + estimated_logp * 10.0
    volume_distribution_l = max(5.0, min(vd, 200.0))

    # Clearance
    clearance_ml_per_min = max(1.0, min(1000.0, 100.0 * (estimated_mw / 300.0) ** 0.5 * (1.0 - protein_binding)))

    # Absorption rate
    absorption_rate_h = max(0.3, min(3.0, 1.0))

    # Hepatic extraction
    hepatic_extraction_ratio = min(0.9, max(0.0, 0.1 + estimated_logp * 0.05))

    # Renal fraction
    renal_fraction = max(0.0, min(1.0, 0.5 - estimated_logp * 0.08 + (1.0 if estimated_mw < 500 else 0.0)))

    return {
        "bioavailability": round(bioavailability, 3),
        "protein_binding": round(protein_binding, 3),
        "half_life_h": round(half_life_h, 1),
        "volume_distribution_l": round(volume_distribution_l, 1),
        "clearance_ml_per_min": round(clearance_ml_per_min, 1),
        "absorption_rate_h": round(absorption_rate_h, 1),
        "hepatic_extraction_ratio": round(hepatic_extraction_ratio, 2),
        "renal_fraction": round(renal_fraction, 2),
        "log_p": round(estimated_logp, 2),
        "molecular_weight_da": round(estimated_mw, 1),
    }
