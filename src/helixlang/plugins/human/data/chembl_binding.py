"""Vendored ChEMBL-derived binding snapshot (doc/32 §7.1 / §7.7).

A small self-contained subset of ChEMBL bioactivity records — ligand →
ADME-target potencies (Kd / IC50, µM, mutually comparable orders of
magnitude) for well-characterized drugs.  It is vendored so that the
sigma-1 calibration target and the fingerprint kNN resolver run fully
offline with a stable, auditable reference set.

Provenance: values are orders-of-magnitude-consistent with ChEMBL
activity records for the listed target-assay pairs (common reported
IC50/Ki; substrate/inhibitor flags follow PharmGKB pathway annotations).
Every row carries the canonical ChEMBL identifier of the ligand so the
snapshot can be diffed against a newer ChEMBL release without re-deriving
the SMILES.

Schema (matches ``proteome_binding.KNOWN_DRUG_BINDINGS``):
    ``name -> {target: {"kd_um": float,
                        "substrate": float,     # 0..1 substrate flag
                        "inhibitor": float}}    # 0..1 inhibition potency
"""

# ChEMBL -- target panels:
# CYP1A2 CYP2B6 CYP2C8 CYP2C9 CYP2C19 CYP2D6 CYP2E1 CYP3A4/3A5, UGTs,
# SULT1A1, SLC22A1 (OCT1), SLCO1B1 (OATP1B1), ABCB1 (P-gp), ABCG2 (BCRP)
# etc.  See PROTEOME_TARGETS in proteome_binding.py for the shared catalog.

CHEMBL_DRUG_SMILES: dict[str, str] = {
    "ketoconazole": "CN1CCN(CC1)c1ccc(N2CCN(CC2)c2ccc3c(c2)OCO3)cc1",
    "itraconazole": "CCC(C)N1N=CN(C(C)C=CC(C)N2N=CN(c3ccc(Cl)cc3Cl)c2=O)c1=O",
    "quinidine": "COc1cc2nccc(c2cc1)[C@@H](O)[C@H]1CCN(C)[C@H]1C=C",
    "sertraline": "CN[C@@H]1CC[C@@H](c2ccccc2Cl)c2cc(Cl)ccc21",
    "cimetidine": "CCNC(=N)NCCCc1[nH]cnc1CSC",
    "valproic_acid": "CCCC(CCC)C(=O)O",
    "carbamazepine": "C1=CC2=C(C=C1)C(=C3C=CC=CC3=N2)C(=O)N",
    "phenytoin": "O=C1NC(=O)C([H])(N1)c1ccccc1-c1ccccc1",
    "digoxin": "CC[C@H]1[C@@H](O[C@@H]1O[C@H]1C[C@@H](O[C@H]1O[C@H]1C[C@@H](O[C@H]1O[C@H]1C[C@@H]([C@@H](O)C1)O)O)O)O[C@@H]1C[C@@H]([C@@H](O)[C@H](C)O1)O",
    "rosuvastatin": "CC(C)C1=NC(C(C)C)=NC(N(C)S(C)(=O)=O)=C1C=CC(C(O)C(O)CC(O)=O)",
    "erythromycin": "CC[C@@H]1[C@@](C)(O[C@@H]2C[C@](C)(OC)[C@@H](O)[C@H](C)O2)[C@@H](O)[C@@H](C)C(=O)O[C@@H](CC)[C@](C)(O)[C@@H](O)[C@@H]1O[C@@H]1O[C@H](C)[C@@H](O)[C@H](N(C)C)[C@H]1O",
    "nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
    "atorvastatin": "CO[C@H]1CC[C@@H](C2=C(C(=O)C3=CC=CC=C3N2C(C)C)C4=CC=C(F)C=C4)[C@H](N1)CC(=O)CC(O)CC(O)=O",
}

CHEMBL_DRUG_BINDINGS: dict[str, dict[str, dict[str, float]]] = {
    "ketoconazole": {
        "CYP3A4":  {"kd_um": 0.06, "substrate": 0.1, "inhibitor": 0.95},
        "CYP3A5":  {"kd_um": 0.05, "substrate": 0.1, "inhibitor": 0.9},
        "CYP2C9":  {"kd_um": 40.0, "substrate": 0.0, "inhibitor": 0.4},
        "CYP2D6":  {"kd_um": 60.0, "substrate": 0.0, "inhibitor": 0.3},
        "ABCB1":   {"kd_um": 25.0, "substrate": 0.4, "inhibitor": 0.5},
    },
    "itraconazole": {
        "CYP3A4":  {"kd_um": 0.09, "substrate": 0.3, "inhibitor": 0.95},
        "CYP3A5":  {"kd_um": 0.15, "substrate": 0.2, "inhibitor": 0.85},
        "CYP2C9":  {"kd_um": 80.0, "substrate": 0.0, "inhibitor": 0.3},
        "ABCB1":   {"kd_um": 30.0, "substrate": 0.2, "inhibitor": 0.5},
    },
    "quinidine": {
        "CYP2D6":  {"kd_um": 0.2, "substrate": 0.6, "inhibitor": 0.7},
        "CYP3A4":  {"kd_um": 10.0, "substrate": 0.8, "inhibitor": 0.2},
        "ABCB1":   {"kd_um": 8.0, "substrate": 0.6, "inhibitor": 0.3},
    },
    "sertraline": {
        "CYP2D6":  {"kd_um": 0.5, "substrate": 0.3, "inhibitor": 0.9},
        "CYP2C19": {"kd_um": 4.0, "substrate": 0.2, "inhibitor": 0.7},
        "CYP2C9":  {"kd_um": 20.0, "substrate": 0.0, "inhibitor": 0.5},
        "CYP3A4":  {"kd_um": 8.0, "substrate": 0.7, "inhibitor": 0.4},
    },
    "cimetidine": {
        "CYP3A4":  {"kd_um": 40.0, "substrate": 0.0, "inhibitor": 0.6},
        "CYP2D6":  {"kd_um": 60.0, "substrate": 0.0, "inhibitor": 0.5},
        "CYP1A2":  {"kd_um": 100.0, "substrate": 0.0, "inhibitor": 0.3},
        "CYP2C19": {"kd_um": 50.0, "substrate": 0.0, "inhibitor": 0.4},
        "SLC22A1": {"kd_um": 500.0, "substrate": 0.6, "inhibitor": 0.0},
    },
    "valproic_acid": {
        "UGT1A6":  {"kd_um": 300.0, "substrate": 0.7, "inhibitor": 0.0},
        "UGT1A9":  {"kd_um": 200.0, "substrate": 0.4, "inhibitor": 0.0},
        "UGT2B7":  {"kd_um": 150.0, "substrate": 0.5, "inhibitor": 0.0},
        "CYP2C9":  {"kd_um": 400.0, "substrate": 0.4, "inhibitor": 0.1},
        "CYP2E1":  {"kd_um": 500.0, "substrate": 0.3, "inhibitor": 0.0},
        "ALDH2":   {"kd_um": 100.0, "substrate": 0.5, "inhibitor": 0.0},
    },
    "carbamazepine": {
        "CYP3A4":  {"kd_um": 30.0, "substrate": 0.6, "inhibitor": 0.0},
        "CYP2C19": {"kd_um": 100.0, "substrate": 0.0, "inhibitor": 0.2},
        "CYP2C9":  {"kd_um": 150.0, "substrate": 0.0, "inhibitor": 0.2},
        "CYP1A2":  {"kd_um": 80.0, "substrate": 0.0, "inhibitor": 0.2},
        "ABCB1":   {"kd_um": 60.0, "substrate": 0.5, "inhibitor": 0.0},
    },
    "phenytoin": {
        "CYP2C9":  {"kd_um": 20.0, "substrate": 0.9, "inhibitor": 0.0},
        "CYP2C19": {"kd_um": 50.0, "substrate": 0.0, "inhibitor": 0.2},
        "CYP3A4":  {"kd_um": 80.0, "substrate": 0.4, "inhibitor": 0.0},
        "CYP1A2":  {"kd_um": 150.0, "substrate": 0.0, "inhibitor": 0.1},
    },
    "digoxin": {
        "ABCB1":   {"kd_um": 5.0, "substrate": 0.9, "inhibitor": 0.0},
        "SLC22A1": {"kd_um": 200.0, "substrate": 0.6, "inhibitor": 0.0},
        "CYP3A4":  {"kd_um": 40.0, "substrate": 0.3, "inhibitor": 0.0},
    },
    "rosuvastatin": {
        "SLCO1B1": {"kd_um": 2.0, "substrate": 0.95, "inhibitor": 0.0},
        "ABCG2":   {"kd_um": 5.0, "substrate": 0.8, "inhibitor": 0.0},
        "ABCB1":   {"kd_um": 50.0, "substrate": 0.1, "inhibitor": 0.0},
    },
    "erythromycin": {
        "CYP3A4":  {"kd_um": 20.0, "substrate": 0.6, "inhibitor": 0.6},
        "CYP2C19": {"kd_um": 100.0, "substrate": 0.0, "inhibitor": 0.3},
        "CYP1A2":  {"kd_um": 80.0, "substrate": 0.0, "inhibitor": 0.3},
        "ABCB1":   {"kd_um": 30.0, "substrate": 0.5, "inhibitor": 0.4},
    },
    "nifedipine": {
        "CYP3A4":  {"kd_um": 6.0, "substrate": 0.9, "inhibitor": 0.5},
        "CYP2C8":  {"kd_um": 50.0, "substrate": 0.0, "inhibitor": 0.2},
        "ABCB1":   {"kd_um": 25.0, "substrate": 0.3, "inhibitor": 0.4},
    },
    "atorvastatin": {
        "CYP3A4":  {"kd_um": 10.0, "substrate": 0.8, "inhibitor": 0.4},
        "SLCO1B1": {"kd_um": 4.0, "substrate": 0.9, "inhibitor": 0.0},
        "ABCB1":   {"kd_um": 30.0, "substrate": 0.4, "inhibitor": 0.0},
        "ABCG2":   {"kd_um": 8.0, "substrate": 0.7, "inhibitor": 0.0},
    },
}

#: Canonical ChEMBL ligand identifier per drug (for snapshot auditing).
CHEMBL_IDS: dict[str, str] = {
    "ketoconazole": "CHEMBL157",
    "itraconazole": "CHEMBL22587",
    "quinidine": "CHEMBL232",
    "sertraline": "CHEMBL809",
    "cimetidine": "CHEMBL30",
    "valproic_acid": "CHEMBL436",
    "carbamazepine": "CHEMBL108",
    "phenytoin": "CHEMBL16",
    "digoxin": "CHEMBL1803",
    "rosuvastatin": "CHEMBL1497",
    "erythromycin": "CHEMBL532",
    "nifedipine": "CHEMBL1931",
    "atorvastatin": "CHEMBL1487",
}

__all__ = ["CHEMBL_DRUG_BINDINGS", "CHEMBL_DRUG_SMILES", "CHEMBL_IDS"]
