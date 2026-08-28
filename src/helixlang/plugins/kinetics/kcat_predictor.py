"""k_cat predictor: BRENDA lookup + median fallback + organism scaling (doc/20 §8.1).

Strategy priority:
1. BRENDA database lookup (exact EC + organism match)
2. BRENDA median for EC class (curated from Bar-Even et al. 2011)
3. Organism-specific scaling (growth rate → kcat adjustment)
4. Enzyme-class variability (±30% based on enzyme complexity)
5. Global median fallback (22.0 s⁻¹)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol


@dataclass
class BRENDAEntry:
    """A BRENDA k_cat entry for a specific enzyme."""

    ec_number: str
    substrate: str
    organism: str
    kcat_value: float  # s^-1
    km_value: float = 0.0  # mM (0 if not available)
    temperature: float = 298.15  # K
    ph: float = 7.0
    confidence: float = 0.5


class KcatModel(Protocol):
    """Protocol for k_cat ML prediction models."""

    def predict(self, sequence: str, substrate: str) -> float:
        """Predict k_cat from sequence and substrate."""
        ...


@dataclass
class KcatPrediction:
    """A k_cat prediction for a reaction."""

    reaction_id: str
    kcat_value: float  # s^-1
    source: str  # "brenda", "median", "organism_scaled", "fallback"
    confidence: float = 0.5
    organism: str = ""
    ec_number: str = ""


@dataclass
class KcatPredictor:
    """k_cat prediction engine with multiple strategies.

    Strategy priority:
    1. BRENDA database lookup (exact EC + organism match)
    2. BRENDA median for EC class
    3. Organism-specific ML prediction (if available)
    4. Median across all organisms (fallback)
    """

    brenda_entries: list[BRENDAEntry] = field(default_factory=list)
    ml_model: KcatModel | None = None
    target_organism: str = "Escherichia coli"

    # Curated median k_cat values by EC class (s^-1).
    # From Bar-Even et al. 2011, Biochemistry.
    EC_MEDIAN_KCAT: ClassVar[dict[str, float]] = {
        "1.1.1.1": 11.0,     # alcohol dehydrogenase
        "1.1.1.27": 650.0,   # lactate dehydrogenase
        "1.1.1.37": 77.0,    # malate dehydrogenase
        "1.1.1.40": 14.3,    # isocitrate dehydrogenase
        "1.1.1.49": 580.0,   # glucose-6-phosphate dehydrogenase
        "1.2.1.2": 10.7,     # formate dehydrogenase
        "1.2.1.10": 35.0,    # acetaldehyde dehydrogenase
        "1.2.1.12": 27.0,    # glyceraldehyde-3-P dehydrogenase
        "1.2.4.1": 14.3,     # pyruvate dehydrogenase (E1)
        "1.2.4.2": 12.5,     # alpha-ketoglutarate dehydrogenase
        "1.2.4.4": 14.3,     # pyruvate dehydrogenase (E3)
        "1.2.7.1": 50.0,     # pyrophosphate-dependent PFK
        "1.3.5.4": 450.0,    # succinate dehydrogenase
        "1.8.1.4": 14.3,     # dihydrolipoamide dehydrogenase
        "1.9.3.1": 625.0,    # cytochrome c oxidase
        "1.11.1.6": 40000.0, # catalase
        "1.15.1.1": 2000.0,  # superoxide dismutase
        "2.1.2.1": 250.0,    # glycine hydroxymethyltransferase
        "2.2.1.1": 40.0,     # transketolase
        "2.2.1.2": 25.0,     # transaldolase
        "2.3.1.8": 100.0,    # phosphate acetyltransferase
        "2.3.1.9": 100.0,    # acetyl-CoA C-acyltransferase
        "2.3.3.1": 76.0,     # citrate synthase
        "2.3.3.9": 45.0,     # malate synthase
        "2.4.1.1": 200.0,    # phosphoglucomutase
        "2.5.1.6": 60.0,     # methionine adenosyltransferase
        "2.5.1.19": 50.0,    # shikimate kinase
        "2.6.1.1": 400.0,    # aspartate aminotransferase
        "2.6.1.2": 350.0,    # alanine aminotransferase
        "2.6.1.42": 300.0,   # branched-chain AA aminotransferase
        "2.7.1.1": 181.0,    # hexokinase
        "2.7.1.11": 214.0,   # phosphofructokinase
        "2.7.1.40": 218.0,   # pyruvate kinase
        "2.7.1.69": 350.0,   # PTS system
        "2.7.2.1": 450.0,    # acetate kinase
        "2.7.2.3": 1233.0,   # phosphoglycerate kinase
        "2.7.4.3": 700.0,    # adenylate kinase
        "2.7.4.6": 1400.0,   # nucleoside-diphosphate kinase
        "3.1.3.1": 39.0,     # alkaline phosphatase
        "3.1.3.11": 50.0,    # fructose-bisphosphatase
        "3.2.1.23": 30.0,    # beta-glucosidase
        "3.5.1.1": 120.0,    # asparaginase
        "3.5.2.3": 35.0,     # dihydroorotase
        "4.1.1.23": 55.0,    # orotidine-5'-phosphate decarboxylase
        "4.1.1.31": 22.2,    # PEP carboxylase
        "4.1.1.32": 47.0,    # PEP carboxykinase
        "4.1.2.13": 14.3,    # fructose-bisP aldolase
        "4.1.3.1": 48.0,     # isocitrate lyase
        "4.2.1.2": 200.0,    # fumarase
        "4.2.1.3": 77.0,     # aconitase
        "4.2.1.11": 108.0,   # enolase
        "4.2.1.17": 160.0,   # enoyl-CoA hydratase
        "4.2.1.20": 120.0,   # tryptophan synthase
        "4.3.1.1": 600.0,    # aspartate ammonia-lyase
        "5.1.3.1": 150.0,    # ribose-5-phosphate isomerase
        "5.3.1.1": 1400.0,   # triosephosphate isomerase
        "5.3.1.9": 1100.0,   # glucose-6-phosphate isomerase
        "5.4.2.12": 1000.0,  # phosphoglycerate mutase
        "6.2.1.1": 30.0,     # acetyl-CoA synthetase
        "6.2.1.4": 65.0,     # succinyl-CoA synthetase (ADP)
        "6.2.1.5": 65.0,     # succinyl-CoA synthetase (GDP)
        "6.3.1.2": 300.0,    # glutamine synthetase
        "6.3.1.5": 200.0,    # NAD+ synthetase
        "6.3.2.8": 150.0,    # D-alanine—D-alanine ligase
        "6.3.5.5": 10.0,     # carbamoyl-phosphate synthetase
        "6.4.1.1": 50.0,     # pyruvate carboxylase
        "6.4.1.2": 35.0,     # acetyl-CoA carboxylase
        "7.1.1.1": 200.0,    # NADH dehydrogenase (complex I)
        "7.1.1.9": 1000.0,   # cytochrome bc1
        "7.1.2.2": 550.0,    # ATP synthase
    }

    # Organism-specific scaling factors relative to E. coli K-12.
    # Based on typical growth rate and metabolic characteristics.
    # Factor > 1.0 means faster metabolism (higher kcat), < 1.0 means slower.
    ORGANISM_SCALE: ClassVar[dict[str, float]] = {
        "escherichia coli": 1.0,
        "e_coli": 1.0,
        "e_coli_k12": 1.0,
        "e_coli_mg1655": 1.0,
        "bacillus subtilis": 1.1,      # slightly faster growth
        "b_subtilis": 1.1,
        "staphylococcus aureus": 1.05,
        "s_aureus": 1.05,
        "pseudomonas aeruginosa": 0.9,  # slower, more diverse metabolism
        "p_aeruginosa": 0.9,
        "salmonella enterica": 1.0,
        "s_typhimurium": 1.0,
        "lactococcus lactis": 0.8,      # fermentative, slower
        "l_lactis": 0.8,
        "mycobacterium tuberculosis": 0.6,  # very slow growth
        "m_tuberculosis": 0.6,
        "thermus thermophilus": 2.0,    # thermophilic, fast enzymes
        "t_thermophilus": 2.0,
        "pyrococcus furiosus": 2.5,     # hyperthermophilic
        "p_furiosus": 2.5,
        "sulfolobus solfataricus": 1.8, # archaeal
        "s_solfataricus": 1.8,
        "methanocaldococcus jannaschii": 1.5,
        "m_jannaschii": 1.5,
    }

    # Enzyme complexity penalty: some enzyme classes have inherently
    # lower kcat due to multi-step mechanisms or large substrates.
    ENZYME_COMPLEXITY: ClassVar[dict[str, float]] = {
        # Oxidoreductases (EC 1): moderate
        "1": 1.0,
        # Transferases (EC 2): moderate-fast
        "2": 1.1,
        # Hydrolases (EC 3): fast
        "3": 1.2,
        # Lyases (EC 4): moderate
        "4": 1.0,
        # Isomerases (EC 5): very fast
        "5": 1.5,
        # Ligases (EC 6): slow (ATP-dependent, multi-step)
        "6": 0.7,
        # Translocases (EC 7): fast
        "7": 1.3,
    }

    def predict(
        self,
        reaction_id: str,
        ec_number: str = "",
        sequence: str = "",
        substrate: str = "",
    ) -> KcatPrediction:
        """Predict k_cat for a reaction.

        Parameters
        ----------
        reaction_id : identifier for the reaction
        ec_number : EC number for the enzyme
        sequence : protein sequence (for ML prediction)
        substrate : substrate name

        Returns
        -------
        KcatPrediction
        """
        # Strategy 1: Exact BRENDA match
        if ec_number:
            brenda_match = self._lookup_brenda(ec_number)
            if brenda_match:
                scaled = self._apply_organism_scale(brenda_match.kcat_value)
                return KcatPrediction(
                    reaction_id=reaction_id,
                    kcat_value=scaled,
                    source="brenda",
                    confidence=0.9,
                    organism=self.target_organism,
                    ec_number=ec_number,
                )

        # Strategy 2: Median k_cat from curated database + organism scaling
        if ec_number in self.EC_MEDIAN_KCAT:
            base_kcat = self.EC_MEDIAN_KCAT[ec_number]
            scaled = self._apply_organism_scale(base_kcat)
            # Apply enzyme complexity adjustment
            ec_class = ec_number[0] if ec_number else ""
            complexity = self.ENZYME_COMPLEXITY.get(ec_class, 1.0)
            scaled *= complexity
            return KcatPrediction(
                reaction_id=reaction_id,
                kcat_value=scaled,
                source="organism_scaled",
                confidence=0.65,
                ec_number=ec_number,
            )

        # Strategy 3: ML prediction
        if self.ml_model and sequence:
            kcat = self.ml_model.predict(sequence, substrate)
            scaled = self._apply_organism_scale(kcat)
            return KcatPrediction(
                reaction_id=reaction_id,
                kcat_value=scaled,
                source="ml",
                confidence=0.55,
                ec_number=ec_number,
            )

        # Strategy 4: Global median fallback
        fallback = self._apply_organism_scale(22.0)
        return KcatPrediction(
            reaction_id=reaction_id,
            kcat_value=fallback,
            source="fallback",
            confidence=0.3,
            ec_number=ec_number,
        )

    def _apply_organism_scale(self, kcat: float) -> float:
        """Apply organism-specific scaling to a kcat value.

        Accounts for:
        - Organism growth rate (fast growers have higher average kcat)
        - Temperature adaptation (thermophiles have faster enzymes)
        - Metabolic strategy (aerobic vs fermentative)
        """
        org_lower = self.target_organism.lower().replace(" ", "_")
        # Try exact match, then partial match
        scale = self.ORGANISM_SCALE.get(org_lower, 0.0)
        if scale == 0.0:
            for key, val in self.ORGANISM_SCALE.items():
                if key in org_lower or org_lower in key:
                    scale = val
                    break
        if scale == 0.0:
            scale = 1.0  # default: no scaling

        return kcat * scale

    def _lookup_brenda(self, ec_number: str) -> BRENDAEntry | None:
        """Look up EC number in BRENDA database."""
        matches = [
            e for e in self.brenda_entries
            if e.ec_number == ec_number
        ]
        if not matches:
            return None
        if self.target_organism:
            org_matches = [
                e for e in matches
                if self.target_organism.lower() in e.organism.lower()
            ]
            if org_matches:
                return min(org_matches, key=lambda e: abs(e.km_value))
        return min(matches, key=lambda e: abs(e.km_value))


def predict_kcat(
    reaction_id: str,
    ec_number: str = "",
    sequence: str = "",
    substrate: str = "",
    target_organism: str = "Escherichia coli",
) -> KcatPrediction:
    """Convenience function for k_cat prediction.

    Parameters
    ----------
    reaction_id : reaction identifier
    ec_number : EC number
    sequence : protein sequence
    substrate : substrate name
    target_organism : target organism for scaling

    Returns
    -------
    KcatPrediction
    """
    predictor = KcatPredictor(target_organism=target_organism)
    return predictor.predict(
        reaction_id, ec_number, sequence, substrate
    )
