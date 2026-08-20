"""KEGG Orthology (KO) → reaction mapping (doc/20 §5.2)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KOMapping:
    """KEGG Orthology term mapped to reactions."""

    ko_id: str
    reaction_ids: list[str] = field(default_factory=list)
    pathway_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0


class KOReactionDB:
    """KEGG Orthology → reaction lookup."""

    def __init__(self) -> None:
        self._ko_to_rxns: dict[str, KOMapping] = {}

    def load_from_dict(
        self, mapping: dict[str, dict[str, list[str]]]
    ) -> None:
        """Load from {ko_id: {"reactions": [...], "pathways": [...]}}."""
        for ko, data in mapping.items():
            self._ko_to_rxns[ko] = KOMapping(
                ko_id=ko,
                reaction_ids=data.get("reactions", []),
                pathway_ids=data.get("pathways", []),
            )

    def lookup(self, ko_id: str) -> KOMapping | None:
        return self._ko_to_rxns.get(ko_id)

    def has_ko(self, ko_id: str) -> bool:
        return ko_id in self._ko_to_rxns

    @property
    def size(self) -> int:
        return len(self._ko_to_rxns)


# Core metabolism KO mappings (E. coli MG1655)
CORE_KO_REACTIONS: dict[str, dict[str, list[str]]] = {
    "K00844": {"reactions": ["HEX1"], "pathways": ["map00010"]},
    "K01810": {"reactions": ["PGI"], "pathways": ["map00010"]},
    "K00850": {"reactions": ["PFK"], "pathways": ["map00010"]},
    "K01623": {"reactions": ["FBA"], "pathways": ["map00010"]},
    "K01624": {"reactions": ["TPI"], "pathways": ["map00010"]},
    "K00134": {"reactions": ["GAPD"], "pathways": ["map00010"]},
    "K00927": {"reactions": ["PGK"], "pathways": ["map00010"]},
    "K01626": {"reactions": ["PGM"], "pathways": ["map00010"]},
    "K01689": {"reactions": ["ENO"], "pathways": ["map00010"]},
    "K00872": {"reactions": ["PYK"], "pathways": ["map00010"]},
    "K00150": {"reactions": ["G3PD1", "G3PD2"], "pathways": ["map00010"]},
    "K00001": {"reactions": ["ADHEr"], "pathways": ["map00010"]},
    "K00128": {"reactions": ["ME1", "ME2"], "pathways": ["map00020"]},
    "K00024": {"reactions": ["ICDH"], "pathways": ["map00020"]},
    "K00244": {"reactions": ["CS"], "pathways": ["map00020"]},
    "K01681": {"reactions": ["ACONa", "ACONb"], "pathways": ["map00020"]},
    "K00239": {"reactions": ["SUCDi"], "pathways": ["map00020"]},
    "K00025": {"reactions": ["MDH"], "pathways": ["map00020"]},
    "K01067": {"reactions": ["FUM"], "pathways": ["map00020"]},
    "K00234": {"reactions": ["SUCOAS"], "pathways": ["map00020"]},
    "K01902": {"reactions": ["ICL"], "pathways": ["map00020"]},
    "K01638": {"reactions": ["MS"], "pathways": ["map00020"]},
}


def build_ko_db() -> KOReactionDB:
    """Build the default core KO → reaction database."""
    db = KOReactionDB()
    db.load_from_dict(CORE_KO_REACTIONS)
    return db
