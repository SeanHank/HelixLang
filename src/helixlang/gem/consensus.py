"""Consensus merge of bottom-up and top-down models (doc/20 §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from helixlang.gem.bottom_up import BottomUpResult, GPRRule, ReactionEntry
from helixlang.gem.top_down import TopDownResult


@dataclass
class ConsensusReaction:
    """A reaction with origin tracking and confidence scoring."""

    reaction_id: str
    equation: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    gpr: GPRRule | None = None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    @property
    def is_medium_confidence(self) -> bool:
        return 0.4 <= self.confidence < 0.8


@dataclass
class ConsensusResult:
    """Result of consensus model merging."""

    reactions: list[ConsensusReaction] = field(default_factory=list)
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    from_bottom_up_only: int = 0
    from_top_down_only: int = 0
    from_both: int = 0

    @property
    def reaction_count(self) -> int:
        return len(self.reactions)

    def reaction_ids(self) -> list[str]:
        return [r.reaction_id for r in self.reactions]

    def high_confidence_ids(self) -> list[str]:
        return [r.reaction_id for r in self.reactions if r.is_high_confidence]


def consensus_merge(
    bottom_up: BottomUpResult,
    top_down: TopDownResult,
) -> ConsensusResult:
    """Merge bottom-up and top-down models into consensus.

    Strategy (GEMsembler-style):
    - Union: keep reaction if EITHER strategy supports it
    - Confidence: HIGH if both agree, MEDIUM if one, LOW if gap-filled
    - GPR: prefer bottom-up GPR (more specific)

    Parameters
    ----------
    bottom_up : result from bottom_up_reconstruct()
    top_down : result from top_down_reconstruct()

    Returns
    -------
    ConsensusResult with merged reactions and confidence scores
    """
    result = ConsensusResult()

    # Index reactions by ID
    bu_by_id: dict[str, ReactionEntry] = {
        r.reaction_id: r for r in bottom_up.reactions
    }
    td_by_id: dict[str, ReactionEntry] = {
        r.reaction_id: r for r in top_down.reactions
    }

    all_ids = set(bu_by_id.keys()) | set(td_by_id.keys())

    for rxn_id in sorted(all_ids):
        bu = bu_by_id.get(rxn_id)
        td = td_by_id.get(rxn_id)

        sources: list[str] = []
        if bu is not None:
            sources.append("bottom_up")
        if td is not None:
            sources.append("top_down")

        # Confidence scoring
        if bu is not None and td is not None:
            confidence = 0.9  # both agree
            result.from_both += 1
        elif bu is not None:
            confidence = 0.7  # bottom-up only
            result.from_bottom_up_only += 1
        else:
            confidence = 0.6  # top-down only
            result.from_top_down_only += 1

        # GPR: prefer bottom-up (more gene-specific)
        gpr = bu.gpr if (bu and bu.gpr) else (td.gpr if td else None)
        equation = (bu.equation if bu and bu.equation
                    else td.equation if td else "")

        result.reactions.append(ConsensusReaction(
            reaction_id=rxn_id,
            equation=equation,
            sources=sources,
            confidence=confidence,
            gpr=gpr,
        ))

    # Count confidence levels
    for rxn in result.reactions:
        if rxn.is_high_confidence:
            result.high_confidence += 1
        elif rxn.is_medium_confidence:
            result.medium_confidence += 1
        else:
            result.low_confidence += 1

    return result
