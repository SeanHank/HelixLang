"""Bottom-up GEM reconstruction: annotation → reaction network (doc/20 §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from helixlang.annotation import GeneAnnotation
from helixlang.annotation.ec_mapping import (
    ECReactionDB, build_ec_db, REACTION_EQUATIONS,
)
from helixlang.annotation.kegg_mapping import KOReactionDB, build_ko_db


@dataclass
class GPRRule:
    """Gene-Protein-Reaction association rule.

    Represents: gene(s) → protein(s) → reaction(s)
    Following S-system notation:  AND for complexes, OR for isoenzymes.
    """

    reaction_id: str
    gene_ids: list[str] = field(default_factory=list)
    and_groups: list[list[str]] = field(default_factory=list)
    rule_string: str = ""

    def __post_init__(self) -> None:
        if not self.rule_string and self.gene_ids:
            if self.and_groups:
                parts = [
                    " and ".join(g) if len(g) > 1 else g[0]
                    for g in self.and_groups
                ]
                self.rule_string = " or ".join(parts)
            else:
                self.rule_string = " or ".join(self.gene_ids)


@dataclass
class ReactionEntry:
    """A single metabolic reaction in the draft model."""

    reaction_id: str
    equation: str
    reversibility: bool = True
    gene_ids: list[str] = field(default_factory=list)
    gpr: GPRRule | None = None
    confidence: float = 1.0
    source: str = "ec_mapping"  # "ec_mapping", "ko_mapping", "spontaneous"


@dataclass
class BottomUpResult:
    """Result of bottom-up GEM reconstruction."""

    reactions: list[ReactionEntry] = field(default_factory=list)
    genes_annotated: int = 0
    genes_with_reactions: int = 0
    ec_matched: int = 0
    ko_matched: int = 0
    spontaneous: int = 0

    @property
    def reaction_count(self) -> int:
        return len(self.reactions)

    @property
    def gene_count(self) -> int:
        return len({g for r in self.reactions for g in r.gene_ids})

    def reaction_ids(self) -> list[str]:
        return [r.reaction_id for r in self.reactions]

    def gpr_rules(self) -> dict[str, GPRRule]:
        return {r.reaction_id: r.gpr for r in self.reactions if r.gpr}


def bottom_up_reconstruct(
    annotations: dict[str, GeneAnnotation],
    ec_db: ECReactionDB | None = None,
    ko_db: KOReactionDB | None = None,
    include_spontaneous: bool = True,
) -> BottomUpResult:
    """Build a draft metabolic network from gene annotations (bottom-up).

    For each annotated gene:
    1. Map EC numbers → reactions via ECReactionDB
    2. Map KO terms → reactions via KOReactionDB
    3. Assign GPR rules
    4. Optionally include spontaneous reactions

    Parameters
    ----------
    annotations : {gene_id: GeneAnnotation}
    ec_db : EC number → reaction database (default: E. coli core)
    ko_db : KEGG Orthology → reaction database
    include_spontaneous : whether to add spontaneous (no-enzyme) reactions

    Returns
    -------
    BottomUpResult with draft reaction network
    """
    if ec_db is None:
        ec_db = build_ec_db()
    if ko_db is None:
        ko_db = build_ko_db()

    result = BottomUpResult()
    result.genes_annotated = len(annotations)

    # Build gene → reactions mapping
    gene_reactions: dict[str, list[str]] = {}
    ec_matched = 0
    ko_matched = 0

    for gene_id, annot in annotations.items():
        # EC number mapping
        for ec in annot.ec_numbers:
            mapping = ec_db.lookup(ec)
            if mapping:
                ec_matched += 1
                for rxn_id in mapping.reaction_ids:
                    gene_reactions.setdefault(rxn_id, []).append(gene_id)

        # KO mapping
        for ko in annot.kegg_ko:
            ko_mapping = ko_db.lookup(ko)
            if ko_mapping:
                ko_matched += 1
                for rxn_id in ko_mapping.reaction_ids:
                    gene_reactions.setdefault(rxn_id, []).append(gene_id)

    result.ec_matched = ec_matched
    result.ko_matched = ko_matched

    # Build reaction entries with GPR rules
    seen_reactions: set[str] = set()
    for rxn_id, gene_ids in gene_reactions.items():
        if rxn_id in seen_reactions:
            continue
        seen_reactions.add(rxn_id)
        unique_genes = list(dict.fromkeys(gene_ids))
        gpr = GPRRule(
            reaction_id=rxn_id,
            gene_ids=unique_genes,
        )
        # Look up equation from reaction database
        equation = REACTION_EQUATIONS.get(rxn_id, "")
        result.reactions.append(ReactionEntry(
            reaction_id=rxn_id,
            equation=equation,
            gene_ids=unique_genes,
            gpr=gpr,
            confidence=1.0,
            source="ec_mapping",
        ))

    # Add spontaneous reactions (no enzyme required)
    if include_spontaneous:
        spontaneous_rxns = [
            "H2O", "PIt2r", "PPR7GK", "r0148", "r0151",
        ]
        for rxn_id in spontaneous_rxns:
            if rxn_id not in seen_reactions:
                seen_reactions.add(rxn_id)
                equation = REACTION_EQUATIONS.get(rxn_id, "")
                result.reactions.append(ReactionEntry(
                    reaction_id=rxn_id,
                    equation=equation,
                    gene_ids=[],
                    gpr=None,
                    confidence=0.8,
                    source="spontaneous",
                ))
                result.spontaneous += 1

    result.genes_with_reactions = result.gene_count
    return result
