"""Example 29: ML-guided directed evolution of GB1 (S8).

Closed-loop "design - build - test - learn" protein engineering in
miniature, mirroring EVOLVEpro (Jiang et al. Science 2025 387:eadr6006),
MULTI-evolve (Tran et al. Science 2025 388:aea1820) and FSFP
(Otalora Ottó et al. Nat Commun 2024 15:5566).  A zero-shot oracle
(ESM-2 pseudo-likelihood, Frazer et al. 2021; BLOSUM62 fallback) ranks
each round's single-residue mutant library against the GB1 wild type
(Wu et al. 2016 interface window), the top-K is screened on the
weighted-BLOSUM62 landscape, and the best hit becomes the next parent.

The guided campaign recovers far more fitness than an oracle-free
random-screening baseline, and the oracle's round-1 predictions align
with the landscape (Spearman, the ProteinGym benchmark metric).

Annotated `.helix` form of this workflow:
`examples/29_directed_evolution.helix`

Run with:  python examples/directed_evolution_workflow.py
"""
from helixlang.apps.protein_evolution import (
    GB1_WT,
    gbi_landscape,
    guided_directed_evolution,
    make_crippled,
)


def main() -> None:
    res = guided_directed_evolution(rounds=8, library_size=60, top_k=5)
    start = make_crippled()

    print(f"wild type fitness   : {gbi_landscape(GB1_WT):.3f}")
    print(f"crippled start      : {res.initial_fitness:.3f} "
          f"(make_crippled() = {gbi_landscape(start):.3f})")

    print("\ncumulative-best fitness per round (guided | random):")
    for rnd, (g, b) in enumerate(zip(res.guided_cumulative_best,
                                     res.baseline_cumulative_best,
                                     strict=True)):
        print(f"  round {rnd}: {g:.3f} | {b:.3f}")

    print(f"\noracle                : {res.oracle_name}")
    print(f"guided recovery       : {res.guided_gain:+.3f}")
    print(f"random recovery       : {res.baseline_gain:+.3f}")
    print(f"oracle vs landscape Spearman: {res.spearman_rho:.3f}")


if __name__ == "__main__":
    main()
