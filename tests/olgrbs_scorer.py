#!/usr/bin/env python
"""Stage-1 self-check for olgrbs.scorer: OSTIR wrapper is deterministic, isolates SD
strength, and handles the no-start-codon case.

Two constructs share an identical layout and start position, differing ONLY in the
Shine-Dalgarno core (strong AGGAGG vs scrambled), so any expression difference is
attributable to the RBS the scorer is meant to read. Run directly:
    python tests/olgrbs_scorer.py
"""
from __future__ import annotations

from olgrbs import score_rbs

LEAD = "A" * 20
SPACER = "A" * 6          # SD -> ATG spacer
CDS = "GCA" * 8           # dummy downstream coding
START = 20 + 6 + 6        # len(LEAD) + len(SD) + len(SPACER) -> index of the A in ATG

STRONG = LEAD + "AGGAGG" + SPACER + "ATG" + CDS
WEAK = LEAD + "ACTCTC" + SPACER + "ATG" + CDS  # scrambled SD, same length/position


def main() -> None:
    assert STRONG[START:START + 3] == "ATG" and WEAK[START:START + 3] == "ATG"

    strong = score_rbs(STRONG, START)
    weak = score_rbs(WEAK, START)
    assert strong is not None and weak is not None, "OSTIR found no start codon"
    assert strong.start_codon == "ATG"

    # determinism (pure + cached): identical inputs -> identical result
    assert score_rbs(STRONG, START) == strong

    # a strong SD must out-express a scrambled one at the same position
    assert strong.expression > weak.expression, (
        f"strong {strong.expression} !> weak {weak.expression}")
    # ...and the mechanism is tighter mRNA:rRNA hybridization (more negative)
    assert strong.dG_mRNA_rRNA < weak.dG_mRNA_rRNA

    # no valid start codon at this position -> None (mid-CDS, not a start triplet)
    assert score_rbs(STRONG, START + 4) is None

    print(f"OK  strong expr={strong.expression:.1f} (dG_total={strong.dG_total:.2f}) "
          f"> weak expr={weak.expression:.1f} (dG_total={weak.dG_total:.2f})")


if __name__ == "__main__":
    main()
