#!/usr/bin/env python
"""Self-check for olg5utr: UTRDesign geometry + 100-nt 5'UTR guard, ACGT one-hot order, the
synonymous move space, and optimize_utr's discrete search (protein-preserving, never worse than
a random valid start, deterministic). Uses a dummy scorer — no Optimus weights needed.

Run:  python tests/olg5utr_driver.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

from olg5utr import UTRDesign, dna_to_onehot, optimize_utr

EGFP = "MVSKGEELFTGVVPILVELD"  # 20-aa outer protein


class DummyOptimus(nn.Module):
    """Deterministic per-sequence scorer (varies with the sequence) — exercises the search."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Parameter(torch.randn(4, 100), requires_grad=False)

    def forward(self, x: torch.Tensor, final_ind: int = 0) -> torch.Tensor:
        return (x * self.w[:, : x.shape[-1]]).sum(dim=(1, 2), keepdim=True)


def main() -> None:
    # geometry: window = free 5'UTR + outer CDS = inner 5'UTR
    d = UTRDesign(outer_protein=EGFP, free_len=20)
    assert d.outer_cds_len == 60 and d.length == 80

    # 100-nt guard: inner 5'UTR must fit the Optimus input
    try:
        UTRDesign(outer_protein="M" * 30, free_len=20)  # 20 + 90 = 110 > 100
        raise AssertionError("expected ValueError for a >100-nt 5'UTR")
    except ValueError:
        pass

    # DNA one-hot uses ACGT order (Optimus training compat — NOT olg's ATGC)
    assert dna_to_onehot(["ACGT"], 4).argmax(1).squeeze(0).tolist() == [0, 1, 2, 3]

    # synonymous move space: Met frozen (1 codon), Leu wobble (6 codons)
    syn = UTRDesign(outer_protein="ML", free_len=6).synonymous_codons()
    assert syn[0] == ["ATG"] and len(syn[1]) == 6

    # discrete search: protein-preserving, >= base, deterministic
    model = DummyOptimus()
    r = optimize_utr(d, model, n_parallel=8, steps=60, seed=0, top=5)
    assert r.best is not None and r.best.score.combined >= r.base.combined
    for c in r.candidates:
        assert d.outer_translation(c.dna) == EGFP, "outer protein changed"
        assert c.dna == c.free_utr + c.outer_cds and len(c.free_utr) == d.free_len
    again = optimize_utr(d, model, n_parallel=8, steps=60, seed=0, top=1)
    assert r.candidates and again.best.dna == optimize_utr(
        d, model, n_parallel=8, steps=60, seed=0, top=1).best.dna, "not deterministic"

    print(f"OK  geometry✓ guard✓ ACGT✓ synonymous✓ | search base={r.base.combined:.3f} "
          f"best={r.best.score.combined:.3f} (+{r.best.score.combined - r.base.combined:.3f}) "
          f"protein-preserved✓ deterministic✓")


if __name__ == "__main__":
    main()
