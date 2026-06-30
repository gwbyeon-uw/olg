#!/usr/bin/env python
"""Stage-3 self-check for olgrbs.search: every candidate is a connected path that preserves
BOTH proteins; count == #paths on a small window; sampler + SA neighbours stay valid; an
infeasible chain yields nothing.

Run:  python tests/olgrbs_search.py
"""
from __future__ import annotations

import torch
from olgrbs_toy import build_toy_design

from olg.constants import Arrangement
from olgrbs.search import OptionChain, build_chain
from olgrbs.window import rbs_window


def proteins(design, nt):
    return design.translate_sequences(nt_seq=nt)


def main() -> None:
    torch.manual_seed(0)
    design = build_toy_design(Arrangement.PLUS_ONE, 16, seed=16)
    base_proteins = None

    # small flank-only window: exact count must equal the enumerated path set
    chain = build_chain(design, rbs_window(design, w_up=3, w_down=0))
    base_nt = chain.to_nt(chain.base_path)
    base_proteins = proteins(design, base_nt)

    paths = list(chain._enumerate_paths())
    assert chain.count() == len(paths), f"count {chain.count()} != {len(paths)} paths"
    nts = list(chain.enumerate_nt())
    assert len(nts) == len(paths) and base_nt in nts
    # the whole point: every enumerated candidate keeps outer AND inner protein intact
    for nt in nts:
        assert proteins(design, nt) == base_proteins, "enumerated candidate changed a protein"
    assert len(set(nts)) > 1, "flank window gave no design freedom — suspicious"

    # larger window incl. locked overlap: sampler + SA neighbours preserve both proteins
    chain2 = build_chain(design, rbs_window(design, w_up=10, w_down=4))
    b2 = proteins(design, chain2.to_nt(chain2.base_path))
    samp = chain2.sample_nt(50, seed=1)
    assert samp, "sampler returned nothing"
    for nt in samp:
        assert proteins(design, nt) == b2, "sampled candidate changed a protein"
    nbrs = list(chain2.neighbors(chain2.base_path))
    assert nbrs, "no SA neighbours"
    for p in nbrs:
        assert proteins(design, chain2.to_nt(p)) == b2, "SA neighbour changed a protein"

    # opening the overlap to dual-synonymous still preserves both proteins
    chain3 = build_chain(design, rbs_window(design, w_up=8, w_down=4), open_overlap=True)
    b3 = proteins(design, chain3.to_nt(chain3.base_path))
    for nt in chain3.sample_nt(40, seed=2):
        assert proteins(design, nt) == b3, "open-overlap candidate changed a protein"

    # infeasible chain -> empty, not a crash
    empty = OptionChain(positions=[0, 1], reach=[[], []], base_quartets=chain.base_quartets,
                        base_path=[chain.base_quartets[0], chain.base_quartets[1]], inner_start_nt=0)
    assert empty.count() == 0 and list(empty.enumerate_nt()) == []

    print(f"OK  enumerated {len(nts)} flank candidates (count==paths), sampler/SA/open-overlap "
          f"all protein-preserving")


if __name__ == "__main__":
    main()
