#!/usr/bin/env python
"""Stage-2 self-check for olgrbs.window: the RBS window is the outer-only upstream codons,
the inner-start NT lands on the forced ATG, and antisense is rejected.

Run:  python tests/olgrbs_window.py
"""
from __future__ import annotations

from olgrbs_toy import build_toy_design

from olg.constants import Arrangement
from olgrbs.window import rbs_window


def _check_sense(arr: Arrangement, offset: int) -> None:
    design = build_toy_design(arr, offset, seed=offset)
    win = rbs_window(design)
    a1 = design.coords.all_to_f1.tolist()
    a2 = design.coords.all_to_f2.tolist()
    nt, _ = design.string_quartet()

    # inner start = frame-2 residue 0, and its NT lands on the forced ATG start codon
    assert win.inner_start_qpos == int(design.coords.f2_to_all[0].item())
    assert nt[win.inner_start_nt:win.inner_start_nt + 3] == "ATG", (
        f"{arr.name} off={offset}: inner start {nt[win.inner_start_nt:win.inner_start_nt+3]} != ATG")

    # flank positions are upstream of the start; outer = frame-1-only, free = neither frame
    for q in win.flank_outer_q:
        assert q < win.inner_start_qpos and a1[q] != -1 and a2[q] == -1
    for q in win.flank_free_q:
        assert q < win.inner_start_qpos and a1[q] == -1 and a2[q] == -1
    # overlap positions are dual-coding (the rarely-designable inner CDS)
    for q in win.overlap_q:
        assert a1[q] != -1 and a2[q] != -1

    # nested inner gene: there IS an outer-synonymous flank to design into
    assert win.flank_outer_q, f"{arr.name} off={offset}: no outer-coding flank found"


def main() -> None:
    for arr in (Arrangement.PLUS_ONE, Arrangement.PLUS_TWO):
        for offset in (14, 18, 22):
            _check_sense(arr, offset)

    # antisense inner gene is out of scope for v1 -> explicit refusal, not silent wrong answer
    anti = build_toy_design(Arrangement.MINUS_ONE, 14, seed=14)
    try:
        rbs_window(anti)
        raise AssertionError("antisense arrangement should raise NotImplementedError")
    except NotImplementedError:
        pass

    print("OK  sense windows map to outer-only upstream codons; ATG start located; antisense rejected")


if __name__ == "__main__":
    main()
