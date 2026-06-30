#!/usr/bin/env python
"""Stage-4 self-check for olgrbs.driver: optimize_rbs returns protein-preserving, ranked
candidates; enumerate finds the brute-force optimum (and fold-window trimming is score-
identical to the full construct); target mode and the SA fallback both work.

Brute-force comparison uses a SMALL window with the driver's seed (so the base chain matches);
a separate smoke run exercises the coverage defaults.

Run:  python tests/olgrbs_driver.py
"""
from __future__ import annotations

from olgrbs_toy import build_toy_design

from olg.constants import Arrangement
from olgrbs import optimize_rbs, score_rbs
from olgrbs.search import build_chain
from olgrbs.window import rbs_window

W_UP, W_DOWN = 6, 3  # small -> enumerable -> brute-forceable


def proteins(design, nt):
    return design.translate_sequences(nt_seq=nt)


def main() -> None:
    design = build_toy_design(Arrangement.PLUS_ONE, 18, l1=60, l2=20, seed=18)

    # open_overlap=False here so the brute chain (build_chain default) matches exactly
    res = optimize_rbs(design, w_up=W_UP, w_down=W_DOWN, open_overlap=False)  # seed=0 default
    assert res.method == "enumerate" and res.candidates, "no candidates"

    # rebuild the SAME base chain (matched seed) for the brute-force comparison
    chain = build_chain(design, rbs_window(design, w_up=W_UP, w_down=W_DOWN), seed=0)
    s = chain.inner_start_nt
    base_nt = chain.to_nt(chain.base_path)
    base_prot = proteins(design, base_nt)
    ctable = design.compatibility.codon_table

    # every ranked candidate preserves both proteins; mutations are genuinely synonymous
    for c in res.candidates:
        assert proteins(design, c.nt) == base_prot, "candidate changed a protein"
        for _idx, bc, cc in c.mutations:
            assert ctable[bc] == ctable[cc], f"non-synonymous outer mutation {bc}->{cc}"

    # best >= base (base is a valid path, so enumerate dominates it)
    assert res.best.score.expression >= res.base_expression

    # enumerate == brute force, scoring the FULL construct (validates dedup + trim==full)
    brute = max(score_rbs(nt, s).expression for nt in chain.enumerate_nt())
    assert abs(res.best.score.expression - brute) < 1e-6, (
        f"driver best {res.best.score.expression} != brute(full) {brute}")

    # target mode: pick the candidate closest to a target rate
    target = 0.5 * (res.rate_range()[0] + res.rate_range()[2])
    res_t = optimize_rbs(design, objective=target, w_up=W_UP, w_down=W_DOWN, open_overlap=False)
    brute_t = min(chain.enumerate_nt(), key=lambda nt: abs(score_rbs(nt, s).expression - target))
    assert abs(res_t.best.score.expression - score_rbs(brute_t, s).expression) < 1e-6

    # SA fallback (forced with a tiny cap): protein-preserving, >= base
    res_sa = optimize_rbs(design, w_up=W_UP, w_down=W_DOWN, enumerate_cap=5,
                          sa_steps=300, sa_restarts=3)
    assert res_sa.method == "anneal" and res_sa.candidates
    for c in res_sa.candidates:
        assert proteins(design, c.nt) == base_prot
    assert res_sa.best.score.expression >= res_sa.base_expression

    # coverage defaults (w_up=w_down=13 cover ±35): just smoke that it runs + preserves proteins
    smoke = optimize_rbs(design, sa_steps=150, sa_restarts=1)
    assert smoke.candidates and all(proteins(design, c.nt) == base_prot for c in smoke.candidates)

    lo, med, hi = res.rate_range()
    print(f"OK  enumerate: {res.n_scored} folds / {res.n_paths} paths | base={res.base_expression:.1f} "
          f"best={res.best.score.expression:.1f} range=[{lo:.1f},{hi:.1f}] "
          f"room={res.design_room_bits():.1f} bits | brute✓ target✓ SA={res_sa.best.score.expression:.1f} "
          f"| defaults={smoke.method}({smoke.n_scored})")


if __name__ == "__main__":
    main()
