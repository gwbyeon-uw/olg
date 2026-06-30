"""Stage 4 — driver + report: optimize the inner gene's RBS via the search ladder.

Picks the search by the (cheap) reachable-path count: enumerate exactly when tractable,
else simulated annealing + Metropolis over quartet moves. Candidates are deduped on the
70-nt OSTIR fold window (changes outside it are invisible) and scored once each.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import median

from olg.constants import Constants

from .scorer import ECOLI_ANTI_SD, RBSScore, score_rbs
from .search import build_chain
from .window import rbs_window

# Trim margin for dedup. Must exceed OSTIR's cutoff (35) on BOTH counts: it captures the full
# ±35 fold window, AND keeps the local start index > 35 so OSTIR's auto-dangles branch
# (none if start>35 else all) matches the full construct. ==35 would flip dangles and inflate scores.
_FOLD_MARGIN = 40


@dataclass
class Candidate:
    nt: str
    score: RBSScore
    mutations: list[tuple[int, str, str]]  # (outer residue idx, base codon, new codon)


@dataclass
class RBSResult:
    method: str
    n_paths: int
    n_scored: int
    base_expression: float
    candidates: list[Candidate] = field(default_factory=list)  # ranked best-first

    @property
    def best(self) -> Candidate | None:
        """Top-ranked candidate, or None if the search scored nothing (e.g. degenerate flank)."""
        return self.candidates[0] if self.candidates else None

    def rate_range(self) -> tuple[float, float, float] | None:
        """(min, median, max) expression over candidates, or None if there are none."""
        if not self.candidates:
            return None
        e = [c.score.expression for c in self.candidates]
        return (min(e), median(e), max(e))

    def design_room_bits(self) -> float:
        """log2 of the number of distinct achievable fold windows scored."""
        return math.log2(self.n_scored) if self.n_scored else float("nan")


def optimize_rbs(design, *, objective="max", open_overlap=True, asd=ECOLI_ANTI_SD,
                 w_up=13, w_down=13, enumerate_cap=100_000, sa_steps=2000, sa_restarts=5,
                 seed=0, top=20) -> RBSResult:
    """Design the inner gene's RBS. ``objective`` is "max" or a target expression value.

    Returns ranked candidates (best-first by the objective), each protein-preserving by
    construction, with the OSTIR score, mutation list, and design-room stats. Defaults
    ``w_up``/``w_down`` (≥12) cover OSTIR's ±35 fold window so scores are base-independent.
    ``open_overlap=True`` (default) also samples the inner CDS's dual-synonymous freedom — free
    where dual-coding is frozen (the common case), useful where it isn't; only synonymous AMP changes.
    """
    window = rbs_window(design, w_up=w_up, w_down=w_down)
    chain = build_chain(design, window, open_overlap=open_overlap, seed=seed)
    s = chain.inner_start_nt
    lo = max(0, s - _FOLD_MARGIN)
    base_nt = chain.to_nt(chain.base_path)
    f1_off = Constants.ARRANGEMENT_CONFIG[window.arrangement][0]
    a1 = design.coords.all_to_f1.tolist()

    # maximise this scalar; "max" = expression, target = negative distance to the target rate
    if objective == "max":
        def key(sc: RBSScore) -> float:
            return sc.expression
    else:
        target = float(objective)

        def key(sc: RBSScore) -> float:
            return -abs(sc.expression - target)

    fold_cache: dict[str, RBSScore | None] = {}

    def score_nt(nt: str) -> RBSScore | None:
        fold = nt[lo:s + _FOLD_MARGIN + 3]  # the only slice OSTIR reads; dedup key
        if fold not in fold_cache:
            fold_cache[fold] = score_rbs(fold, s - lo, asd)
        return fold_cache[fold]

    n_paths = chain.count()
    if n_paths <= enumerate_cap:
        method = "enumerate"
        nts = chain.enumerate_nt()
    else:
        method = "anneal"
        nts = _anneal(chain, score_nt, key, sa_steps, sa_restarts, seed)

    best_by_fold: dict[str, tuple[str, RBSScore]] = {}
    for nt in nts:
        sc = score_nt(nt)
        if sc is None:
            continue
        fold = nt[lo:s + _FOLD_MARGIN + 3]
        if fold not in best_by_fold:
            best_by_fold[fold] = (nt, sc)

    scored = sorted(best_by_fold.values(), key=lambda ns: key(ns[1]), reverse=True)
    base_sc = score_nt(base_nt)
    candidates = [Candidate(nt=nt, score=sc, mutations=_mutations(base_nt, nt, window, f1_off, a1))
                  for nt, sc in scored[:top]]
    return RBSResult(method=method, n_paths=n_paths, n_scored=len(best_by_fold),
                     base_expression=base_sc.expression if base_sc else float("nan"),
                     candidates=candidates)


def _anneal(chain, score_nt, key, steps, restarts, seed):
    """Metropolis SA over single-quartet moves; yields every NT visited (driver dedups)."""
    rng = random.Random(seed)
    for r in range(restarts):
        path = list(chain.base_path)
        cur_nt = chain.to_nt(path)
        cur = score_nt(cur_nt)
        cur_val = key(cur) if cur else -math.inf
        yield cur_nt
        for step in range(steps):
            nbrs = list(chain.neighbors(path))
            if not nbrs:
                break
            cand_path = rng.choice(nbrs)
            cand_nt = chain.to_nt(cand_path)
            sc = score_nt(cand_nt)
            if sc is None:
                continue
            yield cand_nt
            val = key(sc)
            temp = max(1e-6, 1.0 - step / steps)  # linear cool; key units are rate a.u.
            if val >= cur_val or rng.random() < math.exp((val - cur_val) / (temp * (abs(cur_val) + 1.0))):
                path, cur_val = cand_path, val


def _mutations(base_nt, nt, window, f1_off, a1):
    """Outer-frame codon changes vs the base chain (AA preserved -> synonymous). Covers the flank
    AND the overlap (the latter changes only when open_overlap lets it co-design), so reporting is
    complete in both modes. Sorted by outer residue index."""
    muts = []
    for q in (*window.flank_outer_q, *window.overlap_q):  # both carry an outer residue (a1[q] >= 0)
        bc = base_nt[3 * q + f1_off:3 * q + f1_off + 3]
        cc = nt[3 * q + f1_off:3 * q + f1_off + 3]
        if bc != cc:
            muts.append((a1[q], bc, cc))
    return sorted(muts)
