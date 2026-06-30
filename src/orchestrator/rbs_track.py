"""RBS-potential pre-screen track (orchestrator stage).

Per (arrangement, offset): a rough UPPER BOUND on the inner-gene RBS strength achievable from the
outer protein's synonymous flank, so a campaign can drop offsets whose flank can't build a decent
Shine-Dalgarno before real design. Feasible representative designs come from designs.sample_designs
(the workflow's own gate); only the flank is optimized (olgrbs, open_overlap=False). Max over a few
representative designs = optimistic best-case-overlap ceiling (the overlap can only occlude the SD).

Generic over the campaign: the caller supplies the gene, codon table, lock, alphabet, and grid.
"""
from __future__ import annotations

import warnings

from olgrbs import optimize_rbs

from .designs import sample_designs

DEFAULT_OPT = dict(open_overlap=False, enumerate_cap=2000, sa_steps=600, sa_restarts=2)


def region_ceiling(arrangement, offset, lengths, gene, lock0, lk, free_sets, gene_aas, amp_aas,
                   stops, codon_table, alphabet, *, nrep=2, pad=13, seed=None, opt=None, device=None):
    """(feasible, rbs_min, rbs_median, rbs_ceiling, length) for one (arr, offset).

    Tries `lengths` until a feasible design exists (offset is feasible if ANY length works), then
    flank-optimizes the nrep representative designs. Reports the achievable range from the best
    representative's scored synonymous candidates (min/median/ceiling) -- no arbitrary single base.
    `seed` (default = offset) is threaded to sampling and optimization for reproducibility.
    """
    opt = opt or DEFAULT_OPT
    s = offset if seed is None else seed
    for length in lengths:
        designs = sample_designs(arrangement, offset, length, gene, lock0, lk, free_sets, gene_aas,
                                 amp_aas, stops, codon_table, alphabet, k=nrep, pad=pad, seed=s,
                                 device=device)
        if not designs:
            continue
        # guard: a degenerate flank (e.g. offset too close to the 5' end) can score no candidates.
        # seed per design index (s+i) so each representative's SA explores independently.
        scored = [r for r in (optimize_rbs(olg, seed=s + i, **opt) for i, olg in enumerate(designs))
                  if r.candidates]
        if not scored:
            continue
        best_res = max(scored, key=lambda r: r.best.score.expression)  # optimistic best-case overlap
        lo, med, hi = best_res.rate_range()  # achievable range over its synonymous flank candidates
        return True, lo, med, hi, length
    return False, None, None, None, None


def add_ceilings(results, lengths, gene, lock0, lookups, free_sets, gene_aas, amp_aas, stops,
                 codon_table, alphabet, *, pctile=None, nrep=2, pad=13, w_up=0, opt=None,
                 device=None, progress=None):
    """Enrich a Results in place with rbs_base/ceiling/pctile, then return it.

    Per feasible (arrangement, offset) with offset >= `w_up` (a full flank exists; the flank is
    length-independent), runs region_ceiling and broadcasts across that offset's lengths.
    `pctile(x)->percentile` optionally fills rbs_pctile (e.g. vs the E. coli reference).
    """
    for arr, off in results.offsets(feasible_only=True):
        if off < w_up:
            continue
        # reuse the seed feasibility stored for this placement (reproducible); min() is
        # order-independent (seeds are consistent per offset); fall back to offset.
        seeds = {p.seed for p in results
                 if p.arrangement == arr and p.offset == off and p.seed is not None}
        seed = min(seeds) if seeds else off
        ok, lo, med, hi, _length = region_ceiling(
            arr, off, lengths, gene, lock0, lookups[arr], free_sets[arr], gene_aas, amp_aas, stops,
            codon_table, alphabet, nrep=nrep, pad=pad, seed=seed, opt=opt, device=device)
        if not ok:
            continue
        n = results.set_by_offset(arr, off, rbs_min=lo, rbs_median=med, rbs_ceiling=hi,
                                  rbs_pctile=pctile(hi) if pctile else None)
        if n == 0:  # grid misalignment: computed a ceiling for an offset with no placements
            warnings.warn(f"add_ceilings: no placements matched (arr={arr}, offset={off})", stacklevel=2)
        if progress:
            progress(arr, off, hi)
    return results


def track(arrangements, offsets, lengths, gene, lock0, lookups, free_sets, gene_aas, amp_aas,
          stops, codon_table, alphabet, *, nrep=2, pad=13, seed_of=None, opt=None, device=None,
          progress=None):
    """Run region_ceiling over the (arrangement, offset) grid; yield one dict per region.

    `lookups`/`free_sets` are {arrangement: ...} (precomputed once per arrangement by the caller).
    `seed_of(arr, off) -> seed` (default: offset) sets the reproducible seed per region.
    `progress(row)` is an optional callback for live logging.
    """
    seed_of = seed_of or (lambda arr, off: off)
    for arr in arrangements:
        for off in offsets:
            feasible, lo, med, hi, length = region_ceiling(
                arr, off, lengths, gene, lock0, lookups[arr], free_sets[arr], gene_aas, amp_aas,
                stops, codon_table, alphabet, nrep=nrep, pad=pad, seed=seed_of(arr, off), opt=opt,
                device=device)
            row = {"arrangement": arr, "offset": off, "feasible": feasible,
                   "rbs_min": lo, "rbs_median": med, "rbs_ceiling": hi, "length": length}
            if progress:
                progress(row)
            yield row
