"""Feasibility screen stage: the workflow's co-design gate over a placement grid, as a reusable
artifact -- the single source of truth for "what's designable where".

Per placement -> {arrangement, offset, length, feasible, seed}. `feasible` is the gate
(sample_codesign non-empty); `seed` is the draw seed so any consumer reproduces the same samples
deterministically. Consumers (metric scoring, RBS track, codesign) filter on `feasible` and
resample via `seed` instead of recomputing the gate. Model-free; generic over the campaign
(caller supplies gene, codon-derived lookups, lock, alphabets, grid).
"""
from __future__ import annotations

from .feasibility import sample_codesign
from .results import Results


def screen(placements, lk, gene, lock0, gene_aas, amp_aas, free_sets, stops, *,
           seed_of=None, progress=None) -> Results:
    """Run the feasibility gate over the grid into a Results (feasible + seed per placement).

    The pipeline's first stage: produces the Results object subsequent stages enrich in-memory.
    """
    r = Results()
    for row in feasibility_grid(placements, lk, gene, lock0, gene_aas, amp_aas, free_sets, stops,
                                seed_of=seed_of, progress=progress):
        r.upsert(row["arrangement"], row["offset"], row["length"],
                 feasible=row["feasible"], seed=row["seed"])
    return r


def feasibility_grid(placements, lk, gene, lock0, gene_aas, amp_aas, free_sets, stops, *,
                     seed_of=None, progress=None):
    """Yield {arrangement, offset, length, feasible, seed} for each (arr, offset, length).

    `lk`/`free_sets` are {arrangement: ...}. `seed_of(arr, off, length) -> seed` (default: offset,
    matching the screen's gate seed). `progress(row)` is an optional live-logging callback.
    """
    seed_of = seed_of or (lambda arr, off, length: off)
    for arr, off, length in placements:
        seed = seed_of(arr, off, length)
        feasible = bool(sample_codesign(lk[arr], gene, off, length, lock0, gene_aas, amp_aas,
                                        free_sets[arr], 1, allowed_stops=stops, seed=seed))
        row = {"arrangement": arr, "offset": off, "length": length, "feasible": feasible, "seed": seed}
        if progress:
            progress(row)
        yield row
