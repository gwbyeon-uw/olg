"""optimize_utr — discrete search over the synonymous + free-5'UTR space, scored by Optimus MRL.

Mirrors olgrbs's driver shape (entry point + Candidate/Result dataclasses). Batched single-move greedy
(near-greedy Metropolis with tau>0), N sequences in parallel, one Optimus forward per step. Every move
is protein-preserving by construction (free region = any base; outer CDS = synonymous codon swap), so
no protein loss / differentiable translator is needed. See analysis/gd_vs_sampling_pilot for why
discrete beats gradient descent here (+22% MRL).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from .design import NUCLEOTIDES, UTRDesign, dna_to_onehot
from .scorer import MRLScore, combined, mrl_dual, score_mrl


@dataclass
class Candidate:
    """One designed window."""

    dna: str                 # full designed window (free 5'UTR + outer CDS)
    score: MRLScore
    free_utr: str            # the designed free 5'UTR (seq[:free_len]) — the orderable insert
    outer_cds: str           # the synonymous outer CDS (seq[free_len:])


@dataclass
class UTRResult:
    """Ranked designs (best-first) + run metadata."""

    method: str              # "greedy" or "anneal"
    n_evaluated: int         # sequences scored (n_parallel * steps)
    base: MRLScore           # a random valid start, for reference
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


def _random_valid(design: UTRDesign, syn: list[list[str]], rng: random.Random) -> str:
    free = "".join(rng.choice(NUCLEOTIDES) for _ in range(design.free_len))
    cds = "".join(rng.choice(opts) for opts in syn)
    return free + cds


def _propose(dna: str, design: UTRDesign, syn: list[list[str]], rng: random.Random) -> str:
    """One protein-preserving single move: free-region base change, or synonymous codon swap."""
    if rng.random() * design.length < design.free_len:                 # free 5'UTR: change a base
        p = rng.randrange(design.free_len)
        return dna[:p] + rng.choice([n for n in NUCLEOTIDES if n != dna[p]]) + dna[p + 1:]
    ci = rng.randrange(len(design.outer_protein))                       # outer CDS: swap to a synonym
    s = design.free_len + 3 * ci
    alts = [c for c in syn[ci] if c != dna[s:s + 3]]
    if not alts:                                                        # e.g. Met/Trp: no synonym
        return dna
    return dna[:s] + rng.choice(alts) + dna[s + 3:]


def optimize_utr(
    design: UTRDesign,
    model,
    *,
    n_parallel: int = 50,
    steps: int = 4000,
    tau: float = 0.0,
    seed: int = 0,
    top: int = 20,
    device: torch.device | None = None,
) -> UTRResult:
    """Maximize dual Optimus MRL over the design's synonymous + free-5'UTR space.

    Args:
        design: the dual-5'UTR problem (outer protein, free-region length, weights, codon table).
        model: a loaded Optimus model (`model.load_optimus`).
        n_parallel: independent search trajectories run in one batch.
        steps: single-move steps per trajectory.
        tau: acceptance temperature; ``0`` = greedy hill-climb, ``>0`` = Metropolis (escapes optima).
        seed: RNG seed (deterministic).
        top: number of ranked candidates to return.

    Returns:
        UTRResult with candidates ranked best-first by combined MRL.
    """
    device = device or next(model.parameters()).device
    rng = random.Random(seed)
    syn = design.synonymous_codons()

    cur = [_random_valid(design, syn, rng) for _ in range(n_parallel)]
    base = score_mrl(cur[0], model, design)

    def score_batch(seqs: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            mo, mi = mrl_dual(dna_to_onehot(seqs, design.length, device), model, design)
        return combined(mo, mi, design.w_mrl)

    cur_c = score_batch(cur)
    for _ in range(steps):
        prop = [_propose(cur[b], design, syn, rng) for b in range(n_parallel)]
        prop_c = score_batch(prop)
        deltas = (prop_c - cur_c).flatten().tolist()   # single GPU->CPU sync (was one float() per candidate)
        # rng.random() is drawn in candidate order exactly as before (only when d<=0 and tau>0).
        accept = [d > 0 or (tau > 0 and math.exp(d / tau) > rng.random()) for d in deltas]  # greedy / Metropolis
        for b in range(n_parallel):
            if accept[b]:
                cur[b] = prop[b]
        accept_t = torch.tensor(accept, device=cur_c.device).reshape(cur_c.shape)
        cur_c = torch.where(accept_t, prop_c, cur_c)

    order = sorted(range(n_parallel), key=lambda b: float(cur_c[b]), reverse=True)[:top]
    candidates = [
        Candidate(dna=cur[b], score=score_mrl(cur[b], model, design),
                  free_utr=cur[b][: design.free_len], outer_cds=cur[b][design.free_len:])
        for b in order
    ]
    return UTRResult(method="greedy" if tau == 0 else "anneal",
                     n_evaluated=n_parallel * steps, base=base, candidates=candidates)
