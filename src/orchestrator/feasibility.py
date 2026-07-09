"""Quartet-compatibility & reachability for OLG co-design feasibility.

Given an arrangement + codon table, precomputes which quartets encode each (essential_aa, overlap_aa)
pair / overlap codon for cheap start/stop/AA checks, and samples co-design-realizable sequences
(and their quartet paths, via sample_codesign_paths) by reachability over the quartet NT chain.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from olg.constants import Constants


def _codon_from_quartet(quartet: str, frame: int) -> str:
    if frame == 0:  # reference
        return quartet[:3]
    if frame == 1:  # alt (+1 shift)
        return quartet[1:]
    if frame == 2:  # alt neg (RC of +1)
        return Constants._reverse_complement(quartet[1:])
    if frame == 3:  # neg (RC of ref)
        return Constants._reverse_complement(quartet[:3])
    raise ValueError(f"Invalid frame index: {frame}")


@dataclass
class QuartetLookup:
    pair_quartets: dict[tuple[str, str], set[int]]
    f2_codon_quartets: dict[tuple[str, str], set[int]]
    aa1_quartets: dict[str, set[int]]  # all quartets encoding aa1 in frame f1 (any f2)
    q_to_aa2: list[str]                # frame-2 AA for each quartet index (decode sampled paths)


def build_quartet_lookup(arrangement: int, codon_table: dict[str, str]) -> QuartetLookup:
    """Precompute quartet compatibility for an arrangement + codon table."""
    f1_frame = Constants.FRAME_F1[arrangement]
    f2_frame = Constants.FRAME_F2[arrangement]
    pair_quartets: dict[tuple[str, str], set[int]] = defaultdict(set)
    f2_codon_quartets: dict[tuple[str, str], set[int]] = defaultdict(set)
    aa1_quartets: dict[str, set[int]] = defaultdict(set)
    q_to_aa2 = ["X"] * len(Constants.QUARTETS)
    for q_i, q in enumerate(Constants.QUARTETS):
        aa1 = codon_table.get(_codon_from_quartet(q, f1_frame), "X")
        f2_codon = _codon_from_quartet(q, f2_frame)
        aa2 = codon_table.get(f2_codon, "X")
        pair_quartets[(aa1, aa2)].add(q_i)
        f2_codon_quartets[(aa1, f2_codon)].add(q_i)
        aa1_quartets[aa1].add(q_i)
        q_to_aa2[q_i] = aa2
    return QuartetLookup(dict(pair_quartets), dict(f2_codon_quartets),
                         dict(aa1_quartets), q_to_aa2)


# quartet connectivity in the NT chain
_NEXT_Q = [set(v) for v in Constants.QUARTETS_N]           # q -> quartets that may follow q
_PREV_Q = [set() for _ in _NEXT_Q]                          # q' -> quartets that may precede q'
for _q, _nxts in enumerate(_NEXT_Q):
    for _q2 in _nxts:
        _PREV_Q[_q2].add(_q)


def _reach(valids):
    """Forward reachability over connected quartets: reach[i] = quartets at window layer i
    reachable from layer 0 along the NT-chain edges. Returns [list,...] or None if a layer is
    empty (no valid path). Shared by all samplers."""
    reach = [valids[0]]
    for nxt in valids[1:]:
        if not reach[-1]:
            return None
        reach.append(nxt & set().union(*(_NEXT_Q[q] for q in reach[-1])))
    return [list(r) for r in reach] if reach[-1] else None


def _window_valids(lookup: QuartetLookup, gene: str, offset: int, length: int,
                   amp_aas: str, allowed_stops, pad: int):
    """Per-position valid-quartet sets over [offset-pad, offset+length+pad], and the window
    index of the Met. Position roles: gene-only flank, forced Met (ATG), AMP interior, stop."""
    n = len(gene)
    # stop sits at offset+length; if it fell outside the gene it would be silently skipped
    # (feasible without a stop = false positive). Callers bound offset to keep it in-gene.
    assert offset + length < n, "stop position offset+length must be < len(gene)"
    w0, w1 = max(0, offset - pad), min(n, offset + length + pad)
    valids: list[set[int]] = []
    for pos in range(w0, w1):
        if pos < offset or pos > offset + length:
            v = lookup.aa1_quartets.get(gene[pos], set())
        elif pos == offset:
            v = lookup.f2_codon_quartets.get((gene[pos], "ATG"), set())
        elif pos == offset + length:
            v = set().union(*(lookup.f2_codon_quartets.get((gene[pos], s), set()) for s in allowed_stops))
        else:
            v = set().union(*(lookup.pair_quartets.get((gene[pos], a), set()) for a in amp_aas))
        valids.append(v)
    return valids, offset - w0


def sample_sequences(lookup: QuartetLookup, gene: str, offset: int, length: int,
                     amp_aas: str, allowed_stops, k: int, *, pad: int = 3, seed: int = 0,
                     tries_factor: int = 30) -> list[str]:
    """Randomly sample up to `k` UNIQUE valid AMP sequences compatible with the fixed gene.

    Forward reachability gives reach[i] (quartets reachable from the Met-start); a backward
    random walk then samples valid quartet paths -- pick a reachable quartet at the stop, then
    at each earlier position a random reachable quartet that connects to the chosen next one
    (guaranteed non-empty since the next was reached). Each path -> an AMP AA sequence (Met +
    interior; stop excluded). Caps at the space size if it has < k unique sequences.
    """
    valids, met_idx = _window_valids(lookup, gene, offset, length, amp_aas, allowed_stops, pad)
    reach = _reach(valids)
    if reach is None:
        return []
    return list(_walk(reach, met_idx, length, lookup.q_to_aa2, k, seed, tries_factor))


def sample_compatible(lookup: QuartetLookup, gene: str, offset: int, length: int,
                      amp_aas: str, k: int, *, pad: int = 1, seed: int = 0,
                      tries_factor: int = 30, force: dict[int, str] | None = None) -> list[str]:
    """Sample up to `k` UNIQUE AMP-frame AA sequences compatible with the fixed gene, with NO
    start/stop/serine constraint -- EVERY window position is a free compatible residue (the only
    constraint is that the AMP-frame codons spell the fixed gene in frame 1). Same reachability
    backward-walk as sample_sequences; returns [] if no compatible path. For the relaxed screen.

    `force`: {gene_pos (0-indexed): amp_aa} pins the AMP residue at those positions (restricts the
    valid quartets to ones encoding that AMP AA over the fixed gene AA). Returns [] if infeasible.
    """
    force = force or {}
    n = len(gene)
    w0, w1 = max(0, offset - pad), min(n, offset + length + pad)
    valids = []
    for pos in range(w0, w1):
        if pos in force:                             # pinned AMP residue
            v = lookup.pair_quartets.get((gene[pos], force[pos]), set())
        elif offset <= pos < offset + length:        # free compatible AMP residue
            v = set().union(*(lookup.pair_quartets.get((gene[pos], a), set()) for a in amp_aas))
        else:                                         # gene-only flank (connection to fixed gene)
            v = lookup.aa1_quartets.get(gene[pos], set())
        valids.append(v)
    amp_start = offset - w0
    reach = _reach(valids)
    if reach is None:
        return []
    return list(_walk(reach, amp_start, length, lookup.q_to_aa2, k, seed, tries_factor))


def build_free_gene_sets(lookup: QuartetLookup, gene_aas: str, amp_aas: str,
                         codons: list[str]) -> dict:
    """Precompute frame2-keyed quartet sets over FREE gene AAs (frame1 in gene_aas, any of them) --
    used by sample_codesign for positions where the gene may mutate. Compute once per arrangement.
      aa[a]      : quartets with frame2 = AMP AA a   (gene free)
      any        : quartets with frame2 = any AMP AA (gene free)
      codon[c]   : quartets with frame2 = codon c    (gene free)  -- for Met (ATG) / stops
      flank      : quartets with frame1 = any gene AA (gene free, no AMP)  -- gene-only flank
    """
    aa = {a: set().union(*(lookup.pair_quartets.get((g, a), set()) for g in gene_aas))
          for a in amp_aas}
    codon = {c: set().union(*(lookup.f2_codon_quartets.get((g, c), set()) for g in gene_aas))
             for c in codons}
    return {"aa": aa, "any": set().union(*aa.values()) if aa else set(), "codon": codon,
            "flank": set().union(*(lookup.aa1_quartets.get(g, set()) for g in gene_aas))}


def _codesign_reach(lookup, gene, offset, length, lock0, gene_aas, amp_aas, free_sets,
                    allowed_stops, forced_amp, force_start, force_stop, pad):
    """Per-window valid-quartet sets + forward reachability for the co-design constraints.

    Returns (reach, w0, amp0): reach[i] = connectable quartets at window position w0+i,
    amp0 = offset - w0 (the Met index). None if no valid path exists (infeasible).
    """
    forced_amp = forced_amp or {}
    n = len(gene)
    # pad must be >= 2: the window is [offset-pad, offset+length+pad), so pad=1 STOPS at the stop
    # codon and never includes the downstream flank (offset+length+1) -> the stop's connection to the
    # fixed WT flank goes unchecked -> false positives the decoder later rejects. (start side needs
    # only pad>=1.) sample_sequences already defaults pad=3; match it.
    w0, w1 = max(0, offset - pad), min(n, offset + length + pad)
    valids = []
    for pos in range(w0, w1):
        locked, g = pos in lock0, gene[pos]
        if pos < offset or pos > offset + length:                      # gene-only flank
            # the gene is FIXED to WT outside the overlap window (codesign only frees the window
            # minus lock), so the flank must use the fixed WT residue -- NOT free_sets["flank"].
            # Freeing it over-reports feasibility: the forced start/stop then "connects" to a flank
            # residue the decoder (which holds the flank at WT) can never realize -> false positive.
            v = lookup.aa1_quartets.get(g, set())
        elif pos == offset and force_start:                            # Met
            v = lookup.f2_codon_quartets.get((g, "ATG"), set()) if locked else free_sets["codon"]["ATG"]
        elif pos == offset + length and force_stop:                    # stop
            v = (set().union(*(lookup.f2_codon_quartets.get((g, s), set()) for s in allowed_stops))
                 if locked else set().union(*(free_sets["codon"][s] for s in allowed_stops)))
        elif pos in forced_amp:                                        # pinned AMP residue
            a = forced_amp[pos]
            v = lookup.pair_quartets.get((g, a), set()) if locked else free_sets["aa"][a]
        else:                                                          # free AMP residue
            v = (set().union(*(lookup.pair_quartets.get((g, a), set()) for a in amp_aas))
                 if locked else free_sets["any"])
        valids.append(v)

    reach = _reach(valids)
    return (reach, w0, offset - w0) if reach is not None else None


def _walk(reach, amp0, length, q_to_aa2, k, seed, tries_factor):
    """Backward random walk: up to k unique connected paths, keyed/deduped by decoded AA seq.
    Returns dict {amp_aa_seq: path}. Shared by sample_codesign (-> AA) and *_paths (-> path)."""
    import random as _random
    rng = _random.Random(seed)
    out: dict[str, list[int]] = {}
    tries, last = 0, len(reach) - 1
    while len(out) < k and tries < k * tries_factor:
        tries += 1
        path = [0] * len(reach)
        path[last] = rng.choice(reach[last])
        for i in range(last - 1, -1, -1):
            path[i] = rng.choice([q for q in reach[i] if path[i + 1] in _NEXT_Q[q]])
        key = "".join(q_to_aa2[path[amp0 + j]] for j in range(length))  # Met + interior
        out.setdefault(key, list(path))
    return out


def sample_codesign(lookup: QuartetLookup, gene: str, offset: int, length: int, lock0: set[int],
                    gene_aas: str, amp_aas: str, free_sets: dict, k: int, *,
                    allowed_stops=("TAA", "TGA"), forced_amp: dict[int, str] | None = None,
                    force_start: bool = True, force_stop: bool = True,
                    pad: int = 3, seed: int = 0, tries_factor: int = 30) -> list[str]:
    """Sample up to k AMP sequences REALIZABLE in the co-design, so a non-empty result GUARANTEES
    a valid construct exists with the fixed requirements. Per window position the valid quartets
    respect both sides: gene LOCKED -> fixed WT (gene[pos]); gene FREE -> any of gene_aas; AMP is
    Met @offset (force_start), a stop @offset+length (force_stop), a pinned AA at forced_amp
    positions, else free (amp_aas). `free_sets` from build_free_gene_sets (per arrangement)."""
    r = _codesign_reach(lookup, gene, offset, length, lock0, gene_aas, amp_aas, free_sets,
                        allowed_stops, forced_amp, force_start, force_stop, pad)
    if r is None:
        return []
    reach, _w0, amp0 = r
    return list(_walk(reach, amp0, length, lookup.q_to_aa2, k, seed, tries_factor))


def sample_codesign_paths(lookup: QuartetLookup, gene: str, offset: int, length: int,
                          lock0: set[int], gene_aas: str, amp_aas: str, free_sets: dict, k: int, *,
                          allowed_stops=("TAA", "TGA"), forced_amp: dict[int, str] | None = None,
                          force_start: bool = True, force_stop: bool = True,
                          pad: int = 3, seed: int = 0,
                          tries_factor: int = 30) -> list[tuple[list[int], int]]:
    """Like sample_codesign but returns up to k unique (quartet_path, w0): the full both-frame
    realization over window [w0, w0+len(path)), for reconstructing an OLGDesign. Same dedup-by-AA
    as sample_codesign, so the sample set matches at a given seed. Use pad >= the flank you need
    (e.g. pad=13 to cover an upstream RBS flank)."""
    r = _codesign_reach(lookup, gene, offset, length, lock0, gene_aas, amp_aas, free_sets,
                        allowed_stops, forced_amp, force_start, force_stop, pad)
    if r is None:
        return []
    reach, w0, amp0 = r
    return [(p, w0) for p in _walk(reach, amp0, length, lookup.q_to_aa2, k, seed, tries_factor).values()]


def seq_entropy(seqs: list[str]) -> tuple[float, float]:
    """Per-position Shannon entropy (bits) over a list of equal-length sequences.

    Returns (mean_bits_per_position, total_bits). Total ~= log2 of the available sequence-space
    size (= how much design room the fixed gene leaves at this placement); mean is per-residue.
    """
    import math
    from collections import Counter
    if not seqs:
        return float("nan"), float("nan")  # design room undefined for an empty set (intentional NaN)
    length = len(seqs[0])
    per_pos = []
    for i in range(length):
        counts = Counter(s[i] for s in seqs)
        tot = sum(counts.values())
        per_pos.append(-sum((m / tot) * math.log2(m / tot) for m in counts.values()))
    return sum(per_pos) / length, sum(per_pos)
