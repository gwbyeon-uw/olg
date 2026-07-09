"""Stage 3 — candidate generation over the connected quartet option chain.

Only the RBS window varies; the rest of the design is held at one resolved base chain
(positions outside OSTIR's ±35 fold window are invisible to the score anyway). Within the
window each position gets an allowed-quartet set — flank expanded to outer-synonymous, free
positions open, overlap respecting the design's own multiplicity (or expanded to dual-
synonymous if opened). Candidates are **connected paths** through these sets (consecutive
quartets share a boundary nucleotide via ``Constants.QUARTETS_N``), so both proteins are
preserved by construction and the chain decodes to valid NT.

Exposes the search ladder's primitives: ``count`` / ``enumerate_nt`` / ``sample_nt`` / and
``base_path`` + ``neighbors`` for Stage-4 simulated annealing.
"""
from __future__ import annotations

from dataclasses import dataclass

from olg.compatibility import CodonCompatibility
from olg.constants import Constants

# Quartet NT-chain connectivity (built once): NEXT[q] = quartets that may follow q.
_NEXT = [set(v) for v in Constants.QUARTETS_N]
_PREV = [set(v) for v in Constants.QUARTETS_P]
_ALL_Q = set(range(len(Constants.QUARTETS)))


def decode_path(quartets: list[int]) -> str:
    """Quartet path -> nucleotide string (3 nt per quartet + the final boundary nt).

    Shared decoder (also used by orchestrator.designs) -- single source for path->NT.
    """
    return "".join(Constants.QUARTETS[q][:3] for q in quartets) + Constants.QUARTETS[quartets[-1]][3]


@dataclass
class OptionChain:
    """Per-position allowed quartets over the RBS window, spliced onto a fixed base chain."""
    positions: list[int]          # window quartet positions (absolute)
    reach: list[list[int]]        # reachable allowed quartets per position (forward-pruned)
    base_quartets: list[int]      # full resolved chain (window spliced in to make candidates)
    base_path: list[int]          # base assignment restricted to the window positions
    inner_start_nt: int
    _base_nt: str | None = None   # cached full base NT (window positions are contiguous, spliced in to_nt)

    # ---- search-ladder primitives -------------------------------------------------
    def count(self) -> int:
        """Number of distinct connected paths through the window (cheap layer DP)."""
        cnt = {q: 1 for q in self.reach[0]}
        for layer in self.reach[1:]:
            cnt = {q: sum(cnt[p] for p in _PREV[q] if p in cnt) for q in layer}
        return sum(cnt.values())

    def enumerate_nt(self):
        """Yield the NT string of every connected window path (use only when count <= cap)."""
        for path in self._enumerate_paths():
            yield self.to_nt(path)

    def sample_nt(self, k: int, seed: int = 0) -> list[str]:
        """Up to k unique NT strings via random backward walks (huge-space fallback)."""
        import random
        rng = random.Random(seed)
        last = len(self.reach) - 1
        seen: set[str] = set()
        for _ in range(k * 30):
            if len(seen) >= k:
                break
            path = [0] * len(self.reach)
            path[last] = rng.choice(self.reach[last])
            for i in range(last - 1, -1, -1):
                path[i] = rng.choice([q for q in self.reach[i] if path[i + 1] in _NEXT[q]])
            seen.add(self.to_nt(path))
        return list(seen)

    def neighbor_moves(self, path: list[int]):
        """Connected single-position moves as ``(i, q)`` pairs (one quartet swapped, still valid).

        Yields the move, not a spliced path — the SA caller materializes only the chosen one, avoiding
        a full-length path allocation per candidate neighbour.
        """
        last = len(path) - 1
        for i, options in enumerate(self.reach):
            for q in options:
                if q == path[i]:
                    continue
                if (i == 0 or path[i - 1] in _PREV[q]) and (i == last or path[i + 1] in _NEXT[q]):
                    yield i, q

    def neighbors(self, path: list[int]):
        """Connected single-position moves from ``path`` (for SA): one quartet swapped, still valid."""
        for i, q in self.neighbor_moves(path):
            yield path[:i] + [q] + path[i + 1:]

    def to_nt(self, path: list[int]) -> str:
        """Splice a window path into the base chain and decode to NT.

        The window positions are contiguous, so only the window's codons are decoded and spliced into
        a cached full base NT — avoiding a full base_quartets copy + re-decode of unchanged quartets.
        """
        if self._base_nt is None:
            self._base_nt = decode_path(self.base_quartets)
        w0, w1, n = self.positions[0], self.positions[-1] + 1, len(self.base_quartets)
        window = "".join(Constants.QUARTETS[qi][:3] for qi in path)
        if w1 < n:  # boundary NT lies outside the window -> suffix (incl. boundary) is unchanged
            return self._base_nt[:3 * w0] + window + self._base_nt[3 * w1:]
        return self._base_nt[:3 * w0] + window + Constants.QUARTETS[path[-1]][3]  # window ends the chain

    # ---- internals ----------------------------------------------------------------
    def _enumerate_paths(self):
        def rec(i, prefix):
            if i == len(self.reach):
                yield prefix
                return
            for q in self.reach[i]:
                if i == 0 or prefix[-1] in _PREV[q]:
                    yield from rec(i + 1, prefix + [q])
        yield from rec(0, [])


def build_chain(design, window, open_overlap: bool = False, seed: int = 0) -> OptionChain:
    """Build the window option chain for an ``OLGDesign`` + its ``RBSWindow``.

    Resolves the design to one base chain, then over the contiguous window span sets each
    position's allowed quartets: flank_outer -> outer-synonymous (preserve outer AA), flank_free
    -> any, overlap -> the design's own option set (locked) or dual-synonymous (opened), gaps ->
    pinned to base. Forward reachability prunes to connected options under the fixed neighbours.

    ``string_quartet`` randomly picks among synonymous quartets, so the base chain (hence any
    fixed context inside the fold window) is seeded for reproducibility.
    """
    import torch

    torch.manual_seed(seed)
    base_nt, base_quartets = design.string_quartet()
    base_quartets = list(base_quartets)
    total = len(base_quartets)
    arr = window.arrangement
    f1_off, f2_off, _ = Constants.ARRANGEMENT_CONFIG[arr]
    ctable = design.compatibility.codon_table
    crev = design.compatibility.codon_table_rev

    flank_outer = set(window.flank_outer_q)
    flank_free = set(window.flank_free_q)
    overlap = set(window.overlap_q)
    win = flank_outer | flank_free | overlap
    w0, w1 = min(win), max(win) + 1  # contiguous span; any gaps pinned to base below
    positions = list(range(w0, w1))

    def outer_aa(q):
        return ctable[base_nt[3 * q + f1_off:3 * q + f1_off + 3]]

    def inner_aa(q):
        return ctable[base_nt[3 * q + f2_off:3 * q + f2_off + 3]]

    def allowed_at(q):
        if q in flank_outer:  # mutable, preserve outer protein
            return set(CodonCompatibility.compatible_quartets_by_aa(
                arr, (None, outer_aa(q), None), (None, None, None), crev).tolist())
        if q in flank_free:   # true 5'UTR, fully free
            return set(_ALL_Q)
        if q in overlap:
            if open_overlap:  # expand to dual-synonymous (preserve both proteins)
                return set(CodonCompatibility.compatible_quartets_by_aa(
                    arr, (None, outer_aa(q), None), (None, inner_aa(q), None), crev).tolist())
            return {int(x) for x in design.quartet_list[q]}  # respect design multiplicity
        return {base_quartets[q]}  # gap inside span -> pinned

    allowed = [allowed_at(q) for q in positions]

    # clamp window ends to the fixed neighbours just outside the span
    if w0 > 0:
        allowed[0] &= _NEXT[base_quartets[w0 - 1]]
    if w1 < total:
        allowed[-1] &= _PREV[base_quartets[w1]]

    # forward reachability: keep only options connectable from the previous layer
    reach = [allowed[0]]
    for opt in allowed[1:]:
        prev = reach[-1]
        reach.append({q for q in opt if _PREV[q] & prev})
    # backward prune so every kept option also reaches the end
    for i in range(len(reach) - 2, -1, -1):
        nxt = reach[i + 1]
        reach[i] = {q for q in reach[i] if _NEXT[q] & nxt}

    reach = [sorted(r) for r in reach]
    base_path = [base_quartets[p] for p in positions]
    for i, q in enumerate(base_path):
        assert q in reach[i], "base chain must be a valid path through its own window"

    return OptionChain(positions=positions, reach=reach, base_quartets=base_quartets,
                       base_path=base_path, inner_start_nt=window.inner_start_nt)
