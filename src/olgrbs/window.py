"""Stage 2 — map an ``OLGDesign`` to the inner gene's RBS design window.

The inner gene is frame 2 (its Met = the first frame-2 residue). For a SENSE overlap
its RBS sits just 5' of that Met, in quartet positions that code ONLY the outer gene
(frame 1) — those are the mutable, outer-synonymous positions. The early inner CDS
(dual-coding) is locked by default. Purely structural: depends on coords + arrangement,
not on the emitted nucleotides.

NT geometry: residue at absolute quartet position ``q`` occupies codon
``nt[3q + frame_offset : +3]`` (``ARRANGEMENT_CONFIG``), so the inner start codon begins
at NT index ``3*inner_start_qpos + f2_offset``.
"""
from __future__ import annotations

from dataclasses import dataclass

from olg.constants import Arrangement, Constants

# Sense (positive-strand frame 2) arrangements — the only ones v1 supports.
SENSE_ARRANGEMENTS = (Arrangement.PLUS_ONE, Arrangement.PLUS_TWO)


@dataclass(frozen=True)
class RBSWindow:
    """Structural RBS design window for an OLG inner gene (sense overlap).

    Position lists are absolute quartet indices into ``design.quartet_list`` / coords:
      * ``flank_outer_q`` — upstream, frame-1-only → mutable, preserve outer protein (synonymous).
      * ``flank_free_q``  — upstream, neither frame (true 5'UTR) → fully free (rare when nested).
      * ``overlap_q``     — at/after the inner start, dual-coding → locked unless overlap opened.
    """
    arrangement: int
    inner_start_qpos: int
    inner_start_nt: int
    f2_offset: int
    flank_outer_q: tuple[int, ...]
    flank_free_q: tuple[int, ...]
    overlap_q: tuple[int, ...]


def rbs_window(design, w_up: int = 13, w_down: int = 12) -> RBSWindow:
    """Build the inner gene's RBS window from an ``OLGDesign``.

    ``w_up`` upstream / ``w_down`` downstream quartet positions (codons) bound the window;
    the default ``w_up=13`` covers ~39 nt, past OSTIR's ±35 fold window. Raises on antisense
    arrangements (frame-2 reverse strand), which v1 does not support.
    """
    arr = Arrangement(int(design.config.arrangement))
    if arr not in SENSE_ARRANGEMENTS:
        raise NotImplementedError(
            f"olgrbs v1 supports sense inner genes only ({[a.name for a in SENSE_ARRANGEMENTS]}); "
            f"got {arr.name} (antisense). Antisense needs reverse-complementing the window.")

    _, f2_off, f2_rev = Constants.ARRANGEMENT_CONFIG[int(arr)]
    assert not f2_rev, "sense arrangement must not be reverse strand"  # guarded above

    a1 = design.coords.all_to_f1.tolist()
    a2 = design.coords.all_to_f2.tolist()
    inner_start_qpos = int(design.coords.f2_to_all[0].item())  # frame-2 residue 0 = the Met
    total = len(a1)

    flank_outer, flank_free = [], []
    for q in range(max(0, inner_start_qpos - w_up), inner_start_qpos):
        if a1[q] != -1 and a2[q] == -1:
            flank_outer.append(q)
        elif a1[q] == -1 and a2[q] == -1:
            flank_free.append(q)
        # a2[q] != -1 upstream of the start shouldn't happen for a sense nested gene; skip if so.

    overlap = [q for q in range(inner_start_qpos, min(total, inner_start_qpos + w_down))
               if a1[q] != -1 and a2[q] != -1]

    return RBSWindow(
        arrangement=int(arr),
        inner_start_qpos=inner_start_qpos,
        inner_start_nt=3 * inner_start_qpos + f2_off,
        f2_offset=f2_off,
        flank_outer_q=tuple(flank_outer),
        flank_free_q=tuple(flank_free),
        overlap_q=tuple(overlap),
    )
