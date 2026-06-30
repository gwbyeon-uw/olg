"""OSTIR translation-initiation scorer for the inner gene of an OLG design.

Thin, pure wrapper over ``ostir.run_ostir`` pinned to a single known start codon
(the inner gene's Met), so OSTIR skips the start-codon scan and scores exactly that
RBS. Pure + cached on its NT arg so the Stage-4 driver can dedup on the fold window
(OSTIR only reads ``[start-35, start+35]``) and reuse results across candidates.

OSTIR ΔG model: rate ∝ exp(−β·dG_total), dG_total = dG_mRNA:rRNA + dG_start − dG_mRNA
− dG_standby (Salis 2009 / OSTIR). Higher ``expression`` = stronger RBS.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# E. coli 16S anti-Shine-Dalgarno (OSTIR default); exposed so callers can override.
ECOLI_ANTI_SD = "ACCUCCUUA"


@dataclass(frozen=True)
class RBSScore:
    """Core OSTIR result for one start codon. ``expression`` is the rate proxy."""
    expression: float
    dG_total: float
    dG_mRNA_rRNA: float
    dG_mRNA: float
    dG_spacing: float
    dG_standby: float
    dG_start_codon: float
    start_codon: str
    spacing_bp: int


def score_rbs(nt: str, inner_start: int, asd: str = ECOLI_ANTI_SD) -> RBSScore | None:
    """Score the RBS driving the start codon at 0-indexed ``inner_start`` in ``nt``.

    Returns None if OSTIR finds no valid start codon there (e.g. the position is not
    ATG/GTG/TTG, or it is leaderless — too close to the 5' end for an RBS).
    """
    if not 0 <= inner_start < len(nt):
        raise ValueError(f"inner_start {inner_start} out of range for length {len(nt)}")
    return _score_cached(nt.upper(), inner_start, asd)


@lru_cache(maxsize=100_000)
def _score_cached(nt: str, inner_start: int, asd: str) -> RBSScore | None:
    import warnings

    with warnings.catch_warnings():
        # OSTIR warns about a missing RNAfold *binary*; it uses the RNA module, which we have.
        warnings.filterwarnings("ignore", message=".*missing dependency ViennaRNA.*")
        from ostir import run_ostir

    s1 = inner_start + 1  # OSTIR uses 1-indexed start positions
    rows = run_ostir(nt, start=s1, end=s1, aSD=asd, threads=1)
    if not rows:
        return None
    r = rows[0]
    return RBSScore(
        expression=r["expression"],
        dG_total=r["dG_total"],
        dG_mRNA_rRNA=r["dG_rRNA:mRNA"],
        dG_mRNA=r["dG_mRNA"],
        dG_spacing=r["dG_spacing"],
        dG_standby=r["dG_standby"],
        dG_start_codon=r["dG_start_codon"],
        start_codon=r["start_codon"],
        spacing_bp=r["RBS_distance_bp"],
    )
