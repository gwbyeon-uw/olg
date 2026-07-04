"""Fast reimplementation of the OSTIR RBS scorer for the single-sequence / single-start-codon case.

Same ViennaRNA folding engine (via the ``RNA`` C bindings and OSTIR's exact rna2004 parameter object,
so thermodynamics are identical), but a lean control layer purpose-built for one mRNA window + one start
codon — dropping OSTIR's factory/CLI/multiprocessing scaffolding and the per-fold O(n^2) dot-bracket
bookkeeping. Validated field-for-field against the OSTIR golden (tests/ostir_parity.py).

STATUS: incremental port. Ported and parity-verified so far: dG_mRNA, dG_start_codon, start_codon.
Not yet ported (sentinel values below): dG_mRNA_rRNA, dG_spacing, dG_standby, spacing_bp, and hence
dG_total / expression. Do NOT use in production until the parity validator reports PARITY PASS.
"""
from __future__ import annotations

import math
from functools import cache

from .scorer import ECOLI_ANTI_SD, RBSScore

# OSTIR calibration constants (ostir_factory.OSTIRFactory.__init__)
_BETA = 0.40002512
_RT_EFF = 1.0 / _BETA
_K = math.exp(7.279194329)
_CUTOFF = 35                       # nt +/- start codon folded
_TEMP = 37.0
_START_ENERGY = {"ATG": -1.194, "GTG": -0.0748, "TTG": -0.0435, "CTG": -0.03406}
_START_CODONS = frozenset(_START_ENERGY)

_SENTINEL = float("nan")           # marks a term not yet ported


@cache
def _params(dangles: str):
    """OSTIR's exact rna2004 parameter object (reused verbatim so folding is bit-identical)."""
    from ostir.ViennaRNA import get_paramater_object, vienna_constants
    return get_paramater_object(vienna_constants.material, _TEMP, dangles)


def _mfe_energy(seq: str, dangles: str) -> float:
    """MFE of a single strand, matching OSTIR's ViennaRNA.mfe (rounded to 2 dp)."""
    import RNA
    fc = RNA.fold_compound(seq.upper().replace("T", "U"), _params(dangles))
    _, energy = fc.mfe()
    return round(energy, 2)


def score_rbs_fast(nt: str, inner_start: int, asd: str = ECOLI_ANTI_SD) -> RBSScore | None:
    """Score the RBS at 0-indexed ``inner_start``; None if that position is not a start codon.

    Mirrors olgrbs.scorer.score_rbs. See module docstring for port status.
    """
    if not 0 <= inner_start < len(nt):
        raise ValueError(f"inner_start {inner_start} out of range for length {len(nt)}")
    nt = nt.upper()
    codon = nt[inner_start:inner_start + 3]
    if codon not in _START_CODONS or inner_start + 3 > len(nt):
        return None  # OSTIR's find_start_codons yields nothing -> run_ostir returns []

    # dangles switch: "all" (2) within cutoff of the 5' end, else "none" (0)  [factory:230-236]
    dangles = "none" if inner_start > _CUTOFF else "all"
    dG_start = _START_ENERGY[codon]

    # dG_mRNA: MFE of the [-cutoff, +cutoff] window around the start  [calc_dG_mRNA + cutoff_mRNA]
    window = nt[max(0, inner_start - _CUTOFF):min(len(nt), inner_start + _CUTOFF)]
    dG_mRNA = _mfe_energy(window, dangles)

    # TODO(port): dG_mRNA_rRNA (subopt binding-site selection), dG_spacing, dG_standby, spacing_bp.
    return RBSScore(
        expression=_SENTINEL,
        dG_total=_SENTINEL,
        dG_mRNA_rRNA=_SENTINEL,
        dG_mRNA=dG_mRNA,
        dG_spacing=_SENTINEL,
        dG_standby=_SENTINEL,
        dG_start_codon=dG_start,
        start_codon=codon,
        spacing_bp=-1,
    )


def _expression(dG_total: float) -> float:
    return _K * math.exp(-dG_total / _RT_EFF)
