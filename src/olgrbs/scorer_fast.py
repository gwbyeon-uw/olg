"""Fast reimplementation of the OSTIR RBS scorer for the single-sequence / single-start-codon case.

Purpose-built control layer for one mRNA window + one start codon, replacing OSTIR's
``OSTIRFactory``/``find_start_codons``/``_parallel_dG``/CLI/multiprocessing scaffolding with a lean
function. The thermodynamic leaf functions (``calc_dG_mRNA``/``calc_dG_mRNA_rRNA``/
``calc_dG_standby_site``) are currently reused verbatim from ``ostir`` so results are identical; they
will be swapped for fast in-house folding + O(n) base-pair bookkeeping one at a time, each re-verified
against tests/ostir_parity.py. Validated field-for-field against the OSTIR golden (PARITY PASS 427/427).
"""
from __future__ import annotations

import warnings
from functools import cache

from .scorer import ECOLI_ANTI_SD, RBSScore

# From OSTIRFactory.__init__ (ostir_factory.py): calibration/model constants for our single-start path.
_CUTOFF = 35                 # nt +/- start folded, and the 5'-proximity dangles switch
_STANDBY_LEN = 4
_DP = 4                      # run_ostir decimal_places
_NUPACK_HYB = 2.481          # hybridization-penalty offset applied to dG_mRNA_rRNA
# find_start_codons accepts only these (CTG is deliberately excluded there); energies incl. all forms.
_START_CODONS = frozenset({"ATG", "AUG", "GTG", "GUG", "TTG", "UUG"})
_START_ENERGY = {"ATG": -1.194, "AUG": -1.194, "GTG": -0.0748, "GUG": -0.0748,
                 "TTG": -0.0435, "UUG": -0.0435, "CTG": -0.03406, "CUG": -0.03406}


def _leaves():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing dependency ViennaRNA.*")
        from ostir.ostir_calculations import (
            calc_dG_mRNA_rRNA, calc_dG_standby_site, calc_expression_level)
    return calc_dG_mRNA_rRNA, calc_dG_standby_site, calc_expression_level


@cache
def _params(dangles: str):
    """OSTIR's exact rna2004 parameter object, reused so folding is bit-identical."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing dependency ViennaRNA.*")
        from ostir.ViennaRNA import get_paramater_object, vienna_constants
    return get_paramater_object(vienna_constants.material, 37.0, dangles)


def _mfe_energy(seq: str, dangles: str) -> float:
    """MFE of a single strand, matching ostir.ViennaRNA.mfe (energy rounded to 2 dp)."""
    import RNA
    _, energy = RNA.fold_compound(seq.upper().replace("T", "U"), _params(dangles)).mfe()
    return round(energy, 2)


def _dG_mRNA(mRNA: str, start_pos: int, dangles: str) -> float:
    """dG of mRNA folding = MFE of the [-cutoff, +cutoff] window (calc_dG_mRNA, structure unused here)."""
    window = mRNA[max(0, start_pos - _CUTOFF):min(len(mRNA), start_pos + _CUTOFF)]
    return _mfe_energy(window, dangles)


def score_rbs_fast(nt: str, inner_start: int, asd: str = ECOLI_ANTI_SD) -> RBSScore | None:
    """Score the RBS at 0-indexed ``inner_start``; None if OSTIR would report no valid RBS there.

    Drop-in for olgrbs.scorer.score_rbs. Reproduces run_ostir(nt, start=inner_start+1,
    end=inner_start+1, aSD=asd, threads=1)[0], or None when that returns [].
    """
    if not 0 <= inner_start < len(nt):
        raise ValueError(f"inner_start {inner_start} out of range for length {len(nt)}")
    rRNA = asd.upper().replace("T", "U")
    if len(rRNA) != 9:
        return None  # run_ostir rejects a non-9-mer aSD
    mRNA = nt.upper().replace(" ", "")
    seq_len = len(mRNA)

    # find_start_codons for start_range = [inner_start+1, inner_start+1] (1-indexed): a single position,
    # clamped to seq_len-2, yielding it only if it is a start codon.
    start_pos = min(inner_start, seq_len - 2)
    codon = mRNA[start_pos:start_pos + 3]
    if codon.upper() not in _START_CODONS:
        return None

    calc_dG_mRNA_rRNA, calc_dG_standby_site, calc_expression_level = _leaves()
    constraints = None
    dangles = "none" if start_pos > _CUTOFF else "all"   # _parallel_dG auto_dangles
    dG_start = _START_ENERGY[codon.upper()]

    dG_mRNA = _dG_mRNA(mRNA, start_pos, dangles)

    try:
        withspacing, rr_struct, spacing_value = calc_dG_mRNA_rRNA(mRNA, rRNA, start_pos, dangles, constraints)
    except ValueError as e:
        if "leaderless start codon" in str(e):
            return None
        raise
    if not withspacing:  # calc_dG_mRNA_rRNA returned None (subopt found no binding site)
        return None

    withspacing -= _NUPACK_HYB
    nospacing = rr_struct["dG_mRNA_rRNA"] - _NUPACK_HYB
    dG_standby, _corrected = calc_dG_standby_site(rr_struct, dangles, _STANDBY_LEN, constraints, rRNA)

    dG_total = withspacing + dG_start - dG_mRNA - dG_standby
    return RBSScore(
        expression=round(calc_expression_level(dG_total), _DP),
        dG_total=round(float(dG_total), _DP),
        dG_mRNA_rRNA=round(float(nospacing), _DP),
        dG_mRNA=round(float(dG_mRNA), _DP),
        dG_spacing=round(float(rr_struct["dG_spacing"]), _DP),
        dG_standby=round(float(dG_standby), _DP),
        dG_start_codon=round(float(dG_start), _DP),
        start_codon=codon,
        spacing_bp=int(spacing_value),
    )
