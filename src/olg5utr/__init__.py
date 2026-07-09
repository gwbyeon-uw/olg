"""olg5utr — design a 5'UTR shared by two overlapping ORFs to maximize their translation.

Given an outer protein and a free 5' region, design the shared 5'UTR to maximize mean ribosome load
(MRL) of BOTH overlapping ORFs — scored by Optimus 5-Prime (a CNN trained on a human 5'UTR MPRA) —
keeping the outer protein fixed up to synonymous codon choice. Uses discrete search (not gradient
descent).

    from olg5utr import UTRDesign, load_optimus, optimize_utr
    model  = load_optimus("weights/optimus_mrl_multi.pth", device)
    design = UTRDesign(outer_protein="MVSKGEELFTGVVPILVELD", free_len=20, w_mrl=0.5)
    result = optimize_utr(design, model)
    print(result.best.free_utr, result.best.score.combined)
"""
from .design import UTRDesign, dna_to_onehot, reverse_codon_table, translate
from .driver import Candidate, UTRResult, optimize_utr
from .model import MODEL_INPUT_LEN, OPTIMUS_HEADS, Optimus, load_optimus
from .scorer import MRLScore, mrl_dual, score_mrl

__all__ = [
    "UTRDesign", "translate", "reverse_codon_table", "dna_to_onehot",
    "Optimus", "load_optimus", "OPTIMUS_HEADS", "MODEL_INPUT_LEN",
    "MRLScore", "score_mrl", "mrl_dual",
    "Candidate", "UTRResult", "optimize_utr",
]
