"""olgrbs — RBS design for the inner gene of an overlapping-gene (OLG) design.

Consumes olg's native ``OLGDesign`` output and optimizes the inner gene's ribosome
binding site under the outer-protein-synonymous (and dual-synonymous overlap)
constraints, scored with OSTIR. See IMPLEMENTATION_PLAN.md.
"""
from .driver import Candidate, RBSResult, optimize_rbs
from .scorer import ECOLI_ANTI_SD, RBSScore, score_rbs
from .search import OptionChain, build_chain
from .window import RBSWindow, rbs_window

__all__ = [
    "ECOLI_ANTI_SD", "RBSScore", "score_rbs",
    "RBSWindow", "rbs_window",
    "OptionChain", "build_chain",
    "Candidate", "RBSResult", "optimize_rbs",
]
