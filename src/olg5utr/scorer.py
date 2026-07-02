"""Optimus MRL scoring of a dual-5'UTR window (the objective the search maximizes)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .design import UTRDesign, dna_to_onehot
from .model import MODEL_INPUT_LEN


@dataclass
class MRLScore:
    """Optimus MRL for both ORFs of one window + the weighted objective."""

    mrl_outer: float   # MRL of the outer 5'UTR (seq[:free_len])
    mrl_inner: float   # MRL of the inner 5'UTR (whole window)
    combined: float    # w_mrl*outer + (1-w_mrl)*inner  (what the search maximizes)


def mrl_dual(onehot: torch.Tensor, model, design: UTRDesign) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched dual MRL. ``onehot`` is ``(N, 4, L)`` -> (mrl_outer, mrl_inner), each ``(N,)``.

    Each 5'UTR is left-padded to the model's input length (Optimus reads a 100-nt UTR ending at the
    start codon); ``UTRDesign`` guarantees both fit.
    """
    outer = F.pad(onehot[:, :, : design.free_len], (MODEL_INPUT_LEN - design.free_len, 0))
    inner = F.pad(onehot[:, :, : design.length], (MODEL_INPUT_LEN - design.length, 0))
    mo = model(outer, final_ind=design.head).squeeze(1)
    mi = model(inner, final_ind=design.head).squeeze(1)
    return mo, mi


def combined(mo: torch.Tensor, mi: torch.Tensor, w_mrl: float) -> torch.Tensor:
    return w_mrl * mo + (1.0 - w_mrl) * mi


def score_mrl(dna: str, model, design: UTRDesign) -> MRLScore:
    """Score a single designed window (convenience / reporting)."""
    oh = dna_to_onehot([dna], design.length, next(model.parameters()).device)
    with torch.inference_mode():
        mo, mi = mrl_dual(oh, model, design)
    o, i = float(mo[0]), float(mi[0])
    return MRLScore(mrl_outer=o, mrl_inner=i, combined=design.w_mrl * o + (1 - design.w_mrl) * i)
