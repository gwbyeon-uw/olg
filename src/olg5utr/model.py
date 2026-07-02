"""Optimus 5' UTR MRL predictor — the nucleotide scorer for olg5utr.

PyTorch re-implementation of Sample et al.'s Optimus 5-Prime
(https://github.com/pjsample/human_5utr_modeling): a CNN that predicts mean ribosome load
(translation efficiency) from a one-hot 5' UTR. Shared conv backbone + one dense head per training
dataset. This is a plain scorer — NOT an olg protein decoder (no DecoderProtocol / quartets involved).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# eGFP/mCherry MPRA heads the shipped checkpoint was trained with (order fixed by the weights).
OPTIMUS_HEADS = ["egfp_unmod", "egfp_pseudo", "egfp_m1pseudo", "mcherry_unmod"]
MODEL_INPUT_LEN = 100  # Optimus reads a 100-nt 5' UTR; longer 5' UTRs are out of range for the model.


class Optimus(nn.Module):
    """CNN mapping one-hot DNA ``(B, 4, L)`` -> scalar MRL per head."""

    def __init__(
        self,
        inp_len: int = MODEL_INPUT_LEN,
        nbr_filters: int = 120,
        filter_len: int = 8,
        border_mode: int | str = "same",
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        dropout3: float = 0.2,
        nodes: int = 40,
        out_kw: list[str] | None = None,
        n_out_col: int = 1,
    ) -> None:
        super().__init__()
        out_kw = out_kw or list(OPTIMUS_HEADS)
        self.n_out = len(out_kw)
        self.out_ind = {k: v for v, k in enumerate(out_kw)}

        self.conv = nn.Sequential(
            nn.Conv1d(4, nbr_filters, filter_len, padding=border_mode),
            nn.ReLU(),
            nn.Conv1d(nbr_filters, nbr_filters, filter_len, padding=border_mode),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Conv1d(nbr_filters, nbr_filters, filter_len, padding=border_mode),
            nn.ReLU(),
            nn.Dropout(dropout2),
        )
        with torch.no_grad():
            self.conv_output_size = self.conv(torch.zeros(1, 4, inp_len)).flatten(1).shape[1]

        self.head = nn.ModuleList(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.conv_output_size, nodes),
                nn.ReLU(),
                nn.Dropout(dropout3),
                nn.Linear(nodes, n_out_col),
            )
            for _ in range(self.n_out)
        )

    def forward(self, x: torch.Tensor, final_ind: int = 0, final_kw: str | None = None) -> torch.Tensor:
        """One-hot DNA ``(B, 4, L)`` -> MRL ``(B, n_out_col)`` from the selected head."""
        x = self.conv(x)
        return self.head[self.out_ind[final_kw] if final_kw is not None else final_ind](x)


def load_optimus(state_dict_path: str, device: torch.device | None = None) -> Optimus:
    """Load the trained Optimus model (checkpoint holds a ``model_state_dict`` key), eval + frozen."""
    model = Optimus()
    ckpt = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().requires_grad_(False)
    return model.to(device) if device is not None else model
