"""Shared toy-design builder for olgrbs self-checks.

Builds a real ``OLGDesign`` with ZeroOrder decoders (uniform logits, no model weights)
so the inner gene (frame 2) has a forced ATG start + stop nested inside a longer outer
gene (frame 1). Returns the decoded design; ``string_quartet()`` materialises NT.
"""
from __future__ import annotations

import torch

from olg import OLGDesign
from olg.config import DesignConfig, ProteinConfig
from olg.constants import Arrangement, Constants

DEVICE = torch.device("cpu")


def _zeroorder_logits() -> torch.Tensor:
    """Uniform logits over the default alphabet with the stop token suppressed."""
    alphabet = list(Constants.DEFAULT_ALPHABET)
    logits = torch.zeros((1, len(alphabet)), device=DEVICE)
    logits[0, alphabet.index("X")] = Constants.MIN_LOGIT  # no stop inside the body
    return logits


def build_toy_design(arrangement: Arrangement, offset: int, l1: int = 40, l2: int = 12,
                     seed: int = 0, retry: int = 40):
    """Decode a nested inner gene (frame 2, forced Met-start + stop) inside outer gene (frame 1)."""
    cfg = DesignConfig(
        device=DEVICE, arrangement=arrangement, offset=offset, rand_base=seed, tqdm_disable=True,
        protein1=ProteinConfig(device=DEVICE, length=l1),
        protein2=ProteinConfig(device=DEVICE, length=l2, force_start=True, force_stop=True),
    )
    olg = OLGDesign(cfg)
    zo = _zeroorder_logits()
    olg.initialize_decoder("ZeroOrder", frame=0, model=zo)
    olg.initialize_decoder("ZeroOrder", frame=1, model=zo)
    olg.decode_all(dummy_run=(False, False), mask_current=(False, False),
                   force_safe=False, retry=retry)
    return olg
