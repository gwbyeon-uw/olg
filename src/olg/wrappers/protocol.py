"""Decoder protocol — documents what OLGDesign expects from any decoder wrapper.

This is a structural subtyping contract (typing.Protocol). Wrappers satisfy it
implicitly by implementing the required attributes and methods — no inheritance
needed. Use for type annotations and as living documentation for anyone adding
a new model wrapper.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DecoderProtocol(Protocol):
    """Structural interface for OLG decoder wrappers.

    Every decoder used by OLGDesign must expose these attributes and methods.
    BaseWrapper (and its subclasses) already satisfy this protocol.
    """

    S: torch.Tensor
    gap_map_rev: torch.Tensor
    logit_weight: torch.Tensor
    log_prob: torch.Tensor
    selected_log_prob: torch.Tensor

    def decode_next(
        self, dummy_run: bool = False, mask_current: bool = False, use_t: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def update_S(
        self, S_t: torch.Tensor, use_t: int | None = None, alphabet_map: bool = True, dummy_run: bool = False
    ) -> bool:
        # MSA-family wrappers (evodiff, evodiff_seq, gremlin) insert a leading `use_t_msa`
        # arg before `use_t`; `dummy_run` is always last. Call with keywords to be safe.
        ...

    def edit_S(
        self, t: int, S_t: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor | None: ...

    def reset(
        self, decoding_order: torch.Tensor, rand_base: float, seed_S: torch.Tensor | None = None, **kwargs
    ) -> None: ...

    def get_score(
        self, S: torch.Tensor | None = None, positions: torch.Tensor | None = None
    ) -> float: ...

    def get_prot_seq(self, S: torch.Tensor | None = None) -> str: ...

    def get_tied_positions(self) -> list[int]: ...

    def _reset_decoding_order(self, decoding_order: torch.Tensor) -> None: ...
