"""Wrapper for EvoDiff single-sequence (OA-DM) models.

Unlike WrapperEvoDiff (MSA-based), this wrapper operates on a single sequence
[1, L] without any MSA context. It wraps the ByteNetLMTime model from the
evodiff package (OA_DM_38M or OA_DM_640M checkpoints).

The model takes (sequence_tokens, timestep) and outputs per-position logits
over 31 tokens. During OLG decoding, undecoded positions are set to the mask
token (#, index 28) and timestep is set based on fraction of positions decoded.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from olg.constants import Constants
from olg.config import ProteinConfig

from evodiff.utils import Tokenizer
from evodiff.pretrained import OA_DM_38M, OA_DM_640M

from .base_wrapper import BaseWrapper


class WrapperEvoDiffSeq(BaseWrapper):
    """Wrapper for EvoDiff single-sequence OADM models.

    The model operates on a single sequence (no MSA). At each decoding step,
    masked positions hold the mask token and the model predicts per-position
    amino acid distributions conditioned on all unmasked positions.

    Coordinate spaces (with padding):
      - seq_len = model_len + pad_n + pad_c — OLG-visible length
      - model_len — what the model sees (no padding)
      - Padding is handled via a trim/pad adaptor around the forward pass,
        identical to WrapperMSAPairformer.
    """

    # OLG stop 'X' maps to gap '-' in EvoDiff vocabulary
    _DEFAULT_EXTRA_AA_MAP: dict[str, str] = {'X': '-'}

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        length: int,
        prefixed_seq: list[tuple[int, int, str]] | None = None,
        extra_aa_map: dict[str, str] | None = None,
        seed_seq: str | None = None,
        pad: tuple[int, int] = (0, 0),
        **kwargs,
    ):
        """Initialize EvoDiff single-sequence wrapper.

        Args:
            model: EvoDiff ByteNetLMTime model instance.
            tokenizer: EvoDiff Tokenizer instance.
            length: Sequence length (model input length, excluding padding).
            prefixed_seq: List of (start, end, seq) tuples to pre-fill.
            extra_aa_map: Override for alphabet mapping (OLG char -> native char).
            seed_seq: Optional seed sequence string to initialize from.
            pad: (pad_n, pad_c) N/C-terminal padding. Padded positions are not
                passed to the model.
            **kwargs: Passed to BaseWrapper.
        """
        super().__init__(**kwargs)

        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.alphabet)
        self.pad_n, self.pad_c = pad
        self.model_len = length
        self.seq_len = self.model_len + self.pad_n + self.pad_c

        self._build_alphabet_maps(
            self.tokenizer.a_to_i, extra_aa_map, self._DEFAULT_EXTRA_AA_MAP
        )

        self.prefixed_seq = prefixed_seq
        self.seed_seq = seed_seq

        # Gap handling (in OLG space)
        self.gap_positions = None
        self.gap_map = torch.arange(self.seq_len, device=self.device)
        self.gap_map_rev = self.gap_map.clone()
        if self.config.gap_positions is not None:
            self.gap_positions = torch.tensor(
                self.config.gap_positions, device=self.device
            ).sort()[0] - 1
            self.gap_map[self.gap_positions] = -1
            self.gap_map = self.gap_map[self.gap_map != -1]
            self.gap_map_rev[self.gap_positions] = -1
            self.gap_map_rev[self.gap_map_rev != -1] = torch.arange(
                self.gap_map.shape[0], device=self.device
            )

        # Fixed positions
        tmp = torch.zeros(self.seq_len, device=self.device) - 1
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos - 1] = self.alphabet_index[aa]
        self.fixed_positions = tmp.long()

        self.reset(self.decoding_order, self.rand_base)

    @staticmethod
    def _load_model(device: torch.device, size: Literal['38M', '640M'] = '38M'):
        """Load EvoDiff-seq model.

        Args:
            device: Target device.
            size: Model size ('38M' or '640M').

        Returns:
            (model, tokenizer) tuple.
        """
        loader = OA_DM_38M if size == '38M' else OA_DM_640M
        model, _, tokenizer, _ = loader()
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        return model, tokenizer

    def _reset_decoding_order(self, decoding_order: torch.Tensor) -> None:
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)

    def _forward_pass(self) -> torch.Tensor:
        """Run model with trim/pad adaptor.

        Trims pad regions, runs forward pass with timestep based on fraction
        decoded, pads output back to seq_len.

        Returns:
            Logits [seq_len, vocab_size] in OLG space.
        """
        pn, ml = self.pad_n, self.model_len

        # Trim to model-length
        model_input = self.S[:, pn:pn + ml]  # [1, model_len]

        # Timestep: fraction of model positions still masked
        n_masked = (model_input == self.tokenizer.mask_id).sum().float()
        t = (n_masked / ml).unsqueeze(0)  # [1], 1.0 = all masked, 0.0 = all decoded

        with torch.no_grad():
            logits = self.model(model_input, t)  # [1, model_len, vocab_size]

        # Pad back to OLG space
        padded = torch.zeros(
            (1, self.seq_len, self.vocab_size), device=self.device
        )
        padded[:, pn:pn + ml, :] = logits
        return padded[0]  # [seq_len, vocab_size]

    def reset(
        self,
        decoding_order: torch.Tensor,
        rand_base: float,
        seed_S: torch.Tensor | None = None,
    ) -> None:
        """Reset decoding state.

        Args:
            decoding_order: Position decoding order tensor.
            rand_base: Random seed base.
            seed_S: Optional seed sequence in native token space [L] or [1, L].
        """
        self.rand_base = rand_base
        self._reset_decoding_order(decoding_order)

        # Initialize S: pad regions get pad token, real region gets mask token
        self.S = torch.full(
            (1, self.seq_len),
            fill_value=self.tokenizer.pad_id,
            device=self.device,
            dtype=torch.long,
        )
        pn, ml = self.pad_n, self.model_len
        self.S[:, pn:pn + ml] = self.tokenizer.mask_id

        # Reset tracking
        self.next_t = 0
        self.current_pred = None
        self.current_logits = None
        self.decoded_positions = torch.zeros(self.seq_len, device=self.device).unsqueeze(0)
        self.selected_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long()
        self.selected_log_prob = torch.zeros(self.seq_len, device=self.device).unsqueeze(0)
        self.log_prob = torch.zeros((self.seq_len, self.vocab_size), device=self.device)
        self.argmax_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long()

        if seed_S is not None:
            seed = seed_S.clone()
            if seed.dim() == 1:
                seed = seed.unsqueeze(0)
            self.S[:, :seed.shape[1]] = seed
        elif self.seed_seq is not None:
            tokens = torch.tensor(
                [self.tokenizer.a_to_i.get(c, self.tokenizer.mask_id) for c in self.seed_seq],
                device=self.device,
            )
            self.S[0, pn:pn + len(tokens)] = tokens
        else:
            if self.gap_positions is not None:
                gap_token = self.tokenizer.a_to_i['-']
                for p in self.gap_positions:
                    self.decode_next(use_t_msa=p)
                    self.update_S(S_t=gap_token, use_t_msa=p, alphabet_map=False)
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq)

    def edit_S(self, t, S_t, inplace=False):
        """Edit sequence at position(s) t to token(s) S_t."""
        if inplace:
            S = self.S
        else:
            S = self.S.clone()

        in_range = t < self.seq_len if isinstance(t, int) else (t < self.seq_len).all()
        if in_range:
            S[0, t] = S_t

        if not inplace:
            return S

    def decode_next(self, dummy_run=False, mask_current=False, use_t_msa=None, use_t=None):
        """Compute logits for current decoding position."""
        # Coordinate translation
        if use_t is not None:
            t = use_t
            t_msa = self.gap_map[t]
        elif use_t_msa is None:
            t = self.decoding_order[0, self.next_t]
            if t < self.gap_map.shape[0]:
                t_msa = self.gap_map[t]
        else:
            t_msa = use_t_msa
            t = self.gap_map_rev[t_msa]

        if not (self.config.force_stop and (t == self.end_pos)):
            if dummy_run:
                self.current_pred = torch.zeros(
                    (self.seq_len, self.vocab_size), device=self.device
                )
            else:
                if mask_current:
                    self.edit_S(t_msa, self.tokenizer.mask_id, inplace=True)
                self.current_pred = self._forward_pass()

        if t > -1:
            if (use_t_msa is None) and self.config.force_stop and (t == self.end_pos):
                logits = self._force_stop()
                return logits, logits

            self.current_logits = self.current_pred[t_msa, :].unsqueeze(0)  # [1, vocab]

            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map]
                logits_[:, self.stop_index] = Constants.MIN_LOGIT
                logits = logits_.clone()
            else:
                logits_ = self.current_logits.clone()
                logits_ -= logits_.mean()
                logits_ = logits_[:, self.alphabet_map]
                logits_[:, self.stop_index] = Constants.MIN_LOGIT

                logits = logits_.clone()
                logits = self._apply_constraints(logits, t)

            if (use_t_msa is None) and (
                (not self.config.force_stop) or (t != self.end_pos)
            ):
                logits_ = self._penalize_stop(logits_)
                logits = self._penalize_stop(logits)

            if (use_t_msa is None) and self.fixed_positions[t] != -1:
                logits = self._force_fixed_positions(logits, t)

            logits = BaseWrapper._add_noise(logits)
            return logits, logits_

    def update_S(self, S_t, use_t_msa=None, alphabet_map=True, use_t=None, dummy_run=False):
        """Update sequence with selected amino acid."""
        if use_t_msa is None:
            t = self.decoding_order[:, self.next_t]
            if self.config.force_stop and (t == self.end_pos):
                self.next_t += 1
                return False
            t_msa = self.gap_map[t]
            self.next_t += 1
        elif use_t is not None:
            t_msa = self.gap_map[use_t]
        else:
            t_msa = use_t_msa

        if alphabet_map:
            S_t = self.alphabet_map[S_t]

        self.edit_S(t_msa, S_t, inplace=True)
        self.decoded_positions[:, t_msa] = 1.0
        self.selected_aa[:, t_msa] = S_t
        log_softmax = torch.log(F.softmax(self.current_logits[0], dim=-1))
        self.selected_log_prob[:, t_msa] = log_softmax[S_t]
        self.log_prob[t_msa, :] = log_softmax
        self.argmax_aa[:, t_msa] = self.current_logits[0].argmax()
        return True

    def preset_fixed_S(self, fixed_start, fixed_end, fixed_seq):
        """Pre-fill a region that won't be part of OLG decoding."""
        t = torch.arange(fixed_start, fixed_end + 1, device=self.device)
        t_msa = self.gap_map[t]
        fixed_token = torch.tensor(
            [self.tokenizer.a_to_i.get(c, self.tokenizer.a_to_i['-']) for c in fixed_seq],
            device=self.device,
        )
        self.edit_S(t_msa, fixed_token, inplace=True)
        self.decoded_positions[:, t_msa] = 1.0

    def get_score(self, S=None, positions=None):
        """Score sequence by average negative log-likelihood."""
        if S is None:
            S = self.S.clone()
        self.reset(self.decoding_order, self.rand_base, S)
        self.decode_all(use_S=S[0], mask_current=True)
        if positions is not None:
            return (self.selected_log_prob * -1.0)[0, positions].mean()
        pn, ml = self.pad_n, self.model_len
        return (self.selected_log_prob[0, pn:pn + ml] * -1.0).mean()

    def get_prot_seq(self, S=None):
        """Convert internal token sequence to protein string."""
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.config.start_offset:self.seq_len]]
        prot = ''.join([self.alphabet[s.item()] for s in S])
        return prot

    def decode_all(self, temp=1e-12, use_S=None, mask_current=False):
        """Decode all positions sequentially."""
        if not (self.next_t == 0):
            return False
        if use_S is None:
            for i in tqdm(range(self.decoding_order.shape[1]), disable=self.tqdm_disable):
                logits, logits_ = self.decode_next()
                probs = F.softmax(logits / temp, dim=-1)
                S_t = torch.multinomial(probs[0], 1)
                self.update_S(S_t)
        else:
            for i in tqdm(range(self.decoding_order.shape[1]), disable=self.tqdm_disable):
                self.decode_next(mask_current=mask_current)
                t = self.decoding_order[:, i]
                if not (self.config.force_stop and (t == self.end_pos)):
                    S_t = use_S[self.gap_map[t]]
                else:
                    S_t = None
                self.update_S(S_t, alphabet_map=False)
        return True
