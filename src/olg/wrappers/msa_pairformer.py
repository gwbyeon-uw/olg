from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from olg.constants import Constants
from olg.config import ProteinConfig

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import aa2tok_d, tok2aa_d, prepare_msa_masks

from .base_wrapper import BaseWrapper


class WrapperMSAPairformer(BaseWrapper):
    """Wrapper for MSA Pairformer sequence decoder.

    MSA Pairformer is a lightweight MSA transformer that produces per-position
    amino acid logits from multiple sequence alignments. It also provides
    optional contact predictions (Cb-Cb and ConFind).

    The model expects one-hot encoded MSA input [1, N, L, 28] and outputs
    logits of shape [1, L, 26] (20 standard AAs + X, B, Z, U, O, gap).

    Two representations are maintained in sync via edit_S():
      - S_msa_tokens: token indices [1, N, seq_len] (native 28-token space)
      - S_msa_onehot: one-hot [1, N, seq_len, 28] (model input)

    Coordinate spaces:
      - seq_len = model_len + pad_n + pad_c — OLG-visible length
      - model_len = msa_max_length — what the model sees (no padding)
      - All tracking tensors (S, decoded_positions, logit_weight, etc.) are
        in OLG space (seq_len). Only the forward pass trims input to model_len
        and pads output back to seq_len via _forward_pass.

    Padding semantics:
      - Padded positions (the first pad_n and last pad_c positions of S) are
        NOT passed to the model. They get zero logits and must be steered via
        config.logit_weight=0 + config.aa_bias / config.logit_bias.
      - config.length must equal model_len + pad_n + pad_c.
      - config.start_offset should typically be set to pad_n if you want
        get_prot_seq() to skip the N-terminal padding region.
      - config.gap_positions are interpreted as 1-based positions in OLG space
        (the full padded sequence). Specifying gap positions inside the pad
        regions is not meaningful and should be avoided — gaps are properties
        of the model input, and the model never sees pad regions.
      - config.fixed_positions and prefixed_seq are also in OLG space.
      - get_score() auto-excludes pad positions from the score average.
    """

    # OLG stop codon 'X' maps to gap '-' (same convention as EvoDiff/GREMLIN)
    _DEFAULT_EXTRA_AA_MAP: dict[str, str] = {'X': '-'}

    # Full 28-token vocab — NOT truncated to 26. alphabet_map_rev needs entries
    # for <pad> (26) and <mask> (27) to handle undecoded positions in S.
    _NATIVE_VOCAB: dict[str, int] = dict(aa2tok_d)

    MASK_TOKEN: int = aa2tok_d['<mask>']   # 27
    PAD_TOKEN: int = aa2tok_d['<pad>']     # 26
    GAP_TOKEN: int = aa2tok_d['-']         # 25

    # Logit output dimension (20 AA + X, B, Z, U, O, gap — no <pad>/<mask>)
    LOGIT_DIM: int = 26

    def __init__(
        self,
        model: torch.nn.Module,
        msa_seqs: list[str],
        msa_max_length: int,
        msa_n_seq: int = 128,
        msa_selection_type: Literal['random', 'MaxHamming', 'MaxHammingI'] = 'random',
        prefixed_seq: list[tuple[int, int, str]] | None = None,
        extra_aa_map: dict[str, str] | None = None,
        use_bfloat16: bool = True,
        seed_from_msa: bool = False,
        pad: tuple[int, int] = (0, 0),
        **kwargs,
    ):
        """Initialize MSA Pairformer wrapper.

        Args:
            model: MSAPairformer model instance (already on device).
            msa_seqs: Pre-parsed MSA sequences (aligned, uppercase + '-').
            msa_n_seq: Number of sequences to subsample from MSA.
            msa_max_length: Maximum sequence length (MSA columns).
            msa_selection_type: Diversity selection strategy for MSA subsampling.
            prefixed_seq: List of (start, end, seq) tuples to pre-fill before decoding.
            extra_aa_map: Per-call override for alphabet mapping (OLG char -> native char).
            use_bfloat16: Use bfloat16 autocast for inference (recommended for GPU).
            seed_from_msa: If True, seed the query row from the MSA query sequence
                when no explicit seed_S is provided. Recommended — keeps masking ratio
                close to the 15% training distribution during Gibbs refinement.
            pad: (pad_n, pad_c) number of dummy positions to prepend/append.
                Padded positions are NOT passed to the model — the forward pass
                trims input to model_len and pads output back to seq_len. Use
                logit_weight=0 and aa_bias at pad positions to control their AAs.
            **kwargs: Passed to BaseWrapper (device, config, decoding_order, rand_base,
                tqdm_disable, alphabet).
        """
        super().__init__(**kwargs)

        self.model = model
        self.use_bfloat16 = use_bfloat16
        self.seed_from_msa = seed_from_msa
        self.pad_n, self.pad_c = pad
        self.vocab_size = len(aa2tok_d)  # 28 — needed by BaseWrapper._apply_constraints
        self._build_alphabet_maps(self._NATIVE_VOCAB, extra_aa_map, self._DEFAULT_EXTRA_AA_MAP)

        # MSA subsampling
        self.msa_seqs = msa_seqs
        self.msa_n_seq = msa_n_seq
        self.msa_max_length = msa_max_length
        self.model_len = self.msa_max_length  # what the model sees (no pad)
        self.seq_len = self.model_len + self.pad_n + self.pad_c  # OLG-visible length
        self.msa_selection_type = msa_selection_type

        # Seed numpy's global RNG from rand_base so MSA subsampling is reproducible
        # (subsample_msa uses np.random.choice for the slice window and sequence selection)
        if self.rand_base is not None:
            np.random.seed(int(self.rand_base))
        self.valid_msa_, self.query_sequence, _ = self.subsample_msa(
            self.msa_seqs,
            n_sequences=self.msa_n_seq,
            max_seq_len=self.msa_max_length,
            selection_type=self.msa_selection_type,
        )
        # Tokenize with MSA Pairformer vocab
        self.valid_msa = torch.tensor(
            np.array([self._tokenize_seq(seq) for seq in self.valid_msa_]),
            device=self.device,
        )

        self.prefixed_seq = prefixed_seq

        # Gap handling
        self.gap_positions = None
        self.gap_map = torch.arange(self.seq_len, device=self.device)
        self.gap_map_rev = self.gap_map.clone()
        if self.config.gap_positions is not None:
            self.gap_positions = torch.tensor(
                self.config.gap_positions, device=self.device
            ).sort()[0] - 1  # to 0-based
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
    def _tokenize_seq(seq: str | list[str]) -> npt.NDArray[np.int_]:
        """Tokenize a sequence string using MSA Pairformer vocabulary."""
        return np.array([aa2tok_d.get(a, aa2tok_d['X']) for a in seq])

    @staticmethod
    def subsample_msa(
        parsed_msa: list[str],
        n_sequences: int = 64,
        max_seq_len: int = 512,
        selection_type: Literal['random', 'MaxHamming', 'MaxHammingI'] = 'random',
    ) -> tuple[list[str], str, list[str]]:
        """Subsample an MSA based on diversity selection strategies.

        Adapted from EvoDiff (Microsoft), using MSA Pairformer's alphabet.

        Args:
            parsed_msa: List of aligned sequences (uppercase + '-', lowercase = insertions).
            n_sequences: Number of sequences to subsample.
            max_seq_len: Maximum sequence length (random slice if longer).
            selection_type: 'random', 'MaxHamming', or 'MaxHammingI'.

        Returns:
            (aligned_seqs, anchor_seq, unaligned_seqs) tuple.
        """
        from scipy.spatial.distance import cdist

        alpha = np.array(list(tok2aa_d.values()))  # index -> char
        gap_idx = aa2tok_d['-']
        pad_idx = aa2tok_d['<pad>']

        # Extract aligned columns only (uppercase + '-')
        aligned_msa = [
            [c for c in seq if (c.isupper() or c == '-') and c != '.']
            for seq in parsed_msa
        ]

        tokenized_msa = np.array([
            WrapperMSAPairformer._tokenize_seq(seq).tolist() for seq in aligned_msa
        ])
        msa_seq_len = len(tokenized_msa[0])

        if msa_seq_len > max_seq_len:
            slice_start = np.random.choice(msa_seq_len - max_seq_len + 1)
        else:
            slice_start = 0

        sliced_msa_seq = tokenized_msa[:, slice_start:slice_start + max_seq_len]
        anchor_seq = sliced_msa_seq[0]
        seq_len = sliced_msa_seq.shape[1]

        # Remove all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if list(set(seq)) != [gap_idx]]
        msa_num_seqs = len(sliced_msa)

        if msa_num_seqs < n_sequences:
            output = np.full((n_sequences, seq_len), fill_value=pad_idx)
            output[:msa_num_seqs] = sliced_msa
            unal = parsed_msa
            raise ValueError(
                f"MSA has {msa_num_seqs} sequences after filtering, need {n_sequences}"
            )
        elif msa_num_seqs > n_sequences:
            if selection_type == 'random':
                random_idx = np.random.choice(
                    msa_num_seqs - 1, size=n_sequences - 1, replace=False
                ) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate(
                    (anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0
                )
                unal = [parsed_msa[i] for i in random_idx]
            elif selection_type in ('MaxHamming', 'MaxHammingI'):
                unal_inds = [0]
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]

                random_ind = 0 if selection_type == 'MaxHammingI' else np.random.choice(msa_ind)

                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                unal_inds.append(random_ind)
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, random_ind - 1, axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((n_sequences - 2, m))
                msa_ind = np.delete(msa_ind, msa_ind[msa_ind == (random_ind - 1)] - 1)

                for i in range(n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    unal_inds.append(msa_ind[random_ind])
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    msa_ind = np.delete(msa_ind, random_ind)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)

                unal = [parsed_msa[i] for i in unal_inds]
        else:
            unal = parsed_msa
            output = sliced_msa

        output = [''.join(alpha[seq]) for seq in output]
        return output, output[0], unal

    @staticmethod
    def _load_model(device: torch.device, weights_dir: str | None = None) -> MSAPairformer:
        """Load MSA Pairformer model from pretrained weights.

        Args:
            device: Target device.
            weights_dir: Directory for cached weights (downloads from HuggingFace if missing).

        Returns:
            MSAPairformer model ready for inference.
        """
        model = MSAPairformer.from_pretrained(device=device, weights_dir=weights_dir)
        model.eval()
        model.requires_grad_(False)
        return model

    def _forward_pass(self) -> torch.Tensor:
        """Run model forward pass with trim/pad adaptor.

        Trims pad regions from the one-hot input before the model call,
        then pads the output logits back to OLG-space (seq_len).
        Pad positions get zero logits (no model signal).

        Returns:
            Logits tensor of shape [seq_len, LOGIT_DIM] in OLG space.
        """
        pn, ml = self.pad_n, self.model_len
        # Trim: extract model-length slice from OLG-space one-hot
        model_input = self.S_msa_onehot[:, :, pn:pn + ml, :]

        if self.use_bfloat16 and self.device.type == 'cuda':
            with torch.no_grad():
                with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                    results = self.model(
                        msa=model_input,
                        mask=self.mask,
                        msa_mask=self.msa_mask,
                        full_mask=self.full_mask,
                        pairwise_mask=self.pairwise_mask,
                        query_only=True,
                        return_contacts=False,
                    )
        else:
            with torch.no_grad():
                results = self.model(
                    msa=model_input,
                    mask=self.mask,
                    msa_mask=self.msa_mask,
                    full_mask=self.full_mask,
                    pairwise_mask=self.pairwise_mask,
                    query_only=True,
                    return_contacts=False,
                )

        # Pad: place model logits into OLG-space tensor (pad regions stay zero)
        # results['logits'] shape: [1, 1, model_len, 26]
        padded_pred = torch.zeros((self.seq_len, self.LOGIT_DIM), device=self.device)
        padded_pred[pn:pn + ml] = results['logits'][0, 0]
        return padded_pred

    def _reset_decoding_order(self, decoding_order: torch.Tensor) -> None:
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)

    def _build_msa_onehot(self) -> None:
        """Build one-hot MSA tensor and masks from S_msa_tokens.

        One-hot is built from the full OLG-space tensor. Masks are built from
        the model-length slice only (no pad tokens to confuse mask logic).
        """
        self.S_msa_onehot = F.one_hot(
            self.S_msa_tokens, num_classes=len(aa2tok_d)
        ).float()  # [1, N, seq_len, 28]

        # Masks from model-length slice (excludes pad regions)
        model_tokens = self.S_msa_tokens[:, :, self.pad_n:self.pad_n + self.model_len]
        mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(model_tokens)
        self.mask = mask.to(self.device)
        self.msa_mask = msa_mask.to(self.device)
        self.full_mask = full_mask.to(self.device)
        self.pairwise_mask = pairwise_mask.to(self.device)

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
                Must include gap tokens if gap_positions are configured.
        """
        self.rand_base = rand_base
        self._reset_decoding_order(decoding_order)

        # Store original query before masking (model_len, no pad)
        self.S_orig = self.valid_msa[0, :self.model_len].clone()

        # Build token-index MSA in OLG space (seq_len = model_len + pad_n + pad_c).
        # Pad regions get <pad> token; real region starts masked, homologs filled.
        self.S_msa_tokens = torch.full(
            (1, self.msa_n_seq, self.seq_len),
            fill_value=self.PAD_TOKEN,
            device=self.device,
            dtype=torch.long,
        )
        # Real region: query row masked, homolog rows filled
        pn, ml = self.pad_n, self.model_len
        self.S_msa_tokens[:, :, pn:pn + ml] = self.MASK_TOKEN
        self.S_msa_tokens[:, 1:self.msa_n_seq, pn:pn + ml] = (
            self.valid_msa[1:self.msa_n_seq, :ml]
        )

        # S is a view into the query row of S_msa_tokens
        self.S = self.S_msa_tokens[:, 0, :]

        # Reset tracking state
        self.next_t = 0
        self.current_pred = None
        self.current_logits = None
        self.decoded_positions = torch.zeros(self.seq_len, device=self.device).unsqueeze(0)
        self.selected_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long()
        self.selected_log_prob = torch.zeros(self.seq_len, device=self.device).unsqueeze(0)
        self.log_prob = torch.zeros((self.seq_len, self.LOGIT_DIM), device=self.device)
        self.argmax_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long()

        # Use MSA query as seed when seed_from_msa=True and no explicit seed
        if seed_S is None and self.seed_from_msa:
            # S_orig is model_len — place at offset in the real region
            self.S_msa_tokens[0, 0, pn:pn + ml] = self.S_orig
            self._build_msa_onehot()
        elif seed_S is not None:
            # External seed (e.g. from Gibbs) is full seq_len
            seed = seed_S.clone()
            if seed.dim() == 1:
                seed = seed.unsqueeze(0)
            self.S_msa_tokens[:, 0, :seed.shape[1]] = seed
            self._build_msa_onehot()
        else:
            self._build_msa_onehot()
            if self.gap_positions is not None:
                for p in self.gap_positions:
                    self.decode_next(use_t_msa=p)
                    self.update_S(S_t=self.GAP_TOKEN, use_t_msa=p, alphabet_map=False)
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq)

    def edit_S(
        self,
        t: int | torch.Tensor,
        S_t: int | torch.Tensor,
        inplace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Edit query row at position(s) t to token(s) S_t.

        Updates BOTH token-index and one-hot representations to keep them in sync.
        All mutation code paths must go through this method.

        Args:
            t: Position(s) in MSA coordinates (scalar or tensor).
            S_t: Token index(es) in native space (scalar or tensor).
            inplace: If True, modify self state. If False, return cloned copies.

        Returns:
            None if inplace, else (S_clone, S_msa_onehot_clone).
        """
        if inplace:
            S_tokens = self.S_msa_tokens
            S_onehot = self.S_msa_onehot
        else:
            S_tokens = self.S_msa_tokens.clone()
            S_onehot = self.S_msa_onehot.clone()

        # Determine if t is within valid range
        in_range = t < self.seq_len if isinstance(t, int) else (t < self.seq_len).all()
        if in_range:
            # Update token indices
            S_tokens[0, 0, t] = S_t
            # Update one-hot: zero out old, set new
            S_onehot[0, 0, t, :] = 0.0
            S_onehot[0, 0, t, S_t] = 1.0

        if not inplace:
            S_view = S_tokens[:, 0, :self.seq_len]
            return S_view, S_onehot

    def decode_next(
        self,
        dummy_run: bool = False,
        mask_current: bool = False,
        use_t_msa: int | torch.Tensor | None = None,
        use_t: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute logits for the current decoding position.

        Args:
            dummy_run: If True, return zero logits without running the model.
            mask_current: If True, mask the current position before forward pass
                (for pseudolikelihood scoring).
            use_t_msa: Override position in MSA coordinates.
            use_t: Override position in protein coordinates (no gaps).

        Returns:
            (logits_constrained, logits_raw) tuple, or None if position is invalid.
        """
        # Coordinate translation (same logic as EvoDiff)
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
                    (self.seq_len, self.LOGIT_DIM), device=self.device
                )
            else:
                if mask_current:
                    self.edit_S(t_msa, self.MASK_TOKEN, inplace=True)

                self.current_pred = self._forward_pass()

        if t > -1:
            if (use_t_msa is None) and self.config.force_stop and (t == self.end_pos):
                logits = self._force_stop()
                return logits, logits

            self.current_logits = self.current_pred[t_msa, :].unsqueeze(0)  # [1, 26]

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

    def update_S(
        self,
        S_t: int | torch.Tensor,
        use_t_msa: int | torch.Tensor | None = None,
        alphabet_map: bool = True,
        use_t: int | torch.Tensor | None = None,
        dummy_run: bool = False,
    ) -> bool:
        """Update sequence with selected amino acid at current position.

        Args:
            S_t: Selected token (OLG internal if alphabet_map=True, native otherwise).
            use_t_msa: Override position in MSA coordinates.
            alphabet_map: If True, map S_t from OLG to native space.
            use_t: Override position in protein coordinates.
            dummy_run: Unused, kept for interface compatibility.

        Returns:
            True if update succeeded, False if stop codon reached.
        """
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

    def preset_fixed_S(self, fixed_start: int, fixed_end: int, fixed_seq: str) -> None:
        """Pre-fill a region that won't be part of OLG decoding.

        Args:
            fixed_start: Start position (0-based protein coordinates).
            fixed_end: End position (inclusive, 0-based protein coordinates).
            fixed_seq: Amino acid string to fill (may include '-' for gaps).
        """
        t = torch.arange(fixed_start, fixed_end + 1, device=self.device)
        t_msa = self.gap_map[t]
        fixed_token = torch.tensor(
            [aa2tok_d.get(c, self.GAP_TOKEN) for c in fixed_seq],
            device=self.device,
        )
        self.edit_S(t_msa, fixed_token, inplace=True)
        self.decoded_positions[:, t_msa] = 1.0

    def get_score(
        self,
        S: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> float:
        """Score sequence by average negative log-likelihood.

        Args:
            S: Sequence tensor in native space [1, L]. Uses current S if None.
            positions: Subset of positions to score. Scores all if None.

        Returns:
            Average negative log probability.
        """
        if S is None:
            S = self.S.clone()
        self.reset(self.decoding_order, self.rand_base, S)
        self.decode_all(use_S=S[0], mask_current=True)
        if positions is not None:
            return (self.selected_log_prob * -1.0)[0, positions].mean()
        # Auto-exclude pad positions from score average
        pn, ml = self.pad_n, self.model_len
        return (self.selected_log_prob[0, pn:pn + ml] * -1.0).mean()

    def get_prot_seq(self, S: torch.Tensor | None = None) -> str:
        """Convert internal token sequence to protein string.

        Args:
            S: Token indices in OLG internal space. Uses current S if None.

        Returns:
            Protein sequence string.
        """
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.config.start_offset:self.seq_len]]
        prot = ''.join([self.alphabet[s.item()] for s in S])
        return prot

    def decode_all(
        self,
        temp: float = 1e-12,
        use_S: torch.Tensor | None = None,
        mask_current: bool = False,
    ) -> bool:
        """Decode all positions sequentially.

        Args:
            temp: Sampling temperature.
            use_S: Pre-determined sequence for scoring (native token space).
            mask_current: Mask each position before predicting (for pseudolikelihood).

        Returns:
            True if decoding completed, False if already started.
        """
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

    def get_contacts(self) -> dict[str, torch.Tensor]:
        """Run contact prediction on current MSA state.

        Returns:
            Dict with 'cb_contacts' and 'confind_contacts' tensors,
            each of shape [1, model_len, model_len] with values in [0, 1].
            These are in model-space (no padding).
        """
        pn, ml = self.pad_n, self.model_len
        model_input = self.S_msa_onehot[:, :, pn:pn + ml, :]

        if self.use_bfloat16 and self.device.type == 'cuda':
            with torch.no_grad():
                with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                    results = self.model.predict_contacts(
                        msa=model_input,
                        mask=self.mask,
                        msa_mask=self.msa_mask,
                        full_mask=self.full_mask,
                        pairwise_mask=self.pairwise_mask,
                    )
        else:
            with torch.no_grad():
                results = self.model.predict_contacts(
                    msa=model_input,
                    mask=self.mask,
                    msa_mask=self.msa_mask,
                    full_mask=self.full_mask,
                    pairwise_mask=self.pairwise_mask,
                )
        return {
            'cb_contacts': results['predicted_cb_contacts'],
            'confind_contacts': results['predicted_confind_contacts'],
        }
