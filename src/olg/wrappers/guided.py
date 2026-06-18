"""Guided decoder wrapper — adds TAG (Taylor-Approximated Guidance) to any decoder.

Intercepts decode_next() to add classifier gradient signal to base logits,
biasing sampling toward sequences with desired properties (e.g., AMP activity,
low hemolysis). All other methods delegate to the inner decoder.

Usage:
    olg.initialize_decoder("MSAPairformer", frame=0, model=model, ...)
    olg.decoders[0] = GuidedWrapper(
        olg.decoders[0],
        classifiers=[amp_classifier, hemo_classifier],
        guide_temp=0.5,
        weights=[1.0, -1.0],  # maximize AMP, minimize hemolysis
    )

Based on: "Unlocking Guidance for Discrete State-Space Diffusion and Flow Models"
(Nisonoff et al., arXiv:2406.01572)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class GuidanceClassifier(Protocol):
    """Interface for classifiers used in TAG guidance.

    Must accept one-hot encoded sequences with gradients enabled and
    return a scalar log-probability of the desired property per batch element.
    """

    vocab_size: int
    """Size of the classifier's token vocabulary."""

    def log_prob(self, x_onehot: torch.Tensor, t: float) -> torch.Tensor:
        """Compute log p(desired_property | sequence).

        Args:
            x_onehot: One-hot encoded sequence [B, L, vocab_size] with
                requires_grad=True. May contain soft values at masked positions.
            t: Fraction of positions decoded (0.0 = all masked, 1.0 = complete).

        Returns:
            Log probability [B] — higher means more of the desired property.
        """
        ...

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert native decoder token indices to classifier-space indices.

        Args:
            token_ids: Token indices [B, L] in native decoder space (0-27 for
                MSA Pairformer, 0-20 for ProteinMPNN, etc.).

        Returns:
            Token indices [B, L] in classifier's vocabulary space.
        """
        ...

    @property
    def olg_to_clf(self) -> torch.Tensor:
        """Mapping from OLG alphabet index → classifier token index.

        Shape [olg_alphabet_size]. Used to remap classifier gradients
        back to OLG logit space: grad_olg = grad_clf[olg_to_clf].
        """
        ...


class StraightThroughEmbedding(torch.nn.Module):
    """Replace embedding lookup with matrix multiply for gradient flow.

    Standard embedding: argmax(onehot) → lookup → no gradient through discrete choice.
    This module: onehot @ embedding.weight → gradient flows through onehot.
    """

    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        self.weight = embedding.weight  # [vocab_size, d_model]

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """Args: x_onehot [B, L, vocab_size]. Returns: [B, L, d_model]."""
        return x_onehot @ self.weight


class GuidedWrapper:
    """Wraps any DecoderProtocol to add TAG guidance from classifiers.

    Only decode_next() is overridden — all state management, scoring, and
    sequence tracking delegates to the inner decoder via __getattr__.

    The inner decoder's current_logits (used by update_S for log-prob tracking)
    remains unguided, so get_score() returns the generative model's score.
    Classifier scores should be evaluated separately.

    Args:
        inner: Any DecoderProtocol instance (EvoDiff, MSAPairformer, etc.).
        classifiers: List of GuidanceClassifier instances.
        guide_temp: Guidance temperature. Lower = stronger guidance. 1.0 = mild.
        weights: Per-classifier weight. Use negative weight to minimize a property
            (e.g., -1.0 for hemolysis). Defaults to 1.0 for all.
        warmup: Guidance warmup factor. During early decoding (many masked positions),
            effective temperature is scaled up by (1 + warmup * (1 - frac_decoded)).
            Set to 0.0 to disable warmup.
    """

    def __init__(
        self,
        inner,
        classifiers: list[GuidanceClassifier],
        guide_temp: float = 1.0,
        weights: list[float] | None = None,
        warmup: float = 0.0,
    ):
        self._inner = inner
        self.classifiers = classifiers
        self.guide_temp = guide_temp
        self.weights = weights or [1.0] * len(classifiers)
        self.warmup = warmup

        if len(self.weights) != len(self.classifiers):
            raise ValueError(
                f"Got {len(self.weights)} weights for {len(self.classifiers)} classifiers"
            )

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails — delegates to inner
        return getattr(self._inner, name)

    def decode_next(
        self, dummy_run: bool = False, mask_current: bool = False, **kw
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute guided logits: base logits + TAG gradient / temperature."""
        base_logits, base_logits_ = self._inner.decode_next(
            dummy_run, mask_current, **kw
        )

        if dummy_run or not self.classifiers:
            return base_logits, base_logits_

        grad_olg = self._compute_tag_gradient()

        # Effective temperature with warmup
        temp = self.guide_temp
        if self.warmup > 0:
            frac = self._inner.decoded_positions.sum() / max(
                self._inner.decoded_positions.numel(), 1
            )
            temp = temp * (1.0 + self.warmup * (1.0 - frac))

        guided = base_logits + grad_olg / temp
        guided_ = base_logits_ + grad_olg / temp
        return guided, guided_

    def _compute_tag_gradient(self) -> torch.Tensor:
        """Compute TAG gradient at the current decoding position.

        For each classifier:
        1. Convert inner decoder's S to classifier's one-hot space
        2. Forward through classifier with gradient
        3. Backprop to get per-token gradient at current position
        4. Remap gradient to OLG alphabet space
        5. Weight and accumulate

        Returns:
            Gradient tensor [1, olg_alphabet_size] to add to base logits.
        """
        # Current position being decoded
        t_idx = self._inner.next_t - 1  # decode_next already advanced next_t? No.
        # Actually, decode_next doesn't advance next_t — update_S does.
        # So next_t points to the position we just decoded.
        t_pos = self._inner.decoding_order[0, self._inner.next_t]

        # Map through gap_map if it exists
        if hasattr(self._inner, 'gap_map'):
            t_msa = self._inner.gap_map[t_pos]
        else:
            t_msa = t_pos

        olg_alphabet_size = self._inner.alphabet_size
        grad_total = torch.zeros((1, olg_alphabet_size), device=self._inner.S.device)

        # Fraction decoded for time conditioning
        frac_decoded = (
            self._inner.decoded_positions.sum()
            / max(self._inner.decoded_positions.numel(), 1)
        ).item()

        for clf, w in zip(self.classifiers, self.weights):
            # Map native decoder tokens → OLG alphabet → classifier vocab.
            # inner.S is in native model space (e.g., EvoDiff 0-30).
            # alphabet_map_rev maps native → OLG (0-20), with -1 for unmapped.
            olg_tokens = self._inner.alphabet_map_rev[self._inner.S].clamp(min=0)  # [1, L]
            clf_tokens = clf.olg_to_clf[olg_tokens]  # [1, L] in clf space
            x_oh = F.one_hot(clf_tokens, clf.vocab_size).float()  # [1, L, V_clf]

            # Handle undecoded positions based on classifier type.
            decoded = self._inner.decoded_positions[0].bool()  # [L]
            if hasattr(clf, 'mask_token_idx') and clf.mask_token_idx is not None:
                # Noisy-trained classifier: use dedicated mask token.
                # (Spearman ~0.89 vs exact enumeration)
                x_oh[0, ~decoded, :] = 0.0
                x_oh[0, ~decoded, clf.mask_token_idx] = 1.0
            else:
                # Clean-trained classifier: uniform (1/V) at masked positions.
                # (Spearman ~0.73 vs exact enumeration)
                x_oh[0, ~decoded, :] = 1.0 / clf.vocab_size

            x_oh = x_oh.detach().requires_grad_(True)

            # Forward + backward
            log_p = clf.log_prob(x_oh, frac_decoded)  # [B]
            grad = torch.autograd.grad(
                log_p.sum(), x_oh, create_graph=False
            )[0]  # [1, L, V_clf]

            # Taylor approximation: grad_at_z~ - grad_at_zt
            # (following discrete_guidance Eq. 10)
            grad_centered = grad - (x_oh.detach() * grad).sum(dim=-1, keepdim=True)

            # Extract gradient at current position
            grad_t = grad_centered[0, t_msa, :]  # [V_clf]

            # Remap from classifier vocab to OLG alphabet.
            # olg_to_clf[i] = classifier token index for OLG alphabet position i.
            grad_olg = grad_t[clf.olg_to_clf]  # [olg_alphabet_size]

            grad_total[0] += w * grad_olg

        return grad_total
