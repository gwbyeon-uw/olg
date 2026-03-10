"""Frame balancer — heuristic weight adjustment to balance two overlapping frames."""

from __future__ import annotations

import torch


class FrameBalancer:
    """Balances weights between two reading frames based on score history.

    Tracks whether the score gap is consistently favoring one frame, and
    progressively increases the weight on the weaker frame.

    Args:
        max_weight: Maximum allowed balancing weight.
        unit: Increment step for the balancing weight.
        threshold: Minimum score difference to trigger weight escalation.
    """

    def __init__(
        self,
        max_weight: float = 2.0,
        unit: float = 0.5,
        threshold: float = 0.15,
    ):
        self.max_weight = max_weight
        self.unit = unit
        self.threshold = threshold
        self._current_weight = unit

    def reset(self) -> None:
        self._current_weight = self.unit

    def get_weights(
        self,
        scores_pll: list[list[torch.Tensor]],
        shape_f1: torch.Size,
        shape_f2: torch.Size,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-frame logit weight tensors from score history.

        Args:
            scores_pll: List of [score_f1, score_f2] pairs (most recent last).
            shape_f1: Shape for frame-1 weight tensor.
            shape_f2: Shape for frame-2 weight tensor.
            device: Torch device for output tensors.

        Returns:
            (w1, w2) weight tensors, one scaled up for the weaker frame.
        """
        sd0 = scores_pll[-1][1] - scores_pll[-1][0]

        if len(scores_pll) > 1:
            sd1 = scores_pll[-2][1] - scores_pll[-2][0]
            if (torch.sign(sd1) == torch.sign(sd0)) and (torch.abs(sd0) > self.threshold):
                self._current_weight += self.unit
            else:
                self._current_weight = self.unit
        else:
            self._current_weight = self.unit

        diff = torch.abs(sd0)
        weight = min(self.max_weight, diff * self._current_weight + 1.0)

        if scores_pll[-1][1] > scores_pll[-1][0]:
            w1 = torch.ones(shape_f1, device=device)
            w2 = torch.ones(shape_f2, device=device) * weight
        else:
            w1 = torch.ones(shape_f1, device=device) * weight
            w2 = torch.ones(shape_f2, device=device)

        return w1, w2
