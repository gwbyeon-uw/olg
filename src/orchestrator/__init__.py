"""orchestrator — OLG design operations & pipelines (feasibility, screening, RBS tracks).

Top layer over olg (model engine) + olgrbs (RBS design). The user-facing entry point is
`OLGCampaign` (config-driven; inject per-frame objective plug-ins).
"""
from .campaign import CampaignConfig, FrameObjective, OLGCampaign, SequenceScorer

__all__ = ["CampaignConfig", "FrameObjective", "OLGCampaign", "SequenceScorer"]
