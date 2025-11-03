from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import torch

from constants import *

@dataclass
class ProteinConfig:
    """Configuration for a single protein in the OLG design"""
    device: torch.device = torch.device("cuda:0")
    length: int = 100 # Length of the protein, only the overlap encoded region if we are not using the whole protein
    start_offset: int = 0 # For example, this would be 10 if the model input is a sequence of length 100, but we are only overlap-encoding from position 10
    force_stop: bool = False
    force_start: bool = False
    start_codons: List[str] = field(default_factory=lambda: ["ATG"]) # Can be multiple
    fixed_positions: Optional[List[Tuple[int, str]]] = None # 1-based
    gap_positions: Optional[List[int]] = None # For models that use alignments

    # Constraints and biases
    repetition_penalty: Optional[float] = 1.1 # Penalty for repeating amino acids
    repetition_penalty_window: Optional[int] = 4 # Window size for repetition penalty
    logit_weight: Optional[torch.Tensor] = torch.ones(length, device=device) # Weight vectors for each protein's logits
    logit_bias: Optional[torch.Tensor] = torch.zeros((length, Constants.ALPHABET_SIZE), device=device) # Position-specific amino acid biases
    aa_bias: Optional[torch.Tensor] = torch.zeros(Constants.ALPHABET_SIZE, device=device) # Position-invariant amino acid biases
    truncate_topp: Optional[float] = 0.0 # Top-p cutoff for individual protein logits
    max_aa_count: Optional[torch.Tensor] = torch.zeros(Constants.ALPHABET_SIZE, device=device) + Constants.MAX_LOGIT
    max_pos_count: Optional[torch.Tensor] = Constants.MAX_LOGIT
        
@dataclass
class DesignConfig:
    """Main configuration for OLG design"""
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    arrangement: Arrangement = Arrangement.PLUS_ONE
    offset: int = 0 # This is the distance between the N-terminii of the two proteins.
    protein1: ProteinConfig = field(default_factory=lambda: ProteinConfig())
    protein2: ProteinConfig = field(default_factory=lambda: ProteinConfig())
    codon_table: Union[str, Dict[str, str]] = "Standard" # NCBI table name or a dictionary of codon-AA
    decoding_mode: Optional[DecodingMode] = 1
    temperature: Optional[float] = 1.0 # logit/T, applied to the model output
    top_p: Optional[float] = 0.0 # 0.0 for greedy
    complexed: bool = False # Whether to use ProteinMPNN tied decoding
    shared: bool = False # Whether to use EvoDidff split MSA decoding
    balancer_max_weight: float = 2.0 # balancer_max_weight: Maximum weight for balancing the two frames
    balancer_unit: float = 0.5 # balancer_unit: Increment unit for balancing the two frames
    balancer_threshold: float = 0.15 # balancer_threshold: Threshold value for difference in scores for the two frames to trigger balancing weight
    rand_base: Optional[int] = None # rand_base: Random seed for reproducibility
    tqdm_disable: bool = False # tqdm_disable: Whether to disable progress bars