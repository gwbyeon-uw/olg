import copy
import math
import numpy as np
import torch
from tqdm import tqdm

import constants
import utils

from wrappers.proteinmpnn import *
from wrappers.evodiff import *
from wrappers.gremlin import *
from wrappers.protmamba import *
from wrappers.coflow import *
from wrappers.esm3 import *

#Main container for designing two overlapping proteins simultaneously with overlap constraints.
class OLGDesign():
    def __init__(
        self,
        device: torch.device,
        arrangement: int = 0,
        offset: int = 0,
        decoding_mode: int = 0,
        seq_lens: Tuple[int, int] = (None, None),
        seq_starts: Tuple[int, int] = (0, 0),
        codon_table: Union[str, Dict[str, str]] = "Standard",
        top_p: float = 0.0,
        temperature: float = 1.0,
        end_stop: Tuple[bool, bool] = (False, False),
        start: Tuple[bool, bool] = (False, False),
        start_codons: Tuple[List[str], List[str]] = (["ATG"], ["ATG"]),
        fixed_positions: Optional[List[Tuple[int, int, str]]] = None,
        gap_positions: Tuple[Optional[List[int]], Optional[List[int]]] = (None, None),
        repetition_penalty: Tuple[float, float] = (1.1, 1.1),
        repetition_penalty_window: Tuple[int, int] = (4, 4),
        logit_weight: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        logit_bias: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        aa_bias: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        truncate_topp: Tuple[Optional[float], Optional[float]] = (None, None),
        max_aa_count: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        max_pos_count: Tuple[Optional[float], Optional[float]] = (None, None),
        complexed: bool = False,
        shared: bool = False,
        balancer_max_weight: float = 2.0,
        balancer_unit: float = 0.5,
        balancer_threshold: float = 0.15,
        rand_base: Optional[int] = None,
        tqdm_disable: bool = False
    ):
        """        
        Args:
            device: torch.device
            arrangement: Overlap arrangement (0:+1, 1:-1, 2:-0, 3:+2, 4:-2)
            offset: Offset between protein start positions (must be positive)
            decoding_mode: Decoding strategy (0:random, 1:overlap first, 2:overlap last)
            seq_lens: Lengths of the two proteins
            seq_starts: Starting offsets relative to each protein
            codon_table: Genetic code table name or dictionary mapping codons to amino acids (e.g., {"ATG": "M", "TAA": "*"})
            top_p: Top-p cutoff for joint logit matrix sampling
            temperature: Temperature for sampling from joint logit matrix
            end_stop: Whether to add stop codons for each protein
            start: Whether to force start codons for each protein
            start_codons: List of allowed start codons for each protein
            fixed_positions: List of (frame, position, amino_acid) to fix
            gap_positions: Gap positions relative to MSA input
            repetition_penalty: Penalty for repeating amino acids
            repetition_penalty_window: Window size for repetition penalty
            logit_weight: Weight vectors for each protein's logits
            logit_bias: Position-specific amino acid biases
            aa_bias: Position-invariant amino acid biases
            truncate_topp: Top-p cutoff for individual protein logits
            max_aa_count: Maximum count constraints for each amino acid
            max_pos_count: Maximum count for positively charged residues
            complexed: Whether to use ProteinMPNN tied decoding
            shared: Whether to use EvoDidff split MSA decoding
            balancer_max_weight: Maximum weight for balancing the two frames
            balancer_unit: Increment unit for balancing the two frames
            balancer_threshold: Threshold value for difference in scores for the two frames to trigger balancing weight
            rand_base: Random seed for reproducibility
            tqdm_disable: Whether to disable progress bars
        """
        self.device = device
        self.arrangement = arrangement #0:+1, 1:-1, 2:-0, 3:+2, 4:-2; For 3 and 4, offset should be used!
        self.f2_neg = arrangement in [1, 2, 4] #Whether frame 2 is negative strand
        self.offset = offset #Offset between START positions of the two proteins. It must be a positive number (first protein always starts at 0). 
        
        if isinstance(codon_table, dict):
            self.codon_table = codon_table
        else:
            self.codon_table = unambiguous_dna_by_name[codon_table].forward_table #From Biopython NCBI codes 
            for stop_codon in unambiguous_dna_by_name["Standard"].stop_codons:
                self.codon_table[stop_codon] = "X"
        self.codon_table_rev = reverse_codon_table(self.codon_table)

        self.codon_to_aa = torch.zeros((4, 4, 4), device=self.device).long()
        self.codon_to_aa_rc = torch.zeros((4, 4, 4), device=self.device).long()
        for i in [ 0, 1, 2, 3 ]:
            for j in [ 0, 1, 2, 3 ]:
                for k in [ 0, 1, 2, 3 ]:
                    codon = NUCLEOTIDES[i] + NUCLEOTIDES[j] + NUCLEOTIDES[k]
                    codon_rc = reverse_complement(codon)
                    aa = self.codon_table[codon]
                    aa_rc = self.codon_table[codon_rc]
                    self.codon_to_aa[i, j, k] = ALPHABET.index(aa)
                    self.codon_to_aa_rc[i, j, k] = ALPHABET.index(aa_rc)
        
        self.end_stop = end_stop #Enable to add stop codon after each protein
        self.start = start
        self.start_codons = start_codons
        self.top_p = top_p #Top-p cutoff for the joint logit matrix of the amino acid pairs; can set this to very low value for greedy decoding
        self.temperature = temperature #Temperature for sampling from joint logit matrix
        self.decoding_mode = decoding_mode #0:full random, 1:overlap first, 2:overlap last
        self.seq_lens = seq_lens #Lengths of proteins
        self.seq_starts = seq_starts #Starting offsets, relative to each protein

        self.repetition_penalty = repetition_penalty #Repetition penalty; neighboring amino acids that have been already selected are penalized by multiplying/dividing the logits by this amount; 1.2 is common
        self.repetition_penalty_window = repetition_penalty_window
        self.logit_weight = logit_weight #Weight applied to the logit vector for each protein; used to favor one frame over the other
        self.logit_bias = logit_bias #Position-specific bias to favor/penalize some amino acids over others at specific positions
        self.aa_bias = aa_bias #Position-invariant bias to favor/penalize some amino acids over others
        self.fixed_positions_ = fixed_positions #List of tuples: (1-based position, amino acid letter). Beware that there is no check for validity of fixed positions in terms of their compatibility and behavior in such case is not tested.
        self.gap_positions = gap_positions #Put gaps at these positions; relative to and relevant only in the MSA input. These positions do not exist except in the MSA input.
        self.max_aa_count = max_aa_count #Maximum number of each amino acids allowed; sampling an AA is disallowed after reaching this maximum
        self.max_pos_count = max_pos_count #Maximum number of positive AA's allowed; sampling H/K/R is disallowed after reaching this maximum
        self.truncate_topp = truncate_topp #Top-p cutoff for the logit vector for each protein
        self.complexed = complexed #For ProteinMPNN tied decoding
        self.shared = shared #For EvoDidff split MSA decoding
        self.balancer_max_weight = balancer_max_weight #Maximum weight for balancing the two frames
        self.balancer_unit = balancer_unit #Increment unit for balancing the two frames
        self.balancer_threshold = balancer_threshold #Threshold value for difference in scores for the two frames to trigger balancing weight
        self.rand_base = rand_base
        self.tqdm_disable = tqdm_disable
        
        self.codon_compatibility, self.quartets_aa = generate_compatibility_matrix(self.device, self.codon_table)
        
        #First/last nucleotide for each of the 256 quartets
        self.prev_quartet_index = torch.tensor(PREV_QUARTET_INDEX).long()
        self.next_quartet_index = torch.tensor(NEXT_QUARTET_INDEX).long()
        
        #Dictionary that stores compatible next/previous quartets given the current quartet
        self.compatible_prev_quartets = [ torch.tensor(v).unique() for v in QUARTETS_P ]
        self.compatible_next_quartets = [ torch.tensor(v).unique() for v in QUARTETS_N ]

        self.f1_gap_len = self.seq_lens[0] - len(self.gap_positions[0]) if self.gap_positions[0] is not None else self.seq_lens[0]
        self.f2_gap_len = self.seq_lens[1] - len(self.gap_positions[1]) if self.gap_positions[1] is not None else self.seq_lens[1]
        
        #Prepare coordinates for relative positions of the proteins; protein 1 always starts at 0 and is always positive strand
        f1_start_ = 0 #Protein 1 always starts at 0 position
        f2_strand = -1 if self.f2_neg else 1 #-1 indicates negative strand overlap
        f2_start_ = min(self.offset, self.offset + f2_strand * (self.f2_gap_len + self.end_stop[1])) #+1 to make room for stop codon
        all_offset = -1 * min(f2_start_, 0)
        self.f1_start = f1_start_ + all_offset
        self.f1_end = self.f1_start + self.f1_gap_len + self.end_stop[0]
        self.f2_start = f2_start_ + all_offset
        self.f2_end = self.f2_start + self.f2_gap_len + self.end_stop[1]
        self.total_len = max(self.f1_end, self.f2_end)
        
        f1_range = range(self.f1_start, self.f1_end)
        f2_range = range(self.f2_start, self.f2_end)
        overlap_range = range(max(f1_range[0], f2_range[0]), min(f1_range[-1], f2_range[-1])+1) #Intersection
        overlap_start = overlap_range[0]
        overlap_end = overlap_range[-1] + 1
        
        #To help set decoding order so that start codons gets priority
        self.start_mask_all = torch.zeros(self.total_len, device=self.device)
        if self.start[0]:
            f1_start_pos = self.f1_start
            self.start_mask_all[f1_start_pos] = 10 #High value but less than stop codon mask value so that it goes second in decoding order
        if self.start[1]:
            f2_start_pos = (self.f2_end - 1) if self.f2_neg else self.f2_start
            self.start_mask_all[f2_start_pos] = 10
        
        #To help set decoding order so that stop codons gets priority
        self.end_stop_mask_all = torch.zeros(self.total_len, device=self.device)
        if self.end_stop[0]:
            f1_stop_pos = self.f1_end - 1
            self.end_stop_mask_all[f1_stop_pos] = 20 #High value so that it goes first in decoding order
        if self.end_stop[1]:
            f2_stop_pos = self.f2_start if self.f2_neg else (self.f2_end - 1)
            self.end_stop_mask_all[f2_stop_pos] = 20
            
        #To help set decoding order so that overlap region gets priority
        self.overlap_mask_all = torch.zeros(self.total_len, device=self.device)
        self.overlap_mask_all[overlap_start:overlap_end] = 1 
        
        #To help convert positions from absolute coordinates to relative coordinates (to each protein) and vice versa
        self.all_to_f1 = torch.zeros(self.total_len, device=self.device).long()
        self.all_to_f1.fill_(-1) #Positions where there is no overlap
        self.all_to_f1[self.f1_start:self.f1_end] = torch.arange(self.f1_end - self.f1_start) + self.seq_starts[0]
        self.all_to_f2 = torch.zeros(self.total_len, device=self.device).long()
        self.all_to_f2.fill_(-1)
        self.all_to_f2[self.f2_start:self.f2_end] = torch.arange(self.f2_end - self.f2_start) + self.seq_starts[1]
        if self.f2_neg: #Invert if negative
            self.all_to_f2[self.f2_start:self.f2_end] = self.all_to_f2[self.f2_end-1] - self.all_to_f2[self.f2_start:self.f2_end]
        
        self.f1_to_all = torch.stack([ (self.all_to_f1 == pos).nonzero().squeeze(0).squeeze(0) for pos in range(self.seq_starts[0],self.seq_starts[0]+self.f1_gap_len+self.end_stop[0]) ])
        self.f2_to_all = torch.stack([ (self.all_to_f2 == pos).nonzero().squeeze(0).squeeze(0) for pos in range(self.seq_starts[1],self.seq_starts[1]+self.f2_gap_len+self.end_stop[1]) ])
        
        self.f1_to_f2 = self.all_to_f2[self.f1_to_all]
        self.f2_to_f1 = self.all_to_f1[self.f2_to_all]

        self.nuc_total_len = self.total_len * 3 + 1

        self.reset_fixed_positions(self.fixed_positions_)
        
        self.codon_compatibility_start_mask = [None, None]
        self.start_codons_quartets = [[], []]
        if self.start[0]:
            for codon in self.start_codons[0]:
                self.codon_compatibility_start_mask[0] = torch.zeros(self.codon_compatibility.shape, device=self.device).int()
                for q_i in CODONS_TO_QUARTETS[CODONS.index(codon)][FRAME_F1[self.arrangement]]:
                    self.codon_compatibility_start_mask[0][:, :, :, :, :, q_i] = 1
                    self.start_codons_quartets[0] += [ q_i ]
        if self.start[1]:
            for codon in self.start_codons[1]:
                self.codon_compatibility_start_mask[1] = torch.zeros(self.codon_compatibility.shape, device=self.device).int()
                for q_i in CODONS_TO_QUARTETS[CODONS.index(codon)][FRAME_F2[self.arrangement]]:
                    self.codon_compatibility_start_mask[1][:, :, :, :, :, q_i] = 1
                    self.start_codons_quartets[1] += [ q_i ]
            
        self.decoding_orders = [None, None]
        self.decoding_orders_full = [None, None]
        self.decoders = [None, None]
        self.reset_decoding(self.rand_base)

    def reset_fixed_positions(
        self, 
        fixed_positions: Optional[List[Tuple[int, int, str]]] = None
    ) -> None:
        """        
        Args:
            fixed_positions: List of (frame, position, amino_acid) tuples to fix
        """
        if fixed_positions is not None:
            self.fixed_positions_ = fixed_positions
        self.fixed_positions = [[], []]
        self.fixed_positions_set = [ [None] * (self.all_to_f1.max()+1), [None] * (self.all_to_f2.max()+1) ]
        #To help set decoding order so that fixed positions get priority
        self.fixed_positions_mask_all = torch.zeros(self.total_len, device=self.device)
        if fixed_positions is not None:
            for i, (frame, pos, aa) in enumerate(self.fixed_positions_):
                if frame == 0:
                    self.fixed_positions_mask_all[self.f1_to_all[pos-1]] = 1.0#3e8 + i
                    self.fixed_positions[0] += [ [ pos, aa ] ]
                    self.fixed_positions_set[0][pos-1] = aa
                else:
                    self.fixed_positions_mask_all[self.f2_to_all[pos-1]] = 1.0#3e8 + i
                    self.fixed_positions[1] += [ [ pos, aa ] ]
                    self.fixed_positions_set[1][pos-1] = aa
        if len(self.fixed_positions[0]) == 0:
            self.fixed_positions[0] = None
        if len(self.fixed_positions[1]) == 0:
            self.fixed_positions[1] = None

    def reset_decoder_ProteinMPNN(
        self,
        frame: int = 0,
        model: Optional[torch.nn.Module] = None,
        decoder_path: Optional[str] = None,
        ca_only: bool = False,
        pdb_path: Optional[str] = None,
        fixed_chains: List[str] = [],
        design_chains: List[str] = ["A"],
        undecoded_chains: List[str] = [],
        tied: bool = False,
        tied_weight: Optional[List[Tuple[str, float]]] = None,
        fixed_chain_seq: Optional[List[Tuple[str, str]]] = None
    ) -> None:
        """
        Reset ProteinMPNN decoder for specified frame.
        
        Args:
            frame: Which protein frame (0 or 1) to reset decoder for
            model: Pre-loaded ProteinMPNN model
            decoder_path: Path to model checkpoint
            ca_only: Whether to use only CA atoms
            pdb_path: Path to PDB structure file
            fixed_chains: List of chain IDs to keep fixed
            design_chains: List of chain IDs to design
            undecoded_chains: List of chain IDs to leave undecoded
            tied: Whether to use tied decoding
            tied_weight: Weights for tied positions
            fixed_chain_seq: Fixed sequences for chains
        """
        self.decoders[frame] = ProteinMPNNContainer(
            device=self.device,
            decoder_path=decoder_path,
            model=model,
            ca_only=ca_only,
            pdb_path=pdb_path,
            fixed_chains=fixed_chains,
            design_chains=design_chains,
            undecoded_chains=undecoded_chains,
            tied=tied,
            tied_weight=tied_weight,
            fixed_positions=self.fixed_positions[frame],
            fixed_chain_seq=fixed_chain_seq,
            decoding_order=self.decoding_orders[frame],
            repetition_penalty=self.repetition_penalty[frame],
            repetition_penalty_window=self.repetition_penalty_window[frame],
            logit_weight=self.logit_weight[frame],
            logit_bias=self.logit_bias[frame],
            aa_bias=self.aa_bias[frame],
            end_stop=self.end_stop[frame],
            max_aa_count=self.max_aa_count[frame],
            max_pos_count=self.max_pos_count[frame],
            truncate_topp=self.truncate_topp[frame],
            rand_base=self.rand_base,
            tqdm_disable=self.tqdm_disable
        )
        
    def reset_decoder_EvoDiff(
        self,
        frame: int = 0,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        use_esm_msa: bool = False,
        msa_seqs: Optional[List[str]] = None,
        msa_n_seq: Optional[int] = None,
        msa_max_length: Optional[int] = None,
        msa_selection_type: Optional[str] = None,
        prefixed_seq: Optional[str] = None
    ) -> None:
        """
        Reset EvoDiff decoder for specified frame.
        
        Args:
            frame: Which protein frame (0 or 1) to reset decoder for
            model: Pre-loaded EvoDiff model
            tokenizer: Tokenizer for sequence processing
            use_esm_msa: Using ESM-MSA?
            msa_seqs: MSA sequences
            msa_n_seq: Number of sequences in MSA
            msa_max_length: Maximum MSA length
            msa_selection_type: MSA selection strategy
            prefixed_seq: Prefix sequence for conditioning
        """
        self.decoders[frame] = EvoDiffContainer(
            device=self.device,
            model=model,
            tokenizer=tokenizer,
            use_esm_msa=use_esm_msa,
            msa_seqs=msa_seqs,
            msa_n_seq=msa_n_seq,
            msa_max_length=msa_max_length,
            msa_selection_type=msa_selection_type,
            seq_start=self.seq_starts[frame],
            prefixed_seq=prefixed_seq,
            gap_positions=self.gap_positions[frame],
            fixed_positions=self.fixed_positions[frame],
            decoding_order=self.decoding_orders[frame],
            seq_len=self.seq_lens[frame],
            repetition_penalty=self.repetition_penalty[frame],
            repetition_penalty_window=self.repetition_penalty_window[frame],
            logit_weight=self.logit_weight[frame],
            logit_bias=self.logit_bias[frame],
            aa_bias=self.aa_bias[frame],
            end_stop=self.end_stop[frame],
            max_aa_count=self.max_aa_count[frame],
            max_pos_count=self.max_pos_count[frame],
            truncate_topp=self.truncate_topp[frame],
            rand_base=self.rand_base,
            tqdm_disable=self.tqdm_disable
        )
    
    def reset_decoder_GREMLIN(
        self,
        frame: int = 0,
        model: Optional[torch.nn.Module] = None,
        temperature: float = 0.1,
        prefixed_seq: Optional[str] = None
    ) -> None:
        """
        Reset GREMLIN decoder for specified frame.
        
        Args:
            frame: Which protein frame (0 or 1) to reset decoder for
            model: Pre-loaded GREMLIN model
            temperature: Sampling temperature
            prefixed_seq: Prefix sequence for conditioning
        """
        self.decoders[frame] = GREMLINContainer(
            device=self.device,
            model=model,
            temperature=temperature,
            seq_start=self.seq_starts[frame],
            prefixed_seq=prefixed_seq,
            gap_positions=self.gap_positions[frame],
            fixed_positions=self.fixed_positions[frame],
            decoding_order=self.decoding_orders[frame],
            seq_len=self.seq_lens[frame],
            repetition_penalty=self.repetition_penalty[frame],
            repetition_penalty_window=self.repetition_penalty_window[frame],
            logit_weight=self.logit_weight[frame],
            logit_bias=self.logit_bias[frame],
            aa_bias=self.aa_bias[frame],
            end_stop=self.end_stop[frame],
            max_aa_count=self.max_aa_count[frame],
            max_pos_count=self.max_pos_count[frame],
            truncate_topp=self.truncate_topp[frame],
            rand_base=self.rand_base,
            tqdm_disable=self.tqdm_disable
        )
    
    def reset_decoder_ESM3(
        self,
        frame: int = 0,
        model: Optional[torch.nn.Module] = None,
        prefixed_seq: Optional[str] = None
    ) -> None:
        """
        Reset ESM3 decoder for specified frame.
        
        Args:
            frame: Which protein frame (0 or 1) to reset decoder for
            model: Pre-loaded ESM3 model
            prefixed_seq: Prefix sequence for conditioning
        """
        self.decoders[frame] = ESM3Container(
            device=self.device,
            model=model,
            seq_start=self.seq_starts[frame],
            prefixed_seq=prefixed_seq,
            fixed_positions=self.fixed_positions[frame],
            decoding_order=self.decoding_orders[frame],
            seq_len=self.seq_lens[frame],
            repetition_penalty=self.repetition_penalty[frame],
            repetition_penalty_window=self.repetition_penalty_window[frame],
            logit_weight=self.logit_weight[frame],
            logit_bias=self.logit_bias[frame],
            aa_bias=self.aa_bias[frame],
            end_stop=self.end_stop[frame],
            max_aa_count=self.max_aa_count[frame],
            max_pos_count=self.max_pos_count[frame],
            truncate_topp=self.truncate_topp[frame],
            rand_base=self.rand_base,
            tqdm_disable=self.tqdm_disable
        )
    
    def reset_decoder_CoFlow(
        self,
        frame: int = 0,
        model: Optional[torch.nn.Module] = None,
        prefixed_seq: Optional[str] = None
    ) -> None:
        """
        Reset CoFlow decoder for specified frame.
        
        Args:
            frame: Which protein frame (0 or 1) to reset decoder for
            model: Pre-loaded CoFlow model
            prefixed_seq: Prefix sequence for conditioning
        """
        self.decoders[frame] = CoFlowContainer(
            device=self.device,
            model=model,
            seq_start=self.seq_starts[frame],
            prefixed_seq=prefixed_seq,
            fixed_positions=self.fixed_positions[frame],
            decoding_order=self.decoding_orders[frame],
            seq_len=self.seq_lens[frame],
            repetition_penalty=self.repetition_penalty[frame],
            repetition_penalty_window=self.repetition_penalty_window[frame],
            logit_weight=self.logit_weight[frame],
            logit_bias=self.logit_bias[frame],
            aa_bias=self.aa_bias[frame],
            end_stop=self.end_stop[frame],
            max_aa_count=self.max_aa_count[frame],
            max_pos_count=self.max_pos_count[frame],
            truncate_topp=self.truncate_topp[frame],
            rand_base=self.rand_base,
            tqdm_disable=self.tqdm_disable
        )
    
    def reset_decoder_ProtMamba(
        self,
        frame: int = 0,
        model: Optional[torch.nn.Module] = None,
        msa_filepath: Optional[str] = None,
        msa_n_seq: Optional[int] = None,
        msa_max_length: Optional[int] = None,
        msa_selection_type: Optional[str] = None,
        prefixed_seq: Optional[str] = None,
        shuffle_context: bool = False
    ) -> None:
        """
        Reset ProtMamba decoder for specified frame.
        
        Args:
            frame: Which protein frame (0 or 1) to reset decoder for
            model: Pre-loaded ProtMamba model
            msa_filepath: Path to MSA file
            msa_n_seq: Number of sequences in MSA
            msa_max_length: Maximum MSA length
            msa_selection_type: MSA selection strategy
            prefixed_seq: Prefix sequence for conditioning
            shuffle_context: Whether to shuffle context
        """
        self.decoders[frame] = ProtMambaContainer(
            device=self.device,
            model=model,
            msa_seqs=msa_seqs,  # Note: msa_seqs appears undefined in original
            msa_n_seq=msa_n_seq,
            msa_selection_type=msa_selection_type,
            seq_start=self.seq_starts[frame],
            prefixed_seq=prefixed_seq,
            fixed_positions=self.fixed_positions[frame],
            decoding_order=self.decoding_orders[frame],
            seq_len=self.seq_lens[frame],
            repetition_penalty=self.repetition_penalty[frame],
            repetition_penalty_window=self.repetition_penalty_window[frame],
            logit_weight=self.logit_weight[frame],
            logit_bias=self.logit_bias[frame],
            aa_bias=self.aa_bias[frame],
            end_stop=self.end_stop[frame],
            max_aa_count=self.max_aa_count[frame],
            max_pos_count=self.max_pos_count[frame],
            truncate_topp=self.truncate_topp[frame],
            rand_base=self.rand_base,
            tqdm_disable=self.tqdm_disable,
            shuffle_context=shuffle_context
        )
        
    def reset_decoding(
        self,
        rand_base: Optional[int] = None,
        user_order: Optional[torch.Tensor] = None,
        seed_S: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        seed_quartet_list: Optional[List[Optional[torch.Tensor]]] = None
    ) -> None:
        """
        Reset decoding state and reinitialize decoders for each protein.
       
        Args:
           rand_base: Random seed for reproducibility
           user_order: User-provided decoding order (overrides random seed)
           seed_S: Initial sequence tensors for each protein
           seed_quartet_list: Initial quartet list to seed decoding
        """
        self.rand_base = rand_base
        if user_order is not None:
            self.decoding_order_all = user_order #User provided order; supercedes user provided random seed. This has to be in absolute coordinates (quartet position number)
        else:
            if self.rand_base is not None:
                torch.manual_seed(self.rand_base) #User provided random number seed
            rand_order__ = torch.rand(self.total_len, device=self.device) 
            rand_order_ = rand_order__ - self.start_mask_all - self.end_stop_mask_all - self.fixed_positions_mask_all #Stop and start codons first; conflicting specifications are not checked!
            if self.decoding_mode == 0:
                rand_order = rand_order_
            elif self.decoding_mode == 1: #Priortize overlap regions if specified
                rand_order = rand_order_ - self.overlap_mask_all
            elif self.decoding_mode == 2:
                rand_order = rand_order_ + self.overlap_mask_all #This makes overlap region go last instead of priortizing it
            self.decoding_order_all = torch.argsort(rand_order)
            
        self.decoding_orders_full[0] = self.all_to_f1[self.decoding_order_all].clone()
        self.decoding_orders[0] = self.decoding_orders_full[0][self.decoding_orders_full[0]!=-1].unsqueeze(0)
        self.decoding_orders_full[1] = self.all_to_f2[self.decoding_order_all].clone()
        self.decoding_orders[1] = self.decoding_orders_full[1][self.decoding_orders_full[1]!=-1].unsqueeze(0)
        
        self.next_q = 0 #Iteration step
        if seed_quartet_list is None:
            self.quartet_list = [ None ] * self.total_len #This tracks quartets; left-to-right order
        else:
            self.quartet_list = copy.deepcopy(seed_quartet_list)
        self.nuc = None
        self.unmasked_logits_joint = [] #This tracks joint probability matrix at each step
        self.masked_logits_joint = [] #This tracks joint probability matrix at each step masked by compatibility
        self.logits_f1 = [] #This tracks logits at each decoding step
        self.logits_f2 = [] 
        self.logits_f1_ = [] #This will track logits at each decoding step before applying various weights and filters
        self.logits_f2_ = []

        self.current_balancer_weight = self.balancer_unit
        
        if self.decoders[0] is not None:
            self.decoders[0].reset(self.decoding_orders[0], self.rand_base, seed_S[0])
        if self.decoders[1] is not None:
            self.decoders[1].reset(self.decoding_orders[1], self.rand_base, seed_S[1])

    def swap_decoding_position(self, t_q_next: int) -> None:
        """
        Swap decoding order to bring specified position next.
        
        Args:
           t_q_next: Position to decode next
        """
        self.decoding_order_all[torch.where(self.decoding_order_all==t_q_next)[0][0]] = self.decoding_order_all[self.next_q]
        self.decoding_order_all[self.next_q] = t_q_next

        self.decoding_orders_full[0] = self.all_to_f1[self.decoding_order_all].clone()
        self.decoding_orders[0] = self.decoding_orders_full[0][self.decoding_orders_full[0]!=-1].unsqueeze(0)
        self.decoding_orders_full[1] = self.all_to_f2[self.decoding_order_all].clone()
        self.decoding_orders[1] = self.decoding_orders_full[1][self.decoding_orders_full[1]!=-1].unsqueeze(0)   
    
    #Decode next step; this is the key function carrying out each step of iterative OLG decoding
    def decode_next(
        self,
        dummy_run: Tuple[bool, bool] = (False, False),
        mask_current: Tuple[bool, bool] = (False, False),
        mask_quartet: bool = False,
        force_safe: bool = False
    ) -> bool:
        """
        This is the key function that performs each step of iterative sampling with overlap constraint
        
        Args:
           dummy_run: Whether to run in dummy mode for each frame
           mask_current: Whether to mask current position before sampling
           mask_quartet: currently unused
           force_safe: Force safe fallback if no valid choices available
        """
        if (self.decoders[0] is None) or (self.decoders[1] is None):
            print("Decoders not initialized")
            return False

        #Step index and positions
        t_q = self.decoding_order_all[self.next_q] #Position of the current quartet
        t_f1 = self.all_to_f1[t_q].unsqueeze(0) #Position of the current quartet, relative to protein 1; -1 means it's not overlapping
        t_f2 = self.all_to_f2[t_q].unsqueeze(0) #Position of the current quartet, relative to protein 2; -1 means it's not overlapping
        t_q_p = t_q - 1 #Position of neighboring quartet to the left
        t_q_n = t_q + 1 #Position of neighboring quartet to the right
        t_q_p = None if t_q_p < 0 else t_q_p #When it's at the left-most position, t_q_p is None
        t_q_n = None if t_q_n >= self.total_len else t_q_n #When it's at the right-most position, t_q_n is None

        self.quartet_list[t_q] = None
        
        #Get the logits and check if current position is overlapping or not
        overlapping_t = True
        if t_f1 != -1: #If the protein exists at this position, we get the next amino acid logit vector from the decoder
            logits_f1, logits_f1_ = self.decoders[0].decode_next(dummy_run[0], mask_current[0])
        else: 
            logits_f1 = torch.zeros((1, len(ALPHABET)), device=self.device)
            logits_f1_ = torch.zeros((1, len(ALPHABET)), device=self.device)
            overlapping_t = False #If the protein is not overlapping, we zero the logit vector
        if t_f2 != -1: #
            logits_f2, logits_f2_ = self.decoders[1].decode_next(dummy_run[1], mask_current[1])
        else:
            logits_f2 = torch.zeros((1, len(ALPHABET)), device=self.device)
            logits_f2_ = torch.zeros((1, len(ALPHABET)), device=self.device)
            overlapping_t = False
        
        #Joint probabilities is the pair-wise sum of logits
        logits_joint = torch.log(logits_f1.softmax(-1)).unsqueeze(-1) + torch.log(logits_f2.softmax(-1)).unsqueeze(-2)
        logits_joint_safe = logits_f1_.unsqueeze(-1) + logits_f2_.unsqueeze(-2)
        
        #Keep track the logits for sanity checks
        self.logits_f1 += [ logits_f1.clone().detach() ] #Logits from the decoder after applying weights/filtering
        self.logits_f2 += [ logits_f2.clone().detach() ] 
        self.logits_f1_ += [ logits_f1_.clone().detach() ] #Logits from the decoder prior to applying weights/filtering
        self.logits_f2_ += [ logits_f2_.clone().detach() ] 
        self.unmasked_logits_joint += [ logits_joint.clone().detach() ] #Joint logits prior to applying compatibility mask
        
        #If previous quartet was already decoded, then we need to consider this constraint
        q_p = torch.tensor([0, 1, 2, 3], device=self.device).long() #To allow all first nucleotide if previous position was not decoded yet
        if t_q_p is not None:
            if self.quartet_list[t_q_p] is not None:
                q_p = torch.unique(self.prev_quartet_index[self.quartet_list[t_q_p]]) #Last nucleotide of the previous quartets

        #If next quartet was already decoded, then we need to consider this constraint
        q_n = torch.tensor([0, 1, 2, 3], device=self.device).long() #To allow all last nucleotide if next position was not decoded yet
        if t_q_n is not None:
            if self.quartet_list[t_q_n] is not None:
                q_n = torch.unique(self.next_quartet_index[self.quartet_list[t_q_n]]) #First nucleotide of the previous quartets

        #All possible combinations of first and last NUCLEOTIDES; would be 4x4=16 if no previous/next positions were decoded
        p_n = torch.tensor([ (p, n) for p in q_p for n in q_n ]).long()

        compatibility = self.codon_compatibility.clone()
        if self.start[0] and (t_f1 == self.seq_starts[0]):
            compatibility *= self.codon_compatibility_start_mask[0]
        if self.start[1] and (t_f2 == self.seq_starts[1]):
            compatibility *= self.codon_compatibility_start_mask[1]
        
        #Fixed position mask
        fixed_f1 = self.fixed_positions_set[0][t_f1] if t_f1 != -1 else None
        fixed_f1_prev = self.fixed_positions_set[0][t_f1-1] if (t_f1 > 0) else None
        fixed_f1_next = self.fixed_positions_set[0][t_f1+1] if 0 < ((t_f1 + 1) < self.f1_gap_len) else None
        fixed_f2 = self.fixed_positions_set[1][t_f2] if t_f2 != -1 else None
        fixed_f2_prev = self.fixed_positions_set[1][t_f2-1] if (t_f2 > 0) else None
        fixed_f2_next = self.fixed_positions_set[1][t_f2+1] if 0 < ((t_f2 + 1) < self.f2_gap_len) else None

        compatibility_safe = compatibility.clone()
        if not ((fixed_f1 == None) and (fixed_f1_prev == None) and (fixed_f1_next == None) and (fixed_f2 == None) and (fixed_f2_prev == None) and (fixed_f2_next == None)):
            compatible_q_i = compatible_quartets_by_aa(self.arrangement, 
                                                       (fixed_f1_prev, fixed_f1, fixed_f1_next), 
                                                       (fixed_f2_prev, fixed_f2, fixed_f2_next), 
                                                       self.codon_table_rev)
            codon_compatibility_fixed_mask = torch.zeros(self.codon_compatibility.shape, device=self.device).int()
            codon_compatibility_fixed_mask[:, :, :, :, :, compatible_q_i] = 1
            compatibility *= codon_compatibility_fixed_mask
            
        compatibility = (~(compatibility[p_n[:, 0], p_n[:, 1], self.arrangement, :, :, :].bool())) #Get compatibility matrix, for given first and last nucleotide of quartets
        
        quartets_logits_joint = logits_joint.repeat(compatibility.shape[0], 1, 1).unsqueeze(3).repeat(1, 1, 1, len(QUARTETS)) #Joint logits, repeated so that we can mask with compatibility matrix
        quartets_logits_joint[compatibility] = MIN_LOGIT #Mask joint logits matrix with compatibility matrix
        masked_logits_joint = torch.clamp(quartets_logits_joint, min=MIN_LOGIT)
        
        if masked_logits_joint.max() == MIN_LOGIT: #Invalid case
            if force_safe:
                compatibility_safe = (~(compatibility_safe[p_n[:, 0], p_n[:, 1], self.arrangement, :, :, :].bool()))
                quartets_logits_joint = logits_joint_safe.repeat(compatibility_safe.shape[0], 1, 1).unsqueeze(3).repeat(1, 1, 1, len(QUARTETS))
                quartets_logits_joint[compatibility_safe] = MIN_LOGIT #Mask joint logits matrix with compatibility matrix
                masked_logits_joint = torch.clamp(quartets_logits_joint, min=MIN_LOGIT)
            else:
                print("Invalid; no available choice")
                self.errored_compat = compatibility
                return False
            
        self.masked_logits_joint += [ masked_logits_joint.clone().detach() ]
                
        #This implements top-p decoding
        masked_logits_joint_amax = masked_logits_joint.amax([0, 3]) #Collapse to AAs only
        sort_v_, sort_ind = masked_logits_joint_amax.flatten().sort(descending=True) #Sort by logits
        sort_v = torch.nn.functional.softmax(sort_v_/self.temperature, dim=-1) #Apply temperature and softmax
        sort_v_cumsum = sort_v.cumsum(0) #Get cumulative probability of ranked probs for top-P sampling
        cutoff_ind = torch.nonzero(sort_v_cumsum>self.top_p)[0][0] + 1 #Top-P cutoff
        topp_v = sort_v[0:cutoff_ind]
        selected = torch.multinomial(topp_v, 1) #Sampling
        
        best_q = torch.where(masked_logits_joint == sort_v_[selected]) 
        
        #TODO: Add quartet bias (to favor some quartets over others)
        if overlapping_t: #If in overlapping region
            best_q_aa = torch.unique(torch.stack([best_q[1], best_q[2]]), dim=1)
            if (best_q_aa.shape[1] > 1): #In case there is multiple equally likely amino acids, randomly choose a pair
                best_q_aa_rand = torch.randint(0, best_q_aa.shape[1], (1,))
                best_q_aa1 = best_q_aa[0, best_q_aa_rand]
                best_q_aa2 = best_q_aa[1, best_q_aa_rand]
                best_q_uniq_ind = [ bqi for bqi in range(best_q[3].shape[0]) if (best_q[1][bqi] == best_q_aa1) and (best_q[2][bqi] == best_q_aa2) ]
                best_q = [ bq[best_q_uniq_ind] for bq in best_q ]
        else: #Same thing, for not overlapping region
            if t_f1 != -1:
                best_q_aa = torch.unique(best_q[1])
            else:
                best_q_aa = torch.unique(best_q[2])
            if (best_q_aa.shape[0] > 1):
                best_q_aa_rand = torch.randint(0, best_q_aa.shape[0], (1,))
                if t_f1 != -1:
                    best_q_aa1 = best_q_aa[best_q_aa_rand]
                    best_q_uniq_ind = [ bqi for bqi in range(best_q[3].shape[0]) if (best_q[1][bqi] == best_q_aa1) ]
                else:
                    best_q_aa2 = best_q_aa[best_q_aa_rand]
                    best_q_uniq_ind = [ bqi for bqi in range(best_q[3].shape[0]) if (best_q[2][bqi] == best_q_aa2) ]
                best_q = [ bq[best_q_uniq_ind] for bq in best_q ]
        
        #best_q now tracks all available quartets for the selected AA pair. Now we will update protein sequence for next iteration
        if t_f1 != -1:
            f1_S_t = best_q[1][0].unsqueeze(0).unsqueeze(0)
            
            if self.complexed:
                t_full = self.decoders[0].decoding_order[0, self.decoders[0].next_t_full]
                t_ = self.decoders[0].decoding_order_target[0, self.decoders[0].next_t*self.decoders[0].n_design_chains]

                if self.decoders[0].tied:
                    tied_list = self.decoders[0].tied_pos[t_]
                else:
                    tied_list = [ t_ ]
                for t in tied_list:
                    self.decoders[1].edit_S(t, f1_S_t, inplace=True) #Decodes from the other frame if it was part of same complex
            
            elif self.shared:
                self.decoders[1].edit_S(t_f1, f1_S_t, inplace=True)
            
            self.decoders[0].update_S(f1_S_t, dummy_run=dummy_run[0])
        
        if t_f2 != -1:
            f2_S_t = best_q[2][0].unsqueeze(0).unsqueeze(0)
            
            if self.complexed:
                t_full = self.decoders[1].decoding_order[0, self.decoders[1].next_t_full]
                t_ = self.decoders[1].decoding_order_target[0, self.decoders[1].next_t*self.decoders[1].n_design_chains]
            
                if self.decoders[1].tied:
                    tied_list = self.decoders[1].tied_pos[t_]
                else:
                    tied_list = [ t_ ]
                for t in tied_list:
                    self.decoders[0].edit_S(t, f2_S_t, inplace=True) #Decodes from the other frame if it was part of same complex
                
            elif self.shared:
                self.decoders[0].edit_S(t_f2, f2_S_t, inplace=True)
                    
            self.decoders[1].update_S(f2_S_t, dummy_run=dummy_run[1])
        
        best_q = [ q.cpu() for q in best_q ]
        
        #Check that previous position quartets are compatible with current position quartets. This has to be recursive, since choice at each position affects its neighbors
        if (t_q_p is not None) and (self.quartet_list[t_q_p] is not None):
            compatible_prev = np.intersect1d(self.quartet_list[t_q_p], torch.stack([ self.compatible_prev_quartets[q.item()] for q in best_q[3] ]).flatten().unique().cpu())
            self.quartet_list[t_q_p] = compatible_prev
            
            #Recursively check all quartets connected to it
            t_q_p_i = t_q_p - 1
            while (t_q_p_i >= 0) and (self.quartet_list[t_q_p_i] is not None) and (self.quartet_list[t_q_p_i].shape[0] > 1):
                compatible_prev = np.intersect1d(self.quartet_list[t_q_p_i], torch.stack([ self.compatible_prev_quartets[q.item()] for q in self.quartet_list[t_q_p_i+1]]).flatten().unique().cpu())
                self.quartet_list[t_q_p_i] = compatible_prev
                t_q_p_i -= 1
        
        #Check that next position quartets are compatible with current position quartets
        if (t_q_n is not None) and (self.quartet_list[t_q_n] is not None):
            compatible_next = np.intersect1d(self.quartet_list[t_q_n], torch.stack([ self.compatible_next_quartets[q.item()] for q in best_q[3] ]).flatten().unique().cpu())
            self.quartet_list[t_q_n] = compatible_next
            
            #Recursively check all quartets connected to it
            t_q_n_i = t_q_n + 1
            while (t_q_n_i < self.total_len) and (self.quartet_list[t_q_n_i] is not None) and (self.quartet_list[t_q_n_i].shape[0] > 1):
                compatible_next = np.intersect1d(self.quartet_list[t_q_n_i], torch.stack([ self.compatible_next_quartets[q.item()] for q in self.quartet_list[t_q_n_i-1]]).flatten().unique().cpu())
                self.quartet_list[t_q_n_i] = compatible_next
                t_q_n_i += 1
        
        #Save the quartets for current position
        self.quartet_list[t_q] = best_q[3]
        self.next_q += 1
        
        return True
        
    def string_quartet(self) -> Tuple[str, List[int]]:
        """
        Convert decoded quartets into final nucleotide string.
        Randomly selects compatible quartets when multiple choices remain.
        
        Returns:
           tuple: (nucleotide_string, final_quartet_list)
        """
        quartet_list = copy.deepcopy(self.quartet_list)
        
        #First quartet; look to next quartet and choose randomly among the acceptable (connecting) quartets
        acceptable = []
        nt_p_1s = self.prev_quartet_index[quartet_list[0]]
        nt_qn_1s = self.next_quartet_index[quartet_list[1]]
        for n in range(len(nt_p_1s)):
            if nt_p_1s[n] in nt_qn_1s:
                acceptable += [ n ]
        quartet_list[0] = quartet_list[0][np.random.choice(acceptable)]

        #Second to second-last quartet; look to neighboring quartets and choose randomly among the acceptable (connecting) quartets
        for q in range(1, len(quartet_list)-1, 1):
            acceptable = []
            nt_q_1s = self.next_quartet_index[quartet_list[q]]
            nt_p_1s = self.prev_quartet_index[quartet_list[q]]
            nt_qp_4 = self.prev_quartet_index[quartet_list[q-1]]
            nt_qn_1s = self.next_quartet_index[quartet_list[q+1]]    
            for n in range(len(nt_q_1s)):
                if nt_q_1s[n] == nt_qp_4:
                    if nt_p_1s[n] in nt_qn_1s:
                        acceptable += [ n ]
            quartet_list[q] = quartet_list[q][np.random.choice(acceptable)]

        #last quartet; look to previous quartet and choose randomly among the acceptable (connecting) quartets
        acceptable = []
        nt_q_1s = self.next_quartet_index[quartet_list[len(quartet_list)-1]]
        nt_qp_4 = self.prev_quartet_index[quartet_list[len(quartet_list)-2]]
        for n in range(len(nt_q_1s)):
            if nt_q_1s[n] == nt_qp_4:
                acceptable += [ n ]
        quartet_list[len(quartet_list)-1] = quartet_list[len(quartet_list)-1][np.random.choice(acceptable)]

        #Trim last nucleotide if in-phase overlap (-0)
        final_nuc = ''.join([ QUARTETS[q][0:3] for q in quartet_list ] + [ QUARTETS[quartet_list[len(quartet_list)-1]][-1] ])
        self.nuc = torch.tensor([ NUCLEOTIDES.index(c) for c in final_nuc ], device=self.device)
        return final_nuc, quartet_list
    
    def get_prot_seq(self) -> Tuple[str, str]:
        f1_prot = self.decoders[0].get_prot_seq()
        f2_prot = self.decoders[1].get_prot_seq()
        return f1_prot, f2_prot
    
    def get_scores(
        self, 
        positions: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
    ) -> Tuple[float, float]:
        f1_score = self.decoders[0].get_score(positions=positions[0])
        f2_score = self.decoders[1].get_score(positions=positions[1])
        return f1_score, f2_score
    
    def decode_all(
        self,
        dummy_run: Tuple[bool, bool] = (False, False),
        mask_current: Tuple[bool, bool] = (False, False),
        force_safe: bool = False,
        dynamic_order: Optional[str] = None
        ) -> bool:
        for i in tqdm(range(self.total_len), disable=self.tqdm_disable):
            if not self.decode_next(dummy_run=dummy_run, mask_current=mask_current, force_safe=force_safe):
                return False
            if dynamic_order is not None:
                if (self.next_q > 0) and (self.next_q < self.total_len):
                    next_q = self.get_next_order_dyn(dynamic_order, frames=(True, True))
                    self.swap_decoding_position(next_q)
                    self.decoders[0].reset_decoding_order(self.decoding_orders[0])
                    self.decoders[1].reset_decoding_order(self.decoding_orders[1])
        return True

    def get_next_order_dyn(
        self,
        ordering: str = "entropy",
        frames: Tuple[bool, bool] = (True, True),
        priortize_fixed: bool = True
    ) -> int:
        """
        Next position based on current preds
        
        Args:
           ordering: Ordering strategy ("entropy" or "prob")
           frames: Which frames to consider
           priortize_fixed: Whether to prioritize fixed positions
           
        Returns:
           int: Next position to decode
        """
        positions = torch.zeros((2, self.total_len), device=self.device)
        
        if ordering == "entropy":
            if frames[0]:
                f1_curr_pred = self.decoders[0].current_pred[self.seq_starts[0]:self.seq_lens[0], :]
                f1_log_prob = torch.log(torch.nn.functional.softmax(f1_curr_pred, dim=-1))[:, self.decoders[0].alphabet_map]
            else:
                f1_log_prob = torch.zeros(self.decoders[0].current_pred.shape, device=self.device)[self.seq_starts[0]:self.seq_lens[0], self.decoders[0].alphabet_map]
            if frames[1]:
                f2_curr_pred = self.decoders[1].current_pred[self.seq_starts[1]:self.seq_lens[1], :]
                f2_log_prob = torch.log(torch.nn.functional.softmax(f2_curr_pred, dim=-1))[:, self.decoders[1].alphabet_map]
            else:
                f2_log_prob = torch.zeros(self.decoders[1].current_pred.shape, device=self.device)[self.seq_starts[1]:self.seq_lens[1], self.decoders[1].alphabet_map]
                           
            positions[0, self.f1_to_all[self.f1_to_all!=(self.f1_to_all.max()+1-(self.end_stop[0]+0))]] = -1.0 * torch.sum(torch.exp(f1_log_prob)*f1_log_prob, 1)[self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]]
            positions[1, self.f2_to_all[self.f2_to_all!=(self.f2_to_all.max()+1-(self.end_stop[1]+0))]] = -1.0 * torch.sum(torch.exp(f2_log_prob)*f2_log_prob, 1)[self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]]

            positions = positions.mean(0)
            positions = positions / positions.max() #Scale values so that highest value is 1
            positions = positions - self.start_mask_all - self.end_stop_mask_all
            if priortize_fixed:
                 positions = positions - self.fixed_positions_mask_all
            if self.decoding_mode == 1: #Priortize overlap regions if specified
                positions = positions - self.overlap_mask_all
            elif self.decoding_mode == 2:
                positions = positions + self.overlap_mask_all #This makes overlap region go last instead of priortizing it
            next_order = torch.argsort(positions)
            
        elif ordering == "prob":
            if frames[0]:
                f1_curr_pred = self.decoders[0].current_pred[self.seq_starts[0]:self.seq_lens[0], :]
                f1_max_log_prob = torch.log(torch.nn.functional.softmax(f1_curr_pred, dim=-1))[:, self.decoders[0].alphabet_map].max(-1)[0]
            else:
                f1_max_log_prob = torch.zeros(self.decoders[0].current_pred.shape, device=self.device)[self.seq_starts[0]:self.seq_lens[0]]
            if frames[1]:
                f2_curr_pred = self.decoders[1].current_pred[self.seq_starts[1]:self.seq_lens[1], :]
                f2_max_log_prob = torch.log(torch.nn.functional.softmax(f2_curr_pred, dim=-1))[:, self.decoders[1].alphabet_map].max(-1)[0]
            else:
                f2_max_log_prob = torch.zeros(self.decoders[1].current_pred.shape, device=self.device)[self.seq_starts[1]:self.seq_lens[1]]
            
            positions[0, self.f1_to_all[self.f1_to_all!=(self.f1_to_all.max()+1-(self.end_stop[0]+0))]] = f1_max_log_prob[self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]]
            positions[1, self.f2_to_all[self.f2_to_all!=(self.f2_to_all.max()+1-(self.end_stop[1]+0))]] = f2_max_log_prob[self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]]
            #next_order = positions.mean(0).sort()[1]
            positions = positions.mean(0)
            positions = positions / positions.max()
            positions = positions - self.start_mask_all - self.end_stop_mask_all
            if priortize_fixed:
                 positions = positions - self.fixed_positions_mask_all
            if self.decoding_mode == 1: #Priortize overlap regions if specified
                positions = positions - self.overlap_mask_all
            elif self.decoding_mode == 2:
                positions = positions + self.overlap_mask_all #This makes overlap region go last instead of priortizing it
            next_order = torch.argsort(positions)

        return next_order[~torch.isin(next_order, self.decoding_order_all[0:self.next_q])][0]
        
    def get_next_order(
        self,
        ordering: str = "entropy",
        priortize_fixed: bool = True
    ) -> torch.Tensor:
        """
        Generate decoding order based on probs; should be run after computing pseudolikelihoods for both frame
        
        Args:
           ordering: Ordering strategy ("entropy", "prob", "prob_rank", "random", "orig")
           priortize_fixed: Whether to prioritize fixed positions
           
        Returns:
           torch.Tensor: Decoding order indices
        """
        positions = torch.zeros((2, self.total_len), device=self.device)
        if ordering == "entropy":
            f1_log_prob = self.decoders[0].log_prob[:,self.decoders[0].alphabet_map]
            f2_log_prob = self.decoders[1].log_prob[:,self.decoders[1].alphabet_map]
            
            positions[0, self.f1_to_all[self.f1_to_all!=(self.f1_to_all.max()+1-(self.end_stop[0]+0))]] = -1.0 * torch.sum(torch.exp(f1_log_prob)*f1_log_prob, 1)[self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]]
            positions[1, self.f2_to_all[self.f2_to_all!=(self.f2_to_all.max()+1-(self.end_stop[1]+0))]] = -1.0 * torch.sum(torch.exp(f2_log_prob)*f2_log_prob, 1)[self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]]

            positions = positions.mean(0)
            positions = positions / positions.max() #Scale values so that highest value is 1
            positions = positions - self.start_mask_all - self.end_stop_mask_all
            if priortize_fixed:
                 positions = positions - self.fixed_positions_mask_all
            if self.decoding_mode == 1: #Priortize overlap regions if specified
                positions = positions - self.overlap_mask_all
            elif self.decoding_mode == 2:
                positions = positions + self.overlap_mask_all #This makes overlap region go last instead of priortizing it
            next_order = torch.argsort(positions)
        elif ordering == "prob":
            positions[0, self.f1_to_all[self.f1_to_all!=(self.f1_to_all.max()+1-(self.end_stop[0]+0))]] = self.decoders[0].selected_log_prob[0][self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]]
            positions[1, self.f2_to_all[self.f2_to_all!=(self.f2_to_all.max()+1-(self.end_stop[1]+0))]] = self.decoders[1].selected_log_prob[0][self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]]
            #next_order = positions.mean(0).sort()[1]
            positions = positions.mean(0)
            positions = positions / positions.max()
            positions = positions - self.start_mask_all - self.end_stop_mask_all
            if priortize_fixed:
                 positions = positions - self.fixed_positions_mask_all
            if self.decoding_mode == 1: #Priortize overlap regions if specified
                positions = positions - self.overlap_mask_all
            elif self.decoding_mode == 2:
                positions = positions + self.overlap_mask_all #This makes overlap region go last instead of priortizing it
            next_order = torch.argsort(positions)
        elif ordering == "prob_rank":
            prob_1 = self.decoders[0].selected_log_prob[0][self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]]
            prob_2 = self.decoders[1].selected_log_prob[0][self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]]
            prob_rank_1 = self.decoders[0].log_prob.sort(1)[1].gather(1, self.decoders[0].S.permute([1,0]))[:,0][self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]]
            prob_rank_2 = self.decoders[1].log_prob.sort(1)[1].gather(1, self.decoders[1].S.permute([1,0]))[:,0][self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]]
            positions[0, self.f1_to_all[self.f1_to_all!=(self.f1_to_all.max()+1-(self.end_stop[0]+0))]] = prob_rank_1 + prob_1 * 1e-8
            positions[1, self.f2_to_all[self.f2_to_all!=(self.f2_to_all.max()+1-(self.end_stop[1]+0))]] = prob_rank_2 + prob_2 * 1e-8
            #next_order = positions.mean(0).sort()[1]
            positions = positions.mean(0)
            positions = positions / positions.max()
            positions = positions - self.start_mask_all - self.end_stop_mask_all
            if priortize_fixed:
                 positions = positions - self.fixed_positions_mask_all
            if self.decoding_mode == 1: #Priortize overlap regions if specified
                positions = positions - self.overlap_mask_all
            elif self.decoding_mode == 2:
                positions = positions + self.overlap_mask_all #This makes overlap region go last instead of priortizing it
            next_order = torch.argsort(positions)
        elif ordering == "random":
            next_order_ = self.decoding_order_all.clone()
            idx = torch.randperm(next_order_.nelement())
            next_order = next_order_[idx]
        elif ordering == "orig":
            next_order = self.decoding_order_all.clone()
        return next_order

    def get_next_weight(
        self, 
        scores_pll: List[[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Heuristics for calculating weights to balance the two frames.
        It will look at the last two scores in the history and increment the current balancing weight if the signs are the same.

        Args:
            scores_pll: pseudolikelihood history
        Returns:
            tuple: (frame1_weights, frame2_weights)
        """
        scores_0 = [ scores_pll[-1][0], scores_pll[-1][1] ]
        sd0 = scores_0[1] - scores_0[0]
        
        if len(scores_pll) > 1: #Must have at least 2 items in the history to compare, else default to just looking at the last one            
            scores_1 = [ scores_pll[-2][0], scores_pll[-2][1] ]
            sd1 = scores_1[1] - scores_1[0]
            
            if (torch.sign(sd1) == torch.sign(sd0)) and (torch.abs(sd0) > self.balancer_threshold):
                self.current_balancer_weight = self.current_balancer_weight + self.balancer_unit
            else:
                self.current_balancer_weight = self.balancer_unit
        else:
            self.current_balancer_weight = self.balancer_unit

        diff = torch.abs(sd0)
        weight = min(self.balancer_max_weight, diff * self.current_balancer_weight + 1.0)
        if scores_pll[-1][1] > scores_pll[-1][0]:
            w1 = torch.tensor([1.0]*self.decoders[0].logit_weight.shape[0], device=self.device)
            w2 = torch.tensor([weight]*self.decoders[1].logit_weight.shape[0], device=self.device)
        else:
            w1 = torch.tensor([weight]*self.decoders[0].logit_weight.shape[0], device=self.device)
            w2 = torch.tensor([1.0]*self.decoders[1].logit_weight.shape[0], device=self.device)
        return w1, w2
        
    def decode_all_gibbs(
        self,
        next_order: Optional[torch.Tensor] = None,
        weight: Tuple[float, float] = (1.0, 1.0),
        seed_quartet: bool = False,
        force_safe: bool = False,
        dummy_run: Tuple[bool, bool] = (False, False),
        dynamic_order: Optional[str] = None
    ) -> None:
        """
        Run Gibbs/ICM style iterative refinement.
        
        Args:
           next_order: Custom decoding order
           weight: Weights for each frame
           seed_quartet: Whether to seed with current quartets
           force_safe: Force safe fallback for invalid choices
           dummy_run: Whether to run in dummy mode
           dynamic_order: Dynamic ordering strategy
        """
        w1, w2 = weight
        seed_quartet_list = self.quartet_list if seed_quartet else None
        self.reset_decoding(user_order=next_order, seed_S=(self.decoders[0].S.clone(), self.decoders[1].S.clone()), seed_quartet_list=seed_quartet_list)
        self.decoders[0].logit_weight = w1
        self.decoders[1].logit_weight = w2
        self.decode_all(dummy_run=dummy_run, mask_current=(True, True), force_safe=force_safe, dynamic_order=dynamic_order) #Run decoding with current position masking
    
    def nuc_pos_to_aa_pos(
        self,
        nuc_pos: int
    ) -> Tuple[Tuple[Optional[int], Optional[int], Optional[int]], 
        Tuple[Optional[int], Optional[int], Optional[int]]]:
        """
        Given nucleotide position, get amino acid positions relative to each protein
        
        Args:
           nuc_pos: Nucleotide position
           
        Returns:
           tuple: ((f1_pos, f1_pos_target, f1_codon_pos), 
                   (f2_pos, f2_pos_target, f2_codon_pos))
        """
        if self.arrangement == 0:
            f1_pos = math.floor(nuc_pos / 3) #Position of AA
            f1_pos_target = f1_pos - self.f1_start #Relative to protein
            f1_codon_pos = nuc_pos % 3 #1/2/3 on codon
            f1_in_range = (f1_pos >= self.f1_start) and (f1_pos < self.f1_end) and (f1_pos_target >= 0) and (f1_pos_target < (self.seq_lens[0] + self.end_stop[0]))
            f2_pos = math.floor((nuc_pos - 1) / 3)
            f2_pos_target = f2_pos - self.f2_start
            f2_codon_pos = (nuc_pos - 1) % 3
            f2_in_range = (f2_pos >= self.f2_start) and (f2_pos < self.f2_end) and (f2_pos_target >= 0) and (f2_pos_target < (self.seq_lens[1] + self.end_stop[1]))

        elif self.arrangement == 3:
            f2_pos = math.floor(nuc_pos / 3) #Position of AA
            f2_pos_target = f2_pos - self.f2_start #Relative to protein
            f2_codon_pos = nuc_pos % 3 #1/2/3 on codon
            f2_in_range = (f2_pos >= self.f2_start) and (f2_pos < self.f2_end) and (f2_pos_target >= 0) and (f2_pos_target < (self.seq_lens[1] + self.end_stop[1]))
            f1_pos = math.floor((nuc_pos - 1) / 3)
            f1_pos_target = f1_pos - self.f1_start
            f1_codon_pos = (nuc_pos - 1) % 3
            f1_in_range = (f1_pos >= self.f1_start) and (f1_pos < self.f1_end) and (f1_pos_target >= 0) and (f1_pos_target < (self.seq_lens[0] + self.end_stop[0]))

        elif self.arrangement == 1:
            f1_pos = math.floor(nuc_pos / 3) #Position of AA
            f1_pos_target = f1_pos - self.f1_start #Relative to protein
            f1_codon_pos = nuc_pos % 3 #1/2/3 on codon
            f1_in_range = (f1_pos >= self.f1_start) and (f1_pos < self.f1_end) and (f1_pos_target >= 0) and (f1_pos_target < (self.seq_lens[0] + self.end_stop[0]))
            f2_pos = math.floor((nuc_pos - 1) / 3)
            f2_pos_target = self.f2_end - 1 - f2_pos
            f2_codon_pos = (nuc_pos - 1) % 3
            f2_in_range = (f2_pos >= self.f2_start) and (f2_pos < self.f2_end) and (f2_pos_target >= 0) and (f2_pos_target < (self.seq_lens[1] + self.end_stop[1]))

        elif self.arrangement == 4:
            f2_pos = math.floor(nuc_pos / 3) #Position of AA
            f2_pos_target = self.f2_end - 1 - f2_pos #Relative to protein
            f2_codon_pos = nuc_pos % 3 #1/2/3 on codon
            f2_in_range = (f2_pos >= self.f2_start) and (f2_pos < self.f2_end) and (f2_pos_target >= 0) and (f2_pos_target < (self.seq_lens[1] + self.end_stop[1]))
            f1_pos = math.floor((nuc_pos - 1) / 3)
            f1_pos_target = f1_pos - self.f1_start
            f1_codon_pos = (nuc_pos - 1) % 3
            f1_in_range = (f1_pos >= self.f1_start) and (f1_pos < self.f1_end) and (f1_pos_target >= 0) and (f1_pos_target < (self.seq_lens[0] + self.end_stop[0]))

        elif self.arrangement == 2:
            f1_pos = math.floor(nuc_pos / 3) #Position of AA
            f1_pos_target = f1_pos - self.f1_start #Relative to protein
            f1_codon_pos = nuc_pos % 3 #1/2/3 on codon
            f1_in_range = (f1_pos >= self.f1_start) and (f1_pos < self.f1_end) and (f1_pos_target >= 0) and (f1_pos_target < (self.seq_lens[0] + self.end_stop[0]))
            f2_pos = math.floor(nuc_pos  / 3)
            f2_pos_target = self.f2_end - 1 - f2_pos
            f2_codon_pos = nuc_pos % 3
            f2_in_range = (f2_pos >= self.f2_start) and (f2_pos < self.f2_end) and (f2_pos_target >= 0) and (f2_pos_target < (self.seq_lens[1] + self.end_stop[1]))

        f1_res = (f1_pos, f1_pos_target, f1_codon_pos) if f1_in_range else (None, None, None)
        f2_res = (f2_pos, f2_pos_target, f2_codon_pos) if f2_in_range else (None, None, None)
        return f1_res, f2_res
    
    def best_nuc_change(
        self,
        nuc_pos: int,
        nuc_temp: Optional[torch.Tensor] = None
    ) -> None:
        """
        Find and apply best nucleotide change at given position.
        
        Args:
           nuc_pos: Nucleotide position to optimize
           nuc_temp: Temporary nucleotide sequence (uses self.nuc if None)
        """
        if nuc_temp is not None:
            nuc = nuc_temp
        else:
            nuc = self.nuc
            
        (f1_pos, f1_pos_target, f1_codon_pos), (f2_pos, f2_pos_target, f2_codon_pos) = self.nuc_pos_to_aa_pos(nuc_pos)
        
        logits_f1 = torch.zeros((1, len(ALPHABET)), device=self.device)
        logits_f1_ = torch.zeros((1, len(ALPHABET)), device=self.device)
        logits_f2 = torch.zeros((1, len(ALPHABET)), device=self.device)
        logits_f2_ = torch.zeros((1, len(ALPHABET)), device=self.device)
        f1_all_aa = torch.zeros(4, device=self.device).long()
        f2_all_aa = torch.zeros(4, device=self.device).long()

        if f1_pos is not None:
            f1_codon_start = nuc_pos - f1_codon_pos
            f1_codon_end = f1_codon_start + 3
            f1_codon = nuc[f1_codon_start:f1_codon_end].unsqueeze(0) #if positive; else revcomp
            f1_all_codons = f1_codon.repeat((4, 1)) #All 4 possible nucleotides
            f1_all_codons[:, f1_codon_pos] = torch.arange(4, device=self.device) 
            f1_all_aa = self.codon_to_aa[f1_all_codons[:, 0], f1_all_codons[:, 1], f1_all_codons[:, 2]] #Translate
            if f1_pos_target < self.decoders[0].decoded_positions.shape[1]:
                self.decoders[0].decoded_positions[0, f1_pos_target] = 0
            logits_f1, logits_f1_ = self.decoders[0].decode_next(mask_current=True, use_t=f1_pos_target)
        if f2_pos is not None:
            f2_codon_start = nuc_pos - f2_codon_pos
            f2_codon_end = f2_codon_start + 3
            f2_codon = nuc[f2_codon_start:f2_codon_end].unsqueeze(0) #if positive; else revcomp
            f2_all_codons = f2_codon.repeat((4, 1))
            f2_all_codons[:, f2_codon_pos] = torch.arange(4, device=self.device)
            if self.f2_neg:
                f2_all_aa = self.codon_to_aa_rc[f2_all_codons[:, 0], f2_all_codons[:, 1], f2_all_codons[:, 2]]
            else:
                f2_all_aa = self.codon_to_aa[f2_all_codons[:, 0], f2_all_codons[:, 1], f2_all_codons[:, 2]]
            if f2_pos_target < self.decoders[1].decoded_positions.shape[1]:
                self.decoders[1].decoded_positions[0, f2_pos_target] = 0
            logits_f2, logits_f2_ = self.decoders[1].decode_next(mask_current=True, use_t=f2_pos_target)
        
        best_nuc = torch.stack([ logits_f1[0, f1_all_aa], logits_f2[0, f2_all_aa] ]).mean(0).argmax()
        current_nuc = nuc[nuc_pos]
        nuc[nuc_pos] = best_nuc

        if f1_pos_target is not None:
            if f1_pos_target < self.decoders[0].decoded_positions.shape[1]:
                self.decoders[0].update_S(f1_all_aa[best_nuc], use_t=f1_pos_target, alphabet_map=True)
        if f2_pos_target is not None:
            if f2_pos_target < self.decoders[1].decoded_positions.shape[1]:
                self.decoders[1].update_S(f2_all_aa[best_nuc], use_t=f2_pos_target, alphabet_map=True)

    #Iterate Gibbs/ICM style passes at nucleotide level
    def mutate_all_gibbs(
        self,
        ordering: str = "entropy",
        aw_max: float = 0.25,
        aw_scale: float = 0.25,
        scores: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Run Gibbs/ICM style refinement at nucleotide level.
        
        Args:
           ordering: Ordering strategy
           aw_max: Maximum weight adjustment
           aw_scale: Weight scaling factor
           scores: Current scores
        """
        if scores is None:
            scores = self.get_scores()

        next_order = self.get_next_order(ordering) if ordering != "orig" else None
        pos_ind = torch.arange(self.nuc_total_len)
        pos_ind_ = np.concatenate([ np.random.permutation(pos_ind[(quartet_pos*3):(quartet_pos*3+4)]) for quartet_pos in self.decoding_order_all.cpu().numpy() ])
        pos_ind_uniq, pos_ind_uniq_ind = np.unique(pos_ind_, return_index=True)
        next_order_nuc = torch.tensor(pos_ind_uniq[pos_ind_uniq_ind.argsort()], device=self.device)
        
        weight = min(aw_max, scores[1] / scores[0] - 1.0) * aw_scale + 1.0
        w1 = torch.tensor([1.0]*100, device=self.device)
        w2 = torch.tensor([weight]*100, device=self.device)
        
        #self.reset_decoding(user_order=next_order, seed_S=(self.decoders[0].S.clone(), self.decoders[1].S.clone()))
        self.decoders[0].logit_weight = w1
        self.decoders[1].logit_weight = w2

        for nuc_pos in tqdm(next_order_nuc, disable=self.tqdm_disable):
            self.best_nuc_change(nuc_pos)

    #Check if fixed positions/stop/start make sense
    def validate_fixed(
        self,
        prerun: bool = True,
        reset: bool = True,
        print_error: bool = True,
    ) -> bool:
        """
        Do a dry run to check if fixed positions, stop, and start codons are valid.
        
        Args:
           prerun: Whether to run full decoding first
           reset: Whether to reset after validation
           print_error: Whether to print the failures
           
        Returns:
           tuple: (success_flag, list of failed positions)
        """
        if prerun:
            self.decode_all(dummy_run=(True, True), force_safe=True)
            #if not self.decode_all(dummy_run=(True, True), force_safe=True):
#                print("No choices possible at step "+str(self.next_q))
                #return False
            
        S_f1, S_f2 = self.get_prot_seq()
        nuc_seq, quartets = self.string_quartet()
        
        failed = False
        failed_res = []
        if self.fixed_positions[0] is not None:
            for pos, aa in self.fixed_positions[0]:
                if aa != S_f1[pos-1]:
                    failed = True
                    failed_res += [ (0, pos, aa) ]
                    if print_error:
                        print("Fixed residue could not be placed for protein 1: "+str(pos)+" "+aa)

        if self.fixed_positions[1] is not None:
            for pos, aa in self.fixed_positions[1]:
                if aa != S_f2[pos-1]:
                    failed = True
                    failed_res += [ (1, pos, aa) ]
                    if print_error:
                        print("Fixed residue could not be placed for protein 2: "+str(pos)+" "+aa)
                    
        if self.end_stop[0]:
            q_i = quartets[self.f1_to_all[-1]]
            aa_f1 = self.quartets_aa[q_i][FRAME_F1[self.arrangement]] 
            aa_i = STOP_INDEX
            if aa_f1 != aa_i:
                failed = True
                failed_res += [ (0, None, 'Stop') ]
                if print_error:
                    print("Stop could not be placed for protein 1")

        if self.end_stop[1]:
            q_i = quartets[self.f2_to_all[-1]]
            aa_f2 = self.quartets_aa[q_i][FRAME_F2[self.arrangement]] 
            aa_i = STOP_INDEX
            if aa_f2 != aa_i:
                failed = True
                failed_res += [ (1, None, 'Stop') ]
                if print_error:
                    print("Stop could not be placed for protein 2")
                
        if self.start[0]:
            q_i = quartets[self.f1_to_all[0]]
            if q_i not in self.start_codons_quartets[0]:
                failed = True
                failed_res += [ (0, 1, 'Start') ]
                if print_error:
                    print("Start could not be placed for protein 1")
                
        if self.start[1]:
            q_i = quartets[self.f2_to_all[0]]
            if q_i not in self.start_codons_quartets[1]:
                failed = True
                failed_res += [ (0, 1, 'Start') ]
                if print_error:
                    print("Start could not be placed for protein 2")
        
        if reset:
            self.reset_decoding(self.rand_base, user_order=self.decoding_order_all)
        return (not failed, failed_res)