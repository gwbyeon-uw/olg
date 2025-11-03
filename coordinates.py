import numpy as np
import torch

from constants import Arrangement, DecodingMode
from config import DesignConfig

class Coordinates:
    """Manages coordinate transformations between proteins and absolute positions given the design setup"""
    def __init__(
        self, 
        config: DesignConfig
    ):
        self.config = config
        self.f2_neg = self.config.arrangement in [ Arrangement.MINUS_ONE, Arrangement.MINUS_ZERO, Arrangement.MINUS_TWO ] #Whether frame 2 is negative strand
        
        self.f1_gap_len = self.config.protein1.length - len(self.config.protein1.gap_positions) if self.config.protein1.gap_positions is not None else self.config.protein1.length
        self.f2_gap_len = self.config.protein2.length - len(self.config.protein2.gap_positions) if self.config.protein2.gap_positions is not None else self.config.protein2.length
        
        #Prepare coordinates for relative positions of the proteins; protein 1 always starts at 0 and is always positive strand
        f1_start_ = 0 #Protein 1 always starts at 0 position
        f2_strand = -1 if self.f2_neg else 1 #-1 indicates negative strand overlap
        f2_start_ = min(self.config.offset, self.config.offset + f2_strand * (self.f2_gap_len + self.config.protein2.force_stop)) #+1 to make room for stop codon
        all_offset = -1 * min(f2_start_, 0)
        self.f1_start = f1_start_ + all_offset
        self.f1_end = self.f1_start + self.f1_gap_len + self.config.protein1.force_stop
        self.f2_start = f2_start_ + all_offset
        self.f2_end = self.f2_start + self.f2_gap_len + self.config.protein2.force_stop
        self.total_len = max(self.f1_end, self.f2_end)
        
        f1_range = range(self.f1_start, self.f1_end)
        f2_range = range(self.f2_start, self.f2_end)
        overlap_range = range(max(f1_range[0], f2_range[0]), min(f1_range[-1], f2_range[-1])+1) #Intersection
        overlap_start = overlap_range[0]
        overlap_end = overlap_range[-1] + 1
        
        #To help set decoding order so that start codons gets priority
        self.start_mask_all = torch.zeros(self.total_len, device=self.config.device)
        if self.config.protein1.force_start:
            f1_start_pos = self.f1_start
            self.start_mask_all[f1_start_pos] = 10 #High value but less than stop codon mask value so that it goes second in decoding order
        if self.config.protein2.force_start:
            f2_start_pos = (self.f2_end - 1) if self.f2_neg else self.f2_start
            self.start_mask_all[f2_start_pos] = 10
        
        #To help set decoding order so that stop codons gets priority
        self.end_stop_mask_all = torch.zeros(self.total_len, device=self.config.device)
        if self.config.protein1.force_stop:
            f1_stop_pos = self.f1_end - 1
            self.end_stop_mask_all[f1_stop_pos] = 20 #High value so that it goes first in decoding order
        if self.config.protein2.force_stop:
            f2_stop_pos = self.f2_start if self.f2_neg else (self.f2_end - 1)
            self.end_stop_mask_all[f2_stop_pos] = 20
            
        #To help set decoding order so that overlap region gets priority
        self.overlap_mask_all = torch.zeros(self.total_len, device=self.config.device)
        self.overlap_mask_all[overlap_start:overlap_end] = 1 
        
        #Transformation tensors to help convert positions from absolute coordinates to relative coordinates (to each protein) and vice versa
        self.all_to_f1 = torch.zeros(self.total_len, device=self.config.device).long()
        self.all_to_f1.fill_(-1) #Positions where there is no overlap
        self.all_to_f1[self.f1_start:self.f1_end] = torch.arange(self.f1_end - self.f1_start) + self.config.protein1.start_offset
        self.all_to_f2 = torch.zeros(self.total_len, device=self.config.device).long()
        self.all_to_f2.fill_(-1)
        self.all_to_f2[self.f2_start:self.f2_end] = torch.arange(self.f2_end - self.f2_start) + self.config.protein2.start_offset
        if self.f2_neg: #Invert if negative
            self.all_to_f2[self.f2_start:self.f2_end] = self.all_to_f2[self.f2_end-1] - self.all_to_f2[self.f2_start:self.f2_end]

        self.f1_to_all = torch.stack([ (self.all_to_f1 == pos).nonzero().squeeze(0).squeeze(0) for pos in range(self.config.protein1.start_offset,self.config.protein1.start_offset+self.f1_gap_len+self.config.protein1.force_stop) ])
        self.f2_to_all = torch.stack([ (self.all_to_f2 == pos).nonzero().squeeze(0).squeeze(0) for pos in range(self.config.protein2.start_offset,self.config.protein2.start_offset+self.f2_gap_len+self.config.protein2.force_stop) ])
        
        self.f1_to_f2 = self.all_to_f2[self.f1_to_all]
        self.f2_to_f1 = self.all_to_f1[self.f2_to_all]

        self.nuc_total_len = self.total_len * 3 + 1

        # Fixed positions
        self.fixed_positions_set = [ [None] * (self.all_to_f1.max()+1), [None] * (self.all_to_f2.max()+1) ]
        self.fixed_positions_mask_all = torch.zeros(self.total_len, device=self.config.device) #To help set decoding order so that fixed positions get priority

        if self.config.protein1.fixed_positions is not None:
            for pos, aa in self.config.protein1.fixed_positions:
                self.fixed_positions_mask_all[self.f1_to_all[pos-1]] = 1.0 # 3e8 + i
                self.fixed_positions_set[0][pos-1] = aa
        if self.config.protein2.fixed_positions is not None:
            for pos, aa in self.config.protein2.fixed_positions:
                self.fixed_positions_mask_all[self.f2_to_all[pos-1]] = 1.0 # 3e8 + i
                self.fixed_positions_set[1][pos-1] = aa        

    '''
    def absolute_to_protein(
        self, 
        pos: torch.Tensor, 
        protein_idx: int
    ) -> torch.Tensor:
        """Convert absolute position to protein-relative position"""
        if protein_idx == 0:
            return self.all_to_f1[pos]
        else:
            return self.all_to_f2[pos]
    
    def protein_to_absolute(
        self, 
        pos: torch.Tensor,
        protein_idx: int
    ) -> torch.Tensor:
        """Convert protein-relative position to absolute position"""
        if protein_idx == 0:
            return self.f1_to_all[pos]
        else:
            return self.f2_to_all[pos]

    def protein_to_other(
        self, 
        pos: torch.Tensor,
        protein_idx: int
    ) -> torch.Tensor:
        """Convert position relative to one protein to the position relative to the other"""
        if protein_idx == 0:
            return self.f1_to_f2[pos]
        else:
            return self.f2_to_f1[pos]
    '''