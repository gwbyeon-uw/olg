import copy
import math
from typing import Dict, Tuple, List, Optional, Any
from tqdm import tqdm
import functools

import numpy as np
import torch
import torch.nn.functional as F

from constants import *
from config import *
from coordinates import *
from compatibility import *
    
from wrappers.proteinmpnn import *
from wrappers.gremlin import *
from wrappers.base_wrapper import ZeroOrderWrapper

try:
    from wrappers.coflow import *
    coflow_avail = True
except ImportError:
    print("Skipping CoFlow wrapper")
    coflow_avail = False

try:    
    from wrappers.esm3 import *
    esm3_avail = True
except ImportError:
    print("Skipping ESM3 wrapper")
    esm3_avail = False

try:
    from wrappers.evodiff import *
    evodiff_avail = True
except ImportError:
    print("Skiping EvoDiff wrapper")
    evodiff_avail = False

class OLGDesign():
    def __init__(
        self,
        config: DesignConfig
    ):
        self.config = config
        self.coords = Coordinates(self.config)
        self.compatibility = CodonCompatibility(self.config)
        
        self.decoders = [ None, None ]
        self.decoding_orders = [ None, None ]
        self.decoding_orders_full = [ None, None ]
        self.reset_decoding()

    def reset_decoding(
        self,
        user_order: Optional[torch.Tensor] = None,
        seed_S: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        seed_quartet_list: Optional[List[Optional[torch.Tensor]]] = None
    ) -> None:
        """
        Reset decoding state and reinitialize decoders for each protein.
       
        Args:
           user_order: User-provided decoding order (overrides random seed)
           seed_S: Initial sequence tensors for each protein
           seed_quartet_list: Initial quartet list to seed decoding
        """
        if user_order is not None:
            self.decoding_order_all = user_order #User provided order; supercedes user provided random seed. This has to be in absolute coordinates (quartet position number)
        else:
            if self.config.rand_base is not None:
                torch.manual_seed(self.config.rand_base) #User provided random number seed
            _order_score = torch.rand(self.coords.total_len, device=self.config.device)
            order_score = _order_score - self.coords.start_mask_all - self.coords.end_stop_mask_all - self.coords.fixed_positions_mask_all #Stop and start codons first; conflicting specifications are not checked!
    
            if self.config.decoding_mode == DecodingMode.OVERLAP_FIRST:
                order_score -= self.coords.overlap_mask_all
            elif self.config.decoding_mode == DecodingMode.OVERLAP_LAST:
                order_score += self.coords.overlap_mask_all
            self.decoding_order_all = torch.argsort(order_score)
            
        self.decoding_orders_full[0] = self.coords.all_to_f1[self.decoding_order_all].clone()
        self.decoding_orders[0] = self.decoding_orders_full[0][self.decoding_orders_full[0]!=-1].unsqueeze(0)
        self.decoding_orders_full[1] = self.coords.all_to_f2[self.decoding_order_all].clone()
        self.decoding_orders[1] = self.decoding_orders_full[1][self.decoding_orders_full[1]!=-1].unsqueeze(0)
        
        self.next_q = 0 #Iteration step
        if seed_quartet_list is None:
            self.quartet_list = [ None ] * self.coords.total_len #This tracks quartets; left-to-right order
        else:
            self.quartet_list = copy.deepcopy(seed_quartet_list)
            
        self.nuc = None
        self.unmasked_logits_joint = [] #This tracks joint probability matrix at each step
        self.masked_logits_joint = [] #This tracks joint probability matrix at each step masked by compatibility
        self.logits_f1 = [] #This tracks logits at each decoding step
        self.logits_f2 = [] 
        self.logits_f1_ = [] #This will track logits at each decoding step before applying various weights and filters
        self.logits_f2_ = []

        self.config.current_balancer_weight = self.config.balancer_unit
        
        if self.decoders[0] is not None:
            self.decoders[0].reset(self.decoding_orders[0], seed_S[0])
        if self.decoders[1] is not None:
            self.decoders[1].reset(self.decoding_orders[1], seed_S[1])

    def initialize_decoder(self, decoder_type: str, frame: int, model: Any, **kwargs):
        shared_params = {
            "device": self.config.device,
            "config": self.config.protein1 if frame == 0 else self.config.protein2,
            "decoding_order": self.decoding_orders[frame],
            "rand_base": self.config.rand_base,
            "tqdm_disable": self.config.tqdm_disable
        }

        kwargs.update(shared_params)
        
        decoder_classes = {
            "ProteinMPNN": WrapperProteinMPNN,
            "ZeroOrder": ZeroOrderWrapper,
            "ESM3": WrapperESM3 if esm3_avail else None,
            "CoFlow": WrapperCoFlow if coflow_avail else None,
            "EvoDiff": WrapperEvoDiff if evodiff_avail else None
        }
        
        if decoder_type in decoder_classes:
            self.decoders[frame] = decoder_classes[decoder_type](model, **kwargs)
        
    def swap_decoding_position(self, t_q_next: int) -> None:
        """
        Swap decoding order to bring specified position next.
        
        Args:
           t_q_next: Position to decode next
        """
        self.decoding_order_all[torch.where(self.decoding_order_all==t_q_next)[0][0]] = self.decoding_order_all[self.next_q]
        self.decoding_order_all[self.next_q] = t_q_next

        self.decoding_orders_full[0] = self.coords.all_to_f1[self.decoding_order_all].clone()
        self.decoding_orders[0] = self.decoding_orders_full[0][self.decoding_orders_full[0]!=-1].unsqueeze(0)
        self.decoding_orders_full[1] = self.coords.all_to_f2[self.decoding_order_all].clone()
        self.decoding_orders[1] = self.decoding_orders_full[1][self.decoding_orders_full[1]!=-1].unsqueeze(0)   

    @staticmethod
    def move_to_first(tensor, index):
        """
        Move the element at 'index' to the first position and shift others down.
        
        Args:
            tensor: Input tensor (1D)
            index: Index of element to move to first position
            
        Returns:
            Tensor with element at index moved to position 0
        """
        # Extract the element at the given index
        element = tensor[index:index+1]
        
        # Concatenate: element + tensor[:index] + tensor[index+1:]
        result = torch.cat([
            element,
            tensor[:index],
            tensor[index+1:]
        ])
        
        return result

    # Decode next step; this is the key function carrying out each step of iterative OLG decoding
    def decode_next(
        self,
        dummy_run: Tuple[bool] = (False, False),
        mask_current: Tuple[bool] = (False, False),
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
        t_f1 = self.coords.all_to_f1[t_q].unsqueeze(0) #Position of the current quartet, relative to protein 1; -1 means it's not overlapping
        t_f2 = self.coords.all_to_f2[t_q].unsqueeze(0) #Position of the current quartet, relative to protein 2; -1 means it's not overlapping
        t_q_p = t_q - 1 #Position of neighboring quartet to the left
        t_q_n = t_q + 1 #Position of neighboring quartet to the right
        t_q_p = None if t_q_p < 0 else t_q_p #When it's at the left-most position, t_q_p is None
        t_q_n = None if t_q_n >= self.coords.total_len else t_q_n #When it's at the right-most position, t_q_n is None

        self.quartet_list[t_q] = None
        
        #Get the logits and check if current position is overlapping or not
        overlapping_t = True
        if t_f1 != -1: #If the protein exists at this position, we get the next amino acid logit vector from the decoder
            logits_f1, logits_f1_ = self.decoders[0].decode_next(dummy_run[0], mask_current[0])
        else: 
            logits_f1 = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
            logits_f1_ = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
            overlapping_t = False #If the protein is not overlapping, we zero the logit vector
        if t_f2 != -1: #
            logits_f2, logits_f2_ = self.decoders[1].decode_next(dummy_run[1], mask_current[1])
        else:
            logits_f2 = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
            logits_f2_ = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
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
        q_p = torch.tensor([0, 1, 2, 3], device=self.config.device).long() #To allow all first nucleotide if previous position was not decoded yet
        if t_q_p is not None:
            if self.quartet_list[t_q_p] is not None:
                q_p = torch.unique(self.compatibility.prev_quartet_index[self.quartet_list[t_q_p]]) #Last nucleotide of the previous quartets

        #If next quartet was already decoded, then we need to consider this constraint
        q_n = torch.tensor([0, 1, 2, 3], device=self.config.device).long() #To allow all last nucleotide if next position was not decoded yet
        if t_q_n is not None:
            if self.quartet_list[t_q_n] is not None:
                q_n = torch.unique(self.compatibility.next_quartet_index[self.quartet_list[t_q_n]]) #First nucleotide of the previous quartets

        #All possible combinations of first and last NUCLEOTIDES; would be 4x4=16 if no previous/next positions were decoded
        p_n = torch.tensor([ (p, n) for p in q_p for n in q_n ]).long()

        compatibility = self.compatibility.codon_compatibility.clone()
        if self.config.protein1.force_start and (t_f1 == self.config.protein1.start_offset):
            compatibility *= self.compatibility.codon_compatibility_start_mask[0]
        if self.config.protein2.force_start and (t_f2 == self.config.protein2.start_offset):
            compatibility *= self.compatibility.codon_compatibility_start_mask[1]
        
        #Fixed position mask
        fixed_f1 = self.coords.fixed_positions_set[0][t_f1] if t_f1 != -1 else None
        fixed_f1_prev = self.coords.fixed_positions_set[0][t_f1-1] if (t_f1 > 0) else None
        fixed_f1_next = self.coords.fixed_positions_set[0][t_f1+1] if 0 < ((t_f1 + 1) < self.coords.f1_gap_len) else None
        fixed_f2 = self.coords.fixed_positions_set[1][t_f2] if t_f2 != -1 else None
        fixed_f2_prev = self.coords.fixed_positions_set[1][t_f2-1] if (t_f2 > 0) else None
        fixed_f2_next = self.coords.fixed_positions_set[1][t_f2+1] if 0 < ((t_f2 + 1) < self.coords.f2_gap_len) else None

        compatibility_safe = compatibility.clone()
        if not ((fixed_f1 == None) and (fixed_f1_prev == None) and (fixed_f1_next == None) and (fixed_f2 == None) and (fixed_f2_prev == None) and (fixed_f2_next == None)):
            compatible_q_i = self.compatibility.compatible_quartets_by_aa(
                self.config.arrangement, 
                (fixed_f1_prev, fixed_f1, fixed_f1_next), 
                (fixed_f2_prev, fixed_f2, fixed_f2_next), 
                self.compatibility.codon_table_rev
            )
            codon_compatibility_fixed_mask = torch.zeros(self.compatibility.codon_compatibility.shape, device=self.config.device).int()
            codon_compatibility_fixed_mask[:, :, :, :, :, compatible_q_i] = 1
            compatibility *= codon_compatibility_fixed_mask
            
        compatibility = (~(compatibility[p_n[:, 0], p_n[:, 1], self.config.arrangement, :, :, :].bool())) #Get compatibility matrix, for given first and last nucleotide of quartets
        
        quartets_logits_joint = logits_joint.repeat(compatibility.shape[0], 1, 1).unsqueeze(3).repeat(1, 1, 1, Constants.QUARTET_SIZE) #Joint logits, repeated so that we can mask with compatibility matrix
        quartets_logits_joint[compatibility] = Constants.MIN_LOGIT #Mask joint logits matrix with compatibility matrix
        masked_logits_joint = torch.clamp(quartets_logits_joint, min=Constants.MIN_LOGIT)
        
        if masked_logits_joint.max() == Constants.MIN_LOGIT: #Invalid case
            if force_safe:
                compatibility_safe = (~(compatibility_safe[p_n[:, 0], p_n[:, 1], self.config.arrangement, :, :, :].bool()))
                quartets_logits_joint = logits_joint_safe.repeat(compatibility_safe.shape[0], 1, 1).unsqueeze(3).repeat(1, 1, 1, Constants.QUARTET_SIZE)
                quartets_logits_joint[compatibility_safe] = Constants.MIN_LOGIT #Mask joint logits matrix with compatibility matrix
                masked_logits_joint = torch.clamp(quartets_logits_joint, min=Constants.MIN_LOGIT)
            else:
                print("Invalid; no available choice")
                self.errored_compat = compatibility
                self.errored_next_q = self.next_q
                return False
            
        self.masked_logits_joint += [ masked_logits_joint.clone().detach() ]
                
        #This implements top-p decoding
        masked_logits_joint_amax = masked_logits_joint.amax([0, 3]) #Collapse to AAs only
        sort_v_, sort_ind = masked_logits_joint_amax.flatten().sort(descending=True) #Sort by logits
        sort_v = torch.nn.functional.softmax(sort_v_/self.config.temperature, dim=-1) #Apply temperature and softmax
        sort_v_cumsum = sort_v.cumsum(0) #Get cumulative probability of ranked probs for top-P sampling
        cutoff_ind = torch.nonzero(sort_v_cumsum>self.config.top_p)[0][0] + 1 #Top-P cutoff
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
            
            if self.config.complexed:
                t_full = self.decoders[0].decoding_order[0, self.decoders[0].next_t_full]
                t_ = self.decoders[0].decoding_order_target[0, self.decoders[0].next_t*self.decoders[0].n_design_chains]

                if self.decoders[0].tied:
                    tied_list = self.decoders[0].tied_pos[t_]
                else:
                    tied_list = [ t_ ]
                for t in tied_list:
                    self.decoders[1].edit_S(t, f1_S_t, inplace=True) #Decodes from the other frame if it was part of same complex
            
            elif self.config.shared:
                self.decoders[1].edit_S(t_f1, f1_S_t, inplace=True)
            
            self.decoders[0].update_S(f1_S_t, dummy_run=dummy_run[0])
        
        if t_f2 != -1:
            f2_S_t = best_q[2][0].unsqueeze(0).unsqueeze(0)
            
            if self.config.complexed:
                t_full = self.decoders[1].decoding_order[0, self.decoders[1].next_t_full]
                t_ = self.decoders[1].decoding_order_target[0, self.decoders[1].next_t*self.decoders[1].n_design_chains]
            
                if self.decoders[1].tied:
                    tied_list = self.decoders[1].tied_pos[t_]
                else:
                    tied_list = [ t_ ]
                for t in tied_list:
                    self.decoders[0].edit_S(t, f2_S_t, inplace=True) #Decodes from the other frame if it was part of same complex
                
            elif self.config.shared:
                self.decoders[0].edit_S(t_f2, f2_S_t, inplace=True)
                    
            self.decoders[1].update_S(f2_S_t, dummy_run=dummy_run[1])
        
        best_q = [ q.cpu() for q in best_q ]
        
        #Check that previous position quartets are compatible with current position quartets. This has to be recursive, since choice at each position affects its neighbors
        if (t_q_p is not None) and (self.quartet_list[t_q_p] is not None):
            compatible_prev = np.intersect1d(self.quartet_list[t_q_p], torch.stack([ self.compatibility.compatible_prev_quartets[q.item()] for q in best_q[3] ]).flatten().unique().cpu())
            self.quartet_list[t_q_p] = compatible_prev
            
            #Recursively check all quartets connected to it
            t_q_p_i = t_q_p - 1
            while (t_q_p_i >= 0) and (self.quartet_list[t_q_p_i] is not None) and (self.quartet_list[t_q_p_i].shape[0] > 1):
                compatible_prev = np.intersect1d(self.quartet_list[t_q_p_i], torch.stack([ self.compatibility.compatible_prev_quartets[q.item()] for q in self.quartet_list[t_q_p_i+1]]).flatten().unique().cpu())
                self.quartet_list[t_q_p_i] = compatible_prev
                t_q_p_i -= 1
        
        #Check that next position quartets are compatible with current position quartets
        if (t_q_n is not None) and (self.quartet_list[t_q_n] is not None):
            compatible_next = np.intersect1d(self.quartet_list[t_q_n], torch.stack([ self.compatibility.compatible_next_quartets[q.item()] for q in best_q[3] ]).flatten().unique().cpu())
            self.quartet_list[t_q_n] = compatible_next
            
            #Recursively check all quartets connected to it
            t_q_n_i = t_q_n + 1
            while (t_q_n_i < self.coords.total_len) and (self.quartet_list[t_q_n_i] is not None) and (self.quartet_list[t_q_n_i].shape[0] > 1):
                compatible_next = np.intersect1d(self.quartet_list[t_q_n_i], torch.stack([ self.compatibility.compatible_next_quartets[q.item()] for q in self.quartet_list[t_q_n_i-1]]).flatten().unique().cpu())
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
        nt_p_1s = self.compatibility.prev_quartet_index[quartet_list[0]]
        nt_qn_1s = self.compatibility.next_quartet_index[quartet_list[1]]
        for n in range(len(nt_p_1s)):
            if nt_p_1s[n] in nt_qn_1s:
                acceptable += [ n ]
        quartet_list[0] = quartet_list[0][np.random.choice(acceptable)]

        #Second to second-last quartet; look to neighboring quartets and choose randomly among the acceptable (connecting) quartets
        for q in range(1, len(quartet_list)-1, 1):
            acceptable = []
            nt_q_1s = self.compatibility.next_quartet_index[quartet_list[q]]
            nt_p_1s = self.compatibility.prev_quartet_index[quartet_list[q]]
            nt_qp_4 = self.compatibility.prev_quartet_index[quartet_list[q-1]]
            nt_qn_1s = self.compatibility.next_quartet_index[quartet_list[q+1]]    
            for n in range(len(nt_q_1s)):
                if nt_q_1s[n] == nt_qp_4:
                    if nt_p_1s[n] in nt_qn_1s:
                        acceptable += [ n ]
            quartet_list[q] = quartet_list[q][np.random.choice(acceptable)]

        #last quartet; look to previous quartet and choose randomly among the acceptable (connecting) quartets
        acceptable = []
        nt_q_1s = self.compatibility.next_quartet_index[quartet_list[len(quartet_list)-1]]
        nt_qp_4 = self.compatibility.prev_quartet_index[quartet_list[len(quartet_list)-2]]
        for n in range(len(nt_q_1s)):
            if nt_q_1s[n] == nt_qp_4:
                acceptable += [ n ]
        quartet_list[len(quartet_list)-1] = quartet_list[len(quartet_list)-1][np.random.choice(acceptable)]

        #Trim last nucleotide if in-phase overlap (-0)
        final_nuc = ''.join([ Constants.QUARTETS[q][0:3] for q in quartet_list ] + [ Constants.QUARTETS[quartet_list[len(quartet_list)-1]][-1] ])
        self.nuc = torch.tensor([ Constants.NUCLEOTIDES.index(c) for c in final_nuc ], device=self.config.device)
        return final_nuc, quartet_list

    def get_prot_seq(self) -> Tuple[str]:
        f1_prot = self.decoders[0].get_prot_seq()
        f2_prot = self.decoders[1].get_prot_seq()
        return f1_prot, f2_prot
    
    def get_scores(
        self, 
        positions: Tuple[Optional[torch.Tensor]] = (None, None)
    ) -> Tuple[float]:
        f1_score = self.decoders[0].get_score(positions=positions[0])
        f2_score = self.decoders[1].get_score(positions=positions[1])
        return f1_score, f2_score
    
    def decode_all(
        self,
        dummy_run: Tuple[bool] = (False, False),
        mask_current: Tuple[bool] = (False, False),
        retry: int = 0,
        force_safe: bool = False,
        dynamic_order: Optional[str] = None
        ) -> bool:
        if retry > 0:
            seed_S = (self.decoders[0].S.clone(), self.decoders[1].S.clone())
        else:
            seed_S = None

        current_try = 0
        while current_try <= retry:
            for i in tqdm(range(self.coords.total_len), disable=self.config.tqdm_disable):
                valid = self.decode_next(dummy_run=dummy_run, mask_current=mask_current, force_safe=force_safe)
                if (not valid) and (retry > 0):
                    #print("invalid, retrying")
                    new_order = self.move_to_first(self.decoding_order_all, self.errored_next_q)
                    self.reset_decoding(user_order=new_order, seed_S=seed_S)
                    current_try += 1
                    break
                    
                if dynamic_order is not None:
                    if (self.next_q > 0) and (self.next_q < self.coords.total_len):
                        next_q = self.get_next_order_dyn(dynamic_order, frames=(True, True))
                        self.swap_decoding_position(next_q)
                        self.decoders[0]._reset_decoding_order(self.decoding_orders[0])
                        self.decoders[1]._reset_decoding_order(self.decoding_orders[1])

            if valid:
                return True 
                
        return False
            
    def _map_score_positions(self, f1_score: torch.Tensor, f2_score: torch.Tensor) -> torch.Tensor:
        """Helper to handle coordinate mapping between absolute position and protein-relative position for decoding order function"""
        positions = torch.zeros((2, self.coords.total_len), device=self.config.device)
        
        f1_abs_position = self.coords.f1_to_all[self.coords.f1_to_all!=(self.coords.f1_to_all.max()+1-(self.config.protein1.force_stop+0))]
        f1_protein_position = self.decoders[0].gap_map_rev[self.decoders[0].gap_map_rev!=-1]

        f2_abs_position = self.coords.f2_to_all[self.coords.f2_to_all!=(self.coords.f2_to_all.max()+1-(self.config.protein2.force_stop+0))]
        f2_protein_position = self.decoders[1].gap_map_rev[self.decoders[1].gap_map_rev!=-1]
        
        positions[0, f1_abs_position] = f1_score[f1_protein_position]
        positions[1, f2_abs_position] = f2_score[f2_protein_position]
        
        return positions
    
    def _apply_masks_and_sort(self, positions: torch.Tensor, priortize_fixed: bool):
        """Apply masks and sort positions to get decoding order"""
        positions = positions.mean(0)
        positions = positions / positions.max()  # Scale values so that highest value is 1
        positions = positions - self.coords.start_mask_all - self.coords.end_stop_mask_all
        
        if priortize_fixed:
            positions = positions - self.coords.fixed_positions_mask_all
        
        if self.config.decoding_mode == DecodingMode.OVERLAP_FIRST:
            positions = positions - self.coords.overlap_mask_all
        elif self.config.decoding_mode == DecodingMode.OVERLAP_LAST:
            positions = positions + self.coords.overlap_mask_all
        
        return torch.argsort(positions)

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
        if ordering == "entropy":
            if frames[0]:
                f1_curr_pred = self.decoders[0].current_pred[self.config.protein1.start_offset:self.config.protein1.length, :]
                f1_log_prob = torch.log(torch.nn.functional.softmax(f1_curr_pred, dim=-1))[:, self.decoders[0].alphabet_map]
            else:
                f1_log_prob = torch.zeros(self.decoders[0].current_pred.shape, device=self.config.device)[self.config.protein1.start_offset:self.config.protein1.length, self.decoders[0].alphabet_map]
            if frames[1]:
                f2_curr_pred = self.decoders[1].current_pred[self.config.protein2.start_offset:self.config.protein2.length, :]
                f2_log_prob = torch.log(torch.nn.functional.softmax(f2_curr_pred, dim=-1))[:, self.decoders[1].alphabet_map]
            else:
                f2_log_prob = torch.zeros(self.decoders[1].current_pred.shape, device=self.config.device)[self.config.protein2.start_offset:self.config.protein2.length, self.decoders[1].alphabet_map]

            f1_entropy = -1.0 * torch.sum(torch.exp(f1_log_prob) * f1_log_prob, 1)
            f2_entropy = -1.0 * torch.sum(torch.exp(f2_log_prob) * f2_log_prob, 1)
            
            positions = self._map_score_positions(f1_entropy, f2_entropy)
            
        elif ordering == "prob":
            if frames[0]:
                f1_curr_pred = self.decoders[0].current_pred[self.config.protein1.start_offset:self.config.protein1.length, :]
                f1_max_log_prob = torch.log(torch.nn.functional.softmax(f1_curr_pred, dim=-1))[:, self.decoders[0].alphabet_map].max(-1)[0]
            else:
                f1_max_log_prob = torch.zeros(self.decoders[0].current_pred.shape, device=self.config.device)[self.config.protein1.start_offset:self.config.protein1.length]
            if frames[1]:
                f2_curr_pred = self.decoders[1].current_pred[self.config.protein2.start_offset:self.config.protein2.length, :]
                f2_max_log_prob = torch.log(torch.nn.functional.softmax(f2_curr_pred, dim=-1))[:, self.decoders[1].alphabet_map].max(-1)[0]
            else:
                f2_max_log_prob = torch.zeros(self.decoders[1].current_pred.shape, device=self.config.device)[self.config.protein2.start_offset:self.config.protein2.length]

            positions = self._map_score_positions(f1_max_log_prob, f2_max_log_prob)

        next_order = self._apply_masks_and_sort(positions, priortize_fixed)
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
        if ordering == "random":
            next_order = self.decoding_order_all.clone()
            idx = torch.randperm(next_order.nelement())
            return next_order[idx]
        elif ordering == "orig":
            return self.decoding_order_all.clone()

        positions = torch.zeros((2, self.coords.total_len), device=self.config.device)
        
        if ordering == "entropy":
            f1_log_prob = self.decoders[0].log_prob[:, self.decoders[0].alphabet_map]
            f2_log_prob = self.decoders[1].log_prob[:, self.decoders[1].alphabet_map]
            f1_entropy = -1.0 * torch.sum(torch.exp(f1_log_prob) * f1_log_prob, 1)
            f2_entropy = -1.0 * torch.sum(torch.exp(f2_log_prob) * f2_log_prob, 1)
            positions = self._map_score_positions(f1_entropy, f2_entropy)
        elif ordering == "prob":
            f1_selected_log_prob = self.decoders[0].selected_log_prob[0]
            f2_selected_log_prob = self.decoders[1].selected_log_prob[0]
            positions = self._map_score_positions(f1_selected_log_prob, f2_selected_log_prob)
        elif ordering == "prob_rank":
            prob_1 = self.decoders[0].selected_log_prob[0]
            prob_2 = self.decoders[1].selected_log_prob[0]
            prob_rank_1 = self.decoders[0].log_prob.sort(1)[1].gather(1, self.decoders[0].S.permute([1,0]))[:,0]
            prob_rank_2 = self.decoders[1].log_prob.sort(1)[1].gather(1, self.decoders[1].S.permute([1,0]))[:,0]
            prob_rank_1 += prob_1 * Constants.EPS
            prob_rank_2 += prob_2 * Constants.EPS
            positions = self._map_score_positions(prob_rank_1, prob_rank_2)
            
        return self._apply_masks_and_sort(positions, priortize_fixed)

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
            
            if (torch.sign(sd1) == torch.sign(sd0)) and (torch.abs(sd0) > self.config.balancer_threshold):
                self.config.current_balancer_weight = self.config.current_balancer_weight + self.config.balancer_unit
            else:
                self.config.current_balancer_weight = self.config.balancer_unit
        else:
            self.config.current_balancer_weight = self.config.balancer_unit

        diff = torch.abs(sd0)
        weight = min(self.config.balancer_max_weight, diff * self.config.current_balancer_weight + 1.0)
        if scores_pll[-1][1] > scores_pll[-1][0]:
            w1 = torch.ones(self.decoders[0].logit_weight.shape, device=self.decoders[0].logit_weight.device)
            w2 = torch.ones(self.decoders[1].logit_weight.shape, device=self.decoders[1].logit_weight.device) * weight
        else:
            w1 = torch.ones(self.decoders[0].logit_weight.shape, device=self.decoders[0].logit_weight.device) * weight
            w2 = torch.ones(self.decoders[1].logit_weight.shape, device=self.decoders[1].logit_weight.device)
        return w1, w2
        
    def decode_all_gibbs(
        self,
        next_order: Optional[torch.Tensor] = None,
        weight: Tuple[torch.Tensor] = None,
        seed_S: Tuple[torch.Tensor ] = None,
        seed_quartet: bool = False,
        force_safe: bool = False,
        dummy_run: Tuple[bool, bool] = (False, False),
        dynamic_order: Optional[str] = None,
        retry: int = 0
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
        if seed_S is None:
            seed_S = (self.decoders[0].S.clone(), self.decoders[1].S.clone())
        self.reset_decoding(user_order=next_order, seed_S=seed_S, seed_quartet_list=seed_quartet_list)
        self.decoders[0].logit_weight = w1
        self.decoders[1].logit_weight = w2
        self.decode_all(dummy_run=dummy_run, mask_current=(True, True), force_safe=force_safe, retry=retry, dynamic_order=dynamic_order) #Run decoding with current position masking

    def validate(self, keep_S: bool = False, retry: int = 10) -> bool:
        """
        Do a dry run to check if fixed positions, stop, and start codons are valid.
        """
        valid = self.decode_all(dummy_run=(True, True), force_safe=False, retry=retry)
        if keep_S:
            self.reset_decoding(user_order=self.decoding_order_all, seed_S=(self.decoders[0].S.clone(), self.decoders[1].S.clone()))
        else:
            self.reset_decoding(user_order=self.decoding_order_all)
        return valid
    
    #Check if fixed positions/stop/start make sense
    def validate_fixed(
        self,
        reset: bool = True,
        print_error: bool = True,
    ) -> bool:
        """
        Do a dry run to check if fixed positions, stop, and start codons are valid.
        
        Args:
           reset: Whether to reset after validation
           print_error: Whether to print the failures
           
        Returns:
           tuple: (success_flag, list of failed positions)
        """
        self.decode_all(dummy_run=(True, True), force_safe=True)
            
        S_f1, S_f2 = self.get_prot_seq()
        nuc_seq, quartets = self.string_quartet()
        
        failed = False
        failed_res = []
        if self.config.protein1.fixed_positions is not None:
            for pos, aa in self.config.protein1.fixed_positions:
                if aa != S_f1[pos-1]:
                    failed = True
                    failed_res += [ (0, pos, aa) ]
                    if print_error:
                        print("Fixed residue could not be placed for protein 1: "+str(pos)+" "+aa)

        if self.config.protein2.fixed_positions is not None:
            for pos, aa in self.config.protein2.fixed_positions:
                if aa != S_f2[pos-1]:
                    failed = True
                    failed_res += [ (1, pos, aa) ]
                    if print_error:
                        print("Fixed residue could not be placed for protein 2: "+str(pos)+" "+aa)
                    
        if self.config.protein1.force_stop:
            q_i = quartets[self.coords.f1_to_all[-1]]
            aa_f1 = self.compatibility.quartets_aa[q_i][FRAME_F1[self.config.arrangement]] 
            aa_i = Constants.STOP_INDEX
            if aa_f1 != aa_i:
                failed = True
                failed_res += [ (0, None, 'Stop') ]
                if print_error:
                    print("Stop could not be placed for protein 1")

        if self.config.protein2.force_stop:
            q_i = quartets[self.coords.f2_to_all[-1]]
            aa_f2 = self.compatibility.quartets_aa[q_i][FRAME_F2[self.config.arrangement]] 
            aa_i = Constants.STOP_INDEX
            if aa_f2 != aa_i:
                failed = True
                failed_res += [ (1, None, 'Stop') ]
                if print_error:
                    print("Stop could not be placed for protein 2")
                
        if self.config.protein1.force_start:
            q_i = quartets[self.coords.f1_to_all[0]]
            if q_i not in self.compatibility.start_codons_quartets[0]:
                failed = True
                failed_res += [ (0, 1, 'Start') ]
                if print_error:
                    print("Start could not be placed for protein 1")
                
        if self.config.protein2.force_start:
            q_i = quartets[self.coords.f2_to_all[0]]
            if q_i not in self.compatibility.start_codons_quartets[1]:
                failed = True
                failed_res += [ (0, 1, 'Start') ]
                if print_error:
                    print("Start could not be placed for protein 2")
        
        if reset:
            self.reset_decoding(user_order=self.decoding_order_all, seed_S=(self.decoders[0].S.clone(), self.decoders[1].S.clone()))
        return (not failed, failed_res)

    @staticmethod
    def per_position_entropy(indices: torch.Tensor, num_categories: int, epsilon: float = 1e-10):
        """
        Calculate per-position entropy for a BxL tensor of categorical indices.
        
        Args:
            indices: torch.Tensor of shape (B, L) containing category indices
            num_categories: int, total number of possible categories
            
        Returns:
            torch.Tensor of shape (L,) containing entropy at each position
        """
        B, L = indices.shape
        one_hot = F.one_hot(indices, num_classes=num_categories).float()
        probs = one_hot.mean(dim=0)
        entropy = -(probs * torch.log(probs + epsilon)).sum(dim=-1)
        return entropy

### Unused
'''
    def nuc_pos_to_aa_pos(
        self,
        nuc_pos: int
    ) -> Tuple[Tuple[Optional[int]]]:
        """
        Given nucleotide position, get amino acid positions relative to each protein
        
        Args:
           nuc_pos: Nucleotide position
           
        Returns:
           tuple: ((f1_pos, f1_pos_target, f1_codon_pos), 
                   (f2_pos, f2_pos_target, f2_codon_pos))
        """
        f1_offset, f2_offset, reverse = Constants.ARRANGEMENT_CONFIG[self.config.arrangement]
        
        # Calculate f1 position
        f1_pos = math.floor((nuc_pos - f1_offset) / 3)
        f1_pos_target = f1_pos - self.coords.f1_start
        f1_codon_pos = (nuc_pos - f1_offset) % 3
        f1_in_range = (f1_pos >= self.coords.f1_start) and (f1_pos < self.coords.f1_end) and \
                      (f1_pos_target >= 0) and (f1_pos_target < (self.config.protein1.length + self.config.protein1.force_stop))
        
        # Calculate f2 position
        f2_pos = math.floor((nuc_pos - f2_offset) / 3)
        f2_pos_target = (self.coords.f2_end - 1 - f2_pos) if reverse else (f2_pos - self.coords.f2_start)
        f2_codon_pos = (nuc_pos - f2_offset) % 3
        f2_in_range = (f2_pos >= self.coords.f2_start) and (f2_pos < self.coords.f2_end) and \
                      (f2_pos_target >= 0) and (f2_pos_target < (self.config.protein2.length + self.config.protein2.force_stop))
        
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
        
        logits_f1 = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
        logits_f1_ = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
        logits_f2 = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
        logits_f2_ = torch.zeros((1, Constants.ALPHABET_SIZE), device=self.config.device)
        f1_all_aa = torch.zeros(Constants.NUCLEOTIDE_SIZE, device=self.config.device).long()
        f2_all_aa = torch.zeros(Constants.NUCLEOTIDE_SIZE, device=self.config.device).long()

        if f1_pos is not None:
            f1_codon_start = nuc_pos - f1_codon_pos
            f1_codon_end = f1_codon_start + 3
            f1_codon = nuc[f1_codon_start:f1_codon_end].unsqueeze(0) #if positive; else revcomp
            f1_all_codons = f1_codon.repeat((4, 1)) #All 4 possible nucleotides
            f1_all_codons[:, f1_codon_pos] = torch.arange(4, device=self.config.device) 
            f1_all_aa = self.compatibility.codon_to_aa[f1_all_codons[:, 0], f1_all_codons[:, 1], f1_all_codons[:, 2]] #Translate
            if f1_pos_target < self.decoders[0].decoded_positions.shape[1]:
                self.decoders[0].decoded_positions[0, f1_pos_target] = 0
            logits_f1, logits_f1_ = self.decoders[0].decode_next(mask_current=True, use_t=f1_pos_target)
        if f2_pos is not None:
            f2_codon_start = nuc_pos - f2_codon_pos
            f2_codon_end = f2_codon_start + 3
            f2_codon = nuc[f2_codon_start:f2_codon_end].unsqueeze(0) #if positive; else revcomp
            f2_all_codons = f2_codon.repeat((4, 1))
            f2_all_codons[:, f2_codon_pos] = torch.arange(4, device=self.config.device)
            if self.f2_neg:
                f2_all_aa = self.compatibility.codon_to_aa_rc[f2_all_codons[:, 0], f2_all_codons[:, 1], f2_all_codons[:, 2]]
            else:
                f2_all_aa = self.compatibility.codon_to_aa[f2_all_codons[:, 0], f2_all_codons[:, 1], f2_all_codons[:, 2]]
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
        next_order_nuc = torch.tensor(pos_ind_uniq[pos_ind_uniq_ind.argsort()], device=self.config.device)
        
        weight = min(aw_max, scores[1] / scores[0] - 1.0) * aw_scale + 1.0
        w1 = self.decoders[0].logit_weight
        w2 = self.decoders[1].logit_weight * weight * 100

        for nuc_pos in tqdm(next_order_nuc, disable=self.config.tqdm_disable):
            self.best_nuc_change(nuc_pos)
'''