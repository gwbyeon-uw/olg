from tqdm import tqdm

from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
import numpy as np

from olg.constants import *
from olg.config import ProteinConfig

from .base_wrapper import BaseWrapper

class WrapperESM3(BaseWrapper):
    def __init__(
        self,
        model: nn.Module, 
        prefixed_seq: Optional[Tuple[int, int, str]] = None,
        **kwargs
    ):
        """
        """
        super().__init__(**kwargs)

        self.model = model
        self.tokenizer = self.model.tokenizers.sequence.vocab
        self.mask_idx = self.tokenizer['<mask>']
        self.cls_idx = self.tokenizer['<cls>']
        self.eos_idx = self.tokenizer['<eos>']
        self.vocab_size = len(self.tokenizer)
        self.alphabet_map = torch.tensor([ self.tokenizer[l] for l in Constants.ALPHABET ], device=self.device) #Index we use to ESM3 index
        self.alphabet_map_rev = torch.arange(self.vocab_size, device=self.device)
        self.alphabet_map_rev.fill_(-1)
        for a, i in self.tokenizer.items():
            if a in Constants.ALPHABET:
                self.alphabet_map_rev[i] = Constants.ALPHABET.index(a)
        
        self.L = self.config.length + 2
        self.prefixed_seq = prefixed_seq #List of tuples, (start, end, seq)
            
        tmp = torch.zeros(self.config.length, device=self.device) - 1 #Position relative to target protein
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos-1] = Constants.ALPHABET.index(aa)
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and AA index at fixed positions    
        
        self.reset(self.decoding_order, self.rand_base)    

    def _reset_decoding_order(self, decoding_order):
        self.decoding_order = decoding_order #This is relative to target chain's position in the OLG decoder. It is NOT the positions for X/S from PDB and need to be offseted
        self.end_pos = torch.max(self.decoding_order)
    
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None, seed_tracks=None):
        self.rand_base = rand_base
        self._reset_decoding_order(decoding_order)
        
        self.next_t = 0 #Iteration step; used as index for decoding orders

        self.current_logits = None
        self.decoded_positions = torch.zeros(self.config.length, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.config.length, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.config.length, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.config.length, self.vocab_size), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.config.length, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        if seed_S is not None:
            self.S = seed_S.clone()
        else:
            self.S = torch.zeros((1, self.config.length)).long().to(self.device) #Excluding <cls> and <eos>
            self.S = self.S.fill_(self.mask_idx)
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq) #This will update S, S_msa and decoded positions

        if seed_tracks is None: #Other than sequence_tokens
            '''
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.
            '''
            seed_tracks = { 'structure_tokens': None, 'ss8_tokens': None, 'sasa_tokens': None,
                            'function_tokens': None, 'residue_annotation_tokens': None,
                            'average_plddt': torch.tensor(1.0, device=self.device),
                            'per_res_plddt': torch.ones((1, self.L), device=self.device),
                            'structure_coords': None, 'chain_id': None, 'sequence_id': None }
        self.structure_tokens = seed_tracks['structure_tokens']
        self.ss8_tokens = seed_tracks['ss8_tokens']
        self.sasa_tokens = seed_tracks['sasa_tokens']
        self.function_tokens = seed_tracks['function_tokens']
        self.residue_annotation_tokens = seed_tracks['residue_annotation_tokens']
        self.average_plddt = seed_tracks['average_plddt']
        self.per_res_plddt = seed_tracks['per_res_plddt']
        self.structure_coords = seed_tracks['structure_coords']
        self.chain_id = seed_tracks['chain_id']
        self.sequence_id = seed_tracks['sequence_id']
        
        self.gap_map = torch.arange(self.decoding_order.shape[1], device=self.device) #Dummy
        self.gap_map_rev = self.gap_map.clone()

    def get_logits(self):
        input_seq = torch.nn.functional.pad(self.S, (1,1))
        input_seq[0, 0] = self.cls_idx
        input_seq[0, -1] = self.eos_idx
        out = self.model.forward(sequence_tokens=input_seq, 
                                 structure_tokens=self.structure_tokens, ss8_tokens=self.ss8_tokens, 
                                 sasa_tokens=self.sasa_tokens, function_tokens=self.function_tokens,
                                 residue_annotation_tokens=self.residue_annotation_tokens,
                                 average_plddt=self.average_plddt, per_res_plddt=self.per_res_plddt, 
                                 structure_coords=self.structure_coords, chain_id=self.chain_id, 
                                 sequence_id=self.sequence_id)
        return out.sequence_logits[:, 1:-1, 0:self.vocab_size]
        
    #Decode next step; returns AA logits
    def decode_next(self, dummy_run=False, mask_current=False, use_t=None):
        if use_t is not None:
            t = use_t
        else:
            t = self.decoding_order[0, self.next_t] #Decoding position, relative to target protein
        
        if dummy_run:
            self.current_pred = torch.zeros((self.config.length, self.vocab_size), device=self.device)
        else:
            if mask_current:
                self.S[0, t] = self.mask_idx
            self.current_pred = self.get_logits()[0]

        if t > -1:
            if self.config.force_stop and (t == self.end_pos): #Everything is zero except stop index
                logits = self._force_stop()
                return logits, logits
                
            self.current_logits = self.current_pred[t, :].unsqueeze(0) #Logits at current position, unless it's a stop and > length of protein
            
            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
                #logits_[:, Constants.STOP_INDEX] = Constants.MIN_LOGIT #Zero out the index for X
                logits = logits_.clone() 
            else:
                logits_ = self.current_logits.clone()[:, self.alphabet_map]
                logits_ -= logits_.mean()
                logits_[:, Constants.STOP_INDEX] = Constants.MIN_LOGIT
                logits = logits_.clone()
                logits = self._apply_constraints(logits, t)

            if (not self.config.force_stop) or (t != self.end_pos):
                logits_ = self._penalize_stop(logits_)
                logits = self._penalize_stop(logits)

            if self.fixed_positions[t] != -1:
                logits = self._force_fixed_positions(logits, t)

            logits = BaseWrapper._add_noise(logits)
            return logits, logits_
        
    def edit_S(self, t, S_t, inplace=False): #t here is relative to MSA; S is ProtMamba alphabet
        if inplace:
            S = self.S
        else:
            S = self.S.clone()

        if t < self.config.length:
            S[:, t] = S_t #Edit first row only
            
        if not inplace:
            return S
    
    #Update protein sequence vector S for next iteration
    def update_S(self, S_t, use_t=None, alphabet_map=True, dummy_run=False): #t here is relative to protein (no gap); S_t is ESM3
        if use_t is None:
            t = self.decoding_order[:, self.next_t]
            if self.config.force_stop and (t == self.end_pos):
                self.next_t += 1
                return False
            self.next_t += 1 #Moves to next t
        else:
            t = use_t

        if alphabet_map:
            S_t = self.alphabet_map[S_t]

        self.edit_S(t, S_t, inplace=True)
        self.decoded_positions[:, t] = 1.0
        self.selected_aa[:, t] = S_t
        log_softmax = torch.log(torch.nn.functional.softmax(self.current_logits[0], dim=-1))
        self.selected_log_prob[:, t] = log_softmax[S_t]
        self.log_prob[t, :] = log_softmax
        self.argmax_aa[:, t] = self.current_logits[0].argmax()
        return True
    
    #Update protein sequence vector S for fixing some regions that will not be part of OLG decoding
    def preset_fixed_S(self, fixed_start, fixed_end, fixed_seq):
        t = torch.arange(fixed_start, fixed_end + 1, device=self.device)
        fixed_token = self.alphabet_map[torch.tensor([ Constants.ALPHABET.index(c) for c in fixed_seq ], device=self.device)]
        self.edit_S(t, fixed_token, inplace=True) #t here not relative to MSA
        self.decoded_positions[:, t] = 1.0
    
    #Score the sequence by averaging log probabilities of each step
    def get_score(self, S=None, positions=None):
        if S is None: #To rescore with current sequence, use self.S.clone() as input
            S = self.S.clone()
        self.reset(self.decoding_order, self.rand_base, S)
        self.decode_all(use_S=S[0], mask_current=True)
        if positions is not None:
            return (self.selected_log_prob * -1.0)[0, positions].mean()
        return (self.selected_log_prob.mean() * -1.0)
        
    def get_prot_seq(self, S=None):
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.config.start_offset:self.config.length]]
        prot = ''.join([Constants.ALPHABET[s] for s in S])
        return prot

    #Decodes all; this is used to design non-overlapping proteins with the same parameters
    def decode_all(self, temp=1e-12, use_S=None, mask_current=False): #use_S is used to score a sequence. This should include gaps
        if not (self.next_t == 0):
            return False
        if use_S is None:
            for i in tqdm(range(self.decoding_order.shape[1]), disable=self.tqdm_disable):
                logits, logits_ = self.decode_next()
                probs = torch.nn.functional.softmax(logits/temp, dim=-1)
                S_t = torch.multinomial(probs[0], 1)
                self.update_S(S_t)
        else: #Gaps are already decoded at reset
            for i in tqdm(range(self.decoding_order.shape[1]), disable=self.tqdm_disable): 
                self.decode_next(mask_current=mask_current)
                t = self.decoding_order[:, i]
                if not (self.config.force_stop and (t == self.end_pos)):
                    S_t = use_S[self.gap_map[t]]
                else:
                    S_t = None
                self.update_S(S_t, alphabet_map=False)
        return True