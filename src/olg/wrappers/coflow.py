from tqdm import tqdm

from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from esm.tokenization import EsmSequenceTokenizer
import esm.utils.constants.esm3 as C

from ._vendored.CoFlow.model import CoFlowModel

from olg.constants import *
from olg.config import ProteinConfig
from .base_wrapper import BaseWrapper

class WrapperCoFlow(BaseWrapper):
    # No default remapping: standard OLG letters (including 'X') are in ESM3's tokenizer
    _DEFAULT_EXTRA_AA_MAP: dict[str, str] = {}

    def __init__(
        self,
        model: nn.Module,
        prefixed_seq: Optional[Tuple[int, int, str]] = None,
        sample_struct: bool = True,
        sample_struct_temp: float = 0.7,
        extra_aa_map: Optional[dict[str, str]] = None,
        **kwargs
    ):
        """
        """
        super().__init__(**kwargs)

        self.model = model
        self.sample_struct = sample_struct
        self.sample_struct_temp = sample_struct_temp
        self.tokenizer = EsmSequenceTokenizer().vocab
        self.mask_idx = C.SEQUENCE_MASK_TOKEN
        self.bos_idx = C.SEQUENCE_BOS_TOKEN
        self.eos_idx = C.SEQUENCE_EOS_TOKEN
        self.struct_mask_idx = C.STRUCTURE_MASK_TOKEN
        self.struct_bos_idx = C.STRUCTURE_BOS_TOKEN
        self.struct_eos_idx = C.STRUCTURE_EOS_TOKEN
        self.vocab_size = len(self.tokenizer)
        self._build_alphabet_maps(self.tokenizer, extra_aa_map, self._DEFAULT_EXTRA_AA_MAP)

        self.prefixed_seq = prefixed_seq #List of tuples, (start, end, seq)

        tmp = torch.zeros(self.config.length, device=self.device) - 1 #Position relative to target protein
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos-1] = self.alphabet_index[aa]
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and OLG-internal AA index at fixed positions
        
        self.reset(self.decoding_order, self.rand_base)

    @staticmethod
    def _load_coflow_model(device: torch.device, weight_path: str):
        model = CoFlowModel.from_pretrained(weight_path)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        return model

    def _reset_decoding_order(self, decoding_order):
        self.decoding_order = decoding_order #This is relative to target chain's position in the OLG decoder. It is NOT the positions for X/S from PDB and need to be offseted
        self.end_pos = torch.max(self.decoding_order)
    
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None, seed_structure=None):
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

        if seed_structure is None: #Other than sequence_tokens
            self.structure_tokens = torch.ones((1, self.config.length)).long().to(self.device)
            self.structure_tokens = self.structure_tokens.fill_(C.STRUCTURE_MASK_TOKEN)
        else:
            self.structure_tokens = seed_structure
        
        self.gap_map = torch.arange(self.decoding_order.shape[1], device=self.device) #Dummy
        self.gap_map_rev = self.gap_map.clone()
        self.dt = 1.0 / self.config.length
        self.last_struct_logits = None

    def pad_seq(self, input_seq: torch.Tensor, input_type="sequence"):
        input_seq = torch.nn.functional.pad(input_seq, (1,1))
        input_seq[0, 0] = self.bos_idx if input_type == "sequence" else self.struct_bos_idx
        input_seq[0, -1] = self.eos_idx if input_type == "sequence" else self.struct_eos_idx
        return input_seq
        
    def get_logits(self):
        tc = self.get_current_tc(self.S)
        struct_logits, seq_logits = self.model.denoise(structure=self.structure_tokens, sequence=self.S, t=tc)
        return seq_logits[0, :, 0:self.vocab_size], struct_logits[0]

    def get_current_tc(self, S):
        tc = (1.0 - (S[0, ] == self.mask_idx).sum() * self.dt).unsqueeze(0).unsqueeze(0) #Timestep for the model, linear; 1 - masked proportion
        return tc
        
    #Decode next step; returns AA logits
    def decode_next(self, dummy_run=False, mask_current=False, use_t=None, use_current_pred=False, only_pred=False):
        if use_t is not None:
            t = use_t
        else:
            t = self.decoding_order[0, self.next_t] #Decoding position, relative to target protein
        
        if dummy_run:
            self.current_pred = torch.zeros((self.config.length, self.vocab_size), device=self.device)
        else:
            if mask_current:
                self.S[0, t] = self.mask_idx
            if not use_current_pred:
                self.current_pred, self.current_pred_struct = self.get_logits()

        if only_pred:
            return None, None

        if t > -1:
            if self.config.force_stop and (t == self.end_pos): #Everything is zero except stop index
                logits = self._force_stop()
                return logits, logits
                
            self.current_logits = self.current_pred[t, :].unsqueeze(0) #Logits at current position, unless it's a stop and > length of protein
            
            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, self.stop_index] = Constants.MIN_LOGIT #Zero out the index for X
                logits = logits_.clone() 
            else:
                logits_ = self.current_logits.clone()[:, self.alphabet_map]
                logits_ -= logits_.mean()
                logits_[:, self.stop_index] = Constants.MIN_LOGIT
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

        if self.sample_struct:
            self.sample_struct_single(t, dummy_run=dummy_run)

        return True

    def sample_struct_single(self, t, dummy_run=False):
        if not dummy_run:
            if t < self.config.length:
                self.structure_tokens[0, t] = C.STRUCTURE_MASK_TOKEN
                _, struct_logits = self.get_logits()
                struct_probs = torch.softmax(struct_logits/self.sample_struct_temp, dim=-1)    
                self.structure_tokens[0, t] = Categorical(struct_probs[t, :]).sample()

    def sample_struct_all(self, n_steps: int, temp: float, eta: float, purity: bool):
        structure = self.structure_tokens.clone()
        structure_mask = structure == C.STRUCTURE_MASK_TOKEN
        for idx in range(n_steps):
            t = torch.Tensor([[idx/n_steps]]).to(self.device)
            struc_logits, _ = self.model.denoise(structure=structure, sequence=self.S, t=t)
            struc_probs = torch.softmax(struc_logits/self.sample_struct_temp, dim=-1)
            structure = self.model.flow._sample_next_single(
                probs=struc_probs, xt=self.structure_tokens, mask_token=C.STRUCTURE_MASK_TOKEN, mask=structure_mask,
                N=n_steps, step=idx, eta=eta, purity=purity, sample=True)
        return structure
    
    #Update protein sequence vector S for fixing some regions that will not be part of OLG decoding
    def preset_fixed_S(self, fixed_start, fixed_end, fixed_seq):
        t = torch.arange(fixed_start, fixed_end + 1, device=self.device)
        fixed_token = self.alphabet_map[torch.tensor([ self.alphabet_index[c] for c in fixed_seq ], device=self.device)]
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
        prot = ''.join([self.alphabet[s.item()] for s in S])
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