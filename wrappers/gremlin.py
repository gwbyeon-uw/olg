from tqdm import tqdm

import numpy as np
import torch

from constants import *
from config import ProteinConfig

from .base_wrapper import BaseWrapper

#gremlin, pytorch version
#In pytorch, based on https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF.ipynb
class GREMLIN(torch.nn.Module):
    def __init__(self, L, A):
        super(GREMLIN, self).__init__()
        self.L = L
        self.A = A
        self.W = torch.nn.Parameter(torch.zeros(self.L, self.A, self.L, self.A), requires_grad=True) #2-body-term; W
        self.V = torch.nn.Parameter(torch.zeros(self.L, self.A), requires_grad=True) #1-body-term; V
        
    def get_pll(self, X, temp):
        VW = self.V + torch.tensordot(X, self.W, 2) #predicted
        H = torch.mul(X, VW).sum((1, 2)) / temp #Hamiltonian / E
        Z = torch.sum(torch.logsumexp(VW, 2), 1) #Local Z
        PLL = H - Z #Pseudolikelihood
        return PLL
    
    def forward(self, msa_onehot, temp=1.0, weights=None, neff=None, regularized_loss=False, reg_str=0.01): #where X is one hot MSA
        PLL = self.get_pll(msa_onehot, temp)

        if regularized_loss:
            #Regularization
            L2_V = reg_str * torch.sum(torch.square(self.V))
            L2_W = reg_str * torch.sum(torch.square(self.W)) * 0.5 * (self.L-1) * (self.A-1)
            
            #loss
            if weights is not None:
                loss = -torch.sum(PLL*weights)/torch.sum(weights)
            else:
                loss = -torch.sum(PLL)
            if neff is not None:
                loss = loss + (L2_V + L2_W)/neff
            else:
                loss = loss + (L2_V + L2_W)
            return loss
            
        else:
            return PLL     

#Helper class to handle generate a protein sequence with GREMLIN
class WrapperGREMLIN():
    def __init__(
        self, 
        model: torch.nn.Module,
        temperature: float = 1.0,
        prefixed_seq: Optional[Tuple[int, int, str]] = None,
    ):
        super().__init__(**kwargs)
        
        self.model = model
        self.model = self.model.eval()
        self.model.requires_grad_(False)

        self.W = self.model.W
        self.V = self.model.V
        self.temp = temperature
        
        #MSA subsampling        
        self.alphabet_map = torch.tensor([ Constants.GREMLIN_ALPHABET[l] for l in Constants.ALPHABET_GAP ], device=self.device) #Index we use to model index
        self.alphabet_map_rev = torch.tensor([ Constants.ALPHABET_GAP.index(a) if a in Constants.ALPHABET_GAP else -1 for a in Constants.GREMLIN_ALPHABET.keys()], device=self.device) #model index to index we use
        self.alphabet_inds = torch.arange(20, device=self.device) #Dummy

        self.prefixed_seq = prefixed_seq #List of tuples, (start, end, seq)
        
        self.gap_positions =  None
        self.gap_map = torch.arange(self.config.length, device=self.device) #From our protein position to MSA position with gaps
        self.gap_map_rev = self.gap_map.clone()
        if self.config.gap_positions is not None:
            self.gap_positions = torch.tensor(self.config.gap_positions, device=self.device).sort()[0] - 1 #to 0-based
            self.gap_map[self.gap_positions] = -1
            self.gap_map = self.gap_map[self.gap_map!=-1] 
            self.gap_map_rev[self.gap_positions] = -1
            self.gap_map_rev[self.gap_map_rev!=-1] = torch.arange(self.gap_map.shape[0], device=self.device)
            
        tmp = torch.zeros(self.config.length, device=self.device) - 1 #Position relative to target protein
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos-1] = Constants.ALPHABET.index(aa)
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and AA index at fixed positions    
        
        self.reset(self.decoding_order, self.rand_base)    
        
    def _reset_decoding_order(self, decoding_order):
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)
        
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None):
        self.rand_base = rand_base
        self._reset_decoding_order(decoding_order)
        
        self.next_t = 0 #Iteration step; used as index for decoding orders

        self.current_pred = None
        self.current_logits = None
        self.decoded_positions = torch.zeros(self.config.length, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.config.length, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.config.length, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.config.length, len(Constants.GREMLIN_ALPHABET)), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.config.length, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        if seed_S is not None: #The seed must include gaps if there are any
            self.S = seed_S.clone()
        else:
            self.S = torch.randint(0, len(Constants.GREMLIN_ALPHABET), (1, self.config.length), device=self.device).long() #0's for now
            if self.gap_positions is not None:
                for p in self.gap_positions:
                    self.decode_next(use_t_msa=p)
                    self.update_S(S_t=Constants.GAP_TOKEN_GREMLIN, use_t_msa=p, alphabet_map=False)
                    
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq) #This will update S, S_msa and decoded positions

    def get_cond_prob(self, t, use_S=None, temperature=0.1):
        if use_S is None:
            S_repeat = self.S.repeat((len(Constants.GREMLIN_ALPHABET), 1))
        else:
            S_repeat = use_S.repeat((len(Constants.GREMLIN_ALPHABET), 1))
        S_repeat[:, t] = torch.arange(len(Constants.GREMLIN_ALPHABET), device=self.device) 
        S_onehot = torch.nn.functional.one_hot(S_repeat, num_classes=len(Constants.GREMLIN_ALPHABET)) * 1.0
        return self.model.get_pll(S_onehot, temperature).unsqueeze(0)

    #Decode next step; returns AA logits
    def decode_next(self, dummy_run=False, mask_current=False, use_t_msa=None, use_t=None):
        if use_t is not None:
            t = use_t #Decoding position, relative to target protein
            t_msa = self.gap_map[t] #Decoding position, relative to the MSA of the target protein
        elif use_t_msa is None:
            t = self.decoding_order[0, self.next_t] #Decoding position, relative to target protein
            if t < self.gap_map.shape[0]:
                t_msa = self.gap_map[t] #Decoding position, relative to the MSA of the target protein
        else:
            t_msa = use_t_msa
            t = self.gap_map_rev[t_msa]

        if not (self.config.force_stop and (t == self.end_pos)):
            if dummy_run:
                self.current_logits = torch.zeros((1, len(Constants.GREMLIN_ALPHABET)), device=self.device)
            else:
                self.current_logits = self.get_cond_prob(t_msa, temperature=self.temp)
                self.current_logits -= self.current_logits.mean()
                
        if t > -1:
            if (use_t_msa is None) and self.config.force_stop and (t == self.end_pos): #Everything is zero except stop index
                logits = self._force_stop()
                return logits, logits
            
            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, Constants.STOP_INDEX] = Constants.MIN_LOGIT #Zero out the index for X
                logits = logits_.clone() 
            else:
                logits_ = self.current_logits.clone() #Only first row and standard AAs
                logits_ -= logits_.mean()
                logits_ = logits_[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, Constants.STOP_INDEX] = Constants.MIN_LOGIT #Zero out the index for X
    
                logits = logits_.clone()
                
                #Repeat penalty
                logits = self._apply_repetition_penalty(logits, t)
                
                #Final logits x some weight/temperature
                logits = self._apply_weights_and_biases(logits, t)
                
                #These suppress some AA's on hard thresholding of their counts
                aa_count = torch.nn.functional.one_hot(self.S[:,self.decoded_positions[0].bool()], num_classes=len(Constants.GREMLIN_ALPHABET)).sum(1)[:, self.alphabet_map]
                max_aa = (aa_count >= self.config.max_aa_count)
                logits[max_aa] = Constants.MIN_LOGIT
    
                #Positive AA total counts
                if (aa_count[0, 6] + aa_count[0, 8] + aa_count[0, 14]) >= self.config.max_pos_count: #This is for positively charged AA's; H/K/R
                    logits[0, 6] = Constants.MIN_LOGIT
                    logits[0, 8] = Constants.MIN_LOGIT
                    logits[0, 14] = Constants.MIN_LOGIT
    
                logits = BaseWrapper._top_p(logits, self.config.truncate_topp) #Top-p filtering

            if (use_t_msa is None) and ((not self.config.force_stop) or (t != self.end_pos)): #Penalize stop codon if not at last position
                logits_ = self._penalize_stop(logits_)
                logits = self._penalize_stop(logits)
                
            if (use_t_msa is None) and self.fixed_positions[t] != -1: #Everything is zero except fixed position
                logits = self._force_fixed_positions(logits, t)
                
            logits = BaseWrapper._add_noise(logits)
            return logits, logits_
    
    def edit_S(self, t, S_t, inplace=False): #t here is relative to MSA; S is model alphabet
        if inplace:
            S = self.S
        else:
            S = self.S.clone()

        if t < self.config.length:
            S[:, t] = S_t #Edit first row only

        if not inplace:
            return S
    
    #Update protein sequence vector S for next iteration
    def update_S(self, S_t, use_t_msa=None, alphabet_map=True, use_t=None, dummy_run=False): #t here is relative to protein (no gap); S_t is model alphabet
        if use_t is not None:
            t_msa = self.gap_map[use_t]
        elif use_t_msa is None:
            t = self.decoding_order[:, self.next_t]
            if self.config.force_stop and (t == self.end_pos):
                self.next_t += 1
                return False
            t_msa = self.gap_map[t] #Decoding position, relative to the MSA of the target protein
            self.next_t += 1 #Moves to next t
        else:
            t_msa = use_t_msa

        if alphabet_map:
            S_t = self.alphabet_map[S_t]
            
        self.edit_S(t_msa, S_t, inplace=True)
        self.decoded_positions[:, t_msa] = 1.0
        self.selected_aa[:, t_msa] = S_t
        log_softmax = torch.log(torch.nn.functional.softmax(self.current_logits[0], dim=-1))
        self.selected_log_prob[:, t_msa] = log_softmax[S_t]
        self.log_prob[t_msa, :] = log_softmax
        self.argmax_aa[:, t_msa] = self.current_logits[0].argmax()
        return True
    
    #Update protein sequence vector S for fixing some regions that will not be part of OLG decoding
    def preset_fixed_S(self, fixed_start, fixed_end, fixed_seq):
        t = torch.range(fixed_start, fixed_end, device=self.device)
        t_msa = self.gap_map[t] #Decoding position, relative to the MSA of the target protein
        fixed_token = self.alphabet_map[torch.tensor([ Constants.ALPHABET_GAP.index(c) for c in fixed_seq ], device=self.device)]
        self.edit_S(t_msa, fixed_token, inplace=True) #t here not relative to MSA
        self.decoded_positions[:, t_msa] = 1.0
    
    #Score the sequence by averaging log probabilities of each step
    def get_score(self, S=None, pll=False, positions=None):
        if S is None:
            S = self.S.clone()
        if pll:
            return -1.0 * self.model.get_pll(torch.nn.functional.one_hot(S, num_classes=len(Constants.GREMLIN_ALPHABET))*1.0, 1.0)
        else:
            self.reset(self.decoding_order, self.rand_base, S)
            self.decode_all(use_S=S[0]) 
            if positions is not None:
                return (self.selected_log_prob.mean() * -1.0)[0, positions]
            return self.selected_log_prob.mean() * -1.0
        
    def get_prot_seq(self, S=None):
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.config.start_offset:self.config.length]]
        prot = ''.join([Constants.ALPHABET_GAP[s] for s in S])
        return prot

    #Decodes all; this is used to design non-overlapping proteins with the same parameters
    def decode_all(self, temp=1e-12, use_S=None): #use_S is used to score a sequence. This should include gaps
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
                self.decode_next()
                t = self.decoding_order[:, i]
                if not (self.config.force_stop and (t == self.end_pos)):
                    S_t = use_S[self.gap_map[t]]
                else:
                    S_t = None
                self.update_S(S_t, alphabet_map=False)
        return True