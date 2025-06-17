from tqdm import tqdm

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from constants import *
from utils import *

from esm.pretrained import ESM3_structure_decoder_v0
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization import StructureTokenizer, EsmSequenceTokenizer
import esm.utils.constants.esm3 as C

def load_coflow_model(device, weight_path="./coflow_ckpt/"):
    model = CoFlowModel.from_pretrained(weight_path)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    return model

#Helper class to handle generate a protein sequence with CoFlow
class CoFlowContainer():
    def __init__(self, device, model, seq_len, seq_start, fixed_positions, prefixed_seq, 
                 decoding_order, end_stop, repetition_penalty, repetition_penalty_window,
                 logit_weight, logit_bias, aa_bias, 
                 max_aa_count, max_pos_count, 
                 truncate_topp, rand_base, tqdm_disable):

        self.tqdm_disable = tqdm_disable
        self.device = device
        
        if model is None:
            self.model = load_coflow_model(self.device)
        else:
            self.model = model

        self.tokenizer = EsmSequenceTokenizer().vocab
        self.vocab_size = len(self.tokenizer)
        self.alphabet_map = torch.tensor([ self.tokenizer[l] for l in ALPHABET ], device=self.device) #Index we use to ESM3 index
        self.alphabet_map_rev = torch.arange(self.vocab_size, device=self.device)
        self.alphabet_map_rev.fill_(-1)
        for a, i in self.tokenizer.items():
            if a in ALPHABET:
                self.alphabet_map_rev[i] = ALPHABET.index(a)
        self.alphabet_inds = torch.arange(20, device=self.device) #Dummy
        
        self.seq_len = seq_len 
        self.L = self.seq_len + 2
        self.seq_start = seq_start #Offset to start from
        self.prefixed_seq = prefixed_seq #List of tuples, (start, end, seq)
        self.end_stop = end_stop

        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_window = repetition_penalty_window
        
        if logit_weight is None:
            self.logit_weight = torch.ones(self.seq_len, device=self.device)
        else:
            self.logit_weight = logit_weight
        
        if logit_bias is None:
            self.logit_bias = torch.zeros((self.seq_len, len(ALPHABET)), device=self.device)
        else:
            self.logit_bias = logit_bias
            
        #Biases; these get added to logits
        if aa_bias is None:
            self.aa_bias = torch.zeros(len(ALPHABET), device=self.device)
        else:
            self.aa_bias = aa_bias
             
        if max_aa_count is None:
            self.max_aa_count = torch.zeros(len(ALPHABET), device=self.device) + MAX_LOGIT
        else:
            self.max_aa_count = max_aa_count
            
        if max_pos_count is None:
            self.max_pos_count = MAX_LOGIT
        else:
            self.max_pos_count = max_pos_count
            
        if truncate_topp is None:
            self.truncate_topp = 0.0
        else:
            self.truncate_topp = truncate_topp
            
        tmp = torch.zeros(self.seq_len, device=self.device) - 1 #Position relative to target protein
        if fixed_positions is not None:
            for pos, aa in fixed_positions:
                tmp[pos-1] = ALPHABET.index(aa)
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and AA index at fixed positions    
        
        self.reset(decoding_order, rand_base)    

    def reset_decoding_order(self, decoding_order):
        self.decoding_order = decoding_order #This is relative to target chain's position in the OLG decoder. It is NOT the positions for X/S from PDB and need to be offseted
        self.end_pos = torch.max(self.decoding_order)
    
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None, seed_structure=None, tcm=None):
        self.reset_decoding_order(decoding_order)
        self.rand_base = rand_base
        
        if self.rand_base is not None:
            np.random.seed(self.rand_base) #Random seed for subsample MSA
        
        self.next_t = 0 #Iteration step; used as index for decoding orders

        self.current_logits = None
        self.decoded_positions = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.seq_len, self.vocab_size), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        if seed_S is not None:
            self.S = seed_S.clone()
        else:
            self.S = torch.zeros((1, self.seq_len)).long().to(self.device) #Excluding <cls> and <eos>
            self.S = self.S.fill_(self.tokenizer['<mask>'])
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq) #This will update S, S_msa and decoded positions

        if seed_structure is None: #Other than sequence_tokens
            self.structure_tokens = torch.ones((1, self.seq_len)).long().to(self.device)
            self.structure_tokens = self.structure_tokens.fill_(C.STRUCTURE_MASK_TOKEN)
        else:
            self.structure_tokens = seed_structure
        
        self.gap_map = torch.arange(self.decoding_order.shape[1], device=self.device) #Dummy
        self.gap_map_rev = self.gap_map.clone()
        self.tc = torch.Tensor([[0.0]]).to(self.device)
        self.dt = 1.0 / self.seq_len
        self.tcm = tcm
        self.last_struct_logits = None

    def get_logits(self):
        if self.tcm is None:
            tc = self.tc
        else:
            tc = self.tcm
        struct_logits, seq_logits = self.model.denoise(structure=self.structure_tokens, sequence=self.S, t=tc)
        return seq_logits[:, :, 0:self.vocab_size]
        
    #Decode next step; returns AA logits
    def decode_next(self, dummy_run=False, mask_current=False, use_t=None):
        if use_t is not None:
            t = use_t
        else:
            t = self.decoding_order[0, self.next_t] #Decoding position, relative to target protein
        
        if dummy_run:
            self.current_pred = torch.zeros((self.seq_len, self.vocab_size), device=self.device)
        else:
            if mask_current:
                self.S[0, t] = self.tokenizer['<mask>']
            self.current_pred = self.get_logits()[0]

        if t > -1:
            if self.end_stop and (t == self.end_pos): #Everything is zero except stop index
                logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
                logits[0, STOP_INDEX] = MAX_LOGIT #High number to force stop
                logits = add_noise(logits)
                return logits, logits
    
            self.current_logits = self.current_pred[t, :].unsqueeze(0) #Logits at current position, unless it's a stop and > length of protein
            
            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, STOP_INDEX] = MIN_LOGIT #Zero out the index for X
                logits = logits_.clone() 
            else:
                logits_ = self.current_logits.clone() #Only first row and standard AAs
                logits_ = logits_[:, self.alphabet_map] #Only the alphabet we use
                logits_ -= logits_.mean()
                logits_[:, STOP_INDEX] = MIN_LOGIT #Zero out the index for X
    
                logits = logits_.clone()

                #Repeat penalty
                t_left = max(0, t-self.repetition_penalty_window)
                t_right = min(self.decoded_positions.shape[1], t+self.repetition_penalty_window)
                if (t_right + 1 - t_left) > 0:
                    decoded_pos = self.decoded_positions[0, t_left:(t_right+1)].bool()
                    if decoded_pos.sum() > 0:
                        neighbor_aa = self.alphabet_map_rev[self.S[0, t_left:(t_right+1)][decoded_pos]]
                        neighbor_aa = neighbor_aa[neighbor_aa!=-1]
                        if neighbor_aa.shape[0] > 0:
                            uniq_ct = torch.unique(neighbor_aa, return_counts=True)
                            logits_p = logits[0, uniq_ct[0]]
                            rep_pen = self.repetition_penalty**uniq_ct[1]
                            logits_p = torch.where(logits_p < 0, logits_p * rep_pen, logits_p / rep_pen).to(logits.dtype)
                            logits.scatter_(1, uniq_ct[0].unsqueeze(0), logits_p.unsqueeze(0))
                
                #Final logits x some weight/temperature
                logits = self.logit_weight[t] * (logits + self.aa_bias.unsqueeze(0) + self.logit_bias[t:(t+1), :])
                            
                #These suppress some AA's on hard thresholding of their counts
                aa_count = torch.nn.functional.one_hot(self.S[:,self.decoded_positions[0].bool()], num_classes=self.vocab_size).sum(1)[:, self.alphabet_map]
                max_aa = (aa_count >= self.max_aa_count)
                logits[max_aa] = MIN_LOGIT
    
                #Positive AA total counts
                if (aa_count[0, 6] + aa_count[0, 8] + aa_count[0, 14]) >= self.max_pos_count: #This is for positively charged AA's; H/K/R
                    logits[0, 6] = MIN_LOGIT
                    logits[0, 8] = MIN_LOGIT
                    logits[0, 14] = MIN_LOGIT
                            
                logits = top_p(logits, self.truncate_topp) #Top-p filtering
    
            if ((not self.end_stop) or (t != self.end_pos)): #Penalize stop codon if not at last position
                logits_[0, STOP_INDEX] = MIN_LOGIT
                logits[0, STOP_INDEX] = MIN_LOGIT
                
            if self.fixed_positions[t] != -1: #Everything is zero except fixed position
                logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
                logits[0, self.fixed_positions[t]] = MAX_LOGIT #High number to force fixed residue
                
            logits = add_noise(logits)
            return logits, logits_
        
    def edit_S(self, t, S_t, inplace=False): #t here is relative to MSA; S is ProtMamba alphabet
        if inplace:
            S = self.S
        else:
            S = self.S.clone()

        if t < self.seq_len:
            S[:, t] = S_t #Edit first row only
            
        if not inplace:
            return S

    def sample_struct_single(self, t, temp=0.7, dummy_run=False):
        if not dummy_run:
            if t < self.seq_len:
                if self.tcm is None:
                    tc = self.tc
                else:
                    tc = self.tcm
                self.structure_tokens[0, t] = C.STRUCTURE_MASK_TOKEN
                struct_logits, seq_logits = self.model.denoise(structure=self.structure_tokens, sequence=self.S, t=tc)
                struct_probs = torch.softmax(struct_logits/temp, dim=-1)    
                self.structure_tokens[0, t] = Categorical(struct_probs[0, t, :]).sample()
            
    #Update protein sequence vector S for next iteration
    def update_S(self, S_t, use_t=None, alphabet_map=True, dummy_run=False): #t here is relative to protein (no gap); S_t is ProtMamba alphabet
        if use_t is None:
            t = self.decoding_order[:, self.next_t]
            if self.end_stop and (t == self.end_pos):
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

        self.tc += self.dt
        return True
    
    #Update protein sequence vector S for fixing some regions that will not be part of OLG decoding
    def preset_fixed_S(self, fixed_start, fixed_end, fixed_seq):
        t = torch.range(fixed_start, fixed_end, device=self.device)
        fixed_token = self.alphabet_map[torch.tensor([ ALPHABET.index(c) for c in fixed_seq ], device=self.device)]
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
            S = self.alphabet_map_rev[self.S[0, self.seq_start:self.seq_len]]
        prot = ''.join([ALPHABET[s] for s in S])
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
                if not (self.end_stop and (t == self.end_pos)):
                    S_t = use_S[self.gap_map[t]]
                else:
                    S_t = None
                self.update_S(S_t, alphabet_map=False)
        return True