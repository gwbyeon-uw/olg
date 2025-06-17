from tqdm import tqdm

import numpy as np
import torch

from constants import *
from utils import *

#For EvoDiff
from scipy.spatial.distance import hamming, cdist
from sequence_models.constants import PROTEIN_ALPHABET, GAP
from evodiff.utils import Tokenizer
from evodiff.pretrained import MSA_OA_DM_MAXSUB, ESM_MSA_1b #https://github.com/microsoft/evodiff/tree/main#loading-pretrained-models

def load_evodiff_model(device):
    checkpoint = MSA_OA_DM_MAXSUB()
    model, _, tokenizer, _ = checkpoint
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer

def load_esmmsa_model(device):
    checkpoint = ESM_MSA_1b()
    model, _, alphabet, _ = checkpoint
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    evodiff_tokenizer = Tokenizer()
    return model, alphabet, evodiff_tokenizer

#Helper class to handle generate a protein sequence with EvoDiff
class EvoDiffContainer():
    def __init__(self, device, model, tokenizer, 
                 msa_seqs, msa_n_seq, msa_max_length, msa_selection_type, 
                 seq_len, seq_start, fixed_positions, gap_positions, prefixed_seq, 
                 decoding_order, end_stop, repetition_penalty, repetition_penalty_window,
                 logit_weight, logit_bias, aa_bias, 
                 max_aa_count, max_pos_count, 
                 truncate_topp, rand_base, tqdm_disable, use_esm_msa):

        self.tqdm_disable = tqdm_disable
        self.device = device
        self.use_esm_msa = use_esm_msa #Flag to use MSA Transformer instead of EvoDiff-MSA
        if (model is None) or (tokenizer is None):
            if not self.use_esm_msa:
                self.model, self.tokenizer = load_evodiff_model(self.device)
            else:
                self.model, _, self.tokenizer = load_esmmsa_model(self.device)
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        #MSA subsampling        
        self.msa_seqs = msa_seqs
        self.msa_n_seq = msa_n_seq # number of sequences in MSA to subsample
        self.msa_max_length = msa_max_length # maximum sequence length to subsample
        self.seq_len = self.msa_max_length #Set to same as MSA length for now
        self.msa_selection_type = msa_selection_type # or 'MaxHamming'; MSA subsampling scheme

        self.valid_msa_, self.query_sequence, _ = subsample_msa(self.msa_seqs, 
                                                               n_sequences=self.msa_n_seq,
                                                               max_seq_len=self.msa_max_length,
                                                               selection_type=self.msa_selection_type)
        self.valid_msa = torch.tensor(np.array([self.tokenizer.tokenizeMSA(seq) for seq in self.valid_msa_]), device=self.device) #Tokenize sequence
        self.padding = torch.full((self.msa_n_seq, self.msa_max_length-self.seq_len), fill_value=self.tokenizer.pad_id, device=self.device)
        
        self.alphabet_map = torch.tensor([ self.tokenizer.a_to_i[l] for l in ALPHABET_GAP ], device=self.device) #Index we use to EvoDiff index
        self.alphabet_map_rev = torch.tensor([ ALPHABET_GAP.index(a) if a in ALPHABET_GAP else -1 for a in self.tokenizer.alphabet  ], device=self.device) #EvoDiff index to index we use
        #self.alphabet_inds = self.alphabet_map_rev[self.alphabet_map_rev!=-1]
        self.alphabet_inds = torch.arange(20, device=self.device) #Dummy

        self.remap_to_evodiff = REMAP_TO_EVODIFF.to(self.device)
        self.remap_to_esmmsa = REMAP_TO_ESM_MSA.to(self.device)
        
        self.seq_start = seq_start #Offset to start from
        self.prefixed_seq = prefixed_seq #List of tuples, (start, end, seq)
        self.end_stop = end_stop

        self.gap_positions =  None
        self.gap_map = torch.arange(self.seq_len, device=self.device) #From our protein position to MSA position with gaps
        self.gap_map_rev = self.gap_map.clone()
        if gap_positions is not None:
            self.gap_positions = torch.tensor(gap_positions, device=self.device).sort()[0] - 1 #to 0-based
            self.gap_map[self.gap_positions] = -1
            self.gap_map = self.gap_map[self.gap_map!=-1] 
            self.gap_map_rev[self.gap_positions] = -1
            self.gap_map_rev[self.gap_map_rev!=-1] = torch.arange(self.gap_map.shape[0], device=self.device)
            
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
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)
        
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None):
        self.reset_decoding_order(decoding_order)
        self.rand_base = rand_base
        
        if self.rand_base is not None:
            np.random.seed(self.rand_base) #Random seed
        
        self.S_orig = self.valid_msa[0, :self.seq_len]
        self.S_msa = torch.full((1, self.msa_n_seq, self.msa_max_length), fill_value=self.tokenizer.mask_id, device=self.device)
        self.S_msa[:, 1:self.msa_n_seq, :self.seq_len] = self.valid_msa[1:self.msa_n_seq, :self.seq_len] #Everything except the first row; first row is the design target and therefore masked (fully, so if part of it needs to be filled prior to OLG decoding, use prefixed_seq)
        self.S_msa[:, :, self.seq_len:] = self.padding #Change mask token to pad token
        #self.S_msa = self.S_msa.to(self.device) #Will write generated sequences here and used to input to model
        
        self.next_t = 0 #Iteration step; used as index for decoding orders

        self.current_pred = None
        self.current_logits = None
        self.decoded_positions = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.seq_len, len(self.tokenizer.alphabet)), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        if seed_S is not None: #The seed must include gaps if there are any
            self.S_msa[:, 0, :] = seed_S.clone()
            self.S = self.S_msa[:, 0, :] #Just the top row
        else:
            self.S = self.S_msa[:, 0, :] #Just the top row
            if self.gap_positions is not None:
                for p in self.gap_positions:
                    self.decode_next(use_t_msa=p)
                    self.update_S(S_t=GAP_TOKEN, use_t_msa=p, alphabet_map=False)
                    
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq) #This will update S, S_msa and decoded positions
        
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

        if not (self.end_stop and (t == self.end_pos)):
            if dummy_run:
                self.current_pred = torch.zeros((self.S_msa.shape[2], len(self.tokenizer.alphabet)), device=self.device)
            else:
                if mask_current:
                    self.S_msa[:, 0, t_msa] = self.tokenizer.mask_id
    
                if self.use_esm_msa:
                    remapped_input = torch.nn.functional.pad(torch.gather(self.remap_to_esmmsa.expand(self.S_msa.shape[1], -1), 1, self.S_msa[0]), (1, 0)).unsqueeze(0)
                    output_ = self.model(remapped_input)  #Output shape of preds is (BS=1, N=64, L, n_tokens=31)
                    output = torch.zeros((1, self.S_msa.shape[1], self.S_msa.shape[2], len(self.tokenizer.a_to_i)), device=self.device).fill_(MIN_LOGIT)
                    output[:, :, :, self.remap_to_evodiff[self.remap_to_evodiff!=-1]] = output_['logits'][:, :, 1:, self.remap_to_evodiff!=-1]
                    self.current_pred = output[0, 0, :, :]
                else:
                    self.current_pred = self.model(self.S_msa)[0, 0, :, :]  #Output shape of preds is (BS=1, N=64, L, n_tokens=31)

        if t > -1:
            if (use_t_msa is None) and self.end_stop and (t == self.end_pos): #Everything is zero except stop index
                logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
                logits[0, STOP_INDEX] = MAX_LOGIT #High number to force stop
                logits = add_noise(logits)
                return logits, logits
                
            self.current_logits = self.current_pred[t_msa, :].unsqueeze(0) #Logits at current position, unless it's a stop and > length of protein
    
            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, STOP_INDEX] = MIN_LOGIT #Zero out the index for X
                logits = logits_.clone() 
            else:
                logits_ = self.current_logits.clone() #Only first row and standard AAs
                logits_ -= logits_.mean() #Center it
                logits_ = logits_[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, STOP_INDEX] = MIN_LOGIT #Zero out the index for X

                logits = logits_.clone()
                
                #Repeat penalty
                t_left = max(0, t-self.repetition_penalty_window)
                t_right = min(self.decoded_positions.shape[1], t+self.repetition_penalty_window)
                if (t_right + 1 - t_left) > 0:
                    decoded_pos = self.decoded_positions[0, t_left:(t_right+1)].bool()
                    if decoded_pos.sum() > 0:
                        neighbor_aa = self.alphabet_map_rev[self.S[0, t_left:(t_right+1)][decoded_pos]]
                        uniq_ct = torch.unique(neighbor_aa, return_counts=True)
                        if neighbor_aa.shape[0] > 0:
                            logits_p = logits[0, uniq_ct[0]]
                            rep_pen = self.repetition_penalty**uniq_ct[1]
                            logits_p = torch.where(logits_p < 0, logits_p * rep_pen, logits_p / rep_pen)
                            logits.scatter_(1, uniq_ct[0].unsqueeze(0), logits_p.unsqueeze(0))
                
                #Final logits x some weight/temperature
                logits = self.logit_weight[t] * (logits + self.aa_bias.unsqueeze(0) + self.logit_bias[t:(t+1), :])
                
                #These suppress some AA's on hard thresholding of their counts
                aa_count = torch.nn.functional.one_hot(self.S[:,self.decoded_positions[0].bool()], num_classes=len(self.tokenizer.alphabet)).sum(1)[:, self.
                alphabet_map]
                max_aa = (aa_count >= self.max_aa_count)
                logits[max_aa] = MIN_LOGIT
    
                #Positive AA total counts
                if (aa_count[0, 6] + aa_count[0, 8] + aa_count[0, 14]) >= self.max_pos_count: #This is for positively charged AA's; H/K/R
                    logits[0, 6] = MIN_LOGIT
                    logits[0, 8] = MIN_LOGIT
                    logits[0, 14] = MIN_LOGIT
    
                logits = top_p(logits, self.truncate_topp) #Top-p filtering

            if (use_t_msa is None) and ((not self.end_stop) or (t != self.end_pos)): #Penalize stop codon if not at last position
                logits_[0, STOP_INDEX] = MIN_LOGIT
                logits[0, STOP_INDEX] = MIN_LOGIT
            
            if (use_t_msa is None) and self.fixed_positions[t] != -1: #Everything is zero except fixed position
                logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
                logits[0, self.fixed_positions[t]] = MAX_LOGIT #High number to force fixed residue
    
            logits = add_noise(logits)
            return logits, logits_
    
    def edit_S(self, t, S_t, inplace=False): #t here is relative to MSA; S is EvoDiff alphabet
        if inplace:
            S = self.S
            S_msa = self.S_msa
        else:
            S = self.S.clone()
            S_msa = self.S_msa.clone()

        if t < self.seq_len:
            S_msa[:, 0, t] = S_t #Edit first row only
            S = S_msa[:, 0, 0:self.seq_len] #First row slice of MSA

        if not inplace:
            return S, S_msa
    
    #Update protein sequence vector S for next iteration
    def update_S(self, S_t, use_t_msa=None, alphabet_map=True, use_t=None, dummy_run=False): #t here is relative to protein (no gap); S_t is EvoDiff alphabet
        if use_t_msa is None:
            t = self.decoding_order[:, self.next_t]
            if self.end_stop and (t == self.end_pos):
                self.next_t += 1
                return False
            t_msa = self.gap_map[t] #Decoding position, relative to the MSA of the target protein
            self.next_t += 1 #Moves to next t
        elif use_t is not None:
            t_msa = self.gap_map[use_t]
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
        fixed_token = self.alphabet_map[torch.tensor([ ALPHABET_GAP.index(c) for c in fixed_seq ], device=self.device)]
        self.edit_S(t_msa, fixed_token, inplace=True) #t here not relative to MSA
        self.decoded_positions[:, t_msa] = 1.0
    
    #Score the sequence by averaging log probabilities of each step
    def get_score(self, S=None, positions=None):
        if S is None: #To rescore with current sequence, use self.S.clone() as input
            S = self.S.clone()
        self.reset(self.decoding_order, self.rand_base, S)
        self.decode_all(use_S=S[0], mask_current=True) #Seed with given sequence, then mask/predict each token conditional on rest
        if positions is not None:
            return (self.selected_log_prob * -1.0)[0, positions].mean()
        return (self.selected_log_prob.mean() * -1.0)
            
    def get_prot_seq(self, S=None):
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.seq_start:self.seq_len]]
        prot = ''.join([ALPHABET_GAP[s] for s in S])
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