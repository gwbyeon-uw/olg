from tqdm import tqdm

import numpy as np
import torch

from constants import *
from utils import *

from ProtMamba_ssm.modules import MambaLMHeadModelwithPosids, update_graph_cache, load_model
from ProtMamba_ssm.utils import load_from_file, prepare_tokens, prepare_target, generate_sequence, decode_sequence
from ProtMamba_ssm.utils import tokenizer as protmamba_tokenizer
from ProtMamba_ssm.utils import AA_TO_ID as protmamba_AA_TO_ID
from ProtMamba_ssm.dataloaders import Uniclust30_Dataset
from mamba_ssm.utils.generation import InferenceParams

def load_protmamba_model(device, weight_path="./ProtMamba-Long-foundation/"):
    model = load_model(weight_path, model_class=MambaLMHeadModelwithPosids, device=device, dtype=torch.bfloat16, checkpoint_mixer=False)
    model = model.eval()
    model.requires_grad_(False)
    return model

#Helper class to handle generate a protein sequence with ProtMamba
class ProtMambaContainer():
    def __init__(self, device, model, msa_seqs, msa_n_seq, msa_selection_type,
                 seq_len, seq_start, fixed_positions, prefixed_seq, 
                 decoding_order, end_stop, repetition_penalty, repetition_penalty_window,
                 logit_weight, logit_bias, aa_bias, 
                 max_aa_count, max_pos_count, 
                 truncate_topp, rand_base, tqdm_disable, shuffle_context):

        self.tqdm_disable = tqdm_disable
        self.device = device
        
        if model is None:
            self.model = load_protmamba_model(self.device)
        else:
            self.model = model

        self.msa_seqs = msa_seqs
        self.msa_n_seq = msa_n_seq
        self.msa_selection_type = msa_selection_type
        
        self.shuffle_context = shuffle_context

        self.alphabet_map = torch.tensor([ protmamba_AA_TO_ID[l] for l in ALPHABET ], device=self.device) #Index we use to ProtMamba index
        self.alphabet_map_rev = torch.arange(len(protmamba_AA_TO_ID), device=self.device)
        self.alphabet_map_rev.fill_(-1)
        for a, i in protmamba_AA_TO_ID.items():
            if a in ALPHABET:
                self.alphabet_map_rev[i] = ALPHABET.index(a)
        self.alphabet_inds = torch.arange(20, device=self.device) #Dummy
        
        self.seq_len = seq_len 
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
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)
        
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None):
        self.rand_base = rand_base
        self.reset_decoding_order(decoding_order)
        
        if self.rand_base is not None:
            np.random.seed(self.rand_base) #Random seed for subsample MSA

        _, _, self.msa = subsample_msa(self.msa_seqs, self.msa_n_seq, 1000000, self.msa_selection_type)
        
        self.tokenized_seqs = [ torch.tensor([0] + [ protmamba_AA_TO_ID[c] for c in seq if c != '-' ], device=self.device) for seq in self.msa ]
        self.S_orig = self.tokenized_seqs[0].unsqueeze(0)[:, 1:]
        self.context_seqs_orig = self.tokenized_seqs[1:]
        self.pos_ids_orig = [ torch.arange(t.shape[0], device=self.device) for t in self.context_seqs_orig ]
        
        self.next_t = 0 #Iteration step; used as index for decoding orders

        self.current_logits = None
        self.decoded_positions = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.seq_len, 40), device=self.device).bfloat16() #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        if seed_S is not None:
            self.S = seed_S.clone()
        else:
            self.S = self.S_orig.clone() #Seed with top row #torch.zeros((1, self.seq_len)).long().to(self.device)
            #self.S = self.S.fill_(protmamba_AA_TO_ID['<mask>'])
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq) #This will update S, S_msa and decoded positions

        self.gap_map = torch.arange(self.decoding_order.shape[1], device=self.device) #Dummy
        self.gap_map_rev = self.gap_map.clone()

    def decode_single_fim(self, target, mask_pos, shuffle_context=False):
        mask_pos_ = mask_pos + 1
        input_seq = target.clone()
        input_seq = torch.nn.functional.pad(input_seq, (1, 0))
        input_seq[0, mask_pos_] = 33 #<mask-1>
        input_seq = torch.hstack([ input_seq, torch.tensor([[ 2, 33 ]], device=self.device)]) #<eos> <mask-1>
        targ_pos = torch.arange(target.shape[1]+1, dtype=torch.int64, device=self.device)[None,:]
        targ_pos = torch.cat([targ_pos, torch.full((target.shape[0], 1), 0, dtype=torch.int64, device=self.device), torch.tensor([[mask_pos_]], device=self.device)], dim=1) #append 0 and mask position
        
        if shuffle_context:
            randidx = torch.randperm(len(self.context_seqs_orig))
            context_tokens = torch.cat([ self.context_seqs_orig[i] for i in randidx ]).unsqueeze(0)
            context_pos_id = torch.cat([ self.pos_ids_orig[i] for i in randidx ]).unsqueeze(0)
            context_tokens = torch.hstack([ context_tokens, input_seq ])
            context_pos_id = torch.hstack([ context_pos_id, targ_pos ])
        else:
            context_tokens = torch.cat(self.context_seqs_orig).unsqueeze(0)
            context_pos_id = torch.cat(self.pos_ids_orig).unsqueeze(0)
            context_tokens = torch.hstack([ context_tokens, input_seq ])
            context_pos_id = torch.hstack([ context_pos_id, targ_pos ])
        
        batch_size, seqlen_og = context_tokens.shape
        inference_params = InferenceParams(max_seqlen=10000000, max_batch_size=batch_size)
        logits = self.model(context_tokens, position_ids=context_pos_id, 
                       seq_position_ids=None, inference_params=inference_params, num_last_tokens=1).logits.squeeze(0)

        return logits
        
    #Decode next step; returns AA logits
    def decode_next(self, dummy_run=False, mask_current=False, use_t=None):
        if use_t is not None:
            t = use_t
        else:
            t = self.decoding_order[0, self.next_t] #Decoding position, relative to target protein
        
        if dummy_run:
            self.current_logits = torch.zeros((1, 40), device=self.device)
        else:
            self.current_logits = self.decode_single_fim(self.S, t, self.shuffle_context)  #Output shape is (BS=1, n_tokens=40)

        if (use_t_msa is None) and self.end_stop and (t == self.end_pos): #Everything is zero except stop index
            logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
            logits[0, STOP_INDEX] = MAX_LOGIT #High number to force stop
            logits = add_noise(logits)
            return logits, logits
        
        if dummy_run:
            logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
            logits_[:, STOP_INDEX] = MIN_LOGIT #Zero out the index for X
            logits = logits_.clone() 
        else:
            logits_ = self.current_logits.clone() #Only first row and standard AAs
            logits_ -= logits_.mean()
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
            aa_count = torch.nn.functional.one_hot(self.S[:,self.decoded_positions[0].bool()], num_classes=40).sum(1)[:, self.alphabet_map]
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
    
    def edit_S(self, t, S_t, inplace=False): #t here is relative to MSA; S is ProtMamba alphabet
        if inplace:
            S = self.S
        else:
            S = self.S.clone()

        if t < self.seq_len:
            S[:, t] = S_t #Edit first row only
            
        if not inplace:
            return S
    
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
        self.decode_all(use_S=S[0])
        if positions is not None:
            return (self.selected_log_prob * -1.0)[0, positions].mean()
        return (self.selected_log_prob.mean() * -1.0)
        
    def get_prot_seq(self, S=None):
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.seq_start:self.seq_len]]
        prot = ''.join([ALPHABET[s] for s in S])
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
                if not (self.end_stop and (t == self.end_pos)):
                    S_t = use_S[t]
                else:
                    S_t = None
                self.update_S(S_t, alphabet_map=False)
        return True