from tqdm import tqdm

import numpy as np
import torch

from scipy.spatial.distance import pdist,squareform

from constants import *
from utils import *

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
            
def mk_msa(seqs, device, gap_cutoff=1.0, eff_cutoff=0.8):
    '''converts list of sequences to msa'''
    
    alphabet = "ARNDCQEGHILKMFPSTWYV-"
    states = len(alphabet)

    k = torch.tensor(list(alphabet.encode())).to(device)
    v = torch.tensor(list(range(21))).to(device)
    aa2num = torch.zeros(k.max()+1,dtype=v.dtype).to(device) #k,v from approach #1
    aa2num[k] = v
        
    def one_hot(msa, states=21):
        one = torch.eye(states).to(device)
        return one[msa]

    def filt_gaps(msa, gap_cutoff=0.5, states=21):
        '''filters alignment to remove gappy positions'''
        tmp = (msa == states-1) * 1.0
        non_gaps = torch.where(torch.sum(tmp.T,-1).T/msa.shape[0] < gap_cutoff)[0]
        del tmp
        return msa[:,non_gaps],non_gaps

    def get_eff(msa, eff_cutoff=0.8):
        '''compute effective weight for each sequence'''
        ncol = msa.shape[1]
        
        # pairwise identity
        pdist = torch.nn.functional.pdist(msa*1.0,p=0)
        msa_sm = 1.0 - torch.tensor(squareform(pdist.cpu().numpy())).to(device)/ncol
        del pdist
    
        # weight for each sequence
        msa_w = (msa_sm >= eff_cutoff) * 1.0
        del msa_sm
        msa_w = 1/torch.sum(msa_w,-1)
        
        return msa_w

    seq_arr = torch.tensor([ list(p.encode()) for p in seqs ], dtype=torch.long).to(device)
    msa_ori = torch.take(aa2num, seq_arr)
        
    msa, v_idx = filt_gaps(msa_ori, gap_cutoff, states)
    msa_weights = get_eff(msa, eff_cutoff)

    # compute effective number of sequences
    ncol = msa.shape[1] # length of sequence
    w_idx = v_idx[np.stack(np.triu_indices(ncol,1),-1)]

    msa_onehot = one_hot(msa, states)
    
    return {"msa_ori":msa_ori,
            "msa":msa,
            "msa_onehot":msa_onehot,
            "weights":msa_weights,
            "neff":torch.sum(msa_weights),
            "v_idx":v_idx,
            "w_idx":w_idx,
            "nrow":msa.shape[0],
            "ncol":ncol,
            "ncol_ori":msa_ori.shape[1],
            "states":states}



#Helper class to handle generate a protein sequence with GREMLIN
class GREMLINContainer():
    def __init__(self, device, model, temperature,
                 seq_len, seq_start, fixed_positions, gap_positions, prefixed_seq, 
                 decoding_order, end_stop, repetition_penalty, repetition_penalty_window,
                 logit_weight, logit_bias, aa_bias, 
                 max_aa_count, max_pos_count, 
                 truncate_topp, rand_base, tqdm_disable):

        self.tqdm_disable = tqdm_disable
        self.device = device
        
        if model is None:
            self.model = torch.load(model).to(self.device)
        else:
            self.model = model
        self.model = self.model.eval()
        self.model.requires_grad_(False)

        self.W = self.model.W
        self.V = self.model.V
        self.temp = temperature
        
        #MSA subsampling        
        self.alphabet_map = torch.tensor([ GREMLIN_ALPHABET[l] for l in ALPHABET_GAP ], device=self.device) #Index we use to model index
        self.alphabet_map_rev = torch.tensor([ ALPHABET_GAP.index(a) if a in ALPHABET_GAP else -1 for a in GREMLIN_ALPHABET.keys()], device=self.device) #model index to index we use
        self.alphabet_inds = torch.arange(20, device=self.device) #Dummy

        self.seq_len = seq_len #Set to same as MSA length for now
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
        
        self.next_t = 0 #Iteration step; used as index for decoding orders

        self.current_pred = None
        self.current_logits = None
        self.decoded_positions = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.seq_len, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.seq_len, len(GREMLIN_ALPHABET)), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.seq_len, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        if seed_S is not None: #The seed must include gaps if there are any
            self.S = seed_S.clone()
        else:
            self.S = torch.randint(0, len(GREMLIN_ALPHABET), (1, self.seq_len), device=self.device).long() #0's for now
            if self.gap_positions is not None:
                for p in self.gap_positions:
                    self.decode_next(use_t_msa=p)
                    self.update_S(S_t=GAP_TOKEN_GREMLIN, use_t_msa=p, alphabet_map=False)
                    
            if self.prefixed_seq is not None:
                for fixed_start, fixed_end, fixed_seq in self.prefixed_seq:
                    self.preset_fixed_S(fixed_start, fixed_end, fixed_seq) #This will update S, S_msa and decoded positions

    def get_cond_prob(self, t, use_S=None, temperature=0.1):
        if use_S is None:
            S_repeat = self.S.repeat((len(GREMLIN_ALPHABET), 1))
        else:
            S_repeat = use_S.repeat((len(GREMLIN_ALPHABET), 1))
        S_repeat[:, t] = torch.arange(len(GREMLIN_ALPHABET), device=self.device) 
        S_onehot = torch.nn.functional.one_hot(S_repeat, num_classes=len(GREMLIN_ALPHABET)) * 1.0
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

        if not (self.end_stop and (t == self.end_pos)):
            if dummy_run:
                self.current_logits = torch.zeros((1, len(GREMLIN_ALPHABET)), device=self.device)
            else:
                self.current_logits = self.get_cond_prob(t_msa, temperature=self.temp)
                self.current_logits -= self.current_logits.mean()
                
        if t > -1:
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
                aa_count = torch.nn.functional.one_hot(self.S[:,self.decoded_positions[0].bool()], num_classes=len(GREMLIN_ALPHABET)).sum(1)[:, self.alphabet_map]
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
    
    def edit_S(self, t, S_t, inplace=False): #t here is relative to MSA; S is model alphabet
        if inplace:
            S = self.S
        else:
            S = self.S.clone()

        if t < self.seq_len:
            S[:, t] = S_t #Edit first row only

        if not inplace:
            return S
    
    #Update protein sequence vector S for next iteration
    def update_S(self, S_t, use_t_msa=None, alphabet_map=True, use_t=None, dummy_run=False): #t here is relative to protein (no gap); S_t is model alphabet
        if use_t is not None:
            t_msa = self.gap_map[use_t]
        elif use_t_msa is None:
            t = self.decoding_order[:, self.next_t]
            if self.end_stop and (t == self.end_pos):
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
        fixed_token = self.alphabet_map[torch.tensor([ ALPHABET_GAP.index(c) for c in fixed_seq ], device=self.device)]
        self.edit_S(t_msa, fixed_token, inplace=True) #t here not relative to MSA
        self.decoded_positions[:, t_msa] = 1.0
    
    #Score the sequence by averaging log probabilities of each step
    def get_score(self, S=None, pll=False, positions=None):
        if S is None:
            S = self.S.clone()
        if pll:
            return -1.0 * self.model.get_pll(torch.nn.functional.one_hot(S, num_classes=len(GREMLIN_ALPHABET))*1.0, 1.0)
        else:
            self.reset(self.decoding_order, self.rand_base, S)
            self.decode_all(use_S=S[0]) 
            if positions is not None:
                return (self.selected_log_prob.mean() * -1.0)[0, positions]
            return self.selected_log_prob.mean() * -1.0
        
    def get_prot_seq(self, S=None):
        if S is None:
            S = self.alphabet_map_rev[self.S[0, self.seq_start:self.seq_len]]
        prot = ''.join([ALPHABET_GAP[s] for s in S])
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
                    S_t = use_S[self.gap_map[t]]
                else:
                    S_t = None
                self.update_S(S_t, alphabet_map=False)
        return True