from typing import Tuple, Optional
import string

import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform

from olg.constants import *
from olg.config import ProteinConfig

class BaseWrapper:
    """
    Base class for protein sequence decoders with some functions shared across all
    """
    
    def __init__(
        self,
        device: torch.device,
        config: ProteinConfig,
        decoding_order: torch.Tensor,
        rand_base: float,
        tqdm_disable: bool
    ):
        self.device = device
        self.config = config
        self.decoding_order = decoding_order
        self.rand_base = rand_base
        self.tqdm_disable = tqdm_disable

        self.logit_weight = self.config.logit_weight.clone()
        
        self.alphabet_inds = torch.arange(20, device=self.device) # Dummy
        
    #Top p thresholding given logit vector
    @staticmethod
    def _top_p(
        logits: torch.Tensor, 
        thres: float = 0.1, 
        removal_value: float = -1e3
    ) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > (1 - thres)
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
    
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits_ = logits.clone()
        logits_[indices_to_remove] = removal_value
        
        return logits_

    #Adds small noise to tensor; used for tiebreaking stochastically
    @staticmethod
    def _add_noise(
        tensor: torch.Tensor, 
        factor: float = 1e-6
    ) -> torch.Tensor:
        noised_tensor = tensor + torch.rand(tensor.shape, device=tensor.device) * factor
        return noised_tensor

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor, 
        t: int, # Decoding position, relative to target protein
    ) -> torch.Tensor:
        """Apply repetition penalty based on nearby decoded positions."""
        logits_new = logits.clone()
        t_left = max(0, t-self.config.repetition_penalty_window)
        t_right = min(self.decoded_positions.shape[1], t+self.config.repetition_penalty_window)
        if (t_right + 1 - t_left) > 0:
            decoded_pos = self.decoded_positions[0, t_left:(t_right+1)].bool()
            if decoded_pos.sum() > 0:
                neighbor_aa = self.alphabet_map_rev[self.S[0, t_left:(t_right+1)][decoded_pos]]
                uniq_ct = torch.unique(neighbor_aa, return_counts=True)
                if neighbor_aa.shape[0] > 0:
                    logits_p = logits[0, uniq_ct[0]]
                    rep_pen = self.config.repetition_penalty**uniq_ct[1]
                    logits_p = torch.where(logits_p < 0, logits_p * rep_pen, logits_p / rep_pen).to(logits.dtype)
                    logits_new.scatter_(1, uniq_ct[0].unsqueeze(0), logits_p.unsqueeze(0))
                    
        return logits_new

    def _apply_weights_and_biases(
        self, 
        logits: torch.Tensor, 
        t: int
    ) -> torch.Tensor:
        """Apply position-specific weights and biases."""
        return self.logit_weight[t] * (logits + self.config.aa_bias.unsqueeze(0) + self.config.logit_bias[t:(t + 1), :])

    def _force_stop(self) -> torch.Tensor:
        logits = torch.zeros(Constants.ALPHABET_SIZE, device=self.device).unsqueeze(0)
        logits[0, Constants.STOP_INDEX] = Constants.MAX_LOGIT #High number to force stop
        logits = BaseWrapper._add_noise(logits)
        return logits

    def _penalize_stop(self, logits: torch.Tensor) -> torch.Tensor:
        logits_new = logits.clone()
        logits_new[0, Constants.STOP_INDEX] = Constants.MIN_LOGIT
        return logits_new

    def _force_fixed_positions(self, logits: torch.Tensor, t: int) -> torch.Tensor:
        logits = torch.zeros(Constants.ALPHABET_SIZE, device=self.device).unsqueeze(0)
        logits[0, self.fixed_positions[t]] = Constants.MAX_LOGIT #High number to force fixed residue
        return logits

    def _apply_constraints(self, logits: torch.Tensor, t: int) -> torch.Tensor:
        """Apply non-dummy constraints: repetition penalty, weights/biases, AA count suppression, top-p.

        Shared across all non-ProteinMPNN wrappers. Called only in the non-dummy path,
        after logit extraction, centering, alphabet mapping, and stop-index zeroing.
        """
        logits = self._apply_repetition_penalty(logits, t)
        logits = self._apply_weights_and_biases(logits, t)

        aa_count = F.one_hot(
            self.S[:, self.decoded_positions[0].bool()], num_classes=self.vocab_size
        ).sum(1)[:, self.alphabet_map]
        logits[aa_count >= self.config.max_aa_count] = Constants.MIN_LOGIT

        if (aa_count[0, 6] + aa_count[0, 8] + aa_count[0, 14]) >= self.config.max_pos_count:
            logits[0, 6] = Constants.MIN_LOGIT
            logits[0, 8] = Constants.MIN_LOGIT
            logits[0, 14] = Constants.MIN_LOGIT

        logits = BaseWrapper._top_p(logits, self.config.truncate_topp)
        return logits

    @staticmethod
    def parse_fasta(
        filename: str, 
        limit: int = -1
    ) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]:    
        header = []
        sequence = []
        lines = open(filename, "r")
        for line in lines:
            line = line.rstrip()
            if line[0] == ">":
                if len(header) == limit:
                    break
                header.append(line[1:])
                sequence.append([])
            else:
                sequence[-1].append(line)
        lines.close()
        table = str.maketrans('', '', string.ascii_lowercase)
        sequence = [''.join(seq).translate(table) for seq in sequence]
        
        return np.array(header), np.array(sequence)

    @staticmethod
    def load_a3m(
        path_to_msa: str, 
        device: str, 
        gap_cutoff_v: float = 0.5, 
        gap_cutoff_h: float = 0.25
    ) -> List[str]:
        names, seqs = BaseWrapper.parse_fasta(path_to_msa)
        seqs = seqs.view('S4').reshape((seqs.size, -1)).astype('U1')
        seqs = seqs[((seqs=='-').sum(-1)/seqs.shape[1]) < gap_cutoff_h]
        seqs = seqs[:,((seqs=='-').sum(0)/seqs.shape[0]) < gap_cutoff_v]
        seqs = [ ''.join(s) for s in seqs ]
        return seqs

    @staticmethod
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
            pdist = F.pdist(msa*1.0,p=0)
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
        
class ZeroOrderWrapper(BaseWrapper):
    """
    Zero order model for testing / decoding order initialization purposes
    """
    def __init__(
        self, 
        model: torch.Tensor, # Logit tensor for each AA
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.temp = temperature
        self.logits = model
        self.vocab_size = self.logits.shape[1]

        self.alphabet_map = torch.arange(Constants.ALPHABET_SIZE, device=self.device) #Dummy
        self.alphabet_map_rev = torch.arange(Constants.ALPHABET_SIZE, device=self.device) #Dummy

        tmp = torch.zeros(self.config.length, device=self.device) - 1 #Position relative to target protein
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos-1] = Constants.ALPHABET.index(aa)
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and AA index at fixed positions    
        
        self.reset(self.decoding_order, self.rand_base)   

    def reset(
        self, 
        decoding_order: torch.Tensor, 
        rand_base: float, 
        seed_S: Optional[torch.Tensor] = None
    ) -> None:
        """
        Clears sequences & resets tracking variables.
        
        Args:
            decoding_order: Tensor specifying position decoding order
            rand_base: Rand seed
            seed_S: Optional seed sequence to start with
        """
        self.rand_base = rand_base
        self._reset_decoding_order(decoding_order)
        
        if seed_S is not None:
            self.S = seed_S.clone()
        else:
            self.S = torch.zeros((1, self.config.length)).long().to(self.device)
            
        self.gap_map = torch.arange(self.decoding_order.shape[1], device=self.device) #Dummy
        self.gap_map_rev = self.gap_map.clone()
        
        self.current_logits = None
        
        self.next_t = 0 #Iteration step; used as index for decoding orders
        
        self.decoded_positions = torch.zeros(self.config.length, device=self.device).unsqueeze(0) #This will track decoded positions during design iterations
        self.selected_aa = torch.zeros(self.config.length, device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.config.length, device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.config.length, self.logits.shape[1]), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.config.length, device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
            
    def _reset_decoding_order(self, decoding_order):
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)

    def get_logits(self): # Dummy, always returns logit tensor
        return self.logits
        
    def decode_next(self, dummy_run=False, mask_current=False, use_t=None):
        if use_t is not None:
            t = use_t
        else:
            t = self.decoding_order[0, self.next_t]

        if dummy_run:
            self.current_logits = torch.zeros(self.logits.shape, device=self.device)
        else:
            self.current_logits = self.get_logits()

        if t > -1:
            if self.config.force_stop and (t == self.end_pos):
                logits = self._force_stop()
                return logits, logits

            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map]
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

    def get_tied_positions(self) -> list[int]:
        """Positions to update together with current decode step.

        Default: current position only. Override in subclasses that support
        tied/symmetric decoding (e.g. ProteinMPNN multimer).
        """
        t = self.decoding_order[0, self.next_t]
        return [t.item()]

    def decode_all(
        self, 
        temp: float = 1e-12, 
        use_S: Optional[torch.Tensor] = None, 
        mask_current: bool = False
    ) -> bool:
        """
        Mainly used for pseudolikelihood calculation and for designing non-overlapping proteins
        
        Args:
            temp: Temperature for sampling from logits.
            use_S: Optional sequence tensor to use instead of sampling.
                If provided, amino acids are taken from this sequence
            mask_current: Whether to mask the current position during decoding
        """
        if not (self.next_t == 0):
            return False
        for i in tqdm(range(self.decoding_order.shape[1]), disable=self.tqdm_disable):
            logits, logits_ = self.decode_next(mask_current=mask_current)
            if use_S is None:
                probs = torch.nn.functional.softmax(logits/temp, dim=-1)
                S_t = torch.multinomial(probs[0], 1)
            else:
                t = self.decoding_order[:, i]
                if not (self.config.force_stop and (t == self.end_pos)):
                    S_t = use_S[t]
                else:
                    S_t = None
            self.update_S(S_t, alphabet_map=False)
        return True