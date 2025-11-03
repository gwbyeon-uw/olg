from typing import Any, Tuple
import string

import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform

from constants import *
from config import ProteinConfig

class GuidanceWrapper:
    def __init__(
        self,
        device: torch.device,
        config: Any,
        rand_base: float
    ):
        self.device = device
        self.config = config
        self.rand_base = rand_base

    @staticmethod
    def unconditional_rate(xt: torch.Tensor, logits: torch.Tensor, stochasticity: float, t: float, temp: float) -> torch.Tensor: # B, L, C
        pt_x1_probs = F.softmax(logits / temp, dim=-1)
        xt_is_mask = (xt == self.mask_idx).view(1, logits.shape[1], 1).float()
        R_t = xt_is_mask * pt_x1_probs * ((1 + stochasticity * t) / (1 - t)) # B, L, C
        remask_rates = (1 - xt_is_mask) * stochasticity
        R_t += remask_rates
        return R_t

    @staticmethod
    def rate_to_prob(xt: torch.Tensor, R_t: torch.Tensor, dt: float, log_prob: bool) -> torch.Tensor:
        # Set the diagonal of the rates to negative row sum
        R_t.scatter_(-1, xt[:, :, None], 0.0)
        R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

        # Obtain probabilities from the rates
        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        step_probs.scatter_(
            -1,
            xt[:, :, None],
            (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        
        if log_prob:
            return torch.log(step_probs)
        else:
            return step_probs

    @staticmethod
    def guided_rate(
        log_prob_ratio: torch.Tensor, 
        R_t: torch.Tensor, 
        guide_temp: float = 1.0, 
        log_prob_ratio_cutoff: float = 80.0
    ) -> torch.Tensor:
        log_prob_ratio /= guide_temp # Scale log prob ratio by temperature
        log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff) # Clamp the log prob ratio
        prob_ratio = torch.exp(log_prob_ratio) # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
        R_t = R_t * prob_ratio # Multiply the reverse rate elementwise with the density ratio
        return R_t
        
    def conditional_rate(self, current_S: torch.Tensor) -> torch.Tensor: # S = B, L, C
        # Taylor-approximated guidance (TAG) based on https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py
        S = current_S.clone()
        # \grad_{x}{log p(y|x)}(z_t), shape (B, L, C)
        with torch.enable_grad():
            S.requires_grad_(True)
            # log p(y|x=z_t), shape (B,)
            log_prob = self.predictor(S)
            log_prob.sum().backward()
            # Shape (B, L, C)
            grad_log_prob = S.grad
        # 1st order Taylor approximation of the log difference
        # Shape (B, L, C)
        log_prob_ratio = grad_log_prob - (S * grad_log_prob).sum(dim=-1, keepdim=True)

        return log_prob_ratio

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