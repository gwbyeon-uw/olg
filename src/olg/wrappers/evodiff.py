from tqdm import tqdm

from typing import Tuple, List, Union, Literal, Optional

import torch
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from olg.constants import *

from evodiff.utils import Tokenizer
from evodiff.pretrained import MSA_OA_DM_MAXSUB, ESM_MSA_1b #https://github.com/microsoft/evodiff/tree/main#loading-pretrained-models

from .base_wrapper import BaseWrapper

class WrapperEvoDiff(BaseWrapper):
    # 'X' (stop codon) maps to '-' (gap token) in EvoDiff vocabulary
    _DEFAULT_EXTRA_AA_MAP: dict[str, str] = {'X': '-'}

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        msa_seqs: List[str],
        msa_n_seq: int,
        msa_max_length: int,
        msa_selection_type: Literal['random', 'MaxHamming', 'MaxHammingI'] = 'random',
        use_esm_msa: bool = False,
        prefixed_seq: Optional[List[Tuple[int, int, str]]] = None,
        extra_aa_map: Optional[dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_esm_msa = use_esm_msa #Flag to use MSA Transformer instead of EvoDiff-MSA
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.alphabet)
        self._build_alphabet_maps(self.tokenizer.a_to_i, extra_aa_map, self._DEFAULT_EXTRA_AA_MAP)

        #MSA subsampling
        self.msa_seqs = msa_seqs
        self.msa_n_seq = msa_n_seq # number of sequences in MSA to subsample
        self.msa_max_length = msa_max_length # maximum sequence length to subsample
        self.seq_len = self.msa_max_length #Set to same as MSA length for now
        self.msa_selection_type = msa_selection_type # or 'MaxHamming'; MSA subsampling scheme

        # Seed numpy's global RNG from rand_base so MSA subsampling is reproducible
        # (subsample_msa uses np.random.choice for the slice window and sequence selection)
        if self.rand_base is not None:
            np.random.seed(int(self.rand_base))
        self.valid_msa_, self.query_sequence, _ = self.subsample_msa(
            self.msa_seqs, 
            n_sequences=self.msa_n_seq,
            max_seq_len=self.msa_max_length,
            selection_type=self.msa_selection_type
        )
        self.valid_msa = torch.tensor(np.array([self.tokenizer.tokenizeMSA(seq) for seq in self.valid_msa_]), device=self.device) #Tokenize sequence
        self.padding = torch.full((self.msa_n_seq, self.msa_max_length-self.seq_len), fill_value=self.tokenizer.pad_id, device=self.device)
        
        self.prefixed_seq = prefixed_seq #List of tuples, (start, end, seq)
        
        self.remap_to_evodiff = Constants.REMAP_TO_EVODIFF.to(self.device)
        self.remap_to_esmmsa = Constants.REMAP_TO_ESM_MSA.to(self.device)

        self.gap_positions =  None
        self.gap_map = torch.arange(self.seq_len, device=self.device) #From our protein position to MSA position with gaps
        self.gap_map_rev = self.gap_map.clone()
        if self.config.gap_positions is not None:
            self.gap_positions = torch.tensor(self.config.gap_positions, device=self.device).sort()[0] - 1 #to 0-based
            self.gap_map[self.gap_positions] = -1
            self.gap_map = self.gap_map[self.gap_map!=-1] 
            self.gap_map_rev[self.gap_positions] = -1
            self.gap_map_rev[self.gap_map_rev!=-1] = torch.arange(self.gap_map.shape[0], device=self.device)
            
        tmp = torch.zeros(self.seq_len, device=self.device) - 1 #Position relative to target protein
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos-1] = self.alphabet_index[aa]
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and OLG-internal AA index at fixed positions
        
        self.reset(self.decoding_order, self.rand_base)

    @staticmethod
    def _load_evodiff_model(device):
        checkpoint = MSA_OA_DM_MAXSUB()
        model, _, tokenizer, _ = checkpoint
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        return model, tokenizer

    @staticmethod
    def _load_esmmsa_model(device):
        checkpoint = ESM_MSA_1b()
        model, _, alphabet, _ = checkpoint
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        evodiff_tokenizer = Tokenizer()
        return model, alphabet, evodiff_tokenizer

    @staticmethod
    def tokenizeMSA(
        seq: Union[str, List[str]]
    ) -> npt.NDArray[np.int_]:
        return np.array([Constants.EVODIFF_ALPHABET[a] for a in seq])
    
    @staticmethod
    def subsample_msa(
        parsed_msa: List[str], 
        n_sequences: int = 64, 
        max_seq_len: int = 512, 
        selection_type: Literal['random', 'MaxHamming', 'MaxHammingI'] = 'random'
    ) -> Tuple[List[str], str, List[str]]:
        """
        Modified from https://github.com/microsoft/evodiff/blob/main/evodiff/data.py
        Subsample an MSA (Multiple Sequence Alignment) based on different selection strategies.
        
        Args:
            parsed_msa: List of sequences in the MSA, where aligned positions are uppercase
                or '-', and unaligned positions are lowercase or '.'.
            n_sequences: Number of sequences to subsample. Must be <= number of sequences in MSA.
            max_seq_len: Maximum sequence length to consider. If MSA is longer, a random slice
                of this length is used.
            selection_type: Strategy for selecting sequences:
                - 'random': Randomly select sequences
                - 'MaxHamming': Maximize Hamming distance between selected sequences,
                  starting with a random seed
                - 'MaxHammingI': Like MaxHamming but starts with the first sequence as seed
        
        Returns:
            Tuple containing:
                - output: List of aligned sequences as strings
                - anchor_seq: The first (query) sequence from the output
                - unal: List of the original unaligned sequences corresponding to selected sequences
        """    
        alphabet = Constants.EVODIFF_ALPHABET #EvoDiff alphabet
        alpha = np.array(list(alphabet))
        gap_idx = Constants.EVODIFF_ALPHABET['-']
        pad_idx = Constants.EVODIFF_ALPHABET['!']
        
        #Do hamming distance from aligned section
        aligned_msa = [ [ char for char in seq if (char.isupper() or char == '-') and not char == '.' ] for seq in parsed_msa ]   
    
        tokenized_msa = [ WrapperEvoDiff.tokenizeMSA(seq) for seq in aligned_msa ]
        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
        msa_seq_len = len(tokenized_msa[0])
    
        if msa_seq_len > max_seq_len:
            slice_start = np.random.choice(msa_seq_len - max_seq_len + 1)
            seq_len = max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len
    
        sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + max_seq_len]
        anchor_seq = sliced_msa_seq[0]  # This is the query sequence in MSA
    
        # slice out all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [gap_idx])]
        msa_num_seqs = len(sliced_msa)
    
        if msa_num_seqs < n_sequences:
            output = np.full(shape=(n_sequences, seq_len), fill_value=pad_idx)
            output[:msa_num_seqs] = sliced_msa
            unal = parsed_msa
            raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
        elif msa_num_seqs > n_sequences:
            if selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
                unal = [ parsed_msa[i] for i in random_idx ]
            elif selection_type == "MaxHamming" or selection_type == "MaxHammingI":
                unal_inds = [0]
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                
                if selection_type == "MaxHammingI":
                    random_ind = 0
                else:
                    random_ind = np.random.choice(msa_ind)
                    
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                unal_inds.append(random_ind)
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((n_sequences - 2, m))
                msa_ind = np.delete(msa_ind, msa_ind[msa_ind==(random_ind-1)]-1)
    
                for i in range(n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    unal_inds.append(msa_ind[random_ind])
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    msa_ind = np.delete(msa_ind, random_ind)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
                    
                unal = [ parsed_msa[i] for i in unal_inds ]
        else:
            unal = parsed_msa
            output = sliced_msa
    
        output = [''.join(seq) for seq in alpha[output]]
        return output, output[0], unal
    
    def _reset_decoding_order(self, decoding_order):
        self.decoding_order = decoding_order
        self.end_pos = torch.max(self.decoding_order)
    
    #Resets decoding; sequences are emptied and various trackers are set to zero
    def reset(self, decoding_order, rand_base, seed_S=None):
        self.rand_base = rand_base
        self._reset_decoding_order(decoding_order)
        
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
                    self.update_S(S_t=Constants.GAP_TOKEN, use_t_msa=p, alphabet_map=False)
                    
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

        if not (self.config.force_stop and (t == self.end_pos)):
            if dummy_run:
                self.current_pred = torch.zeros((self.S_msa.shape[2], len(self.tokenizer.alphabet)), device=self.device)
            else:
                if mask_current:
                    self.S_msa[:, 0, t_msa] = self.tokenizer.mask_id
    
                if self.use_esm_msa:
                    remapped_input = torch.nn.functional.pad(torch.gather(self.remap_to_esmmsa.expand(self.S_msa.shape[1], -1), 1, self.S_msa[0]), (1, 0)).unsqueeze(0)
                    output_ = self.model(remapped_input)  #Output shape of preds is (BS=1, N=64, L, n_tokens=31)
                    output = torch.zeros((1, self.S_msa.shape[1], self.S_msa.shape[2], len(self.tokenizer.a_to_i)), device=self.device).fill_(Constants.MIN_LOGIT)
                    output[:, :, :, self.remap_to_evodiff[self.remap_to_evodiff!=-1]] = output_['logits'][:, :, 1:, self.remap_to_evodiff!=-1]
                    self.current_pred = output[0, 0, :, :]
                else:
                    self.current_pred = self.model(self.S_msa)[0, 0, :, :]  #Output shape of preds is (BS=1, N=64, L, n_tokens=31)

        if t > -1:
            if (use_t_msa is None) and self.config.force_stop and (t == self.end_pos): #Everything is zero except stop index
                logits = self._force_stop()
                return logits, logits
                
            self.current_logits = self.current_pred[t_msa, :].unsqueeze(0) #Logits at current position, unless it's a stop and > length of protein
    
            if dummy_run:
                logits_ = self.current_logits.clone()[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, self.stop_index] = Constants.MIN_LOGIT #Zero out the index for X
                logits = logits_.clone()
            else:
                logits_ = self.current_logits.clone() #Only first row and standard AAs
                logits_ -= logits_.mean() #Center it
                logits_ = logits_[:, self.alphabet_map] #Only the alphabet we use
                logits_[:, self.stop_index] = Constants.MIN_LOGIT #Zero out the index for X

                logits = logits_.clone()
                logits = self._apply_constraints(logits, t)

            if (use_t_msa is None) and ((not self.config.force_stop) or (t != self.end_pos)):
                logits_ = self._penalize_stop(logits_)
                logits = self._penalize_stop(logits)

            if (use_t_msa is None) and self.fixed_positions[t] != -1:
                logits = self._force_fixed_positions(logits, t)

            logits = BaseWrapper._add_noise(logits)
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
            if self.config.force_stop and (t == self.end_pos):
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
        t = torch.arange(fixed_start, fixed_end + 1, device=self.device)
        t_msa = self.gap_map[t] #Decoding position, relative to the MSA of the target protein
        # Convert fixed_seq chars directly to EvoDiff native tokens (may include '-' for gaps)
        fixed_token = torch.tensor([ self.tokenizer.a_to_i.get(c, self.tokenizer.a_to_i['-']) for c in fixed_seq ], device=self.device)
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
            S = self.alphabet_map_rev[self.S[0, self.config.start_offset:self.seq_len]]
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