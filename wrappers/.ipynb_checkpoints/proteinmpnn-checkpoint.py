from tqdm import tqdm

import numpy as np
import torch

from constants import *
from utils import *
from wrappers.protein_mpnn_utils import gather_nodes, cat_neighbors_nodes, _scores, parse_PDB, StructureDatasetPDB, ProteinMPNN

def process_pdb(
    pdb_path: str,
    ca_only: bool,
    fixed_chains: Optional[List[str]],
    design_chains: List[str],
    undecoded_chains: Optional[List[str]]
) -> Tuple[Dict[str, Any], Tuple[List[str], List[str], List[str], List[str]]]:
    """
    Process a PDB file for ProteinMPNN input, extracting chain information and structure data.
    
    Args:
        pdb_path: Path to the PDB file to process
        ca_only: If True, only use alpha carbon atoms in the structure
        fixed_chains: List of chain IDs to keep fixed
        design_chains: List of chain IDs to be designed
        undecoded_chains: List of chain IDs to remain undecoded
    
    Returns:
        Tuple containing:
            - pdb_data: Processed structure data from StructureDatasetPDB
            - chain_id: Tuple of (fixed_chain_list, design_chain_list, 
              undecoded_chain_list, all_chain_list) where each element is a 
              sorted list of chain IDs
    """
    pdb_dict_list = parse_PDB(pdb_path, ca_only=ca_only)
    pdb_data = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=100000)[0] #Only 1 PDB
    all_chain_list = [ item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain' ]
    
    #Sort alphabetically
    all_chain_list.sort()
    fixed_chains.sort()
    design_chains.sort()
    undecoded_chains.sort()
    
    if (fixed_chains is None) and (undecoded_chains is None):
        fixed_chain_list = []
        design_chain_list = all_chain_list
    else:        
        fixed_chain_list = [ letter for letter in all_chain_list if letter in fixed_chains ]
        undecoded_chain_list = [ letter for letter in all_chain_list if letter in undecoded_chains ]
        design_chain_list = [ letter for letter in all_chain_list if letter in design_chains ]
        
    chain_id = (fixed_chain_list, design_chain_list, undecoded_chain_list, all_chain_list)
    return pdb_data, chain_id 
    
def featurize(
    device: torch.device,
    pdb: Dict[str, Any],
    chain_id: Tuple[List[str], List[str], List[str], List[str]],
    ca_only: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Featurize protein structure data. Simpler rewrite of tied_featurize from ProteinMPNN.
    Orders and concatenates coordinates (X) and sequences (S) across chains.
    Also returns position encoded residue index, mask and chain encoding
    
    Args:
        device: torch.device
        pdb: Dictionary containing protein structure data with keys like:
            - 'seq': Full sequence
            - 'seq_chain_X': Sequence for chain X
            - 'coords_chain_X': Coordinates dictionary for chain X
        chain_id: Tuple of (fixed_chains, design_chains, undecoded_chains, all_chains)
            where each element is a list of chain letters
        ca_only: If True, only use CA (alpha carbon) coordinates
    
    Returns:
        Tuple containing:
            - X: Coordinate tensor of shape [1, L, 3] if ca_only else [1, L, 4, 3]
            - S: Sequence tensor of shape [1, L] with amino acid indices
            - mask: Binary mask tensor indicating valid positions [1, L]
            - chain_encodings: Chain ID encodings [1, L], starting from 1
            - residue_idx: Position-encoded residue indices [1, L]
    """
    total_length = len(pdb['seq']) #sum of chain seq lengths
    if ca_only: #Dimensions for X
        X = np.zeros([1, total_length, 1, 3])
    else:
        X = np.zeros([1, total_length, 4, 3])
        
    S = np.zeros([1, total_length], dtype=np.int32) #Sequence tokens
    fixed_chains, design_chains, undecoded_chains, all_chains = chain_id #Lists of chain letters
    
    chain_encodings = np.zeros([1, total_length], dtype=np.int32) #First chain is 1
    residue_idx = -100 * np.ones([1, total_length], dtype=np.int32) #Residue encodings
    
    X_chains = []
    chain_seqs = []
    chain_start = 0
    for chain_id, chain_letter in enumerate(all_chains):
        chain_seq = pdb["seq_chain_"+chain_letter]
        chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq]) #Replace - with X
        chain_length = len(chain_seq)
        chain_coords = pdb["coords_chain_"+chain_letter] #this is a dictionary
        
        if ca_only:
            X_chain = np.array(chain_coords["CA_chain_"+chain_letter]) #[chain_length,1,3] #CA_diff
            if len(X_chain.shape) == 2:
                X_chain = X_chain[:, None, :]
        else:
            X_chain = np.stack([chain_coords[c] for c in ["N_chain_"+chain_letter, "CA_chain_"+chain_letter, "C_chain_"+chain_letter, "O_chain_"+chain_letter]], 1) #[chain_length,4,3]

        X_chains.append(X_chain)
        chain_seqs.append(chain_seq)
        
        chain_end = chain_start + chain_length
        chain_encodings[0, chain_start:chain_end] = chain_id
        residue_idx[0, chain_start:chain_end] = 100 * chain_id + np.arange(chain_start, chain_end) #Residue indices are encoded as 100*chain_id + position
        chain_start += chain_length
        
    #X and S
    X[0, :, :, :] = np.concatenate(X_chains, 0) #[L, 4, 3]
    isnan = np.isnan(X)
    X[isnan] = 0. #Handle missing coordinates by setting them to 0 and masking
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    
    all_seq = "".join(chain_seqs)
    S = np.asarray([ALPHABET.index(aa) for aa in all_seq], dtype=np.int32)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    
    chain_encodings = torch.from_numpy(chain_encodings).to(dtype=torch.long, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    
    if ca_only:
        X = X[:,:,0]
    else:
        X = X
    return X, S, mask, chain_encodings, residue_idx

#Helper to load ProteinMPNN model
def load_proteinmpnn_model(
    checkpoint_path: str, 
    device: torch.device, 
    ca_only: bool = False
) -> torch.nn.Module:
    hidden_dim = 128
    num_layers = 3
    backbone_noise = 0.00 #Noise is 0 during inference
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    model = ProteinMPNN(ca_only=ca_only, num_letters=len(ALPHABET), 
                        node_features=hidden_dim, edge_features=hidden_dim, 
                        hidden_dim=hidden_dim, num_encoder_layers=num_layers,
                        num_decoder_layers=num_layers, augment_eps=backbone_noise, 
                        k_neighbors=checkpoint['num_edges'])
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.requires_grad_(False)
    return model

class ProteinMPNNContainer():
    def __init__(
        self,
        device: torch.device,
        decoder_path: str,
        model: Optional[torch.nn.Module],
        ca_only: bool,
        pdb_path: str,
        fixed_chains: Optional[List[str]],
        design_chains: List[str],
        undecoded_chains: Optional[List[str]],
        tied: bool,
        tied_weight: Optional[List[Tuple[str, float]]],
        fixed_positions: Optional[List[Tuple[int, str]]],
        fixed_chain_seq: Optional[List[Tuple[str, str]]],
        decoding_order: torch.Tensor,
        end_stop: bool,
        repetition_penalty: float,
        repetition_penalty_window: int,
        logit_weight: Optional[torch.Tensor],
        logit_bias: Optional[torch.Tensor],
        aa_bias: Optional[torch.Tensor],
        max_aa_count: Optional[torch.Tensor],
        max_pos_count: float,
        truncate_topp: Optional[float],
        rand_base: float,
        tqdm_disable: bool
    ):
        """
        Initialize ProteinMPNN container with model and design parameters.
        
        Args:
            device: torch.device
            decoder_path: Path to model checkpoint
            model: Pre-loaded model (if None, loads from decoder_path)
            ca_only: Use only CA atoms
            pdb_path: Path to input PDB file
            fixed_chains: List of chain IDs to keep fixed
            design_chains: List of chain IDs to design
            undecoded_chains: List of chain IDs to leave undecoded
            tied: Whether to use tied decoding for symmetric design
            tied_weight: Weights for tied positions (chain_letter, weight)
            fixed_positions: List of (position, amino_acid) to fix
            fixed_chain_seq: List of (chain_letter, sequence) for fixed chains
            decoding_order: Tensor specifying position decoding order
            end_stop: Whether to use stop codon at sequence end
            repetition_penalty: Penalty for repeating amino acids
            repetition_penalty_window: Window size for repetition penalty
            logit_weight: Per-position weight for logits
            logit_bias: Per-position bias for amino acids
            aa_bias: Global bias for each amino acid
            max_aa_count: Maximum count for each amino acid type
            max_pos_count: Maximum count for positively charged residues
            truncate_topp: Top-p truncation value (0-1)
            rand_base: Rand number seed
            tqdm_disable: Whether to disable progress bars
        """

        self.tqdm_disable = tqdm_disable
        self.device = device
        self.ca_only = ca_only
        if model is None:
            self.model = load_proteinmpnn_model(decoder_path, self.device, ca_only=self.ca_only)
        else:
            self.model = model
        self.pdb_path = pdb_path
        self.pdb_data, self.chain_id = process_pdb(self.pdb_path, self.ca_only, fixed_chains, design_chains, undecoded_chains)
        self.fixed_chains, self.design_chains, self.undecoded_chains, self.all_chains = self.chain_id
        self.fixed_chain_seq = fixed_chain_seq #list of tuples (chain_letter, seq)
        self.n_chains = len(self.all_chains)
        self.n_design_chains = len(self.design_chains)
        self.n_fixed_chains = len(self.fixed_chains)
        self.n_undecoded_chains = len(self.undecoded_chains) #Chains to be left undecoded; useful for pairing with a residue in the other frame's container instance for hetero-multimer design

        self.alphabet_inds = torch.arange(20, device=self.device) #Dummy
        self.alphabet_map = torch.arange(len(ALPHABET), device=self.device) #Dummy
        self.alphabet_map_rev = torch.arange(len(ALPHABET), device=self.device) #Dummy
        
        self.X, self.S_orig, self.mask, self.chain_encoding, self.residue_idx = featurize(self.device, self.pdb_data, self.chain_id, self.ca_only) # Featurize structure
        self.target_chain = self.design_chains[0] #There can be multiple design chains for tied option, but only first one will be the "main" chain
        self.chain_lengths = [ (self.chain_encoding[0, :] == self.all_chains.index(chain_letter)).sum() for chain_letter in self.all_chains ]
        self.chain_offsets = [ torch.nonzero(self.chain_encoding[0, :] == self.all_chains.index(chain_letter))[0][0] for chain_letter in self.all_chains ]
        self.target_chain_id = self.all_chains.index(self.target_chain)
        self.target_chain_length = self.chain_lengths[self.target_chain_id]
        self.target_chain_offset = self.chain_offsets[self.target_chain_id]
        
        tmp = torch.zeros(self.target_chain_length, device=self.device) - 1 #Position relative to target protein
        if fixed_positions is not None:
            for pos, aa in fixed_positions:
                tmp[pos-1] = ALPHABET.index(aa)
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and AA index at fixed positions
        
        self.tied = tied
        self.end_stop = end_stop
        
        #Biases and weights
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_window = repetition_penalty_window
        
        if logit_bias is None:
            self.logit_bias = torch.zeros((self.target_chain_length, len(ALPHABET)), device=self.device)
        else:
            self.logit_bias = logit_bias
        
        if logit_weight is None:
            self.logit_weight = torch.ones(self.target_chain_length, device=self.device)
        else:
            self.logit_weight = logit_weight
            
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
        
        if self.tied: #Tied weight provided as list of tuples (chain_letter, weight)
            self.tied_weight = torch.ones(len(self.all_chains), device=self.device)
            if tied_weight is not None:
                for chain_letter, weight in tied_weight:
                    self.tied_weight[self.all_chains.index(chain_letter)] = weight
            
        self.reset(decoding_order, rand_base)

    def reset_decoding_order(self, decoding_order: torch.Tensor, keep_S: bool = True) -> None:
        """
        Reset the decoding order for protein design.
        
        This method sets up the order in which positions will be decoded during
        the design process, handling tied positions and fixed chains.
        
        Args:
            decoding_order: Tensor specifying the order of positions to decode
            keep_S: Whether to keep the current sequence tensor
        """
        self.decoding_order = decoding_order #This is relative to target chain's position in the OLG decoder. It is NOT the positions for X/S from PDB and need to be offseted
        self.end_pos = torch.max(self.decoding_order)
        self.decoding_order_e = self.decoding_order[self.decoding_order != self.end_pos].unsqueeze(0) if self.end_stop else decoding_order #Same, but excluding stop codon position
        self.decoding_order_target = self.decoding_order_e + self.target_chain_offset #This is the decoding order that corresponds to positions for X/S. Only has target chain's positions.
        
        #Make decoding order for tied decoding.
        all_decoding_order_e_offset = [ (self.decoding_order_e + self.chain_offsets[self.all_chains.index(chain_letter)]).squeeze(0).unsqueeze(1) for chain_letter in self.design_chains ] #List of offseted decoding orders for each tied chain
        all_decoding_order_e_offset_und = [ (self.decoding_order_e + self.chain_offsets[self.all_chains.index(chain_letter)]).squeeze(0).unsqueeze(1) for chain_letter in (self.design_chains+self.undecoded_chains) ] #List of offseted decoding orders for the tied chains + chains that will be updated by OLG decoder (undecoded_chains)
        
        self.decoding_order_target = torch.cat(all_decoding_order_e_offset, dim=-1).view(self.decoding_order_e.shape[1], -1).flatten().unsqueeze(0) #This interleaves the positions so that tied positions would get decoded consecutively
        self.decoding_order_target_und = torch.cat(all_decoding_order_e_offset_und, dim=-1).view(self.decoding_order_e.shape[1], -1).flatten().unsqueeze(0) #This interleaves the positions so that tied positions would get decoded consecutively, including undecoded chains   

        self.tied_pos = [ [] for i in range(self.S_orig.shape[0]) ]  #List of the partner residues given a tied position in the main target chain
        for i in range(0, self.decoding_order_target.shape[1], self.n_design_chains):
            tied = self.decoding_order_target[0, i:(i+self.n_design_chains)]
            for j in tied:
                self.tied_pos[j] = tied #Including self

        #Add the positions for fixed chains to the decoding orders so that it happens first. Note that this doesn't deal with fixed positions for the target chain, which is handled by OLG decoder
        if len(self.fixed_chains) > 0:
            fixed_chain_positions = []
            for chain_letter in self.fixed_chains:
                fixed_chain_positions += [ torch.nonzero(self.chain_encoding[0, :] == (self.all_chains.index(chain_letter)), device=self.device)[:, 0] ] #The positions here are relative to X/S.
            self.fixed_chain_positions = torch.cat(fixed_chain_positions).unsqueeze(0)
            rand_order = torch.argsort(torch.rand(self.fixed_chain_positions.shape[1])) #Randomize the order within fixed chain positions
            self.decoding_order_S = torch.cat([ self.fixed_chain_positions[:, rand_order], self.decoding_order_target_und ], dim=1) #Now the decoding order for X/S has fixed chains before target chain; this is the decoding order to be provided to forward pass function of the model
        else:
            self.decoding_order_S = self.decoding_order_target_und

        #Prepare encoder
        if keep_S:
            _, _, self.E_idx, self.h_E, self.h_EXV_encoder_fw, self.h_V_stack, self.mask_bw, self.h_EXV_encoder = self.prepare_encoder(self.decoding_order_S) #Initially, S and h_S are empty
        else:
            self.S, self.h_S, self.E_idx, self.h_E, self.h_EXV_encoder_fw, self.h_V_stack, self.mask_bw, self.h_EXV_encoder = self.prepare_encoder(self.decoding_order_S) #Initially, S and h_S are empty
            
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
        self.reset_decoding_order(decoding_order, keep_S=False)
        
        if seed_S is not None:
            self.S = seed_S.clone()
            self.h_S = self.model.W_s(self.S)

        self.gap_map = torch.arange(self.decoding_order_S.shape[1], device=self.device) #Dummy; to keep compatible with models with gap
        self.gap_map_rev = self.gap_map.clone()
        
        self.current_logits = None
        
        self.next_t = 0 #Iteration step; used as index for decoding orders
        self.next_t_full = 0 #Iteration step; used as index for decoding orders; including stop
        
        self.decoded_positions = torch.zeros(self.decoding_order_S.shape, device=self.device) #This will track decoded positions during design iterations
        
        self.selected_aa = torch.zeros(self.decoding_order_target.shape[1], device=self.device).unsqueeze(0).long() #This will keep track of AAs decoded at each position
        self.selected_log_prob = torch.zeros(self.decoding_order_target.shape[1], device=self.device).unsqueeze(0) #This will keep track of log probs for selected AA
        self.log_prob = torch.zeros((self.decoding_order_target.shape[1], len(ALPHABET)), device=self.device) #This will keep track of log probs at each step
        self.argmax_aa = torch.zeros(self.decoding_order_target.shape[1], device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        self.preset_fixed_S(self.fixed_chain_seq) #This will update S, h_S and decoded_positions with fixed chains; but not individual fixed positions within design chains
        
        self.mask_eval = (~self.decoded_positions.bool()).to(torch.float32) #Mask for fixed chain / positions; used to get log probs only over the designed regions
        fixed_chain_res = torch.nonzero(self.fixed_positions != -1)
        if fixed_chain_res.shape[0] > 0:
            self.mask_eval[0, fixed_chain_res] = 0.0

    def get_decoding_mask(
        self, 
        E_idx: torch.Tensor, 
        decoding_order: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precalculate decoding masks for forward and backward attention.
        
        Args:
            E_idx: Edge indices tensor
            decoding_order: Decoding order tensor
            
        Returns:
            Tuple of (forward_mask, backward_mask) tensors
        """
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=self.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = self.mask.view([self.mask.size(0), self.mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        return mask_fw, mask_bw
        
    def prepare_encoder(
        self, 
        decoding_order: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Initialize model encoder with input structure and decoding order. From ProteinMPNN sample() function
        
        Args:
            decoding_order: Tensor specifying decoding order
            
        Returns:
            Tuple containing:
                - S: Sequence tensor
                - h_S: Sequence embeddings
                - E_idx: Edge indices
                - h_E: Edge embeddings
                - h_EXV_encoder_fw: Forward encoder features
                - h_V_stack: Node embedding stack
                - mask_bw: Backward attention mask
                - h_EXV_encoder: Full encoder features
        """
        #Prepare node and edge embeddings
        E, E_idx = self.model.features(self.X, self.mask, self.residue_idx, self.chain_encoding+1)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=self.device)
        h_E = self.model.W_e(E)

        #Encoder is unmasked self-attention; decoder uses masked self-attention
        mask_attend = gather_nodes(self.mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = self.mask.unsqueeze(-1) * mask_attend
        for layer in self.model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, self.mask, mask_attend)

        #These precalculate the decoding order mask needed for each decoding step
        mask_fw, mask_bw = self.get_decoding_mask(E_idx, decoding_order)

        #Prepare tensors
        N_batch, N_nodes = self.X.size(0), self.X.size(1)
        h_S = torch.zeros_like(h_V, device=self.device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=self.device) + 20
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=self.device) for _ in range(len(self.model.decoder_layers))]
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)        
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        return S, h_S, E_idx, h_E, h_EXV_encoder_fw, h_V_stack, mask_bw, h_EXV_encoder
        
    def edit_S(
        self, 
        t_m: int, 
        S_t: torch.Tensor, 
        inplace: bool = False
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Edit sequence at specified position. Returns S and h_S with specified positions/AA changed
        
        Args:
            t_m: Position to edit
            S_t: New amino acid token
            inplace: Whether to modify in place
            
        Returns:
            If not inplace, returns tuple of (S, h_S)
        """
        if inplace:
            S = self.S
            h_S = self.h_S
        else:
            S = self.S.clone()
            h_S = self.h_S.clone()

        h_S[:, t_m, :] = self.model.W_s(S_t) #Get embedding
        S[:, t_m] = S_t #Update S
        if not inplace:
            return S, h_S
    
    def get_logits(
        self, 
        t_m: int, 
        h_S_m: Optional[torch.Tensor] = None, 
        mask_current: bool = False
    ) -> torch.Tensor:
        """
        Do a pass of the model and get amino acid logits for a specific position.
        
        Args:
            t_m: Position to get logits for
            h_S_m: Optional sequence embeddings
            mask_current: Whether to mask current position
            
        Returns:
            Logits tensor for amino acids at position
        """
        if not mask_current:
            h_EXV_encoder_t = self.h_EXV_encoder_fw[:, t_m:(t_m+1), :, :]
            mask_bw = self.mask_bw
        else:
            new_decoding_order_1 = self.decoding_order_S[self.decoding_order_S == t_m]
            new_decoding_order_0 = self.decoding_order_S[self.decoding_order_S != t_m]
            new_decoding_order = torch.concatenate((new_decoding_order_0, new_decoding_order_1)).unsqueeze(0) #new_decoding_order decoding order where current position is last
            self.S[:, t_m] = 20
            self.h_S[:, t_m, :] = 0.0 #Get embedding
            mask_fw, mask_bw = self.get_decoding_mask(self.E_idx, new_decoding_order)
            h_EXV_encoder_fw_ = mask_fw * self.h_EXV_encoder
            h_EXV_encoder_t = h_EXV_encoder_fw_[:, t_m:(t_m+1), :, :]

        if h_S_m is None:
            h_S = self.h_S
        else:
            h_S = h_S_m

        #Encoding
        E_idx_t = self.E_idx[:, t_m:(t_m+1), :]
        h_E_t = self.h_E[:, t_m:(t_m+1), :, :]
        h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
        
        #Decoding
        mask_t = self.mask[:, t_m:(t_m+1)]
        h_V_stack = [ h_v.clone() for h_v in self.h_V_stack ]
        for l, layer in enumerate(self.model.decoder_layers):
            h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
            h_V_t = h_V_stack[l][:, t_m:(t_m+1), :]
            h_ESV_t = mask_bw[:, t_m:(t_m+1), :, :] * h_ESV_decoder_t + h_EXV_encoder_t
            h_V_stack[l+1][:, t_m, :] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
        h_V_t = h_V_stack[-1][:, t_m, :]
        if self.tied:
            beta = self.tied_weight[self.chain_encoding[0, t_m]] #Weight for each of the tied chains
        else:
            beta = 1.0
        logits = beta * self.model.W_out(h_V_t)
        
        return logits
    
    def decode_next(
        self, 
        dummy_run: bool = False, 
        mask_current: bool = False, 
        use_t: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the next position in the sequence.
        
        Args:
            dummy_run: Whether this is a dummy run (returns zero logits)
            mask_current: Whether to mask current position
            use_t: Optional specific position to decode
            
        Returns:
            Tuple of (processed_logits with bias/weight/etc, raw_logits)
        """
        if use_t is not None:
            t = use_t #Decoding position, relative to target protein
            if not (self.end_stop and (t == self.end_pos)):
                if not self.tied:
                    t_list = [ t ]    
                else:
                    next_t = (self.decoding_order_e[0]==use_t).nonzero().item()    
                    self.decoding_order_target[0, (next_t*self.n_design_chains):((next_t+1)*self.n_design_chains)]
                
        else:
            t = self.decoding_order[0, self.next_t_full] #Decoding position, relative to target protein
            if not self.tied:
                t_list = [ self.decoding_order_target[0, self.next_t] ]
            else:
                t_list = self.decoding_order_target[0, (self.next_t*self.n_design_chains):((self.next_t+1)*self.n_design_chains)]
        
        if self.end_stop and (t == self.end_pos):
            logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
            logits[0, STOP_INDEX] = MAX_LOGIT #High number to force stop
            logits = add_noise(logits)
            return logits, logits

        if dummy_run: #All zero if dummy running
            logits_ = torch.zeros((1, len(ALPHABET)), device=self.device)
            logits = logits_
            self.current_logits = logits_.clone()
            
        else:                                 
            logits_ = 0.0
            #Decoding position, relative to X/S
            
            for t_m in t_list:
                logits_ += self.get_logits(t_m, mask_current=mask_current)
            self.current_logits = logits_.clone() #Logits at current position, unless it's a stop and > length of protein
            
            logits_ -= logits_.mean()

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

            count_pos = torch.zeros(self.decoded_positions.shape, device=self.device)
            count_pos[:, self.target_chain_offset:(self.target_chain_offset+self.target_chain_length)] = 1
            count_pos = (self.decoded_positions * count_pos) == 1
            
            #These suppress some AA's on hard thresholding of their counts
            #aa_count = torch.nn.functional.one_hot(self.S[:,self.target_chain_offset:(self.target_chain_offset+self.target_chain_length)], num_classes=len(ALPHABET)).sum(1)
            aa_count = torch.nn.functional.one_hot(self.S[:,count_pos[0]], num_classes=len(ALPHABET)).sum(1)
            max_aa = (aa_count >= self.max_aa_count)
            logits[max_aa] = MIN_LOGIT

            #Positive AA total counts
            if (aa_count[0, 6] + aa_count[0, 8] + aa_count[0, 14]) >= self.max_pos_count: #This is for positively charged AA's; H/K/R
                logits[0, 6] = MIN_LOGIT
                logits[0, 8] = MIN_LOGIT
                logits[0, 14] = MIN_LOGIT

            logits = top_p(logits, self.truncate_topp) #Top-p filtering
        
        if (not self.end_stop) or (t != self.end_pos): #Penalize stop codon if not at last position
            logits_[0, STOP_INDEX] = MIN_LOGIT
            logits[0, STOP_INDEX] = MIN_LOGIT
            
        if self.fixed_positions[t] != -1:
            logits = torch.zeros(self.aa_bias.shape, device=self.device).unsqueeze(0)
            logits[0, self.fixed_positions[t]] = MAX_LOGIT #High number to force fixed residue
            
        logits = add_noise(logits)
        return logits, logits_

    def preset_fixed_S(
        self, 
        fixed_chain_seq: Optional[List[Tuple[str, str]]]
    ) -> bool:
        """
        Update protein sequence tensor (S) and embedding (h_S) for fixed chains that won't be part of overlap
        
        Args:
            fixed_chain_seq: Optional list of tuples, where each tuple contains:
                - chain_letter: Single character chain identifier
                - seq: Amino acid sequence string for that chain
                If None, sequences are extracted from PDB data for fixed chains
                    
        Results:
            - Updates self.S with fixed chain sequences
            - Updates self.h_S with embeddings for fixed positions
            - Updates self.decoded_positions to mark fixed positions as decoded
        """
        #Track all fixed positions
        all_fixed_positions = []
        
        #Fixed chains
        if fixed_chain_seq is not None:
            for chain_letter, seq in fixed_chain_seq: #List of tuples (chain, sequence)
                seq_token = torch.tensor([ ALPHABET.index(c) for c in seq ], device=self.device)
                start = self.chain_offsets[self.all_chains.index(chain_letter)]
                end = start + len(seq)
                self.S[:, start:end] = seq_token
                all_fixed_positions += [ pos for pos in range(start, end) ]
        else: #if fixed chain seqs are not provided, then take from PDB
            if self.fixed_chains is not None:
                for chain_letter in self.fixed_chains:
                    seq = self.pdb_data["seq_chain_"+chain_letter]
                    seq_token = torch.tensor([ ALPHABET.index(c) for c in seq ], device=self.device)
                    start = self.chain_offsets[self.all_chains.index(chain_letter)]
                    end = start + len(seq)
                    self.S[:, start:end] = seq_token
                    all_fixed_positions += [ pos for pos in range(start, end) ]

        #Update h_S
        all_fixed_positions = torch.tensor(all_fixed_positions, device=self.device).sort()[0].long()
        self.h_S[:, all_fixed_positions, :] = self.model.W_s(self.S[:, all_fixed_positions]) #Update embedding
        self.decoded_positions[:, all_fixed_positions] = 1.0
        
        return True
        
    def update_S(
        self, 
        S_t: torch.Tensor, 
        alphabet_map: bool = False, 
        use_t: Optional[int] = None, 
        dummy_run: bool = False
    ) -> None:
        """
        Updates the sequence tensor (S) with a newly selected amino acid,
        advances the decoding position counters, and records selection statistics
        including log probabilities and the argmax choice.
        
        Args:
            S_t: Tensor containing the selected amino acid token(s) to insert
            alphabet_map: dummy, always False
            use_t: Optional specific position to update. If None, uses current 
                decoding position from self.next_t
            dummy_run: dummy, unused
        
        Results:
            - Updates self.S with the new amino acid at the current position(s)
            - Updates self.h_S with embeddings for the new amino acid(s)
            - Marks position(s) as decoded in self.decoded_positions
            - Records selected amino acid in self.selected_aa
            - Records log probability of selection in self.selected_log_prob
            - Records full log probability distribution in self.log_prob
            - Records argmax choice in self.argmax_aa
            - Advances decoding counters (next_t, next_t_full)
        
        Notes:
            - For tied decoding, the same amino acid is placed at multiple positions
            - Returns early if at stop codon position when end_stop is True
        """
        if use_t is None:
            t_full = self.decoding_order[0, self.next_t_full]
            if self.end_stop and (self.end_pos == t_full):
                self.next_t_full += 1
                return #Do nothing if stop codon; don't advance next_t
            t = self.decoding_order_target[0, self.next_t]
            t_list = self.decoding_order_target[0, ((self.next_t)*self.n_design_chains):((self.next_t+1)*self.n_design_chains)]
            self.next_t_full += 1
            self.next_t += 1
        else:
            t = use_t
        
        if not self.tied:
            self.edit_S(t, S_t, inplace=True)
            self.decoded_positions[0, t] = 1.0
        else:
            for t in t_list:
                self.edit_S(t, S_t, inplace=True)
                self.decoded_positions[0, t] = 1.0
        
        self.selected_aa[:, t] = S_t
        log_softmax = torch.log(torch.nn.functional.softmax(self.current_logits[0], dim=-1))
        self.selected_log_prob[:, t] = log_softmax[S_t]
        self.log_prob[t, :] = log_softmax
        self.argmax_aa[:, t] = self.current_logits[0].argmax()
        
    def get_likelihoods(
        self, 
        S: Optional[torch.Tensor] = None, 
        decoding_order: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        From ProteinMPNN; compute the log probability distribution over amino acids for each position in the sequence.
        
        Args:
            S: Optional sequence tensor to evaluate. If None, uses self.S
            decoding_order: Optional custom decoding order. If None, uses self.decoding_order_S
        
        Returns:
            torch.Tensor: Log probability tensor of shape [batch, length, alphabet_size]
                containing log probabilities for each amino acid at each position
        
        Side Effects:
            - Updates self.log_prob with the computed log probabilities
            - Updates self.selected_log_prob with negative log likelihood of the sequence
            - Sets all positions in self.decoded_positions as decoded after this operation
        """
        if S is None:
            S = self.S
        if decoding_order is None:
            decoding_order = self.decoding_order_S
        randn = torch.rand(self.mask.shape, device=self.device)  #This is a dummy parameter, using specified decoding order
        
        log_probs = self.model(self.X, S, self.mask, self.mask_eval, self.residue_idx, self.chain_encoding+1, randn, use_input_decoding_order=True, decoding_order=decoding_order)
        self.log_prob = log_probs[0]
        #self.argmax_aa = self.log_prob.argmax()
        criterion = torch.nn.NLLLoss(reduction='none')
        self.selected_log_prob = criterion(log_probs.contiguous().view(-1,log_probs.size(-1)),S.contiguous().view(-1)).view(S.size())
        self.decoded_positions.fill_(1.0)
            
        return log_probs
    
    def get_score(
        self, 
        S: Optional[torch.Tensor] = None, 
        ar_ll: bool = False, 
        decoding_order: Optional[torch.Tensor] = None, 
        positions: Optional[torch.Tensor] = None
    ) -> float:
        """
        Calculate the log likelihood score for a protein sequence.
        
        Args:
            S: Optional sequence tensor to score. If None, uses current sequence (self.S)
                To rescore current sequence, pass self.S.clone() explicitly
            ar_ll: If True, uses autoregressive log likelihood computation from ProteinMPNN utils
                if False, uses pseudolikelihood (mask each position, get conditional prob on rest of seq)
            decoding_order: Optional custom decoding order for ar_ll mode.
                If None, uses self.decoding_order_S
            positions: Optional tensor of specific positions to score.
                If provided, only these positions contribute to the score
        
        Returns:
            float: Negative log likelihood score; lower = better
        """
        if S is None: #To rescore with current sequence, use self.S.clone() as input
            S = self.S.clone()
            
        if not ar_ll:
            self.reset(self.decoding_order, self.rand_base, S)
            self.decode_all(use_S=S[0], mask_current=True)
            if positions is not None:
                return (self.selected_log_prob * -1.0)[0, positions].mean()
            return (self.selected_log_prob.mean() * -1.0)
        else:
            if decoding_order is None:
                decoding_order = self.decoding_order_S
            log_probs = self.get_likelihoods(S, decoding_order)
            mask_for_loss = self.mask * self.mask_eval
            if positions is not None:
                mask_for_loss[positions] = 0
            return _scores(S, log_probs, mask_for_loss)[0]

    def get_prot_seq(self, S: Optional[torch.Tensor] = None) -> Optional[str]:
        if S is None:
            S = self.S[:, self.target_chain_offset:(self.target_chain_offset+self.target_chain_length)] #Sequence for only design target chain
        prot = ''.join([ALPHABET[s.item()] for s in S[0, :]])
        return prot
        
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
        if not ((self.next_t == 0) and (self.next_t_full == 0)):
            return False
        for i in tqdm(range(self.decoding_order.shape[1]), disable=self.tqdm_disable):
            logits, logits_ = self.decode_next(mask_current=mask_current)
            if use_S is None:
                probs = torch.nn.functional.softmax(logits/temp, dim=-1)
                S_t = torch.multinomial(probs[0], 1)
            else:
                t = self.decoding_order[:, i]
                if not (self.end_stop and (t == self.end_pos)):
                    S_t = use_S[t]
                else:
                    S_t = None
            self.update_S(S_t, alphabet_map=False)
        return True