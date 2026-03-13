from tqdm import tqdm

from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from olg.constants import *
from olg.config import ProteinConfig

from ._vendored.protein_mpnn_utils import gather_nodes, cat_neighbors_nodes, _scores, parse_PDB, StructureDatasetPDB, ProteinMPNN
from .base_wrapper import BaseWrapper

class WrapperProteinMPNN(BaseWrapper):
    # ProteinMPNN's fixed training vocabulary (21 tokens, identical ordering to Constants.DEFAULT_ALPHABET)
    _NATIVE_LETTERS: str = 'ACDEFGHIKLMNPQRSTVWYX'
    _NATIVE_VOCAB: dict[str, int] = {c: i for i, c in enumerate('ACDEFGHIKLMNPQRSTVWYX')}
    _NATIVE_VOCAB_SIZE: int = 21
    # No default remapping needed: standard OLG alphabet == ProteinMPNN native vocab
    _DEFAULT_EXTRA_AA_MAP: dict[str, str] = {}

    def __init__(
        self,
        model: torch.nn.Module,
        ca_only: bool,
        pdb_path: str,
        fixed_chains: Optional[List[str]] = [],
        design_chains: List[str] = [ "A" ],
        undecoded_chains: Optional[List[str]] = [],
        tied: bool = False,
        tied_weight: Optional[List[Tuple[str, float]]] = [],
        fixed_chain_seq: Optional[List[Tuple[str]]] = [],
        chain_mask: Optional[List] = None,
        extra_aa_map: Optional[Dict[str, str]] = None,
        pad: Tuple[int, int] = (0, 0),
        **kwargs
    ):
        """
        Initialize ProteinMPNN container with model and design parameters.

        Args:
            model: Pre-loaded model (if None, loads from decoder_path)
            ca_only: Use only CA atoms
            pdb_path: Path to input PDB file
            fixed_chains: List of chain IDs to keep fixed
            design_chains: List of chain IDs to design
            undecoded_chains: List of chain IDs to leave undecoded
            tied: Whether to use tied decoding for symmetric design
            tied_weight: Weights for tied positions (chain_letter, weight)
            fixed_chain_seq: List of (chain_letter, sequence) for fixed chains
            chain_mask: Per-chain masks, one entry per chain in all_chains order.
                Each entry can be: None (no masking), a list of int positions
                within that chain to force mask=0, or an array/tensor of shape
                [chain_length] used as the base mask (1=valid, 0=masked).
            pad: (pad_n, pad_c) number of dummy Gly residues to prepend/append
                to the target (first design) chain.  Padded positions get NaN
                coordinates and mask=0 so ProteinMPNN ignores them structurally,
                but they are real design positions included in config.length.
        """
        super().__init__(**kwargs)

        self.ca_only = ca_only
        self.model = model
        self._build_alphabet_maps(self._NATIVE_VOCAB, extra_aa_map, self._DEFAULT_EXTRA_AA_MAP)
        self._set_target_from_pdb(pdb_path, fixed_chains, design_chains, undecoded_chains,
                                  tied, tied_weight, fixed_chain_seq, chain_mask=chain_mask,
                                  pad=pad)

        self.reset(self.decoding_order, self.rand_base)

    def _set_target_from_pdb(self, pdb_path, fixed_chains=[], design_chains=["A"], undecoded_chains=[], tied=False, tied_weight=[], fixed_chain_seq=[],
                             chain_mask=None, pad=(0, 0)):
        self.pad_n, self.pad_c = pad
        self.pdb_data, self.chain_id = self._process_pdb(pdb_path, self.ca_only, fixed_chains, design_chains, undecoded_chains)
        self.fixed_chains, self.design_chains, self.undecoded_chains, self.all_chains = self.chain_id
        self.fixed_chain_seq = fixed_chain_seq #list of tuples (chain_letter, seq)
        self.n_chains = len(self.all_chains)
        self.n_design_chains = len(self.design_chains)
        self.n_fixed_chains = len(self.fixed_chains)
        self.n_undecoded_chains = len(self.undecoded_chains) #Chains to be left undecoded; useful for pairing with a residue in the other frame's container instance for hetero-multimer design

        # Inject padding into pdb_data before featurization
        if self.pad_n > 0 or self.pad_c > 0:
            if tied:
                raise ValueError("pad != (0,0) is not supported with tied=True")
            target_chain = self.design_chains[0]
            self._pad_chain(self.pdb_data, target_chain, self.pad_n, self.pad_c, self.ca_only)
            target_chain_idx = self.all_chains.index(target_chain)
            chain_mask = self._adjust_chain_mask(chain_mask, target_chain_idx, self.pad_n, self.pad_c)

        self.X, self.S_orig, self.mask, self.chain_encoding, self.residue_idx = self._featurize(
            self.device, self.pdb_data, self.chain_id, self.ca_only, chain_mask=chain_mask
        ) # Featurize structure
        self.target_chain = self.design_chains[0] #There can be multiple design chains for tied option, but only first one will be the "main" chain
        self.chain_lengths = [ (self.chain_encoding[0, :] == self.all_chains.index(chain_letter)).sum() for chain_letter in self.all_chains ]
        self.chain_offsets = [ torch.nonzero(self.chain_encoding[0, :] == self.all_chains.index(chain_letter))[0][0] for chain_letter in self.all_chains ]
        self.target_chain_id = self.all_chains.index(self.target_chain)
        self.target_chain_length = self.chain_lengths[self.target_chain_id]
        self.target_chain_offset = self.chain_offsets[self.target_chain_id]

        tmp = torch.zeros(self.target_chain_length, device=self.device) - 1 #Position relative to target protein
        if self.config.fixed_positions is not None:
            for pos, aa in self.config.fixed_positions:
                tmp[pos-1] = self.alphabet_index[aa]
        self.fixed_positions = tmp.long() #This will have -1 non-fixed positions and OLG-internal AA index at fixed positions
        
        self.tied = tied
        
        if self.tied: #Tied weight provided as list of tuples (chain_letter, weight)
            self.tied_weight = torch.ones(len(self.all_chains), device=self.device)
            if tied_weight is not None:
                for chain_letter, weight in tied_weight:
                    self.tied_weight[self.all_chains.index(chain_letter)] = weight

    #Helper to load ProteinMPNN model
    @staticmethod
    def _load_proteinmpnn_model(
        checkpoint_path: str, 
        device: torch.device, 
        ca_only: bool = False
    ) -> torch.nn.Module:
        hidden_dim = 128
        num_layers = 3
        backbone_noise = 0.00 #Noise is 0 during inference
        checkpoint = torch.load(checkpoint_path, map_location=device) 
        model = ProteinMPNN(ca_only=ca_only, num_letters=WrapperProteinMPNN._NATIVE_VOCAB_SIZE,
                            node_features=hidden_dim, edge_features=hidden_dim, 
                            hidden_dim=hidden_dim, num_encoder_layers=num_layers,
                            num_decoder_layers=num_layers, augment_eps=backbone_noise, 
                            k_neighbors=checkpoint['num_edges'])
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.requires_grad_(False)
        return model

    @staticmethod
    def _process_pdb(
        pdb_path: str,
        ca_only: bool,
        fixed_chains: Optional[List[str]],
        design_chains: List[str],
        undecoded_chains: Optional[List[str]]
    ) -> Tuple[Dict[str, Any], Tuple[List[str]]]:
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

    @staticmethod
    def _pad_chain(
        pdb_data: Dict[str, Any],
        chain_letter: str,
        pad_n: int,
        pad_c: int,
        ca_only: bool,
    ) -> None:
        """Mutate pdb_data to prepend/append dummy Gly residues with NaN coords.

        NaN coordinates cause _featurize to set mask=0 for padded positions
        automatically via its np.isfinite check, so no _featurize changes are
        needed.

        Args:
            pdb_data: Parsed PDB dict (modified in place).
            chain_letter: Which chain to pad (e.g. "A").
            pad_n: Number of Gly residues to prepend (N-terminal).
            pad_c: Number of Gly residues to append (C-terminal).
            ca_only: Whether only CA atoms are present.
        """
        seq_key = f"seq_chain_{chain_letter}"
        pdb_data[seq_key] = "G" * pad_n + pdb_data[seq_key] + "G" * pad_c

        nan3 = [float("nan")] * 3
        coords_dict = pdb_data[f"coords_chain_{chain_letter}"]
        if ca_only:
            # CA coords in ca_only mode are [L, 1, 3] (triply nested lists)
            nan_n = [[nan3] for _ in range(pad_n)]
            nan_c = [[nan3] for _ in range(pad_c)]
            key = f"CA_chain_{chain_letter}"
            coords_dict[key] = nan_n + coords_dict[key] + nan_c
        else:
            # N/CA/C/O coords are [L, 3] (doubly nested lists)
            nan_n = [nan3 for _ in range(pad_n)]
            nan_c = [nan3 for _ in range(pad_c)]
            for atom in ["N", "CA", "C", "O"]:
                key = f"{atom}_chain_{chain_letter}"
                coords_dict[key] = nan_n + coords_dict[key] + nan_c

    @staticmethod
    def _adjust_chain_mask(
        chain_mask: Optional[List],
        target_chain_idx: int,
        pad_n: int,
        pad_c: int,
    ) -> Optional[List]:
        """Expand a user-provided chain_mask entry to account for padding.

        Args:
            chain_mask: Per-chain mask list (may be None).
            target_chain_idx: Index of the padded chain in all_chains.
            pad_n: N-terminal padding count.
            pad_c: C-terminal padding count.

        Returns:
            The (possibly modified) chain_mask list, or None.
        """
        if chain_mask is None:
            return None
        if target_chain_idx >= len(chain_mask):
            return chain_mask
        cm = chain_mask[target_chain_idx]
        if cm is None:
            return chain_mask

        chain_mask = list(chain_mask)  # shallow copy so caller's list is not mutated
        if isinstance(cm, np.ndarray):
            chain_mask[target_chain_idx] = np.concatenate(
                [np.zeros(pad_n, dtype=cm.dtype), cm, np.zeros(pad_c, dtype=cm.dtype)]
            )
        elif isinstance(cm, torch.Tensor):
            chain_mask[target_chain_idx] = torch.cat([
                torch.zeros(pad_n, dtype=cm.dtype, device=cm.device),
                cm,
                torch.zeros(pad_c, dtype=cm.dtype, device=cm.device),
            ])
        else:
            # list of int positions to zero out — shift by pad_n
            chain_mask[target_chain_idx] = [p + pad_n for p in cm]
        return chain_mask

    @staticmethod
    def _featurize(
        device: torch.device,
        pdb: Dict[str, Any],
        chain_id: Tuple[List[str], List[str], List[str], List[str]],
        ca_only: bool = False,
        chain_mask: Optional[List] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Featurize protein structure data. Simpler rewrite of tied_featurize from ProteinMPNN.
        Orders and concatenates coordinates (X) and sequences (S) across chains.
        Also returns position encoded residue index, mask and chain encoding.

        Args:
            device: torch.device
            pdb: Dictionary containing protein structure data with keys like:
                - 'seq': Full sequence
                - 'seq_chain_X': Sequence for chain X
                - 'coords_chain_X': Coordinates dictionary for chain X
            chain_id: Tuple of (fixed_chains, design_chains, undecoded_chains, all_chains)
                where each element is a list of chain letters
            ca_only: If True, only use CA (alpha carbon) coordinates
            chain_mask: Per-chain masks, one entry per chain in all_chains order.
                Each entry can be:
                - None: no additional masking for this chain
                - list[int]: positions within the chain to force mask=0
                - np.ndarray of shape [chain_length]: base mask (1=valid, 0=masked)

        Returns:
            Tuple containing:
                - X: Coordinate tensor of shape [1, L, 3] if ca_only else [1, L, 4, 3]
                - S: Sequence tensor of shape [1, L] with amino acid indices
                - mask: Binary mask tensor indicating valid positions [1, L]
                - chain_encodings: Chain ID encodings [1, L], starting from 1
                - residue_idx: Position-encoded residue indices [1, L]
        """
        fixed_chains, design_chains, undecoded_chains, all_chains = chain_id #Lists of chain letters
        n_chains = len(all_chains)
        n_atoms = 1 if ca_only else 4

        X_chains = []
        chain_seqs = []
        chain_start = 0

        chain_encodings_parts = []
        residue_idx_parts = []
        mask_parts = []

        for c_id, chain_letter in enumerate(all_chains):
            chain_seq = pdb["seq_chain_"+chain_letter]
            chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq]) #Replace - with X
            chain_length = len(chain_seq)
            chain_coords = pdb["coords_chain_"+chain_letter] #this is a dictionary

            if ca_only:
                X_chain = np.array(chain_coords["CA_chain_"+chain_letter]) #[chain_length,1,3]
                if len(X_chain.shape) == 2:
                    X_chain = X_chain[:, None, :]
            else:
                X_chain = np.stack([chain_coords[c] for c in ["N_chain_"+chain_letter, "CA_chain_"+chain_letter, "C_chain_"+chain_letter, "O_chain_"+chain_letter]], 1) #[chain_length,4,3]

            X_chains.append(X_chain)
            chain_seqs.append(chain_seq)

            # Build per-chain mask: start from coordinate validity, then apply user mask
            chain_mask_arr = np.isfinite(np.sum(X_chain, (1, 2))).astype(np.float32) # [chain_length]
            X_chain[np.isnan(X_chain)] = 0. # Replace NaN with 0 after mask computation

            # Apply per-chain user mask
            if chain_mask is not None and c_id < len(chain_mask) and chain_mask[c_id] is not None:
                cm = chain_mask[c_id]
                if isinstance(cm, np.ndarray):
                    # Base mask tensor: multiply with coordinate mask
                    chain_mask_arr *= cm.astype(np.float32)
                elif isinstance(cm, torch.Tensor):
                    chain_mask_arr *= cm.cpu().numpy().astype(np.float32)
                else:
                    # List of positions to zero out
                    for p in cm:
                        chain_mask_arr[p] = 0.0

            mask_parts.append(chain_mask_arr)

            chain_end = chain_start + chain_length
            chain_encodings_parts.append(np.full(chain_length, c_id, dtype=np.int32))
            residue_idx_parts.append(100 * c_id + np.arange(chain_start, chain_end, dtype=np.int32))
            chain_start = chain_end

        total_length = chain_start

        # Build final arrays
        if ca_only:
            X = np.zeros([1, total_length, 1, 3])
        else:
            X = np.zeros([1, total_length, 4, 3])

        X[0, :, :, :] = np.concatenate(X_chains, 0) #[L, n_atoms, 3]
        mask = np.concatenate(mask_parts)[None, :] #[1, L]

        X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
        mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)

        all_seq = "".join(chain_seqs)
        _nv = WrapperProteinMPNN._NATIVE_VOCAB
        S = np.asarray([_nv.get(aa, _nv['X']) for aa in all_seq], dtype=np.int32)
        S = torch.from_numpy(S).to(dtype=torch.long, device=device)

        chain_encodings = np.concatenate(chain_encodings_parts)[None, :]
        residue_idx = np.concatenate(residue_idx_parts)[None, :]
        chain_encodings = torch.from_numpy(chain_encodings).to(dtype=torch.long, device=device)
        residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)

        if ca_only:
            X = X[:,:,0]
        return X, S, mask, chain_encodings, residue_idx
    
    def _reset_decoding_order(self, decoding_order: torch.Tensor, keep_S: bool = True) -> None:
        """
        Sets up the order in which positions will be decoded during
        the design process, handling tied positions and fixed chains.
        
        Args:
            decoding_order: Tensor specifying the order of positions to decode
            keep_S: Whether to keep the current sequence tensor
        """
        self.decoding_order = decoding_order #This is relative to target chain's position in the OLG decoder. It is NOT the positions for X/S from PDB and need to be offseted
        self.end_pos = torch.max(self.decoding_order)
        self.decoding_order_e = self.decoding_order[self.decoding_order != self.end_pos].unsqueeze(0) if self.config.force_stop else decoding_order #Same, but excluding stop codon position
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

        # _get_decoding_mask requires a complete permutation of all model positions.
        # Prepend any uncovered positions — they are masked out so their ordering is irrelevant.
        model_size = self.X.shape[1]
        covered = self.decoding_order_S[0]
        all_pos = torch.arange(model_size, device=self.device)
        uncovered = all_pos[~torch.isin(all_pos, covered)]
        if uncovered.numel() > 0:
            self.decoding_order_S = torch.cat([uncovered.unsqueeze(0), self.decoding_order_S], dim=1)

        #Prepare encoder
        if keep_S:
            _, _, self.E_idx, self.h_E, self.h_EXV_encoder_fw, self.h_V_stack, self.mask_bw, self.h_EXV_encoder = self._prepare_encoder(self.decoding_order_S) #Initially, S and h_S are empty
        else:
            self.S, self.h_S, self.E_idx, self.h_E, self.h_EXV_encoder_fw, self.h_V_stack, self.mask_bw, self.h_EXV_encoder = self._prepare_encoder(self.decoding_order_S) #Initially, S and h_S are empty
            
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
        self._reset_decoding_order(decoding_order, keep_S=False)
        
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
        self.log_prob = torch.zeros((self.decoding_order_target.shape[1], self._NATIVE_VOCAB_SIZE), device=self.device) #This will keep track of log probs at each step (native space)
        self.argmax_aa = torch.zeros(self.decoding_order_target.shape[1], device=self.device).unsqueeze(0).long() #This will keep track of AAs that would have been the argmax
        
        self.preset_fixed_S(self.fixed_chain_seq) #This will update S, h_S and decoded_positions with fixed chains; but not individual fixed positions within design chains
        
        self.mask_eval = (~self.decoded_positions.bool()).to(torch.float32) #Mask for fixed chain / positions; used to get log probs only over the designed regions
        fixed_chain_res = torch.nonzero(self.fixed_positions != -1)
        if fixed_chain_res.shape[0] > 0:
            self.mask_eval[0, fixed_chain_res] = 0.0

    def _get_decoding_mask(
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
        
    def _prepare_encoder(
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
        mask_fw, mask_bw = self._get_decoding_mask(E_idx, decoding_order)

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
            mask_fw, mask_bw = self._get_decoding_mask(self.E_idx, new_decoding_order)
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
            t = use_t # Decoding position, relative to target protein
            if not (self.config.force_stop and (t == self.end_pos)):
                if not self.tied:
                    t_list = [ t ]    
                else:
                    next_t = (self.decoding_order_e[0]==use_t).nonzero().item()
                    t_list = self.decoding_order_target[0, (next_t*self.n_design_chains):((next_t+1)*self.n_design_chains)]
                
        else:
            t = self.decoding_order[0, self.next_t_full] #Decoding position, relative to target protein
            # Check force_stop before accessing decoding_order_target, since the
            # stop position may not have an entry in decoding_order_target.
            if self.config.force_stop and (t == self.end_pos):
                logits = self._force_stop()
                return logits, logits
            if not self.tied:
                t_list = [ self.decoding_order_target[0, self.next_t] ]
            else:
                t_list = self.decoding_order_target[0, (self.next_t*self.n_design_chains):((self.next_t+1)*self.n_design_chains)]

        if self.config.force_stop and (t == self.end_pos):
            logits = self._force_stop()
            return logits, logits

        if dummy_run: #All zero if dummy running
            self.current_logits = torch.zeros((1, self._NATIVE_VOCAB_SIZE), device=self.device)
            logits_ = self.current_logits[:, self.alphabet_map]  # remap to OLG-internal
            logits = logits_.clone()

        else:
            native_logits = 0.0
            #Decoding position, relative to X/S
            for t_m in t_list:
                native_logits += self.get_logits(t_m, mask_current=mask_current)
            self.current_logits = native_logits.clone()  # kept in native space for log_prob

            # Remap to OLG-internal space for all downstream constraint logic
            logits_ = self.current_logits[:, self.alphabet_map]
            logits_ -= logits_.mean()
            logits = logits_.clone()

            #Repeat penalty (uses model-global t to index into model-global S / decoded_positions)
            logits = self._apply_repetition_penalty(logits, t)

            # Per-position weight/bias tensors are 0-indexed from protein start; subtract offset.
            t_local = t - self.config.start_offset
            logits = self._apply_weights_and_biases(logits, t_local)

            count_pos = torch.zeros(self.decoded_positions.shape, device=self.device)
            count_pos[:, self.target_chain_offset:(self.target_chain_offset+self.target_chain_length)] = 1
            count_pos = (self.decoded_positions * count_pos) == 1

            # aa_count in OLG-internal space (S stores native tokens, new-letter slots are zero)
            aa_count = torch.nn.functional.one_hot(self.S[:,count_pos[0]], num_classes=self.alphabet_size).sum(1)
            max_aa = (aa_count >= self.config.max_aa_count)
            logits[max_aa] = Constants.MIN_LOGIT

            #Positive AA total counts (H/K/R)
            if self.pos_charged_indices and (
                sum(aa_count[0, i] for i in self.pos_charged_indices) >= self.config.max_pos_count
            ):
                for i in self.pos_charged_indices:
                    logits[0, i] = Constants.MIN_LOGIT

            logits = BaseWrapper._top_p(logits, self.config.truncate_topp) #Top-p filtering
        
        if (not self.config.force_stop) or (t != self.end_pos): #Penalize stop codon if not at last position
            logits_ = self._penalize_stop(logits_)
            logits = self._penalize_stop(logits)
            
        if self.fixed_positions[t] != -1:
            logits = self._force_fixed_positions(logits, t)
            
        logits = BaseWrapper._add_noise(logits)
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
                seq_token = torch.tensor([ self._NATIVE_VOCAB.get(c, self._NATIVE_VOCAB['X']) for c in seq ], device=self.device)
                start = self.chain_offsets[self.all_chains.index(chain_letter)]
                end = start + len(seq)
                self.S[:, start:end] = seq_token
                all_fixed_positions += [ pos for pos in range(start, end) ]
        else: #if fixed chain seqs are not provided, then take from PDB
            if self.fixed_chains is not None:
                for chain_letter in self.fixed_chains:
                    seq = self.pdb_data["seq_chain_"+chain_letter]
                    seq_token = torch.tensor([ self._NATIVE_VOCAB.get(c, self._NATIVE_VOCAB['X']) for c in seq ], device=self.device)
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
            if self.config.force_stop and (self.end_pos == t_full):
                self.next_t_full += 1
                return #Do nothing if stop codon; don't advance next_t
            t = self.decoding_order_target[0, self.next_t]
            t_list = self.decoding_order_target[0, ((self.next_t)*self.n_design_chains):((self.next_t+1)*self.n_design_chains)]
            self.next_t_full += 1
            self.next_t += 1
        else:
            t = use_t
        
        # S_t is OLG-internal; convert to native for model embedding and log_prob indexing
        native_S_t = self.alphabet_map[S_t]

        if not self.tied:
            self.edit_S(t, native_S_t, inplace=True)
            self.decoded_positions[0, t] = 1.0
        else:
            for t in t_list:
                self.edit_S(t, native_S_t, inplace=True)
                self.decoded_positions[0, t] = 1.0

        # Per-position arrays (selected_aa, log_prob, etc.) are 0-indexed from protein start.
        t_local = t - self.target_chain_offset - self.config.start_offset
        self.selected_aa[:, t_local] = S_t  # store OLG-internal token
        log_softmax = torch.log(torch.nn.functional.softmax(self.current_logits[0], dim=-1))  # native space
        self.selected_log_prob[:, t_local] = log_softmax[native_S_t]
        self.log_prob[t_local, :] = log_softmax
        self.argmax_aa[:, t_local] = self.current_logits[0].argmax()
        
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
        # S stores native tokens; convert to OLG-internal via alphabet_map_rev, then to chars
        prot = ''.join([self.alphabet[self.alphabet_map_rev[s.item()].item()] for s in S[0, :]])
        return prot

    def get_tied_positions(self) -> list[int]:
        """All tied positions for current step (multimer symmetry)."""
        t_ = self.decoding_order_target[0, self.next_t * self.n_design_chains]
        if self.tied:
            return self.tied_pos[t_].tolist()
        return [t_.item()]

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
                S_t = torch.multinomial(probs[0], 1)  # OLG-internal
            else:
                t = self.decoding_order[:, i]
                if not (self.config.force_stop and (t == self.end_pos)):
                    # use_S stores native tokens; convert to OLG-internal for update_S
                    S_t = self.alphabet_map_rev[use_S[t]]
                else:
                    S_t = None
            self.update_S(S_t, alphabet_map=False)
        return True