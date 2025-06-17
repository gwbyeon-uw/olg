import string
from typing import Dict, Tuple, List, Set, Optional, Union, Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import torch

from constants import *

from Bio.Data.CodonTable import standard_dna_table, register_ncbi_table, unambiguous_dna_by_name

def reverse_complement(sequence):
    return ''.join(REVERSE_COMPLEMENT[nucleotide] for nucleotide in reversed(sequence))

def compatible_quartets_by_aa(
    arrangement: int,
    aa1s: Tuple[Optional[str], Optional[str], Optional[str]],
    aa2s: Tuple[Optional[str], Optional[str], Optional[str]],
    codon_table_rev: Dict[str, List[str]]
) -> np.ndarray:
    """
    Find compatible quartet indices based on amino acid constraints across different reading frames.
    
    Parameters:
    -----------
    arrangement : int
        Integer representing the arrangement type (0-4), where each value represents a different
        pattern of overlap between reading frames
        
    aa1s/aa2s : Tuple[Optional[str], Optional[str], Optional[str]]
        Tuple containing amino acid constraints for the first/second frame, in the form (aa1_p, aa1_c, aa1_n):
        - aa1_p: Amino acid constraint for previous position
        - aa1_c: Amino acid constraint for current position
        - aa1_n: Amino acid constraint for next position
        Any of these can be None if no constraint exists for that position.
    
    codon_table_rev : Dict[str, List[str]]
        A reverse codon table mapping amino acids to their possible codons.
        For example: {'M': ['ATG'], 'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'], ...}
    
    Returns:
    --------
    np.ndarray
        A numpy array containing the indices of all compatible quartets that satisfy
        the given amino acid constraints in the specified arrangement.
    """

    aa1_p, aa1_c, aa1_n = aa1s
    aa2_p, aa2_c, aa2_n = aa2s
    
    # Use sets for faster intersection operations
    common_indices = set(range(len(QUARTETS)))
    
    # Helper function to update common indices with new constraints
    def update_common_indices(aa, lookup_func):
        nonlocal common_indices
        if aa is None:
            return
            
        # Get codons for this amino acid
        codons = codon_table_rev[aa]
        
        # Get new indices based on lookup function
        new_indices = set(lookup_func(codons))
        
        # Update common indices
        common_indices &= new_indices
        
    # Define handler functions for each arrangement pattern
    # These functions encapsulate the logic for each arrangement
    arrangement_handlers = {
        0: lambda: (
            update_common_indices(aa1_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][0]}),
            update_common_indices(aa1_n, lambda codons: {x for c in codons for x in N_QUARTETS[NUCLEOTIDE_INDEX[c[0]]]}),
            update_common_indices(aa2_p, lambda codons: {x for c in codons for x in P_QUARTETS[NUCLEOTIDE_INDEX[c[-1]]]}),
            update_common_indices(aa2_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][1]})
        ),
        
        1: lambda: (
            update_common_indices(aa1_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][0]}),
            update_common_indices(aa1_n, lambda codons: {x for c in codons for x in N_QUARTETS[NUCLEOTIDE_INDEX[c[0]]]}),
            update_common_indices(aa2_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][2]}),
            update_common_indices(aa2_n, lambda codons: {x for c in codons for x in P_QUARTETS[NUCLEOTIDE_INDEX[reverse_complement(c[0])]]})
        ),
        
        2: lambda: (
            update_common_indices(aa1_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][0]}),
            update_common_indices(aa1_n, lambda codons: {x for c in codons for x in N_QUARTETS[NUCLEOTIDE_INDEX[c[0]]]}),
            update_common_indices(aa2_p, lambda codons: {x for c in codons for x in N_QUARTETS[NUCLEOTIDE_INDEX[reverse_complement(c[-1])]]}),
            update_common_indices(aa2_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][3]})
        ),
        
        3: lambda: (
            update_common_indices(aa1_p, lambda codons: {x for c in codons for x in P_QUARTETS[NUCLEOTIDE_INDEX[c[-1]]]}),
            update_common_indices(aa1_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][1]}),
            update_common_indices(aa2_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][0]}),
            update_common_indices(aa2_n, lambda codons: {x for c in codons for x in N_QUARTETS[NUCLEOTIDE_INDEX[c[0]]]})
        ),
        
        4: lambda: (
            update_common_indices(aa1_p, lambda codons: {x for c in codons for x in P_QUARTETS[NUCLEOTIDE_INDEX[c[-1]]]}),
            update_common_indices(aa1_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][1]}),
            update_common_indices(aa2_p, lambda codons: {x for c in codons for x in N_QUARTETS[NUCLEOTIDE_INDEX[reverse_complement(c[-1])]]}),
            update_common_indices(aa2_c, lambda codons: {x for c in codons for x in CODONS_TO_QUARTETS[CODON_INDEX[c]][3]})
        )
    }
    
    # Call the appropriate handler for the given arrangement
    if arrangement in arrangement_handlers:
        arrangement_handlers[arrangement]()
    
    # Convert the final result to numpy array
    return np.array(list(common_indices))

def generate_compatibility_matrix(
    device: torch.device,
    codon_table: Dict[str, str]
) -> Tuple[torch.Tensor, List[Optional[Tuple[int, int, int, int]]]]:
    """
    Generate a compatibility matrix for codon pairings across multiple reading frames.
    
    This function creates a tensor representing valid amino acid combinations between
    codons in different reading frames. The resulting compatibility matrix can be used 
    for masking joint distributions to analyze overlapping coding sequences.
    
    Parameters:
    -----------
    device : torch.device    
    codon_table : Union[Dict[str, str], str]
        A dictionary mapping codons to amino acids (e.g., {"ATG": "M", "TAA": "*"})
        
    Returns:
    --------
    Tuple[torch.Tensor, List[Optional[Tuple[int, int, int, int]]]]
        A tuple containing:
        - codon_compatibility: A 6D tensor with shape (4, 4, 6, len(ALPHABET), len(ALPHABET), len(QUARTETS))
          where dimensions represent:
            - Dim 0: First nucleotide (4 options: A, T, G, C)
            - Dim 1: Last nucleotide (4 options: A, T, G, C)
            - Dim 2: Frame arrangement (6 different arrangements)
            - Dim 3: Amino acid 1 index (from ALPHABET, including 'X' for stop codons)
            - Dim 4: Amino acid 2 index (from ALPHABET, including 'X' for stop codons)
            - Dim 5: Quartet index (256 possible nucleotide quartets)
          Each element is binary (0 or 1), indicating whether the amino acid pair is compatible
          for the given quartet and frame arrangement.
        
        - quartets_aa: A list mapping quartet indices to tuples of amino acid indices
          in each alternate reading frame (reference frame aa, alt frame aa, 
          alt frame reverse complement aa, reference frame reverse complement aa)
    """
    
    #Dim 0: first nucleotide (4); ATGC
    #Dim 1: last nucleotide (4)
    #Dim 2: arrangement
    #Dim 3: amino acid 1 (21); 21st letter (X) is used as stop codon
    #Dim 4: amino acid 2 (21)
    #Dim 5: quartet index (256)
    codon_compatibility = torch.zeros((4, 4, 6, len(ALPHABET), len(ALPHABET), len(QUARTETS)), device=device).int()
    quartets_aa = [None] * len(QUARTETS) #Quartet index to amino acids in each alt frames
    
    for q_i in range(len(QUARTETS)):
        q = QUARTETS[q_i]
        q1 = NUCLEOTIDE_INDEX[q[0]]
        q4 = NUCLEOTIDE_INDEX[q[3]]

        #reference frame
        q_ref = q[:3]
        aa_ref = codon_table[q_ref]
        aa_ref = 'X' if aa_ref == '*' else aa_ref
        i_ref = ALPHABET_INDEX[aa_ref]

        #shifted
        q_alt = q[1:]
        aa_alt = codon_table[q_alt]
        aa_alt = 'X' if aa_alt == '*' else aa_alt
        i_alt = ALPHABET_INDEX[aa_alt]

        #reverse complement of shifted
        aa_alt_neg = codon_table[reverse_complement(q_alt)]
        aa_alt_neg = 'X' if aa_alt_neg == '*' else aa_alt_neg
        i_alt_neg = ALPHABET_INDEX[aa_alt_neg]

        #reverse complement of reference
        aa_neg = codon_table[reverse_complement(q_ref)]
        aa_neg = 'X' if aa_neg == '*' else aa_neg
        i_neg = ALPHABET_INDEX[aa_neg]
        
        codon_compatibility[q1, q4, 0, i_ref, i_alt, q_i] = 1
        codon_compatibility[q1, q4, 1, i_ref, i_alt_neg, q_i] = 1
        codon_compatibility[q1, q4, 2, i_ref, i_neg, q_i] = 1
        codon_compatibility[q1, q4, 3, i_alt, i_ref, q_i] = 1
        codon_compatibility[q1, q4, 4, i_alt, i_neg, q_i] = 1
        
        quartets_aa[q_i] = ( i_ref, i_alt, i_alt_neg, i_neg )

    return codon_compatibility, quartets_aa

#Top p thresholding given logit vector
def top_p(
    logits: torch.Tensor, 
    thres: float = 0.1, 
    removal_value: float = -1e3
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

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
def add_noise(
    tensor: torch.Tensor, 
    factor: float = 1e-6
) -> torch.Tensor:
    noised_tensor = tensor + torch.rand(tensor.shape, device=tensor.device) * factor
    return noised_tensor

#function to parse fasta
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

def load_a3m(
    path_to_msa: str, 
    device: str, 
    gap_cutoff_v: float = 0.5, 
    gap_cutoff_h: float = 0.25
) -> List[str]:
    names, seqs = parse_fasta(path_to_msa)
    seqs = seqs.view('S4').reshape((seqs.size, -1)).astype('U1')
    seqs = seqs[((seqs=='-').sum(-1)/seqs.shape[1]) < gap_cutoff_h]
    seqs = seqs[:,((seqs=='-').sum(0)/seqs.shape[0]) < gap_cutoff_v]
    seqs = [ ''.join(s) for s in seqs ]
    return seqs

def tokenizeMSA(
    seq: Union[str, List[str]]
) -> npt.NDArray[np.int_]:
    return np.array([EVODIFF_ALPHABET_INDEX[a] for a in seq])
    
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
    alphabet = EVODIFF_ALPHABET #EvoDiff alphabet
    alpha = np.array(list(alphabet))
    gap_idx = EVODIFF_ALPHABET_INDEX['-']
    pad_idx = EVODIFF_ALPHABET_INDEX['!']
    
    #Do hamming distance from aligned section
    aligned_msa = [ [ char for char in seq if (char.isupper() or char == '-') and not char == '.' ] for seq in parsed_msa ]   

    tokenized_msa = [ tokenizeMSA(seq) for seq in aligned_msa ]
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