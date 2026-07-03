import numpy as np
import torch

from typing import Dict, Tuple, List, Optional

from Bio.Data.CodonTable import unambiguous_dna_by_name

from olg.constants import *
from olg.config import DesignConfig

# (codon_table, alphabet, device) -> (codon_compatibility, quartets_aa). The matrix is design-independent
# and never mutated in place (decode clones before writing), so one copy is shared across all designs.
_COMPAT_CACHE: dict = {}


class CodonCompatibility:
    """Manages codon compatibility matrices and quartet operations"""
    def __init__(
        self, 
        config: DesignConfig
    ):
        self.config = config
        self.alphabet = config.alphabet
        self.alphabet_index = config.alphabet_index
        self.stop_index = config.stop_index
        self.codon_table = self.config.codon_table
        if isinstance(self.codon_table, dict):
            pass  # already a dict, use as-is
        else:
            ncbi_table = unambiguous_dna_by_name[self.codon_table] #From Biopython NCBI codes
            # copy forward_table (it excludes stops) so we don't mutate Biopython's global,
            # and use THIS table's own stop codons (Standard's stops are wrong for alt codes,
            # e.g. mito TGA->W, and miss alt-only stops like AGA/AGG -> KeyError later)
            self.codon_table = dict(ncbi_table.forward_table)
            for stop_codon in ncbi_table.stop_codons:
                self.codon_table[stop_codon] = "X"
        self.codon_table_rev = Constants._reverse_codon_table(self.codon_table)
        key = (frozenset(self.codon_table.items()), tuple(self.alphabet), str(self.config.device))
        if key not in _COMPAT_CACHE:
            _COMPAT_CACHE[key] = self.generate_compatibility_matrix(
                self.config.device, self.codon_table, self.alphabet, self.alphabet_index
            )
        self.codon_compatibility, self.quartets_aa = _COMPAT_CACHE[key]
        
        #First/last nucleotide for each of the 256 quartets
        self.prev_quartet_index = torch.tensor(Constants.PREV_QUARTET_INDEX).long()
        self.next_quartet_index = torch.tensor(Constants.NEXT_QUARTET_INDEX).long()
        
        #Dictionary that stores compatible next/previous quartets given the current quartet
        self.compatible_prev_quartets = [ torch.tensor(v).unique() for v in Constants.QUARTETS_P ]
        self.compatible_next_quartets = [ torch.tensor(v).unique() for v in Constants.QUARTETS_N ]

        self.codon_compatibility_start_mask = [None, None]
        self.start_codons_quartets = [[], []]
        if self.config.protein1.force_start:
            # Allocate the mask ONCE; OR in the quartets of every start codon (not just the last)
            self.codon_compatibility_start_mask[0] = torch.zeros(self.codon_compatibility.shape, device=self.config.device).int()
            for codon in self.config.protein1.start_codons:
                for q_i in Constants.CODONS_TO_QUARTETS[Constants.CODONS.index(codon)][Constants.FRAME_F1[self.config.arrangement]]:
                    self.codon_compatibility_start_mask[0][:, :, :, :, :, q_i] = 1
                    self.start_codons_quartets[0] += [ q_i ]
        if self.config.protein2.force_start:
            self.codon_compatibility_start_mask[1] = torch.zeros(self.codon_compatibility.shape, device=self.config.device).int()
            for codon in self.config.protein2.start_codons:
                for q_i in Constants.CODONS_TO_QUARTETS[Constants.CODONS.index(codon)][Constants.FRAME_F2[self.config.arrangement]]:
                    self.codon_compatibility_start_mask[1][:, :, :, :, :, q_i] = 1
                    self.start_codons_quartets[1] += [ q_i ]
                
        self.codon_to_aa = torch.zeros((Constants.NUCLEOTIDE_SIZE, Constants.NUCLEOTIDE_SIZE, Constants.NUCLEOTIDE_SIZE), device=self.config.device).long()
        self.codon_to_aa_rc = torch.zeros((Constants.NUCLEOTIDE_SIZE, Constants.NUCLEOTIDE_SIZE, Constants.NUCLEOTIDE_SIZE), device=self.config.device).long()
        for i in range(Constants.NUCLEOTIDE_SIZE):
            for j in range(Constants.NUCLEOTIDE_SIZE):
                for k in range(Constants.NUCLEOTIDE_SIZE):
                    codon = Constants.NUCLEOTIDES[i] + Constants.NUCLEOTIDES[j] + Constants.NUCLEOTIDES[k]
                    codon_rc = Constants._reverse_complement(codon)
                    aa = self.codon_table[codon]
                    aa_rc = self.codon_table[codon_rc]
                    self.codon_to_aa[i, j, k] = self.alphabet_index[aa]
                    self.codon_to_aa_rc[i, j, k] = self.alphabet_index[aa_rc]

    @staticmethod
    def generate_compatibility_matrix(
        device: torch.device,
        codon_table: Dict[str, str],
        alphabet: List[str] = None,
        alphabet_index: Dict[str, int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int]]]:
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
        Tuple[torch.Tensor, List[Tuple[int]]
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
        
        # Fall back to global constants for callers that don't supply alphabet
        if alphabet is None:
            alphabet = Constants.DEFAULT_ALPHABET
        if alphabet_index is None:
            alphabet_index = Constants.DEFAULT_ALPHABET_INDEX

        #Dim 0: first nucleotide (4); ATGC
        #Dim 1: last nucleotide (4)
        #Dim 2: arrangement
        #Dim 3: amino acid 1; len(alphabet) entries, last letter (X) is stop codon
        #Dim 4: amino acid 2
        #Dim 5: quartet index (256)
        codon_compatibility = torch.zeros((4, 4, 6, len(alphabet), len(alphabet), len(Constants.QUARTETS)), device=device, dtype=torch.int)
        quartets_aa = [ None ] * len(Constants.QUARTETS) #Quartet index to amino acids in each alt frames

        for q_i, q in enumerate(Constants.QUARTETS):
            q1 = Constants.NUCLEOTIDE_INDEX[q[0]]
            q4 = Constants.NUCLEOTIDE_INDEX[q[3]]

            #reference frame
            q_ref = q[:3]
            aa_ref = codon_table[q_ref]
            aa_ref = 'X' if aa_ref == '*' else aa_ref
            i_ref = alphabet_index[aa_ref]

            #shifted
            q_alt = q[1:]
            aa_alt = codon_table[q_alt]
            aa_alt = 'X' if aa_alt == '*' else aa_alt
            i_alt = alphabet_index[aa_alt]

            #reverse complement of shifted
            aa_alt_neg = codon_table[Constants._reverse_complement(q_alt)]
            aa_alt_neg = 'X' if aa_alt_neg == '*' else aa_alt_neg
            i_alt_neg = alphabet_index[aa_alt_neg]

            #reverse complement of reference
            aa_neg = codon_table[Constants._reverse_complement(q_ref)]
            aa_neg = 'X' if aa_neg == '*' else aa_neg
            i_neg = alphabet_index[aa_neg]
            
            codon_compatibility[q1, q4, 0, i_ref, i_alt, q_i] = 1
            codon_compatibility[q1, q4, 1, i_ref, i_alt_neg, q_i] = 1
            codon_compatibility[q1, q4, 2, i_ref, i_neg, q_i] = 1
            codon_compatibility[q1, q4, 3, i_alt, i_ref, q_i] = 1
            codon_compatibility[q1, q4, 4, i_alt, i_neg, q_i] = 1
            
            quartets_aa[q_i] = ( i_ref, i_alt, i_alt_neg, i_neg )
    
        return codon_compatibility, quartets_aa

    @staticmethod
    def compatible_quartets_by_aa(
        arrangement: int,
        aa1s: Tuple[Optional[str]],
        aa2s: Tuple[Optional[str]],
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
        common_indices = set(range(len(Constants.QUARTETS)))
        
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
                update_common_indices(aa1_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][0]}),
                update_common_indices(aa1_n, lambda codons: {x for c in codons for x in Constants.N_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[0]]]}),
                update_common_indices(aa2_p, lambda codons: {x for c in codons for x in Constants.P_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[-1]]]}),
                update_common_indices(aa2_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][1]})
            ),
            
            1: lambda: (
                update_common_indices(aa1_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][0]}),
                update_common_indices(aa1_n, lambda codons: {x for c in codons for x in Constants.N_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[0]]]}),
                update_common_indices(aa2_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][2]}),
                update_common_indices(aa2_n, lambda codons: {x for c in codons for x in Constants.P_QUARTETS[Constants.NUCLEOTIDE_INDEX[Constants._reverse_complement(c[0])]]})
            ),
            
            2: lambda: (
                update_common_indices(aa1_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][0]}),
                update_common_indices(aa1_n, lambda codons: {x for c in codons for x in Constants.N_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[0]]]}),
                update_common_indices(aa2_p, lambda codons: {x for c in codons for x in Constants.N_QUARTETS[Constants.NUCLEOTIDE_INDEX[Constants._reverse_complement(c[-1])]]}),
                update_common_indices(aa2_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][3]})
            ),
            
            3: lambda: (
                update_common_indices(aa1_p, lambda codons: {x for c in codons for x in Constants.P_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[-1]]]}),
                update_common_indices(aa1_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][1]}),
                update_common_indices(aa2_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][0]}),
                update_common_indices(aa2_n, lambda codons: {x for c in codons for x in Constants.N_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[0]]]})
            ),
            
            4: lambda: (
                update_common_indices(aa1_p, lambda codons: {x for c in codons for x in Constants.P_QUARTETS[Constants.NUCLEOTIDE_INDEX[c[-1]]]}),
                update_common_indices(aa1_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][1]}),
                update_common_indices(aa2_p, lambda codons: {x for c in codons for x in Constants.N_QUARTETS[Constants.NUCLEOTIDE_INDEX[Constants._reverse_complement(c[-1])]]}),
                update_common_indices(aa2_c, lambda codons: {x for c in codons for x in Constants.CODONS_TO_QUARTETS[Constants.CODON_INDEX[c]][3]})
            )
        }
        
        # Call the appropriate handler for the given arrangement
        if arrangement in arrangement_handlers:
            arrangement_handlers[arrangement]()
        
        # Convert the final result to numpy array
        return np.array(list(common_indices))