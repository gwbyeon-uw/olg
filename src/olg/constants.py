from typing import Dict, List
from enum import IntEnum
import itertools
import torch

class Arrangement(IntEnum):
    """Enumeration for overlap arrangements between frames"""
    PLUS_ONE = 0   # +1 frame
    MINUS_ONE = 1  # -1 frame  
    MINUS_ZERO = 2 # -0 frame
    PLUS_TWO = 3   # +2 frame
    MINUS_TWO = 4  # -2 frame

class DecodingMode(IntEnum):
    """Enumeration for decoding strategies"""
    RANDOM = 0
    OVERLAP_FIRST = 1
    OVERLAP_LAST = 2
    
class Constants:
    EPS = 1e-8

    # (f1_offset, f2_offset, reverse)
    ARRANGEMENT_CONFIG = {
        0: (0, 1, False),
        3: (1, 0, False),
        1: (0, 1, True),
        4: (1, 0, True),
        2: (0, 0, True),
    }
    
    #High/low values for clamping and strong penalty (float16-safe: max ~65504)
    MAX_LOGIT = 1e4
    MIN_LOGIT = -1e4
    
    #Alphabets and quartets that we use
    NUCLEOTIDES = list('ATGC')
    ALPHABET = list('ACDEFGHIKLMNPQRSTVWYX')
    ALPHABET_SIZE = len(ALPHABET)
    ALPHABET_GAP = list('ACDEFGHIKLMNPQRSTVWY-') #21 with gap
    NUCLEOTIDE_SIZE = len(NUCLEOTIDES)
    
    QUARTETS = [ ''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=4) ] #All possible combination of 4 nucleotides
    CODONS = [ ''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=3) ] #All possible combination of 3 nucleotides
    QUARTET_SIZE = len(QUARTETS)
    
    ALPHABET_INDEX = { alphabet: index for index, alphabet in enumerate(ALPHABET) }
    CODON_INDEX = { codon: index for index, codon in enumerate(CODONS) }
    NUCLEOTIDE_INDEX = { nucleotide: index for index, nucleotide in enumerate(NUCLEOTIDES) }
    
    STOP_INDEX = ALPHABET.index('X') #X will be used as stop codon 
    GAP_TOKEN = 26 #As in EvoDiff alphabet
    GAP_TOKEN_GREMLIN = 20 #As in EvoDiff alphabet
    
    REVERSE_COMPLEMENT = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G'
    }
    
    @staticmethod
    def _reverse_complement(seq):
        return ''.join(Constants.REVERSE_COMPLEMENT[n] for n in reversed(seq))
    
    #Indexing these lists with quartet id (q_i) gives first and fourth nucleotide of each quartet
    PREV_QUARTET_INDEX = [0] * QUARTET_SIZE
    NEXT_QUARTET_INDEX = [0] * QUARTET_SIZE
    for q_i in range(QUARTET_SIZE):
        q = QUARTETS[q_i]
        q1 = NUCLEOTIDES.index(q[0])
        q4 = NUCLEOTIDES.index(q[3])
        PREV_QUARTET_INDEX[q_i] = q4
        NEXT_QUARTET_INDEX[q_i] = q1
    
    #List, where index is a quartet id (q_i) and value is the list of compatible quartets that would be compatible with it in the previous or next position
    QUARTETS_P = [ [] for i in range(QUARTET_SIZE) ]
    QUARTETS_N = [ [] for i in range(QUARTET_SIZE) ]
    
    #List, given nucleotides in first and last position, compatible quartets
    P_QUARTETS = [ [] for i in range(NUCLEOTIDE_SIZE) ]
    N_QUARTETS = [ [] for i in range(NUCLEOTIDE_SIZE) ]
    
    for q_i in range(QUARTET_SIZE):
        q = QUARTETS[q_i]
        q1_i = NUCLEOTIDES.index(q[0])
        q4_i = NUCLEOTIDES.index(q[3])
        
        #Compatible next quartet
        for q_j in range(QUARTET_SIZE):
            q = QUARTETS[q_j]
            q1_j = NUCLEOTIDES.index(q[0])
            if q1_j == q4_i:
                QUARTETS_N[q_i] += [ q_j ]
        
        #Compatible previous quartet
        for q_k in range(QUARTET_SIZE):
            q = QUARTETS[q_k]
            q4_k = NUCLEOTIDES.index(q[3])
            if q4_k == q1_i:
                QUARTETS_P[q_i] += [ q_k ]
        
        P_QUARTETS[q1_i] += [ q_i ]
        N_QUARTETS[q4_i] += [ q_i ]
    
    #List of which frames of each arrangement encodes protein 1 or 2; index is arrangement; values are 0=ref, 1=alt, 2=alt_neg, 3=neg
    FRAME_F1 = [ 0, 0, 0, 1, 1]
    FRAME_F2 = [ 1, 2, 3, 0, 3 ]

    #List mapping codons to which quartets; index is [codon_index][frame_index]; values are list of quartet indices
    @staticmethod
    def _build_codons_to_quartets(reverse_complement: Dict[str, str], quartets: List[str], codons: List[str]):
        result = [[[] for j in range(4)] for i in range(len(codons))]
        for q_i, q in enumerate(quartets):
            q_ref = q[:3]
            q_alt = q[1:]
            q_rc = ''.join(reverse_complement[n] for n in reversed(q))
            q_alt_neg = q_rc[:3]
            q_neg = q_rc[1:]
            
            result[codons.index(q_ref)][0] += [q_i]
            result[codons.index(q_alt)][1] += [q_i]
            result[codons.index(q_alt_neg)][2] += [q_i]
            result[codons.index(q_neg)][3] += [q_i]
        return result
    
    CODONS_TO_QUARTETS = _build_codons_to_quartets(REVERSE_COMPLEMENT, QUARTETS, CODONS)

    #Standard codons dict; { codon: AA }
    STANDARD_CODONS = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "TAT": "Y", "TAC": "Y", "TAA": "X", "TAG": "X",
            "TGT": "C", "TGC": "C", "TGA": "X", "TGG": "W",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        }

    
    #Reverse dict of standard codon table; { AA: codon }
    @staticmethod
    def _reverse_codon_table(
        codon_table: Dict[str, str]
    ) -> Dict[str, str]:
        codon_table_rev = { a:[] for c, a in codon_table.items() }    
        for c, a in codon_table.items():
            codon_table_rev[a] += [ c ]
        return codon_table_rev
    
    STANDARD_CODONS_REV = _reverse_codon_table(STANDARD_CODONS)
    
    EVODIFF_ALPHABET = { c: i for i, c in enumerate(list('ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!')) }
    EVODIFF_ALPHABET_INDEX = { u: i for i, u in enumerate(EVODIFF_ALPHABET) }

    ESM_MSA_ALPHABET = list('@!*uLAGVSERTIDPKQNFYMHWCXBUZO.-n#') # @: start token <cls>, !: pad token <pad>, *: stop token <eos>, -: gap token, #: mask token <mask>
    ESM_MSA_ALPHABET[3] = '<unk>'
    ESM_MSA_ALPHABET[31] = '<null_1>'
    ESM_MSA_ALPHABET = { c: i for i, c in enumerate(ESM_MSA_ALPHABET) }
    
    REMAP_TO_ESM_MSA = torch.zeros(len(EVODIFF_ALPHABET)).long() - 1
    for k, v in EVODIFF_ALPHABET.items():
        if k in ESM_MSA_ALPHABET.keys():
            REMAP_TO_ESM_MSA[v] = ESM_MSA_ALPHABET[k]
    
    REMAP_TO_EVODIFF = torch.zeros(len(ESM_MSA_ALPHABET)).long() - 1
    for k, v in ESM_MSA_ALPHABET.items():
        if k in EVODIFF_ALPHABET.keys():
            REMAP_TO_EVODIFF[v] = EVODIFF_ALPHABET[k]
    
    GREMLIN_ALPHABET_ = list("ARNDCQEGHILKMFPSTWYV-")
    GREMLIN_ALPHABET = dict(zip(GREMLIN_ALPHABET_, range(len(GREMLIN_ALPHABET_))))