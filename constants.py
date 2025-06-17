import itertools
import torch

#High/low values for clamping and strong penalty
MAX_LOGIT = 1e3
MIN_LOGIT = -1e3

#Alphabets and quartets that we use
ALPHABET = list('ACDEFGHIKLMNPQRSTVWYX')
ALPHABET_DICT = { ALPHABET[i]: i for i in range(len(ALPHABET)) }
STOP_INDEX = ALPHABET.index('X') #X will be used as stop codon 
NUCLEOTIDES = list('ATGC')
QUARTETS = [ ''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=4) ] #All possible combination of 4 nucleotides
CODONS = [ ''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=3) ] #All possible combination of 3 nucleotides

ALPHABET_INDEX = { alphabet: index for index, alphabet in enumerate(ALPHABET) }
CODON_INDEX = { codon: index for index, codon in enumerate(CODONS) }
NUCLEOTIDE_INDEX = { nucleotide: index for index, nucleotide in enumerate(NUCLEOTIDES) }

REVERSE_COMPLEMENT = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
}

ALPHABET_GAP = list('ACDEFGHIKLMNPQRSTVWY-') #21 with gap

GAP_TOKEN = 26 #As in EvoDiff alphabet
GAP_TOKEN_GREMLIN = 20 #As in EvoDiff alphabet

#Indexing these lists with quartet id (q_i) gives first and fourth nucleotide of each quartet
PREV_QUARTET_INDEX = [0] * len(QUARTETS)
NEXT_QUARTET_INDEX = [0] * len(QUARTETS)
for q_i in range(len(QUARTETS)):
    q = QUARTETS[q_i]
    q1 = NUCLEOTIDES.index(q[0])
    q4 = NUCLEOTIDES.index(q[3])
    PREV_QUARTET_INDEX[q_i] = q4
    NEXT_QUARTET_INDEX[q_i] = q1

#List, where index is a quartet id (q_i) and value is the list of compatible quartets that would be compatible with it in the previous or next position
QUARTETS_P = [ [] for i in range(len(QUARTETS)) ]
QUARTETS_N = [ [] for i in range(len(QUARTETS)) ]

#List, given nucleotides in first and last position, compatible quartets
P_QUARTETS = [ [] for i in range(len(NUCLEOTIDES)) ]
N_QUARTETS = [ [] for i in range(len(NUCLEOTIDES)) ]

for q_i in range(len(QUARTETS)):
    q = QUARTETS[q_i]
    q1_i = NUCLEOTIDES.index(q[0])
    q4_i = NUCLEOTIDES.index(q[3])
    
    #Compatible next quartet
    for q_j in range(len(QUARTETS)):
        q = QUARTETS[q_j]
        q1_j = NUCLEOTIDES.index(q[0])
        if q1_j == q4_i:
            QUARTETS_N[q_i] += [ q_j ]
    
    #Compatible previous quartet
    for q_k in range(len(QUARTETS)):
        q = QUARTETS[q_k]
        q4_k = NUCLEOTIDES.index(q[3])
        if q4_k == q1_i:
            QUARTETS_P[q_i] += [ q_k ]
    
    P_QUARTETS[q1_i] += [ q_i ]
    N_QUARTETS[q4_i] += [ q_i ]

            
#List of which frames of each arrangement encodes protein 1 or 2; index is arrangement; values are 0=ref, 1=alt, 2=alt_neg, 3=neg
FRAME_F1 = [ 0, 0, 0, 1, 1]
FRAME_F2 = [ 1, 2, 3, 0, 3 ]

def reverse_complement(sequence):
    return ''.join(REVERSE_COMPLEMENT[nucleotide] for nucleotide in reversed(sequence))

#List mapping codons to which quartets; index is [codon_index][frame_index]; values are list of quartet indices
CODONS_TO_QUARTETS = [ [ [] for j in range(4) ] for i in range(len(CODONS)) ]
for q_i in range(len(QUARTETS)):
    q = QUARTETS[q_i]
    q_ref = q[:3]
    q_alt = q[1:]
    q_rc = reverse_complement(q)
    q_alt_neg = q_rc[:3]
    q_neg = q_rc[1:]
    
    CODONS_TO_QUARTETS[CODONS.index(q_ref)][0] += [ q_i ]
    CODONS_TO_QUARTETS[CODONS.index(q_alt)][1] += [ q_i ]
    CODONS_TO_QUARTETS[CODONS.index(q_alt_neg)][2] += [ q_i ]
    CODONS_TO_QUARTETS[CODONS.index(q_neg)][3] += [ q_i ]

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
def reverse_codon_table(codon_table):
    codon_table_rev = { a:[] for c, a in codon_table.items() }    
    for c, a in codon_table.items():
        codon_table_rev[a] += [ c ]
    return codon_table_rev
STANDARD_CODONS_REV = reverse_codon_table(STANDARD_CODONS)

EVODIFF_ALPHABET = {'A': 0,
                    'C': 1,
                    'D': 2,
                    'E': 3,
                    'F': 4,
                    'G': 5,
                    'H': 6,
                    'I': 7,
                    'K': 8,
                    'L': 9,
                    'M': 10,
                    'N': 11,
                    'P': 12,
                    'Q': 13,
                    'R': 14,
                    'S': 15,
                    'T': 16,
                    'V': 17,
                    'W': 18,
                    'Y': 19,
                    'B': 20,
                    'Z': 21,
                    'X': 22,
                    'J': 23,
                    'O': 24,
                    'U': 25,
                    '-': 26,
                    '*': 27,
                    '#': 28,
                    '@': 29,
                    '!': 30}

EVODIFF_ALPHABET_INDEX = {u: i for i, u in enumerate(EVODIFF_ALPHABET)}

ESM_MSA_ALPHABET = {'@': 0, #start token #{'<cls>': 0,
                    '!': 1, #pad token #'<pad>': 1,
                    '*': 2, #stop token #'<eos>': 2,
                    '<unk>': 3,
                    'L': 4,
                    'A': 5,
                    'G': 6,
                    'V': 7,
                    'S': 8,
                    'E': 9,
                    'R': 10,
                    'T': 11,
                    'I': 12,
                    'D': 13,
                    'P': 14,
                    'K': 15,
                    'Q': 16,
                    'N': 17,
                    'F': 18,
                    'Y': 19,
                    'M': 20,
                    'H': 21,
                    'W': 22,
                    'C': 23,
                    'X': 24,
                    'B': 25,
                    'U': 26,
                    'Z': 27,
                    'O': 28,
                    '.': 29,
                    '-': 30, #gap token
                    '<null_1>': 31,
                    '#': 32} #mask token #'<mask>': 32}

REMAP_TO_ESM_MSA = torch.zeros(len(EVODIFF_ALPHABET)).long() - 1
for k, v in EVODIFF_ALPHABET.items():
    if k in ESM_MSA_ALPHABET.keys():
        REMAP_TO_ESM_MSA[v] = ESM_MSA_ALPHABET[k]

REMAP_TO_EVODIFF = torch.zeros(len(ESM_MSA_ALPHABET)).long() - 1
for k, v in ESM_MSA_ALPHABET.items():
    if k in EVODIFF_ALPHABET.keys():
        REMAP_TO_EVODIFF[v] = EVODIFF_ALPHABET[k]

GREMLIN_ALPHABET_ = list("ARNDCQEGHILKMFPSTWYV-")
GREMLIN_ALPHABET = { GREMLIN_ALPHABET_[i]:i for i in range(len(GREMLIN_ALPHABET_)) }