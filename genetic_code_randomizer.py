#From https://github.com/parizkh/resource-conservation-in-genetic-code

import numpy as np
import pandas

transitions = {'A':'G', 'G':'A', 'C':'T', 'T':'C'}
transversions = {'A':['C', 'T'], 'G':['C', 'T'], 'C':['A', 'G'], 'T':['A', 'G']}
stop_letter = 'X'

def gen_neighbouring_codons(c):
    res = []
    for i in [0,1,2]:
        for b in ['A', 'C', 'G', 'T']:
            if c[i]!=b:
                res.append(c[:i] + b + c[(i+1):])
    return res

def gen_neighbouring_codons_ti(c):
    res = []
    for i in [0,1,2]:
        for b in ['A', 'C', 'G', 'T']:
            if c[i]!=b and b in transitions[c[i]]:
                res.append(c[:i] + b + c[(i+1):])
    return res

def gen_neighbouring_codons_tv(c):
    res = []
    for i in [0,1,2]:
        for b in ['A', 'C', 'G', 'T']:
            if c[i]!=b and b in transversions[c[i]]:
                res.append(c[:i] + b + c[(i+1):])
    return res

# the Grantham matrix of amino acid distances; Grantham, Science, 1974
grantham_matrix = pandas.read_csv("data/grantham.tsv", sep="\t", index_col = 0)

def read_code(fileName):
    code = {}
    with open(fileName, 'r') as f:
        lines = f.read().split("\n")
        # remove the header
        lines.pop(0)
        for l in lines:
            if l!="":
                splitLine = l.split("\t")
                #print(splitLine)
                code[splitLine[1]] = splitLine[0]
    return code

def gen_all_codons():
    res = []
    bases = ['A', 'C', 'G', 'T']
    for c1 in bases:
        for c2 in bases:
            for c3 in bases:
                res = res+[c1+c2+c3]
    return res

#Amino acid permutation
#Maintains the block structure of the standard genetic code, but changes which codons correspond to which amino acid. Position of the stop codons is fixed.
def rand_aa_permutation(standard_code, seed = 0):
    np.random.seed(seed)
    # all amino acids
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', stop_letter]
    # dictionary to store the code
    code = {}
    # randomly shuffle the amino acids 
    shuffled_aas = np.random.permutation(20)
    # the last one is always stop
    shuffled_aas = np.append(shuffled_aas, [20])
    # go over the standard code and always change amino acid i to shuffled_aas[i]
    for codon, aa in standard_code.items():
        code[codon] = aas[shuffled_aas[aas.index(aa)]]
    return code
    
#Amino acid permutation with restriction on the change in number of codons per amino acid
#Amino acid frequency in proteins is positively correlated with number of codons of the amino acid (Gilis, Genome Biology, 2001). This function generates randomized codes using amino acid permutation, but only permutations where number of codons of each amino acid changes by at most max_diff are allowed. Position of the stop codons is fixed.
# max_diff ... maximal allowed change in number of codons per amino acid
def rand_aa_permutation_restricted(standard_code, max_diff, seed = 0):
    np.random.seed(seed)

    # count the number of codons for each amino acid in the standard code
    old_aas, counts = np.unique(list(standard_code.values()), return_counts=True)
    old_aas = old_aas.tolist()
    counts = counts.tolist()
    # delete entries corresponding to stop codons, as they are not permuted
    stop_index = old_aas.index(stop_letter)
    del old_aas[stop_index]
    del counts[stop_index]
    # an array to store the new permutation of amino acids
    new_aas = ['']*20
    # an array indicating whether an amino acid has been assigned a codon block already
    used = [False]*20

    # iterate over blocks of given size, from the largest to the smallest
    for blockSize in [6,4,3,2,1]:
        # choose all blocks of given size and shuffle them
        blocks = [x for x in range(len(counts)) if counts[x]==blockSize]
        np.random.shuffle(blocks)
        # amino acids that need to be placed in this step (otherwise their number of codons would change to much)
        must_have_pos = [old_aas[x] for x in range(len(old_aas)) if counts[x]==blockSize+max_diff and not used[x]]
        # place those amino acids to the first len(must_have_pos) blocks and mark them as used
        for i in range(len(must_have_pos)):
            aa = must_have_pos[i]
            new_aas[blocks[i]] = aa
            used[old_aas.index(aa)] = True
        # amino acids that may get a position now
        may_have_pos = [old_aas[x] for x in range(len(old_aas)) if counts[x]>=blockSize-max_diff and counts[x]<blockSize+max_diff and not used[x]]
        # shuffle the potential amino acids and fill in the remaining blocks
        np.random.shuffle(may_have_pos)
        for i in range(len(must_have_pos), len(blocks)):
            aa = may_have_pos[i-len(must_have_pos)]
            new_aas[blocks[i]] = aa
            used[old_aas.index(aa)] = True

    # construct the new code
    old_aas += [stop_letter]
    new_aas += [stop_letter]
    code = {}
    for codon, aa in standard_code.items():
        code[codon] = new_aas[old_aas.index(aa)]

    return code	

#Quartet shuffling
#Randomizes the first two positions of codons, while maintaining that the two sets of first and second positions in which the stop codons reside are separated by a single transition mutation.
def rand_quartet_shuffling(standard_code, seed = 0):
    np.random.seed(seed)
    code = standard_code.copy()
    bases = ['A', 'C', 'G', 'T']
    # all possible pairs of bases
    pairs = ['TA', 'TG', 'TT', 'TC', 'CT', 'CC', 'CA', 'CG', 'AT', 'AC', 'AA', 'AG', 'GT', 'GC', 'GA', 'GG']
    # permutation of the pairs to construct the alternative codon
    perm_pairs = pairs.copy()
    np.random.shuffle(perm_pairs)
    
    # the "opal" stop codon is separated by a transition from the "ambre" stop codon
    stop_pair_1 = perm_pairs[0]
    # the transition can happen in the first, or in the second position
    transition_pos = np.random.randint(2)
    if transition_pos==0:
        stop_pair_2 = transitions[stop_pair_1[0]]+stop_pair_1[1]
    else:
        stop_pair_2 = stop_pair_1[0]+transitions[stop_pair_1[1]]
    tmp_pos = perm_pairs.index(stop_pair_2)
    perm_pairs[tmp_pos] = perm_pairs[1]
    perm_pairs[1] = stop_pair_2
    # construct the new code
    for i in range(len(pairs)):   
        for b in bases:
            old_codon = pairs[i]+b
            new_codon = perm_pairs[i]+b
            code[new_codon] = standard_code[old_codon]
                                            
    return code
    
#N-block Shuffler, Caporaso, J Mol Evol, 2005
#"The N-Blocks of the genetic code are held constant. We define an N-Block as a 1-, 2-, or 4-codon block in which all codons specify the same amino acid. The block structure is defined based on the canonical code, and all blocks of the same size are permuted among themselves. (...) The N-Block Shuffler maintains the number of codons per amino acid and the block structure of the genetic code." The position of stop codons is fixed.
def rand_N_block_shuffler(seed = 0):
    np.random.seed(seed)
    
    # the block structure of the standard code
    N_blocks_1 = ['ATA', 'ATG', 'TGG']
    N_blocks_1_aas = ['I', 'M', 'W']
    N_blocks_2 = [['TTT', 'TTC'], ['TTA', 'TTG'], ['TAT', 'TAC'], ['TGT', 'TGC'], ['CAT', 'CAC'], ['CAA', 'CAG'],
        ['ATT', 'ATC'], ['AAT', 'AAC'], ['AAA', 'AAG'], ['AGT', 'AGC'], ['AGA', 'AGG'], ['GAT', 'GAC'], ['GAA', 'GAG']]
    N_blocks_2_aas = ['F', 'L', 'Y', 'C', 'H', 'Q', 'I', 'N', 'K', 'S', 'R', 'D', 'E']
    N_blocks_4 = [['TCT', 'TCC', 'TCA', 'TCG'], ['CTT', 'CTC', 'CTA', 'CTG'], ['CCT', 'CCC', 'CCA', 'CCG'], 
        ['CGT', 'CGC', 'CGA', 'CGG'], ['ACT', 'ACC', 'ACA', 'ACG'], ['GTT', 'GTC', 'GTA', 'GTG'], ['GCT', 'GCC', 'GCA', 'GCG'],
        ['GGT', 'GGC', 'GGA', 'GGG']]
    N_blocks_4_aas = ['S', 'L', 'P', 'R', 'T', 'V', 'A', 'G']
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    code = {}
    # shuffle 1-blocks
    perm1 = np.random.permutation(len(N_blocks_1))
    for i in range(len(N_blocks_1)):
        code[N_blocks_1[i]] = N_blocks_1_aas[perm1[i]]
    # shuffle 2-blocks
    perm2 = np.random.permutation(len(N_blocks_2))
    for i in range(len(N_blocks_2)):
        for codon in N_blocks_2[i]:
            code[codon] = N_blocks_2_aas[perm2[i]]
    # shuffle 4-blocks
    perm4 = np.random.permutation(len(N_blocks_4))
    for i in range(len(N_blocks_4)):
        for codon in N_blocks_4[i]:
            code[codon] = N_blocks_4_aas[perm4[i]]
    # add stop codons
    for codon in stop_codons:
        code[codon] = stop_letter

    return code

#Codon Shuffler, Caporaso, J Mol Evol, 2005
#Amino acids are assigned to codons randomly, but the number of codons each amino acid has is preserved. Maintains the number of codons per amino acid, but produces codes without block structure. Position of stop codons fixed.
def rand_codon_shuffler(standard_code, seed = 0):
    np.random.seed(seed)
    # dictionary to store the code
    code = standard_code.copy()
    # count the number of codons for each amino acid
    values, counts = np.unique(list(standard_code.values()), return_counts=True)
    # shuffle the codons
    # stop codons are fixed - don't include them in the permutation
    codons = list(standard_code.keys())
    perm = np.random.permutation(64)
    codons = [codons[x] for x in perm if codons[x] not in ['TAA', 'TAG', 'TGA']]
    # assign the codons to amino acids
    start = 0
    for i in range(len(values)):
        aa = values[i]
        if aa!=stop_letter:
            count = counts[i]
            for j in range(start, start+count):
                code[codons[j]]=str(aa)
            start = start+count
    return code
    
#AAAGALOC Shuffler, Caporaso, J Mol Evol, 2005
#Generates completely random codes, the only restriction is that each amino acid has at least one codon. The resulting codes do not in general have a block structure. Number of codons per amino acid is not preserved. Position of stop codons fixed.
def rand_aaagaloc_shuffler(standard_code, seed = 0):
    np.random.seed(seed)
    # dictionary to store the code
    code = {}
    # list of amino acids
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # fix stop codons
    code['TAA'] = stop_letter
    code['TAG'] = stop_letter
    code['TGA'] = stop_letter
    # shuffle the codons
    # stop codons are fixed - don't include them in the permutation
    codons = list(standard_code.keys())
    perm = np.random.permutation(64)
    codons = [codons[x] for x in perm if codons[x] not in ['TAA', 'TAG', 'TGA']]
    # the first 20 codons are assigned each to the corresponding amino acid
    # to fulfill the requirement that each amino acid has at least one codon
    for i in range(20):
        code[codons[i]] = aas[i]
    # the remaining codons are assigned to amino acids randomly
    for i in range(41):
        codon = codons[20+i]
        aa = aas[np.random.randint(0,20)]
        code[codon] = aa

    return code

#The Massey models, J Mol Evol, 2008

#The models by Massey (J Mol Evol, 2008) attempt to model the evolution of the genetic code, in particular the stepwise addition of new amino acids to the growing code.
#The amino acid to be added next is chosen randomly from all not yet placed amino acids with probability proportional to its physicochemical similarity to the parent acid. The physicochemical similarity is defined as 1/d, where d is the Grantham distance of the proposed amino acid and the parent amino acid.
#These models preserve the block structure of the genetic code. Further, the models tend to create codes in which neighbouring amino acids are physicochemically similar.

#Random expansion
#In Random expansion (Model 1 in Massey, J Mol Evol 2008), we start with a random amino acid assigned to a random codon block. Subsequent amino acids are chosen based on their similarity with the amino acid previously added to the code and placed in a randomly chosen codon block neighbouring the previous one. If no such codon block exists, a codon block is chosen randomly from the remaining free blocks.
'''
Helper function to get all neighbouring codon blocks of a given block.
bl ... a codon block; list of codons
blocks ... a list of blocks, each list of codons
'''
def get_neighbouring_blocks(bl, blocks):
    neighbours = []
    for codon in bl:
        neighbours.append(gen_neighbouring_codons_ti(codon)+gen_neighbouring_codons_tv(codon))
    neighbours = np.unique(neighbours)
    # return blocks that share at least one element with neighbours
    return [b for b in blocks if set(b) & set(neighbours)]

def rand_random_expansion(seed = 0):
    np.random.seed(seed)
    # dictionary to store the code
    code = {}
    # define the blocks of the code
    blocks = [["TTT", "TTC"], ["TAT", "TAC"], ["ATA", "ATC", "ATT"], ["AAT", "AAC"], 
        ["AGT", "AGC", "TCA", "TCC", "TCG", "TCT"], ["ACA", "ACC", "ACG", "ACT"], ["TGT", "TGC"], 
        ["CGA", "CGC", "CGG", "CGT", "AGA", "AGG"], ["TTA", "TTG", "CTA", "CTC", "CTG", "CTT"],
        ["CCA", "CCC", "CCT", "CCG"], ["CAT", "CAC"], ["GTG", "GTT", "GTA", "GTC"], 
        ["GCA", "GCG", "GCC", "GCT"], ["GAT", "GAC"], ["GGA", "GGC", "GGG", "GGT"], 
        ["TGG"], ["CAA", "CAG"], ["ATG"], ["AAA", "AAG"], ["GAA", "GAG"]]
    # list of aminoacids 
    aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    previous_block = None
    previous_aa = None
    for i in range(20):
        # the first amino acid and block
        if previous_aa == None:
            # choose random block and random amino acid
            next_block = blocks[np.random.randint(len(blocks))]
            next_aa = aminoacids[np.random.randint(len(aminoacids))]
        else:
            # choose a block neighbouring the previous one
            candidate_blocks = get_neighbouring_blocks(previous_block, blocks)
            # at least one neighbouring block is free
            if len(candidate_blocks)>0:
                # choose a random one from the neighbouring blocks
                next_block = candidate_blocks[np.random.randint(len(candidate_blocks))]
            # there are no free neighbouring blocks
            else:
                # choose a random block from all remaining free blocks
                next_block = blocks[np.random.randint(len(blocks))]
            # probabilities of adding the individual amino acids
            probs = [1/grantham_matrix.at[previous_aa, aa] for aa in aminoacids]
            probs = [x/sum(probs) for x in probs]
            # choose the next amino acid
            next_aa = aminoacids[np.random.choice(len(aminoacids), p=probs)]            
        # update aminoacids and blocks    
        del aminoacids[aminoacids.index(next_aa)]
        del blocks[blocks.index(next_block)]
        # add the new codons to the code
        for codon in next_block:
            code[codon] = next_aa
        previous_block = next_block
        previous_aa = next_aa  

    # add stop codons
    stop_codons = ["TAA", "TAG", "TGA"]
    for stop in stop_codons:
        code[stop] = stop_letter

    return code
    
#Ambiguity reduction 1, Ambiguity reduction 2 and 2-1-3 model
#These are Model 2a, 2b and 3 in Massey, J Mol Evol, 2008. The codon blocks are added in order specified by Fig. 2a, 2b and 3 in Massey, J Mol Evol, 2008. In addition, in Model 3 the four starting amino acids are fixed (V, A, D, and G).
#Below we define the order of adding blocks, as well as the parent of each newly added block (0 if no parent) for models 2a, 2b and the 213 model:
# Model 2a
blocks_massey_2a = [["TTT", "TTC"], ["TAT", "TAC"], ["ATA", "ATC", "ATT"], ["AAT", "AAC"], 
        ["AGT", "AGC", "TCA", "TCC", "TCG", "TCT"], ["ACA", "ACC", "ACG", "ACT"], ["TGT", "TGC"], 
        ["CGA", "CGC", "CGG", "CGT", "AGA", "AGG"], ["TTA", "TTG", "CTA", "CTC", "CTG", "CTT"],
        ["CCA", "CCC", "CCT", "CCG"], ["CAT", "CAC"], ["GTG", "GTT", "GTA", "GTC"], 
        ["GCA", "GCG", "GCC", "GCT"], ["GAT", "GAC"], ["GGA", "GGC", "GGG", "GGT"], 
        ["TGG"], ["CAA", "CAG"], ["ATG"], ["AAA", "AAG"], ["GAA", "GAG"]]
parents_massey_2a = [0, 0, 1, 2, 1, 3, 2, 4, 1, 5, 2, 3, 6, 4, 0, 7, 11, 3, 4, 14]

# Model 2b
blocks_massey_2b = [["TTA", "TTG", "CTA", "CTC", "CTG", "CTT"], ["CAT", "CAC"], ["GTG", "GTT", "GTA", "GTC"],
    ["GAT", "GAC"], ["CCA", "CCC", "CCT", "CCG"], ["CGA", "CGC", "CGG", "CGT", "AGA", "AGG"], 
    ["GCA", "GCG", "GCC", "GCT"], ["GGA", "GGC", "GGG", "GGT"], ["TTT", "TTC"], 
    ["AGT", "AGC", "TCA", "TCC", "TCG", "TCT"], ["TAT", "TAC"], ["TGT", "TGC"], ["ATA", "ATC", "ATT"], 
    ["ACA", "ACC", "ACG", "ACT"], ["AAT", "AAC"], ["TGG"], ["ATG"], ["CAA", "CAG"], ["AAA", "AAG"], 
    ["GAA", "GAG"]]
parents_massey_2b = [0, 0, 1, 2, 1, 2, 3, 4, 1, 5, 2, 6, 3, 7, 4, 12, 13, 2, 15, 4]

# Model 3 (213 model)
blocks_213 = [["GTG", "GTT", "GTA", "GTC"], ["GCA", "GCG", "GCC", "GCT"], ["GAT", "GAC"],
    ["GGA", "GGC", "GGG", "GGT"], ["TTA", "TTG", "CTA", "CTC", "CTG", "CTT"],  ["CCA", "CCC", "CCT", "CCG"],
    ["CAT", "CAC"], ["CGA", "CGC", "CGG", "CGT", "AGA", "AGG"], ["TTT", "TTC"], 
    ["AGT", "AGC", "TCA", "TCC", "TCG", "TCT"], ["TAT", "TAC"], ["TGT", "TGC"], ["ATA", "ATC", "ATT"], 
    ["ACA", "ACC", "ACG", "ACT"], ["AAT", "AAC"], ["TGG"], ["CAA", "CAG"], ["ATG"], ["AAA", "AAG"], 
    ["GAA", "GAG"]]
parents_213 = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 12, 7, 13, 15, 3]
fixed_aas_213 = {1: 'V', 2: 'A', 3: 'D', 4: 'G'}

'''
blocks ... the codon blocks, in the order in which they are added to the code; list of lists of codons
parents ... a list of the same length as blocks, specifying the parent for each block in blocks; 
    0 if a block has no parent
fixed_aas ... a dictionary in the form index:aa, index is index of a block, aa the amino acid that will be
    assigned to this block; None if not applicable (no blocks with fixed meaning)
'''
def rand_massey_2_3_shuffler(blocks, parents, fixed_aas = None, seed = 0):
    np.random.seed(seed)
    # list of amino acids
    aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # list to store the amino acids added to the code
    new_aas = []
    # a dictionary to store the code
    code = {}
    for i in range(len(blocks)):
        # no parent: either use the specified fixed aas or choose a random amino acid
        if parents[i]==0:
            if fixed_aas == None:
                aa = aminoacids[np.random.randint(len(aminoacids))]
            else:
                aa = fixed_aas[i+1]
        # choose amino acid with probability proportional to 1/d(parent, aa), d the Grantham distance
        else:
            parent_aa = new_aas[parents[i]-1]
            # the probabilities of choosing the individual amino acids
            probs = [1/grantham_matrix.at[parent_aa, aa] for aa in aminoacids]
            probs = [x/sum(probs) for x in probs]
            # choose a random amino acid with probability proportional to probs
            aa = aminoacids[np.random.choice(len(aminoacids), p=probs)]
        # update new_aas, delete the newly added amino acid from aminoacids
        new_aas.append(aa)
        del aminoacids[aminoacids.index(aa)]
        # update the code
        for codon in blocks[i]:
            code[codon] = aa

    # add stop codons
    stop_codons = ["TAA", "TAG", "TGA"]
    for stop in stop_codons:
        code[stop] = stop_letter

    return code

standard_code_ = read_code("data/code_standard.tsv")
standard_code = {}
for k,v in standard_code_.items():
    if v != '*':
        standard_code[k] = v
    else:
        standard_code[k] = stop_letter

# maximum difference between number of codons per aa in the alternative codes and the standard one
# only used when randomization_type = "aa_permutation_restricted"
max_diff = 2

def generate_random_code(randomization_type, num_codes, seed_off=0):
    codes = [ None for i in range(num_codes) ]
    for seed_ in range(num_codes):
        seed = seed_ + seed_off
        if randomization_type=="aa_permutation":
            code = rand_aa_permutation(standard_code, seed)
        elif randomization_type=="aa_permutation_restricted":
            code = rand_aa_permutation_restricted(standard_code, max_diff, seed)
        elif randomization_type=="quartet":
            code = rand_quartet_shuffling(standard_code, seed)
        elif randomization_type=="nblock":
            code = rand_N_block_shuffler(seed)
        elif randomization_type=="codon":
            code = rand_codon_shuffler(standard_code, seed)
        elif randomization_type=="aaagaloc":
            code = rand_aaagaloc_shuffler(standard_code, seed)
        elif randomization_type=="random_expansion":
            code = rand_random_expansion(seed)
        elif randomization_type=="ambiguity_reduction_1":
            code = rand_massey_2_3_shuffler(blocks_massey_2a, parents_massey_2a, None, seed)
        elif randomization_type=="ambiguity_reduction_2":
            code = rand_massey_2_3_shuffler(blocks_massey_2b, parents_massey_2b, None, seed)
        elif randomization_type=="213":
            code = rand_massey_2_3_shuffler(blocks_213, parents_213, fixed_aas_213, seed)
    
        codes[seed_] = code
        
    return codes

#Example
"""
randomization_types = [ "aa_permutation", "aa_permutation_restricted", 
                       "quartet", "nblock", "codon", "aaagaloc", 
                       "random_expansion", "ambiguity_reduction_1", 
                       "ambiguity_reduction_2", "213"]

num_codes = 100
seed_offset = 100

rand_codes = { rt: generate_random_code(rt, num_codes, seed_offset) for rt in randomization_types }
rand_codes_type = [ k for k, v in rand_codes.items() for vv in v ]
rand_codes_code = [ vv for k, v in rand_codes.items() for vv in v ]
"""