#gremlin, pytorch version
#In pytorch, based on https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF.ipynb
import string
import numpy as np
import torch
from scipy.spatial.distance import pdist,squareform

def parse_fasta(filename,limit=-1):
    '''function to parse fasta'''
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

def load_a3m(path_to_msa, device, gap_cutoff_v=0.5, gap_cutoff_h=0.25):
    names, seqs = parse_fasta(path_to_msa)
    seqs = seqs.view('S4').reshape((seqs.size, -1)).astype('U1')
    seqs = seqs[((seqs=='-').sum(-1)/seqs.shape[1]) < gap_cutoff_h]
    seqs = seqs[:,((seqs=='-').sum(0)/seqs.shape[0]) < gap_cutoff_v]
    seqs = [ ''.join(s) for s in seqs ]
    return seqs
    
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
        pdist = torch.nn.functional.pdist(msa*1.0,p=0)
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

class GREMLIN(torch.nn.Module):
    def __init__(self, L, A):
        super(GREMLIN, self).__init__()
        self.L = L
        self.A = A
        self.W = torch.nn.Parameter(torch.zeros(self.L, self.A, self.L, self.A), requires_grad=True) #2-body-term; W
        self.V = torch.nn.Parameter(torch.zeros(self.L, self.A), requires_grad=True) #1-body-term; V
        
    def get_pll(self, X, temp):
        VW = self.V + torch.tensordot(X, self.W, 2) #predicted
        H = torch.mul(X, VW).sum((1, 2)) / temp #Hamiltonian / E
        Z = torch.sum(torch.logsumexp(VW, 2), 1) #Local Z
        PLL = H - Z #Pseudolikelihood
        return PLL
    
    def forward(self, msa_onehot, temp=1.0, weights=None, neff=None, regularized_loss=False, reg_str=0.01): #where X is one hot MSA
        PLL = self.get_pll(msa_onehot, temp)

        if regularized_loss:
            #Regularization
            L2_V = reg_str * torch.sum(torch.square(self.V))
            L2_W = reg_str * torch.sum(torch.square(self.W)) * 0.5 * (self.L-1) * (self.A-1)
            
            #loss
            if weights is not None:
                loss = -torch.sum(PLL*weights)/torch.sum(weights)
            else:
                loss = -torch.sum(PLL)
            if neff is not None:
                loss = loss + (L2_V + L2_W)/neff
            else:
                loss = loss + (L2_V + L2_W)
            return loss
            
        else:
            return PLL     