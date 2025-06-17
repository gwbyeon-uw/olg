import torch
from torch import nn
from st import *

#Translation CNN net
class Translator(nn.Module):
    def __init__(self, n_channel=512): #For some reason, having a conv layer with high ch (=512) has best performance
        super(Translator, self).__init__()        
        self.n_channel = n_channel
        self.n_nuc = 4
        self.n_aa = 20
        
        self.codon = nn.Conv1d(self.n_nuc, self.n_channel, 3, 1)
        self.aa_withstop_f = nn.Conv1d(self.n_channel, self.n_aa + 1, 1, 3)
        self.aa_withstop_r = nn.Conv1d(self.n_channel, self.n_aa + 1, 1, 3)
        self.aa_stop_f = nn.Conv1d(self.n_channel, 2, 1, 3)
        self.aa_stop_r = nn.Conv1d(self.n_channel, 2, 1, 3)
        
        self.stargsoftmax_nostop = STArgmaxSoftmaxGeneric(self.n_aa)
        self.stargsoftmax_withstop = STArgmaxSoftmaxGeneric(self.n_aa + 1)
        self.stargsoftmax_stop = STArgmaxSoftmaxGeneric(2)
        
        self.norm_0 = nn.InstanceNorm1d(1, affine=True, eps=1e-16)
        self.norm_1 = nn.InstanceNorm1d(1, affine=True, eps=1e-16)
        self.norm_2 = nn.InstanceNorm1d(1, affine=True, eps=1e-16)
        
    def forward(self, input, temperature=1.0):
        codon_ = self.codon(input) #stride=1 with size=3; corresponds to 3nt "codon" convolutional filter applied across nucleotide input
        
        #Normalization; seems to give better performance
        codon = torch.reshape(self.norm_0(torch.reshape(codon_, (codon_.shape[0], 1, codon_.shape[1]*codon_.shape[2]))), (codon_.shape[0], codon_.shape[1], codon_.shape[2]))
        
        #Shift +0, +1, +2 for alignment
        codon_frames = [ codon, codon[:, :, 1:], codon[:, :, 2:] ]
        
        #3 frames x 2 strands = 6 protein sequence outputs; stride=3 here
        aa_withstop_ = [ self.aa_withstop_f(c) for c in codon_frames ] + [ self.aa_withstop_r(c) for c in codon_frames ] #21 channels for 20 AA + stop
        aa_stop_ = [ self.aa_stop_f(c) for c in codon_frames ] + [ self.aa_stop_r(c) for c in codon_frames ] #2 channel for nostop/stop
        
        #Normalization; seems to give better performance
        aa_withstop = [ torch.reshape(self.norm_1(torch.reshape(a, (a.shape[0], 1, a.shape[1]*a.shape[2]))), (a.shape[0], a.shape[1], a.shape[2])) for a in aa_withstop_ ]
        aa_stop = [ torch.reshape(self.norm_2(torch.reshape(a, (a.shape[0], 1, a.shape[1]*a.shape[2]))), (a.shape[0], a.shape[1], a.shape[2])) for a in aa_stop_ ]
        
        #Temperature in case we want to use sampling
        aa_withstop_temperature = [ aa * temperature for aa in aa_withstop ]
        aa_stop_temperature = [ aa * temperature for aa in aa_stop ]
        
        #Straight-through estimator; argmax forward, softmax backward
        sampled_withstop = [ self.stargsoftmax_withstop(aa) for aa in aa_withstop_temperature ]
        sampled_stop = [ self.stargsoftmax_stop(aa) for aa in aa_stop_temperature ]
        
        return sampled_withstop, sampled_stop