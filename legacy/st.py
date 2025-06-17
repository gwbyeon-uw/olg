import torch
from torch import nn
import torch.nn.functional as F

#Various straight-through estimators for backpropagation

#Sigmoid thresholding
class STThresholdGeneric(nn.Module):
    def __init__(self, threshold):
        super(STThresholdGeneric, self).__init__()
        self.threshold = threshold
    
    def forward(self, x): #X should be logits
        thres = (x>self.threshold).float()
        back = torch.sigmoid(x)
        ret = thres - back.detach() + back
        
        return ret
    
#Gumbel softmax
class STArgmaxGumbelSoftmaxGeneric(nn.Module):
    def __init__(self, onehot_dim, tau):
        super(STArgmaxGumbelSoftmaxGeneric, self).__init__()
        self.onehot_dim = onehot_dim
        self.tau = tau
    
    def forward(self, x, argmax=True): #X should be logits
        gumbels = -torch.empty_like(x).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (x + gumbels) / self.tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(1)
        
        if argmax:
            argmax_seq = 1.0 * F.one_hot(torch.argmax(x, 1), self.onehot_dim).permute([0, 2, 1])
            ret = argmax_seq - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        return ret

#Argmax forward, softmax backward
class STArgmaxSoftmaxGeneric(nn.Module):
    def __init__(self, onehot_dim):
        super(STArgmaxSoftmaxGeneric, self).__init__()
        self.onehot_dim = onehot_dim
    
    def forward(self, x): #X should be logits
        softmax_seq = F.softmax(x, dim=1)
        argmax_seq = 1.0 * F.one_hot(torch.argmax(softmax_seq, 1), self.onehot_dim).permute([0, 2, 1])
        ret = argmax_seq - softmax_seq.detach() + softmax_seq
        
        return ret