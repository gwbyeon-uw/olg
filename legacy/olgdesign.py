import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_optimizer as optim

import numpy as np

from st import *
from translator import *

import pickle, json, importlib, copy, sys

import matplotlib
import matplotlib.pyplot as plt    
import matplotlib.animation as anim
import matplotlib.colors as colors
import matplotlib.patches as patches

try:
    get_ipython
    from IPython.display import clear_output
    ipy = True
except NameError:
    ipy = False    
    
CLR = "\x1B[0K" #for printing progress
amino_acids = list("ARNDCQEGHILKMFPSTWYV*")
AA_1_N = {a:n for n,a in enumerate(amino_acids)} #

#Container for optimizing nucleotide sequence overlapping genes
class SeqOptN(nn.Module):
    def __init__(self, device, network_name, seq_length, offset, f1_len, f2_len, 
                 translator, model, frames, last_stop=True, gap_len=200, 
                 retain_grad=False):
        super(SeqOptN, self).__init__()
        self.device = device
        self.seq_length = seq_length * 3
        self.f1_length = f1_len
        self.f2_length = f2_len
        self.offset = offset
        self.one_seed = torch.ones(1)        
        self.onehot_dim = 4
        self.batch_size = 1
        self.dense = nn.Linear(1, self.seq_length * self.onehot_dim, bias=False)
        torch.nn.init.xavier_normal_(self.dense.weight) #glorot initialization
        self.instance_norm = nn.InstanceNorm2d(1, affine=True, eps=1e-16) #Affine intializes to weight 1, bias 0
        self.stsampler = STArgmaxSoftmaxGeneric(4) #Straight-through estimator (Softmax backward, Argmax forward)
        self.f1 = frames[0]
        self.f2 = frames[1]
        self.frame_phase = (self.f2 - self.f1) % 3
        self.frame_phase_offset = int(self.frame_phase > 0)
        self.last_stop = True
        self.last_stop_offset = int(self.last_stop)
        self.retain_grad = retain_grad
        self.gap_len = gap_len
        
        self.joined_length = self.f1_length + self.f2_length
        self.idx_pdb = torch.arange(self.joined_length).long().view(1, self.joined_length)
        self.idx_pdb[:, self.f2_length:] += self.gap_len
        
        self.translator = translator
        self.predictor = model
        self.set_device(device)
        
        self.network_name = network_name

        if network_name == 'rf_Jul31_2022':
            #Prepare templates and other constant inputs to rosettafold
            #self.fake_template_f1 = self.predictor.prep_fake_template(self.f1_length, self.device)
            #self.fake_template_f2 = self.predictor.prep_fake_template(self.f2_length, self.device)
            self.term_info_f1 = torch.zeros((self.f1_length, 2), device=self.device).float() #N/C term info
            self.term_info_f1[0, 0] = 1.0 # flag for N-term
            self.term_info_f1[-1, 1] = 1.0 # flag for C-term
            self.term_info_f2 = torch.zeros((self.f2_length, 2), device=self.device).float()
            self.term_info_f2[0, 0] = 1.0 # flag for N-term
            self.term_info_f2[-1, 1] = 1.0 # flag for C-term
        else:
            pass

        self.layers = {}
    
    def set_device(self, device):
        self = self.to(device)
        self.translator = self.translator.to(device)
        self.predictor = self.predictor.to(device)
        self.one_seed = self.one_seed.to(device)
        self.idx_pdb = self.idx_pdb.to(device)
    
    def set_weights(self, new_params):
        model_state = self.state_dict()
        for name, param in new_params.items():
            if isinstance(param, torch.nn.parameter.Parameter):
                param = param.data
            model_state[name].copy_(param)
            
    def forward(self):
        seed = self.one_seed #Dummy, always 1.0
        dense = self.dense(seed) #Weights here are the "logits" for nucleotide sequence
        reshaped = torch.reshape(dense, (1, self.onehot_dim, self.seq_length)) #Reshape to 1x4xLength for one-hot nucleotide
        normed = self.instance_norm(reshaped.unsqueeze(0))[0] #Normalize
        sampled = torch.cat([ self.stsampler(normed[i:(i+1)]) for i in range(self.batch_size) ], 0) #Argmax forward pass; softmax backward        
                
        sampled_slice1 = sampled[:, :, 0:((self.f1_length+self.last_stop_offset)*3)] #Nucleotides for first protein
        sampled_slice2 = sampled[:, :, (self.offset*3):((self.offset+self.f2_length+self.last_stop_offset+self.frame_phase_offset)*3)] #Nucleotides for second protein (offseted)
        
        withstop_slice1, stop_slice1 = self.translator(sampled_slice1) #Translation for first protein
        withstop_slice2, stop_slice2 = self.translator(sampled_slice2) #Translation for second protein
        
        #Get the protein sequences for the specified frames
        prot_f1 = withstop_slice1[self.f1]
        stop_f1 = stop_slice1[self.f1]
        prot_f2 = withstop_slice2[self.f2]
        stop_f2 = stop_slice2[self.f2]
        
        #Flip sequence if negative strand
        if self.f1 >= 3:
            prot_f1 = prot_f1.flip(2)
            stop_f1 = stop_f1.flip(2)
        if self.f2 >= 3:
            prot_f2 = prot_f2.flip(2)
            stop_f2 = stop_f2.flip(2)
        
        #Last AA
        last_f1 = prot_f1[:, :, -1]
        last_f2 = prot_f2[:, :, -1]
        
        #Excluding last if forcing stop codon at end
        if self.last_stop:
            prot_f1 = prot_f1[:, :, :-1]
            stop_f1 = stop_f1[:, :, :-1]
            prot_f2 = prot_f2[:, :, :-1]
            stop_f2 = stop_f2[:, :, :-1]

        #RosettaFold
        #Rosetta update
        pinput_f1 = prot_f1.permute([0, 2, 1]).unsqueeze(0)
        pinput_f2 = prot_f2.permute([0, 2, 1]).unsqueeze(0)
        pinput_f1_argmax = torch.argmax(pinput_f1, axis=-1)
        pinput_f2_argmax = torch.argmax(pinput_f2, axis=-1)
        
        if self.network_name == 'rf_Jul31_2022':
            (xyz_t_f1, t1d_f1, t2d_f1, alpha_t_f1), _ = self.fake_template_f1
            (xyz_t_f2, t1d_f2, t2d_f2, alpha_t_f2), _ = self.fake_template_f2

            pred_f1 = self.predictor(pinput_f1_argmax, pinput_f1, self.term_info_f1, t1d_f1, t2d_f1, xyz_t_f1, xyz_t_f1[:,0], alpha_t_f1, nmer=1, L_s=[], device=self.device)
            pred_f2 = self.predictor(pinput_f1_argmax, pinput_f1, self.term_info_f2, t1d_f1, t2d_f1, xyz_t_f1, xyz_t_f1[:,0], alpha_t_f1, nmer=1, L_s=[], device=self.device)
        else:
            pred_f1 = self.predictor(pinput_f1_argmax, msa_one_hot=pinput_f1, use_transf_checkpoint=True)
            pred_f2 = self.predictor(pinput_f2_argmax, msa_one_hot=pinput_f2, use_transf_checkpoint=True)
        
        pred_ret = [pred_f1, pred_f2]
        pinput = [pinput_f1, pinput_f2]

        #For access to various tensors
        self.layers['dense'] = dense
        self.layers['reshaped'] = reshaped
        self.layers['sampled'] = sampled
        self.layers['sampled_slice1'] = sampled_slice1
        self.layers['sampled_slice2'] = sampled_slice2
        self.layers['prot_slice1'] = prot_f1
        self.layers['prot_slice2'] = prot_f2
        self.layers['stop_slice1'] = stop_f1
        self.layers['stop_slice2'] = stop_f2
        self.layers['last_slice1'] = last_f1
        self.layers['last_slice2'] = last_f2
        self.layers['pred'] = pred_ret
        
        #For access to gradients w.r.t. different layers
        if self.retain_grad:
            for k, v in self.layers.items():
                if k != 'pred':
                    v.retain_grad()
        
        return pred_ret, [stop_f1, stop_f2], [prot_f1, prot_f2], [last_f1, last_f2], [pinput_f1, pinput_f2]

#Class to help calculate losses and extract various layers
class SeqOptNLoss():
    def __init__(self, model, device, alpha_max, 
                 mask_f1, mask_f2, bkg_f1, bkg_f2, 
                 force_aa_f1, force_aa_f2, no_f2,
                 weight_kl, weight_ce, weight_lddt,
                 weight_stop, weight_force, weight_last, 
                 weight_rog, weight_rog_sum, rog_thres, rog_sum_thres,                
                 weight_surfnp, surfnp_thres, 
                 weight_nc, nc_thres,
                 aloss_max):
        
        self.device = device
        self.model = model
        self.alpha_max = alpha_max #For weighted average of losses for each protein
        self.weights = {'kl': weight_kl, #Weights for each loss
                        'ce': weight_ce,
                        'lddt': weight_lddt,
                        'stop': weight_stop,
                        'force': weight_force,
                        'last': weight_last,
                        'rog': weight_rog,
                        'rog_sum': weight_rog_sum,
                        'surfnp': weight_surfnp,
                        'nc': weight_nc}
        self.thres = {'rog': rog_thres,
                      'rog_sum': rog_sum_thres,
                      'surfnp': surfnp_thres,
                      'nc': nc_thres}
        self.bkg_f1 = bkg_f1 #Background for KLD loss
        self.bkg_f2 = bkg_f2
        self.no_f2 = no_f2
        
        #Default masks if none provided
        if mask_f1 is None:
            self.mask_f1 = torch.ones(model.f1_length, model.f1_length).to(device)
            self.mask_f1.fill_diagonal_(0.0)
            self.mask_f1 = mask.unsqueeze(0)
        else:
            self.mask_f1 = mask_f1.to(device)
        
        if mask_f2 is None:
            self.mask_f2 = torch.ones(model.f2_length, model.f2_length).to(device)
            self.mask_f2.fill_diagonal_(0.0)
            self.mask_f2 = mask.unsqueeze(0)
        else:
            self.mask_f2 = mask_f2.to(device)

        #Default force_aa if none provided
        self.force_aa_f1 = force_aa_f1.to(device)
        self.force_aa_f2 = force_aa_f2.to(device)
        if self.force_aa_f1 is None:
            self.force_aa_f1_indices = None
        else:
            self.force_aa_f1_indices = torch.sum(self.force_aa_f1, 1)[0].nonzero()[:, 0]
            self.force_aa_f1 = self.force_aa_f1[:, :, self.force_aa_f1_indices]
        
        if self.force_aa_f2 is None:
            self.force_aa_f2_indices = None
        else:
            self.force_aa_f2_indices = torch.sum(self.force_aa_f2, 1)[0].nonzero()[:, 0]
            self.force_aa_f2 = self.force_aa_f2[:, :, self.force_aa_f2_indices]
        
        #Total number of forced AA's
        self.force_aa_f1_total = torch.sum(self.force_aa_f1)
        self.force_aa_f2_total = torch.sum(self.force_aa_f2)
        
        #For forcing stop at the end
        self.stop_onehot = torch.zeros(1, 21).to(device)
        self.stop_onehot[:, 20] = 1.0
        
        #Losses
        self.total_loss = None
        self.nloss = None
        self.hloss = None
        self.hloss_kw = [ 'kl', 'ce', 'lddt']
        self.aloss_kw = [ 'rog', 'rog_sum', 'surfnp', 'nc' ]
        self.nloss_kw = [ 'stop', 'force', 'last' ]
        self.aloss_max = aloss_max
        self.unweighted_losses_f1 = {'kl': None,
                                     'ce': None,
                                     'lddt': None,
                                     'stop': None,
                                     'force': None,
                                     'last': None,
                                     'rog': None,
                                     'rog_sum': None,
                                     'surfnp': None,
                                     'nc': None}
        self.unweighted_losses_f2 = {'kl': None,
                                     'ce': None,
                                     'lddt': None,
                                     'stop': None,
                                     'force': None,
                                     'last': None,
                                     'rog': None,
                                     'rog_sum': None,
                                     'surfnp': None,
                                     'nc': None}
        
        self.pred = None
        self.stop = None
        self.prot = None
        self.last_aa = None
        self.pinput = None
        
        self.c6d_dict_f1 = None
        self.c6d_dict_f2 = None
        self.probs_f1 = None
        self.probs_f2 = None
        self.out_f1 = None
        self.out_f2 = None
        self.pinput_f1 = None
        self.pinput_f2 = None
        
        self.last_grad_norm = None
        self.h_history = []
        
    #For printing losses
    def __str__(self):
        st = ""
        for k in self.unweighted_losses_f1.keys():
            v_f1 = self.unweighted_losses_f1[k].item()
            v_f2 = self.unweighted_losses_f2[k].item()
            st += f'{k:<8}{v_f1:>8.4f}{v_f2:>8.4f}{CLR}\n'
        
        total_loss = self.total_loss
        h_loss = self.hloss
        a_loss = self.aloss
        n_loss = self.nloss
        st += f'{"h_loss":<8}{h_loss:>8.4f}{CLR}\n'
        st += f'{"a_loss":<8}{a_loss:>8.4f}{CLR}\n'
        st += f'{"n_loss":<8}{n_loss:>8.4f}{CLR}\n'
        st += f'{"total_loss":<8}{total_loss:>8.4f}{CLR}\n'
        
        if self.last_grad_norm is not None:
            last_grad_norm = self.last_grad_norm
            st += f'{"grad_norm":<8}{last_grad_norm:>8.4f}{CLR}\n'
        else:
            st += f'{CLR}\n'
            
        return st
    
    #Backward pass helper function
    def update_gradient(self, norm_factor=1.0):#, percentile_zero=0.75):
        self.total_loss.backward()    
        
        #Normalize gradient
        grad = [p for p in self.model.dense.parameters() ][0].grad
        grad_norm = torch.norm(grad, p=2.0).detach()
        self.last_grad_norm = grad_norm
        grad.detach().mul_(norm_factor / grad_norm)
         
    #Grab reshaped gradient on nucleotide one-hot matrix
    def get_detached_gradient(self, unnormed=False):
        grad = [p for p in self.model.dense.parameters() ][0].grad
        grad_reshaped = torch.reshape(grad, (1, self.model.onehot_dim, self.model.seq_length)).detach().clone()
        if unnormed:
            return grad_reshaped.mul(self.last_grad_norm)
        else:
            return grad_reshaped
    
    #Run current sequence thru rosettafold and update predictions
    def update_model(self):
        self.pred, self.stop, self.prot, self.last_aa, self.pinput = self.model()
        
        self.c6d_dict_f1, self.probs_f1 = get_c6d_dict(self.pred[0], grad=True)
        self.c6d_dict_f2, self.probs_f2 = get_c6d_dict(self.pred[1], grad=True)
        self.out_f1 = {'c6d': self.c6d_dict_f1, 'xyz': self.pred[0].get('xyz', None)}
        self.out_f2 = {'c6d': self.c6d_dict_f2, 'xyz': self.pred[1].get('xyz', None)}
        self.pinput_f1 = self.pinput[0]
        self.pinput_f2 = self.pinput[1]
    
    #KL divergence loss for C6D vs background
    def update_kl_loss(self):
        kl_f1 = -1.0 * get_kl_loss(self.out_f1, self.bkg_f1, self.mask_f1)
        kl_f2 = -1.0 * get_kl_loss(self.out_f2, self.bkg_f2, self.mask_f2)
        self.unweighted_losses_f1['kl'] = kl_f1
        self.unweighted_losses_f2['kl'] = kl_f2
    
    #Cross entropy loss for C6D
    def update_ce_loss(self, beta=1.0):
        ce_f1 = get_entropy_loss(self.out_f1, self.mask_f1, beta=beta)
        ce_f2 = get_entropy_loss(self.out_f2, self.mask_f2, beta=beta)
        self.unweighted_losses_f1['ce'] = ce_f1
        self.unweighted_losses_f2['ce'] = ce_f2
    
    #LDDT loss
    def update_lddt_loss(self):
        mean_lddt_f1 = -1.0 * torch.mean(self.pred[0]['lddt'])
        mean_lddt_f2 = -1.0 * torch.mean(self.pred[1]['lddt'])
        self.unweighted_losses_f1['lddt'] = mean_lddt_f1
        self.unweighted_losses_f2['lddt'] = mean_lddt_f2
    
    #Radius of gyration loss
    def update_rog_loss(self):
        rog_f1_ = get_rog_loss(self.pred[0]['xyz'])#, self.thres['rog'])[0]
        rog_f2_ = get_rog_loss(self.pred[1]['xyz'])#, self.thres['rog'])[0]
        rog_f1 = F.relu(rog_f1_ - self.thres['rog'])[0]
        rog_f2 = F.relu(rog_f2_ - self.thres['rog'])[0]
        rog_sum = rog_f1_ + rog_f2_
        rog_sum_loss = F.relu(rog_sum - self.thres['rog_sum'])[0] * 0.5
        self.unweighted_losses_f1['rog'] = rog_f1
        self.unweighted_losses_f2['rog'] = rog_f2
        self.unweighted_losses_f1['rog_sum'] = rog_sum_loss
        self.unweighted_losses_f2['rog_sum'] = rog_sum_loss
    
    #Surface non polar loss
    def update_surfnp_loss(self):
        surfnp_f1 = get_surfnp_loss(self.out_f1, self.pinput_f1, target_polar=self.thres['surfnp'])
        surfnp_f2 = get_surfnp_loss(self.out_f2, self.pinput_f2, target_polar=self.thres['surfnp'])
        self.unweighted_losses_f1['surfnp'] = surfnp_f1
        self.unweighted_losses_f2['surfnp'] = surfnp_f2
    
    #Net charge loss
    def update_nc_loss(self):
        nc_f1 = get_nc_loss(self.pinput_f1, target_charge=self.thres['nc'])
        nc_f2 = get_nc_loss(self.pinput_f2, target_charge=self.thres['nc'])
        self.unweighted_losses_f1['nc'] = nc_f1
        self.unweighted_losses_f2['nc'] = nc_f2
    
    #Loss to minimize stop codon
    def update_stop_loss(self):
        stop_f1 = F.relu(torch.sum(torch.square(self.stop[0][:,0:1,:])) - 1e-16)
        stop_f2 = F.relu(torch.sum(torch.square(self.stop[1][:,0:1,:])) - 1e-16)
        self.unweighted_losses_f1['stop'] = stop_f1
        self.unweighted_losses_f2['stop'] = stop_f2
    
    #Loss to force specified positions
    def update_force_loss(self):
        match_f1 = F.relu(self.force_aa_f1_total-torch.sum(self.force_aa_f1*self.prot[0][:, :, self.force_aa_f1_indices])-1e-16)
        match_f2 = F.relu(self.force_aa_f2_total-torch.sum(self.force_aa_f2*self.prot[1][:, :, self.force_aa_f2_indices])-1e-16)
        self.unweighted_losses_f1['force'] = match_f1
        self.unweighted_losses_f2['force'] = match_f2
        
    #Loss to specify last codon must be a stop
    def update_last_loss(self):    
        last_stop_f1 = F.relu(1.0-torch.sum(self.last_aa[0]*self.stop_onehot)-1e-16)
        last_stop_f2 = F.relu(1.0-torch.sum(self.last_aa[1]*self.stop_onehot)-1e-16)
        self.unweighted_losses_f1['last'] = last_stop_f1
        self.unweighted_losses_f2['last'] = last_stop_f2

    #Calculate and update all losses
    def update_loss_all(self, kl_alpha=1e8):
        self.update_kl_loss()
        self.update_ce_loss()
        self.update_lddt_loss()
        self.update_rog_loss()
        self.update_surfnp_loss()
        self.update_nc_loss()
        self.update_stop_loss()
        self.update_force_loss()
        self.update_last_loss()
        
        #Loss for forced AA's and stop codons; scaled by length of protein
        self.nloss_f1 = torch.sum(torch.stack([ v*self.weights[k] for k, v in self.unweighted_losses_f1.items() if k in self.nloss_kw ])) / self.model.f1_length
        self.nloss_f2 = torch.sum(torch.stack([ v*self.weights[k] for k, v in self.unweighted_losses_f2.items() if k in self.nloss_kw ])) / self.model.f2_length
        
        #Hallucination losses
        hloss_f1 = [ self.unweighted_losses_f1[k] * self.weights[k] for k in self.hloss_kw ]
        hloss_f2 = [ self.unweighted_losses_f2[k] * self.weights[k] for k in self.hloss_kw ]
        self.hloss_f1 = torch.sum(torch.stack(hloss_f1))
        self.hloss_f2 = torch.sum(torch.stack(hloss_f2))
        
        #Aux losses
        aloss_f1 = torch.sum(torch.stack([ self.unweighted_losses_f1[k] * self.weights[k] for k in self.aloss_kw ]))
        aloss_f2 = torch.sum(torch.stack([ self.unweighted_losses_f2[k] * self.weights[k] for k in self.aloss_kw ]))
        aloss_f1_iw = aloss_f1.detach() / self.aloss_max if aloss_f1 > self.aloss_max else 1.0
        aloss_f2_iw = aloss_f2.detach() / self.aloss_max if aloss_f2 > self.aloss_max else 1.0
        self.aloss_f1 = aloss_f1 / aloss_f1_iw
        self.aloss_f2 = aloss_f2 / aloss_f2_iw
        
        #Weighted average factor to penalize against big difference in loss between two proteins
        lower_kl = -1.0 * torch.max(torch.stack([self.unweighted_losses_f1['kl'], self.unweighted_losses_f2['kl']])).detach()
        if lower_kl >= kl_alpha: #Disable penalty for large difference in loss between the two proteins if worse KL divergence is above the threshold
            alpha = 0.5
        else:
            hloss_diff = torch.std(torch.stack([self.hloss_f1, self.hloss_f2])).detach()
            swfactor = torch.tanh(hloss_diff * 2.0)
            alpha = self.alpha_max * swfactor + 0.5 if swfactor < 1.0 else 1.0     

        if not self.no_f2:
            self.hloss = 0.5 * (alpha * torch.max(torch.stack([self.hloss_f1, self.hloss_f2])) + (1 - alpha) * torch.min(torch.stack([self.hloss_f1, self.hloss_f2])))
            self.aloss = 0.5 * (self.aloss_f1 + self.aloss_f2)
            self.nloss = 0.5 * (self.nloss_f1 + self.nloss_f2)
        else:
            self.hloss = self.hloss_f1
            self.aloss = self.aloss_f1
            self.nloss = self.nloss_f1
        
        self.haloss = self.hloss + self.aloss
        self.total_loss = self.nloss + self.haloss #Total loss
    
    #Grab losses before scaling and weighting
    def get_detached_unweighted_losses(self):
        loss_dict = { k: (self.unweighted_losses_f1[k].detach().clone(), self.unweighted_losses_f2[k].detach().clone()) for k in self.unweighted_losses_f1.keys() }
        return loss_dict
    
    #Grab current sequences
    def get_detached_seqs(self):
        params = self.model.state_dict()        
        seqs_dict = {'weight': { k: params[k].detach().clone() for k in ['dense.weight', 'instance_norm.weight', 'instance_norm.bias']},
                     'nuc': self.model.layers['sampled'].detach().clone(),
                     'prot': (self.model.layers['prot_slice1'].detach().clone(),
                              self.model.layers['prot_slice2'].detach().clone()),
                     'stop': (self.model.layers['stop_slice1'].detach().clone(), 
                              self.model.layers['stop_slice2'].detach().clone()),
                     'last': (self.model.layers['last_slice1'].detach().clone(), 
                              self.model.layers['last_slice2'].detach().clone())}
        return seqs_dict
    
    #Grab LDDT and distogram for plotting purpose
    def get_detached_dists(self):
        pred_dict = {'lddt': (self.model.layers['pred'][0]['lddt'].detach().clone(), 
                              self.model.layers['pred'][1]['lddt'].detach().clone()),
                     'dist_argmax': (self.c6d_dict_f1['p_dist'].argmax(3)[0],
                                     self.c6d_dict_f2['p_dist'].argmax(3)[0])}
        return pred_dict

#Make background for KL divergence loss
def mk_bkg(Net, L, device, L2=0, gap_len=None, n_runs=100, net_kwargs={}):
    logits = {k:[] for k in ['dist','omega','theta','phi']}
    
    joined_length = L + L2
    
    if L2 > 0:
        idx_pdb = torch.arange(joined_length).long().view(1, joined_length).to(device)
        idx_pdb[:, L2:] += gap_len

    with torch.no_grad():
        for i in range(n_runs):
            inpt_cat = torch.randint(20, [1, 1, joined_length])  #No gaps in background calculation
            msa_one_hot = F.one_hot(inpt_cat, 21).type(torch.float32)
            input_seq = msa_one_hot[:,0]
            if L2 > 0: #For gapped interaction prediction
                out = Net(torch.argmax(msa_one_hot, axis=-1).to(device), 
                          seq=input_seq, idx=idx_pdb,
                          msa_one_hot=msa_one_hot.to(device), 
                          **net_kwargs)
            else:
                out = Net(torch.argmax(msa_one_hot, axis=-1).to(device), 
                          msa_one_hot=msa_one_hot.to(device), 
                          **net_kwargs)
            for k, v in logits.items():
                v.append(out[k].permute([0,2,3,1]))    

    logits = {k: torch.stack(v, axis=0).mean(0) for k, v in logits.items()} #Average of predictions on N random inputs
    probs = {k: F.softmax(v, dim=3) for k, v in logits.items()}
    
    return probs
    
#For formatting output
def get_c6d_dict(out, grad=True):
    C6D_KEYS = ['dist','omega','theta','phi']
    if grad:
        logits = [out[key].float() for key in C6D_KEYS]
    else:
        logits = [out[key].float().detach() for key in C6D_KEYS]
    probs = [F.softmax(l, dim=1) for l in logits]
    dict_pred = {}
    dict_pred['p_dist'] = probs[0].permute([0,2,3,1])
    dict_pred['p_omega'] = probs[1].permute([0,2,3,1])
    dict_pred['p_theta'] = probs[2].permute([0,2,3,1])
    dict_pred['p_phi'] = probs[3].permute([0,2,3,1])
    return dict_pred, probs    

#Calculates cross entropy loss of C6D
def entropy(p, mask, eps=1e-16):
    S_ij = -(p * torch.log(p + eps)).sum(axis=-1)
    S_ave = torch.sum(mask * S_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
    return S_ave

#From RFDesign; dist_bins=16 excludes >20A; beta modulates sharpness (logit multiplier)
def get_entropy_loss(net_out, mask=None, beta=10.0, dist_bins=-1, eps=1e-16):
    pd = torch.softmax(torch.log(beta * net_out['c6d']['p_dist'][...,:dist_bins] + eps), axis = -1)
    po = torch.softmax(torch.log(beta * net_out['c6d']['p_omega'][...,:36] + eps), axis = -1)
    pt = torch.softmax(torch.log(beta * net_out['c6d']['p_theta'][...,:36] + eps), axis = -1)
    pp = torch.softmax(torch.log(beta * net_out['c6d']['p_phi'][...,:18] + eps), axis = -1)

    #Entropy loss    
    S_d = entropy(pd, mask)
    S_o = entropy(po, mask)
    S_t = entropy(pt, mask)
    S_p = entropy(pp, mask)
    loss = torch.mean(torch.stack((S_d, S_o, S_t, S_p)))

    return loss

#Calculates KL divergence loss of C6D vs. background
def kl(p, q, mask, eps=1e-16):
    p_clamp = torch.clamp(p, eps, 1-eps)
    q_clamp = torch.clamp(q, eps, 1-eps)
    kl_ij = (p_clamp * torch.log(p_clamp/q_clamp)).sum(-1)
    kl_ave = (mask * kl_ij).sum((1,2)) / (mask.sum((1,2)) + eps)
    return kl_ave

def get_kl_loss(net_out, bkg, mask=None, eps=1e-16):
    kl_d = kl(net_out['c6d']['p_dist'], bkg['dist'], mask)
    kl_o = kl(net_out['c6d']['p_omega'], bkg['omega'], mask)
    kl_t = kl(net_out['c6d']['p_theta'], bkg['theta'], mask)
    kl_p = kl(net_out['c6d']['p_phi'], bkg['phi'], mask)
    loss = torch.mean(torch.stack((kl_d, kl_o, kl_t, kl_p)))
    return loss

#Radius of gyration loss; from RFDesign but with ReLU thresholding
def get_rog_loss(pred_xyz):#, thres=None):
    ca_xyz = pred_xyz[:,:,1]
    sq_dist = torch.pow(ca_xyz - ca_xyz.mean(1),2).sum(-1).mean(-1)
    rog = sq_dist.sqrt()
    return rog
    #loss = F.relu(rog - thres)
    #return loss
    
def get_pair_dist(a, b):
    dist = torch.cdist(a, b, p=2)
    return dist

#Dihedral func; from RFDesign
def get_dih(a, b, c, d):
    b0  = a - b
    b1r = c - b
    b2  = d - c

    b1 = b1r/torch.norm(b1r, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)
    ang = torch.atan2(y, x)

    return ang

#Angle func; from RFDesign
def get_ang(a, b, c):
    v = a - b
    w = c - b
    v = v / torch.norm(v, dim=-1, keepdim=True)
    w = w / torch.norm(w, dim=-1, keepdim=True)
    
    y = torch.norm(v-w,dim=-1)
    x = torch.norm(v+w,dim=-1)
    ang = 2*torch.atan2(y, x)
    
    return ang

#Convert cartesian coordinates to 2d dist map; from RFDesign
def xyz_to_c6d_smooth(xyz, params):
    batch = xyz.shape[0]
    nres = xyz.shape[2]

    # three anchor atoms
    N  = xyz[:,0]
    Ca = xyz[:,1]
    C  = xyz[:,2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    dist = get_pair_dist(Cb,Cb)
    mask = (dist + 999.9*torch.eye(nres,device=xyz.device)[None,...])<params['DMAX']
    b,i,j = torch.where(mask)

    omega = torch.zeros([batch,nres,nres],dtype=xyz.dtype,device=xyz.device)
    theta = torch.zeros([batch,nres,nres],dtype=xyz.dtype,device=xyz.device)
    phi = torch.zeros([batch,nres,nres],dtype=xyz.dtype,device=xyz.device)

    omega[b,i,j] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    theta[b,i,j] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    phi[b,i,j] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    return torch.stack([dist, omega, theta, phi],dim=-1), mask
    
#n neighbors, from RFDesign
def n_neighbors(pred_xyz, n=1, m=9, a=0.5, b=2):
    '''
        n :   distance exponent
        m :   distance falloff midpoint
        a :   offset that controls the angular width of cone
        b :   angular sharpness exponent
    '''

    c6d, mask = xyz_to_c6d_smooth(pred_xyz.permute(0,2,1,3), {'DMAX':20.0})

    dist = c6d[...,0]
    phi = c6d[...,3]

    f_dist = 1 / (1 + torch.exp(n * (dist * mask - m)))
    f_ang = ((torch.cos(np.pi - phi * mask) + a) / (1 + a)) ** b
    n_nbr = torch.nansum(f_dist * f_ang * mask, axis=2)
    
    return n_nbr
    
#Surface non-polar loss; from RFDesign but with ReLU thresholding
def get_surfnp_loss(net_out, pinput, nonpolar = 'VILMWF', nbr_thres=2.5, target_polar=0.2):
    i_nonpolar = [ AA_1_N[a] for a in nonpolar ]
    n_nbrs = n_neighbors(net_out['xyz'])
    surface = 1 - torch.sigmoid(n_nbrs - nbr_thres)
    surf_nonpol = torch.square(pinput[:,0,:][..., i_nonpolar]).sum(-1) * surface
    net_polar = surf_nonpol.sum(-1)[0] / surface.sum(-1)[0]
    
    loss = F.relu(net_polar - target_polar - 1e-16)
    return loss
   
#Charged residues loss; from RFDesign but with ReLU thresholding
def get_nc_loss(pinput, target_charge=0): 
    i_pos = [ AA_1_N[a] for a in 'KR' ]
    i_neg = [ AA_1_N[a] for a in 'ED' ]
    pos = torch.square(pinput[:,0,:][...,i_pos]).sum(-1)
    neg = torch.square(pinput[:,0,:][...,i_neg]).sum(-1)
    net_charge = (pos - neg).sum(-1)[0]
    loss = F.relu(net_charge - target_charge - 1e-16)
    
    return loss

#For annealing in simulated annealing
def anneal(loss, loss_perturbed, tau):    
    if loss_perturbed < loss: #If decreases loss, always accept
        return loss_perturbed, True
    else: #If increases loss, accept with a probability modulated by temperature
        ap = torch.exp((loss - loss_perturbed) / tau) #Acceptance probability
        if ap > np.random.rand():
            return loss_perturbed, True
        else:
            return loss, False #Don't accept

#Main simulated annealing iterator function
def optimize_SA(model, device, n_step, alpha_max,
                mask_f1, mask_f2, bkg_f1, bkg_f2,
                force_aa_f1=None, force_aa_f2=None, no_f2=False,
                weight_kl=1.0, weight_ce=1.0, weight_lddt=1.0,
                weight_stop=5.0, weight_force=20.0, weight_last=10.0,
                weight_rog=1.0, weight_rog_sum=1.0, rog_thres=15.0, rog_sum_thres=28.0,
                weight_surfnp=1.0, surfnp_thres=0.2,
                weight_nc=1.0, nc_thres=0.0,
                max_mut=1, tau0=1e-2, anneal_rate=1e-8, min_temp=1e-4,
                min_steps=500, patience=100, early_stop_thres=0.01, aloss_max=1e8, kl_alpha=1.0,
                print_loss=True):
    
    loss = SeqOptNLoss(model, device, alpha_max,
                       mask_f1, mask_f2, bkg_f1, bkg_f2, 
                       force_aa_f1, force_aa_f2, no_f2,
                       weight_kl, weight_ce, weight_lddt,
                       weight_stop, weight_force, weight_last, 
                       weight_rog, weight_rog_sum, rog_thres, rog_sum_thres,
                       weight_surfnp, surfnp_thres,
                       weight_nc, nc_thres, aloss_max)

    #Things to keep track of across optimization steps
    trk_dict = {'min_step': [],
                'grad': [],
                'seq': [],
                'unweighted_loss': [],
                'plot': [],
                'params': None}
    
    #Save parameters for logging purpose
    trk_dict['params'] = {
        'nuc_length': model.seq_length,
        'f1_length': model.f1_length,
        'f2_length': model.f2_length,
        'offset': model.offset,
        'f1': model.f1,
        'f2': model.f2,
        'gap_length': model.gap_len,
        'weights': {'kl': weight_kl,
                     'ce': weight_ce,
                     'lddt': weight_lddt,
                     'stop': weight_stop,
                     'force': weight_force,
                     'last': weight_last,
                     'rog': weight_rog,
                     'rog_sum': weight_rog_sum,
                     'surfnp': weight_surfnp,
                     'nc': weight_nc},
        'n_step': n_step,
        'f1_force': force_aa_f1.detach().clone(),
        'f2_force': force_aa_f2.detach().clone(),
        'last_stop': model.last_stop,
        'no_f2': no_f2,
        'alpha_max': alpha_max,
        'thres': {'rog': rog_thres,
                  'rog_sum': rog_sum_thres,
                  'surfnp': surfnp_thres,
                  'nc': nc_thres},
        'tau0': tau0,
        'anneal_rate': anneal_rate,
        'min_temp': min_temp,
        'max_mut': max_mut,
        'min_steps': min_steps,
        'patience': patience,
        'early_stop_thres': early_stop_thres,
        'aloss_max': aloss_max,
        'kl_alpha': kl_alpha }
    
    tau = tau0 #Initial temperature
    
    with torch.no_grad():
        loss.update_model() #Run one step and get initial losses
    loss.update_loss_all()
    last_loss = loss.total_loss.detach().clone() #Will hold loss of last accepted sequence
    trk_dict['seq'] += [ loss.get_detached_seqs() ]
    trk_dict['unweighted_loss'] += [ loss.get_detached_unweighted_losses() ]
    trk_dict['plot'] += [ loss.get_detached_dists() ]
    trk_dict['min_step'] += [ 0 ]
    last_min_st = str(loss)
    
    params = model.state_dict()
    current_min_loss = last_loss
    best_avg_kl = 0
    num_bad_steps = 0
    for i in range(1, n_step):
        sa_params = params['dense.weight']
        cloned_params = sa_params.clone() #Save current weights to return to if mutation is not accepted
        num_mut = torch.randint(1, max_mut+1, (1, 1))[0] #Randomly choose number of mutations < max_mut
        reshaped_params = torch.reshape(sa_params, (1, model.onehot_dim, model.seq_length))
        chosen_indices = torch.randint(0, model.seq_length, (1, num_mut))[0] #Choose random indices for N number of mutations to perform
        shuffle_indices = torch.randperm(model.onehot_dim) #TODO: this should be multiple for each position. But currently running only num_mut=1
        reshaped_params[:, :, chosen_indices] = torch.stack([ reshaped_params[:, shuffle_indices, chosen_indices[j]] for j in range(chosen_indices.shape[0]) ], 2) #Perform mutation
        new_params = torch.reshape(reshaped_params, (sa_params.shape[0], 1))
        sa_params.copy_(torch.reshape(reshaped_params, (sa_params.shape[0], 1))) #Update weights

        tau = np.maximum(tau0 * np.exp(-anneal_rate * i), min_temp) #New temperature based on temperature decay parameters

        with torch.no_grad():
            loss.update_model() #Run thru predictor without backward pass
        
        loss.update_loss_all(kl_alpha) #Calculate losses and track
        trk_dict['seq'] += [ loss.get_detached_seqs() ]
        trk_dict['unweighted_loss'] += [ loss.get_detached_unweighted_losses() ]
        trk_dict['plot'] += [ loss.get_detached_dists() ]
        
        final_loss, is_swapped = anneal(last_loss, loss.total_loss, tau) #Annealing function to accept/reject mutation
        
        #Additional acceptability; not accetable if stop codons are present or forced AA's are not matching (weighted)
        acceptable = loss.nloss < 1e-16
        if is_swapped and acceptable: #Updated if accepted by annealing function
            last_loss = loss.total_loss.detach().clone()
            if (last_loss < current_min_loss):
                current_min_loss = last_loss
                trk_dict['min_step'] += [ i ]
                last_min_st = str(loss)
        else: #Otherwise return weights to original
            sa_params.copy_(cloned_params)
        
        #Print current losses
        if print_loss:
            if ipy:
                clear_output(wait=True)
                
            if i > 1:
                n_line = print_str.count('\n') + 1
                up_str = "\x1B[" + str(n_line) + "A"
                print_str = up_str
                print_str += f'{"SA step:":<8}{i:>8}{CLR}\n' + f'{"---":<8}{"P1":^8}{"P2":^8}{CLR}\n' + str(loss)
            else:
                print_str = f'{"SA step:":<8}{i:>8}{CLR}\n' + f'{"---":<8}{"P1":^8}{"P2":^8}{CLR}\n' + str(loss)
            
            if len(trk_dict['min_step']) > 0:
                print_str += f'---{CLR}\nLast acceptable minimum{CLR}\n'
                min_step = trk_dict['min_step'][-1]
                print_str += f'{"Last min step:":<8}{min_step:>8}{CLR}\n' + f'{"---":<8}{"P1":^8}{"P2":^8}{CLR}\n'
                print_str += last_min_st
            
            print(print_str)
        
        #Early stop criteria on plateau; uses average KLD of two proteins
        if i >= min_steps:
            current_avg_kl = -0.5 * (loss.unweighted_losses_f1['kl'] + loss.unweighted_losses_f2['kl']) if not no_f2 else (-1.0 * loss.unweighted_losses_f1['kl'])
            
            if current_avg_kl > (best_avg_kl + early_stop_thres):
                best_avg_kl = current_avg_kl
                num_bad_steps = 0
            else:
                num_bad_steps += 1
                
            if num_bad_steps > patience:
                trk_dict['params']['n_step'] = i + 1
                break
        
    return trk_dict

#Main gradient descent iterator function
def optimize_GD(model, device, n_step, n_step_h, n_max_h, alpha_max,
                mask_f1, mask_f2, bkg_f1, bkg_f2,
                force_aa_f1=None, force_aa_f2=None, no_f2=False,
                weight_kl=1.0, weight_ce=1.0, weight_lddt=1.0,
                weight_stop=5.0, weight_force=20.0, weight_last=10.0,
                weight_rog=1.0, weight_rog_sum=1.0, rog_thres=15.0, rog_sum_thres=28.0,
                weight_surfnp=1.0, surfnp_thres=0.2,
                weight_nc=1.0, nc_thres=0.0,
                early_gd_stop=0.5, reset_kl=0.5, reset_step=100, max_total_step=1000,
                lr0=0.1, betas=(0.5, 0.5), eps=1e-3,
                lookahead_k=10, lookahead_alpha=0.5,
                decay_kl_threshold=0.1, decay_min_lr=0.0005, decay_patience=100, decay_factor_lr=0.1,
                min_steps=500, patience=100, early_stop_thres=0.01, aloss_max=0.2, reset_kl_final=0.7, kl_alpha=0.7,
                print_loss=True):
    
    #Model parameter, i.e. nucleotide "logits"
    opt_params = [ i for i in model.dense.parameters() ]
    
    def reset_opt(lr, weight=True, optimizer=True):
        if weight:
            torch.nn.init.xavier_normal_(model.dense.weight) #glorot initialization
        if optimizer:
            optimizer_ = torch.optim.Adam(opt_params, lr=lr, betas=betas, eps=eps)
            optimizer = optim.Lookahead(optimizer_, k=lookahead_k, alpha=lookahead_alpha) #Lookahead outer optimizer wrapping the inner GD optimizer
        return optimizer

    optimizer = reset_opt(lr0, weight=False, optimizer=True)

    loss = SeqOptNLoss(model, device, alpha_max,
                       mask_f1, mask_f2, bkg_f1, bkg_f2, 
                       force_aa_f1, force_aa_f2, no_f2,
                       weight_kl, weight_ce, weight_lddt,
                       weight_stop, weight_force, weight_last, 
                       weight_rog, weight_rog_sum, rog_thres, rog_sum_thres,
                       weight_surfnp, surfnp_thres,
                       weight_nc, nc_thres,
                       aloss_max)
    
    #Things to keep track of across optimization steps
    trk_dict = {'min_step': [],
                'grad': [],
                'seq': [],
                'unweighted_loss': [],
                'plot': [],
                'params': None,
                'decay_step': [],
                'reset_step': []}
    
    #Save parameters for logging purpose
    trk_dict['params'] = {
        'nuc_length': model.seq_length,
        'f1_length': model.f1_length,
        'f2_length': model.f2_length,
        'offset': model.offset,
        'f1': model.f1,
        'f2': model.f2,
        'gap_length': model.gap_len,
        'weights': {'kl': weight_kl,
                     'ce': weight_ce,
                     'lddt': weight_lddt,
                     'stop': weight_stop,
                     'force': weight_force,
                     'last': weight_last,
                     'rog': weight_rog,
                     'rog_sum': weight_rog_sum,
                     'surfnp': weight_surfnp,
                     'nc': weight_nc},
        'n_step': n_step,
        'n_step_h': n_step_h,
        'n_max_h': n_max_h,
        'early_stop_kl': early_gd_stop,
        'f1_force': force_aa_f1.detach().clone(),
        'f2_force': force_aa_f2.detach().clone(),
        'last_stop': model.last_stop,
        'no_f2': no_f2,
        'alpha_max': alpha_max,
        'thres': {'rog': rog_thres,
                  'rog_sum': rog_sum_thres,
                  'surfnp': surfnp_thres,
                  'nc': nc_thres},
        'opt_params': {'lr0': lr0, 
                       'betas': betas, 
                       'eps': eps, 
                       'lookahead_k': lookahead_k, 
                       'lookahead_alpha': lookahead_alpha,
                       'decay_kl_threshold': decay_kl_threshold, 
                       'decay_min_lr': decay_min_lr, 
                       'decay_patience': decay_patience, 
                       'decay_factor_lr': decay_factor_lr,
                       'reset_kl': reset_kl,
                       'reset_step': reset_step,
                       'reset_kl_final': reset_kl_final,
                       'max_total_step': max_total_step},
        'aloss_max': aloss_max,
        'kl_alpha': kl_alpha
    }
                
    #Gradient descent steps
    current_min_loss = 1e16
    early_stopping = False
    early_stop_kl = -1.0 * early_gd_stop
    h_kl = -1.0 * n_step_h
    h_counter = 0
    reset_kl_final_ = -1.0 * reset_kl_final
    
    #For LR decay
    best_avg_kl = 0.0
    best_avg_kl_thres = 0.0
    num_bad_steps = 0
    last_best_step = 0
    lr = lr0

    inner_i = 0
    for i in range(max_total_step):  
        loss.update_model() #Run model with current sequences
        loss.update_loss_all(kl_alpha) #And get losses
        
        #Track history of seqs and losses
        trk_dict['seq'] += [ loss.get_detached_seqs() ]
        trk_dict['unweighted_loss'] += [ loss.get_detached_unweighted_losses() ]
        trk_dict['plot'] += [ loss.get_detached_dists() ]
        
        #Zero out gradient if any nucleotides were swapped, otherwise we will accumulate gradient
        nts_changed = 0
        if i > 0:
            nts_changed = torch.sum(torch.abs(trk_dict['seq'][-1]['nuc'] - trk_dict['seq'][-2]['nuc']))
        if nts_changed > 0:
            optimizer.zero_grad()
            
        #Acceptability = stop codons and forced AA's
        acceptable = loss.nloss < 1e-16
        if (loss.haloss < current_min_loss): #If new solution has better loss than current best
            if acceptable: #And if it's acceptable
                current_min_loss = loss.haloss.detach().clone() #Only use hallucination loss for tracking current best solution
                trk_dict['min_step'] += [ i ]
                if print_loss: #For printing current losses
                    last_min_st = str(loss)
            elif (loss.unweighted_losses_f1['kl'] <= h_kl) and (loss.unweighted_losses_f2['kl'] <= h_kl): #If the new minimal solution was not acceptable and we are below some threshold for the minimal solution, then:
                loss.total_loss = loss.nloss + loss.haloss.detach() #Zero out gradient from hallucination loss so that it tries changing required fixed positions
                h_counter = n_max_h - 1 #We will do this for next ~n_max_h iterations
        elif h_counter > 0:
            if acceptable: 
                h_counter = 0
            else:
                loss.total_loss = loss.nloss + loss.haloss.detach()
                h_counter -= 1
 
        #Update gradient
        loss.update_gradient()
        trk_dict['grad'] += [ loss.get_detached_gradient(unnormed=True) ]
        
        #Update weights
        optimizer.step()

        #Print current losses
        if print_loss:
            if ipy:
                clear_output(wait=True)
                
            if i > 0:
                n_line = print_str.count('\n') + 1
                up_str = "\x1B[" + str(n_line) + "A"
                print_str = up_str 
                print_str += f'{"GD step:":<8}{i:>8}{CLR}\n' + f'{"---":<8}{"P1":^8}{"P2":^8}{CLR}\n' + str(loss)
            else:
                print_str = f'{"GD step:":<8}{i:>8}{CLR}\n' + f'{"---":<8}{"P1":^8}{"P2":^8}{CLR}\n' + str(loss)
               
            if i > 0:
                print_str += f'{"num mut:":<8}{nts_changed:>8}{CLR}\n'
                print_str += f'{"last best step:":<8}{last_best_step:>8}{CLR}\n'
                print_str += f'{"best avg KLD:":<8}{best_avg_kl:>8.4f}{CLR}\n'
                print_str += f'{"current LR:":<8}{lr:>8.4f}{CLR}\n'
            
            if len(trk_dict['min_step']) > 0:
                print_str += f'---{CLR}\nLast acceptable minimum{CLR}\n'
                min_step = trk_dict['min_step'][-1]
                print_str += f'{"Last min step:":<8}{min_step:>8}{CLR}\n' + f'{"---":<8}{"P1":^8}{"P2":^8}{CLR}\n'
                print_str += last_min_st
             
            print(print_str)
        
        #Early stopping or LR decay conditions
        if i > 0:            
            #Stop if maximum number of iterations reached
            if (inner_i == (n_step - 1)) or (i == (max_total_step - 1 )):
                trk_dict['params']['n_step'] = i + 1
                break
            
            #Condition 1: hard threshold - trigger early stopping if specified KL loss reached for both proteins
            if len(trk_dict['min_step']) > 0:
                if (trk_dict['min_step'][-1] == i) and ((loss.unweighted_losses_f1['kl'] <= early_stop_kl) and ((loss.unweighted_losses_f2['kl'] <= early_stop_kl) or no_f2)):
                    trk_dict['params']['n_step'] = i + 1
                    break

            #Track best average KL loss
            current_avg_kl = -0.5 * (loss.unweighted_losses_f1['kl'] + loss.unweighted_losses_f2['kl']) if not no_f2 else (-1.0 * loss.unweighted_losses_f1['kl'])  
            if current_avg_kl > best_avg_kl:
                best_avg_kl = current_avg_kl
                last_best_step = i
            
            #Condition 2: reset everything if specified average KL loss not reached by specified step number
            if (best_avg_kl < reset_kl) and (inner_i == reset_step):
                inner_i = 0
                current_min_loss = 1e16
                best_avg_kl = 0.0
                best_avg_kl_thres = 0.0
                num_bad_steps = 0
                last_best_step = 0
                lr = lr0
                optimizer = reset_opt(lr0, weight=True, optimizer=True)
                trk_dict['reset_step'] += [ i ]

            #Condition 3: detect plateau in average KL loss - no improved loss (by a threshold) in specified number of steps
            else:    
                if current_avg_kl > (best_avg_kl_thres + decay_kl_threshold):
                    best_avg_kl_thres = current_avg_kl
                    num_bad_steps = 0
                else:
                    num_bad_steps += 1

                #If plateau detected, then we will either decay LR, early stop or reset
                if (num_bad_steps > decay_patience):
                    lr = lr * decay_factor_lr #New LR

                    #If new LR will be lower than minimum decay, then early stop or reset
                    if (lr < decay_min_lr) or (decay_factor_lr == 1.0):
                        #Reset if specified KL loss not reached for both proteins
                        min_kl_f1 = trk_dict['unweighted_loss'][trk_dict['min_step'][-1]]['kl'][0]
                        min_kl_f2 = trk_dict['unweighted_loss'][trk_dict['min_step'][-1]]['kl'][1]
                        if (min_kl_f1 >= reset_kl_final_) or (min_kl_f2 >= reset_kl_final_): 
                            inner_i = 0
                            current_min_loss = 1e16
                            best_avg_kl = 0.0
                            best_avg_kl_thres = 0.0
                            num_bad_steps = 0
                            last_best_step = 0
                            lr = lr0
                            optimizer = reset_opt(lr0, weight=True, optimizer=True)
                            trk_dict['reset_step'] += [ i ]
                        else: #Otherwise early stop
                            trk_dict['params']['n_step'] = i + 1
                            break

                    #Decay LR
                    elif (lr >= decay_min_lr) and (h_counter == 0) and (decay_factor_lr < 1.0):
                        #Set weights to last best step
                        model.set_weights(trk_dict['seq'][last_best_step]['weight'])

                        #Reset optimizer with lower LR
                        optimizer = reset_opt(lr, weight=False, optimizer=True)
                        num_bad_steps = 0
                        trk_dict['decay_step'] += [ i ]        

        inner_i += 1
        
    return trk_dict

#Wrapper function to run design loops
def run_design(device, net_rosetta, net_translator,
               network_name,
               total_length, f1_frame, f2_frame, offset,
               f1_force, f2_force, bkg_f1, bkg_f2, mask_f1, mask_f2, 
               last_stop=True, no_f2=False,
               lr=0.1, betas=(0.5, 0.9), eps=1e-3,
               lookahead_k=5, lookahead_alpha=0.5,
               n_step_gd=500, n_step_gd_n=400, n_max_h=5, 
               early_gd_stop=0.5, reset_kl=0.5, reset_step=100, max_total_step=1000,
               n_step_sa=500, alpha_gd=0.5, alpha_sa=0.75,
               weight_kl_gd=2.0, weight_ce_gd=0.0, weight_lddt_gd=1.0,
               weight_stop_gd=1.0, weight_force_gd=5.0, weight_last_gd=2.5,
               weight_rog_gd=1.0, weight_rog_sum_gd=1.0, rog_thres_gd=15.0, rog_sum_thres_gd=28.0,
               weight_surfnp_gd=1.0, surfnp_thres_gd=0.2,
               weight_nc_gd=1.0, nc_thres_gd=0.0,
               weight_kl_sa=2.0, weight_ce_sa=0.0, weight_lddt_sa=1.0,
               weight_stop_sa=1.0, weight_force_sa=5.0, weight_last_sa=2.5,
               weight_rog_sa=1.0, weight_rog_sum_sa=1.0, rog_thres_sa=15.0, rog_sum_thres_sa=28.0,
               weight_surfnp_sa=1.0, surfnp_thres_sa=0.2,
               weight_nc_sa=1.0, nc_thres_sa=0.0,
               max_mut=1, tau0=1e-2, anneal_rate=2e-3, min_temp=1e-3, start_weights=None, 
               decay_kl_threshold=0.1, decay_min_lr=0.0005, decay_patience=100, decay_factor_lr=0.1,
               min_steps=500, patience=100, early_stop_thres=0.01, aloss_max_gd=0.2, aloss_max_sa=1e8, reset_kl_final=0.7,
               kl_alpha_gd=0.7, kl_alpha_sa=1.0,
               print_loss=True):
    
    f1_length = mask_f1.shape[1]
    f2_length = mask_f2.shape[1]
    
    #Set up model to optimize sequence
    pm = SeqOptN(device, network_name, total_length, offset, f1_length, f2_length, 
                 net_translator, net_rosetta, [f1_frame, f2_frame], last_stop)
    
    #Eval mode; this should zero dropouts
    pm.translator.eval()
    pm.predictor.eval()
                
    #Disable saving grad on rosetta and translator
    for p in pm.predictor.parameters():
        p.requires_grad_(False) 
    for p in pm.translator.parameters():
        p.requires_grad_(False) 
    
   #Set weights to pre-loaded state if provided
    if start_weights is not None:
        pm.set_weights(start_weights)

    #Run gradient descent
    res_gd = optimize_GD(pm, device, n_step_gd, n_step_gd_n, n_max_h, alpha_gd,
                         mask_f1, mask_f2, bkg_f1, bkg_f2, f1_force, f2_force, no_f2,
                         weight_kl_gd, weight_ce_gd, weight_lddt_gd,
                         weight_stop_gd, weight_force_gd, weight_last_gd,
                         weight_rog_gd, weight_rog_sum_gd, rog_thres_gd, rog_sum_thres_gd,
                         weight_surfnp_gd, surfnp_thres_gd, 
                         weight_nc_gd, nc_thres_gd,
                         early_gd_stop, reset_kl, reset_step, max_total_step,
                         lr, betas, eps, 
                         lookahead_k, lookahead_alpha,
                         decay_kl_threshold, decay_min_lr, decay_patience, decay_factor_lr, aloss_max_gd, reset_kl_final, kl_alpha_gd,
                         print_loss)
    
    #Set weights to last acceptable minimum in GD
    pm.set_weights(res_gd['seq'][res_gd['min_step'][-1]]['weight'])
    
    #Run simulated annealing
    res_sa = optimize_SA(pm, device, n_step_sa, alpha_sa,
                         mask_f1, mask_f2, bkg_f1, bkg_f2, f1_force, f2_force, no_f2,
                         weight_kl_sa, weight_ce_sa, weight_lddt_sa,
                         weight_stop_sa, weight_force_sa, weight_last_sa,
                         weight_rog_sa, weight_rog_sum_sa, rog_thres_sa, rog_sum_thres_sa,
                         weight_surfnp_sa, surfnp_thres_sa, 
                         weight_nc_sa, nc_thres_sa, 
                         max_mut, tau0, anneal_rate, min_temp, 
                         min_steps, patience, early_stop_thres, aloss_max_sa, kl_alpha_sa,
                         print_loss)

    result = (res_gd, res_sa)
        
    return result

#Function to load RosettaFold network; taken from RFDesign
def load_model(include_dir, network_name, weights_dir, device):
    reg_models = json.load(open(include_dir+"/models/models.json"))
    sel = reg_models[network_name]
    chks = [weights_dir+'/'+sel['weights_path']+'/'+chk for chk in sel['checkpoints']]

    module, method = sel['code_path'].rsplit('.',1)
    sys.path.insert(0, include_dir+'/'+module.rsplit('.',1)[0].replace('.','/'))
    module = importlib.import_module(module)
    NetClass = getattr(module, method)

    net_params = json.load(open(weights_dir+'/'+sel['weights_path']+'/'+sel['params_path']))
    
    Net = NetClass(**net_params)
    weights = torch.load(chks[0], map_location=torch.device(device))
    if 'model_state_dict' in weights.keys():
        weights = weights['model_state_dict']
    Net.load_state_dict(weights, strict=False)

    Net = Net.to(device)
    
    return Net, net_params

#Function for visualizing optimization run
def anim_res(use_res, plot_interval=1, frame_interval=50):
    plt.style.use('default')
    matplotlib.rcParams.update({'font.size': 10})
    
    #Total number of steps GD and SA combined
    total_steps = len(use_res[0]['seq']) + len(use_res[1]['seq'])#use_res[1]['params']['n_step']

    #Combine trajectories from GD and SA
    all_loss = {}
    all_seq = {}
    all_plot = {}
    all_grad = {}

    for ck in use_res[0]['unweighted_loss'][0].keys():
        all_loss[ck] = [ v for d in use_res for x in d['unweighted_loss'] for k, v in x.items() if k == ck ]
        all_loss[ck] = (torch.stack([ l[0] for l in all_loss[ck]]).cpu().numpy(), torch.stack([ l[1] for l in all_loss[ck]]).cpu().numpy())

    for ck in use_res[0]['seq'][0].keys():
        all_seq[ck] = [ v for d in use_res for x in d['seq'] for k, v in x.items() if k == ck ]
        if ck == "weight":
            continue
        elif ck == "nuc": 
            all_seq[ck] = torch.stack(all_seq[ck]).squeeze(1).cpu().numpy()
        else:
            all_seq[ck] = (torch.stack([ s[0] for s in all_seq[ck] ]).squeeze(1).cpu().numpy(), torch.stack([ s[1] for s in all_seq[ck] ]).squeeze(1).cpu().numpy())

    for ck in use_res[0]['plot'][0].keys():
        all_plot[ck] = [ v for d in use_res for x in d['plot'] for k, v in x.items() if k == ck ]
        all_plot[ck] = (torch.stack([ s[0] for s in all_plot[ck] ]).squeeze(1).cpu().numpy(), torch.stack([ s[1] for s in all_plot[ck] ]).squeeze(1).cpu().numpy())

    all_grad = [ x for d in use_res for x in d['grad'] ]
    all_grad = torch.stack(all_grad).squeeze(1).cpu().numpy()

    #Plotting setup
    layout = [
        [ "orf", "orf", "orf", "orf", "orf" ],
        [ "nuc_seq", "nuc_seq", "nuc_seq", "nuc_seq", "nuc_seq" ],
        [ "grad", "grad", "grad", "grad", "grad" ],
        [ "text", "text", "f1_seq", "f1_dist", "f2_dist"],
        [ "text", "text", "f2_seq", "f1_dist", "f2_dist"],
        [ "net_loss", "net_loss", "f1_lddt", "f1_dist_min", "f2_dist_min" ],
        [ "gradnorm", "gradnorm", "f2_lddt", "f1_dist_min", "f2_dist_min" ]]

    f = plt.figure(constrained_layout=True, figsize=(19, 12))
    layout_axd = f.subplot_mosaic(layout, empty_sentinel="X", gridspec_kw={
        'width_ratios': [0.1, 0.05, 0.25, 0.15, 0.15],
        'height_ratios': [0.1, 0.1, 0.15, 0.25, 0.25, 0.25, 0.25]})

    #Assuming loss weights and constraints are the same for GD and SA
    gd_params = use_res[0]['params']
    sa_params = use_res[1]['params']
    f1_dir = 1.0 if gd_params['f1'] < 3 else -1
    f2_dir = 1.0 if gd_params['f2'] < 3 else -1
    f1_frame = gd_params['f1'] % 3
    f2_frame = gd_params['f2'] % 3
    f1_offset = f1_frame
    f2_offset = (gd_params['offset'] * 3) + f2_frame
    f1_len = gd_params["f1_length"]
    f2_len = gd_params["f2_length"]
    nuc_len = gd_params["nuc_length"]
    stop_offset = 3.0 if gd_params["last_stop"] else 0.0

    gdstep = gd_params["n_step"]
    sastep = total_steps - gdstep

    #For showing ORF orientation/arrangement
    orf_f1_plot = layout_axd["orf"].arrow(
        f1_offset-0.5 if f1_dir==1 else (f1_offset+f1_len*3)+stop_offset-0.5, 0.7,
        f1_dir*(f1_len*3+0.5+stop_offset), 0, 
        width=0.15, length_includes_head=True, 
        head_starts_at_zero=False, head_length=f1_len*0.075, facecolor="red")
    orf_f2_plot = layout_axd["orf"].arrow(
        f2_offset-0.5 if f2_dir==1 else (f2_offset+f2_len*3)+stop_offset-0.5, 0.3,
        f2_dir*(f2_len*3+0.5+stop_offset), 0, 
        width=0.15, length_includes_head=True, 
        head_starts_at_zero=False, head_length=f1_len*0.075, facecolor="blue")
    layout_axd["orf"].set_ylim([0, 1])
    layout_axd["orf"].get_yaxis().set_visible(False)
    layout_axd["orf"].set_title("ORF", loc='left')    
    layout_axd["orf"].set_xlim([-0.5, nuc_len-0.5])

    #For showing nucleotide sequence one-hot matrix
    bw_cmap = colors.ListedColormap(['white', 'black'])
    bw_norm = colors.BoundaryNorm([0.0, 0.5, 1.0], bw_cmap.N)

    nuc_plot = layout_axd["nuc_seq"].imshow(all_seq["nuc"][0], cmap=bw_cmap, norm=bw_norm)
    layout_axd["nuc_seq"].get_yaxis().set_visible(False)
    layout_axd["nuc_seq"].set_title("Nucleotide sequence", loc='left')
    layout_axd["nuc_seq"].set_ylim([-0.5, 3.5])
    layout_axd["nuc_seq"].sharex(layout_axd["orf"])

    #For showing gradients
    grad_normfunc = colors.CenteredNorm(vcenter=0.0, halfrange=np.percentile(np.abs(all_grad[0]), 99), clip=True) #Normalize gradient to 99 percentile for plotting
    grad_plot = layout_axd["grad"].imshow(grad_normfunc(all_grad[0]), cmap="magma", vmin=0, vmax=1)
    layout_axd["grad"].get_yaxis().set_visible(False)
    layout_axd["grad"].set_title("Gradient", loc='left')
    layout_axd["grad"].set_ylim([-0.5, 3.5])
    layout_axd["grad"].sharex(layout_axd["orf"])

    #For showing protein distograms
    f1_dist_plot = layout_axd["f1_dist"].imshow(all_plot["dist_argmax"][0][0], cmap="Reds", vmin=0, vmax=36)   
    layout_axd["f1_dist"].set_title("Protein 1 distogram", loc='left')
    
    f2_dist_plot = layout_axd["f2_dist"].imshow(all_plot["dist_argmax"][0][1], cmap="Reds", vmin=0, vmax=36)  
    layout_axd["f2_dist"].set_title("Protein 2 distogram", loc='left')
    
    #Concatenate last AA to protein sequence one-hot matrix
    f1_last = all_seq["last"][0]
    f2_last = all_seq["last"][1]
    f1_seq = np.concatenate((all_seq["prot"][0], np.expand_dims(f1_last, 2)), 2)
    f2_seq = np.concatenate((all_seq["prot"][1], np.expand_dims(f2_last, 2)), 2)

    #For protein 1
    f1_seq_plot = layout_axd["f1_seq"].imshow(f1_seq[0], cmap=bw_cmap, norm=bw_norm)
    layout_axd["f1_seq"].get_yaxis().set_visible(False)
    layout_axd["f1_seq"].set_title("Protein 1 sequence", loc='left')
    layout_axd["f1_seq"].set_ylim([-0.5, 20.5])

    #Draw rectangle overlays to mark stop codons for protein 1
    f1_seq_stop = all_seq["prot"][0][0][20]
    f1_seq_stop = f1_seq_stop * -1.0
    for x in np.where(f1_seq_stop)[0]:
        rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='red', alpha=0.5)
        layout_axd["f1_seq"].add_patch(rect)

    #For protein 2
    f2_seq_plot = layout_axd["f2_seq"].imshow(f2_seq[1], cmap=bw_cmap, norm=bw_norm)
    layout_axd["f2_seq"].get_yaxis().set_visible(False)
    layout_axd["f2_seq"].set_title("Protein 2 sequence", loc='left')
    layout_axd["f2_seq"].set_ylim([-0.5, 20.5])

    #Draw rectangle overlays to mark stop codons for protein 2
    f2_seq_stop = all_seq["prot"][0][1][20]
    f2_seq_stop = f2_seq_stop * -1.0
    for x in np.where(f2_seq_stop)[0]:
        rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='red', alpha=0.5)
        layout_axd["f2_seq"].add_patch(rect)

    #Draw rectangle overlays to mark whether there is a stop codon at the end
    stop_onehot = np.zeros(21)
    stop_onehot[20] = 1.0
    stop_onehot = np.expand_dims(stop_onehot, 0).repeat(total_steps, 0)
    f1_last_stop = np.sum(f1_last * stop_onehot, 1)
    f2_last_stop = np.sum(f2_last * stop_onehot, 1)
    
    if f1_last_stop[0] < 1.0:
        rect_f1_last_stop = patches.Rectangle((f1_len-0.5, -0.5), 1.0, 21, facecolor='blue', alpha=0.5)
        layout_axd["f1_seq"].add_patch(rect_f1_last_stop)    
    if f2_last_stop[0] < 1.0:
        rect_f2_last_stop = patches.Rectangle((f2_len-0.5, -0.5), 1.0, 21, facecolor='blue', alpha=0.5)
        layout_axd["f2_seq"].add_patch(rect_f2_last_stop)
    
    #If stop codon was not forced at the end, then don't show this rectangle
    if not gd_params["last_stop"]:
        rect_f1_last_stop.set_alpha(0.0)
        rect_f2_last_stop.set_alpha(0.0)
    
    #Draw rectangle overlays to mark whether forced AA position is matching
    f1_force = np.sum(np.repeat(gd_params['f1_force'].cpu().numpy(), total_steps, 0) * all_seq['prot'][0], 1)
    f2_force = np.sum(np.repeat(gd_params['f2_force'].cpu().numpy(), total_steps, 0) * all_seq['prot'][1], 1)
    f1_force_ind = np.where(np.sum(gd_params['f1_force'].cpu().numpy(), 1)[0])[0]
    f2_force_ind = np.where(np.sum(gd_params['f2_force'].cpu().numpy(), 1)[0])[0]
    for x in f1_force_ind:
        if f1_force[0, x] < 1.0:
            rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='orange', alpha=0.5)
            layout_axd["f1_seq"].add_patch(rect)
    for x in f2_force_ind:
        if f2_force[0, x] < 1.0:
            rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='orange', alpha=0.5)
            layout_axd["f2_seq"].add_patch(rect)

    #Plotting hallucination losses; only KLD for now
    losses_f1 = all_loss['kl'][0]# + all_loss['ce'][0] + all_loss['lddt'][0]
    losses_f2 = all_loss['kl'][1]# + all_loss['ce'][1] + all_loss['lddt'][1]
    losses_ymin = np.min(np.concatenate((losses_f1, losses_f2)))
    losses_ymax = np.max(np.concatenate((losses_f1, losses_f2)))
    losses_range = losses_ymax - losses_ymin
    loss_plot_f1 = layout_axd["net_loss"].plot(losses_f1, linewidth=0.5, alpha=0.5, color='red', label="Protein 1")
    loss_plot_f2 = layout_axd["net_loss"].plot(losses_f2, linewidth=0.5, alpha=0.5, color='blue', label="Protein 2")
    layout_axd["net_loss"].legend(loc="upper right", framealpha=0.5)
    layout_axd["net_loss"].axvline(x=gdstep-0.5, color='black', linewidth=1.0)
    layout_axd["net_loss"].set_ylim([losses_ymin-losses_range*0.05, losses_ymax+losses_range*0.05])
    layout_axd["net_loss"].set_title("Hallucination loss", loc='left')
    
    #Mark the loss at current step
    loss_current_f1 = layout_axd["net_loss"].plot(0, losses_f1[0], marker='o', markerfacecolor='red', markeredgecolor='black', markersize=6)
    loss_current_f2 = layout_axd["net_loss"].plot(0, losses_f2[0], marker='o', markerfacecolor='blue', markeredgecolor='black', markersize=6)

    #Plotting grad norms
    gradnorms = np.array([ np.linalg.norm(g, 2) for g in all_grad ])
    gradnorm_plot = layout_axd["gradnorm"].plot(gradnorms, linewidth=0.5)
    #gradnorms_ymax = np.percentile(gradnorms, 99)
    gradnorms_ymax = np.max(gradnorms)
    gradnorms_ymin = np.min(gradnorms)
    gradnorms_range = gradnorms_ymax - gradnorms_ymin
    layout_axd["gradnorm"].set_ylim([gradnorms_ymin-gradnorms_range*0.05, gradnorms_ymax+gradnorms_range*0.05])
    layout_axd["gradnorm"].set_title("Gradient norm", loc='left')
    layout_axd["gradnorm"].sharex(layout_axd["net_loss"])
    
    #Mark the grad norm at current step
    gradnorm_current = layout_axd["gradnorm"].plot(0, gradnorms[0], marker='o', markerfacecolor='red', markeredgecolor='black', markersize=6)

    #Text to display weights and constraints
    step0 = 1
    text0 = f"{'Step':<20}{step0:>5}{'/':^3}{total_steps:>5}\n"
    tp = f"{'Gradient descent':<20}{step0:>5}{'/':^3}{gdstep:>5}\n"
    fst = str(gd_params["f1"]) + ' & ' + str(gd_params["f2"])
    textf = f"\n{'Frames':<20}{fst:>7}\n"
    wst = ""
    for k, v in gd_params['weights'].items():
        wk = 'weight ' + k
        wst += f'{wk:<20}{v:>7.4f}\n'
    alpha_gd = gd_params["alpha_max"]
    alpha_sa = sa_params["alpha_max"]
    ast_gd = f"{'maxalpha_gd':<20}{alpha_gd:>7.4f}\n"
    ast_sa = f"{'maxalpha_sa':<20}{alpha_sa:>7.4f}"
    tst = ""
    for k, v in gd_params['thres'].items():
        tk = 'thres ' + k
        tst += f'{tk:<20}{v:>7.4f}\n'
    sst = textf + wst + tst + ast_gd + ast_sa
    text_plot = layout_axd["text"].text(0, 0.5, text0+tp+sst, fontsize=10, horizontalalignment='left', verticalalignment='center', fontfamily="monospace")
    layout_axd["text"].axis('off')
    
    #Decide which steps to plot given the interval. Also find the last minimal step before current
    plot_steps = np.array(list(range(0, total_steps, plot_interval)))
    min_steps = np.array(use_res[0]['min_step'] + [ s + gd_params['n_step'] for s in use_res[1]['min_step'] ])

    def find_nearest_below(arr, target):
        diff = target - arr + 1
        mask = np.ma.less_equal(diff, 0)
        if np.all(mask):
            return None
        masked_diff = np.ma.masked_array(diff, mask)
        return masked_diff.argmin()

    plot_steps_min = [ min_steps[find_nearest_below(min_steps, x)] if x >= min_steps[0] else None for x in range(total_steps) ]

    #Plot the last best solution distogram for protein 1
    dummy_data = np.zeros_like(all_plot["dist_argmax"][0][0])
    f1_dist_min_plot = layout_axd["f1_dist_min"].imshow(dummy_data, cmap="Reds", vmin=0, vmax=36)
    layout_axd["f1_dist_min"].set_title("Protein 1 dist. min", loc='left')
    layout_axd["f1_dist_min"].sharex(layout_axd["f1_dist"])
    
    #Plot the last best solution distogram for protein 2
    f2_dist_min_plot = layout_axd["f2_dist_min"].imshow(dummy_data, cmap="Reds", vmin=0, vmax=36)
    layout_axd["f2_dist_min"].set_title("Protein 2 dist. min", loc='left')
    layout_axd["f2_dist_min"].sharex(layout_axd["f2_dist"])
    
    #Plot LDDT for protein 1
    f1_lddt_ymax = np.max(all_plot["lddt"][0])
    f1_lddt_ymin = np.min(all_plot["lddt"][0])
    f1_lddt_range = f1_lddt_ymax - f1_lddt_ymin
    f1_lddt_plot = layout_axd["f1_lddt"].plot(all_plot["lddt"][0][0], label="Current")
    f1_lddt_min_plot = layout_axd["f1_lddt"].plot(all_plot["lddt"][0][0], label="Min", alpha=0.0) #Dummy last best solution LDDT
    layout_axd["f1_lddt"].set_ylim([f1_lddt_ymin-f1_lddt_range*0.05, f1_lddt_ymax+f1_lddt_range*0.05])
    layout_axd["f1_lddt"].set_title("Protein 1 LDDT")

    #Plot LDDT for protein 2
    f2_lddt_ymax = np.max(all_plot["lddt"][1])
    f2_lddt_ymin = np.min(all_plot["lddt"][1])
    f2_lddt_range = f2_lddt_ymax - f2_lddt_ymin
    f2_lddt_plot = layout_axd["f2_lddt"].plot(all_plot["lddt"][1][0], label="Current")
    f2_lddt_min_plot = layout_axd["f2_lddt"].plot(all_plot["lddt"][1][0], label="Min", alpha=0.0) #Dummy last best solution LDDT
    layout_axd["f2_lddt"].set_ylim([f2_lddt_ymin-f2_lddt_range*0.05, f2_lddt_ymax+f2_lddt_range*0.05])
    layout_axd["f2_lddt"].set_title("Protein 2 LDDT")

    #Animation function
    def animate(frame):
        step = plot_steps[frame]
                
        nuc_plot.set_data(all_seq["nuc"][step]) #Nuc sequence for current step
        f1_dist_plot.set_data(all_plot["dist_argmax"][0][step]) #Distogram for current step
        f2_dist_plot.set_data(all_plot["dist_argmax"][1][step])
        f1_seq_plot.set_data(f1_seq[step]) #Prot sequence for current step
        f2_seq_plot.set_data(f2_seq[step])

        #Remove previous rectangle overlays and add new ones for the current step
        f1_seq_stop = all_seq["prot"][0][step][20]
        f1_seq_stop = f1_seq_stop * -1.0
        [ p.remove() for p in reversed(layout_axd["f1_seq"].patches) ]
        for x in np.where(f1_seq_stop)[0]:
            rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='red', alpha=0.5)
            layout_axd["f1_seq"].add_patch(rect)

        f2_seq_stop = all_seq["prot"][1][step][20]
        f2_seq_stop = f2_seq_stop * -1.0
        [ p.remove() for p in reversed(layout_axd["f2_seq"].patches) ]
        for x in np.where(f2_seq_stop)[0]:
            rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='red', alpha=0.5)
            layout_axd["f2_seq"].add_patch(rect)
            
        if f1_last_stop[step] < 1.0:
            rect = patches.Rectangle((f1_len-0.5, -0.5), 1.0, 21, facecolor='blue', alpha=0.5)
            layout_axd["f1_seq"].add_patch(rect)    
        if f2_last_stop[step] < 1.0:
            rect = patches.Rectangle((f2_len-0.5, -0.5), 1.0, 21, facecolor='blue', alpha=0.5)
            layout_axd["f2_seq"].add_patch(rect)    
        if not gd_params["last_stop"]:
            rect_f1_last_stop.set_alpha(0.0)
            rect_f2_last_stop.set_alpha(0.0)   
        
        for x in f1_force_ind:
            if f1_force[step, x] < 1.0:
                rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='orange', alpha=0.5)
                layout_axd["f1_seq"].add_patch(rect)
        for x in f2_force_ind:
            if f2_force[step, x] < 1.0:
                rect = patches.Rectangle((x-0.5, -0.5), 1.0, 21, facecolor='orange', alpha=0.5)
                layout_axd["f2_seq"].add_patch(rect)
                
        loss_current_f1[0].set_data(step, losses_f1[step]) #Hal. loss at current step
        loss_current_f2[0].set_data(step, losses_f2[step])

        #Update gradient plot
        if step < gdstep:
            grad_normfunc = colors.CenteredNorm(vcenter=0.0, halfrange=np.percentile(np.abs(all_grad[step]), 99), clip=True) #Normalize gradient to 99 percentile for plotting
            grad_plot.set_data(grad_normfunc(all_grad[step]))

            gradnorm_current[0].set_data(step, gradnorms[step]) #Update grad norm for current step
            step_ = step + 1
            tp = f"{'Gradient descent':<20}{step_:>5}{'/':^3}{gdstep:>5}\n"
        else:
            if step == gdstep: #Remove grad norm plot when we enter SA steps
                gradnorm_plot[0].remove()
                grad_plot.remove()
                gradnorm_current[0].remove()
            step_ = step - gdstep + 1
            tp = f"{'Simulated annealing':<20}{step_:>5}{'/':^3}{sastep:>5}\n"
        step0 = step + 1
        text0 = f"{'Step':<20}{step0:>5}{'/':^3}{total_steps:>5}\n"
        text_plot.set_text(text0+tp+sst)

        #Update last best solution distograms and LDDTs
        if plot_steps_min[step] is not None:
            f1_dist_min_plot.set_data(all_plot["dist_argmax"][0][plot_steps_min[step]])
            f2_dist_min_plot.set_data(all_plot["dist_argmax"][1][plot_steps_min[step]])            
            f1_lddt_min_plot[0].set_data(f1_lddt_plot[0].get_data()[0], all_plot["lddt"][0][plot_steps_min[step]])
            f2_lddt_min_plot[0].set_data(f2_lddt_plot[0].get_data()[0], all_plot["lddt"][1][plot_steps_min[step]])
            if f1_lddt_min_plot[0].get_alpha() == 0.0: #First last best solution LDDT
                f1_lddt_min_plot[0].set_alpha(1.0)
                f2_lddt_min_plot[0].set_alpha(1.0)
                layout_axd["f1_lddt"].legend(loc="upper right", framealpha=0.5)
                layout_axd["f2_lddt"].legend(loc="upper right", framealpha=0.5)

        #Update LDDT
        f1_lddt_plot[0].set_data(f1_lddt_plot[0].get_data()[0], all_plot["lddt"][0][step])
        f2_lddt_plot[0].set_data(f2_lddt_plot[0].get_data()[0], all_plot["lddt"][1][step])

        return

    n_frame = len(plot_steps) - 1
    f_anim = anim.FuncAnimation(f, animate, frames=n_frame, interval=frame_interval)
    
    return f_anim

def min_dist(use_res):
    min_step = use_res[1]['min_step'][-1]
    f1_min_dist = use_res[1]['plot'][min_step]["dist_argmax"][0].cpu().numpy()
    f2_min_dist = use_res[1]['plot'][min_step]["dist_argmax"][1].cpu().numpy()
    f1_min_lddt = use_res[1]['plot'][min_step]["lddt"][0][0].cpu().numpy() * 100.0
    f2_min_lddt = use_res[1]['plot'][min_step]["lddt"][1][0].cpu().numpy() * 100.0

    layout = [
        [ "f1_dist", "f2_dist", "dist_cbar" ],
        [ "lddt", "lddt", "lddt" ]]

    f = plt.figure(constrained_layout=True, figsize=(6, 5))
    layout_axd = f.subplot_mosaic(layout, empty_sentinel="X", gridspec_kw={
        'width_ratios': [0.5, 0.5, 0.05],
        'height_ratios': [0.5, 0.5]})

    f1_dist_plot = layout_axd["f1_dist"].imshow(f1_min_dist, cmap="Reds", vmin=0, vmax=36)
    f2_dist_plot = layout_axd["f2_dist"].imshow(f2_min_dist, cmap="Reds", vmin=0, vmax=36)
    dist_cbar = f.colorbar(f1_dist_plot, cax=layout_axd["dist_cbar"])
    layout_axd["f1_dist"].set_title("Protein 1 distogram", loc="left")
    layout_axd["f2_dist"].set_title("Protein 2 distogram", loc="left")
    
    f1_lddt_plot = layout_axd["lddt"].plot(f1_min_lddt, label="Protein 1")
    f2_lddt_plot = layout_axd["lddt"].plot(f2_min_lddt, label="Protein 2")
    layout_axd["lddt"].set_ylim([0, 100])
    layout_axd["lddt"].legend(loc="upper right", framealpha=0.5)
    layout_axd["lddt"].set_title("pLDDT", loc="left")
    
    return f
