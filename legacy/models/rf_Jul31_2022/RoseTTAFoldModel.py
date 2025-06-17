import sys
import numpy as np
import torch
import torch.nn as nn
from common.util_module import ComputeAllAtomCoords
from .featurizing import MSABlockDeletion, MSAFeaturize
from .Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from .Track_module import IterativeSimulator
from .AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, LDDTNetwork 
from common.kinematics import xyz_to_t2d, get_init_xyz
from common import util

def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

class RoseTTAFoldModule(nn.Module):
    def __init__(
        self, n_extra_block=4, n_main_block=8, n_ref_block=4,\
        d_msa=256, d_msa_full=64, d_pair=128, d_templ=64,
        n_head_msa=8, n_head_pair=4, n_head_templ=4,
        d_hidden=32, d_hidden_templ=64,
        rbf_sigma=1.0, p_drop=0.15,
        SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
        use_deeper_ref=False,
        aamask=None, ljlk_parameters=None, lj_correction_parameters=None, num_bonds=None, lj_lin=0.75, device='cuda:0'
    ):
        super(RoseTTAFoldModule, self).__init__()
        #
        # Input Embeddings
        d_state = SE3_param['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair,  d_state=d_state, p_drop=p_drop)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25, p_drop=p_drop)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state, n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25)
        
        if aamask is None:
            aamask = util.allatom_mask.to(device)
        if ljlk_parameters is None:
            ljlk_parameters = util.ljlk_parameters.to(device)
        if lj_correction_parameters is None:
            lj_correction_parameters = util.lj_correction_parameters.to(device)
        if num_bonds is None:
            num_bonds = util.num_bonds.to(device)
            
        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state, rbf_sigma=rbf_sigma)
        #
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block,
            n_main_block=n_main_block,
            n_ref_block=n_ref_block,
            d_msa=d_msa, 
            d_msa_full=d_msa_full,
            d_pair=d_pair, 
            d_hidden=d_hidden,
            n_head_msa=n_head_msa,
            n_head_pair=n_head_pair,
            SE3_param=SE3_param,
            rbf_sigma=rbf_sigma,
            p_drop=p_drop,
            aamask=aamask, 
            ljlk_parameters=ljlk_parameters,
            lj_correction_parameters=lj_correction_parameters, 
            num_bonds=num_bonds,
            lj_lin=lj_lin,
            use_deeper_ref=use_deeper_ref
        )

        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
    
    def prep_fake_template(self, L, device):
        xyz_t = torch.full((1,L,27,3),np.nan).float()
        xyz_t = xyz_t.float().unsqueeze(0).to(device)
        t2d = xyz_to_t2d(xyz_t)

        t1d = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,L,1)).float()), -1)
        t1d = t1d.float().unsqueeze(0).to(device)

        xyz_t_msa = torch.full((1,L,27,3),np.nan).float()
        xyz_t_msa = xyz_t_msa.float().unsqueeze(0).to(device)
        t2d_msa = xyz_to_t2d(xyz_t_msa)

        t1d_msa = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=21).float() # all gaps
        t1d_msa = torch.cat((t1d_msa, torch.zeros((1,L,1)).float()), -1)
        t1d_msa = t1d_msa.float().unsqueeze(0).to(device)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(-1,L,27,3),
            seq_tmp,
            util.torsion_indices.to(device),
            util.torsion_can_flip.to(device),
            util.reference_angles.to(device))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))

        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        xyz_t = get_init_xyz(xyz_t) # initialize coordinates with first template

        seq_tmp = t1d_msa[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t_msa.reshape(-1,L,27,3),
            seq_tmp,
            util.torsion_indices.to(device),
            util.torsion_can_flip.to(device),
            util.reference_angles.to(device))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))

        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t_msa = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        xyz_t_msa = get_init_xyz(xyz_t_msa) # initialize coordinates with first template
        
        return (xyz_t, t1d, t2d, alpha_t), (xyz_t_msa, t1d_msa, t2d_msa, alpha_t_msa)
    
    #def _run_iter(self, msa_latent=None, msa_full=None, seq=None, xyz=None, sctors=None, idx=None,
    def _run_iter(self, msa_latent=None, msa_full=None, seq=None, seq1hot=None, xyz=None, sctors=None, idx=None,
                        t1d=None, t2d=None, xyz_t=None, alpha_t=None, 
                        msa_prev=None, pair_prev=None, state_prev=None,
                        use_checkpoint=True):
        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, seq1hot)
        msa_full = self.full_emb(msa_full, seq, idx, seq1hot)
        #
        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(msa_prev, pair_prev, xyz, state_prev, sctors)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle

        # add template embedding
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint)

        # Predict coordinates from given inputs
        msa, pair, xyz, alpha_s, state = self.simulator(
            seq, msa_latent, msa_full, pair, xyz[:,:,:3], state, idx, use_checkpoint=use_checkpoint)

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)

        # predict distogram & orientograms
        logits = self.c6d_pred(pair)

        # Predict LDDT
        lddt = self.lddt_pred(state)

        return logits, xyz, alpha_s, lddt, msa[:,0], pair, state, logits_aa

    #def forward(self, msa, ins, t1d, t2d, xyz_t, xyz, alpha_t, nmer=1, L_s=[], device='cuda:0', max_cycle=15, max_seq=10000, use_amp=True):
    def forward(self, msa, msa_one_hot, term_info, t1d, t2d, xyz_t, xyz, alpha_t, nmer=1, L_s=[], device='cuda:0'):
        compute_allatom_coords = ComputeAllAtomCoords().to(device)
        
        #BNLA, A=21
        B, N, L = msa.shape #Assume B=1, N=1
        pad_size = 22 - msa_one_hot.shape[-1]
        pad_row = torch.zeros(1, 1, L, pad_size).to(msa_one_hot.device)
        msa_one_hot_pad = torch.cat([msa_one_hot, pad_row], -1)[0] # pad to 22 tokens
        msa_feat = torch.cat([msa_one_hot_pad, msa_one_hot_pad, torch.zeros(1, L, 2).to(msa_one_hot.device), term_info.unsqueeze(0)], dim=-1).float() #Seed
        extra_feat = torch.cat([msa_one_hot_pad, torch.zeros(1, L, 1).to(msa_one_hot.device), term_info.unsqueeze(0)], dim=-1).float() #Extra
        seq = msa[:, 0][0]
        seq1hot = msa_one_hot_pad[0].float()
        
        idx_pdb = torch.arange(L, device=device).long().view(1, L)
        if len(L_s) > 1:
            start_L = 0
            for L_subunit in L_s[:-1]:
                start_L += L_subunit
                idx_pdb[:,start_L:] += 100
        
        msa_prev = None
        pair_prev = None
        state_prev = None
        alpha_prev = torch.zeros((1, L, 10, 2), device=device)
        xyz_prev=xyz

        best_lddt = torch.tensor([-1.0], device=device)
        best_xyz = None
        best_logit = None

        inputs = {}
        inputs['idx'] = idx_pdb
        inputs['t1d'] = t1d
        inputs['t2d'] = t2d
        inputs['xyz_t'] = xyz_t
        inputs['alpha_t'] = alpha_t

        seq = seq.unsqueeze(0)
        inputs['seq'] = seq
        inputs['seq1hot'] = seq1hot.unsqueeze(0)
        inputs['msa_latent'] = msa_feat.unsqueeze(0)
        inputs['msa_full'] = extra_feat.unsqueeze(0)
        inputs['msa_prev'] = msa_prev
        inputs['pair_prev'] = pair_prev
        inputs['state_prev'] = state_prev
        inputs['xyz'] = xyz_prev
        inputs['sctors'] = alpha_prev

        
        #with torch.cuda.amp.autocast(use_amp):
        logit_s, init_crds, alpha_tmp, pred_lddt_binned, msa_tmp, pair_tmp, state_tmp, logits_aa = self._run_iter(**inputs)
        pred_lddt = lddt_unbin(pred_lddt_binned)

        #xyz_prev = init_crds[-1]
        #alpha_prev = alpha_tmp[-1]    
        #_, all_crds = compute_allatom_coords(seq, xyz_prev, alpha_prev)
        '''
            #sys.stdout.write("RECYCLE [%02d/%02d] pred LDDT: %.4f / best LDDT: %.4f\n"%(i_cycle, max_cycle, pred_lddt.mean(), best_lddt.mean()))
            if torch.isnan(pred_lddt).any(): # something went wrong -- not using this results
                continue

            xyz_prev = init_crds[-1]
            alpha_prev = alpha_tmp[-1]
            msa_prev = msa_tmp
            pair_prev = pair_tmp
            state_prev = state_tmp

            if pred_lddt.mean() < best_lddt.mean():
                continue
            _, all_crds = compute_allatom_coords(seq, xyz_prev, alpha_prev)
            best_xyz = all_crds.clone()
            best_logit = logit_s
            best_lddt = pred_lddt.clone()
        '''

        prob_s = list()
        #for logit in best_logit:
        for logit in logit_s:
            prob = nn.Softmax(dim=1)(logit.float()) # distogram
            prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
            prob_s.append(prob)

        for prob in prob_s:
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
        prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        
        out = dict(
            dist = logit_s[0],
            omega = logit_s[1],
            theta = logit_s[2],
            phi = logit_s[3],
            xyz = init_crds.reshape(-1, B, L, 3, 3)[-1],
            lddt = pred_lddt.view(B, L),
            prob_s = prob_s,
            logits_aa = logits_aa,
            pair = pair_tmp,
            msa = msa_tmp
        )
        
        return out
        
        #return best_xyz[0].detach().cpu(), best_lddt[0].float().detach().cpu(), prob_s
