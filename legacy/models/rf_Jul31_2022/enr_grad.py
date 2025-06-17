import torch
import numpy as np

def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B,L = a.shape[:2]

    u1 = a-b
    u2 = c-b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B*L, 3)
    u2 = u2.reshape(B*L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1) # (B,L,1)
    cos_theta = torch.matmul(u1[:,None,:], u2[:,:,None]).reshape(B, L, 1)
    
    return torch.cat([cos_theta, sin_theta], axis=-1) # (B, L, 2)

def length(a, b):
    return torch.norm(a-b, dim=-1)

def torsion(a,b,c,d, eps=1e-6):
    #A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b-a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c-b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d-c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1) #[B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)
    
    cos_angle = torch.matmul(t1[:,:,None,:], t2[:,:,:,None])[:,:,0]
    sin_angle = torch.norm(u2, dim=-1,keepdim=True)*(torch.matmul(u1[:,:,None,:], t2[:,:,:,None])[:,:,0])
    
    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1)/(t1_norm*t2_norm+eps) #[B,L,2]
    return cos_sin

# ideal N-C distance, ideal cos(CA-C-N angle), ideal cos(C-N-CA angle)
def calc_BB_bond_geom(pred, idx, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, sig_len=0.02, sig_ang=0.05):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    B, L = pred.shape[:2]

    bonded = (idx[:,1:] - idx[:,:-1])==1

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    CN_loss = torch.clamp( torch.abs(blen_CN_pred - ideal_NC) - sig_len, min=0.0 )
    CN_loss = (bonded*CN_loss).sum() / (bonded.sum())
    blen_loss = CN_loss   #fd squared loss

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    bang_CNCA_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    CACN_loss = torch.clamp( torch.abs(bang_CACN_pred - ideal_CACN) - sig_ang,  min=0.0 )
    CACN_loss = (bonded*CACN_loss).sum() / (bonded.sum())
    CNCA_loss = torch.clamp( torch.abs(bang_CNCA_pred - ideal_CNCA) - sig_ang,  min=0.0 )
    CNCA_loss = (bonded*CNCA_loss).sum() / (bonded.sum())
    bang_loss = CACN_loss + CNCA_loss

    return blen_loss+bang_loss

# LJ loss
#  custom backwards for mem efficiency
class LJLoss(torch.autograd.Function):
    @staticmethod
    def ljVdV(deltas, sigma, epsilon, lj_lin, eps):
        dist = torch.sqrt( torch.sum ( torch.square( deltas ), dim=-1 ) + eps )
        linpart = dist<lj_lin*sigma
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])

        # works for linpart too
        dljEdd_over_r = epsilon * (-12 * sd12/deff + 12 * sd6/deff) / (dist)

        return ljE.sum(), dljEdd_over_r

    @staticmethod
    def forward(
        ctx, xs, seq, aamask, ljparams, ljcorr, num_bonds, 
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training
    ):
        L,A = xs.shape[:2]

        ds_res = torch.sqrt( torch.sum ( torch.square( 
            xs.detach()[:,None,1,:]-xs.detach()[None,:,1,:]), dim=-1 ))
        rs = torch.triu_indices(L,L,0, device=xs.device)
        ri,rj = rs[0],rs[1]

        # batch during inference for huge systems
        BATCHSIZE = len(ri)
        if (not training):
            BATCHSIZE = 65536

        #print (BATCHSIZE, (len(ri)-1)//BATCHSIZE + 1)

        ljval = 0
        dljEdx = torch.zeros_like(xs, dtype=torch.float)

        for i_batch in range((len(ri)-1)//BATCHSIZE + 1):
            idx = torch.arange(
                i_batch*BATCHSIZE, 
                min( (i_batch+1)*BATCHSIZE, len(ri)),
                device=xs.device
            )
            rii,rjj = ri[idx],rj[idx] 

            ridx,ai,aj = (
                aamask[seq[rii]][:,:,None]*aamask[seq[rjj]][:,None,:]
            ).nonzero(as_tuple=True)
            deltas = xs[rii,:,None,:]-xs[rjj,None,:,:]
            seqi,seqj = seq[rii[ridx]], seq[rjj[ridx]]

            mask = torch.ones_like(ridx, dtype=torch.bool) # are atoms defined?

            intrares = (rii[ridx]==rjj[ridx])
            mask[intrares*(ai<aj)] = False  # upper tri (atoms)

            # count-pair
            mask[intrares] *= num_bonds[seqi[intrares],ai[intrares],aj[intrares]]>=4
            pepbondres = ri[ridx]+1==rj[ridx]
            mask[pepbondres] *= (
                num_bonds[seqi[pepbondres],ai[pepbondres],2]
                + num_bonds[seqj[pepbondres],0,aj[pepbondres]]
                + 1) >=4

            # apply mask.  only interactions to be scored remain
            ai,aj,seqi,seqj,ridx = ai[mask],aj[mask],seqi[mask],seqj[mask],ridx[mask]
            deltas = deltas[ridx,ai,aj]

            # hbond correction
            use_hb_dis = (
                ljcorr[seqi,ai,0]*ljcorr[seqj,aj,1] 
                + ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] ).nonzero()
            use_ohdon_dis = ( # OH are both donors & acceptors
                ljcorr[seqi,ai,0]*ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] 
                +ljcorr[seqi,ai,0]*ljcorr[seqj,aj,0]*ljcorr[seqj,aj,1] 
            ).nonzero()
            use_hb_hdis = (
                ljcorr[seqi,ai,2]*ljcorr[seqj,aj,1] 
                +ljcorr[seqi,ai,1]*ljcorr[seqj,aj,2] 
            ).nonzero()

            # disulfide correction
            potential_disulf = (ljcorr[seqi,ai,3]*ljcorr[seqj,aj,3] ).nonzero()

            ljrs = ljparams[seqi,ai,0] + ljparams[seqj,aj,0]
            ljrs[use_hb_dis] = lj_hb_dis
            ljrs[use_ohdon_dis] = lj_OHdon_dis
            ljrs[use_hb_hdis] = lj_hbond_hdis

            ljss = torch.sqrt( ljparams[seqi,ai,1] * ljparams[seqj,aj,1] + eps )
            ljss [potential_disulf] = 0.0

            natoms = torch.sum(aamask[seq])
            ljval_i,dljEdd_i = LJLoss.ljVdV(deltas,ljrs,ljss,lj_lin,eps)

            ljval += ljval_i / natoms

            # sum per-atom-pair grads into per-atom grads
            # note this is stochastic op on GPU
            idxI,idxJ = rii[ridx]*A + ai, rjj[ridx]*A + aj
            dljEdx.view(-1,3).index_add_(0, idxI, dljEdd_i[:,None]*deltas, alpha=1.0/natoms)
            dljEdx.view(-1,3).index_add_(0, idxJ, dljEdd_i[:,None]*deltas, alpha=-1.0/natoms)

        ctx.save_for_backward(dljEdx)

        return ljval

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        dljEdx, = ctx.saved_tensors
        return (
            grad_output * dljEdx, 
            None, None, None, None, None, None, None, None, None, None, None
        )

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8,
    training=True
):
    lj = LJLoss.apply
    ljval = lj(
        xs, seq, aamask, ljparams, ljcorr, num_bonds, 
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training)

    return ljval

@torch.enable_grad()
def calc_BB_bond_geom_grads(pred, idx, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, sig_len=0.02, sig_ang=0.05):
    pred.requires_grad_(True)
    Ebond = calc_BB_bond_geom(pred, idx, eps, ideal_NC, ideal_CACN, ideal_CNCA, sig_len, sig_ang)
    return torch.autograd.grad(Ebond, pred)

@torch.enable_grad()
def calc_lj_grads(
    seq, xyz, alpha, toaa, 
    aamask, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8, training=False
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Elj = calc_lj(
        seq[0], 
        xyzaa[0,...,:3], 
        aamask, 
        ljparams, 
        ljcorr, 
        num_bonds, 
        lj_lin, 
        lj_hb_dis, 
        lj_OHdon_dis, 
        lj_hbond_hdis, 
        lj_maxrad,
        eps,
        training
    )
    return torch.autograd.grad(Elj, (xyz,alpha))
