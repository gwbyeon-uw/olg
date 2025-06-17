#Helper function for running single sequences on OpenFold. Mostly modified functions from OpenFold repo as well as ColabFold
import os

import numpy as np 
import torch

from openfold import config
from openfold.model import model
from openfold.data import data_pipeline
from openfold.data import feature_pipeline
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein, residue_constants

import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib import collections as mcoll

#From OpenFold notebook example
def placeholder_template_feats(num_templates_, num_res_):
    return {
      'template_aatype': torch.zeros(num_templates_, num_res_, 22).long(),
      'template_all_atom_positions': torch.zeros(num_templates_, num_res_, 37, 3),
      'template_all_atom_mask': torch.zeros(num_templates_, num_res_, 37),
      'template_domain_names': torch.zeros(num_templates_),
      'template_sum_probs': torch.zeros(num_templates_, 1),
    }

def prepare_openfold_input(seq, cfg): #Input is string of protein sequences
    num_templates = 1 #Dummy
    num_res = len(seq)

    feature_dict = {}
    feature_dict.update(data_pipeline.make_sequence_features(seq, 'test', num_res))
    feature_dict.update(data_pipeline.make_msa_features(([seq],), deletion_matrices=([[0]*len(seq)],)))
    feature_dict.update(placeholder_template_feats(num_templates, num_res))

    pipeline = feature_pipeline.FeaturePipeline(cfg.data)
    processed_feature_dict = pipeline.process_features(feature_dict, mode='predict')
    processed_feature_dict = tensor_tree_map(lambda t: t.cuda(), processed_feature_dict)
    
    return processed_feature_dict, feature_dict

def load_openfold(weight_path, max_recycle, weight_name="model_1"):
    weight_name_spl = weight_name.split("_")
    if(weight_name_spl[-1] == "ptm"):
        of_model_name = "finetuning_ptm_2.pt"
        model_name = "model_1_ptm"
    else:
        model_name = "model_1"
        of_model_name = f"finetuning_{weight_name_spl[-1]}.pt"
    
    cfg = config.model_config(model_name)
    cfg['data']['common']['max_recycling_iters'] = max_recycle #Try setting recycling high up to 48; recycling sometimes significantly improves prediction    
    openfold_model = model.AlphaFold(cfg)
    openfold_model = openfold_model.eval()
    
    params_name = os.path.join(weight_path, of_model_name)
    d = torch.load(params_name)
    openfold_model.load_state_dict(d)
    openfold_model = openfold_model.cuda()
    
    return openfold_model, cfg

#Plots that summarize OpenFold predictions
def plot_2d(result_f1, result_f2):
    if 'predicted_aligned_error' in result_f1.keys():
        layout = [
            [ "f1_pseudo3d", "f2_pseudo3d", "lddt_cbar" ],
            [ "f1_dist", "f2_dist", "dist_cbar" ],
            [ "f1_pae", "f2_pae", "pae_cbar"],
            [ "lddt", "lddt", "lddt" ]]
        
        f = plt.figure(constrained_layout=True, figsize=(6, 10))
        layout_axd = f.subplot_mosaic(layout, empty_sentinel="X", gridspec_kw={
            'width_ratios': [0.5, 0.5, 0.05],
            'height_ratios': [0.5, 0.5, 0.5, 0.5]})
        
    else:
        layout = [
            [ "f1_pseudo3d", "f2_pseudo3d", "lddt_cbar" ],
            [ "f1_dist", "f2_dist", "dist_cbar" ],
            [ "lddt", "lddt", "lddt" ]]
        
        f = plt.figure(constrained_layout=True, figsize=(6, 7.5))
        layout_axd = f.subplot_mosaic(layout, empty_sentinel="X", gridspec_kw={
            'width_ratios': [0.5, 0.5, 0.05],
            'height_ratios': [0.5, 0.5, 0.5]})

    f1_dist = result_f1['distogram_logits'].argmax(2).cpu().numpy()
    f1_dist_plot = layout_axd["f1_dist"].imshow(f1_dist, cmap="Reds", vmin=0, vmax=64)
    dist_cbar = f.colorbar(f1_dist_plot, cax=layout_axd["dist_cbar"])
    layout_axd["f1_dist"].set_title("Protein 1 distogram", loc="left")

    if result_f2 is not None:
        f2_dist = result_f2['distogram_logits'].argmax(2).cpu().numpy()
        f2_dist_plot = layout_axd["f2_dist"].imshow(f2_dist, cmap="Reds", vmin=0, vmax=64)
        layout_axd["f2_dist"].set_title("Protein 2 distogram", loc="left")

    if 'predicted_aligned_error' in result_f1.keys():
        f1_pae = result_f1['predicted_aligned_error'].cpu().numpy()
        f1_pae_plot = layout_axd["f1_pae"].imshow(f1_pae, cmap="magma", vmin=0, vmax=30)
        pae_cbar = f.colorbar(f1_pae_plot, cax=layout_axd["pae_cbar"])
        layout_axd["f1_pae"].set_title("Protein 1 PAE", loc="left")
        
        if result_f2 is not None:
            f2_pae = result_f2['predicted_aligned_error'].cpu().numpy()
            f2_pae_plot = layout_axd["f2_pae"].imshow(f2_pae, cmap="magma", vmin=0, vmax=30)
            layout_axd["f2_pae"].set_title("Protein 2 PAE", loc="left")

    f1_lddt = result_f1['plddt'].cpu().numpy()
    f1_lddt_plot = layout_axd["lddt"].plot(f1_lddt, label="Protein 1")
    layout_axd["lddt"].set_ylim([0, 100])
    layout_axd["lddt"].set_title("pLDDT", loc="left")
    
    if result_f2 is not None:
        f2_lddt = result_f2['plddt'].cpu().numpy()
        f2_lddt_plot = layout_axd["lddt"].plot(f2_lddt, label="Protein 2")
        layout_axd["lddt"].legend(loc="upper right", framealpha=0.5)

    f1_pseudo3d_plot = plot_pseudo_3D(result_f1["final_atom_positions"], result_f1["plddt"], layout_axd["f1_pseudo3d"], cmap="gist_rainbow", cmin=50, cmax=90)
    lddt_cbar = f.colorbar(matplotlib.cm.ScalarMappable(cmap="gist_rainbow", norm=matplotlib.colors.Normalize(vmin=50.0, vmax=90.0)), cax=layout_axd["lddt_cbar"])
    layout_axd["f1_pseudo3d"].axis('off')
    layout_axd["f1_pseudo3d"].set_title("Protein 1 Pseudo-3D", loc="left")
    
    if result_f2 is not None:
        f2_pseudo3d_plot = plot_pseudo_3D(result_f2["final_atom_positions"], result_f2["plddt"], layout_axd["f2_pseudo3d"], cmap="gist_rainbow", cmin=50, cmax=90)
        layout_axd["f2_pseudo3d"].axis('off')
        layout_axd["f2_pseudo3d"].set_title("Protein 2 Pseudo-3D", loc="left")

    return f

#Various helper functions from OpenFold and ColabFold repo's
def prep_output(out, batch, feature_dict, feature_processor, model_name="model_1_ptm", multimer_ri_gap=200):
    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    # Prep protein metadata
    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset={model_name}",
    ])
    
    ri = feature_dict["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if(c != cur_chain):
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remark=remark,
    )

    return unrelaxed_protein

def get_pdb(processed_feature_dict, prediction_result, feature_dict, config, model_name="model_1_ptm"):
    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict_np = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()), 
        processed_feature_dict
    )
    
    prediction_result_np = tensor_tree_map(lambda x: np.array(x.cpu()), prediction_result)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    
    unrelaxed_protein = prep_output(
        prediction_result_np, 
        processed_feature_dict_np, 
        feature_dict,
        feature_processor, 
        model_name
    )
    
    pdb = protein.to_pdb(unrelaxed_protein)
    return pdb

def kabsch(a, b, weights=None, return_v=False):
    a = np.asarray(a)
    b = np.asarray(b)
    if weights is None: weights = np.ones(len(b))
    else: weights = np.asarray(weights)
    B = np.einsum('ji,jk->ik', weights[:, None] * a, b)
    u, s, vh = np.linalg.svd(B)
    if np.linalg.det(u @ vh) < 0: u[:, -1] = -u[:, -1]
    if return_v: return u
    else: return u @ vh

def ca_align_to_last(positions, ref, Ls):
    def align(P, Q):
        if Ls is None or len(Ls) == 1:
            P_,Q_ = P,Q
        else:
        # align relative to first chain
            P_,Q_ = P[:Ls[0]],Q[:Ls[0]]
        p = P_ - P_.mean(0,keepdims=True)
        q = Q_ - Q_.mean(0,keepdims=True)
        return ((P - P_.mean(0,keepdims=True)) @ kabsch(p,q)) + Q_.mean(0,keepdims=True)

    pos = positions[ref,:,1,:] - positions[ref,:,1,:].mean(0,keepdims=True)
    best_2D_view = pos @ kabsch(pos,pos,return_v=True)

    new_positions = []
    for i in range(len(positions)):
        new_positions.append(align(positions[i,:,1,:],best_2D_view))
    return np.asarray(new_positions)

def plot_pseudo_3D(xyz_in, c_in=None, ax=None, chainbreak=5,
                   cmap="gist_rainbow", line_w=2.0,
                   cmin=None, cmax=None, zmin=None, zmax=None):

    xyz = ca_align_to_last(xyz_in.unsqueeze(0).cpu().numpy(), 0, [xyz_in.shape[0]])[0]
    c = c_in.cpu().numpy()
    
    def rescale(a,amin=None,amax=None):
        a = np.copy(a)
        if amin is None: amin = a.min()
        if amax is None: amax = a.max()
        a[a < amin] = amin
        a[a > amax] = amax
        return (a - amin)/(amax - amin)

    # make segments
    xyz = np.asarray(xyz)
    seg = np.concatenate([xyz[:-1,None,:],xyz[1:,None,:]],axis=-2)
    seg_xy = seg[...,:2]
    seg_z = seg[...,2].mean(-1)
    ord = seg_z.argsort()

    # set colors
    if c is None: c = np.arange(len(seg))[::-1]
    else: c = (c[1:] + c[:-1])/2
    c = rescale(c,cmin,cmax)  

    if isinstance(cmap, str):
        if cmap == "gist_rainbow": c *= 0.75
        colors = matplotlib.cm.get_cmap(cmap)(c)
    else:
        colors = cmap(c)
  
    if chainbreak is not None:
        dist = np.linalg.norm(xyz[:-1] - xyz[1:], axis=-1)
        colors[...,3] = (dist < chainbreak).astype(float)

    # add shade/tint based on z-dimension
    z = rescale(seg_z,zmin,zmax)[:,None]
    tint, shade = z/3, (z+2)/3
    colors[:,:3] = colors[:,:3] + (1 - colors[:,:3]) * tint
    colors[:,:3] = colors[:,:3] * shade
  
    set_lim = False
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(5)
        fig.set_figheight(5)
        set_lim = True
    else:
        fig = ax.get_figure()
        if ax.get_xlim() == (0,1):
            set_lim = True

    if set_lim:
        xy_min = xyz[:,:2].min() - line_w
        xy_max = xyz[:,:2].max() + line_w
        ax.set_xlim(xy_min,xy_max)
        ax.set_ylim(xy_min,xy_max)

    ax.set_aspect('equal')

    # determine linewidths
    width = fig.bbox_inches.width * ax.get_position().width
    linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

    lines = mcoll.LineCollection(seg_xy[ord], colors=colors[ord], linewidths=linewidths,
                               path_effects=[matplotlib.patheffects.Stroke(capstyle="round")])

    return ax.add_collection(lines)