"""Boltz-2 structure-prediction wrapper.

Builds model inputs (sequences, MSA, pocket conditioning), runs Boltz prediction (monomer fold or
multi-chain co-folding), and writes/scores the predicted structures. A standalone utility for the
structure-conditioned loops the caller assembles manually (see the README binder-design example);
it is not invoked by ``OLGDesign``.
"""
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict
from pathlib import Path
import gc
import os
import random
import subprocess

import torch
import torch.nn as nn
import numpy as np

from boltz.model.models.boltz2 import Boltz2
from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, PairformerArgsV2, MSAModuleArgs
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.types import Coords, Ensemble, StructureV2, MSA, Input
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.write.pdb import to_pdb

# ---------------------------------------------------------------------------
# Constants & helpers adapted from Protein-Hunter
# ---------------------------------------------------------------------------

CHAIN_TO_NUMBER = {chr(ord("A") + i): i for i in range(26)}


def smart_split(s: str) -> list[str]:
    """Split a string on comma, colon, or whitespace (in that priority order)."""
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",")]
    if ":" in s:
        return [x.strip() for x in s.split(":")]
    return [x.strip() for x in s.split()]


def get_cif(cif_code: str = "") -> str:
    """Return local path to a CIF/PDB file, fetching from RCSB or AlphaFold if needed."""
    if not cif_code:
        raise ValueError("No CIF code or path specified.")
    if os.path.isfile(cif_code):
        return os.path.abspath(cif_code)
    if len(cif_code) == 4:
        local_cif = f"{cif_code}.cif"
        if not os.path.isfile(local_cif):
            subprocess.run(
                ["wget", "-qnc", f"https://files.rcsb.org/download/{cif_code}.cif"],
                check=True,
            )
        return os.path.abspath(local_cif)
    # AlphaFold ID
    local_cif = f"AF-{cif_code}-F1-model_v3.cif"
    if not os.path.isfile(local_cif):
        subprocess.run(
            ["wget", "-qnc", f"https://alphafold.ebi.ac.uk/files/AF-{cif_code}-F1-model_v3.cif"],
            check=True,
        )
    return os.path.abspath(local_cif)


def process_msa(chain_id: str, sequence: str, msa_dir: Path) -> Path:
    """Run MMseqs2 MSA search and return path to Boltz-format .npz file."""
    msa_chain_dir = msa_dir / f"{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(exist_ok=True, parents=True)

    unpaired_msa = run_mmseqs2(
        [sequence],
        str(msa_chain_dir),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy",
    )

    msa_a3m_path = env_dir / "msa.a3m"
    msa_a3m_path.write_text(unpaired_msa[0])

    msa_npz_path = env_dir / "msa.npz"
    if not msa_npz_path.exists():
        msa = parse_a3m(msa_a3m_path, taxonomy=None, max_seqs=4096)
        msa.dump(msa_npz_path)

    return msa_npz_path

@dataclass
class BoltzPHConfig:
    """
    Reference: https://github.com/yehlincho/Protein-Hunter/blob/main/protein_hunter_boltz.ipynb
    """
    device: torch.device = torch.device("cuda:0")
    name: str = ""  # @param {type:"string"}
    mode: Literal["binder", "unconditional"] = "unconditional"  # @param ["unconditional", "binder"]
    save_dir: str = "" # Dummy
    cyclic: bool = False  # @param {type:"boolean"}
    protein_seqs: Optional[str] =  None
    msa_mode: Literal["single", "mmseqs"] = "single" # @param ["single", "mmseqs"]
    ligand_smiles: Optional[str] = None #""  # @param {type:"string"}
    ligand_ccd: Optional[str] = None #"SAM"  # @param {type:"string"}
    nucleic_seq: Optional[str] = None# = ""  # @param {type:"string"}
    nucleic_type: Optional[str] = None# = "dna" # @param ["dna", "rna"]
    template_path: Optional[str] = None# = ""
    template_cif_chain_id: Optional[str] = None # = "" # for mmCIF files, which chain from mmcif file to use for the template
    contact_residues: Optional[str] = None# = "|2,3,1" # @param {type:"string"}  # e.g., "1,2,5,10" on target chain 
    #@markdown - Specify which target chain residues must contact the binder (currently only supports protein contacts). For more than two chains, separate by |, e.g., "1,2,5,10 | 3,5,10".
    contact_cutoff: Optional[int] = None# = 10.0
    max_contact_filter_retries: Optional[int] = None# = 6
    no_contact_filter: Optional[bool] = None# = False
    diffuse_steps: int = 200  # @param {type:"integer"}
    recycling_steps: int = 3  # @param {type:"integer"}
    boltz_model_version: Literal["boltz2"] = "boltz2"  # @param ["boltz1", "boltz2"]
    logmd: bool = False  # @param {type:"boolean"}
    randomly_kill_helix_feature: bool = False  # @param {type:"boolean"}
    negative_helix_constant: float = 0.2  # @param {type:"number"}
    ccd_path: str = "./mols"
    
    #temperature: float = 0.1  # @param {type:"number"}
    #alanine_bias: bool = True  # @param {type:"boolean"}
    #alanine_bias_start: float = -0.5  # @param {type:"number"}
    #alanine_bias_end: float = -0.1  # @param {type:"number"}
    
    #omit_AA = "C"  # @param {type:"string"}
    #exclude_P = False  # @param {type:"boolean"}
    #percent_X = 80  # @param {type:"number"}
    #high_iptm_threshold = 0.8  # @param {type:"number"}
    #high_plddt_threshold = 0.8  # @param {type:"number"}
    
class BoltzPHWrapper:
    def __init__(
        self,
        model: nn.Module,
        ph_args: BoltzPHConfig,
    ):
        self.ph_args = ph_args
        self.data_builder = InputDataBuilder(self.ph_args)
        self.ccd_lib = load_canonicals(self.ph_args.ccd_path)
        self.device = self.ph_args.device
        self.model = model
        self.reset()

    def reset(self):
        self.data, self.pocket_conditioning = self.data_builder.build()
        
    def run_prediction(self, seq, chain, pdb_filename):
        self._update_binder_sequence(self.data, seq, chain)
        output, structure = self._run_prediction(
            self.data, chain, None, False, None, 
            self.ccd_lib, self.ph_args.ccd_path, self.model, 
            self.ph_args.randomly_kill_helix_feature, self.ph_args.negative_helix_constant,
            self.device, "boltz2", self.pocket_conditioning
        )

        self._save_pdb(structure, output['coords'], output["plddt"].detach().cpu().numpy()[0], pdb_filename)
        return output, structure

    #See https://github.com/yehlincho/Protein-Hunter/blob/main/boltz_ph/model_utils.py
    @staticmethod
    def _run_prediction(
        data,
        binder_chain,
        seq=None,
        logmd=False,
        name=None,
        ccd_lib=None,
        ccd_path=None,
        boltz_model=None,
        randomly_kill_helix_feature=False,
        negative_helix_constant=0.1,
        device="cpu",
        boltz_model_version="boltz2",
        pocket_conditioning=False,
    ):
        """Parses data, generates batch, and runs a single Boltz prediction step."""
        # 1. Update sequence if provided
        if seq is not None:
            # Assumes data["sequences"] is sorted by chain ID where binder_chain is in the position corresponding to its CHAIN_TO_NUMBER value
            try:
                binder_idx = CHAIN_TO_NUMBER.get(binder_chain, None)
                if binder_idx is not None and len(data["sequences"]) > binder_idx:
                     data["sequences"][binder_idx]["protein"]["sequence"] = seq
                else:
                    # Fallback search if sorting is unexpected
                    for entry in data["sequences"]:
                        if "protein" in entry and binder_chain in entry["protein"].get("id", []):
                            entry["protein"]["sequence"] = seq
                            break
                    else:
                        raise KeyError(f"Binder chain {binder_chain} not found in sequences for update.")
    
            except Exception as e:
                print(f"Error updating sequence in data dict: {e}")
                
        # 2. Parse data schema
        target = parse_boltz_schema(
            name,
            data,
            ccd_lib,
            ccd_path,
            boltz_2=boltz_model_version == "boltz2",
        )
        
        # 3. Generate batch and structure
        batch, structure = BoltzPHWrapper._get_batch(
            target,
            ccd_path,
            ccd_lib,
            pocket_conditioning=pocket_conditioning,
        )
        
        # 4. Move batch to device
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        
        # 5. Run prediction
        output = boltz_model.predict_step(
            batch,
            batch_idx=0,
            dataloader_idx=0,
            randomly_kill_helix_feature=randomly_kill_helix_feature,
            negative_helix_constant=negative_helix_constant,
            binder_chain=binder_chain,
            logmd=logmd,
            structure=structure,
        )
        return output, structure

    @staticmethod
    def _update_binder_sequence(data_cp, new_seq, binder_chain):
        for seq_entry in data_cp["sequences"]:
            if (
                "protein" in seq_entry
                and binder_chain in seq_entry["protein"]["id"]
            ):
                seq_entry["protein"]["sequence"] = new_seq
                return
        # Should not happen if data_cp is built correctly
        raise ValueError("Binder chain not found in data dictionary.")

    @staticmethod
    def _save_pdb(structure, coords, plddts, filename):
        """Saves the predicted structure coordinates to a PDB file."""
        structure.atoms["coords"] = (
            coords[0].detach().cpu().numpy()[: structure.atoms["coords"].shape[0]]
        )
        
        with open(filename, "w") as f:
            f.write(to_pdb(structure, plddts, boltz2=True))

    @staticmethod
    def sample_seq(length: int, exclude_P: bool = True, frac_X: float = 0.0) -> str:
        """Samples a random sequence of the given length, optionally excluding Proline (P) and including 'X' residues."""
        aas = "ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")
        num_x = round(length * frac_X)
        pool = aas if aas else "X"
        seq_list = ["X"] * num_x + random.choices(pool, k=length - num_x)
        random.shuffle(seq_list)
        return "".join(seq_list)
    
    #See https://github.com/yehlincho/Protein-Hunter/blob/main/boltz_ph/model_utils.py
    @staticmethod
    def load_model(
        checkpoint_path,
        device,
        no_potentials: bool = True,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
    ):
        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
            "max_parallel_samples": 1,
        }

        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.638

        steering_args = BoltzSteeringParams()
        if no_potentials:
            steering_args.fk_steering = False
            steering_args.physical_guidance_update = False
            steering_args.contact_guidance_update = False
        # else: defaults (fk=False, physical=False, contact=True) match
        # Protein-Hunter's binder mode with contact steering enabled.

        pairformer_args = PairformerArgsV2()
        pairformer_args.v2 = True
        pairformer_args.activation_checkpointing = True

        msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
        msa_args.activation_checkpointing = True

        model_module = Boltz2.load_from_checkpoint(
            checkpoint_path=os.path.expanduser(checkpoint_path),
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
            use_templates=True,
            use_templates_v2=True,
            use_trifast=False,
            max_parallel_samples=1,
            steering_args=asdict(steering_args),
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
        ).eval()

        return model_module
    
    #From https://github.com/yehlincho/Protein-Hunter/blob/main/boltz_ph/model_utils.py
    @staticmethod
    def _get_batch(
        target,
        ccd_path,
        ccd_lib,
        max_seqs=0,
        pocket_conditioning=False,
        keep_record=False,
    ):
        max_seqs = 4096
        structure = target.structure
    
        coords = np.array([(atom["coords"],) for atom in structure.atoms], dtype=Coords)
        ensemble = np.array([(0, len(coords))], dtype=Ensemble)
    
        structure = StructureV2(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            coords=coords,
            ensemble=ensemble,
        )

        msas = {}
        for chain in target.record.chains:
            msa_id = chain.msa_id
            if msa_id != -1:
                msa = np.load(msa_id)
                msas[chain.chain_id] = MSA(**msa)
    
        input = Input(
            structure,
            msas,
            record=target.record,
            residue_constraints=target.residue_constraints,
            templates=target.templates,
            extra_mols=target.extra_mols,
        )
    
        tokenizer = Boltz2Tokenizer()
        featurizer = Boltz2Featurizer()
    
        tokenized = tokenizer.tokenize(input)
    
        # seed = 42
        # random = np.random.default_rng(seed)
        random = np.random.default_rng()
        
        molecules = {}
        molecules.update(ccd_lib)
        molecules.update(input.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(ccd_path, mol_names))
        
        options = target.record.inference_options
        if pocket_conditioning:
            pocket_constraints, contact_constraints = (
                options.pocket_constraints,
                options.contact_constraints,
            )
        
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                inference_contact_constraints=contact_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        
        else:
            pocket_constraints = None
        
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
            
        if keep_record:
            batch["record"] = target.record
    
        return batch, structure

    @staticmethod
    def compute_iptm(output, binder_chain="A"):
        """Extract mean inter-chain ipTM for the binder from Boltz2 output.

        Returns 0.0 for single-chain (monomer) predictions.
        """
        pair_chains = output.get("pair_chains_iptm")
        if pair_chains is None or len(pair_chains) <= 1:
            return 0.0
        binder_idx = CHAIN_TO_NUMBER[binder_chain]
        values = [
            (
                pair_chains[binder_idx][i].detach().cpu().item()
                + pair_chains[i][binder_idx].detach().cpu().item()
            )
            / 2.0
            for i in pair_chains
            if i != binder_idx
        ]
        return float(np.mean(values)) if values else 0.0

    #https://github.com/yehlincho/Protein-Hunter/blob/main/boltz_ph/model_utils.py
    @staticmethod
    def aggressive_memory_cleanup():
        """Performs aggressive CUDA and Python memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.synchronize()
    
        for _ in range(3):
            gc.collect()
    
        # Reset torch dynamo cache if available
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'reset'):
            torch._dynamo.reset()
        
        # Clear cublas workspaces if function exists
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()

    #https://github.com/yehlincho/Protein-Hunter/blob/main/boltz_ph/model_utils.py
    @staticmethod
    def clean_memory():
        """Wrapper for general garbage collection and aggressive cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        BoltzPHWrapper.aggressive_memory_cleanup()

#From https://github.com/yehlincho/Protein-Hunter/blob/main/boltz_ph/pipeline.py
class InputDataBuilder:
    """Handles parsing command-line arguments and constructing the base Boltz input data dictionary."""

    def __init__(self, args):
        self.args = args
        self.protein_hunter_save_dir = self.args.save_dir
        """
        self.save_dir = (
            args.save_dir if args.save_dir else f"./results_boltz/{args.name}"
        )
        self.protein_hunter_save_dir = f"{self.save_dir}/0_protein_hunter_design"
        os.makedirs(self.protein_hunter_save_dir, exist_ok=True)
        """

    def _process_sequence_inputs(self):
        """
        Parses and groups protein sequences and MSAs from command line arguments.
        Ensures protein_seqs_list and protein_msas_list are aligned and padded.
        """
        a = self.args
        
        protein_seqs_list = smart_split(a.protein_seqs) if a.protein_seqs else []

        # Handle special "empty" case: --protein_msas "empty" applies to all seqs
        if a.msa_mode == "single":
            protein_msas_list = ["empty"] * len(protein_seqs_list)
        elif a.msa_mode == "mmseqs":
            protein_msas_list = ["mmseqs"] * len(protein_seqs_list)
        else:
            raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        return protein_seqs_list, protein_msas_list


    def build(self):
        """
        Constructs the base Boltz input data dictionary (sequences, templates, constraints).
        """
        a = self.args

        if a.mode == "unconditional":
            data = self._build_unconditional_data()
            pocket_conditioning = False
        else:
            data, pocket_conditioning = self._build_conditional_data()

        # Sort sequences by chain ID for consistent processing
        data["sequences"] = sorted(
            data["sequences"], key=lambda entry: list(entry.values())[0]["id"][0]
        )

        return data, pocket_conditioning


    def _build_unconditional_data(self):
        """Constructs data for unconditional binder design."""
        data = {
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": "X",
                        "msa": "empty",
                    }
                }
            ]
        }
        return data


    def _build_conditional_data(self):
        """Constructs data for conditioned design (binder + target + optional non-protein)."""
        a = self.args
        protein_seqs_list, protein_msas_list = (
            self._process_sequence_inputs()
        )
        sequences = []
        
        # Assign chain IDs to proteins first
        protein_chain_ids = [chr(ord('B') + i) for i in range(len(protein_seqs_list))]
        
        # Find next available chain letter
        next_chain_idx = len(protein_chain_ids)
        
        ligand_chain_id = None
        if a.ligand_smiles or a.ligand_ccd:
            ligand_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
            
        nucleic_chain_id = None
        if a.nucleic_seq:
            nucleic_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
        # --- END NEW ---

        # Step 1: Determine canonical MSA for each unique target sequence
        seq_to_indices = defaultdict(list)
        for idx, seq in enumerate(protein_seqs_list):
            if seq:
                seq_to_indices[seq].append(idx)
        
        seq_to_final_msa = {}
        for seq, idx_list in seq_to_indices.items():
            chosen_msa = next(
                (
                    protein_msas_list[i]
                    for i in idx_list
                ),
                None
            )
            chosen_msa = chosen_msa if chosen_msa is not None else ""

            if chosen_msa == "mmseqs":
                pid = protein_chain_ids[idx_list[0]]
                msa_value = process_msa(pid, seq, Path(self.protein_hunter_save_dir))
                seq_to_final_msa[seq] = str(msa_value)
            elif chosen_msa == "empty":
                seq_to_final_msa[seq] = "empty"
            else:
                raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        # Step 2: Build sequences list for target proteins
        for i, (seq, msa) in enumerate(zip(protein_seqs_list, protein_msas_list)):
            if not seq:
                continue
            pid = protein_chain_ids[i]
            final_msa = seq_to_final_msa.get(seq, "empty")
            sequences.append(
                {
                    "protein": {
                        "id": [pid],
                        "sequence": seq,
                        "msa": final_msa,
                    }
                }
            )

        # Step 3: Add binder chain
        sequences.append(
            {
                "protein": {
                    "id": ["A"], # Hardcoded 'A'
                    "sequence": "X",
                    "msa": "empty",
                    "cyclic": a.cyclic,
                }
            }
        )

        # Step 4: Add ligands/nucleic acids
        if a.ligand_smiles:
            sequences.append(
                {"ligand": {"id": [ligand_chain_id], "smiles": a.ligand_smiles}}
            )
        elif a.ligand_ccd:
            sequences.append({"ligand": {"id": [ligand_chain_id], "ccd": a.ligand_ccd}})
        if a.nucleic_seq:
            sequences.append(
                {a.nucleic_type: {"id": [nucleic_chain_id], "sequence": a.nucleic_seq}}
            )

        # Step 5: Add templates
        templates = self._build_templates(protein_chain_ids)

        data = {"sequences": sequences}
        if templates:
            data["templates"] = templates

        # Step 6: Add constraints (pocket conditioning)
        pocket_conditioning = bool(a.contact_residues and a.contact_residues.strip())
        if pocket_conditioning:
            contacts = []
            residues_chains = a.contact_residues.split("|")
            for i, residues_chain in enumerate(residues_chains):
                residues = residues_chain.split(",")
                contacts.extend([
                    [protein_chain_ids[i], int(res)]
                    for res in residues
                    if res.strip() != ""
                ])
            constraints = [{"pocket": {"binder": "A", "contacts": contacts}}]
            data["constraints"] = constraints

        return data, pocket_conditioning

    def _build_templates(self, protein_chain_ids):
        """
        Constructs the list of template dictionaries.
        """
        a = self.args
        templates = []
        if a.template_path:
            template_path_list = smart_split(a.template_path)
            # We use the internal protein_chain_ids list
            template_cif_chain_id_list = (
                smart_split(a.template_cif_chain_id)
                if a.template_cif_chain_id
                else []
            )
            
            # Use protein_chain_ids to determine the number of expected templates
            num_proteins = len(protein_chain_ids)
            
            # Pad template paths list to match number of proteins
            if len(template_path_list) != num_proteins:
                print(f"Warning: Mismatch between number of proteins ({num_proteins}) and template paths ({len(template_path_list)}). Padding with empty entries.")
                while len(template_path_list) < num_proteins:
                    template_path_list.append("")
            
            # Pad cif chains list to match number of proteins
            if len(template_cif_chain_id_list) != num_proteins:
                print(f"Warning: Mismatch between number of proteins ({num_proteins}) and template CIF chains ({len(template_cif_chain_id_list)}). Padding with empty entries.")
                while len(template_cif_chain_id_list) < num_proteins:
                    template_cif_chain_id_list.append("")
            
            # Now, iterate up to num_proteins, linking them
            for i in range(num_proteins):
                template_file_path = template_path_list[i]
                if not template_file_path:
                    continue # Skip if no template path for this protein
                    
                template_file = get_cif(template_file_path)
                
                t_block = (
                    {"cif": template_file}
                    if template_file.endswith(".cif")
                    else {"pdb": template_file}
                )
                
                t_block["chain_id"] = protein_chain_ids[i] # e.g., 'B'
                
                # Only add cif_chain_id if provided for this template
                cif_chain = template_cif_chain_id_list[i]
                if cif_chain:
                    t_block["cif_chain_id"] = cif_chain # e.g., 'P'
                
                templates.append(t_block)
                
        return templates