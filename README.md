# OLGDesign

Design two proteins whose coding sequences overlap in different reading frames of the same DNA.

![OLG iterative design](olg_iteration.gif)

## Background

This framework uses generative protein models to design synthetic OLGs. The core algorithm is an iterative constrained sampling procedure over quartet positions (4-nucleotide windows that span two codons in overlapping frames), using model logits from both frames simultaneously.

## Supported models

| Model | Notes |
|-------|-------|
| **ProteinMPNN** | Supports complexes via tied decoding |
| **MSA Pairformer** | Requires `msa-pairformer` package; MSA-based with contact prediction |
| **ESM3** | Requires `esm` package; supports function-conditioned generation |
| **EvoDiff-seq** | Requires `evodiff` package; single-sequence OADM (no MSA needed) |
| **CoFlow** | Requires `coflow` package |
| **EvoDiff-MSA** | Requires `evodiff` package; supports shared MSA mode |
| **GREMLIN** | Built-in, no extra deps |
| **APEX** | AMP MIC regressor; GREMLIN-style enumeration decoder (vendored model, no submodule) |
| **ZeroOrder** | Built-in dummy model (uniform logits) |

Any combination of models can be used (one per frame).

## Installation

```bash
pip install -e .

```

## Quick start

```python
import torch
from olg import *
from olg.config import DesignConfig
from olg.wrappers.proteinmpnn import WrapperProteinMPNN

device = torch.device("cuda:0")

# Load model
model = WrapperProteinMPNN._load_proteinmpnn_model(
    "weights/proteinmpnn/v_48_010.pt", device, ca_only=True
)

# Configure
config = DesignConfig.from_yaml("my_design.yaml")  # or DesignConfig() for defaults
olg = OLGDesign(config)

# Initialize both frames
olg.initialize_decoder("ProteinMPNN", frame=0, model=model, ca_only=True, pdb_path="scaffold_1.pdb")
olg.initialize_decoder("ProteinMPNN", frame=1, model=model, ca_only=True, pdb_path="scaffold_2.pdb")

# Design
with torch.inference_mode():
    olg.decode_all()

    scores = [[*olg.get_scores()]]
    for i in range(9):
        order = olg.get_next_order("entropy")
        w1, w2 = olg.get_next_weight(scores)
        olg.decode_all_gibbs(next_order=order, weight=(w1, w2))
        scores.append([*olg.get_scores()])

    prot_f1, prot_f2 = olg.get_prot_seq()
    nuc_seq, _ = olg.string_quartet()
```

## Configuration

All defaults live in `src/olg/base.yaml`. User YAML files only need to specify overrides:

```yaml
# my_design.yaml
temperature: 0.1
arrangement: 0
protein1:
  length: 120
  repetition_penalty: 1.2
protein2:
  length: 95
  force_stop: true
```

Load with `DesignConfig.from_yaml("my_design.yaml")`. Calling `DesignConfig.from_yaml()` with no arguments returns base defaults.

### Design-level parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `cuda:0` | Torch device |
| `arrangement` | `0` | Reading frame arrangement (see below) |
| `offset` | `0` | Distance between protein N-termini (in amino acids) |
| `codon_table` | `Standard` | NCBI genetic code name, or a `{codon: AA}` dict |
| `decoding_mode` | `1` | `0`=random, `1`=overlap first, `2`=overlap last |
| `temperature` | `1.0` | Sampling temperature for joint logits |
| `top_p` | `0.0` | Top-p nucleus sampling (`0.0`=greedy) |
| `complexed` | `false` | Tied decoding for ProteinMPNN multimers |
| `shared` | `false` | Shared MSA mode for EvoDiff |
| `balancer_max_weight` | `2.0` | Max frame-balancing weight |
| `balancer_unit` | `0.5` | Frame-balancing increment |
| `balancer_threshold` | `0.15` | Score difference to trigger rebalancing |
| `rand_base` | `null` | Random seed |
| `tqdm_disable` | `false` | Suppress progress bars |

### Per-protein parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length` | `100` | Protein length (overlap region only if partial) |
| `start_offset` | `0` | Starting position within the model input |
| `force_stop` | `false` | Enforce stop codon at C-terminus |
| `force_start` | `false` | Enforce start codon at N-terminus |
| `start_codons` | `["ATG"]` | Allowed start codons |
| `fixed_positions` | `null` | Fixed residues as `[[pos, "AA"], ...]` (1-indexed) |
| `gap_positions` | `null` | Gap positions for MSA models (1-indexed) |
| `repetition_penalty` | `1.1` | Penalty multiplier per repeated AA (`1.0`=off) |
| `repetition_penalty_window` | `4` | Window for repetition counting |
| `truncate_topp` | `0.0` | Per-protein top-p before joint calculation |
| `max_pos_count` | `10000` | Max positively charged residues (H, K, R) |

Tensor fields (`logit_weight`, `logit_bias`, `aa_bias`, `max_aa_count`) can be specified in YAML as inline lists or paths to `.pt`/`.npy` files. They default to uniform/zero and are omitted from `base.yaml`.

## Extended alphabets

The OLG alphabet can be extended beyond the standard 20 amino acids + stop (`X`) to represent codon-level distinctions. A common case is splitting serine into two tokens based on which codons encode it, then restricting each frame to one group.

```python
from olg.constants import Constants, build_restricted_codon_table

# --- Codon table: reassign specific codons to new tokens ---
# AGT, AGC are taken from serine and reassigned to 'J'; TCN codons stay as 'S'.
# Any number of new letters and codon sets can be specified simultaneously.
codon_table = build_restricted_codon_table({"J": ["AGT", "AGC"]})

# Extend the alphabet with the new token. Any ordering is valid;
# indices are derived from letter identity, not position.
alphabet = list(Constants.DEFAULT_ALPHABET) + ["J"]
alphabet_index = {a: i for i, a in enumerate(alphabet)}

# --- Per-frame exclusion via aa_bias ---
# Setting aa_bias[idx] = MIN_LOGIT makes that token unsamplable in that frame.
p1_aa_bias = torch.zeros(len(alphabet), device=device)
p1_aa_bias[alphabet_index["S"]] = Constants.MIN_LOGIT   # frame 0: only J-serine (AGT/AGC)

p2_aa_bias = torch.zeros(len(alphabet), device=device)
p2_aa_bias[alphabet_index["J"]] = Constants.MIN_LOGIT   # frame 1: only S-serine (TCN)

# --- Config ---
config = DesignConfig(
    codon_table=codon_table,
    alphabet=alphabet,
    protein1=ProteinConfig(length=100, aa_bias=p1_aa_bias, ...),
    protein2=ProteinConfig(length=100, aa_bias=p2_aa_bias, force_start=True, ...),
)
olg = OLGDesign(config)

# --- extra_aa_map: tell the model wrapper how to score extended tokens ---
# ProteinMPNN has a fixed native vocabulary; map 'J' to native 'S' so it
# scores restricted-serine positions with the same logits as regular serine.
olg.initialize_decoder(
    "ProteinMPNN", frame=0, model=mpnn_model,
    ca_only=True, pdb_path="scaffold_1.pdb",
    extra_aa_map={"J": "S"},
)

# --- Translating the result ---
# Use translate_sequences() to recover S/J distinctions from the NT sequence.
# This is independent of the model wrapper vocabulary and handles all arrangements.
prot1, prot2 = olg.translate_sequences()
```

### Padding (extending proteins beyond PDB length)

When a designed protein needs to be longer than the PDB scaffold, the ProteinMPNN wrapper can inject dummy residues with NaN coordinates at the N- and/or C-terminus via the `pad` parameter. These positions get `mask=0` automatically (no structural contribution from the model), while biases and codon constraints still apply.

```python
# 100-residue PDB, but we want a 102-AA protein (1 extra on each end)
olg.initialize_decoder(
    "ProteinMPNN", frame=1, model=model,
    ca_only=True, pdb_path="scaffold.pdb",
    pad=(1, 1),  # (n_terminal, c_terminal)
)
# ProteinConfig.length must include padding: length=102
```

`config.length` **includes** padding — padded positions are real design positions in the final protein. To control amino acid selection at padded positions (where the model contributes no information), use `logit_weight` and `logit_bias`:

```python
# Zero out model logits at extension positions, bias toward S and G
logit_weight = torch.ones(length, device=device)
logit_bias = torch.zeros((length, alphabet_size), device=device)

for pos in extension_positions:
    logit_weight[pos] = 0.0
    logit_bias[pos, alphabet_index["S"]] = 1.0
    logit_bias[pos, alphabet_index["G"]] = 1.0

protein2 = ProteinConfig(
    length=length, logit_weight=logit_weight, logit_bias=logit_bias, ...
)
```

Padding is not supported with `tied=True` (complexes).

### Reading frame arrangements

| Value | Name | Description |
|-------|------|-------------|
| `0` | +1 | Protein 2 in +1 frame (same strand) |
| `1` | -1 | Protein 2 in -1 frame (reverse strand) |
| `2` | -0 | Protein 2 in -0 frame (reverse strand, in-phase) |
| `3` | +2 | Protein 2 in +2 frame (same strand) |
| `4` | -2 | Protein 2 in -2 frame (reverse strand) |

### Multi-chain context (binder design)

When designing an overlap protein that should bind a target, ProteinMPNN can condition on both the binder structure and the target structure simultaneously. Pass `fixed_chains` to keep target chains as structural context while designing only the binder chain:

```python
# Load a complex PDB with chain A (binder) and chain B (target)
olg.initialize_decoder(
    "ProteinMPNN", frame=1, model=model,
    ca_only=True, pdb_path="complex.pdb",
    fixed_chains=["B"],       # target structure as fixed context
    design_chains=["A"],      # binder chain to design
)
```

Fixed chains are encoded first in the autoregressive pass, so every logit prediction for the design chain is conditioned on the target's structure and sequence.

### MSA Pairformer

[MSA Pairformer](https://github.com/yoakiyama/MSA_Pairformer) is a lightweight MSA transformer that produces per-position amino acid logits from multiple sequence alignments. It also provides optional contact predictions (Cb-Cb and ConFind).

**Installation:**

```bash
git clone https://github.com/yoakiyama/MSA_Pairformer.git
pip install -e MSA_Pairformer
```

**Basic usage:**

```python
from olg.wrappers.msa_pairformer import WrapperMSAPairformer

# Load model (downloads weights from HuggingFace on first run)
model = WrapperMSAPairformer._load_model(device, weights_dir="weights/msa_pairformer")

# Parse MSA (list of aligned sequences from a3m/fasta file)
headers, seqs = WrapperMSAPairformer.parse_fasta("my_protein.a3m")
msa_seqs = [str(s) for s in seqs]

# Initialize decoder
olg.initialize_decoder(
    "MSAPairformer", frame=0, model=model,
    msa_seqs=msa_seqs,
    msa_n_seq=128,             # MSA depth (default 128; 64-256 recommended)
    msa_max_length=100,        # sequence length (must match config.length minus padding)
    msa_selection_type='MaxHamming',  # diversity selection: 'random', 'MaxHamming', 'MaxHammingI'
    seed_from_msa=True,        # seed query row from MSA (recommended)
)
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `msa_n_seq` | `128` | MSA depth. 64-256 recommended for good contact prediction |
| `msa_max_length` | — | Sequence length matching the MSA columns |
| `msa_selection_type` | `'random'` | `'MaxHamming'` for diverse subsampling |
| `seed_from_msa` | `False` | Seed query from MSA query sequence. Recommended — keeps masking close to training distribution |
| `use_bfloat16` | `True` | bfloat16 autocast for GPU inference |
| `pad` | `(0, 0)` | N/C-terminal padding (positions excluded from model, steered via `logit_weight`/`aa_bias`) |

**Contact predictions** (optional, outside DecoderProtocol):

```python
contacts = olg.decoders[0].get_contacts()
cb_contacts = contacts['cb_contacts']         # [1, L, L] Cb-Cb contacts
confind_contacts = contacts['confind_contacts']  # [1, L, L] interface contacts
```

**Notes:**
- MSA Pairformer is trained with 15% BERT-style masking. Starting from a fully masked query (default) is out-of-distribution. Use `seed_from_msa=True` for better initial sequences.
- Padding positions are trimmed before the model forward pass and zero-padded on output. Use `logit_weight=0` and `aa_bias` to steer padded positions.
- The model runs a full forward pass per decoding position (~80-180ms depending on MSA depth). Gibbs refinement iterations each take ~20s for 100-residue proteins on an L4 GPU.

### ESM3 with function conditioning

ESM3 supports function-conditioned generation via InterPro annotations and keywords. This enables generating sequences with specific functional properties (e.g., antimicrobial activity) built into the generative model itself.

```python
olg.initialize_decoder(
    "ESM3", frame=0, model=esm3_model,
    function_annotations=[
        {"label": "IPR004275", "start": 1, "end": 23},   # Frog AMP propeptide
        {"label": "antimicrobial peptide", "start": 1, "end": 23},
    ],
)
```

AMP-related InterPro entries available in ESM3:

| InterPro ID | Description |
|------------|-------------|
| `IPR004275` | Frog antimicrobial peptide, propeptide |
| `IPR012520` | Frog antimicrobial peptide, brevinin-1 type |
| `IPR012521` | Frog antimicrobial peptide, brevinin-2/esculentin type |
| `IPR012524` | Abaecin, antimicrobial peptide |
| `IPR001542` | Defensin, invertebrate/fungal |
| `IPR010851` | Defensin-like protein |

Keywords: `antimicrobial`, `antimicrobial peptide`, `defensin`, `bacteriocin`.

Function conditioning can be combined with TAG guidance for dual control: ESM3 generates from the functional family distribution, while the classifier steers toward specific properties (e.g., potency, low hemolysis).

### APEX (AMP potency decoder)

[APEX](https://gitlab.com/machine-biology-group-public/apex) is a per-organism MIC regressor (34 strains). The wrapper uses it like GREMLIN — at each position it enumerates all 20 amino acids, scores the candidates with the ensemble, and builds a sampling distribution from predicted potency. It is a **frame decoder**, so the designed peptide is driven directly toward low MIC against a chosen organism.

```python
from olg.wrappers.apex import WrapperAPEX

# Load the 40-model ensemble (checkpoints from the APEX GitLab repo)
models = WrapperAPEX._load_ensemble("weights/apex", device)

olg.initialize_decoder(
    "APEX", frame=0, model=models,
    organism="E. coli ATCC11775",  # single target organism (no aggregation)
    temperature=0.5,               # softmax temperature on -log10(MIC)
)
```

**Energy function** — for each candidate AA at position `t`, the ensemble predicts MIC (µM) for the target organism (averaged in linear µM space). The Boltzmann mapping is:

```
score(aa) = -log10(MIC_target(aa)) / temperature      # higher = more potent
P(aa | context) = softmax(score)
```

so energy `E(aa) = log10(MIC_target(aa))` — lower MIC → lower energy → higher sampling probability.

**Notes:**
- APEX alphabet is the 20 standard amino acids only (no gap/stop/extended); other OLG alphabet entries (`X`, `J`, …) are held at `MIN_LOGIT` and never sampled.
- Max peptide length 50; no MSA, no gaps.
- The model class is vendored (`wrappers/_vendored/apex/model.py`) — no git submodule required. Download the 40 checkpoints into `weights/apex/`.
- `decoders[frame].get_mic(organism=...)` returns the predicted MIC of the current sequence for any organism (see `WrapperAPEX.organism` list).
- Two frames can target different pathogens simultaneously (e.g. E. coli // S. aureus) in overlapping reading frames.

### Classifier-guided decoding (TAG)

The `GuidedWrapper` adds [Taylor-Approximated Guidance](https://arxiv.org/abs/2406.01572) to any decoder, biasing sampling toward sequences with desired properties (e.g., antimicrobial activity, low hemolysis) without modifying the base model.

```python
from olg.wrappers.guided import GuidedWrapper

# Initialize base decoder as usual
olg.initialize_decoder("MSAPairformer", frame=0, model=model, ...)

# Wrap with guidance
olg.decoders[0] = GuidedWrapper(
    olg.decoders[0],
    classifiers=[amp_classifier, hemo_classifier],
    guide_temp=0.5,          # lower = stronger guidance
    weights=[1.0, -1.0],    # maximize AMP activity, minimize hemolysis
)

# Design as normal — guidance is transparent to OLG
olg.decode_all()
```

The wrapper only intercepts `decode_next()` to add the classifier gradient signal. All state management, scoring, and sequence tracking delegates to the inner decoder. `get_score()` returns the unguided generative model score (for frame balancing); classifier scores should be evaluated separately.

**Classifier interface** — any classifier implementing `GuidanceClassifier`:

```python
class MyClassifier:
    vocab_size: int = 21                                # classifier's token vocabulary size
    olg_to_clf: torch.Tensor                            # [olg_alphabet_size] mapping
    def encode_tokens(self, token_ids) -> torch.Tensor: # native → classifier tokens
        ...
    def log_prob(self, x_onehot, t) -> torch.Tensor:   # [B, L, V] → [B] log probability
        ...
```

The classifier must accept one-hot input (`requires_grad=True`) and return a differentiable scalar log-probability. Use `StraightThroughEmbedding` to make embedding-based classifiers (ESM-2, ProtBERT) differentiable through discrete tokens.

### Structure hallucination with Boltz2

The `structure/boltz.py` module wraps [Boltz2](https://github.com/jwohlwend/boltz) for structure prediction, adapted from [Protein-Hunter](https://github.com/yehlincho/Protein-Hunter). It supports both monomer fold prediction and multi-chain binder hallucination:

```python
from olg.structure.boltz import BoltzPHWrapper, BoltzPHConfig

# Load model (no_potentials=False enables contact steering for binder mode)
boltz_model = BoltzPHWrapper.load_model("boltz2_conf.ckpt", device, no_potentials=False)

# Configure for binder design
config = BoltzPHConfig(
    mode="binder",                         # multi-chain: A=binder, B=target
    protein_seqs="MKTLLF...",              # target protein sequence
    msa_mode="single",                     # or "mmseqs" for MSA search
    contact_residues="5,10,15",            # optional: pocket steering
    randomly_kill_helix_feature=True,      # encourage diverse folds
    ccd_path="weights/mols",
)

# Predict complex
boltz_ph = BoltzPHWrapper(boltz_model, config)
boltz_ph.reset()
output, structure = boltz_ph.run_prediction(binder_seq, "A", "complex.pdb")

# Extract binding metrics
iptm = BoltzPHWrapper.compute_iptm(output, "A")   # inter-chain TM-score
plddt = output["plddt"].mean().item()               # predicted LDDT
```

Binder hallucination tricks (from Protein-Hunter):
- **Multi-chain input**: target sequence as chain B; binder as chain A with "X" placeholder
- **Contact steering**: pocket constraints guide diffusion toward specific target residues
- **Helix killing**: disrupts i→i+4 pairwise features on the binder chain, encouraging diverse folds
- **ipTM tracking**: inter-chain predicted TM-score as the binding quality metric

## Project structure

```
src/olg/
  design.py              # OLGDesign — main orchestrator
  config.py              # DesignConfig / ProteinConfig dataclasses + YAML I/O
  base.yaml              # Default configuration values
  constants.py           # Alphabets, quartets, codon tables, enums
  coordinates.py         # Frame coordinate mapping
  compatibility.py       # Codon compatibility constraints
  balancer.py            # Frame weight balancer
  exceptions.py          # Exception hierarchy
  genetic_code_randomizer.py  # Alternative genetic code generation
  wrappers/
    protocol.py          # DecoderProtocol (typing.Protocol)
    base_wrapper.py      # BaseWrapper mixin + ZeroOrderWrapper
    proteinmpnn.py       # ProteinMPNN wrapper (supports multi-chain with fixed context)
    esm3.py              # ESM3 wrapper (supports function-conditioned generation)
    evodiff_seq.py       # EvoDiff single-sequence OADM wrapper
    coflow.py            # CoFlow wrapper
    evodiff.py           # EvoDiff-MSA wrapper
    msa_pairformer.py    # MSA Pairformer wrapper (MSA-based + contact prediction)
    guided.py            # GuidedWrapper — TAG classifier guidance for any decoder
    gremlin.py           # GREMLIN wrapper
    apex.py              # APEX AMP MIC regressor (GREMLIN-style enumeration decoder)
    _vendored/apex/      # vendored APEX model class (for unpickling checkpoints)
  structure/
    boltz.py             # Boltz2 wrapper (monomer + binder hallucination)
```

## Citation

```bibtex
@article{olg2025,
  title   = {Design of overlapping genes using deep generative models of protein sequences},
  author  = {},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.05.06.652464}
}
```
