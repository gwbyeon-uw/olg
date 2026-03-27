#!/usr/bin/env python
"""Minimal smoke test: S/J extended-alphabet OLG design.

Two proteins co-designed with ProteinMPNN on two separate scaffolds.

protein1: 100 AA, no force_start/stop.
protein2: 100-AA PDB extended to 102 via pad=(1,1), force_start + force_stop;
  DesignConfig.offset=5 shifts protein2's start 5 NT positions into the sequence.

Serine is split into two tokens via a restricted codon table:
  S  — TCN codons (standard serine)
  J  — AGT / AGC codons (restricted serine)

Per-frame aa_bias enforces:
  protein1: no S  (forced to use J-serine codons, AGT/AGC)
  protein2: no J  (forced to use S-serine codons, TCN)

Checks:
  1. protein2 starts with M            (force_start=True)
  2. No 'S' in protein1                (S excluded by aa_bias)
  3. No 'J' in protein2                (J excluded by aa_bias)
  4. Lengths correct                   (100 AA for p1, 102 AA for p2)
  5. MPNN scores finite and positive
  6. NT sequence valid ACGT
  7. Padded positions have mask=0
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from olg import OLGDesign
from olg.config import DesignConfig, ProteinConfig
from olg.constants import Constants, build_restricted_codon_table
from olg.wrappers.proteinmpnn import WrapperProteinMPNN

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights" / "proteinmpnn" / "v_48_010.pt"
PDB1 = str(HERE / "104_16.pdb")
PDB2 = str(HERE / "131_12.pdb")  # 100-residue PDB; extended to 102 via pad=(1,1)
P2_PAD = (1, 1)  # 1 Gly prepended, 1 Gly appended → 102 total positions
P2_LENGTH = 102  # config.length includes padding
N_GIBBS = 9
OFFSET = 5   # NT offset for protein2: shifts its start position in NT space


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Codon table: AGT and AGC reassigned from S to J; all other codons unchanged.
    codon_table = build_restricted_codon_table({"J": ["AGT", "AGC"]})

    # Extended alphabet: standard 21 + J. Any ordering is valid;
    # indices are derived from letter identity, not position.
    alphabet = list(Constants.DEFAULT_ALPHABET) + ["J"]
    alphabet_index = {a: i for i, a in enumerate(alphabet)}

    # aa_bias: exclude S from protein1 and J from protein2.
    p1_aa_bias = torch.zeros(len(alphabet), device=device)
    p1_aa_bias[alphabet_index["S"]] = Constants.MIN_LOGIT

    p2_aa_bias = torch.zeros(len(alphabet), device=device)
    p2_aa_bias[alphabet_index["J"]] = Constants.MIN_LOGIT

    # logit_weight=0 and logit_bias for S/G at extension positions for protein2.
    # Extension positions: padded positions minus forced start.
    #   N-term: positions 1..pad_n-1 (position 0 is forced Met)
    #   C-term: positions (length - pad_c)..(length - 1)
    #   (force_stop adds stop BEYOND config.length; all P2_LENGTH positions are real protein)
    p2_logit_weight = torch.ones(P2_LENGTH, device=device)
    p2_logit_bias = torch.zeros((P2_LENGTH, len(alphabet)), device=device)

    ext_positions = []
    ext_positions.extend(range(1, P2_PAD[0]))           # N-term extensions (skip Met at 0)
    ext_positions.extend(range(P2_LENGTH - P2_PAD[1],    # C-term extensions
                               P2_LENGTH))
    for pos in ext_positions:
        p2_logit_weight[pos] = 0.0
        p2_logit_bias[pos, alphabet_index["S"]] = 1.0
        p2_logit_bias[pos, alphabet_index["G"]] = 1.0

    config = DesignConfig(
        device=device,
        codon_table=codon_table,
        alphabet=alphabet,
        rand_base=42,
        tqdm_disable=True,
        offset=OFFSET,
        protein1=ProteinConfig(device=device, length=100,
                               alphabet_size=len(alphabet), aa_bias=p1_aa_bias),
        protein2=ProteinConfig(device=device, length=P2_LENGTH,
                               alphabet_size=len(alphabet), aa_bias=p2_aa_bias,
                               force_start=True, force_stop=True,
                               logit_weight=p2_logit_weight,
                               logit_bias=p2_logit_bias),
    )

    print(f"Alphabet ({len(alphabet)}): {''.join(alphabet)}")
    print("Serine split:", {c: aa for c, aa in sorted(codon_table.items()) if aa in ("S", "J")})

    print(f"\nLoading ProteinMPNN from {WEIGHTS} ...")
    model = WrapperProteinMPNN._load_proteinmpnn_model(str(WEIGHTS), device, ca_only=True)

    olg = OLGDesign(config)
    extra_aa_map = {"J": "S"}  # J → native S in ProteinMPNN
    olg.initialize_decoder("ProteinMPNN", frame=0, model=model,
                            ca_only=True, pdb_path=PDB1, extra_aa_map=extra_aa_map)
    olg.initialize_decoder("ProteinMPNN", frame=1, model=model,
                            ca_only=True, pdb_path=PDB2, extra_aa_map=extra_aa_map,
                            pad=P2_PAD)

    with torch.inference_mode():
        olg.decode_all()
        s1, s2 = olg.get_scores()
        scores_pll = [[s1, s2]]
        print(f"\n[init] p1={s1:.4f}  p2={s2:.4f}")

        for i in range(N_GIBBS):
            ordering = olg.get_next_order("entropy")
            w1, w2 = olg.get_next_weight(scores_pll)
            olg.decode_all_gibbs(next_order=ordering, weight=(w1, w2))
            s1, s2 = olg.get_scores()
            scores_pll.append([s1, s2])
            prot1, prot2 = olg.translate_sequences()
            print(f"[{i+1:2d}]   p1={s1:.4f}  p2={s2:.4f}  "
                  f"| S/J p1={prot1.count('S')}/{prot1.count('J')}  "
                  f"S/J p2={prot2.count('S')}/{prot2.count('J')}")

    nt_seq, _ = olg.string_quartet()
    prot1, prot2 = olg.translate_sequences()
    s1, s2 = float(olg.decoders[0].get_score()), float(olg.decoders[1].get_score())

    # Padded position mask check
    p2_wrapper = olg.decoders[1]
    off = p2_wrapper.target_chain_offset
    p2_mask = p2_wrapper.mask[0]
    pad_n_mask = p2_mask[off : off + P2_PAD[0]]
    pad_c_mask = p2_mask[off + P2_LENGTH - P2_PAD[1] : off + P2_LENGTH]

    checks = [
        (prot2 and prot2[0] == "M",       "protein2 starts with M"),
        ("S" not in prot1,                 "no S in protein1"),
        ("J" not in prot2,                 "no J in protein2"),
        (len(prot1) == 100,                f"protein1 length = {len(prot1)}"),
        (len(prot2) == P2_LENGTH,          f"protein2 length = {len(prot2)}"),
        (0.0 < s1 < 100.0,                f"p1 score = {s1:.4f}"),
        (0.0 < s2 < 100.0,                f"p2 score = {s2:.4f}"),
        (nt_seq and all(c in "ACGT" for c in nt_seq), f"NT valid ({len(nt_seq)} nt)"),
        (pad_n_mask.sum() == 0 and pad_c_mask.sum() == 0, "padded positions mask=0"),
    ]

    print(f"\n{'='*68}")
    failed = []
    for i, (ok, msg) in enumerate(checks, 1):
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {i}: {msg}")
        if not ok:
            failed.append(msg)

    print(f"{'='*68}")
    if failed:
        print(f"FAILED ({len(failed)} error(s))")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
