# Smoke Test: S/J Extended Alphabet with Padding

## Design goal

Co-design two overlapping proteins where serine codons are partitioned between frames:
- **protein1** (100 AA): uses only J-serine (AGT/AGC codons)
- **protein2** (102 AA): uses only S-serine (TCN codons), extended 1 residue on each terminus beyond its 100-residue PDB scaffold

## Parameters

### Extended alphabet and codon table

Serine is split into two tokens via `build_restricted_codon_table({"J": ["AGT", "AGC"]})`. The OLG alphabet becomes 22 tokens (standard 21 + J).

`extra_aa_map={"J": "S"}` tells ProteinMPNN to score J positions using its native serine logits. After remapping to OLG-internal space (`[1, 22]`), S (index 17) and J (index 21) receive identical raw model logits. Per-frame `aa_bias` then differentiates them:

- protein1: `aa_bias[S] = MIN_LOGIT` → S excluded, only J-serine sampled
- protein2: `aa_bias[J] = MIN_LOGIT` → J excluded, only S-serine sampled

The codon table enforces the distinction at the nucleotide level: S → TCN, J → AGT/AGC.

### Padding

```python
olg.initialize_decoder("ProteinMPNN", frame=1, model=model,
                        ca_only=True, pdb_path="131_12.pdb", pad=(1, 1))
```

`pad=(1, 1)` injects dummy Gly residues with NaN coordinates at the N- and C-terminus of the parsed PDB. NaN coords produce `mask=0` in featurization, masking attention at those positions. ProteinMPNN still outputs logits there, but they lack structural basis — `logit_weight=0` and `logit_bias` (below) override them. `config.length=102` includes padding — padded positions are real design positions.

### Logit weight and bias at extension positions

At extension positions (padded positions not consumed by forced Met/Stop), `logit_weight=0` zeroes the model logits and `logit_bias` provides equal S/G priors. Biases are added after weighting (`weight * model_logits + biases`), so they remain effective when weight is zero. With `pad=(1, 1)` no free extension positions exist; the code is generic for larger pad values.

## Checks

| # | Check | Validates |
|---|-------|-----------|
| 1 | protein2 starts with M | `force_start` with padding |
| 2 | No S in protein1 | `aa_bias` exclusion |
| 3 | No J in protein2 | `aa_bias` exclusion |
| 4 | protein1 length = 100 | Correct length |
| 5 | protein2 length = 102 | Length includes padding |
| 6-7 | MPNN scores in range | Model scoring with padding |
| 8 | NT sequence valid ACGT | Valid nucleotide output |
| 9 | Padded positions mask=0 | NaN coords → automatic masking |

## Files

- `smoke_test_sj.py` — test script
- `104_16.pdb` — protein1 scaffold (100 residues)
- `131_12.pdb` — protein2 scaffold (100 residues, extended to 102 via pad)
