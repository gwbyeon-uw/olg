#!/usr/bin/env python
"""Round-trip sanity check for the OLG decode -> DNA -> protein pipeline.

For every arrangement at several offsets, decode two proteins (ZeroOrder, no
model weights needed), then verify three INDEPENDENT protein representations
agree:

  * ``get_prot_seq()``        -- read straight from each decoder's token array (self.S)
  * ``translate_sequences()`` -- olg's own translation of the nucleotides emitted
                                 by ``string_quartet()`` (uses the configured
                                 codon table + ARRANGEMENT_CONFIG geometry)
  * Biopython ``Seq.translate`` -- external translation of the same emitted NT,
                                 extracted per frame (offset + reverse-complement
                                 for minus strands) and translated with the
                                 Standard table

The decoder leg shares only the decode step with the other two; it diverges at
the quartet -> NT -> codon path. The Biopython leg is fully external to olg's
codon machinery, so it also guards against bugs *inside* translate_sequences.
(This is the check that fails on the string_quartet selection bug: non-connecting
quartets corrupt the NT and the translation no longer matches the decoder-held
sequence.)

Stop codons are excluded from the ZeroOrder body so sequences are full-length
and compared exactly; arrangements/offsets that cannot form a valid overlap are
reported as DECODE-FAIL (distinct from a translation MISMATCH).

Run:  python tests/sanity_roundtrip.py
Exit code is nonzero if any (arrangement, offset) MISMATCHes.
"""

import sys

import torch
from Bio.Seq import Seq

from olg import OLGDesign
from olg.config import DesignConfig, ProteinConfig
from olg.constants import Arrangement, Constants

DEVICE = torch.device("cpu")
LENGTH = 30
OFFSETS = [0, 5, 10, 15, 20, 25]
ARRANGEMENTS = list(Arrangement)
RETRY = 25  # reseed budget so random ZeroOrder sequences find a compatible quartet chain


def biopython_translate(nt: str, abs_positions: list[int], nt_offset: int, reverse: bool) -> str:
    """Externally translate one frame from the emitted NT string.

    Residue i sits at absolute quartet position ``abs_positions[i]``; its codon is
    the 3 nt starting at ``3*pos + nt_offset`` (reverse-complemented for minus
    strands). Translation uses Biopython's Standard table. Stops terminate.
    """
    out = []
    for pos in abs_positions:
        codon = nt[3 * pos + nt_offset: 3 * pos + nt_offset + 3]
        if len(codon) < 3:
            break
        if reverse:
            codon = str(Seq(codon).reverse_complement())
        aa = str(Seq(codon).translate(table="Standard"))
        if aa == "*":
            break
        out.append(aa)
    return "".join(out)


def _zeroorder_logits() -> tuple[torch.Tensor, int]:
    """Uniform logits over the default alphabet with the stop token suppressed."""
    alphabet = list(Constants.DEFAULT_ALPHABET)
    x_idx = alphabet.index("X")
    logits = torch.zeros((1, len(alphabet)), device=DEVICE)
    logits[0, x_idx] = Constants.MIN_LOGIT  # never sample a stop in the body
    return logits, x_idx


def run_one(arrangement: Arrangement, offset: int, seed: int) -> dict:
    """Decode one design and return both protein representations (or a failure)."""
    cfg = DesignConfig(
        device=DEVICE, arrangement=arrangement, offset=offset,
        rand_base=seed, tqdm_disable=True,
        protein1=ProteinConfig(device=DEVICE, length=LENGTH),
        protein2=ProteinConfig(device=DEVICE, length=LENGTH),
    )
    try:
        olg = OLGDesign(cfg)
    except Exception as e:  # invalid geometry for this (arrangement, offset)
        return {"status": "BUILD-FAIL", "detail": f"{type(e).__name__}: {e}"}

    zo, _ = _zeroorder_logits()
    olg.initialize_decoder("ZeroOrder", frame=0, model=zo)
    olg.initialize_decoder("ZeroOrder", frame=1, model=zo)

    try:
        olg.decode_all(dummy_run=(False, False), mask_current=(False, False),
                       force_safe=False, retry=RETRY)
    except Exception as e:
        return {"status": "DECODE-FAIL", "detail": f"{type(e).__name__}: {e}"}

    nt, _ = olg.string_quartet()
    t1, t2 = olg.translate_sequences()  # olg's translation of emitted NT
    g1, g2 = olg.get_prot_seq()         # decoder-held tokens

    # External Biopython translation of the same NT, per frame geometry
    f1_off, f2_off, f2_rev = Constants.ARRANGEMENT_CONFIG[int(arrangement)]
    pos1 = [p.item() for p in olg.coords.f1_to_all]
    pos2 = [p.item() for p in olg.coords.f2_to_all]
    b1 = biopython_translate(nt, pos1, f1_off, False)
    b2 = biopython_translate(nt, pos2, f2_off, f2_rev)

    return {"status": "OK", "nt": nt, "t1": t1, "t2": t2, "g1": g1, "g2": g2, "b1": b1, "b2": b2}


def main() -> int:
    print(f"Round-trip sanity check  (L={LENGTH}, retry={RETRY})")
    print("Each frame must agree across: decoder tokens == olg translate == Biopython translate")
    print(f"{'arrangement':12} {'offset':>6} {'frame1':>8} {'frame2':>8}  result")
    print("-" * 56)

    n_pass = n_mismatch = n_skip = 0
    failures = []

    for arrangement in ARRANGEMENTS:
        for offset in OFFSETS:
            res = run_one(arrangement, offset, seed=offset)
            if res["status"] != "OK":
                n_skip += 1
                print(f"{arrangement.name:12} {offset:>6} {'':>8} {'':>8}  {res['status']} ({res['detail']})")
                continue

            # three-way agreement per frame: decoder (g) == olg translate (t) == biopython (b)
            f1_ok = res["g1"] == res["t1"] == res["b1"]
            f2_ok = res["g2"] == res["t2"] == res["b2"]
            tag = "PASS" if (f1_ok and f2_ok) else "MISMATCH"
            if f1_ok and f2_ok:
                n_pass += 1
            else:
                n_mismatch += 1
                failures.append((arrangement.name, offset, res))
            print(f"{arrangement.name:12} {offset:>6} {('OK' if f1_ok else 'BAD'):>8} "
                  f"{('OK' if f2_ok else 'BAD'):>8}  {tag}")

    print("-" * 56)
    print(f"PASS={n_pass}  MISMATCH={n_mismatch}  SKIP(build/decode-fail)={n_skip}")

    for name, offset, res in failures:
        print(f"\n--- MISMATCH detail: {name} offset={offset} ---")
        for fr, g, t, b in [("frame1", res["g1"], res["t1"], res["b1"]),
                            ("frame2", res["g2"], res["t2"], res["b2"])]:
            if not (g == t == b):
                print(f"  {fr} decoder  : {g}")
                print(f"  {fr} olg-trans: {t}")
                print(f"  {fr} biopython: {b}")

    return 1 if n_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
