"""Sample feasible OLGDesigns at a placement (orchestrator).

The single "I want a usable design" operation: composes the model-free feasibility primitive
(feasibility.sample_codesign_paths) with realization (path -> OLGDesign), so consumers never
touch paths or reconstruction. Consumers that only need feasibility / AA samples call
orchestrator.feasibility directly (lighter -- no torch, no OLGDesign).

Realization decodes a path -> NT, translates BOTH frames, and fixes both proteins (gene with
overlap co-design muts + inner peptide) -> a fully-specified, consistent decode (reliable, unlike
fixing gene=WT against an AMP that needs muts). The result is a LOCAL design (just the window),
which is all olgrbs needs (flank + OSTIR's ±35 fold window are local).
"""
from __future__ import annotations

import torch

import logging

from olg import OLGDesign
from olg.config import DesignConfig, ProteinConfig
from olg.constants import Arrangement, Constants
from olg.exceptions import OLGError
from olgrbs.search import decode_path as _nt_from_path  # shared path->NT decoder (single source)

from .feasibility import sample_codesign_paths

logger = logging.getLogger(__name__)


def _realize(arrangement: int, offset: int, length: int, path: list[int], w0: int,
             codon_table: dict, alphabet: list[str], device=None) -> OLGDesign:
    """Build the local OLGDesign realized by `path` (window starts at gene pos `w0`)."""
    device = device or torch.device("cpu")
    nt = _nt_from_path(path)
    f1_off, f2_off, f2_rev = Constants.ARRANGEMENT_CONFIG[arrangement]

    def codon(p: int, nt_off: int, rev: bool) -> str:
        c = nt[3 * p + nt_off: 3 * p + nt_off + 3]
        return Constants._reverse_complement(c) if rev else c

    npos = len(path)                 # gene (frame 1) spans the whole window
    amp0 = offset - w0               # inner Met index within the window
    gene_aas = [codon_table.get(codon(i, f1_off, False), "X") for i in range(npos)]
    amp_aas = [codon_table.get(codon(amp0 + j, f2_off, f2_rev), "X") for j in range(length)]

    a_idx = {a: i for i, a in enumerate(alphabet)}
    logits = torch.zeros((1, len(alphabet)), device=device)
    logits[0, a_idx["X"]] = Constants.MIN_LOGIT
    cfg = DesignConfig(
        device=device, arrangement=Arrangement(arrangement), offset=amp0, codon_table=codon_table,
        alphabet=alphabet, rand_base=0, tqdm_disable=True,
        protein1=ProteinConfig(device=device, length=npos, alphabet_size=len(alphabet),
                               fixed_positions=[(i + 1, gene_aas[i]) for i in range(npos)]),
        protein2=ProteinConfig(device=device, length=length, alphabet_size=len(alphabet),
                               force_start=True, force_stop=True,
                               fixed_positions=[(j + 1, amp_aas[j]) for j in range(length)]),
    )
    olg = OLGDesign(cfg)
    olg.initialize_decoder("ZeroOrder", frame=0, model=logits)
    olg.initialize_decoder("ZeroOrder", frame=1, model=logits)
    olg.decode_all(dummy_run=(False, False), mask_current=(False, False), force_safe=False, retry=40)
    return olg


def sample_designs(arrangement: int, offset: int, length: int, gene: str, lock0: set[int],
                   lk, free_sets: dict, gene_aas: str, amp_aas: str, stops, codon_table: dict,
                   alphabet: list[str], *, k: int = 1, pad: int = 13, seed: int | None = None,
                   device=None) -> list[OLGDesign]:
    """Up to k valid OLGDesigns at this placement; [] if infeasible (the workflow's own gate).

    `pad` >= the upstream flank you need (e.g. 13 for an RBS flank). `seed` defaults to `offset`.
    """
    # fail fast on a codon_table/alphabet contract violation (else _realize raises a bare KeyError
    # deep in decoder init -- which we deliberately do NOT swallow as "infeasible" below).
    missing = (set(codon_table.values()) | {"X"}) - set(alphabet)
    if missing:
        raise ValueError(f"sample_designs: codon_table amino acids not in alphabet: {sorted(missing)}")
    paths = sample_codesign_paths(lk, gene, offset, length, lock0, gene_aas, amp_aas, free_sets, k,
                                  allowed_stops=stops, seed=offset if seed is None else seed, pad=pad)
    out, failed = [], 0
    for path, w0 in paths:
        try:
            out.append(_realize(arrangement, offset, length, path, w0, codon_table, alphabet, device))
        except OLGError:  # decode miss despite a valid path; narrow so real bugs still propagate
            failed += 1
    if failed:  # not silent: a systematic decode failure would otherwise masquerade as infeasibility
        logger.warning("sample_designs(arr=%s, offset=%s, length=%s): %d/%d paths failed to decode",
                       arrangement, offset, length, failed, len(paths))
    return out
