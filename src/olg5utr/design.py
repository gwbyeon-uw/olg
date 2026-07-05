"""Dual-5'UTR design geometry + the discrete (synonymous) move space.

Layout of the designed window (length L = free_len + 3*len(outer_protein)):

    [ free 5'UTR : free_len nt ][ outer-gene CDS : len(outer_protein) codons ]  | inner ATG (downstream)
      positions 0..free_len-1      positions free_len..L-1                        position L
      outer 5'UTR = seq[0:free_len]           inner 5'UTR = seq[0:L]

Two ORFs, dual MRL; the free region drives BOTH 5'UTRs, the outer CDS is single-synonymous, and there
is NO dual-coded region (the inner CDS is downstream of L, outside either 5'UTR). Same strand only.

Reuses olg's genetic-code table (`olg.constants`) for synonymous codons; the DNA one-hot uses ACGT
order (fixed by Optimus's training — NOT olg's ATGC nucleotide order).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from olg.constants import Constants, build_restricted_codon_table  # noqa: F401  (genetic code, native)

from .model import MODEL_INPUT_LEN

NUCLEOTIDES = "ACGT"                       # Optimus one-hot order (do NOT use olg's ATGC here)
_BASE = {b: i for i, b in enumerate(NUCLEOTIDES)}


def reverse_codon_table(codon_table: dict[str, str]) -> dict[str, list[str]]:
    """AA -> list of synonymous codons, from a codon->AA table."""
    rev: dict[str, list[str]] = {}
    for codon, aa in codon_table.items():
        rev.setdefault(aa, []).append(codon)
    return rev


def translate(dna: str, codon_table: dict[str, str], offset: int = 0) -> str:
    """Translate ``dna`` in one reading frame starting at ``offset``."""
    return "".join(codon_table.get(dna[i:i + 3], "?") for i in range(offset, len(dna) - 2, 3))


@dataclass
class UTRDesign:
    """Spec for one dual-5'UTR design problem.

    Args:
        outer_protein: outer gene amino-acid sequence (including the start ``M``); its CDS is
            ``3*len(outer_protein)`` nt and is held fixed up to synonymous codon choice.
        free_len: length of the free 5' region (the outer gene's 5'UTR), fully unconstrained.
        w_mrl: dual-MRL weight — objective is ``w_mrl*MRL_outer + (1-w_mrl)*MRL_inner``.
        codon_table: genetic code (codon->AA); defaults to olg's standard table. Use
            ``build_restricted_codon_table`` for a custom code.
        head: Optimus output head to score with (dataset the model was trained on).
    """

    outer_protein: str
    free_len: int
    w_mrl: float = 0.5
    codon_table: dict[str, str] = field(default_factory=lambda: dict(Constants.STANDARD_CODONS))
    head: int = 0

    def __post_init__(self) -> None:
        if self.length > MODEL_INPUT_LEN:
            raise ValueError(
                f"inner 5'UTR is {self.length} nt but Optimus reads only {MODEL_INPUT_LEN} nt — "
                "shorten outer_protein or free_len (don't design a >~100-nt 5'UTR with this model)."
            )
        self._rev = reverse_codon_table(self.codon_table)
        missing = [a for a in self.outer_protein if a not in self._rev]
        if missing:
            raise ValueError(f"outer_protein has residues not in the codon table: {sorted(set(missing))}")

    @property
    def outer_cds_len(self) -> int:
        return 3 * len(self.outer_protein)

    @property
    def length(self) -> int:
        """L — the designed window = the inner gene's 5'UTR length."""
        return self.free_len + self.outer_cds_len

    def synonymous_codons(self) -> list[list[str]]:
        """Per outer-CDS codon position, the codons encoding that residue."""
        return [self._rev[a] for a in self.outer_protein]

    def outer_translation(self, dna: str) -> str:
        """Read the outer protein out of a window (frame starts at free_len)."""
        return translate(dna, self.codon_table, offset=self.free_len)


def dna_to_onehot(seqs: list[str], length: int, device: torch.device | None = None) -> torch.Tensor:
    """List of DNA strings -> ``(N, 4, length)`` one-hot (ACGT order).

    Builds the index array on CPU and scatters once, then a single host->device transfer — avoids
    N*length scalar writes into a device tensor. All sequences must be at least ``length`` long.
    """
    idx = torch.tensor([[_BASE[ch] for ch in s[:length]] for s in seqs], dtype=torch.long)  # (N, L) CPU
    oh = torch.zeros(len(seqs), 4, length)
    oh.scatter_(1, idx.unsqueeze(1), 1.0)
    return oh.to(device) if device is not None else oh
