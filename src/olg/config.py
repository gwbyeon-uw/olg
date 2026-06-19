from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from olg.constants import *
from olg.exceptions import ConfigValidationError

_BASE_YAML = Path(__file__).parent / "base.yaml"

# Fields on ProteinConfig that hold tensors and can be specified in YAML
# as either a list of numbers or a path to a .pt/.npy file.
_TENSOR_FIELDS = {"logit_weight", "logit_bias", "aa_bias", "max_aa_count"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _resolve_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert a YAML tensor value (list or file path) to a torch.Tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return torch.tensor(value, device=device)
    if isinstance(value, str):
        p = Path(value)
        if p.suffix == ".pt":
            return torch.load(p, map_location=device, weights_only=True)
        if p.suffix in (".npy", ".npz"):
            return torch.from_numpy(np.load(p)).to(device)
        raise ConfigValidationError(f"Unsupported tensor file format: {p.suffix}")
    raise TypeError(f"Cannot convert {type(value).__name__} to tensor")


@dataclass
class ProteinConfig:
    """Configuration for a single protein in the OLG design"""
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    length: int = 100  # Length of the protein, only the overlap encoded region if we are not using the whole protein
    start_offset: int = 0  # For example, this would be 10 if the model input is a sequence of length 100, but we are only overlap-encoding from position 10
    force_stop: bool = False
    force_start: bool = False
    start_codons: List[str] = field(default_factory=lambda: ["ATG"])  # Can be multiple
    fixed_positions: Optional[List[Tuple[int, str]]] = None  # 1-based
    gap_positions: Optional[List[int]] = None  # For models that use alignments
    alphabet_size: int = Constants.DEFAULT_ALPHABET_SIZE  # Set by DesignConfig from its alphabet field

    # Constraints and biases
    repetition_penalty: float = 1.1  # Penalty for repeating amino acids
    repetition_penalty_window: int = 4  # Window size for repetition penalty
    logit_weight: Optional[torch.Tensor] = None  # Weight vectors for each protein's logits
    logit_bias: Optional[torch.Tensor] = None  # Position-specific amino acid biases
    aa_bias: Optional[torch.Tensor] = None  # Position-invariant amino acid biases
    truncate_topp: float = 0.0  # Top-p cutoff for individual protein logits
    max_aa_count: Optional[torch.Tensor] = None  # Max count per amino acid type
    max_pos_count: float = Constants.MAX_LOGIT  # Max total count of positively charged residues

    def __post_init__(self):
        self._validate()
        if self.logit_weight is None:
            self.logit_weight = torch.ones(self.length, device=self.device)
        if self.logit_bias is None:
            self.logit_bias = torch.zeros((self.length, self.alphabet_size), device=self.device)
        if self.aa_bias is None:
            self.aa_bias = torch.zeros(self.alphabet_size, device=self.device)
        if self.max_aa_count is None:
            self.max_aa_count = torch.zeros(self.alphabet_size, device=self.device) + Constants.MAX_LOGIT

    def _validate(self):
        if self.length <= 0:
            raise ConfigValidationError(f"length must be positive, got {self.length}")
        if self.repetition_penalty < 1.0:
            raise ConfigValidationError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
        if self.repetition_penalty_window < 0:
            raise ConfigValidationError(f"repetition_penalty_window must be >= 0, got {self.repetition_penalty_window}")
        if not (0.0 <= self.truncate_topp <= 1.0):
            raise ConfigValidationError(f"truncate_topp must be in [0, 1], got {self.truncate_topp}")

        # start_offset: consumed by coordinate maps (all_to_f1 = arange(...) + start_offset)
        # and get_prot_seq slicing; must index validly into a length-`length` protein.
        if not (0 <= self.start_offset < self.length):
            raise ConfigValidationError(
                f"start_offset must be in [0, length={self.length}), got {self.start_offset}")

        # start codons must be DNA triplets (looked up in CODONS_TO_QUARTETS by exact match)
        for codon in self.start_codons:
            if not (isinstance(codon, str) and len(codon) == 3 and all(b in "ACGT" for b in codon)):
                raise ConfigValidationError(
                    f"start_codons must be length-3 strings over ACGT, got {codon!r}")

        # fixed_positions are 1-based in the gap-reduced residue space and must be unique.
        # coordinates.py indexes f1_to_all[pos-1], and f1_to_all has (length - len(gap_positions))
        # entries, so the bound is the post-gap residue count, not raw length.
        if self.fixed_positions is not None:
            n_gaps = len(self.gap_positions) if self.gap_positions else 0
            effective_len = self.length - n_gaps
            seen = set()
            for fp in self.fixed_positions:
                pos = fp[0]
                if not (1 <= pos <= effective_len):
                    raise ConfigValidationError(
                        f"fixed_positions position {pos} out of range [1, {effective_len}] "
                        f"(length {self.length} minus {n_gaps} gap positions)")
                if pos in seen:
                    raise ConfigValidationError(f"duplicate fixed_positions position {pos}")
                seen.add(pos)

        # gap_positions: 1-based, in [1, length], unique, and cannot remove every residue
        # (coordinates.py computes f1_gap_len = length - len(gap_positions))
        if self.gap_positions is not None:
            if len(set(self.gap_positions)) != len(self.gap_positions):
                raise ConfigValidationError("gap_positions contains duplicate entries")
            for g in self.gap_positions:
                if not (1 <= g <= self.length):
                    raise ConfigValidationError(
                        f"gap_positions entry {g} out of range [1, {self.length}]")
            if len(self.gap_positions) >= self.length:
                raise ConfigValidationError(
                    f"gap_positions removes all residues ({len(self.gap_positions)} >= length {self.length})")

        # User-supplied tensor shapes must match (length, alphabet_size) couplings.
        # Defaults are filled in after _validate, so only non-None (user) tensors are checked.
        # _validate runs at this config's own alphabet_size; DesignConfig._resize_for_alphabet
        # legitimately extends tensors later only when pc.alphabet_size < design.alphabet_size.
        expected = {
            "logit_weight": (self.length,),
            "logit_bias": (self.length, self.alphabet_size),
            "aa_bias": (self.alphabet_size,),
            "max_aa_count": (self.alphabet_size,),
        }
        for name, shape in expected.items():
            val = getattr(self, name)
            if val is not None and tuple(val.shape) != shape:
                raise ConfigValidationError(
                    f"{name} must have shape {shape}, got {tuple(val.shape)}")

    @classmethod
    def _from_dict(cls, d: dict, device: torch.device, alphabet_size: int = Constants.DEFAULT_ALPHABET_SIZE) -> ProteinConfig:
        """Build a ProteinConfig from a plain dict (e.g. parsed from YAML)."""
        d = dict(d)  # shallow copy
        d["device"] = device
        # alphabet_size is injected from DesignConfig; don't let the YAML value override it
        d["alphabet_size"] = alphabet_size
        for key in _TENSOR_FIELDS:
            if key in d and d[key] is not None:
                d[key] = _resolve_tensor(d[key], device)
        if "fixed_positions" in d and d["fixed_positions"] is not None:
            d["fixed_positions"] = [tuple(fp) for fp in d["fixed_positions"]]
        return cls(**d)

    def to_dict(self, include_defaults: bool = False) -> dict:
        """Serialize to a plain dict suitable for YAML output.

        Tensor fields at their default (all-same) values are omitted
        unless ``include_defaults`` is True, keeping YAML output clean.
        Non-default tensor values are serialized as lists; file-path
        references are not preserved (the resolved values are written).
        """
        d = {}
        for f in fields(self):
            if f.name in ("device", "alphabet_size"):
                continue  # device lives on DesignConfig; alphabet_size is derived from DesignConfig.alphabet
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                if not include_defaults and self._is_default_tensor(f.name, val):
                    continue
                d[f.name] = val.cpu().tolist()
            elif f.name == "fixed_positions" and val is not None:
                # store as plain lists, not tuples, so yaml.safe_load can round-trip them
                # (tuples serialize as !!python/tuple tags that safe_load rejects)
                d[f.name] = [list(fp) for fp in val]
            else:
                d[f.name] = val
        return d

    def _is_default_tensor(self, name: str, val: torch.Tensor) -> bool:
        """Check if a tensor field is at its __post_init__ default."""
        if name == "logit_weight":
            return val.shape == (self.length,) and torch.all(val == 1.0)
        if name == "logit_bias":
            return val.shape == (self.length, self.alphabet_size) and torch.all(val == 0.0)
        if name == "aa_bias":
            return val.shape == (self.alphabet_size,) and torch.all(val == 0.0)
        if name == "max_aa_count":
            return val.shape == (self.alphabet_size,) and torch.all(val == Constants.MAX_LOGIT)
        return False

    def _resize_for_alphabet(self, new_size: int) -> None:
        """Extend alphabet-dependent tensors for a larger alphabet.

        Called by DesignConfig when a ProteinConfig was built with the default
        (smaller) alphabet_size but the parent DesignConfig uses a larger one.
        New letter positions receive neutral defaults: zero bias, MAX_LOGIT count.
        """
        if new_size <= self.alphabet_size:
            return
        device = self.logit_weight.device
        extra = new_size - self.alphabet_size
        self.logit_bias = torch.cat(
            [self.logit_bias, torch.zeros((self.length, extra), device=device)], dim=1
        )
        self.aa_bias = torch.cat(
            [self.aa_bias, torch.zeros(extra, device=device)]
        )
        self.max_aa_count = torch.cat(
            [self.max_aa_count, torch.full((extra,), Constants.MAX_LOGIT, device=device)]
        )
        self.alphabet_size = new_size


@dataclass
class DesignConfig:
    """Main configuration for OLG design"""
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    arrangement: Arrangement = Arrangement.PLUS_ONE
    offset: int = 0  # This is the distance between the N-terminii of the two proteins.
    protein1: ProteinConfig = field(default_factory=lambda: ProteinConfig())
    protein2: ProteinConfig = field(default_factory=lambda: ProteinConfig())
    codon_table: Union[str, Dict[str, str]] = "Standard"  # NCBI table name or a dictionary of codon-AA
    decoding_mode: Optional[DecodingMode] = DecodingMode.OVERLAP_FIRST
    temperature: float = 1.0  # logit/T, applied to the model output
    top_p: float = 0.0  # 0.0 for greedy
    complexed: bool = False  # Whether to use ProteinMPNN tied decoding
    shared: bool = False  # Whether to use EvoDiff split MSA decoding
    balancer_max_weight: float = 2.0  # Maximum weight for balancing the two frames
    balancer_unit: float = 0.5  # Increment unit for balancing the two frames
    balancer_threshold: float = 0.15  # Threshold for difference in scores to trigger balancing
    rand_base: Optional[int] = None  # Random seed for reproducibility
    tqdm_disable: bool = False  # Whether to disable progress bars
    alphabet: List[str] = field(default_factory=lambda: list(Constants.DEFAULT_ALPHABET))
    # Any ordering is valid — indices are derived from letter identity, not position.

    def __post_init__(self):
        # Derived attributes — computed from alphabet by content, never by position
        self.alphabet_size: int = len(self.alphabet)
        self.alphabet_index: Dict[str, int] = {a: i for i, a in enumerate(self.alphabet)}
        if 'X' not in self.alphabet_index:
            raise ConfigValidationError("alphabet must contain 'X' (stop codon marker)")
        self.stop_index: int = self.alphabet_index['X']
        # Validate no duplicates
        if self.alphabet_size != len(set(self.alphabet)):
            dupes = sorted({a for a in self.alphabet if self.alphabet.count(a) > 1})
            raise ConfigValidationError(f"alphabet contains duplicate letters: {dupes}")
        # Validate codon_table values are all in alphabet
        if isinstance(self.codon_table, dict):
            unknown = set(self.codon_table.values()) - set(self.alphabet)
            if unknown:
                raise ConfigValidationError(
                    f"codon_table references amino acids not in alphabet: {sorted(unknown)}. "
                    f"Extend DesignConfig.alphabet to include them."
                )
        # If ProteinConfigs were built with a smaller alphabet_size (e.g. default 21
        # when this DesignConfig uses an extended alphabet), extend their tensors now.
        for pc in (self.protein1, self.protein2):
            if pc.alphabet_size < self.alphabet_size:
                pc._resize_for_alphabet(self.alphabet_size)
            elif pc.alphabet_size > self.alphabet_size:
                raise ConfigValidationError(
                    f"ProteinConfig.alphabet_size ({pc.alphabet_size}) exceeds "
                    f"DesignConfig.alphabet_size ({self.alphabet_size}). "
                    f"Ensure ProteinConfigs are built with the same alphabet as DesignConfig."
                )

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> DesignConfig:
        """Load a DesignConfig from base.yaml defaults, optionally overridden by a user YAML.

        Always loads ``base.yaml`` (shipped with the package) first,
        then deep-merges user overrides from *path* on top. If *path*
        is None, returns base defaults only.

        User YAML files can be minimal — only specify what differs::

            # my_design.yaml
            temperature: 0.1
            protein1:
              length: 120
              repetition_penalty: 1.2

        Tensor fields on ProteinConfig can be specified as:
          - A list of numbers (converted to torch.Tensor)
          - A path to a .pt or .npy file
          - Omitted entirely (defaults from base.yaml / __post_init__)
        """
        with open(_BASE_YAML) as f:
            d = yaml.safe_load(f)
        if path is not None:
            path = Path(path)
            with open(path) as f:
                user = yaml.safe_load(f) or {}
            d = _deep_merge(d, user)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> DesignConfig:
        """Build a DesignConfig from a plain dict."""
        d = dict(d)
        device = torch.device(d.pop("device", "cuda:0"))

        # Parse alphabet first — needed to determine alphabet_size for ProteinConfigs
        alphabet = list(d.get("alphabet", list(Constants.DEFAULT_ALPHABET)))
        alphabet_size = len(alphabet)

        # Parse enum fields
        if "arrangement" in d:
            d["arrangement"] = Arrangement(d["arrangement"])
        if "decoding_mode" in d and d["decoding_mode"] is not None:
            d["decoding_mode"] = DecodingMode(d["decoding_mode"])

        # Parse nested ProteinConfigs, injecting the derived alphabet_size
        p1 = d.pop("protein1", {}) or {}
        p2 = d.pop("protein2", {}) or {}
        protein1 = ProteinConfig._from_dict(p1, device, alphabet_size=alphabet_size) if p1 else ProteinConfig(device=device, alphabet_size=alphabet_size)
        protein2 = ProteinConfig._from_dict(p2, device, alphabet_size=alphabet_size) if p2 else ProteinConfig(device=device, alphabet_size=alphabet_size)

        return cls(device=device, protein1=protein1, protein2=protein2, **d)

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # safe_dump (vs dump) refuses to emit python-specific tags, so any non-plain
            # value surfaces as an error at write time instead of breaking safe_load on read
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Serialize to a plain dict. Tensors become lists, enums become ints."""
        d = {
            "device": str(self.device),
            "arrangement": int(self.arrangement),
            "offset": self.offset,
            "protein1": self.protein1.to_dict(),
            "protein2": self.protein2.to_dict(),
            "codon_table": self.codon_table,
            "decoding_mode": int(self.decoding_mode) if self.decoding_mode is not None else None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "complexed": self.complexed,
            "shared": self.shared,
            "balancer_max_weight": self.balancer_max_weight,
            "balancer_unit": self.balancer_unit,
            "balancer_threshold": self.balancer_threshold,
            "rand_base": self.rand_base,
            "tqdm_disable": self.tqdm_disable,
            "alphabet": list(self.alphabet),
        }
        return d
