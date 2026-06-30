"""In-memory results object for the design pipeline.

Stages pass a `Results` (a keyed set of `Placement`s) and enrich it, instead of wiring through
intermediate CSVs. Each stage fills its own fields (None until it runs). Persistence
(`save`/`load`) is serialization for traceability/resume -- not the inter-module channel.

Keyed by (arrangement, offset, length). RBS ceiling is a per-(arr,offset) property (the flank is
length-independent), so `set_by_offset` broadcasts it across the lengths present.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import pandas as pd

_KEYS = ("arrangement", "offset", "length")
_INT_FIELDS = {"arrangement", "offset", "length", "seed", "n_compat"}
_BOOL_FIELDS = {"feasible"}


def _coerce(name: str, v):
    """Coerce a CSV-loaded value to the field's Python type (pandas dtype inference is unreliable:
    bools can arrive as strings, int columns become float when any cell was NaN). Treats blank
    strings as None and rejects non-finite numerics for int fields; callers add row context."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if name in _BOOL_FIELDS:
        return v.strip().lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
    if name in _INT_FIELDS:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"non-finite value {v!r} for integer field {name!r}")
        return int(round(f))
    return float(v)


@dataclass
class Placement:
    """One placement + the outputs stages accumulate on it (None until the stage runs)."""
    arrangement: int
    offset: int
    length: int
    # feasibility stage
    feasible: bool | None = None
    seed: int | None = None
    # rbs-track stage (per-(arr,offset), broadcast over lengths): achievable RBS range over
    # synonymous flank realizations. ceiling = best; pctile = ceiling's E. coli percentile.
    rbs_min: float | None = None
    rbs_median: float | None = None
    rbs_ceiling: float | None = None
    rbs_pctile: float | None = None
    # metric stages
    mic_mean: float | None = None
    mic_median: float | None = None
    mic_min: float | None = None
    entropy_bits_per_pos: float | None = None
    entropy_total_bits: float | None = None
    n_compat: int | None = None

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.arrangement, self.offset, self.length)


class Results:
    """A keyed collection of Placements that stages enrich in place."""

    def __init__(self):
        self._p: dict[tuple[int, int, int], Placement] = {}

    def upsert(self, arrangement: int, offset: int, length: int, **vals) -> Placement:
        """Create or update the placement, setting any provided stage fields."""
        key = (arrangement, offset, length)
        p = self._p.get(key) or Placement(arrangement, offset, length)
        for k, v in vals.items():
            setattr(p, k, v)
        self._p[key] = p
        return p

    def get(self, arrangement: int, offset: int, length: int) -> Placement | None:
        return self._p.get((arrangement, offset, length))

    def set_by_offset(self, arrangement: int, offset: int, **vals) -> int:
        """Broadcast fields to every placement sharing (arrangement, offset); return the number
        matched. Callers should check 0 (a silent no-match = grid misalignment)."""
        n = 0
        for p in self._p.values():
            if p.arrangement == arrangement and p.offset == offset:
                for k, v in vals.items():
                    setattr(p, k, v)
                n += 1
        return n

    def offsets(self, feasible_only: bool = True) -> list[tuple[int, int]]:
        """Distinct (arrangement, offset) present (optionally only feasible) -- for per-offset stages."""
        return sorted({(p.arrangement, p.offset) for p in self
                       if (p.feasible or not feasible_only)})

    def filter(self, pred) -> list[Placement]:
        return [p for p in self if pred(p)]

    def __iter__(self):
        return iter(self._p.values())

    def __len__(self):
        return len(self._p)

    # ---- serialization (traceability / resume), not the inter-module channel ----
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(p) for p in self])

    def save(self, path) -> None:
        self.to_frame().to_csv(path, index=False)

    @classmethod
    def load(cls, path) -> "Results":
        r = cls()
        names = {f.name for f in fields(Placement)}
        df = pd.read_csv(path)
        missing = set(_KEYS) - set(df.columns)
        if missing:
            raise ValueError(f"Results.load: {path} missing required key column(s): {sorted(missing)}")
        for i, row in enumerate(df.itertuples(index=False)):
            try:
                vals = {k: _coerce(k, v) for k, v in row._asdict().items() if k in names and pd.notna(v)}
            except (ValueError, TypeError, OverflowError) as e:
                raise ValueError(f"Results.load: {path} row {i} bad value: {e}") from e
            vals = {k: v for k, v in vals.items() if v is not None}  # blank-string coercions -> drop
            if not set(_KEYS) <= vals.keys():
                raise ValueError(f"Results.load: {path} row {i} missing key value(s): {row._asdict()}")
            r.upsert(vals.pop("arrangement"), vals.pop("offset"), vals.pop("length"), **vals)
        return r
