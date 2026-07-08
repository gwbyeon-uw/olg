#!/usr/bin/env python
"""OSTIR parity validator — the ground truth any RBS-scoring reimplementation must reproduce.

Snapshots the full RBSScore (expression + every ΔG component term) that the current
``ostir.run_ostir`` path produces, over a representative set of fold windows: the real windows
olgrbs scores during ``optimize_rbs`` (harvested), plus synthetic edge cases (GTG/TTG starts,
tight/loose spacing, no-SD, leaderless). ``tests/golden_ostir.json`` is the committed snapshot.

  python tests/ostir_parity.py                 # self-check: score_rbs vs the committed golden
  python tests/ostir_parity.py --save          # regenerate the golden (only when OSTIR itself changes)
  python tests/ostir_parity.py --check mymodule.my_score_rbs   # diff a candidate scorer vs the golden

A candidate has the score_rbs signature: fn(nt:str, inner_start:int, asd:str) -> RBSScore | None.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for olgrbs_toy

GOLDEN = Path(__file__).resolve().parent / "golden_ostir.json"
FIELDS = ["expression", "dG_total", "dG_mRNA_rRNA", "dG_mRNA", "dG_spacing",
          "dG_standby", "dG_start_codon", "start_codon", "spacing_bp"]
ABS_TOL = {"expression": 1e-6, "dG_total": 1e-4, "dG_mRNA_rRNA": 1e-4, "dG_mRNA": 1e-4,
           "dG_spacing": 1e-4, "dG_standby": 1e-4, "dG_start_codon": 1e-4}
REL_TOL_EXPR = 1e-4  # expression spans orders of magnitude -> also allow a relative match


def harvest_windows(max_windows: int = 400) -> list[tuple[str, int, str]]:
    """Real (nt, inner_start, asd) windows olgrbs actually scores, across diverse toy designs."""
    import olgrbs.driver as drv
    from olgrbs import optimize_rbs
    from olg.constants import Arrangement
    from olgrbs_toy import build_toy_design

    seen: dict[tuple[str, int, str], None] = {}
    orig = drv.score_rbs

    def rec(nt, start, asd):
        seen.setdefault((nt, start, asd), None)
        return orig(nt, start, asd)

    drv.score_rbs = rec
    try:
        for arr in (Arrangement.PLUS_ONE, Arrangement.PLUS_TWO):
            for seed in range(6):
                optimize_rbs(build_toy_design(arr, 18, l1=60, l2=20, seed=seed),
                             sa_steps=800, sa_restarts=2)
                if len(seen) >= max_windows:
                    break
            if len(seen) >= max_windows:
                break
    finally:
        drv.score_rbs = orig
    return list(seen)[:max_windows]


def synthetic_windows() -> list[tuple[str, int, str]]:
    """Edge cases the harvest under-samples: alt start codons over varied spacing, no-SD, leaderless."""
    from olgrbs.scorer import ECOLI_ANTI_SD as SD
    tail = "GCTACGGTACGTACGATCGTAAAT"
    cases = []
    for codon in ("ATG", "GTG", "TTG"):
        for pad in range(0, 12, 2):
            cases.append(("AAGGAGGA" + "T" * pad + codon + tail, 8 + pad, SD))
    cases += [("ATG" + tail, 0, SD), ("GTG" + tail, 0, SD),
              ("AA" + "ATG" + tail, 2, SD), ("AAAA" + "ATG" + tail, 4, SD)]
    cases += [("GGGGGGGGGGGGATGCCCCCCCCCCCC", 12, SD),
              ("AAAAAAAAAAAAATGTTTTTTTTTTTTT", 12, SD),
              ("CGCGCGCGCGCGATGCGCGCGCGCGCG", 12, SD)]
    cases += [("A" * 40 + "AGGAGG" + "A" * 5 + "ATG" + "GCT" * 8, 54, SD),
              ("AGGAGG" + "A" * 30 + "ATG" + "GCT" * 8, 39, SD)]
    return [(nt.upper(), s, a) for nt, s, a in cases]


def _golden_score(nt, start, asd):
    from olgrbs.scorer import score_rbs
    r = score_rbs(nt, start, asd)
    return None if r is None else asdict(r)


def build_golden():
    windows = harvest_windows() + synthetic_windows()
    return [{"nt": nt, "start": s, "asd": a, "score": _golden_score(nt, s, a)} for (nt, s, a) in windows]


def _cmp(gold, got) -> list[str]:
    if (gold is None) != (got is None):
        return [f"None-mismatch: golden={'None' if gold is None else 'score'} got={'None' if got is None else 'score'}"]
    if gold is None:
        return []
    diffs = []
    for f in FIELDS:
        gv, cv = gold[f], got[f]
        if f == "start_codon":
            if gv != cv:
                diffs.append(f"{f}: {gv!r} != {cv!r}")
        elif f == "spacing_bp":
            if int(gv) != int(cv):
                diffs.append(f"{f}: {gv} != {cv}")
        elif not math.isfinite(cv):  # NaN/inf never silently passes a numeric tolerance
            diffs.append(f"{f}: golden {gv:.6g} vs non-finite {cv!r}")
        elif f == "expression":
            ad = abs(gv - cv); rd = ad / max(abs(gv), 1e-12)
            if ad > ABS_TOL[f] and rd > REL_TOL_EXPR:
                diffs.append(f"{f}: {gv:.6g} vs {cv:.6g} (abs {ad:.2e}, rel {rd:.2e})")
        else:
            ad = abs(gv - cv)
            if ad > ABS_TOL[f]:
                diffs.append(f"{f}: {gv:.6g} vs {cv:.6g} (abs {ad:.2e})")
    return diffs


def validate(candidate_fn, golden) -> bool:
    n_fail = 0
    per_field: dict[str, int] = {}
    for i, row in enumerate(golden):
        got = candidate_fn(row["nt"], row["start"], row["asd"])
        got = None if got is None else asdict(got)
        d = _cmp(row["score"], got)
        if d:
            n_fail += 1
            for item in d:
                per_field[item.split(":")[0]] = per_field.get(item.split(":")[0], 0) + 1
            if n_fail <= 8:
                print(f"  MISMATCH [{i}] start={row['start']} len={len(row['nt'])}: {d[:4]}")
    if per_field:
        print("  per-field mismatch counts:", dict(sorted(per_field.items(), key=lambda kv: -kv[1])))
    ok = n_fail == 0
    print(f"{'PARITY PASS' if ok else 'PARITY FAIL'}: {len(golden) - n_fail}/{len(golden)} rows match")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="regenerate tests/golden_ostir.json from OSTIR")
    ap.add_argument("--check", metavar="MODULE.FN", help="validate a candidate scorer against the golden")
    args = ap.parse_args()

    if args.save:
        golden = build_golden()
        n_ok = sum(1 for g in golden if g["score"] is not None)
        GOLDEN.write_text(json.dumps(golden))
        print(f"saved {len(golden)} windows ({n_ok} scored, {len(golden) - n_ok} None) -> {GOLDEN}")
        return 0

    golden = json.loads(GOLDEN.read_text())
    if args.check:
        mod, fn = args.check.rsplit(".", 1)
        candidate = getattr(importlib.import_module(mod), fn)
    else:
        from olgrbs.scorer import score_rbs as candidate  # self-check
    return 0 if validate(candidate, golden) else 1


if __name__ == "__main__":
    sys.exit(main())
