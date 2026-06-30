"""Step-2 scan loop (orchestrator). The loop is identical - walk placements by rank,
probe-skip the ones that don't model-decode, resume from a checkpoint CSV, stop at top_k successes.
What DIFFERS is the per-placement work, which each session passes in:

    make_cells(arr, off, length) -> list of cell dicts (each has a 'mode' key + any params)
    run_cell(arr, off, length, cell) -> dict of result fields (raises on a decode failure)

`OLGCampaign.design()` supplies make_cells/run_cell (the per-session design step) while this holds the
robustness/bookkeeping plumbing.
"""
from __future__ import annotations

import time

import pandas as pd

from olg.constants import Arrangement


def run_scan(ranked, make_cells, run_cell, out, *, top_k, max_tried):
    """Returns nothing; writes `out` incrementally (resumable). See module docstring for callbacks."""
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["arrangement"], r["offset"], r["length"], r["mode"]) for r in rows if r.get("status") == "OK"}
    succeeded = {(r["arrangement"], r["offset"], r["length"]) for r in rows if r.get("status") == "OK"}

    tried = 0
    for arr, off, length in ranked:
        if len(succeeded) >= top_k or tried >= max_tried:
            break
        if (arr, off, length) in succeeded:
            continue
        tried += 1
        probed = False
        for cell in make_cells(arr, off, length):
            mode = cell["mode"]
            if (arr, off, length, mode) in done:
                continue
            first_attempt = not probed
            probed = True
            base = {"arrangement": arr, "arrangement_name": Arrangement(arr).name,
                    "offset": off, "length": length, **cell}
            t0 = time.time()
            try:
                row = {**base, "status": "OK", **run_cell(arr, off, length, cell)}
                succeeded.add((arr, off, length))
            except Exception as e:  # noqa: BLE001  (a decode failure is expected -> skip this placement)
                row = {**base, "status": f"error:{type(e).__name__}"}
                print(f"  off{off} L{length} {mode}: {type(e).__name__} [{time.time()-t0:.0f}s]")
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)          # checkpoint every cell
            if row["status"] == "OK":
                extra = (f" | plaus {row['gene_plausibility']:.2f} | {row['n_mut']} muts"
                         if row.get("gene_plausibility") == row.get("gene_plausibility")   # not NaN
                         and "n_mut" in row else "")
                print(f"  off{off} L{length} {mode}: MIC {row['mic_uM']:.1f}{extra} | "
                      f"RBS {row['rbs_rate']} (p{row['rbs_pctile']}) [{time.time()-t0:.0f}s]")
            elif first_attempt:
                print(f"  off{off} L{length}: probe failed to decode -- skipping placement")
                break

    print(f"\n-> {out.name}  ({len(succeeded)} placements succeeded, {tried} tried)")
