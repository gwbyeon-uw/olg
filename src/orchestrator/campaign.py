"""OLGCampaign — a config-driven front-end over the orchestrator primitives.

One YAML + two injected per-frame objectives (model-agnostic plug-ins) drive the whole workflow:

    camp = OLGCampaign.from_yaml("config.yaml", session="codesign")
    camp.set_objectives(gene=..., amp=...)        # FrameObjective plug-ins (see protocols below)
    screen  = camp.screen()                        # feasibility + RBS ceiling + potency proxy
    designs = camp.design(screen)                  # target -> rank -> probe-skip -> co-design -> RBS

The campaign owns *structure + search* (genetic code, lock, quartet lookups, feasibility, the design
cell, OLGDesign wiring, the E. coli percentile). It imports `olg` / `olgrbs` only — never a concrete
protein model; the scoring models live behind the injected objectives so this stays reusable for a
different gene / peptide / model.
"""
from __future__ import annotations

import bisect
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
import yaml

from olg import OLGDesign
from olg.config import DesignConfig, ProteinConfig
from olg.constants import Arrangement, Constants, build_restricted_codon_table

from .feasibility import build_free_gene_sets, build_quartet_lookup


# ── injected per-frame objective plug-ins (orchestrator-defined, caller-implemented) ──────────────
class FrameObjective(Protocol):
    """A pluggable per-frame design objective. Keeps orchestrator model-agnostic: the caller wires the
    concrete model behind this protocol."""

    metric_name: str  # result-row column for this frame's score (e.g. "mic_uM", "gene_plausibility")

    def attach(self, olg: OLGDesign, frame: int) -> None:
        """Initialize this frame's decoder on `olg` (wraps olg.initialize_decoder)."""

    def score(self, olg: OLGDesign, frame: int, free: list[int]) -> float:
        """Scalar objective for the just-decoded frame (`free` = co-designed gene positions)."""


class SequenceScorer(Protocol):
    """Batch-scores raw AA sequences (the screen's cheap potency proxy)."""

    def score_sequences(self, seqs: list[str]) -> list[float]:
        ...


class AmpObjective(FrameObjective, SequenceScorer, Protocol):
    """The AMP-frame objective: a FrameObjective that ALSO batch-scores sequences (the screen proxy).
    `screen()` needs only this `score_sequences`; `design()` needs the FrameObjective half too."""


# ── config ────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class CampaignConfig:
    """Parsed campaign YAML. Holds the shared blocks + one session block (`fixed`/`codesign`); input
    paths are resolved relative to the YAML file. The committed demo `config.yaml` is used as-is."""

    raw: dict
    session: str
    base: Path

    @classmethod
    def from_yaml(cls, path: str | Path, session: str) -> CampaignConfig:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text())
        base = path.parent
        for k, v in raw["inputs"].items():
            if not os.path.isabs(v):
                raw["inputs"][k] = str((base / v).resolve())
        if session not in raw:
            raise KeyError(f"session '{session}' not in {path} (have {[k for k in raw if isinstance(raw[k], dict) and 'scan' in raw[k]]})")
        return cls(raw, session, base)

    @property
    def inputs(self) -> dict:
        return self.raw["inputs"]

    @property
    def block(self) -> dict:
        return self.raw[self.session]

    def screen_cfg(self) -> dict:
        return self.block["screen"]

    def scan_cfg(self) -> dict:
        return self.block["scan"]


# ── campaign ────────────────────────────────────────────────────────────────────────────────────────
class OLGCampaign:
    """Config-driven OLG design campaign. Construct via `from_yaml`, inject objectives via
    `set_objectives`, then call `screen()` / `design()`."""

    def __init__(self, config: CampaignConfig):
        self.cfg = config
        c = config.raw

        # genetic code (S/J split) + alphabet -- traceable from config, not hardcoded
        self.alt_code = {k: list(v) for k, v in c["genetic_code"].items()}
        self.codon_table = build_restricted_codon_table(self.alt_code)
        self.alphabet = list(Constants.DEFAULT_ALPHABET) + list(self.alt_code)
        self.a_idx = {a: i for i, a in enumerate(self.alphabet)}
        self.j_idx, self.s_idx, self.z_idx = self.a_idx["J"], self.a_idx["S"], self.a_idx["Z"]
        # per-frame free AA alphabets: gene serine -> J (no S); AMP standard 20 (no J)
        self.gene_aas = "".join(a for a in Constants.DEFAULT_ALPHABET if a not in "SX") + "J"
        self.amp_aas = "".join(a for a in Constants.DEFAULT_ALPHABET if a != "X")
        self.stops = c["design"]["allowed_stop_codons"]
        self.arrangements = c["design"]["arrangements"]

        # gene + MSA + lock
        self.gene = _load_fasta(config.inputs["essential_fasta"])           # WT FabD (standard code)
        self.gene_j = "".join("J" if a == "S" else a for a in self.gene)    # serines -> J token
        self.n_gene = len(self.gene)
        self.cons = np.load(config.inputs["conservation_npy"])
        tau = c["lock"]["conservation_tau"]
        lock = sorted(set(c["lock"]["catalytic"]) | {int(i) + 1 for i in np.where(self.cons >= tau)[0]})
        self.lock1 = lock                                                    # 1-indexed (for preservation checks)
        self.lock0 = {p - 1 for p in lock}                                  # 0-indexed
        self.catalytic0 = sorted(p - 1 for p in c["lock"]["catalytic"])     # 0-indexed catalytic set
        self.msa_seqs = [ln.strip() for ln in Path(config.inputs["msa_a3m"]).read_text().splitlines()
                         if not ln.startswith(">")]

        # quartet feasibility tables (S/J), per arrangement
        self.lookups = {a: build_quartet_lookup(a, self.codon_table) for a in self.arrangements}
        self.free_sets = {a: build_free_gene_sets(self.lookups[a], self.gene_aas, self.amp_aas,
                                                  ["ATG"] + self.stops) for a in self.arrangements}

        self.gene_obj: FrameObjective | None = None
        self.amp_obj: AmpObjective | None = None      # FrameObjective + SequenceScorer (screen + design)
        self._pctile = None

    @classmethod
    def from_yaml(cls, path: str | Path, session: str) -> OLGCampaign:
        return cls(CampaignConfig.from_yaml(path, session))

    def set_objectives(self, *, gene: FrameObjective | None = None, amp: AmpObjective | None = None) -> None:
        """Inject the frame objectives. `screen()` needs only `amp` (a SequenceScorer); `design()` needs
        both. Set whichever a step requires (so the screen never loads the design models)."""
        if gene is not None:
            self.gene_obj = gene
        if amp is not None:
            self.amp_obj = amp

    # ── grid / positions ──
    def placement_grid(self, lengths, step):
        """(arrangement, offset, length) over the grid (offset keeps the stop in-gene)."""
        for arr in self.arrangements:
            for length in lengths:
                for off in range(0, self.n_gene - length, step):
                    yield arr, off, length

    def free_gene_positions(self, offset: int, length: int) -> list[int]:
        """0-indexed co-designed gene positions = overlap [offset, offset+length] inclusive minus lock."""
        return [i for i in range(offset, min(offset + length + 1, self.n_gene)) if i not in self.lock0]

    def _gene_fixed_positions(self, offset: int, length: int, *, fix_all: bool) -> list[tuple[int, str]]:
        free = set() if fix_all else set(self.free_gene_positions(offset, length))
        return [(i + 1, self.gene_j[i]) for i in range(self.n_gene) if i not in free]

    def _bias(self, device, *exclude: int) -> torch.Tensor:
        b = torch.zeros(len(self.alphabet), device=device)
        for idx in exclude:
            b[idx] = Constants.MIN_LOGIT
        return b

    # ── OLGDesign construction ──
    def build_design_config(self, arrangement: int, offset: int, length: int, rand_base: int,
                            device, *, fix_gene: bool = False) -> DesignConfig:
        """Gene free in the overlap-minus-lock (or fully fixed if fix_gene); AMP forces start/stop."""
        return DesignConfig(
            device=device, arrangement=Arrangement(arrangement), offset=offset,
            codon_table=self.codon_table, alphabet=self.alphabet, rand_base=rand_base, tqdm_disable=True,
            top_p=0.0, temperature=1.0,
            protein1=ProteinConfig(device=device, length=self.n_gene, alphabet_size=len(self.alphabet),
                                   fixed_positions=self._gene_fixed_positions(offset, length, fix_all=fix_gene),
                                   aa_bias=self._bias(device, self.s_idx, self.z_idx)),
            protein2=ProteinConfig(device=device, length=length, alphabet_size=len(self.alphabet),
                                   force_start=True, force_stop=True,
                                   aa_bias=self._bias(device, self.j_idx, self.z_idx)),
        )

    def reconstruct(self, arr: int, offset: int, amp_seq: str, mut_aa: str, device) -> OLGDesign:
        """Rebuild the designed OLG with both proteins fixed (ZeroOrder, CPU-friendly) -- for olgrbs."""
        gene = list(self.gene_j)
        for m in (mut_aa.split(";") if isinstance(mut_aa, str) and mut_aa else []):
            gene[int(m[1:-1]) - 1] = m[-1]
        logits = torch.zeros((1, len(self.alphabet)), device=device)
        cfg = DesignConfig(
            device=device, arrangement=Arrangement(arr), offset=offset,
            codon_table=self.codon_table, alphabet=self.alphabet, rand_base=0, tqdm_disable=True,
            protein1=ProteinConfig(device=device, length=self.n_gene, alphabet_size=len(self.alphabet),
                                   fixed_positions=[(i + 1, gene[i]) for i in range(self.n_gene)]),
            protein2=ProteinConfig(device=device, length=len(amp_seq), alphabet_size=len(self.alphabet),
                                   force_start=True, force_stop=True,
                                   fixed_positions=[(i + 1, amp_seq[i]) for i in range(len(amp_seq))]),
        )
        olg = OLGDesign(cfg)
        olg.initialize_decoder("ZeroOrder", frame=0, model=logits)
        olg.initialize_decoder("ZeroOrder", frame=1, model=logits)
        olg.decode_all(dummy_run=(False, False), mask_current=(False, False), force_safe=False, retry=80)
        return olg

    # ── realize a design into actual sequences (protein + nucleotide, downstream-useful forms) ──
    def sequences(self, arr, off, amp_seq, mut_aa, *, final_nt=None, rbs_opt=None, seed=0, device=None) -> dict:
        """Realize a design's actual sequences from the final RBS-optimized DNA. Pass `final_nt` (the scan's
        stored `rbs_nt`) to extract directly; if omitted, the RBS optimization is re-run (slow, but
        deterministic at `seed`) to recover it. Returns:

          full_dna        the complete designed inner-gene CDS (FabD, RBS-optimized) -- the orderable construct
          fabd_protein    the gene-frame translation (FabD + co-design mutations)
          amp_protein     the AMP peptide (Met + interior)
          amp_cds         the nested AMP ORF nucleotides (ATG ... stop)
          rbs_upstream    ~30 nt 5' of the AMP start (the SD/spacer the RBS optimization tuned)
          rbs_fold_window the exact slice OSTIR scores (+ rbs_start_in_window) -- reproduces rbs_rate
        """
        from olgrbs import optimize_rbs, rbs_window, score_rbs
        device = device or torch.device("cpu")
        olg = self.reconstruct(arr, off, amp_seq, mut_aa, device)
        if final_nt is None:                                       # no stored DNA -> re-run the RBS optimization
            res = optimize_rbs(olg, seed=seed, **dict(rbs_opt or self.cfg.scan_cfg()["rbs_opt"]))
            final_nt = res.best.nt if res.best else olg.string_quartet()[0]
        fabd_protein, amp_protein = olg.translate_sequences(final_nt)
        s = rbs_window(olg).inner_start_nt                          # AMP ATG index in the DNA
        lo = max(0, s - 40)                                        # OSTIR fold window start (driver's _FOLD_MARGIN)
        sc = score_rbs(final_nt[lo:s + 43], s - lo)               # cheap re-score of the final RBS (one OSTIR call)
        return {
            "arrangement": Arrangement(arr).name, "offset": int(off), "length": len(amp_protein),
            "fabd_protein": fabd_protein, "amp_protein": amp_protein,
            "full_dna": final_nt,
            "amp_cds": final_nt[s:s + 3 * (len(amp_protein) + 1)],  # Met + interior + stop
            "amp_start_nt": s,
            "rbs_upstream": final_nt[max(0, s - 30):s],
            "rbs_fold_window": final_nt[lo:s + 43], "rbs_start_in_window": s - lo,
            "rbs_rate": round(sc.expression, 1) if sc else None,
            "rbs_pctile": round(self.ecoli_percentile()(sc.expression), 1) if sc else None,
            "n_gene_mut": len(mut_aa.split(";")) if isinstance(mut_aa, str) and mut_aa else 0,
        }

    # ── E. coli RBS percentile ──
    def ecoli_percentile(self):
        """Percentile fn vs the E. coli OSTIR reference (cached). x -> percentile of real genes <= x."""
        if self._pctile is None:
            ref = sorted(float(r["expression"]) for r in csv.DictReader(open(self.cfg.inputs["ecoli_rbs_reference"])))
            self._pctile = lambda x: 100.0 * bisect.bisect_right(ref, x) / len(ref)
        return self._pctile

    # ── Step 1: screen ──
    def _screen_lengths(self) -> list[int]:
        sc = self.cfg.screen_cfg()
        return sc["lengths"] if "lengths" in sc else list(range(sc["length_min"], sc["length_max"] + 1))

    def screen(self, *, device=None, progress=None) -> pd.DataFrame:
        """Step 1 over the placement grid: feasibility + cheap per-placement metrics (no design models;
        only the AMP objective's batch `score_sequences`). `gene_free` (codesign) adds the RBS ceiling
        and gates on co-design feasibility; otherwise (fixed) the gate is WT-hosting. Returns one row
        per feasible placement."""
        if self.amp_obj is None:
            raise RuntimeError("screen() needs set_objectives(amp=...) first (the potency scorer)")
        lengths, sc = self._screen_lengths(), self.cfg.screen_cfg()
        placements = list(self.placement_grid(lengths, sc["offset_step"]))
        if self.cfg.block.get("gene_free", True):
            return self._screen_free(placements, lengths, sc["k_per_placement"], device, progress)
        return self._screen_fixed(placements, sc["k_per_placement"])

    def _score_potency(self, rows: list[dict], all_seqs: list[str]) -> pd.DataFrame:
        """Fill mic_mean/median/min from the AMP objective's batch scorer over the sampled space."""
        scores = np.asarray(self.amp_obj.score_sequences(all_seqs))
        for r in rows:
            s = scores[r.pop("_i0"):r.pop("_i1")]
            r["mic_mean"], r["mic_median"], r["mic_min"] = float(s.mean()), float(np.median(s)), float(s.min())
        return pd.DataFrame(rows)

    def _screen_free(self, placements, lengths, k, device, progress) -> pd.DataFrame:
        from .feasibility import sample_compatible, seq_entropy
        from .feasibility_screen import screen as feasibility_screen
        from .rbs_track import add_ceilings
        rbs = self.cfg.raw["rbs"]
        std_lk = {a: build_quartet_lookup(a, build_restricted_codon_table({})) for a in self.arrangements}
        results = feasibility_screen(placements, self.lookups, self.gene_j, self.lock0,
                                     self.gene_aas, self.amp_aas, self.free_sets, self.stops)
        add_ceilings(results, lengths, self.gene_j, self.lock0, self.lookups, self.free_sets,
                     self.gene_aas, self.amp_aas, self.stops, self.codon_table, self.alphabet,
                     pctile=self.ecoli_percentile(), nrep=rbs["nrep"], pad=rbs["pad"], w_up=rbs["w_up"],
                     opt=dict(rbs["opt"]), device=device, progress=progress)
        rows, all_seqs = [], []
        for p in results.filter(lambda p: p.feasible):
            seqs = sample_compatible(std_lk[p.arrangement], self.gene, p.offset, p.length, self.amp_aas,
                                     k, seed=1000 * p.offset + p.length)
            if not seqs:
                continue
            mean_bits, total_bits = seq_entropy(seqs)
            rows.append({"arrangement": p.arrangement, "arrangement_name": Arrangement(p.arrangement).name,
                         "offset": p.offset, "length": p.length, "n_compat": len(seqs),
                         "entropy_bits_per_pos": mean_bits, "entropy_total_bits": total_bits,
                         "rbs_min": p.rbs_min, "rbs_median": p.rbs_median, "rbs_ceiling": p.rbs_ceiling,
                         "rbs_pctile": p.rbs_pctile, "_i0": len(all_seqs), "_i1": len(all_seqs) + len(seqs)})
            all_seqs.extend(seqs)
        return self._score_potency(rows, all_seqs)

    def _screen_fixed(self, placements, k) -> pd.DataFrame:
        from .feasibility import sample_sequences, seq_entropy
        rows, all_seqs = [], []
        for arr, off, length in placements:
            seqs = sample_sequences(self.lookups[arr], self.gene_j, off, length, self.amp_aas,
                                    self.stops, k, seed=off)
            if not seqs:
                continue
            mean_bits, total_bits = seq_entropy(seqs)
            rows.append({"arrangement": arr, "arrangement_name": Arrangement(arr).name, "offset": off,
                         "length": length, "n_compat": len(seqs), "entropy_bits_per_pos": mean_bits,
                         "entropy_total_bits": total_bits, "_i0": len(all_seqs), "_i1": len(all_seqs) + len(seqs)})
            all_seqs.extend(seqs)
        return self._score_potency(rows, all_seqs)

    # ── Step 2: design ──
    def _target(self, screen_df, scan) -> list[tuple[int, int, int]]:
        """Targeted candidate placements, ranked by `select_by`. For gene_free (codesign) with a
        `cons_top_pct`: the top X% most-conserved windows OR any window over a catalytic residue
        (nest the AMP in the constrained core); otherwise every feasible placement."""
        df = screen_df.copy()
        if self.cfg.block.get("gene_free", True) and "cons_top_pct" in scan:
            df["_cons"] = [self.cons[o:o + ln + 1].mean() for o, ln in zip(df.offset, df.length)]
            df["_cat"] = [any(o <= p <= o + ln for p in self.catalytic0) for o, ln in zip(df.offset, df.length)]
            thr = df["_cons"].quantile(1 - scan["cons_top_pct"] / 100)
            df = df[(df["_cons"] >= thr) | df["_cat"]]
        return [(int(r.arrangement), int(r.offset), int(r.length))
                for r in df.sort_values(scan["select_by"]).itertuples()]

    def _design_cell(self, arr, off, length, *, w_gene, w_amp, free, fix_gene, rand_base,
                     n_sweep, retry, device) -> dict:
        """One greedy-ICM co-descent cell: attach both objectives, co-decode at fixed weights, return
        the converged AA-level design dict (objective scores keyed by each objective's metric_name)."""
        olg = OLGDesign(self.build_design_config(arr, off, length, rand_base, device, fix_gene=fix_gene))
        self.gene_obj.attach(olg, 0)
        self.amp_obj.attach(olg, 1)
        w1 = torch.ones(olg.decoders[0].logit_weight.shape, device=device) * w_gene
        w2 = torch.ones(olg.decoders[1].logit_weight.shape, device=device) * w_amp
        olg.decoders[0].logit_weight, olg.decoders[1].logit_weight = w1, w2
        olg.decode_all(dummy_run=(False, False), mask_current=(True, False), force_safe=False, retry=retry)
        for _ in range(n_sweep):
            order = olg.get_next_order("entropy")
            olg.decode_all_gibbs(dummy_run=(False, False), next_order=order, weight=(w1, w2),
                                 force_safe=False, retry=retry)
        t1, t2 = olg.translate_sequences()
        muts = [i for i in free if t1[i] != self.gene_j[i]]
        # gene_preserved checks the gene is WT OUTSIDE the overlap-minus-lock window; recompute that
        # window from (off, length) -- independent of the passed `free` (which is [] for the fixed
        # session) -- to match the original gene_preserved_outside semantics.
        outside_free = set(self.free_gene_positions(off, length))
        return {
            "amp_seq": t2,
            self.amp_obj.metric_name: self.amp_obj.score(olg, 1, free),
            self.gene_obj.metric_name: self.gene_obj.score(olg, 0, free),
            "n_mut": len(muts),
            "mut_aa": ";".join(f"{self.gene[i]}{i + 1}{t1[i]}" for i in muts),
            "mut_cons_mean": float(np.mean([self.cons[i] for i in muts])) if muts else float("nan"),
            "gene_preserved": all(t1[i] == self.gene_j[i] for i in range(self.n_gene) if i not in outside_free)
                              and all(t1[p - 1] == self.gene_j[p - 1] for p in self.lock1),
            "amp_n_ser": t2.count("S"),
        }

    def _rbs_step(self, arr, off, amp_seq, mut_aa, scan) -> dict:
        """Reconstruct the designed OLG and maximize its AMP RBS over FabD-synonymous codons (olgrbs)."""
        from olgrbs import optimize_rbs
        olg = self.reconstruct(arr, off, amp_seq, mut_aa, torch.device("cpu"))
        r = optimize_rbs(olg, **dict(scan["rbs_opt"]))
        base = r.base_expression   # base path depends only on the RNG seed (which this run fixes), not
                                   # on the search opts, so no separate default-opt run is needed
        if r.best is None:
            return {"rbs_base": round(base, 1), "rbs_rate": None, "rbs_pctile": None,
                    "rbs_xbase": None, "rbs_n_mut": None, "rbs_method": r.method, "rbs_nt": None}
        rate = r.best.score.expression
        return {"rbs_base": round(base, 1), "rbs_rate": round(rate, 1),
                "rbs_pctile": round(self.ecoli_percentile()(rate), 1),
                "rbs_xbase": round(rate / base, 1) if base else None,
                "rbs_n_mut": len(r.best.mutations), "rbs_method": r.method,
                "rbs_nt": r.best.nt}                              # the final RBS-optimized DNA (for sequences())

    def design(self, screen_df, out_csv, *, device=None) -> pd.DataFrame:
        """Step 2: design the targeted top_k placements. Per placement runs a cell per `weights` w_gene
        (codesign) or per de-novo `n_seeds` restart (fixed), each = co-design + RBS. Walks by rank,
        probe-skips placements that don't model-decode, resumes from `out_csv`. Returns the scan table.

        Both sessions emit the same columns (n_free, w_gene, fixed_gene, rbs_xbase, rbs_method, ...) -- a
        superset of the old per-session outputs; consumers read by name so the extra fixed-session columns
        are harmless."""
        from .scan_loop import run_scan
        if self.gene_obj is None or self.amp_obj is None:
            raise RuntimeError("design() needs set_objectives(gene=..., amp=...) first")
        scan, gene_free = self.cfg.scan_cfg(), self.cfg.block.get("gene_free", True)
        ranked = self._target(screen_df, scan)

        def make_cells(arr, off, length):
            if gene_free:
                cells = [{"mode": f"w{w}", "w_gene": w, "fixed_gene": False} for w in scan["weights"]]
                if scan.get("fixed_gene_cell"):
                    cells.append({"mode": "fixed", "w_gene": 1.0, "fixed_gene": True})
                return cells
            return [{"mode": f"seed{s}", "seed": s, "w_gene": 1.0, "fixed_gene": True}
                    for s in range(scan["n_seeds"])]

        def run_cell(arr, off, length, cell):
            fix = cell["fixed_gene"]
            free = [] if fix else self.free_gene_positions(off, length)
            d = self._design_cell(arr, off, length, w_gene=cell["w_gene"], w_amp=1.0, free=free,
                                  fix_gene=fix, rand_base=1000 * cell.get("seed", 0),
                                  n_sweep=scan["n_sweep"], retry=scan["retry"], device=device)
            return {"n_free": len(free), **d, **self._rbs_step(arr, off, d["amp_seq"], d["mut_aa"], scan)}

        with torch.inference_mode():
            run_scan(ranked, make_cells, run_cell, Path(out_csv),
                     top_k=scan["top_k"], max_tried=scan["max_placements_tried"])
        return pd.read_csv(out_csv)

    # ── gene plausibility (masked marginal log-prob over the free positions) ──
    @staticmethod
    def masked_marginal_logprob(gene_decoder, positions, tokens) -> float:
        """Mean unweighted masked-marginal log-prob of the chosen gene AAs over `positions`."""
        gene_decoder.logit_weight = torch.ones_like(gene_decoder.logit_weight)
        lls = []
        for p in positions:
            cur = int(tokens[p])
            gene_decoder.edit_S(p, gene_decoder.MASK_TOKEN, inplace=True)
            lp = torch.log_softmax(gene_decoder._forward_pass()[p].float(), dim=-1)
            gene_decoder.edit_S(p, cur, inplace=True)
            lls.append(lp[cur].item())
        return float(np.mean(lls)) if lls else float("nan")


def _load_fasta(path: str) -> str:
    return "".join(ln.strip() for ln in Path(path).read_text().splitlines()
                   if ln and not ln.startswith(">"))
