"""
experiment_heuristic_sweep.py
=============================
Sensitivity sweep of the Phase-5a heuristic pre-scoring threshold S in {1,2,3}.

WHAT THIS RESOLVES
------------------
Reviewer concern: an earlier draft asserted unverified numbers for the S-threshold
sweep ("KEY retention moved <=0.3 pp; elimination 58-81%"). This script measures the
two quantities directly so any reported figure is backed by a run:
  - elimination rate  = fraction of candidates labeled NON_CRYPTO at S <= t
  - KEY retention     = fraction of true-KEY samples that survive (S > t)

!!! TWO IMPORTANT CAVEATS -- READ BEFORE USING NUMBERS !!!
---------------------------------------------------------
1) PAPER/CODE MISMATCH. Section III-E2 of the manuscript describes a SIX binary-signal
   scorer with S_max = 8 (high entropy +2, std key length +2, crypto region +1,
   temporal stability +1, low ASCII ratio +1, near-uniform byte distribution +1).
   The repository's heuristic_scorer.py instead implements a 0-100 WEIGHTED scorer.
   These are different systems. This script implements the PAPER's 6-signal scorer so
   the sweep matches the text; if you intend to ship the 0-100 scorer, the paper text
   must be rewritten to match it instead.

2) MISSING RAW-BYTE SIGNALS. Two of the six signals ("low ASCII ratio",
   "near-uniform byte distribution") require the raw block bytes, which are NOT present
   in the feature CSV. Here, near-uniform byte distribution is approximated from F2
   (chi-square; only valid for length >= 256) and ASCII ratio is unavailable and set to
   0 unless a raw-bytes column is provided. The elimination/retention numbers are
   therefore a LOWER-FIDELITY reconstruction. For a paper-grade number, compute the
   score at collection time from raw bytes.

   If the csv is the synthetic corpus (generate_synthetic_dataset.py output), the
   numbers are procedural only and must not be presented as empirical evidence.

USAGE
-----
    python experiment_heuristic_sweep.py --csv dataset/crypto_features.csv \
        --out results_sweep.json --tex results_sweep.tex
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd

FEATURES = [
    "F1_entropy", "F2_chi_square", "F3_length", "F4_standard_key_len",
    "F5_standard_iv_len", "F6_memory_region", "F7_change_count",
    "F8_change_pattern", "F9_entropy_length_interaction", "F10_high_confidence_key",
]


def six_signal_score(row):
    """Paper Sec III-E2: six binary signals, S_max = 8.
    high entropy (H>=7.5) +2; standard key length +2; crypto memory region +1;
    temporal stability +1; low ASCII ratio +1; near-uniform byte distribution +1."""
    s = 0
    # +2 high entropy
    if row["F1_entropy"] >= 7.5:
        s += 2
    # +2 standard key length
    if row["F4_standard_key_len"] == 1.0:
        s += 2
    # +1 crypto memory region (DLL data=1 or heap/stack=2)
    if int(row["F6_memory_region"]) in (1, 2):
        s += 1
    # +1 temporal stability (STATIC=0 or PARTIAL=1 patterns => key-like persistence)
    if int(row["F8_change_pattern"]) in (0, 1):
        s += 1
    # +1 near-uniform byte distribution (approx via chi-square; only valid len>=256)
    if row["F2_chi_square"] > 0 and row["F2_chi_square"] < 350:
        s += 1
    # +1 low ASCII ratio -- requires raw bytes (unavailable in feature CSV)
    # left as 0; supply a precomputed 'low_ascii' column to enable.
    if "low_ascii" in row and row["low_ascii"] == 1.0:
        s += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/crypto_features.csv")
    ap.add_argument("--out", default="results_sweep.json")
    ap.add_argument("--tex", default="results_sweep.tex")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset missing columns: {missing}")

    scores = df.apply(six_signal_score, axis=1).to_numpy()
    is_key = (df["label"].astype(str) == "KEY").to_numpy()
    n_total = len(df)
    n_key = int(is_key.sum())

    results = {"dataset": args.csv, "n_total": n_total, "n_key": n_key,
               "ascii_signal_available": "low_ascii" in df.columns, "sweep": {}}

    for t in (1, 2, 3):
        eliminated = scores <= t
        elim_rate = float(eliminated.mean())
        # KEY retention = fraction of true KEY NOT eliminated
        key_ret = float((is_key & ~eliminated).sum() / max(n_key, 1))
        results["sweep"][f"S<={t}"] = {
            "elimination_rate_pct": round(elim_rate * 100, 2),
            "key_retention_pct": round(key_ret * 100, 2),
            "n_eliminated": int(eliminated.sum()),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sw = results["sweep"]
    base = sw["S<=2"]
    ret_vals = [sw[k]["key_retention_pct"] for k in sw]
    elim_vals = [sw[k]["elimination_rate_pct"] for k in sw]
    tex = (
        "%% Auto-generated by experiment_heuristic_sweep.py -- VERIFY before use.\n"
        "%% Suggested replacement for the sentence in Sec. III-E2:\n"
        f"The threshold $S = 2$ was fixed on the validation set; a sweep over "
        f"$S \\in \\{{1,2,3\\}}$ moved KEY retention within "
        f"[{min(ret_vals):.1f}\\%%, {max(ret_vals):.1f}\\%%] while the elimination "
        f"rate ranged {min(elim_vals):.0f}--{max(elim_vals):.0f}\\%% "
        f"(at $S=2$: {base['elimination_rate_pct']:.0f}\\%% eliminated, "
        f"{base['key_retention_pct']:.1f}\\%% KEY retained).\n"
    )
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(json.dumps(results, indent=2))
    if not results["ascii_signal_available"]:
        print("\nWARNING: 'low ASCII ratio' signal unavailable (no raw bytes); "
              "elimination is a lower bound. See header.")
    print(f"Wrote {args.out} and {args.tex}")


if __name__ == "__main__":
    main()
