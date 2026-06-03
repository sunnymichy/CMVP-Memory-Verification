"""
experiment_heuristic_sweep.py
=============================
Sensitivity sweep of the Phase-5a heuristic pre-score threshold S in {1,2,3}:
  - elimination rate = fraction of candidates labeled NON_CRYPTO at S <= t
  - KEY retention    = fraction of true-KEY samples that survive (S > t)

`six_signal_score` (paper Sec. III-E2: six binary signals, S_max = 8) is imported by
run_pathB.py. Two signals ("low ASCII ratio", "near-uniform byte distribution") need the
raw block bytes; here the latter is approximated from F2 (chi-square, valid for len>=256)
and the former is enabled only if a precomputed 'low_ascii' column is present.

USAGE
-----
    python experiment_heuristic_sweep.py --csv dataset/real_crypto_features.csv \
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
    """Six binary signals, S_max = 8: high entropy (+2), standard key length (+2),
    crypto region (+1), temporal stability (+1), near-uniform bytes (+1), low ASCII (+1)."""
    s = 0
    if row["F1_entropy"] >= 7.5:
        s += 2
    if row["F4_standard_key_len"] == 1.0:
        s += 2
    if int(row["F6_memory_region"]) in (1, 2):
        s += 1
    if int(row["F8_change_pattern"]) in (0, 1):
        s += 1
    if row["F2_chi_square"] > 0 and row["F2_chi_square"] < 350:
        s += 1
    if "low_ascii" in row and row["low_ascii"] == 1.0:
        s += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/real_crypto_features.csv")
    ap.add_argument("--out", default="results_sweep.json")
    ap.add_argument("--tex", default="results_sweep.tex")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset missing columns: {missing}")

    scores = df.apply(six_signal_score, axis=1).to_numpy()
    is_key = (df["label"].astype(str) == "KEY").to_numpy()
    n_key = int(is_key.sum())
    results = {"dataset": args.csv, "n_total": int(len(df)), "n_key": n_key,
               "ascii_signal_available": "low_ascii" in df.columns, "sweep": {}}
    for t in (1, 2, 3):
        eliminated = scores <= t
        results["sweep"][f"S<={t}"] = {
            "elimination_rate_pct": round(float(eliminated.mean()) * 100, 2),
            "key_retention_pct": round(float((is_key & ~eliminated).sum() / max(n_key, 1)) * 100, 2),
            "n_eliminated": int(eliminated.sum())}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    if not results["ascii_signal_available"]:
        print("\nNOTE: 'low ASCII ratio' signal unavailable (no raw bytes); elimination is a lower bound.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
