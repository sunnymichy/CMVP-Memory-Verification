"""
field_eval.py  --  honest field evaluation on real CMVP-style modules
=====================================================================
Scores the read-only pipeline against ground truth obtained by Phase-6 cross-
verification (DLL injection / API hooking), per module, and emits per-module metrics,
overall recall with a Wilson 95% CI, and a paste-ready LaTeX table. It computes NOTHING
unless you supply real per-module inputs, so no number can be invented.

WORKFLOW (all on the Windows host; see FIELD_EVAL_PROTOCOL.md)
-------------------------------------------------------------
For each target module you produce ONE json file in --indir:

  {
    "module": "Module A (OpenSSL)",          # display name
    "library": "OpenSSL",                     # library family
    "ground_truth_keys": [                    # from Phase-6 hooked run (actual key bytes)
       {"id": "k1", "algo": "AES-256-CBC", "value_hex": "ab12...", "len": 32},
       ...
    ],
    "predicted_key_blocks": [                 # blocks the read-only pipeline labelled KEY
       {"addr": "0x1f3a...", "value_hex": "ab12...", "region": 2, "confidence": 0.93},
       ...
    ],
    "funnel": {                               # OPTIONAL: candidate counts per phase
       "phase1_regions": 50, "phase3_blocks": 2100, "phase5a_blocks": 630, "phase5b_key": 12
    }
  }

A ground-truth key counts as DETECTED (logical-key TP) if at least one predicted KEY
block matches it by value (exact, or GT value contained in the block, or block contained
in GT value -- to allow for padding/headers). A predicted KEY block that matches no
ground-truth key is a false-positive block (FP). Definitions follow the protocol doc.

USAGE
-----
    python field_eval.py --indir field_modules --out field_results.json --tex field_table.tex
"""

import argparse
import glob
import json
import math
import os


def _norm(h):
    return (h or "").lower().replace("0x", "").replace(" ", "")


def _matches(gt_hex, blk_hex):
    g, b = _norm(gt_hex), _norm(blk_hex)
    if not g or not b:
        return False
    return g == b or g in b or b in g


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_module(m):
    gts = m.get("ground_truth_keys", [])
    preds = m.get("predicted_key_blocks", [])
    detected_ids, matched_blocks = set(), set()
    for gi, gt in enumerate(gts):
        for bi, blk in enumerate(preds):
            if _matches(gt.get("value_hex"), blk.get("value_hex")):
                detected_ids.add(gi)
                matched_blocks.add(bi)
    n_keys = len(gts)
    det = len(detected_ids)
    miss = n_keys - det
    fp_block = len(preds) - len(matched_blocks)
    tp_block = len(matched_blocks)
    recall = det / n_keys if n_keys else 0.0
    block_prec = tp_block / len(preds) if preds else 0.0
    act_prec = det / (det + fp_block) if (det + fp_block) else 0.0
    return {
        "module": m.get("module", "?"), "library": m.get("library", "?"),
        "keys": n_keys, "detected": det, "missed": miss, "fp_block": fp_block,
        "recall_pct": round(recall * 100, 1),
        "block_precision_pct": round(block_prec * 100, 1),
        "actionable_precision_pct": round(act_prec * 100, 1),
        "funnel": m.get("funnel"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="field_modules", help="dir of per-module json files")
    ap.add_argument("--out", default="field_results.json")
    ap.add_argument("--tex", default="field_table.tex")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.json")))
    # Score ONLY assembled module files (both keys present). This skips the intermediate
    # *.groundtruth.json / *.predictions.json files, which are also "*.json".
    per_module = []
    timings = []   # per-module wall-clock from field_predict.py (see FIELD_EVAL_PROTOCOL.md sec. 6)
    for f in files:
        if f.endswith(".groundtruth.json") or f.endswith(".predictions.json"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        if "ground_truth_keys" not in d or "predicted_key_blocks" not in d:
            print(f"  [skip non-assembled] {os.path.basename(f)}")
            continue
        per_module.append(score_module(d))
        # timing lives in the assembled json if carried over, else the sibling predictions.json
        t = d.get("timing")
        if t is None:
            sib = f[:-5] + ".predictions.json"
            if os.path.exists(sib):
                t = json.load(open(sib, encoding="utf-8")).get("timing")
        if t and "detect_s" in t:
            timings.append(t)
    if not per_module:
        raise SystemExit(f"No assembled module files in {args.indir}/ "
                         f"(need both ground_truth_keys and predicted_key_blocks; "
                         f"run field_assemble.py first). See FIELD_EVAL_PROTOCOL.md.")
    tot_keys = sum(r["keys"] for r in per_module)
    tot_det = sum(r["detected"] for r in per_module)
    tot_fp = sum(r["fp_block"] for r in per_module)
    tot_tp_block = sum(r["detected"] for r in per_module)  # block-level TP approx = detected
    lo, hi = wilson_ci(tot_det, tot_keys)
    overall = {
        "n_modules": len(per_module), "total_keys": tot_keys, "total_detected": tot_det,
        "total_missed": tot_keys - tot_det, "total_fp_block": tot_fp,
        "recall_pct": round(tot_det / tot_keys * 100, 1) if tot_keys else 0.0,
        "wilson95_lo_pct": round(lo * 100, 1), "wilson95_hi_pct": round(hi * 100, 1),
        "actionable_precision_pct": round(tot_det / (tot_det + tot_fp) * 100, 1) if (tot_det + tot_fp) else 0.0,
    }
    if timings:
        import statistics
        det = [t["detect_s"] for t in timings]
        tot = [t["total_s"] for t in timings if t.get("total_s") is not None]
        overall["n_timed_modules"] = len(det)
        overall["mean_detect_s"] = round(statistics.mean(det), 1)
        overall["mean_total_s"] = round(statistics.mean(tot), 1) if tot else None
    results = {"per_module": per_module, "overall": overall}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # paste-ready LaTeX table
    rows = []
    for r in per_module:
        rows.append(f"{r['module']} & {r['keys']} & {r['detected']} & {r['missed']} & "
                    f"{r['fp_block']} & {r['recall_pct']:.0f}\\% & {r['actionable_precision_pct']:.0f}\\% \\\\")
    body = "\n".join(rows)
    tex = (
        "% Auto-generated by field_eval.py from REAL field runs -- verify before use.\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Per-module field key detection (ground truth via Phase-6 cross-verification). "
        "Recall is at the logical-key level; Act.\\ Prec.\\ counts each false-positive KEY block as a "
        "separate review action.}\n\\label{tab:field}\n\\scriptsize\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "\\textbf{Module (Library)} & \\textbf{Keys} & \\textbf{Det.} & \\textbf{Miss} & \\textbf{FP} & "
        "\\textbf{Rec.} & \\textbf{Act.\\ Prec.} \\\\\n\\midrule\n"
        f"{body}\n\\midrule\n"
        f"\\textbf{{Total}} & \\textbf{{{overall['total_keys']}}} & \\textbf{{{overall['total_detected']}}} & "
        f"\\textbf{{{overall['total_missed']}}} & \\textbf{{{overall['total_fp_block']}}} & "
        f"\\textbf{{{overall['recall_pct']:.0f}\\%}} & \\textbf{{{overall['actionable_precision_pct']:.0f}\\%}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        f"% Overall recall {overall['recall_pct']}% "
        f"(95% Wilson CI [{overall['wilson95_lo_pct']}%, {overall['wilson95_hi_pct']}%]); "
        f"{overall['total_detected']}/{overall['total_keys']} keys.\n"
    )
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out} and {args.tex}")
    print(f"Overall: {tot_det}/{tot_keys} keys recovered "
          f"({overall['recall_pct']}%, 95% Wilson CI [{overall['wilson95_lo_pct']}%, {overall['wilson95_hi_pct']}%])")
    if timings:
        msg = f"Per-module wall-clock: detect mean {overall['mean_detect_s']}s ({overall['n_timed_modules']} modules)"
        if overall.get("mean_total_s") is not None:
            msg += f", total mean {overall['mean_total_s']}s (incl. one-time model load)"
        print(msg)
    else:
        print("Per-module wall-clock: no timing found (re-run field_predict.py to record it).")
    print("Paste field_table.tex into the paper ONLY after a real field run.")


if __name__ == "__main__":
    main()
