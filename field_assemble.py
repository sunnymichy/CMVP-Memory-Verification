"""
field_assemble.py  --  merge predictions + ground truth into a field_eval module file
=====================================================================================
Combines one module's prediction file (field_predict.py) and ground-truth file
(field_hook.py, or a Detours dump) into the single JSON that field_eval.py scores.

USAGE
-----
    python field_assemble.py \
        --predictions field_modules/moduleA.predictions.json \
        --groundtruth field_modules/moduleA.groundtruth.json \
        --out field_modules/moduleA.json
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--funnel", default=None, help="optional JSON file with per-phase counts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pred = json.load(open(a.predictions, encoding="utf-8"))
    gt = json.load(open(a.groundtruth, encoding="utf-8"))
    if pred.get("module") != gt.get("module"):
        print(f"WARNING: module names differ: '{pred.get('module')}' vs '{gt.get('module')}'")

    merged = {
        "module": gt.get("module") or pred.get("module"),
        "library": gt.get("library") or pred.get("library"),
        "ground_truth_keys": gt.get("ground_truth_keys", []),
        "predicted_key_blocks": pred.get("predicted_key_blocks", []),
    }
    if a.funnel:
        merged["funnel"] = json.load(open(a.funnel, encoding="utf-8"))

    json.dump(merged, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"merged -> {a.out}  "
          f"({len(merged['ground_truth_keys'])} GT keys, "
          f"{len(merged['predicted_key_blocks'])} predicted KEY blocks)")


if __name__ == "__main__":
    main()
