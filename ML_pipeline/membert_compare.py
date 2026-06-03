"""
membert_compare.py  --  fair, common-task benchmark vs MemBERT
==============================================================
A like-for-like comparison is only fair on a COMMON task. We reduce both methods to
block-level binary KEY detection on the SAME labeled raw-byte blocks, and report
precision/recall/F1 for each method:

  * structure verifier  -- per block: a private key whose DER loads (csp_verify);
  * GBDT classifier      -- the deployed model on features extracted from the block
                            (feature-limited: region/temporal default, since a lone block
                            carries no temporal context -- disclose this);
  * MemBERT              -- via the adapter `membert_predict(raw_bytes) -> float in [0,1]`
                            that YOU wire to the real model (stub raises until implemented).

The labeled set is built from the field artifacts: positives = ground-truth secret keys
(field_hook), hard negatives = system-flagged blocks that are NOT ground-truth keys
(field_predict/field_detect outputs). No numbers are written into the paper until this is
actually run with a real MemBERT adapter.

USAGE
-----
    python membert_compare.py --indir field_modules --csv dataset/real_crypto_features.csv \
        --out membert_compare.json --tex membert_compare.tex
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collector"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, FEATURE_NAMES
import csp_verify as cv

CLASSES = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]


# ---------------------------------------------------------------- MemBERT adapter (WIRE ME)
def membert_predict(raw_bytes):
    """Return P(KEY) in [0,1] for a raw memory block, using the real MemBERT model.
    Implement this with the published MemBERT weights/code (see MEMBERT_BENCHMARK.md).
    Until implemented it raises, and MemBERT is omitted from the comparison."""
    raise NotImplementedError("Wire membert_predict() to the real MemBERT model "
                              "(see MEMBERT_BENCHMARK.md). MemBERT row is skipped until then.")


# ---------------------------------------------------------------- our methods
def verifier_is_key(raw):
    """Per-block structure verification (DER private key). AES/Curve25519 need region
    context and are therefore out of scope for a lone block -- a documented limitation."""
    try:
        from cryptography.hazmat.primitives.serialization import load_der_private_key
        load_der_private_key(raw, password=None)
        return True
    except Exception:
        return False


def load_classifier(csv):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(csv)
    X = df[FEATURE_NAMES].astype(float).values
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(df["label"].astype(str).values)
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=1000, depth=9, learning_rate=0.05,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
    except Exception:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    m.fit(X, y)
    return m, le


def classifier_is_key(model, le, raw):
    feats = extract_features(raw, 2, 0, 1)   # region=heap, no temporal context (feature-limited)
    p = np.asarray(model.predict_proba(np.array([feats])))[0]
    return int(np.argmax(p)) == list(le.classes_).index("KEY")


# ---------------------------------------------------------------- dataset + metrics
def _norm(h): return (h or "").lower().replace("0x", "")


def build_labeled_blocks(indir):
    """positives = ground-truth secret keys; negatives = flagged blocks that match no GT key."""
    gt_hex, pred_hex = set(), set()
    for f in glob.glob(os.path.join(indir, "*.groundtruth.json")):
        for k in json.load(open(f, encoding="utf-8")).get("ground_truth_keys", []):
            gt_hex.add(_norm(k.get("value_hex")))
    for f in glob.glob(os.path.join(indir, "*.predictions.json")):
        for b in json.load(open(f, encoding="utf-8")).get("predicted_key_blocks", []):
            pred_hex.add(_norm(b.get("value_hex")))
    pos = [h for h in gt_hex if h]
    neg = [h for h in pred_hex if h and not any(h == g or h in g or g in h for g in gt_hex)]
    return pos, neg


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p * 100, 1), round(r * 100, 1), round(f * 100, 1)


def evaluate(name, predict_fn, pos, neg):
    tp = sum(1 for h in pos if predict_fn(bytes.fromhex(h)))
    fn = len(pos) - tp
    fp = sum(1 for h in neg if predict_fn(bytes.fromhex(h)))
    p, r, f = prf(tp, fp, fn)
    return {"method": name, "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="field_modules")
    ap.add_argument("--csv", default="dataset/real_crypto_features.csv")
    ap.add_argument("--out", default="membert_compare.json")
    ap.add_argument("--tex", default="membert_compare.tex")
    a = ap.parse_args()

    pos, neg = build_labeled_blocks(a.indir)
    if not pos:
        raise SystemExit("No positives (ground-truth keys) found; run the field collection first.")
    print(f"common-task block set: {len(pos)} KEY positives, {len(neg)} hard negatives")

    rows = [evaluate("Structure verifier (DER)", verifier_is_key, pos, neg)]
    try:
        model, le = load_classifier(a.csv)
        rows.append(evaluate("GBDT classifier (feature-limited)",
                              lambda raw: classifier_is_key(model, le, raw), pos, neg))
    except Exception as e:
        print(f"[warn] classifier skipped: {e}")
    try:
        membert_predict(b"\x00" * 16)   # probe adapter
        rows.append(evaluate("MemBERT", lambda raw: membert_predict(raw) >= 0.5, pos, neg))
    except NotImplementedError:
        print("[note] MemBERT adapter not implemented -> MemBERT row omitted "
              "(wire membert_predict(); see MEMBERT_BENCHMARK.md).")
    except Exception as e:
        print(f"[warn] MemBERT skipped: {e}")

    json.dump({"n_pos": len(pos), "n_neg": len(neg), "rows": rows}, open(a.out, "w"), indent=2)
    body = "\n".join(f"{r['method']} & {r['precision']}\\% & {r['recall']}\\% & {r['f1']}\\% \\\\" for r in rows)
    tex = ("% Auto-generated by membert_compare.py -- common-task (block-level binary KEY detection).\n"
           "\\begin{table}[t]\\centering\\scriptsize\n"
           f"\\caption{{Common-task comparison (block-level binary KEY detection; {len(pos)} positives, "
           f"{len(neg)} hard negatives). MemBERT row appears once \\code{{membert\\_predict}} is wired.}}\n"
           "\\label{tab:membert_quant}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
           "\\textbf{Method} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    open(a.tex, "w", encoding="utf-8").write(tex)
    print(json.dumps(rows, indent=2))
    print(f"\nWrote {a.out} and {a.tex}")


if __name__ == "__main__":
    main()
