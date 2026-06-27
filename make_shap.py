"""
make_shap.py  --  generate SHAP figures for the deployed XGBoost model (real corpus)
=====================================================================================
Produces:
  ../fig.2.png            : SHAP waterfall for one true-positive KEY prediction
  ../fig_shap_summary.png : SHAP beeswarm summary over the test set (KEY class)

All explanations are computed on the REAL corpus; no values are hand-set.

USAGE
-----
    cd ml_pipeline
    python make_shap.py --csv dataset/real_crypto_features.csv --out ..
Requires: catboost, shap, matplotlib  (pip install catboost shap matplotlib)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CLASSES = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]
FEATURES = ["F1_entropy", "F2_chi_square", "F3_length", "F4_standard_key_len",
            "F5_standard_iv_len", "F6_memory_region", "F7_change_count",
            "F8_change_pattern", "F9_entropy_length_interaction", "F10_high_confidence_key"]
PRETTY = ["F1 entropy", "F2 chi2", "F3 length", "F4 key-len", "F5 iv-len",
          "F6 region", "F7 changes", "F8 pattern", "F9 ent*len", "F10 hi-conf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/real_crypto_features.csv")
    ap.add_argument("--out", default="..")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import shap
    import xgboost as xgb

    df = pd.read_csv(args.csv)
    X = df[FEATURES].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    le = LabelEncoder().fit(CLASSES)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    # Deployed model: XGBoost (matches run_pathB.py --deployed XGBoost).
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              objective="multi:softprob", num_class=5, eval_metric="mlogloss",
                              random_state=args.seed, verbosity=0)
    model.fit(Xtr, le.transform(ytr))
    pred = le.inverse_transform(np.asarray(model.predict(Xte)).ravel().astype(int))

    key_idx = le.transform(["KEY"])[0]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)            # list (per class) or (n, f, c)
    if isinstance(sv, list):
        sv_key = sv[key_idx]
        base = explainer.expected_value[key_idx]
    else:
        sv_key = sv[:, :, key_idx]
        base = np.asarray(explainer.expected_value)[key_idx]

    # pick a confident true-positive KEY sample
    tp = np.where((yte == "KEY") & (pred == "KEY"))[0]
    if len(tp) == 0:
        tp = np.where(yte == "KEY")[0]
    j = tp[int(np.argmax(np.abs(sv_key[tp]).sum(axis=1)))]

    # waterfall
    expl = shap.Explanation(values=sv_key[j], base_values=base,
                            data=Xte[j], feature_names=PRETTY)
    plt.figure()
    try:
        shap.plots.waterfall(expl, show=False, max_display=10)
    except Exception:
        shap.waterfall_plot(expl, show=False)
    plt.tight_layout()
    wpath = os.path.join(args.out, "fig.2.png")
    plt.savefig(wpath, dpi=200, bbox_inches="tight"); plt.close()

    # beeswarm summary (KEY class)
    plt.figure()
    shap.summary_plot(sv_key, Xte, feature_names=PRETTY, show=False, plot_size=(7, 4))
    plt.tight_layout()
    spath = os.path.join(args.out, "fig_shap_summary.png")
    plt.savefig(spath, dpi=200, bbox_inches="tight"); plt.close()

    print(f"Saved {wpath} (true-positive KEY waterfall, sample index {j})")
    print(f"Saved {spath} (KEY-class SHAP summary)")
    print("Top mean|SHAP| features (KEY):")
    order = np.argsort(np.abs(sv_key).mean(axis=0))[::-1]
    for k in order[:5]:
        print(f"  {PRETTY[k]:<12s} {np.abs(sv_key[:,k]).mean():.4f}")


if __name__ == "__main__":
    main()
