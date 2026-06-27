"""
reproduce_tabnet.py  --  TUNED TabNet baseline for Table 7 / Table 10
=====================================================================
Run from ml_pipeline/. Needs pytorch-tabnet. SLOW (neural net). Deployed comparison
model = XGBoost (same config as run_pathB.py).

Fair-baseline protocol (fixes the earlier untuned collapse):
  * StandardScaler on the 10 features (neural nets need scaling; tree models do not).
  * Small validation-set search over (n_d=n_a, n_steps); pick best by validation W-F1.
  * Single 20% test split + 5-fold StratifiedKFold(shuffle, seed=42); paired t-test vs
    XGBoost on the same folds.

USAGE
-----
    python reproduce_tabnet.py            # writes results_real_tabnet.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats as st

CSV = "dataset/real_crypto_features.csv"
CLASSES = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]
LE = LabelEncoder().fit(CLASSES)
CONFIGS = [(16, 3), (32, 4), (64, 5)]   # (n_d=n_a, n_steps)


def make_tabnet(n_d, n_steps):
    from pytorch_tabnet.tab_model import TabNetClassifier
    return TabNetClassifier(n_d=n_d, n_a=n_d, n_steps=n_steps, gamma=1.5,
                            lambda_sparse=1e-4, optimizer_params=dict(lr=2e-2),
                            mask_type="sparsemax", seed=42, verbose=0)


def xgb():
    import xgboost as x
    return x.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           objective="multi:softprob", num_class=5,
                           eval_metric="mlogloss", random_state=42, verbosity=0)


def fit_tn(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr.astype(np.float32), ytr, eval_set=[(Xva.astype(np.float32), yva)],
              eval_metric=["accuracy"], max_epochs=200, patience=30,
              batch_size=1024, virtual_batch_size=128)
    return model


def wf1(model, sc, Xte, yte):
    return f1_score(yte, model.predict(sc.transform(Xte).astype(np.float32)),
                    average="weighted", zero_division=0) * 100


def train_eval(Xtr, ytr, Xte, yte, n_d, n_steps, ret_m=False):
    """Scale on (inner) train, carve val for early stopping, fit, eval on Xte."""
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
    sc = StandardScaler().fit(Xa)
    m = fit_tn(make_tabnet(n_d, n_steps), sc.transform(Xa), ya, sc.transform(Xv), yv)
    w = wf1(m, sc, Xte, yte)
    mac = f1_score(yte, m.predict(sc.transform(Xte).astype(np.float32)), average="macro", zero_division=0) * 100
    return (w, mac, m, sc) if ret_m else (w, mac)


def main():
    df = pd.read_csv(CSV)
    X = df.iloc[:, :10].values.astype(np.float64)
    y = LE.transform(df["label"].values.astype(str))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- validation-set search for (n_d, n_steps) ---
    Xt2, Xval, yt2, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
    val_scores = {}
    for (nd, ns) in CONFIGS:
        w, _ = train_eval(Xt2, yt2, Xval, yval, nd, ns)
        val_scores[(nd, ns)] = w
        print(f"  config n_d={nd}, n_steps={ns}: val W-F1 = {w:.2f}")
    best = max(val_scores, key=val_scores.get)
    print(f"  selected config: n_d={best[0]}, n_steps={best[1]} (val W-F1 {val_scores[best]:.2f})")

    # --- single 20% split with best config ---
    w, mac = train_eval(Xtr, ytr, Xte, yte, best[0], best[1])
    single = {"W_F1": round(w, 2), "M_F1": round(mac, 2)}

    # --- 5-fold CV (best config) + paired t-test vs XGBoost on same folds ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tn_f, xg_f = [], []
    for tr, te in skf.split(X, y):
        w_, _ = train_eval(X[tr], y[tr], X[te], y[te], best[0], best[1])
        tn_f.append(w_)
        xm = xgb(); xm.fit(X[tr], y[tr])
        xg_f.append(f1_score(y[te], xm.predict(X[te]), average="weighted", zero_division=0) * 100)
    tn_f, xg_f = np.array(tn_f), np.array(xg_f)
    t, pv = st.ttest_rel(tn_f, xg_f)
    out = {"config": {"n_d": best[0], "n_steps": best[1], "scaled": True},
           "val_scores": {f"{k[0]}_{k[1]}": round(v, 2) for k, v in val_scores.items()},
           "single_split": single,
           "cv5_mean": round(float(tn_f.mean()), 2), "cv5_sd": round(float(tn_f.std(ddof=1)), 2),
           "cv5_folds": [round(v, 2) for v in tn_f.tolist()],
           "xgb_cv5_mean_thisrun": round(float(xg_f.mean()), 2),
           "ttest_vs_xgb": {"delta": round(float(tn_f.mean() - xg_f.mean()), 2),
                            "t": round(float(t), 3), "p": round(float(pv), 4)}}
    json.dump(out, open("results_real_tabnet.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    print("DONE_TABNET_TUNED")


if __name__ == "__main__":
    main()
