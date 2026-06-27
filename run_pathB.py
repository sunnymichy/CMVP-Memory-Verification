"""
run_pathB.py  --  Path-B real-data orchestrator
================================================
Regenerates EVERY paper number from a given dataset CSV and emits:
  - results_real.json : machine-readable results
  - results_real.tex  : LaTeX \\newcommand macros that overwrite the hard-coded
                        \\perf... values at the top of the manuscript

Run this AFTER collecting REAL data (see RUNBOOK_pathB.md). It reuses the existing
run_all_evaluations.py functions and adds (a) CatBoost (the deployed model in the
paper) and (b) the two reviewer-requested experiments (fraction-matched MIDH,
heuristic S-sweep).

USAGE
-----
    python run_pathB.py --csv dataset/real_crypto_features.csv

IMPORTANT
---------
* If --csv points at the SYNTHETIC corpus (generate_synthetic_dataset.py output),
  the emitted numbers describe synthetic data and MUST NOT be presented as real
  measurements. Use a REAL collected CSV.
* The dominant temporal features F7/F8 are only meaningful if the collector captured
  multi-snapshot temporal change (see RUNBOOK_pathB.md, step 1 fix). Otherwise F7/F8
  will be ~0 and results will differ markedly from the paper narrative.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from scipy import stats as _stats

REAL_LIBS = {"OpenSSL", "PyCryptodome", "pyaes", "PyNaCl", "Windows CNG"}
FEATURE_GROUPS = {"Statistical": [0, 1], "Structural": [2, 3],
                  "Contextual": [4, 5], "Temporal": [6, 7], "Derived": [8, 9]}

CLASSES = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]
# XGBoost requires integer class labels; encode here and decode predictions back to
# strings so all downstream string-based logic (per-class, KEY recall, MIDH) is unchanged.
LE = LabelEncoder().fit(CLASSES)
FEATURES = [
    "F1_entropy", "F2_chi_square", "F3_length", "F4_standard_key_len",
    "F5_standard_iv_len", "F6_memory_region", "F7_change_count",
    "F8_change_pattern", "F9_entropy_length_interaction", "F10_high_confidence_key",
]


def build_models(seed=42):
    models = {}
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=5, eval_metric="mlogloss",
            random_state=seed, verbosity=0)
    except Exception as e:
        print(f"[warn] XGBoost unavailable: {e}")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=1000, depth=9, learning_rate=0.05, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=seed, verbose=False)
    except Exception as e:
        print(f"[warn] CatBoost unavailable (paper's deployed model): {e}")
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.1,
            random_state=seed, verbose=-1)
    except Exception as e:
        print(f"[warn] LightGBM unavailable: {e}")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1)
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), max_iter=500, early_stopping=True,
        validation_fraction=0.2, random_state=seed)
    return models


def fit_predict(model, Xtr, ytr, Xte):
    """Train on integer-encoded labels (required by XGBoost), return string predictions."""
    ytr_int = LE.transform(np.asarray(ytr).astype(str))
    model.fit(Xtr, ytr_int)
    pred_int = np.asarray(model.predict(Xte)).ravel().astype(int)
    return LE.inverse_transform(pred_int)


def scores(y_true, y_pred):
    return {
        "W_F1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "M_F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
    }


def _wf1_cv(Xm, y, seed, model_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc = []
    for tr, te in skf.split(Xm, y):
        m = build_models(seed)[model_name]
        pred = fit_predict(m, Xm[tr], y[tr], Xm[te])
        sc.append(f1_score(y[te], pred, average="weighted", zero_division=0) * 100)
    return float(np.mean(sc)), float(np.std(sc, ddof=1))


def compute_baselines(Xtr, ytr, Xte, yte, seed, model_name):
    """Naive + entropy + heuristic baselines (5-class), single split."""
    out = {}
    for strat, key in [("uniform", "Random"), ("most_frequent", "Majority")]:
        d = DummyClassifier(strategy=strat, random_state=seed)
        d.fit(Xtr, LE.transform(ytr))
        pred = LE.inverse_transform(np.asarray(d.predict(Xte)).astype(int))
        out[key] = scores(yte, pred)
    for cols, key in [([0], "Entropy-only(F1)"), ([0, 2], "Entropy+Length(F1,F3)")]:
        m = build_models(seed)[model_name]
        pred = fit_predict(m, Xtr[:, cols], ytr, Xte[:, cols])
        out[key] = scores(yte, pred)
    try:
        from heuristic_scorer import compute_heuristic_score, heuristic_to_class
        preds = np.array([CLASSES[heuristic_to_class(compute_heuristic_score(r), r)] for r in Xte])
        out["Heuristic-scoring"] = scores(yte, preds)
    except Exception as e:
        out["Heuristic-scoring"] = {"error": str(e)}
    return out


def compute_feature_importance(Xtr, ytr, seed, model_name):
    m = build_models(seed)[model_name]
    m.fit(Xtr, LE.transform(ytr))
    imp = None
    try:
        imp = m.get_feature_importance()          # CatBoost
    except Exception:
        imp = getattr(m, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum() * 100.0
    return {FEATURES[i]: round(float(imp[i]), 2) for i in range(len(FEATURES))}


def compute_ablation(X, y, seed, model_name):
    full_mean, _ = _wf1_cv(X, y, seed, model_name)
    out = {"Full(10)": round(full_mean, 2)}
    for g, idx in FEATURE_GROUPS.items():
        keep = [i for i in range(10) if i not in idx]
        mean, _ = _wf1_cv(X[:, keep], y, seed, model_name)
        out[g] = {"W_F1": round(mean, 2), "delta": round(mean - full_mean, 2)}
    return out


def compute_calibration_tiers(Xtr, ytr, Xte, yte, seed, model_name):
    m = build_models(seed)[model_name]
    m.fit(Xtr, LE.transform(ytr))
    proba = np.asarray(m.predict_proba(Xte))
    pmax = proba.max(axis=1)
    pred = LE.inverse_transform(proba.argmax(axis=1).astype(int))
    correct = (pred == np.asarray(yte)).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        msk = (pmax > bins[b]) & (pmax <= bins[b + 1])
        if msk.sum() > 0:
            ece += abs(correct[msk].mean() - pmax[msk].mean()) * msk.mean()
    tiers = {}
    for name, lo, hi in [("high(>=0.80)", 0.80, 1.01), ("medium(0.50-0.80)", 0.50, 0.80),
                         ("low(<0.50)", -0.01, 0.50)]:
        msk = (pmax >= lo) & (pmax < hi)
        tiers[name] = {"frac": round(float(msk.mean()), 3),
                       "accuracy": round(float(correct[msk].mean() * 100), 1) if msk.sum() else None}
    return {"ECE_pct": round(float(ece * 100), 2), "tiers": tiers}


def compute_learning_curve(X, y, seed, model_name):
    """5-fold CV W-F1 at increasing training fractions, using a STRATIFIED subsample
    of each fold's training indices (fixes the earlier class-skew artifact)."""
    out = {}
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        sc = []
        for tr, te in skf.split(X, y):
            if frac < 1.0:
                sub, _ = train_test_split(tr, train_size=frac, stratify=y[tr],
                                          random_state=seed)
            else:
                sub = tr
            m = build_models(seed)[model_name]
            pred = fit_predict(m, X[sub], y[sub], X[te])
            sc.append(f1_score(y[te], pred, average="weighted", zero_division=0) * 100)
        out[f"{int(frac*100)}pct"] = round(float(np.mean(sc)), 2)
    return out


def compute_3class(X, y, seed, model_name):
    """Re-scope to KEY / IV / OTHER (CIPHERTEXT+PLAINTEXT+NON_CRYPTO -> OTHER), to avoid
    the random-plaintext CIPHERTEXT/PLAINTEXT confound. Reports single-split per-class,
    aggregates, and 5-fold CV W-F1 for the deployed model."""
    remap = {"KEY": "KEY", "IV": "IV", "CIPHERTEXT": "OTHER",
             "PLAINTEXT": "OTHER", "NON_CRYPTO": "OTHER"}
    y3 = np.array([remap[v] for v in y])
    classes3 = ["KEY", "IV", "OTHER"]
    le3 = LabelEncoder().fit(classes3)

    Xtr, Xte, ytr, yte = train_test_split(X, y3, test_size=0.2, stratify=y3, random_state=seed)
    m = build_models(seed)[model_name]
    m.fit(Xtr, le3.transform(ytr))
    pred = le3.inverse_transform(np.asarray(m.predict(Xte)).ravel().astype(int))

    out = {"classes": classes3, "per_class": {}}
    for c in classes3:
        yt = (yte == c).astype(int); yp = (pred == c).astype(int)
        out["per_class"][c] = {
            "precision": round(precision_score(yt, yp, zero_division=0) * 100, 1),
            "recall": round(recall_score(yt, yp, zero_division=0) * 100, 1),
            "f1": round(f1_score(yt, yp, zero_division=0) * 100, 1),
            "n": int((yte == c).sum())}
    out["W_F1"] = round(f1_score(yte, pred, average="weighted", zero_division=0) * 100, 2)
    out["M_F1"] = round(f1_score(yte, pred, average="macro", zero_division=0) * 100, 2)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc = []
    for tr, te in skf.split(X, y3):
        mm = build_models(seed)[model_name]
        mm.fit(X[tr], le3.transform(y3[tr]))
        pr = le3.inverse_transform(np.asarray(mm.predict(X[te])).ravel().astype(int))
        sc.append(f1_score(y3[te], pr, average="weighted", zero_division=0) * 100)
    out["cv5_WF1_mean"] = round(float(np.mean(sc)), 2)
    out["cv5_WF1_sd"] = round(float(np.std(sc, ddof=1)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/real_crypto_features.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default="results_real.json")
    ap.add_argument("--tex", default="results_real.tex")
    ap.add_argument("--deployed", default="XGBoost",
                    help="deployed model whose detailed metrics are reported "
                         "(default XGBoost; the previously-hardcoded choice was CatBoost)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        sys.exit("ERROR: csv has no 'label' column")
    X = df[FEATURES].astype(float).to_numpy() if all(c in df.columns for c in FEATURES) \
        else df.iloc[:, :10].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    library = df["library"].astype(str).to_numpy() if "library" in df.columns else None

    R = {"csv": args.csv, "n_total": int(len(df)),
         "class_counts": {c: int((y == c).sum()) for c in CLASSES}}
    if library is not None:
        R["library_counts"] = {l: int((library == l).sum()) for l in np.unique(library)}

    models = build_models(args.seed)
    print(f"Models available: {list(models)}")

    # ---- single split 60/20/20 ----
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=args.seed)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=args.seed)
    R["single_split"] = {}
    deployed_pred = None
    for name, m in models.items():
        pred = fit_predict(m, Xtr, ytr, Xte)
        R["single_split"][name] = scores(yte, pred)
        if name == args.deployed:
            deployed_pred = pred
    # per-class + confusion for the deployed model (default XGBoost; override with --deployed)
    dep_name = args.deployed if args.deployed in models else ("XGBoost" if "XGBoost" in models else list(models)[0])
    if deployed_pred is None:
        deployed_pred = fit_predict(models[dep_name], Xtr, ytr, Xte)
    R["deployed_model"] = dep_name
    R["per_class"] = {}
    for c in CLASSES:
        yt = (yte == c).astype(int); yp = (deployed_pred == c).astype(int)
        R["per_class"][c] = {
            "precision": round(precision_score(yt, yp, zero_division=0) * 100, 1),
            "recall": round(recall_score(yt, yp, zero_division=0) * 100, 1),
            "f1": round(f1_score(yt, yp, zero_division=0) * 100, 1),
            "n": int((yte == c).sum()),
        }
    R["binary_key"] = {
        "precision": R["per_class"]["KEY"]["precision"],
        "recall": R["per_class"]["KEY"]["recall"],
        "f1": R["per_class"]["KEY"]["f1"],
    }
    cm = confusion_matrix(yte, deployed_pred, labels=CLASSES)
    R["confusion_matrix_rownorm"] = (cm / cm.sum(axis=1, keepdims=True)).round(3).tolist()

    # ---- 5-fold CV ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    R["cv5"] = {name: [] for name in models}
    for tr, te in skf.split(X, y):
        for name, m in build_models(args.seed).items():
            if name not in models:
                continue
            pred = fit_predict(m, X[tr], y[tr], X[te])
            R["cv5"][name].append(f1_score(y[te], pred, average="weighted", zero_division=0) * 100)
    R["cv5_summary"] = {n: {"mean": round(np.mean(v), 2), "sd": round(np.std(v, ddof=1), 2)}
                         for n, v in R["cv5"].items() if v}

    # ---- LOLO ----
    if library is not None:
        R["lolo"] = {}
        libs = list(np.unique(library))
        for name in models:
            per = {}
            for lib in libs:
                te = library == lib; trm = ~te
                m = build_models(args.seed)[name]
                pred = fit_predict(m, X[trm], y[trm], X[te])
                per[lib] = round(f1_score(y[te], pred, average="weighted", zero_division=0) * 100, 1)
            R["lolo"][name] = {"per_library": per,
                                "mean": round(np.mean(list(per.values())), 1),
                                "sd": round(np.std(list(per.values()), ddof=1), 1)}

    # ---- reviewer experiments ----
    try:
        import experiment_fraction_matched_midh as midh
        mids = midh.assign_mids(df)
        tr_c, te_c = midh.mid_disjoint_split(mids, test_mid_frac=0.20, seed=args.seed, train_frac=0.60)
        m = build_models(args.seed)[dep_name]
        pred = fit_predict(m, X[tr_c], y[tr_c], X[te_c])
        R["fraction_matched_midh"] = {
            "model": dep_name, **scores(y[te_c], pred),
            "KEY_recall": round(recall_score((y[te_c] == "KEY").astype(int),
                                              (pred == "KEY").astype(int), zero_division=0) * 100, 1),
            "train_row_frac": round(float(tr_c.mean()), 4)}
    except Exception as e:
        print(f"[warn] fraction-matched MIDH skipped: {e}")

    try:
        import experiment_heuristic_sweep as sweep
        sc = df.apply(sweep.six_signal_score, axis=1).to_numpy()
        isk = (y == "KEY")
        R["heuristic_sweep"] = {}
        for t in (1, 2, 3):
            elim = sc <= t
            R["heuristic_sweep"][f"S<={t}"] = {
                "elimination_pct": round(float(elim.mean()) * 100, 2),
                "key_retention_pct": round(float((isk & ~elim).sum() / max(isk.sum(), 1)) * 100, 2)}
    except Exception as e:
        print(f"[warn] heuristic sweep skipped: {e}")

    # ---- restored tables (baselines, importance, ablation, calibration, curve) ----
    try:
        R["baselines"] = compute_baselines(Xtr, ytr, Xte, yte, args.seed, dep_name)
    except Exception as e:
        print(f"[warn] baselines skipped: {e}")
    try:
        R["feature_importance_pct"] = compute_feature_importance(Xtr, ytr, args.seed, dep_name)
    except Exception as e:
        print(f"[warn] feature importance skipped: {e}")
    try:
        R["ablation_logo"] = compute_ablation(X, y, args.seed, dep_name)
    except Exception as e:
        print(f"[warn] ablation skipped: {e}")
    try:
        R["calibration_tiers"] = compute_calibration_tiers(Xtr, ytr, Xte, yte, args.seed, dep_name)
    except Exception as e:
        print(f"[warn] calibration/tiers skipped: {e}")
    try:
        R["learning_curve"] = compute_learning_curve(X, y, args.seed, dep_name)
    except Exception as e:
        print(f"[warn] learning curve skipped: {e}")
    try:
        R["three_class"] = compute_3class(X, y, args.seed, dep_name)
    except Exception as e:
        print(f"[warn] 3-class skipped: {e}")

    # CV paired t-tests vs deployed model
    if R.get("cv5", {}).get(dep_name):
        base = np.array(R["cv5"][dep_name])
        R["cv5_ttest_vs_deployed"] = {}
        for name, vals in R["cv5"].items():
            if name == dep_name or not vals:
                continue
            t, p = _stats.ttest_rel(np.array(vals), base)
            R["cv5_ttest_vs_deployed"][name] = {
                "delta": round(float(np.mean(vals) - np.mean(base)), 2),
                "t": round(float(t), 3), "p": round(float(p), 4)}

    # LOLO restricted to the five real libraries (exclude memory/synthetic/sim groups)
    for name, d in R.get("lolo", {}).items():
        reals = [v for k, v in d.get("per_library", {}).items() if k in REAL_LIBS]
        if reals:
            d["real_library_mean"] = round(float(np.mean(reals)), 2)
            d["real_library_sd"] = round(float(np.std(reals, ddof=1)), 2)

    # KEY-class FPR + deployment PPV (real, from confusion at single split)
    try:
        cm = np.array(R["confusion_matrix_rownorm"])
        n_by_class = [R["per_class"][c]["n"] for c in CLASSES]
        fp_key = sum(cm[i][0] * n_by_class[i] for i in range(1, 5))   # non-KEY rows -> KEY col
        nonkey_n = sum(n_by_class[1:])
        fpr = fp_key / nonkey_n if nonkey_n else 0.0
        rec = R["per_class"]["KEY"]["recall"] / 100.0
        pi = 1e-4
        ppv = rec * pi / (rec * pi + fpr * (1 - pi)) if (rec * pi + fpr) > 0 else 0.0
        R["deployment_ppv"] = {"key_fpr_pct": round(fpr * 100, 2),
                                "prior": pi, "ppv_pct": round(ppv * 100, 4)}
    except Exception as e:
        print(f"[warn] PPV skipped: {e}")

    # ---- write JSON ----
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)

    # ---- write LaTeX macros (overwrite paper's hard-coded \perf... block) ----
    def pc(name, default="??"):
        return R["single_split"].get(name, {}).get("W_F1", default)
    cb = R["single_split"].get("CatBoost", {})
    xgbm = R["single_split"].get("XGBoost", {})
    lolo_cb = R.get("lolo", {}).get(dep_name, {})
    cv_cb = R["cv5_summary"].get(dep_name, {})
    lines = [
        "%% Auto-generated by run_pathB.py from REAL data -- VERIFY, then \\input this",
        "%% file in place of the hard-coded \\perf... block at the top of the manuscript.",
        f"%% dataset = {args.csv}; n = {R['n_total']}; deployed = {dep_name}",
        f"\\renewcommand{{\\perfCatBoostFone}}{{{cb.get('W_F1','??')}\\%}}",
        f"\\renewcommand{{\\perfXGBoostFone}}{{{xgbm.get('W_F1','??')}\\%}}",
        f"\\renewcommand{{\\perfCatBoostMacFone}}{{{cb.get('M_F1','??')}\\%}}",
        f"\\renewcommand{{\\perfKeyPrecision}}{{{R['binary_key']['precision']}\\%}}",
        f"\\renewcommand{{\\perfKeyRecall}}{{{R['binary_key']['recall']}\\%}}",
        f"\\renewcommand{{\\perfLoloWeightedFone}}{{{lolo_cb.get('mean','??')}\\%}}",
        f"\\renewcommand{{\\perfLoloCatBoost}}{{{lolo_cb.get('mean','??')}\\%}}",
    ]
    if cv_cb and lolo_cb:
        deg = round(cv_cb["mean"] - lolo_cb["mean"], 1)
        lines.append(f"\\renewcommand{{\\perfLoloDegradation}}{{{deg}}}")
    if "fraction_matched_midh" in R:
        fm = R["fraction_matched_midh"]
        lines.append(f"%% fraction-matched MIDH: W-F1={fm['W_F1']}%, M-F1={fm['M_F1']}%, "
                     f"KEY recall={fm['KEY_recall']}%, train_frac={fm['train_row_frac']}")
    if "heuristic_sweep" in R:
        lines.append(f"%% heuristic sweep: {json.dumps(R['heuristic_sweep'])}")
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(R, indent=2))
    print(f"\nWrote {args.json} and {args.tex}")
    print("Next: results_real.json holds the exact numbers for the manuscript, "
          "or \\input results_real.tex.")


if __name__ == "__main__":
    main()
