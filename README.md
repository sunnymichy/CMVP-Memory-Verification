# CMVP-Memory-Verification

Reproducibility artifact for the paper
**"Automating Triage for CMVP Key-Management Verification: A Read-Only Live Memory Analysis Approach."**

A read-only, non-invasive memory-monitoring pipeline that triages in-memory cryptographic key
residues for CMVP / FIPS 140-3 module assurance: a multi-stage funnel, a 10-dimensional
gradient-boosting classifier (5-class), and a structure-aware verifier (ASN.1 DER private-key /
Curve25519-Ed25519 / AES key-schedule invariants).

## Dataset

`dataset/real_crypto_features.csv` — **8,112** labeled memory-block samples (real collection) from
five cryptographic libraries (OpenSSL, PyCryptodome, Windows CNG, PyNaCl, pyaes). 13 columns:
the 10 features `F1..F10`, the `label` (KEY / IV / CIPHERTEXT / PLAINTEXT / NON_CRYPTO), and the
`library` / `algorithm` metadata used to derive Material-IDs (MID) for the leakage-controlled
holdout. This single file is all that is needed to reproduce every classifier number below.

## Environment

```
python -m pip install -r requirements.txt      # Python 3.10+, scikit-learn, xgboost, lightgbm,
                                                # catboost, shap, pandas, numpy, scipy
```
All scripts use fixed seeds (`random_state=42`). Run them from this directory (paths are relative).

## Reproducing the paper

| Paper result | Command | Output |
|---|---|---|
| 5-class baselines, single-split + 5-fold CV, per-class, confusion, deployed XGBoost (Tables 8-10) | `python run_pathB.py` | `results_real.json` |
| Group-wise holdout (79.2%), per-library LOLO breakdown, MID stats, calibration/tier diagnostics, AESKeyFind baseline, contextual-ablation portability check | `python reproduce_extra.py` | `results_real_extra.json` |
| TabNet baseline, tuned (Tables 7, 11) | `python reproduce_tabnet.py` | `results_real_tabnet.json` |
| Material-ID holdout (MIDH, Section 5.2.4) | imported via `experiment_fraction_matched_midh.py` | (in `results_real_extra.json`) |
| Field evaluation funnel + per-module recall (Tables 2, 15) | `python field_eval.py` / `python funnel_table.py` | `field_results.json` |
| Non-intrusiveness (Table 3) | `python perturbation_bench.py` *(needs a live target; Windows)* | `perturbation.tex` |

Pre-computed reference outputs are included at the repository root (`results_real*.json`,
`field_results.json`) and the generated paper-table snippets in `paper_tables/`.

## Layout

```
run_pathB.py reproduce_extra.py reproduce_tabnet.py   classifier + extra numbers
run_group_holdout.py cv_significance.py end_to_end.py experiment_*.py   evaluation
field_predict.py field_hook.py field_eval.py funnel_table.py field_*.py field harness
perturbation_bench.py                                  non-intrusiveness measurement
csp_verify.py                                          structure-aware verifier
feature_extractor.py model_trainer.py                  feature vector + training
make_shap.py learning_curve_analysis.py pr_roc_dump.py fp_diagnose.py   diagnostics
data_collector/        ReadProcessMemory-based collection (Windows) + crypto ops
models/                trained classifiers (deployed: xgb_classifier.pkl) + label encoder
field_modules/         six live-module field-run records (config / ground truth / predictions)
dataset/               real_crypto_features.csv (8,112 samples)
paper_tables/          generated .tex table snippets used in the manuscript
results_real*.json field_results.json                  reference result outputs
FIELD_EVAL_PROTOCOL.md RUNBOOK_pathB.md                 evaluation protocol & runbook
```

The field harness (`field_*`, `data_collector/win_memory.py`) and `perturbation_bench.py` require
Windows with `SeDebugPrivilege` and live target processes; the classifier/benchmark scripts run on
any platform from the released CSV alone.
