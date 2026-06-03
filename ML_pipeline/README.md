# Read-Only Memory-Triage and Structure-Aware Verification for Cryptographic Key Detection in CMVP Evaluation

Reproducibility artifact for the IEEE TDSC submission. This repository reproduces every
**classifier** number from the released corpus and documents the **field**, **throughput**,
and **non-intrusiveness** measurements (which require the original Windows environment).

> **Platform.** The classifier evaluation runs on any OS. The field study and the
> non-intrusiveness benchmark use Windows-only APIs (`ReadProcessMemory`,
> `NtSuspendProcess`, Windows CNG); they were measured on a Windows x86-64 PC
> (12th-gen Intel Core i9 @ 3.2 GHz, 64 GB RAM).

---

## 1. Install

```bash
python -m venv .venv && . .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 2. Reproduce the classifier results (any OS)

The 8,112-block feature corpus is in `dataset/real_crypto_features.csv`. One command
regenerates every classifier table/figure number (5-fold CV, LOLO, fraction-matched MIDH,
calibration, ablation, learning curve, deployment-prior PPV):

```bash
python run_pathB.py --csv dataset/real_crypto_features.csv --deployed XGBoost
# -> results_real.json  (machine-readable, every paper number)
# -> results_real.tex   (LaTeX macros)
```

Supplementary analyses on the same corpus:

```bash
python cv_significance.py --csv dataset/real_crypto_features.csv --seeds 10   # model-comparison t-tests
python pr_roc_dump.py     --csv dataset/real_crypto_features.csv --prior 1e-4 # KEY PR/ROC (Fig.)
python make_shap.py       --csv dataset/real_crypto_features.csv --out .      # SHAP figures
```

## 3. Reproduce the field study (Windows only)

Ground truth is captured by an instrumented target (`field_hook.py`); the blind detector
attaches read-only. Run from an **Administrator** PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File run_field_all.ps1     # per-module detection -> field_modules/
powershell -ExecutionPolicy Bypass -File run_funnel_all.ps1    # candidate-reduction funnel
python field_eval.py   --indir field_modules                   # Table: per-module recall (Wilson CI)
python funnel_table.py --indir field_modules                   # Table: funnel reduction
python end_to_end.py   --indir field_modules                   # Table: path-separated end-to-end
python fp_diagnose.py  --indir field_modules                   # false-positive identity check
```

## 4. Efficiency and non-intrusiveness (Windows only)

```bash
python bench_throughput.py --size-mb 256 --stride 8 --verify                 # scan/verify throughput
python perturbation_bench.py --role driver --reps 5 --duration 20 --tex perturbation.tex
#   -> 0 output mismatches (data preserved) + bounded, tunable scheduling cost
```

## 5. (Optional) Re-collect the corpus from live operations (Windows only)

```bash
python data_collector/temporal_capture.py    # writes dataset/real_crypto_features.csv
```

## 6. Baseline positioning (MemBERT / SmartKex / PointerKex)

`membert_compare.py` builds a common-task block-level benchmark; see `MEMBERT_BENCHMARK.md`
for the fairness protocol. A quantitative head-to-head is not run because those methods
assume Linux/glibc/OpenSSH (see the paper's Related Work).

---

## Repository layout

```
run_pathB.py                     # MAIN: regenerates all classifier numbers
feature_extractor.py             # 10-dim feature vector (F1..F10)
heuristic_scorer.py              # rule-based pre-score baseline
experiment_fraction_matched_midh.py  # value-level-leakage holdout (MIDH)
experiment_heuristic_sweep.py    # heuristic elimination sweep
csp_verify.py                    # structure verifier (AES schedule, DER, Curve25519)
field_hook.py                    # instrumented target (ground truth)
field_predict.py                 # statistical blind detector (classifier path)
field_detect.py                  # structure-aware blind detector (verifier path)
field_assemble.py field_eval.py fp_diagnose.py funnel_table.py end_to_end.py  # field scorers/tables
bench_throughput.py              # scan/verify throughput
perturbation_bench.py            # read-only non-intrusiveness (data-state + scheduling cost)
make_shap.py                     # SHAP figures (deployed XGBoost)
pr_roc_dump.py cv_significance.py# PR/ROC + model-comparison significance
membert_compare.py MEMBERT_BENCHMARK.md  # baseline-comparison harness + protocol
run_field_all.ps1 run_funnel_all.ps1     # field batch runners
data_collector/
    temporal_capture.py          # multi-snapshot temporal collection (F7/F8 measured)
    crypto_ops.py                # per-library key/IV/plaintext generation
    win_memory.py                # read-only Windows memory access helpers
dataset/
    real_crypto_features.csv     # released 8,112-block feature corpus
```

Raw process memory is **not** redistributed (confidentiality); the field/collection stages
are fully documented and require the original Windows environment.

## License / use
Intended for use within accredited Cryptographic and Security Testing Laboratories under
KCMVP(Korea CMVP) governance. See the paper's Ethical Considerations.
