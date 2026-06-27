# Path B — Real-data collection & paper regeneration (runbook)

Goal: replace the synthetic benchmark with a **real** one collected from live crypto
libraries, regenerate every paper number, and incorporate the real numbers into the
manuscript. **All collection steps run on the Windows host** (Windows CNG, `ReadProcessMemory`,
native crypto libs); the resulting numbers are then integrated into the manuscript.

---

## 0. Prerequisites (one-time)

```powershell
# from an ADMIN PowerShell (SeDebugPrivilege / memory access)
cd D:\doc\project\DLL_mem_dump_Analyzer2\ml_pipeline
python -m pip install cryptography pycryptodome pynacl pyaes ^
    xgboost catboost lightgbm scikit-learn pandas numpy scipy shap
```
- Windows CNG (`bcrypt.dll`) ships with Windows; no install needed.
- Run everything in the **same** Python process/bitness you trained with (x64).

---

## 1. Temporal features (F7/F8) — RESOLVED via option 1a

Your dominant features (F7 change-count, F8 change-pattern; ~40% importance) were
**zeroed for library samples** in `collect_real_data.py` (single snapshot). This is fixed
by the new `data_collector/temporal_capture.py`, which runs real crypto operations at
**fixed addresses across N records** and measures the true change counts. Use it instead
of `collect_real_data.py` for collection. No target distribution is imposed; F7/F8 are
whatever the measured lifecycle produces (master keys STATIC, session keys
PARTIAL/FREQUENT, IV/ciphertext ALWAYS).

---

## 2. Collect real data (option 1a)

```powershell
cd D:\doc\project\DLL_mem_dump_Analyzer2\ml_pipeline\data_collector
# --reps controls dataset size; --records = snapshots per session; --interval ms
python temporal_capture.py --reps 40 --records 12 --interval 150 ^
    --output ../dataset/real_crypto_features.csv
```
Outputs `real_crypto_features.csv` (+ `_ml.csv`) and prints the F7/F8 distribution.
Symmetric AES (OpenSSL/PyCryptodome/pyaes) gets real temporal dynamics; asymmetric/
stream/MAC/NON_CRYPTO are gathered via the existing collectors and measured (long-lived
keys are legitimately STATIC). Note the **actual** class counts/total — they will NOT
equal the synthetic 10,000 / 3460-1640-…, so the class-distribution and per-library
tables must be regenerated from these.

**VALIDATE the printed distribution before continuing:**
- KEY should span STATIC..ALWAYS (not all STATIC),
- IV and CIPHERTEXT should be mostly ALWAYS.
If KEY is still ~all STATIC, stop and report — buffer reuse may not be observed on your
build. Quick re-check:
```powershell
python -c "import pandas as pd; d=pd.read_csv('../dataset/real_crypto_features.csv'); print(d.groupby('label')['F8_change_pattern'].value_counts())"
```

---

## 3. Regenerate every paper number

```powershell
cd D:\doc\project\DLL_mem_dump_Analyzer2\ml_pipeline
python run_pathB.py --csv dataset/real_crypto_features.csv
```
Produces:
- `results_real.json` — all numbers (single-split, per-class, confusion, 5-fold CV,
  LOLO, fraction-matched MIDH, heuristic sweep).
- `results_real.tex` — `\renewcommand` macros for the manuscript's `\perf...` values.

(You can also run the two experiments standalone:)
```powershell
python experiment_fraction_matched_midh.py --csv dataset/real_crypto_features.csv
python experiment_heuristic_sweep.py       --csv dataset/real_crypto_features.csv
```

---

## 4. Field / practical evaluation (Table: per-module, 36/39 keys)

The 7-module field result is a **separate** real experiment (run the core pipeline on
the 7 KCMVP modules + Phase-6 cross-verification). It is not produced by `run_pathB.py`.
Record per-module keys/detected/missed/FP and recompute recall + Wilson CI. Keep these
as in-distribution (all 7 use trained library families) per the v3 wording.

---

## 5. Integrate into the manuscript

Using `results_real.json` (and per-module field numbers from step 4), update the manuscript:
- replace the hard-coded `\perf...` block and every results table/number in
  `IEEE_TDSC_en_v3.tex` with the real values,
- regenerate the class-distribution, per-library, baseline, per-class, confusion,
  CV/LOLO/MIDH, and ablation tables,
- correct Sec. IV-A wording to describe the actual collection, and the data-availability
  section to release the real corpus.

Or, for a quick compile: replace the `\newcommand{\perf...}` block at the top of the
`.tex` with `\input{results_real.tex}` (this only updates the macro-driven numbers,
not the table bodies).

---

## Notes / known code↔paper mismatches to reconcile while you're here
- **Deployed model:** paper says CatBoost; `run_all_evaluations.py` evaluates XGB/RF/MLP.
  `run_pathB.py` adds CatBoost so the deployed-model numbers are real.
- **Heuristic scorer:** paper Sec. III-E2 = 6 binary signals (S_max=8); repo
  `heuristic_scorer.py` = 0–100 weighted. Pick one and make text+code agree.
- **MID:** on real data, prefer assigning MIDs from the recorded ground-truth key bytes
  (available from the known-key-value collection) rather than entropy clustering.
