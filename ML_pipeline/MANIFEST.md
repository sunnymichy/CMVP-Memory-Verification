# Upload manifest

This folder is the curated, submission-ready subset of `ml_pipeline/`. Directory structure
is preserved so that the scripts' relative imports keep working (do **not** flatten).

## STATUS (what is already in `upload/`)
Placed by the assistant (classifier-reproduction core + scaffolding):
`README.md`, `requirements.txt`, `.gitignore`, `MANIFEST.md`,
`run_pathB.py`, `feature_extractor.py`, `heuristic_scorer.py`,
`experiment_fraction_matched_midh.py`, `experiment_heuristic_sweep.py`.

**=> You can already reproduce every classifier number** once you drop
`dataset/real_crypto_features.csv` in (see below):
```
python run_pathB.py --csv dataset/real_crypto_features.csv --deployed XGBoost
```

Still to copy from `ml_pipeline/` (Windows-only field/collection + figure/analysis scripts;
plain file copies, no edits needed) -- listed under "Included" below and flagged [COPY].

## Included (reproduction-essential)

**Classifier (any OS)**
- `run_pathB.py`, `feature_extractor.py`, `heuristic_scorer.py`,
  `experiment_fraction_matched_midh.py`, `experiment_heuristic_sweep.py`

**Structure verifier + field study (Windows)**
- `csp_verify.py`, `field_hook.py`, `field_predict.py`, `field_detect.py`,
  `field_assemble.py`, `field_eval.py`, `fp_diagnose.py`, `funnel_table.py`, `end_to_end.py`,
  `run_field_all.ps1`, `run_funnel_all.ps1`

**Efficiency / figures / baselines**
- `bench_throughput.py`, `perturbation_bench.py`, `make_shap.py`, `pr_roc_dump.py`,
  `cv_significance.py`, `membert_compare.py`, `MEMBERT_BENCHMARK.md`

**Collection (Windows)**
- `data_collector/temporal_capture.py`, `data_collector/crypto_ops.py`, `data_collector/win_memory.py`

**Data**
- `dataset/real_crypto_features.csv` (8,112-block corpus)

## Two large files to copy in manually
To avoid any risk of truncation, copy these two **directly** from `ml_pipeline/` (drag-and-drop):
1. `ml_pipeline/dataset/real_crypto_features.csv`  ->  `upload/dataset/real_crypto_features.csv`
2. `ml_pipeline/data_collector/crypto_ops.py`      ->  `upload/data_collector/crypto_ops.py`

## Excluded (legacy / superseded -- do NOT upload)
These belong to an earlier pipeline and are not used by the paper:
- `generate_synthetic_dataset.py`  (synthetic generator; the paper uses the **real** corpus)
- `run_all_evaluations.py`, `run_full_evaluation.py`, `run_experiment.py`, `run_group_holdout.py`
- `model_trainer.py`, `evaluate.py`, `hybrid_ensemble.py`, `learning_curve_analysis.py`
- `data_collector/collect_real_data.py`  (superseded by `temporal_capture.py`)
