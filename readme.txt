# CSP-Monitor: CMVP Key Management Automated Testing Tool

This program is an automated testing tool for Cryptographic Security Parameter (CSP) management for CMVP (Cryptographic Module Validation Program). It combines non-intrusive memory analysis with machine learning (CatBoost) to detect and classify cryptographic key residues in memory in real-time.

---

## 1. Environment Setup

### 1.1 Prerequisites
*   **OS**: Windows 10/11 (64-bit)
*   **Compiler**: Visual Studio 2019 or higher (for compiling the C++ monitor)
*   **Python**: 3.10.x or higher
*   **Privileges**: Administrator privileges (Required for ReadProcessMemory API usage)

### 1.2 Python Library Installation
Navigate to the `ml_pipeline` directory and install the required packages.
```bash
cd ml_pipeline
pip install -r requirements.txt
```
*Key packages: catboost, xgboost, scikit-learn, shap, pandas, numpy*

---

## 2. Build Monitor Module

1.  Open the `DLL_mem_dump_Analyzer2.sln` file in Visual Studio.
2.  Set the build configuration to `Release / x64`.
3.  Perform a `Solution Build (Ctrl+Shift+B)` to generate the following files:
    *   `Monitor_DLL.dll`: For target process injection and memory capture (optional, for result verification)
    *   `DLL_mem_dump_Analyzer2.exe`: Standalone real-time memory monitoring program

---

## 3. Machine Learning Pipeline Usage

### 3.1 Step 1: Generate Training Dataset
Generate 10,000 synthetic data entries based on the distribution described in the paper.
```bash
python ml_pipeline/generate_synthetic_dataset.py
```
*   Output: `dataset/real_crypto_features_ml.csv`

### 3.2 Step 2: Model Training and Evaluation
Train the CatBoost model and verify performance metrics (F1-score, Precision, etc.).
```bash
python ml_pipeline/run_full_evaluation.py
```
*   This process includes SHAP-based feature contribution analysis and Learning Curve analysis.

---

## 4. Live Verification

### 4.1 Snapshot Capture and Feature Extraction
Run the target cryptographic module process, then collect memory snapshots via the monitor.
```bash
# Capture snapshots for a specific process (PID) at 200ms intervals
DLL_mem_dump_Analyzer2.exe --pid <Target_PID> --interval 200 --snapshots 256
```

### 4.2 Crypto Key Detection and Classification (Inference)
Input the collected memory block features into the trained model to identify CSPs.
```bash
python ml_pipeline/run_experiment.py --input captured_memory.csv
```

---

## 5. Result Interpretation

### 5.1 Confidence-based Tiered Triage
Classification results are categorized into three tiers based on probability:
*   **High (>= 0.8)**: Automated detection complete. Very high probability of being a cryptographic key.
*   **Medium (0.5 ~ 0.8)**: Requires analyst review. Refer to SHAP explanations.
*   **Low (< 0.5)**: Can be ignored or requires detailed manual analysis.

### 5.2 Evidence Verification via SHAP Explanations
If you need to understand why a specific block was classified as a 'KEY', check the generated `shap_waterfall.png` file to identify which features ($F_1 \sim F_{10}$) were decisive.

---

## 6. Important Notes

1.  **Administrator Privileges**: Since it uses `ReadProcessMemory` and `NtSuspendProcess`, it must be executed from a **terminal running with administrator privileges**.
2.  **Data Integrity**: This program only reads the target process's memory and does not modify it, making it safe for use in CMVP operational environments.
3.  **Performance Optimization**: When analyzing processes with large memory footprints, ensure that Phase 1 area filtering is enabled (Default: Enabled).
