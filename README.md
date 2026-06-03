Read-Only Memory-Triage and Structure-Aware Verification for Cryptographic Key Detection in CMVP Evaluation

This repository contains the Python source code and dataset to reproduce the experiments in our IEEE TDSC paper. It implements a read-only, non-invasive approach to detecting cryptographic key material in process memory for CMVP/FIPS 140-3 evaluation, combining a machine-learning triage classifier with a structure-aware verifier.

The released code reproduces every reported result: the classifier evaluation on the released 8,112-block feature corpus, the blind field study on live module processes, and the throughput and non-intrusiveness measurements. All read-only memory access uses standard Windows APIs via Python (ctypes); the classifier evaluation runs on any OS.

For installation and step-by-step reproduction commands, see the readme.md file in the ml_pipeline folder.
