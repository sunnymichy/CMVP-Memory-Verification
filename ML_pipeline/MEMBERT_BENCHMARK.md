# Fair quantitative comparison vs MemBERT — protocol

A like-for-like accuracy comparison with MemBERT~[Fellicious 2025] is only meaningful on a
**common task**, because the two methods differ in input and objective (MemBERT: a
transformer over **raw memory bytes** specialized for **OpenSSH** key extraction; ours:
read-only feature/structure-based **multi-class triage + verified detection**). We therefore
reduce both to **block-level binary KEY detection** on the **same labeled raw-byte blocks**
and report precision/recall/F1. `membert_compare.py` runs this; numbers go into the paper
only after a real run.

## What is compared (common task)
Given a raw memory block, decide KEY vs non-KEY. Methods:
- **Structure verifier** — per-block DER private-key validation (RSA/EC/Ed25519/X25519).
  (AES/raw-Curve25519 need region context, so they are out of scope for a *lone* block —
  state this explicitly; it bounds the verifier's recall on this task.)
- **GBDT classifier** — the deployed model on features from the block. A lone block has no
  temporal/region context, so $F_6$/$F_7$/$F_8$ take defaults; report this as
  "feature-limited" (it is a lower bound for the classifier).
- **MemBERT** — `membert_predict(raw_bytes) -> P(KEY)`; threshold 0.5.

## Step 1 — labeled block set
`membert_compare.py` builds it from the field artifacts:
- **positives** = ground-truth secret keys (`*.groundtruth.json`);
- **hard negatives** = system-flagged blocks that match no ground-truth key
  (`*.predictions.json`), i.e. the confusable high-entropy windows.
Disclose this construction in the paper (negatives are *hard* by design). For a broader
negative pool, add random non-key memory blocks (e.g., extend `field_predict.py` to also
dump a labelled sample of scanned windows).

## Step 2 — wire MemBERT (the work you must do)
1. Obtain MemBERT (published weights/code). If only OpenSSH-trained weights exist, either
   (a) use them as-is (zero-shot) and report it as zero-shot, or (b) **re-train/fine-tune on
   the same positives/negatives** for a matched-supervision comparison (the fairer option).
2. Implement `membert_predict(raw_bytes)` in `membert_compare.py` to return P(KEY).
   - MemBERT consumes byte sequences; pad/truncate each block to its expected input length.
   - If MemBERT only scans whole regions, wrap a single block as a minimal region.
3. Re-run:
   ```
   python membert_compare.py --indir field_modules --csv dataset/real_crypto_features.csv
   ```
   The MemBERT row now appears in `membert_compare.tex`.

## Step 3 — fairness caveats to state in the paper
- Common-task reduction (block-level binary) is favorable to neither method by construction;
  state the reduction explicitly.
- If MemBERT is used zero-shot (OpenSSH-trained) it is disadvantaged; if re-trained on our
  blocks it gets matched supervision. Report which, and prefer the re-trained setting.
- The verifier's recall on this task is bounded to DER-loadable asymmetric keys (no region
  context for AES/raw scalars) — not a defect of the full system, only of the per-block
  reduction.

## Step 4 — integrate
Paste `membert_compare.tex` (Table~`tab:membert_quant`) into Related Work / Discussion next
to the qualitative Table~`tab:membert`, and report all three rows with the fairness caveats.
Do NOT add a MemBERT number that was not produced by an actual run.
