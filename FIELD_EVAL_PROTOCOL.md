# Field evaluation protocol — adding the "36/39"-style result honestly

The field result (per-module key recovery, Wilson CI, optional funnel reduction) may be
added to the paper **only after** a real run on real modules. This document defines the
experiment so the numbers are defensible, and `field_eval.py` turns the real outputs into
the paper table. **Do not write any field number in the paper until `field_eval.py` has
produced it from real runs.**

---

## 1. What counts (definitions — fix these before running)

- **Module.** One target process running a cryptographic module under a realistic
  exercise (key gen + encrypt/decrypt/sign/HMAC). Pick modules you are authorized to test
  (KCMVP lab modules), ideally including at least one library family **absent from the
  training corpus** to test true generalization (currently OpenSSL, PyCryptodome, pyaes,
  PyNaCl, CNG are in-distribution).
- **Logical key.** A distinct key/CSP the module uses (a master key, a session key, a
  private key…). Decide per module how many logical keys exist *before* scoring.
- **Detected (TP).** A logical key is detected if ≥1 memory block that the read-only
  pipeline labelled KEY matches the key's actual bytes (Section 3 matching rule).
- **False positive (FP).** A predicted-KEY block matching no actual key.
- **Recall** = detected / total logical keys. **Actionable precision** =
  detected / (detected + FP blocks). **Block precision** = matched blocks / predicted blocks.
- **Statistics.** Report overall recall with a 95% **Wilson** interval (small-sample safe);
  `field_eval.py` computes it.

---

## 2. Producing the two inputs per module

### 2a. Predictions (read-only, the method under test)
Run the deployed read-only pipeline against the live module and record every block it
classifies as KEY. This uses your existing components: read-only capture
(`data_collector/win_memory.py`), feature extraction (`feature_extractor.py`), and the
deployed CatBoost model (`models/`). Emit, per predicted KEY block: address, the block
bytes (`value_hex`), region, and confidence.

> Important: predictions must come from the **read-only** path only (no hooking), exactly
> as the method would be deployed. Do not use any ground-truth knowledge here.

### 2b. Ground truth (Phase-6 cross-verification, separate run)
In a **separate** execution, capture the module's actual key bytes by API hooking / DLL
injection (the `Detours-main/` tooling). Intercept the key-bearing calls (e.g.,
`BCryptGenerateSymmetricKey`, `EVP_CipherInit_ex`, key import/generate) and dump the true
key bytes + algorithm. This is the oracle; it never touches the read-only run.

### 2c. Assemble one JSON per module
Write `field_modules/<module>.json` in the schema documented at the top of
`field_eval.py` (`ground_truth_keys[]` from 2b, `predicted_key_blocks[]` from 2a, optional
`funnel{}` with candidate counts per phase if you want the reduction table).

---

## 3. Matching rule (exact, reproducible)
A ground-truth key `g` is detected iff some predicted block `b` satisfies
`hex(g) == hex(b)` OR `hex(g) ⊂ hex(b)` OR `hex(b) ⊂ hex(g)` (containment covers
padding/headers and partial residues). This is implemented in `field_eval._matches`. If
you need a stricter rule (e.g., exact only), change it there and state it in the paper.

---

## 4. Run the scorer
```powershell
cd D:\doc\project\DLL_mem_dump_Analyzer2\ml_pipeline
python field_eval.py --indir field_modules --out field_results.json --tex field_table.tex
```
Outputs:
- `field_results.json` — per-module + overall (recall, Wilson CI, actionable precision).
- `field_table.tex` — paste-ready per-module table (same layout as the paper).

(Optional funnel table: if every module json includes a `funnel{}`, you can summarize
mean candidate counts per phase the same way — only report it if you actually measured it.)

---

## 5. Adding it to the paper (only now)
1. Add a "Practical Evaluation on Field Modules" subsection in Section Results.
2. Insert `field_table.tex` and quote the overall recall + Wilson CI **from the scorer
   output verbatim**.
3. Update the abstract/conclusion to mention the field result, scoped honestly:
   - state it is in-distribution if all modules use trained library families, or highlight
     any unseen-family module as the stronger evidence;
   - keep the standalone-PPV caveat (the field result is the funnel+classifier+analyst
     pipeline, not the classifier alone).
4. Remove the corresponding "not measured" bullet from Limitations.

### What you may NOT do
- Do not hand-edit the numbers, round favorably, or drop unfavorable modules.
- Do not reuse the synthetic "36/39"; whatever the real run yields is what goes in.
- If only a few modules are available, report exactly that N with its (wide) Wilson CI and
  say so — a small honest N is fine; a fabricated large one is not.

---

## 5b. Path B — blind prediction + hooked ground truth (chosen path)

Three helper scripts implement Path B. The predictor never sees keys (blind); ground truth
comes from instrumentation. **Run as Administrator (SeDebugPrivilege).**

Per module, two shells:

Shell 1 (target module = ground truth + keeps keys resident):
```powershell
python field_hook.py --library OpenSSL --module "Module A (OpenSSL)" --reps 5 --hold 120 ^
    --out field_modules/moduleA.groundtruth.json
# prints PID and holds for 120 s
```

Shell 2 (external, read-only, blind prediction against that PID):
```powershell
python field_predict.py --pid <PID-from-shell-1> --module "Module A (OpenSSL)" --library OpenSSL ^
    --snapshots 12 --interval 150 --static-scan --csv dataset/real_crypto_features.csv ^
    --out field_modules/moduleA.predictions.json
```

Merge, repeat for every module, then score:
```powershell
python field_assemble.py --predictions field_modules/moduleA.predictions.json ^
    --groundtruth field_modules/moduleA.groundtruth.json --out field_modules/moduleA.json
python field_eval.py --indir field_modules --out field_results.json --tex field_table.tex
```

Notes / validity:
- The predictor (`field_predict.py`) uses only external `ReadProcessMemory` + the deployed
  model; it has no key knowledge. Ground truth (`field_hook.py`) is recorded by the target
  itself. Same process instance ⇒ identical key values ⇒ value matching is valid.
- Tune `--min-conf`, `--snapshots`, `--static-scan`, and `--cap` and report the settings.
- For the strongest claim, include at least one module on a library family **absent** from
  the training corpus.

### Opaque third-party native modules (Detours)
If the module is a binary you cannot exercise from Python, replace `field_hook.py` with a
Detours-based hook DLL (use `Detours-main/`). Hook the key-bearing entry points and dump
`{id, algo, value_hex, len}` to a `*.groundtruth.json` in the same schema:
- Windows CNG: \code{BCryptGenerateSymmetricKey}, \code{BCryptImportKey}, \code{BCryptImportKeyPair}.
- OpenSSL: \code{EVP\_CipherInit\_ex}/\code{EVP\_EncryptInit\_ex} (key arg), \code{EVP\_PKEY} import/keygen.
- Capture the key buffer at call time; write hex. Then use `field_predict.py` +
  `field_assemble.py` + `field_eval.py` unchanged.
The read-only prediction run stays exactly the same (`field_predict.py --pid`).

## 5c. Path 3 — structure-aware verified detection (recommended for a strong field claim)

`field_detect.py` (+ `csp_verify.py`) replaces the statistical `field_predict.py`: it reports
ONLY key material that satisfies a cryptographic invariant, so false positives are near-zero.
Ground truth is restricted to secret/private keys (public keys are not CSPs).

**Clean stale files first** (the earlier runs left duplicate module JSONs that inflate the
aggregate):
```powershell
Remove-Item field_modules\*.json, field_modules\*.predictions.json, field_modules\*.pid -ErrorAction SilentlyContinue
```

Per module, two ADMIN shells:
```powershell
# Shell 1 -- target + ground truth (secret keys only) + hold
python field_hook.py --library OpenSSL --module "Module A (OpenSSL)" --reps 5 --hold 300 ^
    --out field_modules/moduleA.groundtruth.json
```
```powershell
# Shell 2 -- VERIFIED read-only detection against that PID
python field_detect.py --pidfile field_modules/moduleA.groundtruth.pid ^
    --module "Module A (OpenSSL)" --library OpenSSL --snapshots 3 --interval 200 ^
    --derive-curve25519 --out field_modules/moduleA.predictions.json
```
Merge + score:
```powershell
python field_assemble.py --predictions field_modules/moduleA.predictions.json ^
    --groundtruth field_modules/moduleA.groundtruth.json --out field_modules/moduleA.json
python field_eval.py --indir field_modules --out field_results.json --tex field_table.tex
```

What the verifiers cover (and don't):
- **AES** (any library that keeps the schedule contiguous, e.g., OpenSSL software path): verified
  via key-schedule expansion present in memory.
- **RSA / EC / Ed25519 / X25519 private keys**: verified by `cryptography.load_der_private_key`
  (DER) or by deriving and locating the public key (raw Curve25519/Ed25519 scalars).
- **Not structurally verifiable**: raw stream-cipher session keys (e.g., ChaCha20) with no
  persisted structure, and AES schedules stored as non-contiguous int arrays (e.g., pure-Python
  pyaes). These will be MISSED by the verifier -- report that honestly (it bounds recall), and
  note them as cases requiring the statistical path or library-specific extractors.

Interpretation: expect **precision ~100% (FP ~0)** and a recall that reflects which key types are
structurally verifiable in the tested module. That is a defensible, honest field result.

## 6. Efficiency (tool wall-clock + causal time-saving)
**Tool wall-clock is now measured.** `field_predict.py` records per-module timing with
`time.perf_counter` and writes a `timing` block to each `*.predictions.json`
(`detect_s` = capture + scan + classify = the per-module field-detection time; `total_s`
additionally includes the one-time model load). Report the mean `detect_s` across the six
modules as the tool's measured per-module time (aggregated automatically by
`field_eval.py` / `funnel_table.py`).

**Causal time-saving vs. analysts still needs a controlled study.** The old "≈92% time
reduction" was uncontrolled, and the absolute tool time above is *not* a controlled saving
against human analysts. For a causal claim, run a small **crossover** study (analysts do some
modules manually and some with the tool, counterbalanced, wall-clock recorded, paired test).
Until that study is run, frame efficiency as the tool's absolute measured time, not a
controlled saving over manual analysis.
```
```
