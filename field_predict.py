"""
field_predict.py  --  BLIND read-only KEY-block prediction on an EXTERNAL process (Path B)
==========================================================================================
Attaches to a running target process by PID, takes N read-only snapshots (NtSuspendProcess
not required for reads; we read live), reconstructs Phase 3-4 candidate blocks by
differential change detection (+ a capped static high-entropy scan), classifies each
candidate with the deployed model, and writes the blocks predicted KEY. NO key knowledge
and NO hooking are used here -- this is the method exactly as deployed.

Pair with field_hook.py (ground truth, SEPARATE run) and field_assemble.py.

!!! This is real Windows process-memory code. Run as Administrator (SeDebugPrivilege).
    It requires a live target process and cannot run in a sandbox; validate output before use. !!!

USAGE
-----
    python field_predict.py --pid 1234 --module "Module A (OpenSSL)" --library OpenSSL \
        --snapshots 12 --interval 150 --csv dataset/real_crypto_features.csv \
        --out field_modules/moduleA.predictions.json
"""
import argparse, ctypes, json, math, os, sys, time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collector"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_memory as wm
from feature_extractor import (extract_features, FEATURE_NAMES,
                               STANDARD_KEY_LENGTHS, STANDARD_IV_LENGTHS,
                               compute_shannon_entropy)

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STD_LENGTHS = sorted(set(STANDARD_KEY_LENGTHS) | set(STANDARD_IV_LENGTHS))
CLASSES = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]


def enable_se_debug():
    """Enable SeDebugPrivilege for this process (needs an elevated/admin shell)."""
    from ctypes import wintypes, POINTER, c_void_p, byref, get_last_error

    adv = ctypes.WinDLL("advapi32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, POINTER(wintypes.HANDLE)]
    adv.OpenProcessToken.restype = wintypes.BOOL
    adv.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, POINTER(LUID)]
    adv.LookupPrivilegeValueW.restype = wintypes.BOOL
    adv.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL,
                                          POINTER(TOKEN_PRIVILEGES), wintypes.DWORD,
                                          c_void_p, c_void_p]
    adv.AdjustTokenPrivileges.restype = wintypes.BOOL
    wm.kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY, SE_PRIV_ENABLED = 0x0020, 0x0008, 0x0002
    hTok = wintypes.HANDLE()
    if not adv.OpenProcessToken(wm.kernel32.GetCurrentProcess(),
                                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, byref(hTok)):
        return False
    luid = LUID()
    if not adv.LookupPrivilegeValueW(None, "SeDebugPrivilege", byref(luid)):
        return False
    tp = TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    tp.Privileges[0].Luid = luid
    tp.Privileges[0].Attributes = SE_PRIV_ENABLED
    adv.AdjustTokenPrivileges(hTok, False, byref(tp), 0, None, None)
    return get_last_error() == 0


def open_external(pid):
    try:
        enable_se_debug()
    except Exception as e:
        print(f"[warn] could not enable SeDebugPrivilege ({e}); "
              f"continuing -- running as Administrator may still suffice.")
    for access in (PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
                   PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION):
        h = wm.kernel32.OpenProcess(access, False, pid)
        if h:
            return h
    err = ctypes.get_last_error()
    raise SystemExit(
        f"OpenProcess failed for pid {pid} (WinError {err}). Checklist: "
        f"(1) run THIS shell as Administrator; "
        f"(2) field_hook.py must still be running and holding that PID (check it didn't exit); "
        f"(3) use the SAME Python bitness as the target (64-bit target needs 64-bit Python); "
        f"(4) WinError 5 = access denied (elevation/SeDebugPrivilege), 87 = no such PID, "
        f"299 = partial read across bitness.")


def load_deployed_model(csv):
    """Train the deployed model from the released corpus so the field predictor uses the
    same model family as the paper (CatBoost; XGBoost fallback)."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(csv)
    X = df[FEATURE_NAMES].astype(float).values
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(df["label"].astype(str).values)
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=1000, depth=9, learning_rate=0.05,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
    except Exception:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=42, verbosity=0)
    m.fit(X, y)
    return m, le


def snapshot(h, max_region=8 * 1024 * 1024):
    snap = {}
    for r in wm.enumerate_regions(h):
        if r.region_type in (1, 2, 3) and 0 < r.size <= max_region:
            d = wm.read_memory(h, r.base, r.size)
            if d is not None:
                snap[(r.base, r.region_type)] = d
    return snap


def changed_runs(series, tau=8):
    """Offsets that change across snapshots, merged into runs with gap<=tau."""
    n = min(len(x) for x in series)
    changed = bytearray(n)
    for j in range(1, len(series)):
        a, b = series[j - 1], series[j]
        for k in range(n):
            if a[k] != b[k]:
                changed[k] = 1
    runs, i = [], 0
    while i < n:
        if changed[i]:
            s = i
            while i < n and changed[i]:
                i += 1
            e = i
            if runs and s - runs[-1][1] <= tau:
                runs[-1] = (runs[-1][0], e)
            else:
                runs.append((s, e))
        else:
            i += 1
    return runs, n


def looks_random_key(block):
    """Byte-level Phase-5a pre-filter: keep only blocks that look like random key
    material, rejecting the dominant field false positives (printable text / DER-with-text,
    pointer & heap-metadata blocks, and any non-random/repetitive data). Operates on the
    real bytes, which the feature-vector-only pipeline could not."""
    n = len(block)
    if n == 0:
        return False
    printable = sum(1 for b in block if 32 <= b < 127)
    if printable / n > 0.30:          # text, paths, DER strings, identifiers
        return False
    if block.count(0) / n > 0.12:     # pointers / heap metadata (zero-heavy)
        return False
    if block.count(0xFF) / n > 0.12:  # pointer high bytes (..ff7f0000..)
        return False
    if len(set(block)) / min(n, 256) < 0.88:   # repetitive / non-uniform
        return False
    if compute_shannon_entropy(block) < 0.90 * math.log2(n):  # below near-maximal randomness
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--pidfile", default=None, help="read target PID from this file (written by field_hook.py)")
    ap.add_argument("--module", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--snapshots", type=int, default=12)
    ap.add_argument("--interval", type=int, default=150)
    ap.add_argument("--csv", default="dataset/real_crypto_features.csv")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--static-scan", action="store_true",
                    help="also scan static (unchanging) high-entropy std-length blocks (master keys)")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="disable the byte-level Phase-5a pre-filter (NOT recommended; floods FPs)")
    ap.add_argument("--cap", type=int, default=300000, help="max candidate blocks (safety)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pid = a.pid
    if pid is None and a.pidfile:
        pid = int(open(a.pidfile, encoding="utf-8").read().strip())
    if pid is None:
        raise SystemExit("Provide --pid or --pidfile (from field_hook.py).")
    t0 = time.perf_counter()
    h = open_external(pid)
    t_open = time.perf_counter()
    model, le = load_deployed_model(a.csv)
    t_model = time.perf_counter()
    key_idx = list(le.classes_).index("KEY")

    print(f"capturing {a.snapshots} snapshots @ {a.interval} ms ...")
    snaps = []
    for _ in range(a.snapshots):
        snaps.append(snapshot(h))
        time.sleep(a.interval / 1000.0)
    common = set.intersection(*[set(s) for s in snaps]) if snaps else set()
    t_capture = time.perf_counter()

    feats_list, meta = [], []
    seen = set()

    prefilter = not a.no_prefilter
    stats = {"std_len_candidates": 0}   # end-to-end funnel counters

    def add_candidate(base, off, rtype, block, changes):
        if len(block) not in STD_LENGTHS:
            return
        stats["std_len_candidates"] += 1
        if prefilter and not looks_random_key(block):
            return
        addr = base + off
        if (addr, len(block)) in seen:
            return
        seen.add((addr, len(block)))
        feats_list.append(extract_features(block, rtype, changes, a.snapshots))
        meta.append((addr, rtype, block))

    for (base, rtype) in common:
        if len(feats_list) >= a.cap:
            break
        series = [s[(base, rtype)] for s in snaps]
        runs, n = changed_runs(series)
        last = series[-1]
        # diff-driven candidates
        for (s, e) in runs:
            for L in STD_LENGTHS:
                if e - s < L:
                    continue
                for off in range(s - (s % 8), e - L + 1, 8):
                    if off < 0:
                        continue
                    vals = [x[off:off + L] for x in series]
                    changes = sum(1 for j in range(1, len(vals)) if vals[j] != vals[j - 1])
                    add_candidate(base, off, rtype, last[off:off + L], changes)
                    if len(feats_list) >= a.cap:
                        break
        # optional static scan (master keys don't change, so the diff path misses them).
        # Use a LENGTH-AWARE entropy gate: an N-byte block can reach at most log2(N)
        # bits/byte, so a fixed 7.0 threshold wrongly rejects all short keys (a 32-byte
        # key tops out near 5.0). We keep blocks whose entropy is near-maximal for their
        # size (>= 0.85*log2(L)) and let the model decide.
        if a.static_scan and rtype in (1, 2):
            for L in (16, 24, 32, 48, 64):
                thr = 0.85 * math.log2(L)
                for off in range(0, n - L + 1, 8):
                    block = last[off:off + L]
                    if compute_shannon_entropy(block) >= thr:
                        add_candidate(base, off, rtype, block, 0)
                    if len(feats_list) >= a.cap:
                        break

    t_scan = time.perf_counter()
    preds = []
    if feats_list:
        proba = model.predict_proba(np.array(feats_list, dtype=float))
        cls_dist = Counter(le.classes_[int(np.argmax(p))] for p in proba)
        print("candidate predicted-class distribution:", dict(cls_dist))
        print(f"max KEY-probability over candidates: {max(float(p[key_idx]) for p in proba):.3f}")
        for i, p in enumerate(proba):
            if int(np.argmax(p)) == key_idx and float(np.max(p)) >= a.min_conf:
                addr, rtype, block = meta[i]
                preds.append({"addr": hex(addr), "value_hex": block.hex(),
                              "region": int(rtype), "confidence": round(float(p[key_idx]), 3)})

    t_classify = time.perf_counter()
    funnel = {
        "regions_scanned": len(common),
        "std_length_candidates": stats["std_len_candidates"],
        "prefilter_passed": len(feats_list),
        "ml_key_predictions": len(preds),
    }
    # Wall-clock timing (perf_counter). detect_s = capture+scan+classify is the
    # per-module field-detection time; total_s additionally includes the one-time
    # model load/train (amortizable across modules).
    timing = {
        "open_s": round(t_open - t0, 3),
        "model_load_s": round(t_model - t_open, 3),
        "capture_s": round(t_capture - t_model, 3),
        "scan_s": round(t_scan - t_capture, 3),
        "classify_s": round(t_classify - t_scan, 3),
        "detect_s": round(t_classify - t_capture, 3),
        "total_s": round(t_classify - t0, 3),
        "snapshots": a.snapshots,
        "interval_ms": a.interval,
    }
    out = {"module": a.module, "library": a.library, "predicted_key_blocks": preds,
           "funnel": funnel, "timing": timing}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"{len(feats_list)} candidates scanned; {len(preds)} predicted KEY -> {a.out}")
    print(f"end-to-end funnel: regions {funnel['regions_scanned']} -> "
          f"std-length candidates {funnel['std_length_candidates']} -> "
          f"pre-filter-passed {funnel['prefilter_passed']} -> ML-KEY {funnel['ml_key_predictions']}")
    print(f"timing (s): model_load {timing['model_load_s']}, capture {timing['capture_s']}, "
          f"scan {timing['scan_s']}, classify {timing['classify_s']} | "
          f"detect (capture+scan+classify) {timing['detect_s']}, total {timing['total_s']}")


if __name__ == "__main__":
    main()
