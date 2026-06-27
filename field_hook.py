"""
field_hook.py  --  ground-truth capture for a target module (Path B, Phase 6)
=============================================================================
Runs a target library as a live "module", records the ACTUAL key bytes it uses (the
instrumented oracle), writes them as ground truth, and then HOLDS the process alive with
the keys resident so that field_predict.py (a SEPARATE, external, read-only run) can
snapshot the same process. Because the predictor never sees these keys and reads memory
externally, the prediction remains blind; this file only supplies ground truth.

For the five libraries exercised by our harness we obtain ground truth directly (the
harness generates the keys). For an opaque third-party native module you cannot modify,
replace this step with Detours API hooking (see FIELD_EVAL_PROTOCOL.md, Path B).

USAGE (run as the target process)
---------------------------------
    python field_hook.py --library OpenSSL --module "Module A (OpenSSL)" --reps 5 \
        --hold 90 --out field_modules/moduleA.groundtruth.json
    # note the printed PID, then in another shell:
    #   python field_predict.py --pid <PID> --module "Module A (OpenSSL)" --library OpenSSL ...
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collector"))
import crypto_ops as co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True,
                    help="OpenSSL | PyCryptodome | pyaes | PyNaCl | Windows CNG")
    ap.add_argument("--module", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--hold", type=int, default=90, help="seconds to keep keys resident")
    ap.add_argument("--include-public", action="store_true",
                    help="also count public keys as ground truth (default: secret/private keys "
                         "only, since public keys are not CSPs / zeroization targets)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    collectors = [(n, fn) for (n, fn) in co.ALL_COLLECTORS if n.startswith(a.library)]
    if not collectors:
        raise SystemExit(f"No collectors for library '{a.library}'. "
                         f"Options: OpenSSL, PyCryptodome, pyaes, PyNaCl, Windows CNG.")

    gt, seen = [], set()
    for _ in range(a.reps):
        for name, fn in collectors:
            try:
                for s in fn():
                    if s.label != "KEY":
                        continue
                    desc = (s.description or "").lower()
                    is_public = any(t in desc for t in ("public", "verify", "peer"))
                    if is_public and not a.include_public:
                        continue   # public keys are not CSPs
                    vh = s.data.hex()
                    if vh in seen:
                        continue
                    seen.add(vh)
                    gt.append({"id": f"k{len(gt)+1}", "algo": s.algorithm,
                               "value_hex": vh, "len": len(s.data),
                               "kind": "public" if is_public else "secret"})
            except Exception as e:
                print(f"  [skip {name}] {e}")

    json.dump({"module": a.module, "library": a.library, "ground_truth_keys": gt},
              open(a.out, "w", encoding="utf-8"), indent=2)
    pidfile = os.path.splitext(a.out)[0] + ".pid"
    with open(pidfile, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    print(f"ground truth: {len(gt)} keys -> {a.out}")
    print(f"PID = {os.getpid()}  (written to {pidfile}; keys held resident for {a.hold}s)")
    print(f"NOW run, in another ADMIN shell, WHILE this is still holding:")
    print(f"  python field_predict.py --pidfile {pidfile} --module \"{a.module}\" "
          f"--library {a.library} --static-scan --out {a.out.replace('groundtruth','predictions')}")
    # keep the process (and its key buffers, pinned by crypto_ops) alive
    time.sleep(a.hold)
    print("hold elapsed; exiting -- the PID is now invalid (re-run field_hook before predicting).")


if __name__ == "__main__":
    main()
