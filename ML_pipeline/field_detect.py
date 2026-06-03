"""
field_detect.py  --  structure-aware CSP detection on an EXTERNAL process (Path 3)
==================================================================================
Read-only scan of a live process that reports ONLY cryptographically VERIFIED key
material (csp_verify.py): AES keys whose schedule is present, DER private keys that
`cryptography` accepts, and X25519/Ed25519 scalars whose public key is co-resident.
Because every finding satisfies an algebraic/structural invariant, false positives are
near-zero -- this is the high-precision replacement for the statistical field_predict.py.

Output is the same predictions schema field_eval.py consumes, so:
    field_hook.py (ground truth, holding) -> field_detect.py (verified) ->
    field_assemble.py -> field_eval.py

USAGE
-----
    python field_detect.py --pidfile field_modules/moduleA.groundtruth.pid \
        --module "Module A (OpenSSL)" --library OpenSSL --snapshots 3 --interval 200 \
        --derive-curve25519 --out field_modules/moduleA.predictions.json
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_predict as fp          # reuse open_external + snapshot + SeDebugPrivilege
import csp_verify as cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--pidfile", default=None)
    ap.add_argument("--module", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--snapshots", type=int, default=3)
    ap.add_argument("--interval", type=int, default=200)
    ap.add_argument("--aes-scan", action="store_true",
                    help="enable AES key-schedule scan (slower; only finds contiguously-stored "
                         "schedules, e.g. software AES). Off by default.")
    ap.add_argument("--max-aes-region", type=int, default=2 * 1024 * 1024,
                    help="skip AES key-schedule scan for regions larger than this (bytes)")
    ap.add_argument("--derive-curve25519", action="store_true",
                    help="also confirm X25519/Ed25519 scalars by deriving their public key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pid = a.pid
    if pid is None and a.pidfile:
        pid = int(open(a.pidfile, encoding="utf-8").read().strip())
    if pid is None:
        raise SystemExit("Provide --pid or --pidfile (from field_hook.py).")

    h = fp.open_external(pid)
    findings = {}   # (addr, value_hex) -> record

    def add(addr, value_hex, ktype, region):
        key = (addr, value_hex)
        if key not in findings:
            findings[key] = {"addr": hex(addr), "value_hex": value_hex,
                             "region": int(region), "type": ktype,
                             "verified": True, "confidence": 1.0}

    for snap_i in range(a.snapshots):
        snap = fp.snapshot(h)
        regs = list(snap.items())
        print(f"snapshot {snap_i+1}/{a.snapshots}: {len(regs)} readable regions, scanning ...")
        for ri, ((base, rtype), data) in enumerate(regs):
            # T1b: DER private keys (RSA / EC / Ed25519 / X25519) -- fast scan + crypto validation
            for (addr, der, ktype) in cv.find_der_private_keys(data, base):
                add(addr, der.hex(), f"DER:{ktype}", rtype)
            # T1a: AES keys with materialized schedule (opt-in; slower). Two passes:
            # natural byte order, and word-byte-swapped (little-endian 32-bit words, e.g. OpenSSL).
            if a.aes_scan and len(data) <= a.max_aes_region:
                for (addr, bits, key) in cv.verify_aes_keys(data, base):
                    add(addr, key.hex(), f"AES-{bits}", rtype)
                for (addr, bits, key) in cv.verify_aes_keys(cv.word_swap(data), base):
                    add(addr, key.hex(), f"AES-{bits}-le", rtype)
            # T2: X25519/Ed25519 raw scalars confirmed by co-resident public key
            if a.derive_curve25519:
                for (addr, sc, label) in cv.verify_curve25519_scalars(data, base):
                    add(addr, sc.hex(), label, rtype)
            if (ri + 1) % 200 == 0:
                print(f"  ... {ri+1}/{len(regs)} regions; {len(findings)} findings so far")
        print(f"snapshot {snap_i+1} done; {len(findings)} verified findings cumulative")
        if snap_i < a.snapshots - 1:
            time.sleep(a.interval / 1000.0)

    preds = list(findings.values())
    by_type = {}
    for f in preds:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1
    out = {"module": a.module, "library": a.library, "predicted_key_blocks": preds}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"verified CSP findings: {len(preds)} -> {a.out}")
    print("by type:", by_type)
    print("All findings satisfy a cryptographic invariant (near-zero false positives).")


if __name__ == "__main__":
    main()
