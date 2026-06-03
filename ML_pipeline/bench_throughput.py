"""
bench_throughput.py -- measure blind-scan + structure-verify throughput.
=========================================================================
Reports the *real* wall-clock throughput of the dominant linear scan (the byte-level
pre-filter that every candidate window pays) and, optionally, the per-candidate DER
verification cost. It then projects per-module scan time using the candidate counts the
field run actually produced (Table II / funnel_table.tex). No number is invented: run this
on the field workstation and paste the printed MB/s and per-module seconds into the
manuscript placeholders in Section "Computational complexity and throughput".

USAGE
-----
    python bench_throughput.py                 # 256 MB synthetic buffer, stride 8
    python bench_throughput.py --size-mb 512 --stride 8 --verify
    python bench_throughput.py --dump some_region.bin    # time over a real saved region

The synthetic buffer is high-entropy random bytes, i.e. the *worst case* for the
pre-filter (the largest fraction of windows pass and reach the classifier/verifier), so the
reported MB/s is a conservative lower bound for real heaps.
"""
import argparse, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collector"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Real candidate workload per module (sums match funnel_table.tex / Table II).
MODULE_CANDIDATES = {
    "A (OpenSSL)": 12_504_684, "B (PyCryptodome)": 10_737_935, "C (pyaes)": 10_313_383,
    "D (Windows CNG)": 10_300_070, "E (PyNaCl)": 10_440_614, "F (PurePy)": 11_134_661,
}

# ---- pre-filter: prefer the deployed implementation; fall back to a faithful re-impl ----
try:
    from field_predict import looks_random_key as _looks_random_key   # deployed gate
    _SRC = "field_predict.looks_random_key"
except Exception:
    _SRC = "builtin replica (field_predict import failed)"
    def _looks_random_key(b):
        n = len(b)
        if n == 0:
            return False
        cnt = [0] * 256
        printable = zeros = ffs = 0
        for x in b:
            cnt[x] += 1
            if 32 <= x < 127:
                printable += 1
            if x == 0:
                zeros += 1
            if x == 0xFF:
                ffs += 1
        if printable / n >= 0.30 or zeros / n >= 0.12 or ffs / n >= 0.12:
            return False
        distinct = sum(1 for c in cnt if c)
        if distinct / n < 0.88:
            return False
        ent = 0.0
        for c in cnt:
            if c:
                p = c / n
                ent -= p * math.log2(p)
        return ent >= 0.90 * math.log2(n)


def _verify_der(raw):
    try:
        from cryptography.hazmat.primitives.serialization import load_der_private_key
        load_der_private_key(bytes(raw), password=None)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mb", type=int, default=256)
    ap.add_argument("--stride", type=int, default=8, help="scan stride in bytes (pointer alignment)")
    ap.add_argument("--win", type=int, default=32, help="candidate window length")
    ap.add_argument("--verify", action="store_true", help="also time DER verification on survivors")
    ap.add_argument("--dump", default=None, help="time over a real saved .bin region instead of synthetic")
    a = ap.parse_args()

    if a.dump:
        buf = bytearray(open(a.dump, "rb").read())
    else:
        buf = bytearray(np.random.randint(0, 256, size=a.size_mb * 1024 * 1024, dtype=np.uint8).tobytes())
    nbytes = len(buf)
    mv = memoryview(buf)

    print(f"pre-filter source : {_SRC}")
    print(f"buffer            : {nbytes/1e6:.1f} MB  stride={a.stride}  win={a.win}")

    t0 = time.perf_counter()
    windows = survivors = 0
    last = nbytes - a.win
    off = 0
    while off <= last:
        if _looks_random_key(mv[off:off + a.win]):
            survivors += 1
        windows += 1
        off += a.stride
    dt = time.perf_counter() - t0

    win_per_s = windows / dt if dt else 0.0
    mb_per_s = (nbytes / 1e6) / dt if dt else 0.0
    print(f"\nscanned {windows:,} windows in {dt:.2f} s")
    print(f"  -> {mb_per_s:,.1f} MB/s")
    print(f"  -> {win_per_s:,.0f} windows/s")
    print(f"  -> pre-filter pass rate {100*survivors/windows:.2f}% (synthetic worst case)")

    if a.verify and survivors:
        t1 = time.perf_counter()
        vchecks = 0
        off = 0
        while off <= last and vchecks < 200_000:
            w = mv[off:off + a.win]
            if _looks_random_key(w):
                _verify_der(w)
                vchecks += 1
            off += a.stride
        vdt = time.perf_counter() - t1
        if vchecks:
            print(f"  DER verify attempt: {1e6*vdt/vchecks:.1f} us/candidate over {vchecks:,} survivors")

    print("\nprojected per-module blind-scan time (real candidate counts, Table II):")
    for m, c in MODULE_CANDIDATES.items():
        secs = c / win_per_s if win_per_s else float("nan")
        print(f"  {m:<18} {c:>12,} windows  ->  {secs:6.1f} s")
    print("\nPaste MB/s and a representative per-module time into the manuscript "
          "placeholders (Section: Computational complexity and throughput).")


if __name__ == "__main__":
    main()
