"""
temporal_capture.py  --  Path-B option 1a: faithful F7/F8 (temporal) collection
================================================================================
PROBLEM THIS FIXES
------------------
collect_real_data.py assigns change_count = 0 to library samples (single snapshot),
so the dominant temporal features F7 (change count) and F8 (change pattern) collapse
to STATIC for almost everything. This module instead measures REAL temporal change:
it runs actual crypto operations at FIXED memory addresses across N records, snapshots
those addresses between records, and counts how often each value changes.

No target distribution is imposed. F7/F8 are whatever the measured lifecycle produces:
  - master key  : written once, reused every record        -> measured STATIC
  - session key : rotated every record (or every 3 records) -> measured FREQUENT/PARTIAL
  - IV / nonce  : fresh every record                        -> measured ALWAYS
  - plaintext   : fresh every record                        -> measured ALWAYS
  - ciphertext  : changes every record                      -> measured ALWAYS

Symmetric AES (CBC/CTR/GCM) over OpenSSL(cryptography), PyCryptodome, and pyaes get
real temporal dynamics here. The remaining algorithms (asymmetric / stream / MAC) and
NON_CRYPTO are gathered via the existing crypto_ops collectors and measured over a real
snapshot pass; long-lived asymmetric keys are legitimately STATIC, which is the truthful
label, not an artifact.

!!! VALIDATE AFTER RUNNING !!!
Print F7/F8 distributions (the runbook gives the one-liner). If KEY is still ~all STATIC
or IV/CIPHERTEXT are not mostly ALWAYS, something is wrong with buffer reuse on your
build -- stop and report before using the numbers.

USAGE
-----
    cd ml_pipeline/data_collector
    python temporal_capture.py --reps 40 --records 12 --interval 150 \
        --output ../dataset/real_crypto_features.csv
"""

import os
import sys
import time
import ctypes
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from win_memory import get_own_process_handle, get_region_type_at, read_memory
from feature_extractor import extract_features, FEATURE_NAMES

_KEEP = []  # prevent GC / address movement


def _pin(nbytes):
    buf = (ctypes.c_ubyte * nbytes)()
    _KEEP.append(buf)
    return ctypes.addressof(buf), buf


def _write(buf, data):
    ctypes.memmove(buf, data, len(data))


import random as _random

# High-DIVERSITY structured plaintext, so PLAINTEXT is (a) separable from high-entropy
# CIPHERTEXT and (b) not memorizable as a handful of fixed strings (which caused the
# value-level leakage seen under fraction-matched MIDH). Each sample draws a random
# template and fills it with random realistic tokens, yielding a near-unique low-entropy
# (~4-4.7 bits/byte) string per record. Keys/IVs are os.urandom (already maximally diverse).
_PT_TEMPLATES = [
    b"GET /api/v2/{}/{} HTTP/1.1\r\nHost: {}.example.org\r\nAuth: {}",
    b"POST /upload/{} HTTP/1.1\r\nContent-Length: {}\r\nUser: {}",
    b"INSERT INTO {}(id,ts,msg) VALUES({},'{}','{}');",
    b"UPDATE {} SET status='{}' WHERE id={} AND owner='{}';",
    b"SELECT {} FROM {} WHERE date='{}' LIMIT {};",
    b"{ \"user\":\"{}\", \"id\":{}, \"role\":\"{}\", \"ts\":\"{}\" }",
    b"{ \"event\":\"{}\", \"path\":\"/var/{}/{}\", \"code\":{} }",
    b"name={}; session={}; record={}; role={}; region={}",
    b"Dear {}, your order {} shipped on {} to {} via {}.",
    b"From: {}@{}.org\r\nTo: {}@example.com\r\nSubject: {}",
    b"2026-{}-{} {}:{}:00 INFO {} processed entry {} ok",
    b"/usr/local/{}/{}/config_{}.yaml owner={} mode=0644",
    b"transaction_id={} account={} amount={} currency={} memo={}",
    b"key=value; {}={}; {}={}; {}={}; padding=........",
    b"<record id=\"{}\"><user>{}</user><action>{}</action></record>",
    b"cn={},ou={},dc={},dc=org uid={} gidNumber={}",
    b"hostname {} role {} datacenter {} rack {} unit {}",
    b"msg \"{} {} completed\" id={} user={} elapsed_ms={}",
]

_WORDS = [
    b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf",
    b"server", b"client", b"payment", b"orders", b"users", b"audit", b"login",
    b"session", b"report", b"invoice", b"backup", b"region", b"tenant",
    b"alice", b"bob", b"carol", b"dave", b"erin", b"frank", b"grace", b"heidi",
    b"active", b"pending", b"closed", b"member", b"admin", b"guest", b"ok", b"fail",
    b"seoul", b"tokyo", b"paris", b"london", b"daejeon", b"busan", b"prod", b"stage",
]


_PT_RNG = _random.Random(0xC0FFEE)  # persistent: every call is distinct, run-reproducible


def _structured_plaintext(nbytes, i=0):
    rng = _PT_RNG

    def tok():
        k = rng.randrange(4)
        if k == 0:
            return rng.choice(_WORDS)
        if k == 1:
            return str(rng.randint(0, 99999999)).encode()
        if k == 2:
            return ("%02d" % rng.randint(1, 28)).encode()
        return rng.choice(_WORDS) + b"_" + str(rng.randint(0, 999)).encode()

    tmpl = _PT_TEMPLATES[rng.randrange(len(_PT_TEMPLATES))]
    out = tmpl
    while b"{}" in out:
        out = out.replace(b"{}", tok(), 1)
    if len(out) < nbytes:
        filler = b" " + b" ".join(rng.choice(_WORDS) for _ in range(12))
        out = out + filler
    return out[:nbytes]


# ───────────────────────── per-library AES encryptors ─────────────────────────
# Each returns ciphertext bytes of a length that is constant for constant pt length.

def _enc_openssl_cbc(key, iv, pt):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    e = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return e.update(pt) + e.finalize()


def _enc_openssl_ctr(key, iv, pt):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    e = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    return e.update(pt) + e.finalize()


def _enc_openssl_gcm(key, iv, pt):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).encrypt(iv, pt, None)


def _enc_pycrypto_cbc(key, iv, pt):
    from Crypto.Cipher import AES
    return AES.new(key, AES.MODE_CBC, iv=iv).encrypt(pt)


def _enc_pycrypto_ctr(key, iv, pt):
    from Crypto.Cipher import AES
    return AES.new(key, AES.MODE_CTR, nonce=iv[:8]).encrypt(pt)


def _enc_pycrypto_gcm(key, iv, pt):
    from Crypto.Cipher import AES
    c = AES.new(key, AES.MODE_GCM, nonce=iv[:12])
    ct, _tag = c.encrypt_and_digest(pt)
    return ct


def _enc_pyaes_cbc(key, iv, pt):
    import pyaes
    aes = pyaes.AESModeOfOperationCBC(key, iv=iv)
    out = b""
    for i in range(0, len(pt), 16):
        out += aes.encrypt(pt[i:i + 16])
    return out


def _enc_pyaes_ctr(key, iv, pt):
    import pyaes
    ctr = pyaes.Counter(initial_value=int.from_bytes(iv, "big"))
    return pyaes.AESModeOfOperationCTR(key, counter=ctr).encrypt(pt)


# (library, algorithm, key_size, iv_size, encryptor)
SYMMETRIC_SPECS = [
    ("OpenSSL", "AES-128-CBC", 16, 16, _enc_openssl_cbc),
    ("OpenSSL", "AES-192-CBC", 24, 16, _enc_openssl_cbc),
    ("OpenSSL", "AES-256-CBC", 32, 16, _enc_openssl_cbc),
    ("OpenSSL", "AES-256-CTR", 32, 16, _enc_openssl_ctr),
    ("OpenSSL", "AES-256-GCM", 32, 12, _enc_openssl_gcm),
    ("PyCryptodome", "AES-128-CBC", 16, 16, _enc_pycrypto_cbc),
    ("PyCryptodome", "AES-192-CBC", 24, 16, _enc_pycrypto_cbc),
    ("PyCryptodome", "AES-256-CBC", 32, 16, _enc_pycrypto_cbc),
    ("PyCryptodome", "AES-256-CTR", 32, 16, _enc_pycrypto_ctr),
    ("PyCryptodome", "AES-256-GCM", 32, 12, _enc_pycrypto_gcm),
    ("pyaes", "AES-128-CTR", 16, 16, _enc_pyaes_ctr),
    ("pyaes", "AES-192-CTR", 24, 16, _enc_pyaes_ctr),
    ("pyaes", "AES-256-CTR", 32, 16, _enc_pyaes_ctr),
    ("pyaes", "AES-128-CBC", 16, 16, _enc_pyaes_cbc),
    ("pyaes", "AES-192-CBC", 24, 16, _enc_pyaes_cbc),
    ("pyaes", "AES-256-CBC", 32, 16, _enc_pyaes_cbc),
]

PT_SIZE = 64  # multiple of 16 for CBC


def _measure_session(handle, library, algorithm, key_size, iv_size, encrypt,
                     n_records, interval_ms, key_mode, emit_labels):
    """Run a session at fixed addresses; return list of (label, library, algo, features).
    key_mode: 'master' (static) | 'session' (rotate each record) | 'session3' (every 3)."""
    key_addr, key_buf = _pin(key_size)
    iv_addr, iv_buf = _pin(iv_size)
    pt_addr, pt_buf = _pin(PT_SIZE)

    master_key = os.urandom(key_size)
    _write(key_buf, master_key)
    iv0, pt0 = os.urandom(iv_size), _structured_plaintext(PT_SIZE, 0)
    _write(iv_buf, iv0); _write(pt_buf, pt0)
    try:
        ct0 = encrypt(bytes(key_buf[:key_size]), bytes(iv_buf[:iv_size]), bytes(pt_buf[:PT_SIZE]))
    except Exception as e:
        print(f"    [skip] {library} {algorithm}: {e}")
        return []
    ct_size = len(ct0)
    ct_addr, ct_buf = _pin(ct_size)
    _write(ct_buf, ct0)

    addrs = {"KEY": (key_addr, key_size), "IV": (iv_addr, iv_size),
             "PLAINTEXT": (pt_addr, PT_SIZE), "CIPHERTEXT": (ct_addr, ct_size)}
    history = {k: [] for k in addrs}

    for i in range(n_records):
        if i > 0:
            if key_mode == "session":
                _write(key_buf, os.urandom(key_size))
            elif key_mode == "session3" and (i % 3 == 0):
                _write(key_buf, os.urandom(key_size))
            new_iv, new_pt = os.urandom(iv_size), _structured_plaintext(PT_SIZE, i)
            _write(iv_buf, new_iv); _write(pt_buf, new_pt)
            ct = encrypt(bytes(key_buf[:key_size]), bytes(iv_buf[:iv_size]), bytes(pt_buf[:PT_SIZE]))
            _write(ct_buf, ct[:ct_size])
        # snapshot the fixed addresses (read own process memory)
        for label, (addr, sz) in addrs.items():
            history[label].append(read_memory(handle, addr, sz))
        time.sleep(interval_ms / 1000.0)

    rows = []
    for label in emit_labels:
        addr, sz = addrs[label]
        seq = [b for b in history[label] if b is not None]
        changes, prev = 0, None
        for b in seq:
            if prev is not None and b != prev:
                changes += 1
            prev = b
        data = seq[-1] if seq else b"\x00" * sz
        region = get_region_type_at(handle, addr)
        feats = extract_features(data, region, changes, n_records)
        rows.append((label, library, algorithm, feats))
    return rows


def collect_temporal(reps, n_records, interval_ms):
    """Symmetric temporal collection across all SYMMETRIC_SPECS."""
    handle = get_own_process_handle()
    rows = []
    for rep in range(reps):
        for (library, algo, ks, ivs, enc) in SYMMETRIC_SPECS:
            # master-key session: emit KEY(static) + IV/PT/CT(dynamic)
            rows += _measure_session(handle, library, algo, ks, ivs, enc,
                                     n_records, interval_ms, "master",
                                     emit_labels=("KEY", "IV", "PLAINTEXT", "CIPHERTEXT"))
            # session-key sessions: emit KEY only (dynamic), to populate PARTIAL/FREQUENT/ALWAYS
            rows += _measure_session(handle, library, algo, ks, ivs, enc,
                                     n_records, interval_ms, "session3",
                                     emit_labels=("KEY",))
            rows += _measure_session(handle, library, algo, ks, ivs, enc,
                                     n_records, interval_ms, "session",
                                     emit_labels=("KEY",))
        print(f"  temporal rep {rep+1}/{reps} done ({len(rows)} rows so far)")
    return rows


def collect_static_remainder(reps, n_records, boost_mult=1):
    """Asymmetric / stream / MAC / NON_CRYPTO via existing collectors, measured over a
    real snapshot pass (long-lived keys are legitimately STATIC).

    boost_mult: extra repetition factor applied ONLY to the under-represented libraries
    (PyNaCl, Windows CNG) so leave-one-library-out coverage is balanced. Keys are
    os.urandom per call, so oversampling adds genuine value diversity, not duplicates."""
    from crypto_ops import (
        collect_openssl_chacha20, collect_openssl_rsa, collect_openssl_ec,
        collect_openssl_ecdsa_p384, collect_openssl_hmac,
        collect_pycryptodome_des3, collect_pycryptodome_chacha20,
        collect_pycryptodome_salsa20, collect_pycryptodome_rsa,
        collect_pynacl_secretbox, collect_pynacl_sealedbox, collect_pynacl_ed25519,
        collect_pynacl_box, collect_pynacl_generichash,
        collect_cng_aes, collect_non_crypto,
    )
    normal = [
        collect_openssl_chacha20, lambda: collect_openssl_rsa(2048),
        collect_openssl_ec, collect_openssl_ecdsa_p384, collect_openssl_hmac,
        collect_pycryptodome_des3, collect_pycryptodome_chacha20,
        collect_pycryptodome_salsa20, lambda: collect_pycryptodome_rsa(2048),
    ]
    boost = [   # under-represented libraries: PyNaCl + Windows CNG
        collect_pynacl_secretbox, collect_pynacl_sealedbox, collect_pynacl_ed25519,
        collect_pynacl_box, collect_pynacl_generichash,
        lambda: collect_cng_aes(16), lambda: collect_cng_aes(24), lambda: collect_cng_aes(32),
    ]
    handle = get_own_process_handle()
    rows = []
    crypto_patterns = []

    def run_once(fn):
        try:
            for s in fn():
                seq = []
                for _ in range(n_records):
                    seq.append(read_memory(handle, s.address, len(s.data)))
                    time.sleep(0.02)
                changes, prev = 0, None
                for b in [x for x in seq if x is not None]:
                    if prev is not None and b != prev:
                        changes += 1
                    prev = b
                feats = extract_features(s.data, s.region_type, changes, n_records)
                rows.append((s.label, s.library, s.algorithm, feats))
                if s.label in ("KEY", "IV"):
                    crypto_patterns.append(s.data)
        except Exception as e:
            print(f"    [skip remainder] {e}")

    for rep in range(reps):
        for fn in normal:
            run_once(fn)
    for rep in range(reps * max(1, boost_mult)):
        for fn in boost:
            run_once(fn)
    print(f"  remainder collected (boost x{boost_mult} for PyNaCl/CNG); {len(rows)} rows")

    # NON_CRYPTO
    try:
        nc_target = max(50, int(len(rows) * 0.15 / 0.85))
        for s in collect_non_crypto(nc_target, crypto_patterns):
            feats = extract_features(s.data, s.region_type, 0, n_records)
            rows.append((s.label, s.library, s.algorithm, feats))
    except Exception as e:
        print(f"    [skip non_crypto] {e}")
    return rows


def write_csv(rows, out_path):
    recs = []
    for (label, library, algorithm, feats) in rows:
        d = {name: float(feats[i]) for i, name in enumerate(FEATURE_NAMES)}
        d["label"] = label; d["library"] = library; d["algorithm"] = algorithm
        recs.append(d)
    df = pd.DataFrame(recs, columns=FEATURE_NAMES + ["label", "library", "algorithm"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    df[FEATURE_NAMES + ["label"]].to_csv(out_path.replace(".csv", "_ml.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40, help="repetitions per algorithm")
    ap.add_argument("--records", type=int, default=12, help="snapshots/records per session")
    ap.add_argument("--interval", type=int, default=150, help="ms between records")
    ap.add_argument("--output", default="../dataset/real_crypto_features.csv")
    ap.add_argument("--no-remainder", action="store_true",
                    help="skip asymmetric/stream/MAC/NON_CRYPTO (symmetric only)")
    ap.add_argument("--boost", type=int, default=1,
                    help="oversampling factor for PyNaCl/CNG. NOTE: empirically this does NOT "
                         "improve their leave-one-library-out scores (LOLO excludes the held-out "
                         "library from training), so the default is 1 (off); see paper Sec. LOLO.")
    args = ap.parse_args()

    print("=" * 64)
    print(" Temporal (F7/F8) faithful collection  --  REAL data")
    print("=" * 64)

    rows = collect_temporal(args.reps, args.records, args.interval)
    if not args.no_remainder:
        print("  collecting asymmetric/stream/MAC/NON_CRYPTO remainder ...")
        rows += collect_static_remainder(max(1, args.reps // 3), args.records, boost_mult=args.boost)

    df = write_csv(rows, args.output)

    print(f"\n총 표본: {len(df)}")
    print("\n클래스 분포:")
    print(df["label"].value_counts().to_string())
    print("\nF8(시간패턴) 분포  [0=STATIC,1=PARTIAL,2=FREQUENT,3=ALWAYS]:")
    print(df.groupby("label")["F8_change_pattern"].value_counts().to_string())
    print("\nF7(변경횟수) 요약:")
    print(df.groupby("label")["F7_change_count"].describe()[["mean", "min", "max"]].to_string())
    print(f"\nCSV 저장: {args.output}")
    print("VALIDATE: KEY should span STATIC..ALWAYS; IV/CIPHERTEXT should be mostly ALWAYS.")


if __name__ == "__main__":
    main()
