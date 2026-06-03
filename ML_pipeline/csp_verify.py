"""
csp_verify.py  --  structure-aware CSP verification (Path 3)
============================================================
Confirms that candidate bytes are REAL key material by checking cryptographic /
structural invariants, not statistics. This yields near-zero false positives:

  T1a  AES key  : expand the candidate key schedule and confirm the expansion is
                  materialized in the same memory region (aeskeyfind principle).
  T1b  DER priv : locate ASN.1 DER and load it with `cryptography` -- success means the
                  bytes are a mathematically valid RSA/EC/Ed25519/X25519 private key.
  T2   X25519/  : for a 32-byte scalar, derive its public key and confirm the public key
       Ed25519    appears in the same region (context confirmation).

Random/structured-but-non-key data (text, DER public keys, pointers, ciphertext, RSA
modulus fragments) fails all three, so it is not reported. Generic stream-cipher session
keys with no persisted structure are NOT structurally verifiable and are intentionally
left to the (separately reported, lower-confidence) statistical path.
"""

import struct

# ---------------------------------------------------------------- AES S-box / Rcon
_SBOX = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16")
_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D]


def aes_key_expansion(key):
    """Standard Rijndael key expansion -> full expanded schedule bytes (16*(Nr+1))."""
    Nk = len(key) // 4
    Nr = {4: 10, 6: 12, 8: 14}[Nk]
    w = [list(key[4 * i:4 * i + 4]) for i in range(Nk)]
    for i in range(Nk, 4 * (Nr + 1)):
        t = list(w[i - 1])
        if i % Nk == 0:
            t = t[1:] + t[:1]                      # RotWord
            t = [_SBOX[b] for b in t]              # SubWord
            t[0] ^= _RCON[i // Nk - 1]
        elif Nk > 6 and i % Nk == 4:
            t = [_SBOX[b] for b in t]
        w.append([w[i - Nk][j] ^ t[j] for j in range(4)])
    return bytes(b for word in w for b in word)


def _keyish(block):
    """Cheap gate before attempting expensive expansion: near-random, not text/pointer."""
    n = len(block)
    if n == 0 or block.count(0) / n > 0.12 or block.count(0xFF) / n > 0.12:
        return False
    if sum(1 for b in block if 32 <= b < 127) / n > 0.30:
        return False
    return len(set(block)) / min(n, 256) >= 0.80


def word_swap(b):
    """Reverse the byte order within each 32-bit word. Many implementations (incl. OpenSSL
    on little-endian) store the AES key schedule as little-endian 32-bit words; running the
    matcher on word_swap(region) recovers those schedules, and the extracted key comes back
    in natural byte order (so it matches the raw key)."""
    n = len(b) - (len(b) % 4)
    return b"".join(b[i:i + 4][::-1] for i in range(0, n, 4))


def verify_aes_keys(region, base_addr=0):
    """Find a CONTIGUOUS, materialized AES key schedule at any 4-byte offset, using an
    INCREMENTAL match with early exit: for each offset we derive one schedule word at a
    time and compare it to memory, bailing at the first mismatch. A non-key offset is
    rejected after ~1 derived word, so the scan is fast; a full match (the whole 176/208/
    240-byte schedule reproduced exactly) is astronomically unlikely unless it is a real
    key. Returns [(addr, key_size_bits, key_bytes)].
    NOTE: this finds schedules stored contiguously in standard word layout (e.g., software
    AES). Libraries that store the schedule as non-contiguous int arrays, byte-swapped, or
    only in AES-NI registers are not matched -- a documented limitation, not a false miss."""
    found = []
    n = len(region)
    for ks in (16, 24, 32):
        Nk = ks // 4
        Nr = {4: 10, 6: 12, 8: 14}[Nk]
        tw = 4 * (Nr + 1)                 # total 32-bit words
        needed = 4 * tw                   # total schedule bytes
        off = 0
        while off + needed <= n:
            w = [region[off + 4 * i: off + 4 * i + 4] for i in range(Nk)]
            ok = True
            for i in range(Nk, tw):
                prev = w[i - 1]
                if i % Nk == 0:
                    t = bytes(_SBOX[b] for b in prev[1:] + prev[:1])
                    t = bytes([t[0] ^ _RCON[i // Nk - 1]]) + t[1:]
                elif Nk > 6 and i % Nk == 4:
                    t = bytes(_SBOX[b] for b in prev)
                else:
                    t = prev
                base = w[i - Nk]
                wi = bytes((base[j] ^ t[j]) for j in range(4))
                if region[off + 4 * i: off + 4 * i + 4] != wi:
                    ok = False
                    break
                w.append(wi)
            if ok:
                found.append((base_addr + off, ks * 8, bytes(region[off:off + ks])))
                off += needed             # skip past the confirmed schedule
            else:
                off += 4
    return found


def find_der_private_keys(region, base_addr=0):
    """Scan for ASN.1 DER private keys and validate them with `cryptography`
    (load succeeds only for mathematically valid RSA/EC/Ed25519/X25519 private keys).
    Returns [(addr, der_bytes, key_type)]."""
    try:
        from cryptography.hazmat.primitives.serialization import load_der_private_key
    except Exception:
        return []
    found, n, i = [], len(region), 0
    while True:
        i = region.find(b"\x30", i)               # ASN.1 SEQUENCE tag
        if i < 0 or i >= n - 4:
            break
        ln_byte = region[i + 1]
        try:
            if ln_byte == 0x82 and i + 4 <= n:     # 2-byte length
                total = 4 + struct.unpack(">H", region[i + 2:i + 4])[0]
            elif ln_byte == 0x81 and i + 3 <= n:   # 1-byte length
                total = 3 + region[i + 2]
            elif ln_byte < 0x80:                    # short form
                total = 2 + ln_byte
            else:
                i += 1
                continue
        except Exception:
            i += 1
            continue
        if 16 <= total <= 8192 and i + total <= n:
            cand = bytes(region[i:i + total])
            try:
                k = load_der_private_key(cand, password=None)
                found.append((base_addr + i, cand, type(k).__name__))
            except Exception:
                pass
        i += 1
    return found


def verify_curve25519_scalars(region, base_addr=0, cap=20000):
    """For 32-byte candidates, derive X25519/Ed25519 public keys and confirm the public
    key appears in the same region (context confirmation). Returns [(addr, scalar, type)]."""
    try:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization as _s
        raw = _s.Encoding.Raw
        pub_fmt = _s.PublicFormat.Raw
    except Exception:
        return []
    found, n, off, tried = [], len(region), 0, 0
    while off + 32 <= n and tried < cap:
        sc = region[off:off + 32]
        if _keyish(sc):
            tried += 1
            for ctor, label in ((X25519PrivateKey, "X25519"), (Ed25519PrivateKey, "Ed25519")):
                try:
                    pub = ctor.from_private_bytes(sc).public_key().public_bytes(raw, pub_fmt)
                    if pub in region:
                        found.append((base_addr + off, sc, label))
                        break
                except Exception:
                    pass
        off += 4
    return found


if __name__ == "__main__":
    import os
    # self-test: a real AES-128 key + its expansion is detected; random 16B is not.
    k = os.urandom(16)
    blob = b"\x00" * 64 + k + aes_key_expansion(k) + b"\x11" * 64
    hits = verify_aes_keys(blob)
    print("AES self-test:", "PASS" if any(h[2] == k for h in hits) else "FAIL", hits[:1])
    rnd = os.urandom(16) + os.urandom(176)        # key not followed by its real expansion
    print("random (should be empty):", verify_aes_keys(rnd))
