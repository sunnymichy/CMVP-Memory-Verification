"""
feature_extractor.py
10-dimensional feature vector extraction module for KCMVP cryptographic key classification.

Paper Section 3.4.1: Extracts a 10-dimensional feature vector [F1..F10] from 6 categories
(entropy, byte distribution, length, memory region, temporal pattern, cross-features).
"""

import numpy as np
from math import log2

# ─── Standard key/IV lengths (Paper Table 2, same 28 types as C++ implementation) ───
STANDARD_KEY_LENGTHS = {
    7,      # DES (excluding parity bits, 56-bit key = 7 bytes)
    8,      # DES (including parity bits)
    10,     # RC2-80
    16,     # AES-128, SEED, ARIA-128
    20,     # ChaCha20-Poly1305
    24,     # AES-192, 3DES
    28,     # EC P-224
    32,     # AES-256, ChaCha20
    33,     # EC P-256 compressed
    40,     # RC5/RC2-extended (5 bytes * 8 = 40 bits variant)
    48,     # HMAC-SHA384
    49,     # EC P-384 compressed
    56,     # 3DES-EDE3 (168-bit = 3 × 56-bit = 3 × 7 bytes, with parity: 3 × 8)
    64,     # BLAKE2b
    65,     # EC P-256 uncompressed
    66,     # EC P-521 compressed (ceil(521/8)+1)
    67,     # EC P-521 compressed variant
    97,     # EC P-384 uncompressed
    128,    # RSA-1024
    133,    # EC P-521 uncompressed
    192,    # RSA-1536
    256,    # RSA-2048
    384,    # RSA-3072
    512,    # RSA-4096
    768,    # RSA-6144
    1024,   # RSA-8192
    2048,   # DH-16384
    3072,   # DH-24576
}

STANDARD_IV_LENGTHS = {8, 12, 16, 24}  # 24: XSalsa20/XChaCha20 nonce (PyNaCl)


def compute_shannon_entropy(data: bytes) -> float:
    """
    F1: Shannon entropy (bits/byte).
    H(X) = -sum(p(i) * log2(p(i))) for i in 0..255
    """
    n = len(data)
    if n == 0:
        return 0.0
    byte_counts = np.bincount(
        np.frombuffer(data, dtype=np.uint8), minlength=256
    )
    probs = byte_counts / n
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def compute_chi_square(data: bytes) -> float:
    """
    F2: Chi-square uniformity test.
    Returns 0.0 for data shorter than 256 bytes due to insufficient statistical significance.
    """
    n = len(data)
    if n < 256:
        return 0.0
    byte_counts = np.bincount(
        np.frombuffer(data, dtype=np.uint8), minlength=256
    ).astype(np.float64)
    expected = np.full(256, n / 256.0)
    return float(np.sum((byte_counts - expected) ** 2 / expected))


def classify_change_pattern(change_count: int, total_snapshots: int) -> int:
    """
    F8: Temporal change pattern classification.
    Based on change frequency ratio, providing consistent classification
    regardless of snapshot count.
      STATIC(0):   <= 10%
      PARTIAL(1):  10-40%
      FREQUENT(2): 40-70%
      ALWAYS(3):   > 70%
    """
    if total_snapshots <= 1:
        return 0
    freq = change_count / (total_snapshots - 1)
    if freq <= 0.1:
        return 0   # STATIC
    elif freq <= 0.4:
        return 1   # PARTIAL
    elif freq <= 0.7:
        return 2   # FREQUENT
    else:
        return 3   # ALWAYS


def extract_features(data: bytes,
                     memory_region: int,
                     change_count: int,
                     total_snapshots: int) -> np.ndarray:
    """
    Extract a 10-dimensional feature vector from a memory block.

    Args:
        data: Target memory block for analysis (8-3072 bytes)
        memory_region: Memory region type
            0=Unknown, 1=DLL Data, 2=Stack/Heap, 3=Other
        change_count: Number of changes at this address across snapshots
        total_snapshots: Total number of snapshots

    Returns:
        np.ndarray: [F1, F2, ..., F10] 10-dimensional float64 array
    """
    n = len(data)

    # F1: Shannon entropy
    entropy = compute_shannon_entropy(data)

    # F2: Chi-square uniformity test
    chi2 = compute_chi_square(data)

    # F3: Data length
    length = float(n)

    # F4: Standard key length match (binary)
    is_standard_key_len = 1.0 if n in STANDARD_KEY_LENGTHS else 0.0

    # F5: Standard IV length match (binary)
    is_standard_iv_len = 1.0 if n in STANDARD_IV_LENGTHS else 0.0

    # F6: Memory region type (categorical 0-3)
    region = float(memory_region)

    # F7: Change count
    changes = float(change_count)

    # F8: Pattern type
    pattern = float(classify_change_pattern(change_count, total_snapshots))

    # F9: Entropy x log2(length+1) interaction term
    interaction = entropy * log2(length + 1)

    # F10: Simultaneous high entropy (>=7.5) and standard key length match
    high_confidence = 1.0 if (entropy >= 7.5 and is_standard_key_len == 1.0) else 0.0

    return np.array([
        entropy,              # F1
        chi2,                 # F2
        length,               # F3
        is_standard_key_len,  # F4
        is_standard_iv_len,   # F5
        region,               # F6
        changes,              # F7
        pattern,              # F8
        interaction,          # F9
        high_confidence,      # F10
    ], dtype=np.float64)


FEATURE_NAMES = [
    'F1_entropy',
    'F2_chi_square',
    'F3_length',
    'F4_standard_key_len',
    'F5_standard_iv_len',
    'F6_memory_region',
    'F7_change_count',
    'F8_change_pattern',
    'F9_entropy_length_interaction',
    'F10_high_confidence_key',
]


# ─── Test when run standalone ───
if __name__ == '__main__':
    # Test 1: Random 32 bytes (AES-256 key simulation)
    import os
    key_data = os.urandom(32)
    feats = extract_features(key_data, memory_region=1, change_count=2, total_snapshots=20)
    print("=== AES-256 Key Simulation (32B random) ===")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:>35s} = {val:.4f}")

    # Test 2: ASCII text (plaintext simulation)
    text_data = b"Hello, this is a plain text message for testing purposes!!"
    feats2 = extract_features(text_data, memory_region=2, change_count=0, total_snapshots=20)
    print("\n=== Plaintext Simulation (ASCII text) ===")
    for name, val in zip(FEATURE_NAMES, feats2):
        print(f"  {name:>35s} = {val:.4f}")

    # Test 3: Random 16 bytes (AES-128 key / IV simulation)
    iv_data = os.urandom(16)
    feats3 = extract_features(iv_data, memory_region=1, change_count=15, total_snapshots=20)
    print("\n=== IV Simulation (16B random) ===")
    for name, val in zip(FEATURE_NAMES, feats3):
        print(f"  {name:>35s} = {val:.4f}")
