"""
feature_extractor.py
10-dimensional feature vector [F1..F10] for cryptographic-block classification
(paper Sec. III: statistical, structural, contextual, and derived cues).
"""

import numpy as np
from math import log2

# Standard key/parameter lengths (bytes) used by F4.
STANDARD_KEY_LENGTHS = {
    7, 8, 10, 16, 20, 24, 28, 32, 33, 40, 48, 49, 56, 64, 65, 66, 67, 97,
    128, 133, 192, 256, 384, 512, 768, 1024, 2048, 3072,
}
STANDARD_IV_LENGTHS = {8, 12, 16, 24}  # 24: XSalsa20/XChaCha20 nonce (PyNaCl)


def compute_shannon_entropy(data: bytes) -> float:
    """F1: Shannon entropy (bits/byte). H(X) = -sum p(i) log2 p(i)."""
    n = len(data)
    if n == 0:
        return 0.0
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = byte_counts / n
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def compute_chi_square(data: bytes) -> float:
    """F2: chi-square uniformity; 0.0 below 256 bytes (statistic unreliable)."""
    n = len(data)
    if n < 256:
        return 0.0
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    expected = np.full(256, n / 256.0)
    return float(np.sum((byte_counts - expected) ** 2 / expected))


def classify_change_pattern(change_count: int, total_snapshots: int) -> int:
    """F8: temporal pattern from change ratio (snapshot-count-invariant).
       STATIC(0)<=10%, PARTIAL(1)10-40%, FREQUENT(2)40-70%, ALWAYS(3)>70%."""
    if total_snapshots <= 1:
        return 0
    freq = change_count / (total_snapshots - 1)
    if freq <= 0.1:
        return 0
    elif freq <= 0.4:
        return 1
    elif freq <= 0.7:
        return 2
    else:
        return 3


def extract_features(data: bytes, memory_region: int, change_count: int,
                     total_snapshots: int) -> np.ndarray:
    """Return [F1..F10] for a memory block.
       memory_region: 0=Unknown, 1=DLL data, 2=Stack/Heap, 3=Other."""
    n = len(data)
    entropy = compute_shannon_entropy(data)
    chi2 = compute_chi_square(data)
    length = float(n)
    is_standard_key_len = 1.0 if n in STANDARD_KEY_LENGTHS else 0.0
    is_standard_iv_len = 1.0 if n in STANDARD_IV_LENGTHS else 0.0
    region = float(memory_region)
    changes = float(change_count)
    pattern = float(classify_change_pattern(change_count, total_snapshots))
    interaction = entropy * log2(length + 1)
    high_confidence = 1.0 if (entropy >= 7.5 and is_standard_key_len == 1.0) else 0.0
    return np.array([entropy, chi2, length, is_standard_key_len, is_standard_iv_len,
                     region, changes, pattern, interaction, high_confidence], dtype=np.float64)


FEATURE_NAMES = [
    'F1_entropy', 'F2_chi_square', 'F3_length', 'F4_standard_key_len',
    'F5_standard_iv_len', 'F6_memory_region', 'F7_change_count',
    'F8_change_pattern', 'F9_entropy_length_interaction', 'F10_high_confidence_key',
]


if __name__ == '__main__':
    import os
    for label, data, reg, cc in [("AES-256 key (32B)", os.urandom(32), 1, 2),
                                 ("plaintext", b"Hello, plain text for testing!!", 2, 0),
                                 ("IV (16B)", os.urandom(16), 1, 15)]:
        print(f"=== {label} ===")
        for name, val in zip(FEATURE_NAMES, extract_features(data, reg, cc, 20)):
            print(f"  {name:>35s} = {val:.4f}")
