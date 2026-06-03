"""
heuristic_scorer.py
Rule-based pre-score (0-100) over the 10-dim feature vector, used as a baseline and as
the rule path of the hybrid ensemble (paper Sec. III). Six weighted signals.
"""

import numpy as np

CLASS_KEY, CLASS_IV, CLASS_CIPHERTEXT, CLASS_PLAINTEXT, CLASS_NON_CRYPTO = 0, 1, 2, 3, 4
CLASS_NAMES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def compute_heuristic_score(features: np.ndarray) -> float:
    """Heuristic key-likelihood score (0-100) from a 10-dim feature vector."""
    entropy, chi2, length = features[0], features[1], features[2]
    is_key_len, is_iv_len, region, pattern = features[3], features[4], features[5], features[7]
    score = 0.0
    # entropy (<=30)
    if entropy >= 7.81: score += 30
    elif entropy >= 7.5: score += 25
    elif entropy >= 7.0: score += 15
    elif entropy >= 6.5: score += 10
    elif entropy >= 5.0: score += 5
    # chi-square uniformity (<=25; valid only for >=256 bytes -> chi2>0)
    if chi2 > 0:
        if chi2 < 350: score += 25
        elif chi2 < 400: score += 15
        else: score += 5
    # length constraint (<=15)
    if is_key_len == 1.0: score += 15
    elif is_iv_len == 1.0: score += 12
    elif length >= 16: score += 5
    # memory region (<=10)
    region_int = int(region)
    if region_int == 1: score += 10
    elif region_int == 2: score += 7
    else: score += 3
    # synergy bonus (+20): high entropy AND standard key length
    if entropy >= 7.5 and is_key_len == 1.0: score += 20
    # temporal pattern (<=10)
    score += {3: 10, 2: 8, 0: 6, 1: 4}.get(int(pattern), 0)
    return min(score, 100.0)


def heuristic_to_class(score: float, features: np.ndarray) -> int:
    """Map (score, features) to a class index (rule path)."""
    entropy, is_key_len, is_iv_len = features[0], features[3], features[4]
    if score >= 75 and is_key_len == 1.0:
        return CLASS_KEY
    elif score >= 70 and is_iv_len == 1.0:
        return CLASS_IV
    elif score >= 60 and entropy >= 7.0:
        return CLASS_CIPHERTEXT
    elif entropy < 5.0:
        return CLASS_PLAINTEXT
    else:
        return CLASS_NON_CRYPTO


if __name__ == '__main__':
    tests = [
        ("AES-256 key", np.array([7.9, 250.0, 32.0, 1.0, 0.0, 1.0, 18.0, 3.0, 39.5, 1.0])),
        ("AES-CBC IV", np.array([7.6, 0.0, 16.0, 1.0, 1.0, 2.0, 0.0, 0.0, 30.4, 1.0])),
        ("ciphertext", np.array([7.8, 260.0, 256.0, 1.0, 0.0, 2.0, 5.0, 1.0, 62.4, 1.0])),
        ("plaintext", np.array([4.2, 800.0, 64.0, 1.0, 0.0, 2.0, 0.0, 0.0, 17.6, 0.0])),
        ("non-crypto", np.array([3.1, 1200.0, 100.0, 0.0, 0.0, 3.0, 12.0, 2.0, 10.4, 0.0])),
    ]
    for name, feat in tests:
        s = compute_heuristic_score(feat)
        print(f"{name:<14s} score={s:5.1f} -> {CLASS_NAMES[heuristic_to_class(s, feat)]}")
