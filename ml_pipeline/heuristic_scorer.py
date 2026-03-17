"""
heuristic_scorer.py
Paper Section 3.4.2: Heuristic scoring system (0-100 points).

Assigns weights to 6 signals to compute a cryptographic key likelihood score.
This score is used as (1) a standalone baseline classifier and
(2) the rule-based path in the hybrid ensemble.
"""

import numpy as np

# Class indices
CLASS_KEY = 0
CLASS_IV = 1
CLASS_CIPHERTEXT = 2
CLASS_PLAINTEXT = 3
CLASS_NON_CRYPTO = 4

CLASS_NAMES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def compute_heuristic_score(features: np.ndarray) -> float:
    """
    Computes a heuristic score (0-100) from a 10-dimensional feature vector.

    Score allocation:
      Signal 1 - Entropy:          max 30 points
      Signal 2 - Chi-square:       max 25 points
      Signal 3 - Length constraint: max 15 points
      Signal 4 - Region type:      max 10 points
      Synergy bonus:               20 points
      Signal 5 - Temporal pattern:  max 10 points
      Total cap:                   100 points
    """
    entropy = features[0]       # F1
    chi2 = features[1]          # F2
    length = features[2]        # F3
    is_key_len = features[3]    # F4
    is_iv_len = features[4]     # F5
    region = features[5]        # F6
    pattern = features[7]       # F8

    score = 0.0

    # ── Signal 1: Entropy (max 30 points) ──
    # Confidence-level thresholds based on empirical data
    if entropy >= 7.81:
        score += 30    # 99% confidence
    elif entropy >= 7.5:
        score += 25    # 95% confidence
    elif entropy >= 7.0:
        score += 15
    elif entropy >= 6.5:
        score += 10
    elif entropy >= 5.0:
        score += 5

    # ── Signal 2: Chi-square uniformity (max 25 points) ──
    # chi2 > 0 is valid only when data >= 256 bytes
    if chi2 > 0:
        if chi2 < 350:      # Uniform distribution (threshold < 350)
            score += 25
        elif chi2 < 400:    # Partially uniform
            score += 15
        else:               # Non-uniform
            score += 5

    # ── Signal 3: Length constraint (max 15 points) ──
    if is_key_len == 1.0:
        score += 15          # Matches one of 28 standard key lengths
    elif is_iv_len == 1.0:
        score += 12          # Standard IV length (8/12/16)
    elif length >= 16:
        score += 5

    # ── Signal 4: Memory region type (max 10 points) ──
    region_int = int(region)
    if region_int == 1:      # DLL Data Section
        score += 10
    elif region_int == 2:    # Stack/Heap
        score += 7
    else:                    # Unknown(0) or Other(3)
        score += 3

    # ── Synergy bonus (20 points) ──
    # High entropy + standard key length simultaneously satisfied
    if entropy >= 7.5 and is_key_len == 1.0:
        score += 20

    # ── Signal 5: Temporal change pattern (max 10 points) ──
    pattern_int = int(pattern)
    pattern_scores = {
        3: 10,   # ALWAYS  → Short-lived session key characteristic
        2: 8,    # FREQUENT
        0: 6,    # STATIC  → Long-term master key characteristic
        1: 4,    # PARTIAL
    }
    score += pattern_scores.get(pattern_int, 0)

    return min(score, 100.0)


def heuristic_to_class(score: float, features: np.ndarray) -> int:
    """
    Determines the class from the heuristic score and features.
    Paper Section 3.4.5: Path 3 rules of the hybrid ensemble.

    Returns:
        Class index (0=KEY, 1=IV, 2=CIPHERTEXT, 3=PLAINTEXT, 4=NON_CRYPTO)
    """
    entropy = features[0]
    is_key_len = features[3]
    is_iv_len = features[4]

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


def check_short_block_threshold(features: np.ndarray,
                                score: float,
                                sensitivity: str = 'normal') -> bool:
    """
    Enhanced decision criteria for 8-15 byte short blocks.

    Paper: 8-byte blocks have a higher probability of exhibiting
    high entropy by chance due to small sample size, so stricter
    decision criteria are applied.

    Args:
        features: 10-dimensional feature vector
        score: Heuristic score
        sensitivity: 'high' / 'normal' / 'low'
    Returns:
        True if the block is accepted as a cryptographic data candidate
    """
    length = int(features[2])
    entropy = features[0]

    if length >= 16:
        # Normal block: apply standard thresholds
        thresholds = {'high': 40, 'normal': 60, 'low': 75}
        return score >= thresholds.get(sensitivity, 60)

    # Short block (8-15 bytes): apply stricter criteria
    short_thresholds = {'high': 55, 'normal': 70, 'low': 85}
    min_entropy = 6.5 if sensitivity == 'high' else 7.0

    return (score >= short_thresholds.get(sensitivity, 70)
            and entropy >= min_entropy)


# ─── Test when run standalone ───
if __name__ == '__main__':
    # Test: various scenarios
    test_cases = [
        ("AES-256 key (DLL, always changing)",
         np.array([7.9, 250.0, 32.0, 1.0, 0.0, 1.0, 18.0, 3.0, 39.5, 1.0])),
        ("AES-CBC IV (heap, static)",
         np.array([7.6, 0.0, 16.0, 1.0, 1.0, 2.0, 0.0, 0.0, 30.4, 1.0])),
        ("Ciphertext (256B, heap)",
         np.array([7.8, 260.0, 256.0, 1.0, 0.0, 2.0, 5.0, 1.0, 62.4, 1.0])),
        ("Plaintext (heap)",
         np.array([4.2, 800.0, 64.0, 1.0, 0.0, 2.0, 0.0, 0.0, 17.6, 0.0])),
        ("Non-crypto data",
         np.array([3.1, 1200.0, 100.0, 0.0, 0.0, 3.0, 12.0, 2.0, 10.4, 0.0])),
    ]

    for name, feat in test_cases:
        score = compute_heuristic_score(feat)
        cls = heuristic_to_class(score, feat)
        print(f"{name:<30s} → Score: {score:5.1f} → Class: {CLASS_NAMES[cls]}")
