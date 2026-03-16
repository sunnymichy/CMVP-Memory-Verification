"""
feature_extractor.py
KCMVP 암호키 분류를 위한 10차원 특징 벡터 추출 모듈.

논문 섹션 3.4.1: 6개 범주(엔트로피, 바이트분포, 길이, 메모리영역, 시간패턴, 교차특징)에서
10차원 특징 벡터 [F1..F10]을 추출한다.
"""

import numpy as np
from math import log2

# ─── 표준 규격 키/IV 길이 (논문 표2, C++ 구현과 동일한 28종) ───
STANDARD_KEY_LENGTHS = {
    7,      # DES (parity bits 제외, 56-bit key = 7 bytes)
    8,      # DES (parity bits 포함)
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
    F1: Shannon 엔트로피 (bits/byte).
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
    F2: 카이제곱 균일성 검정.
    256바이트 미만이면 통계적 유의성 부족으로 0.0 반환.
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
    F8: 시간적 변화 패턴 분류.
    변경 빈도 비율 기반 → 스냅샷 수에 무관한 일관된 분류.
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
    메모리 블록에서 10차원 특징 벡터를 추출한다.

    Args:
        data: 분석 대상 메모리 블록 (8~3072 bytes)
        memory_region: 메모리 영역 유형
            0=Unknown, 1=DLL Data, 2=Stack/Heap, 3=Other
        change_count: 스냅샷 간 해당 주소의 변경 횟수
        total_snapshots: 전체 스냅샷 수

    Returns:
        np.ndarray: [F1, F2, ..., F10] 10차원 float64 배열
    """
    n = len(data)

    # F1: Shannon 엔트로피
    entropy = compute_shannon_entropy(data)

    # F2: 카이제곱 균일성 검정
    chi2 = compute_chi_square(data)

    # F3: 데이터 길이
    length = float(n)

    # F4: 표준 키 길이 일치 여부 (이진)
    is_standard_key_len = 1.0 if n in STANDARD_KEY_LENGTHS else 0.0

    # F5: 표준 IV 길이 일치 여부 (이진)
    is_standard_iv_len = 1.0 if n in STANDARD_IV_LENGTHS else 0.0

    # F6: 메모리 영역 유형 (범주형 0-3)
    region = float(memory_region)

    # F7: 변경 횟수
    changes = float(change_count)

    # F8: 패턴 유형
    pattern = float(classify_change_pattern(change_count, total_snapshots))

    # F9: 엔트로피 x log2(길이+1) 교호작용
    interaction = entropy * log2(length + 1)

    # F10: 고엔트로피(>=7.5) + 표준키길이 동시 충족
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


# ─── 단독 실행 시 테스트 ───
if __name__ == '__main__':
    # 테스트 1: 랜덤 32바이트 (AES-256 키 시뮬레이션)
    import os
    key_data = os.urandom(32)
    feats = extract_features(key_data, memory_region=1, change_count=2, total_snapshots=20)
    print("=== AES-256 키 시뮬레이션 (32B 랜덤) ===")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:>35s} = {val:.4f}")

    # 테스트 2: ASCII 텍스트 (PLAINTEXT 시뮬레이션)
    text_data = b"Hello, this is a plain text message for testing purposes!!"
    feats2 = extract_features(text_data, memory_region=2, change_count=0, total_snapshots=20)
    print("\n=== 평문 시뮬레이션 (ASCII 텍스트) ===")
    for name, val in zip(FEATURE_NAMES, feats2):
        print(f"  {name:>35s} = {val:.4f}")

    # 테스트 3: 16바이트 랜덤 (AES-128 키 / IV 시뮬레이션)
    iv_data = os.urandom(16)
    feats3 = extract_features(iv_data, memory_region=1, change_count=15, total_snapshots=20)
    print("\n=== IV 시뮬레이션 (16B 랜덤) ===")
    for name, val in zip(FEATURE_NAMES, feats3):
        print(f"  {name:>35s} = {val:.4f}")
