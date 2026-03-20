"""
generate_synthetic_dataset.py
합성 데이터셋 생성기.

논문 섹션 3.4.2 기준: 5개 라이브러리(OpenSSL, PyCryptodome, Windows CNG, PyNaCl, pyaes)에서
20종 고유 알고리즘·키 크기 조합(라이브러리별 중복 포함 33건), 총 10,000개 표본.

각 클래스별 통계적 특성과 라이브러리별 메모리 영역(F6) 분포를 차별화하여
현실적인 특징 분포를 모사한다.
"""

import os
import numpy as np
import pandas as pd
from math import log2

from feature_extractor import (
    STANDARD_KEY_LENGTHS,
    STANDARD_IV_LENGTHS,
    FEATURE_NAMES,
)

np.random.seed(42)

# ─── 논문 기준 클래스별 총 표본 수 (합계 10,000) ───
CLASS_COUNTS = {
    'KEY': 3460,
    'IV': 1640,
    'CIPHERTEXT': 1730,
    'PLAINTEXT': 1730,
    'NON_CRYPTO': 1440,
}

# ─── 라이브러리-클래스 분포 행렬 (LOLO 표 기준) ───
# 행합 = 라이브러리 총 표본 수, 열합 = 클래스 총 표본 수
LIBRARY_CLASS_MATRIX = {
    #                   KEY    IV   CIP   PLN   NON   Total
    'OpenSSL':        (960,  425,  480,  500,  425),   # 2,790
    'PyCryptodome':   (855,  385,  435,  425,  365),   # 2,465
    'Windows CNG':    (770,  335,  375,  375,  355),   # 2,210
    'PyNaCl':         (480,  250,  240,  230,  185),   # 1,385
    'pyaes':          (395,  245,  200,  200,  110),   # 1,150
}

CLASS_NAMES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']

# ─── 알고리즘-라이브러리 매핑 (논문 lines 928-934, 33개 조합) ───
LIBRARY_ALGORITHMS = {
    'OpenSSL': [
        'AES-128-CBC', 'AES-192-CBC', 'AES-256-CBC',
        'AES-256-GCM', 'AES-256-CTR', 'ChaCha20-Poly1305',
        'RSA-2048', 'ECDSA-P256', 'ECDSA-P384', 'HMAC-SHA256',
    ],
    'PyCryptodome': [
        'AES-128-CBC', 'AES-192-CBC', 'AES-256-CBC',
        'AES-256-GCM', 'AES-256-CTR', '3DES-CBC',
        'ChaCha20', 'Salsa20', 'RSA-2048',
    ],
    'Windows CNG': [
        'AES-128-CBC', 'AES-192-CBC', 'AES-256-CBC',
    ],
    'PyNaCl': [
        'XSalsa20-Poly1305', 'Curve25519-SealedBox', 'Curve25519-Box',
        'Ed25519', 'BLAKE2b',
    ],
    'pyaes': [
        'AES-128-CTR', 'AES-192-CTR', 'AES-256-CTR',
        'AES-128-CBC', 'AES-192-CBC', 'AES-256-CBC',
    ],
}

# ─── 라이브러리별 F6 메모리 영역 분포 ───
# 0=Unknown, 1=DLL Data, 2=Stack/Heap, 3=Other
# 라이브러리별 메모리 할당 패턴 차이를 반영 (클래스 무관)
LIBRARY_REGION_DIST = {
    'OpenSSL':      ([1, 2, 3], [0.40, 0.35, 0.25]),
    'PyCryptodome': ([1, 2, 3], [0.15, 0.55, 0.30]),
    'Windows CNG':  ([1, 2, 3], [0.30, 0.40, 0.30]),
    'PyNaCl':       ([1, 2, 3], [0.10, 0.55, 0.35]),
    'pyaes':        ([1, 2, 3], [0.08, 0.55, 0.37]),
}

# ─── 알고리즘별 키/IV/암호문 길이 매핑 ───
ALGO_KEY_LENGTHS = {
    'AES-128-CBC': 16, 'AES-128-CTR': 16,
    'AES-192-CBC': 24, 'AES-192-CTR': 24,
    'AES-256-CBC': 32, 'AES-256-GCM': 32, 'AES-256-CTR': 32,
    'ChaCha20-Poly1305': 32, 'ChaCha20': 32, 'Salsa20': 32,
    'XSalsa20-Poly1305': 32,
    '3DES-CBC': 24,
    'RSA-2048': 256, 'ECDSA-P256': 32, 'ECDSA-P384': 48,
    'Ed25519': 32, 'BLAKE2b': 64,
    'HMAC-SHA256': 32,
    'Curve25519-SealedBox': 32, 'Curve25519-Box': 32,
}

ALGO_IV_LENGTHS = {
    'AES-128-CBC': 16, 'AES-192-CBC': 16, 'AES-256-CBC': 16,
    'AES-256-GCM': 12, 'AES-256-CTR': 16,
    'AES-128-CTR': 16, 'AES-192-CTR': 16,
    'ChaCha20-Poly1305': 12, 'ChaCha20': 8, 'Salsa20': 8,
    'XSalsa20-Poly1305': 24,
    '3DES-CBC': 8,
}


def _clip(val, lo, hi):
    return max(lo, min(hi, val))


def _pick_region(library: str, class_name: str = 'NON_CRYPTO') -> float:
    """라이브러리별 메모리 영역(F6) 선택."""
    values, probs = LIBRARY_REGION_DIST[library]
    return float(np.random.choice(values, p=probs))


def _pick_algorithm(library: str) -> str:
    """라이브러리에서 랜덤 알고리즘 선택."""
    return np.random.choice(LIBRARY_ALGORITHMS[library])


def generate_key_sample(library: str, algorithm: str) -> np.ndarray:
    """KEY 클래스 표본: 고엔트로피 + 표준 키 길이."""
    entropy = _clip(np.random.normal(7.7, 0.25), 6.5, 8.0)
    key_len = ALGO_KEY_LENGTHS.get(algorithm, 32)
    length = float(key_len)
    chi2 = _clip(np.random.normal(260, 50), 100, 400) if length >= 256 else 0.0
    is_key = 1.0 if key_len in STANDARD_KEY_LENGTHS else 0.0
    is_iv = 1.0 if key_len in STANDARD_IV_LENGTHS else 0.0
    region = _pick_region(library, 'KEY')
    changes = float(np.random.randint(0, 20))
    pattern = float(np.random.choice([0, 1, 2, 3], p=[0.3, 0.15, 0.25, 0.3]))
    interaction = entropy * log2(length + 1)
    high_conf = 1.0 if entropy >= 7.5 and is_key == 1.0 else 0.0
    return np.array([entropy, chi2, length, is_key, is_iv,
                     region, changes, pattern, interaction, high_conf])


def generate_iv_sample(library: str, algorithm: str) -> np.ndarray:
    """IV 클래스 표본: 고엔트로피 + IV 길이 (8/12/16/24)."""
    entropy = _clip(np.random.normal(7.4, 0.4), 5.5, 8.0)
    iv_len = ALGO_IV_LENGTHS.get(algorithm, 16)
    length = float(iv_len)
    chi2 = 0.0
    is_key = 1.0 if iv_len in STANDARD_KEY_LENGTHS else 0.0
    is_iv = 1.0
    region = _pick_region(library, 'IV')
    changes = float(np.random.randint(0, 10))
    pattern = float(np.random.choice([0, 0, 1, 2], p=[0.4, 0.2, 0.2, 0.2]))
    interaction = entropy * log2(length + 1)
    high_conf = 1.0 if entropy >= 7.5 and is_key == 1.0 else 0.0
    return np.array([entropy, chi2, length, is_key, is_iv,
                     region, changes, pattern, interaction, high_conf])


def generate_ciphertext_sample(library: str, algorithm: str) -> np.ndarray:
    """CIPHERTEXT 클래스 표본: 고엔트로피 + 비표준 길이 가능."""
    entropy = _clip(np.random.normal(7.6, 0.3), 6.0, 8.0)
    length = float(np.random.choice([16, 32, 48, 64, 128, 256, 512, 1024]))
    chi2 = _clip(np.random.normal(270, 60), 100, 500) if length >= 256 else 0.0
    is_key = 1.0 if int(length) in STANDARD_KEY_LENGTHS else 0.0
    is_iv = 1.0 if int(length) in STANDARD_IV_LENGTHS else 0.0
    region = _pick_region(library, 'CIPHERTEXT')
    changes = float(np.random.randint(5, 20))
    pattern = float(np.random.choice([2, 3, 3, 1], p=[0.3, 0.35, 0.25, 0.1]))
    interaction = entropy * log2(length + 1)
    high_conf = 1.0 if entropy >= 7.5 and is_key == 1.0 else 0.0
    return np.array([entropy, chi2, length, is_key, is_iv,
                     region, changes, pattern, interaction, high_conf])


def generate_plaintext_sample(library: str, algorithm: str) -> np.ndarray:
    """PLAINTEXT 클래스 표본: 저~중간 엔트로피."""
    entropy = _clip(np.random.normal(4.5, 0.8), 2.0, 6.5)
    length = float(np.random.choice([16, 32, 64, 128, 256, 512]))
    chi2 = _clip(np.random.normal(800, 200), 400, 2000) if length >= 256 else 0.0
    is_key = 1.0 if int(length) in STANDARD_KEY_LENGTHS else 0.0
    is_iv = 1.0 if int(length) in STANDARD_IV_LENGTHS else 0.0
    region = _pick_region(library, 'PLAINTEXT')
    changes = float(np.random.randint(0, 5))
    pattern = float(np.random.choice([0, 0, 1], p=[0.5, 0.3, 0.2]))
    interaction = entropy * log2(length + 1)
    high_conf = 0.0
    return np.array([entropy, chi2, length, is_key, is_iv,
                     region, changes, pattern, interaction, high_conf])


def generate_non_crypto_sample(library: str, algorithm: str) -> np.ndarray:
    """NON_CRYPTO 클래스 표본: 다양한 엔트로피 (높은 엔트로피도 포함)."""
    if np.random.random() < 0.2:
        entropy = _clip(np.random.normal(7.0, 0.5), 6.0, 7.8)
    else:
        entropy = _clip(np.random.normal(3.5, 1.5), 0.5, 6.5)
    length = float(np.random.choice([8, 12, 16, 20, 24, 32, 48, 64, 100, 200, 500]))
    chi2 = _clip(np.random.normal(900, 300), 300, 3000) if length >= 256 else 0.0
    is_key = 1.0 if int(length) in STANDARD_KEY_LENGTHS else 0.0
    is_iv = 1.0 if int(length) in STANDARD_IV_LENGTHS else 0.0
    region = _pick_region(library, 'NON_CRYPTO')
    changes = float(np.random.randint(0, 20))
    pattern = float(np.random.choice([0, 1, 2, 3]))
    interaction = entropy * log2(length + 1)
    high_conf = 1.0 if entropy >= 7.5 and is_key == 1.0 else 0.0
    return np.array([entropy, chi2, length, is_key, is_iv,
                     region, changes, pattern, interaction, high_conf])


GENERATORS = {
    'KEY': generate_key_sample,
    'IV': generate_iv_sample,
    'CIPHERTEXT': generate_ciphertext_sample,
    'PLAINTEXT': generate_plaintext_sample,
    'NON_CRYPTO': generate_non_crypto_sample,
}


def _generate_fake_address() -> str:
    """가상 메모리 주소 생성."""
    high = np.random.randint(0x1000, 0x7FFF)
    low = np.random.randint(0x00000000, 0x7FFFFFFF)
    return f"0x{high:04X}{low:08X}"


def generate_dataset(output_dir: str = 'dataset'):
    """논문 분포에 맞는 합성 데이터셋 생성.

    생성 파일:
      - crypto_features.csv: 10개 특징 + label + library + algorithm (10,000개)
      - real_crypto_features.csv: 메타데이터 포함 버전
      - real_crypto_features_ml.csv: 10개 특징 + label만 (ML 학습용)
    """
    all_features = []
    all_labels = []
    all_libraries = []
    all_algorithms = []

    for lib_name, class_counts in LIBRARY_CLASS_MATRIX.items():
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = class_counts[cls_idx]
            gen_fn = GENERATORS[cls_name]
            for _ in range(count):
                algo = _pick_algorithm(lib_name)
                feat = gen_fn(lib_name, algo)
                all_features.append(feat)
                all_labels.append(cls_name)
                all_libraries.append(lib_name)
                all_algorithms.append(algo)

    X = np.array(all_features)

    # ── crypto_features.csv: 10개 특징 + label + library + algorithm ──
    columns = FEATURE_NAMES + ['label', 'library', 'algorithm']
    df = pd.DataFrame(
        np.column_stack([X, all_labels, all_libraries, all_algorithms]),
        columns=columns,
    )
    for col in FEATURE_NAMES:
        df[col] = df[col].astype(float)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    crypto_path = os.path.join(output_dir, 'crypto_features.csv')
    df.to_csv(crypto_path, index=False)

    # ── real_crypto_features.csv: 메타데이터 포함 ──
    df_real = df.copy()
    df_real['data_size'] = df_real['F3_length'].astype(int)
    df_real['address'] = [_generate_fake_address() for _ in range(len(df_real))]
    df_real['description'] = df_real.apply(
        lambda r: f"{r['algorithm']} {r['label'].lower()} from {r['library']}", axis=1
    )
    # 열 순서: 10개 특징, label, algorithm, library, data_size, address, description
    real_cols = FEATURE_NAMES + ['label', 'algorithm', 'library',
                                  'data_size', 'address', 'description']
    df_real = df_real[real_cols]

    real_path = os.path.join(output_dir, 'real_crypto_features.csv')
    df_real.to_csv(real_path, index=False)

    # ── real_crypto_features_ml.csv: 10개 특징 + label ──
    ml_cols = FEATURE_NAMES + ['label']
    df_ml = df[ml_cols].copy()

    ml_path = os.path.join(output_dir, 'real_crypto_features_ml.csv')
    df_ml.to_csv(ml_path, index=False)

    # ── 통계 출력 ──
    total = len(df)
    print(f"합성 데이터셋 생성 완료: {output_dir}/")
    print(f"  총 표본: {total}")
    print()
    print("클래스별 분포:")
    for cls_name in CLASS_NAMES:
        count = len(df[df['label'] == cls_name])
        print(f"  {cls_name:>12s}: {count:>5d} ({count/total*100:.1f}%)")
    print()
    print("라이브러리별 분포:")
    for lib_name in LIBRARY_CLASS_MATRIX:
        count = len(df[df['library'] == lib_name])
        print(f"  {lib_name:>15s}: {count:>5d} ({count/total*100:.1f}%)")
    print()
    print(f"생성 파일:")
    print(f"  {crypto_path} ({total} rows)")
    print(f"  {real_path} ({total} rows)")
    print(f"  {ml_path} ({total} rows)")

    return df


if __name__ == '__main__':
    generate_dataset()
