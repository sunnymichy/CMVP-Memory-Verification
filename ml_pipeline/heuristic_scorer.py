"""
heuristic_scorer.py
논문 섹션 3.4.2: 휴리스틱 점수 산출 체계 (0~100점).

6개 신호에 가중치를 부여하여 암호키 가능성 점수를 산출한다.
이 점수는 (1) 단독 baseline 분류기로, (2) 하이브리드 앙상블의 규칙 기반 경로로 활용된다.
"""

import numpy as np

# 클래스 인덱스
CLASS_KEY = 0
CLASS_IV = 1
CLASS_CIPHERTEXT = 2
CLASS_PLAINTEXT = 3
CLASS_NON_CRYPTO = 4

CLASS_NAMES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def compute_heuristic_score(features: np.ndarray) -> float:
    """
    10차원 특징 벡터로부터 휴리스틱 점수(0~100)를 산출한다.

    점수 배분:
      신호1 - 엔트로피:    최대 30점
      신호2 - 카이제곱:    최대 25점
      신호3 - 길이 제약:   최대 15점
      신호4 - 영역 유형:   최대 10점
      시너지 보너스:       20점
      신호5 - 시간 패턴:   최대 10점
      합계 상한:           100점
    """
    entropy = features[0]       # F1
    chi2 = features[1]          # F2
    length = features[2]        # F3
    is_key_len = features[3]    # F4
    is_iv_len = features[4]     # F5
    region = features[5]        # F6
    pattern = features[7]       # F8

    score = 0.0

    # ── 신호 1: 엔트로피 (최대 30점) ──
    # 실측 데이터 기반 신뢰도별 임계값
    if entropy >= 7.81:
        score += 30    # 99% 신뢰도
    elif entropy >= 7.5:
        score += 25    # 95% 신뢰도
    elif entropy >= 7.0:
        score += 15
    elif entropy >= 6.5:
        score += 10
    elif entropy >= 5.0:
        score += 5

    # ── 신호 2: 카이제곱 균일성 (최대 25점) ──
    # chi2 > 0 은 데이터 >= 256바이트일 때만 유효
    if chi2 > 0:
        if chi2 < 350:      # 균등 분포 (임계값 < 350)
            score += 25
        elif chi2 < 400:    # 부분 균등
            score += 15
        else:               # 비균등
            score += 5

    # ── 신호 3: 길이 제약 (최대 15점) ──
    if is_key_len == 1.0:
        score += 15          # 28종 표준 키 길이 일치
    elif is_iv_len == 1.0:
        score += 12          # 표준 IV 길이 (8/12/16)
    elif length >= 16:
        score += 5

    # ── 신호 4: 메모리 영역 유형 (최대 10점) ──
    region_int = int(region)
    if region_int == 1:      # DLL Data Section
        score += 10
    elif region_int == 2:    # Stack/Heap
        score += 7
    else:                    # Unknown(0) 또는 Other(3)
        score += 3

    # ── 시너지 보너스 (20점) ──
    # 고엔트로피 + 표준 키 길이 동시 충족
    if entropy >= 7.5 and is_key_len == 1.0:
        score += 20

    # ── 신호 5: 시간적 변화 패턴 (최대 10점) ──
    pattern_int = int(pattern)
    pattern_scores = {
        3: 10,   # ALWAYS  → 단기 세션 키 특성
        2: 8,    # FREQUENT
        0: 6,    # STATIC  → 장기 마스터 키 특성
        1: 4,    # PARTIAL
    }
    score += pattern_scores.get(pattern_int, 0)

    return min(score, 100.0)


def heuristic_to_class(score: float, features: np.ndarray) -> int:
    """
    휴리스틱 점수와 특징으로부터 클래스를 결정한다.
    논문 섹션 3.4.5 하이브리드 앙상블의 경로 3 규칙.

    Returns:
        클래스 인덱스 (0=KEY, 1=IV, 2=CIPHERTEXT, 3=PLAINTEXT, 4=NON_CRYPTO)
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
    8~15바이트 단축 블록에 대한 강화 판정.

    논문: 8바이트 블록은 표본 크기가 작아 우연히 높은 엔트로피를 나타낼
    확률이 증가하므로 강화된 판정 조건을 적용한다.

    Args:
        features: 10차원 특징 벡터
        score: 휴리스틱 점수
        sensitivity: 'high' / 'normal' / 'low'
    Returns:
        True면 암호 데이터 후보로 채택
    """
    length = int(features[2])
    entropy = features[0]

    if length >= 16:
        # 일반 블록: 표준 임계값 적용
        thresholds = {'high': 40, 'normal': 60, 'low': 75}
        return score >= thresholds.get(sensitivity, 60)

    # 단축 블록 (8~15바이트): 강화 조건
    short_thresholds = {'high': 55, 'normal': 70, 'low': 85}
    min_entropy = 6.5 if sensitivity == 'high' else 7.0

    return (score >= short_thresholds.get(sensitivity, 70)
            and entropy >= min_entropy)


# ─── 단독 실행 시 테스트 ───
if __name__ == '__main__':
    # 테스트: 다양한 시나리오
    test_cases = [
        ("AES-256 키 (DLL, 항상변경)",
         np.array([7.9, 250.0, 32.0, 1.0, 0.0, 1.0, 18.0, 3.0, 39.5, 1.0])),
        ("AES-CBC IV (힙, 정적)",
         np.array([7.6, 0.0, 16.0, 1.0, 1.0, 2.0, 0.0, 0.0, 30.4, 1.0])),
        ("암호문 (256B, 힙)",
         np.array([7.8, 260.0, 256.0, 1.0, 0.0, 2.0, 5.0, 1.0, 62.4, 1.0])),
        ("평문 텍스트 (힙)",
         np.array([4.2, 800.0, 64.0, 1.0, 0.0, 2.0, 0.0, 0.0, 17.6, 0.0])),
        ("비암호 데이터",
         np.array([3.1, 1200.0, 100.0, 0.0, 0.0, 3.0, 12.0, 2.0, 10.4, 0.0])),
    ]

    for name, feat in test_cases:
        score = compute_heuristic_score(feat)
        cls = heuristic_to_class(score, feat)
        print(f"{name:<30s} → 점수: {score:5.1f} → 분류: {CLASS_NAMES[cls]}")
