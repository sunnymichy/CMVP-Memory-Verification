"""
collect_real_data.py
실제 암호 라이브러리로부터 데이터를 수집하여 학습용 CSV를 생성하는 메인 스크립트.

수집 절차:
  1) 암호 연산 수행 + 키/IV/암호문/평문 메모리 주소 확보 (ground truth)
  2) 시간적 변화 패턴 수집 (다중 스냅샷)
  3) 10차원 특징 벡터 추출
  4) CSV 출력

사용법:
  python collect_real_data.py                    # 기본 설정
  python collect_real_data.py --reps 10          # 10회 반복
  python collect_real_data.py --output real.csv  # 출력 파일 지정
"""

import os
import sys
import time
import argparse
import ctypes
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# 상위 디렉토리의 모듈 참조
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from win_memory import (
    get_own_process_handle,
    enumerate_regions,
    read_memory,
    get_region_type_at,
    take_snapshots,
    count_changes,
    MemorySnapshot,
)
from crypto_ops import (
    CryptoSample,
    collect_all,
    collect_non_crypto,
    _pin_bytes,
    _keep_alive_list,
)
from feature_extractor import (
    extract_features,
    FEATURE_NAMES,
    STANDARD_KEY_LENGTHS,
)


def collect_temporal_samples(repetitions: int = 30,
                             snapshot_count: int = 10,
                             snapshot_interval_ms: int = 200,
                             ) -> Tuple[List[CryptoSample], List[np.ndarray]]:
    """
    시간적 변화 패턴을 포함한 표본 수집.

    세션 키 시뮬레이션: 반복마다 새 키 생성 → 같은 주소에서 값이 변경됨
    마스터 키 시뮬레이션: 최초 1회 생성 후 유지 → 정적 패턴

    Returns:
        samples: CryptoSample 리스트
        feature_vectors: 각 표본의 10차원 특징 벡터 리스트
    """
    handle = get_own_process_handle()
    all_samples = []
    all_features = []

    # ─── Phase A: 정적 키 (마스터 키 시뮬레이션) ───
    print("\n[Phase A] 정적 키 수집 (마스터 키 시뮬레이션)...")
    static_keys = []
    static_addrs = []

    # 다양한 크기의 정적 키 생성
    for key_size in [16, 24, 32]:
        key_data = os.urandom(key_size)
        addr, buf = _pin_bytes(key_data)
        static_keys.append(key_data)
        static_addrs.append((addr, key_size))

    # 정적 키에 대한 스냅샷 (값이 변하지 않음)
    print(f"  스냅샷 {snapshot_count}회 캡처 중 ({snapshot_interval_ms}ms 간격)...")
    snapshots = take_snapshots(handle, static_addrs,
                               count=snapshot_count,
                               interval_ms=snapshot_interval_ms)

    for i, (key_data, (addr, size)) in enumerate(zip(static_keys, static_addrs)):
        changes = count_changes(snapshots, addr)
        region_type = get_region_type_at(handle, addr)

        sample = CryptoSample(
            label='KEY', data=key_data, address=addr,
            region_type=region_type,
            algorithm=f'AES-{size*8}', library='static_sim',
            description=f'static master key (changes={changes})'
        )
        features = extract_features(
            key_data, region_type, changes, snapshot_count
        )
        all_samples.append(sample)
        all_features.append(features)
        print(f"  정적 키 {size}B: changes={changes}, pattern=F8={features[7]}")

    # ─── Phase B: 동적 키 (세션 키 시뮬레이션) ───
    print("\n[Phase B] 동적 키 수집 (세션 키 시뮬레이션)...")

    # 고정 주소에 반복적으로 새 키를 덮어씀
    dynamic_bufs = {}
    for key_size in [16, 32]:
        buf = (ctypes.c_ubyte * key_size)()
        _keep_alive_list().append(buf)
        addr = ctypes.addressof(buf)
        dynamic_bufs[key_size] = (addr, buf)

    dynamic_snapshots = []
    for snap_i in range(snapshot_count):
        # 매 스냅샷마다 새 키 값으로 덮어쓰기
        for key_size, (addr, buf) in dynamic_bufs.items():
            new_key = os.urandom(key_size)
            ctypes.memmove(buf, new_key, key_size)

        snap = MemorySnapshot()
        addrs_list = [(addr, ks) for ks, (addr, _) in dynamic_bufs.items()]
        snap.capture(handle, addrs_list)
        dynamic_snapshots.append(snap)
        time.sleep(snapshot_interval_ms / 1000.0)

    for key_size, (addr, buf) in dynamic_bufs.items():
        current_data = bytes(buf[:key_size])
        changes = count_changes(dynamic_snapshots, addr)
        region_type = get_region_type_at(handle, addr)

        sample = CryptoSample(
            label='KEY', data=current_data, address=addr,
            region_type=region_type,
            algorithm=f'AES-{key_size*8}', library='dynamic_sim',
            description=f'dynamic session key (changes={changes})'
        )
        features = extract_features(
            current_data, region_type, changes, snapshot_count
        )
        all_samples.append(sample)
        all_features.append(features)
        print(f"  동적 키 {key_size}B: changes={changes}, pattern=F8={features[7]}")

    # ─── Phase C: 실제 라이브러리 수집 ───
    print(f"\n[Phase C] 암호 라이브러리 수집 ({repetitions}회 반복)...")
    lib_samples = collect_all(
        repetitions=repetitions,
        non_crypto_count=0  # NON_CRYPTO는 별도 처리
    )

    for sample in lib_samples:
        # 시간 패턴: 라이브러리 표본은 단일 스냅샷이므로 change_count=0 (정적)
        # 실제 환경에서는 외부 프로세스 스냅샷에서 계산해야 함
        change_count = 0
        features = extract_features(
            sample.data, sample.region_type,
            change_count, snapshot_count
        )
        all_samples.append(sample)
        all_features.append(features)

    # ─── Phase D: NON_CRYPTO 수집 ───
    # 목표: 전체의 ~15%
    total_crypto = len(all_samples)
    non_crypto_target = max(50, int(total_crypto * 0.15 / 0.85))

    print(f"\n[Phase D] NON_CRYPTO 수집 (목표: {non_crypto_target}개)...")
    crypto_patterns = [s.data for s in all_samples if s.label in ('KEY', 'IV')]
    nc_samples = collect_non_crypto(non_crypto_target, crypto_patterns)

    for sample in nc_samples:
        features = extract_features(
            sample.data, sample.region_type, 0, snapshot_count
        )
        all_samples.append(sample)
        all_features.append(features)

    return all_samples, all_features


def build_csv(samples: List[CryptoSample],
              features: List[np.ndarray],
              output_path: str):
    """수집 결과를 CSV로 저장한다."""
    rows = []
    for sample, feat in zip(samples, features):
        row = {}
        for i, fname in enumerate(FEATURE_NAMES):
            row[fname] = feat[i]
        row['label'] = sample.label
        row['algorithm'] = sample.algorithm
        row['library'] = sample.library
        row['data_size'] = len(sample.data)
        row['address'] = f"0x{sample.address:X}"
        row['description'] = sample.description
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def print_summary(df: pd.DataFrame):
    """데이터셋 요약을 출력한다."""
    print(f"\n{'='*60}")
    print(f"  데이터셋 요약")
    print(f"{'='*60}")
    print(f"  총 표본: {len(df)}")

    print(f"\n  클래스별 분포:")
    for label in ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']:
        count = len(df[df['label'] == label])
        pct = count / len(df) * 100
        print(f"    {label:<12s}: {count:>5d} ({pct:>5.1f}%)")

    print(f"\n  라이브러리별 분포:")
    for lib in sorted(df['library'].unique()):
        count = len(df[df['library'] == lib])
        print(f"    {lib:<20s}: {count:>5d}")

    print(f"\n  알고리즘별 분포:")
    for algo in sorted(df['algorithm'].unique()):
        count = len(df[df['algorithm'] == algo])
        print(f"    {algo:<25s}: {count:>5d}")

    # 특징 통계
    print(f"\n  특징 통계:")
    feat_cols = FEATURE_NAMES
    for col in feat_cols:
        vals = df[col].astype(float)
        print(f"    {col:<35s}: "
              f"mean={vals.mean():>8.3f}  std={vals.std():>8.3f}  "
              f"min={vals.min():>8.3f}  max={vals.max():>8.3f}")

    # 클래스별 엔트로피 분포
    print(f"\n  클래스별 평균 엔트로피:")
    for label in ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']:
        subset = df[df['label'] == label]['F1_entropy'].astype(float)
        if len(subset) > 0:
            print(f"    {label:<12s}: {subset.mean():.3f} +/- {subset.std():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='실제 암호 라이브러리로부터 ML 학습 데이터를 수집한다.'
    )
    parser.add_argument('--reps', type=int, default=30,
                        help='각 암호 연산 반복 횟수 (기본: 30)')
    parser.add_argument('--snapshots', type=int, default=10,
                        help='시간적 패턴 분석용 스냅샷 수 (기본: 10)')
    parser.add_argument('--interval', type=int, default=200,
                        help='스냅샷 캡처 간격 ms (기본: 200)')
    parser.add_argument('--output', type=str,
                        default='../dataset/real_crypto_features.csv',
                        help='출력 CSV 경로')
    args = parser.parse_args()

    print("=" * 60)
    print("  KCMVP ML 데이터셋 수집기 (5 라이브러리, 33 알고리즘-라이브러리 조합)")
    print("=" * 60)
    print(f"  반복 횟수:     {args.reps}")
    print(f"  스냅샷 수:     {args.snapshots}")
    print(f"  캡처 간격:     {args.interval}ms")
    print(f"  출력 파일:     {args.output}")

    start_time = time.time()

    samples, features = collect_temporal_samples(
        repetitions=args.reps,
        snapshot_count=args.snapshots,
        snapshot_interval_ms=args.interval,
    )

    elapsed = time.time() - start_time
    print(f"\n수집 완료: {len(samples)}개 표본 ({elapsed:.1f}초)")

    # CSV 저장
    df = build_csv(samples, features, args.output)
    print_summary(df)

    print(f"\n  CSV 저장 완료: {args.output}")

    # ML 학습용 CSV (특징 + 레이블만)
    ml_cols = FEATURE_NAMES + ['label']
    ml_path = args.output.replace('.csv', '_ml.csv')
    df[ml_cols].to_csv(ml_path, index=False)
    print(f"  ML 학습용 CSV: {ml_path}")

    print(f"\n  다음 단계:")
    print(f"    cd .. && python model_trainer.py {ml_path}")
    print(f"    cd .. && python evaluate.py {ml_path}")


if __name__ == '__main__':
    main()
