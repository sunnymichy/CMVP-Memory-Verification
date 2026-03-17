"""
collect_real_data.py
Main script that collects data from real cryptographic libraries and generates
training CSV files.

Collection procedure:
  1) Perform cryptographic operations + obtain memory addresses for
     key/IV/ciphertext/plaintext (ground truth)
  2) Collect temporal change patterns (multiple snapshots)
  3) Extract 10-dimensional feature vectors
  4) Output CSV

Usage:
  python collect_real_data.py                    # default settings
  python collect_real_data.py --reps 10          # 10 repetitions
  python collect_real_data.py --output real.csv  # specify output file
"""

import os
import sys
import time
import argparse
import ctypes
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Reference modules from the parent directory
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
    Collect samples including temporal change patterns.

    Session key simulation: generate a new key per repetition -> value changes at the same address
    Master key simulation: generate once and retain -> static pattern

    Returns:
        samples: list of CryptoSample objects
        feature_vectors: list of 10-dimensional feature vectors for each sample
    """
    handle = get_own_process_handle()
    all_samples = []
    all_features = []

    # --- Phase A: Static keys (master key simulation) ---
    print("\n[Phase A] Collecting static keys (master key simulation)...")
    static_keys = []
    static_addrs = []

    # Generate static keys of various sizes
    for key_size in [16, 24, 32]:
        key_data = os.urandom(key_size)
        addr, buf = _pin_bytes(key_data)
        static_keys.append(key_data)
        static_addrs.append((addr, key_size))

    # Snapshots for static keys (values do not change)
    print(f"  Capturing {snapshot_count} snapshots ({snapshot_interval_ms}ms interval)...")
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
        print(f"  Static key {size}B: changes={changes}, pattern=F8={features[7]}")

    # --- Phase B: Dynamic keys (session key simulation) ---
    print("\n[Phase B] Collecting dynamic keys (session key simulation)...")

    # Repeatedly overwrite new keys at a fixed address
    dynamic_bufs = {}
    for key_size in [16, 32]:
        buf = (ctypes.c_ubyte * key_size)()
        _keep_alive_list().append(buf)
        addr = ctypes.addressof(buf)
        dynamic_bufs[key_size] = (addr, buf)

    dynamic_snapshots = []
    for snap_i in range(snapshot_count):
        # Overwrite with new key values at each snapshot
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
        print(f"  Dynamic key {key_size}B: changes={changes}, pattern=F8={features[7]}")

    # --- Phase C: Crypto library collection ---
    print(f"\n[Phase C] Collecting from crypto libraries ({repetitions} repetitions)...")
    lib_samples = collect_all(
        repetitions=repetitions,
        non_crypto_count=0  # NON_CRYPTO is handled separately
    )

    for sample in lib_samples:
        # Temporal pattern: library samples are single snapshots so change_count=0 (static)
        # In a real environment, this should be computed from external process snapshots
        change_count = 0
        features = extract_features(
            sample.data, sample.region_type,
            change_count, snapshot_count
        )
        all_samples.append(sample)
        all_features.append(features)

    # --- Phase D: NON_CRYPTO collection ---
    # Goal: approximately 15% of the total
    total_crypto = len(all_samples)
    non_crypto_target = max(50, int(total_crypto * 0.15 / 0.85))

    print(f"\n[Phase D] Collecting NON_CRYPTO (target: {non_crypto_target} samples)...")
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
    """Save the collected results to a CSV file."""
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
    """Print a summary of the dataset."""
    print(f"\n{'='*60}")
    print(f"  Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total samples: {len(df)}")

    print(f"\n  Distribution by class:")
    for label in ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']:
        count = len(df[df['label'] == label])
        pct = count / len(df) * 100
        print(f"    {label:<12s}: {count:>5d} ({pct:>5.1f}%)")

    print(f"\n  Distribution by library:")
    for lib in sorted(df['library'].unique()):
        count = len(df[df['library'] == lib])
        print(f"    {lib:<20s}: {count:>5d}")

    print(f"\n  Distribution by algorithm:")
    for algo in sorted(df['algorithm'].unique()):
        count = len(df[df['algorithm'] == algo])
        print(f"    {algo:<25s}: {count:>5d}")

    # Feature statistics
    print(f"\n  Feature statistics:")
    feat_cols = FEATURE_NAMES
    for col in feat_cols:
        vals = df[col].astype(float)
        print(f"    {col:<35s}: "
              f"mean={vals.mean():>8.3f}  std={vals.std():>8.3f}  "
              f"min={vals.min():>8.3f}  max={vals.max():>8.3f}")

    # Entropy distribution by class
    print(f"\n  Mean entropy by class:")
    for label in ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']:
        subset = df[df['label'] == label]['F1_entropy'].astype(float)
        if len(subset) > 0:
            print(f"    {label:<12s}: {subset.mean():.3f} +/- {subset.std():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect ML training data from real cryptographic libraries.'
    )
    parser.add_argument('--reps', type=int, default=30,
                        help='Number of repetitions for each crypto operation (default: 30)')
    parser.add_argument('--snapshots', type=int, default=10,
                        help='Number of snapshots for temporal pattern analysis (default: 10)')
    parser.add_argument('--interval', type=int, default=200,
                        help='Snapshot capture interval in ms (default: 200)')
    parser.add_argument('--output', type=str,
                        default='../dataset/real_crypto_features.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print("=" * 60)
    print("  KCMVP ML Dataset Collector (5 libraries, 33 algorithm-library combinations)")
    print("=" * 60)
    print(f"  Repetitions:       {args.reps}")
    print(f"  Snapshots:         {args.snapshots}")
    print(f"  Capture interval:  {args.interval}ms")
    print(f"  Output file:       {args.output}")

    start_time = time.time()

    samples, features = collect_temporal_samples(
        repetitions=args.reps,
        snapshot_count=args.snapshots,
        snapshot_interval_ms=args.interval,
    )

    elapsed = time.time() - start_time
    print(f"\nCollection complete: {len(samples)} samples ({elapsed:.1f}s)")

    # Save CSV
    df = build_csv(samples, features, args.output)
    print_summary(df)

    print(f"\n  CSV saved: {args.output}")

    # CSV for ML training (features + labels only)
    ml_cols = FEATURE_NAMES + ['label']
    ml_path = args.output.replace('.csv', '_ml.csv')
    df[ml_cols].to_csv(ml_path, index=False)
    print(f"  ML training CSV: {ml_path}")

    print(f"\n  Next steps:")
    print(f"    cd .. && python model_trainer.py {ml_path}")
    print(f"    cd .. && python evaluate.py {ml_path}")


if __name__ == '__main__':
    main()
