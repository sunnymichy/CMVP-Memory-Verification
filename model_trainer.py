"""
model_trainer.py
논문 섹션 3.4.3 / 4장: 6개 모델의 학습 및 비교 평가.

- XGBoost: 주 분류기 (gradient boosting 앙상블)
- Random Forest: 비교 모델 (bagging 앙상블, 100 trees)
- MLP: 비교 모델 (4개 은닉층 [10-64-32-16-5])
- LightGBM: 비교 모델 (gradient boosting, leaf-wise)
- CatBoost: 비교 모델 (gradient boosting, ordered boosting)
- TabNet: 비교 모델 (attention-based deep learning)
- 60/20/20 train/val/test 분할
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

from feature_extractor import FEATURE_NAMES

# ─── 클래스 정의 ───
CLASSES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def load_dataset(csv_path: str):
    """
    CSV 데이터셋을 로드한다.

    CSV 형식:
      F1_entropy, F2_chi_square, ..., F10_high_confidence_key, label
      label: KEY / IV / CIPHERTEXT / PLAINTEXT / NON_CRYPTO

    Returns:
        X: (n_samples, 10) 특징 배열
        y: (n_samples,) 인코딩된 레이블 배열
        le: LabelEncoder 인스턴스
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    return X, y, le


def split_dataset(X, y, random_state=42):
    """60/20/20 분할 (stratified)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def grid_search_xgboost(X_train, y_train):
    """
    논문 표 hp_search: XGBoost 하이퍼파라미터 그리드 탐색.
    5×6×4×3×3×3×3×3 = 29,160 조합, 3-겹 CV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.5, 1.0, 2.0],
    }
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(CLASSES),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        base_model, param_grid, scoring='f1_weighted',
        cv=cv, n_jobs=-1, verbose=1, refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"\n  GridSearchCV 최적 파라미터: {gs.best_params_}")
    print(f"  GridSearchCV 최적 F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost 분류기 학습 (주 분류기)."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(CLASSES),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_random_forest(X_train, y_train):
    """Random Forest 분류기 학습 (비교 모델, 100 trees)."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    """
    MLP 분류기 학습 (비교 모델).
    논문 구조: [10 → 64 → 32 → 16 → 5]
    MLP는 스케일링 필수이므로 StandardScaler를 함께 반환.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_lightgbm(X_train, y_train):
    """LightGBM 분류기 학습 (비교 모델, leaf-wise gradient boosting)."""
    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    """CatBoost 분류기 학습 (주 분류기, ordered boosting + balanced weights)."""
    model = CatBoostClassifier(
        iterations=1000,
        depth=9,
        learning_rate=0.05,
        bootstrap_type='MVS',
        subsample=0.9,
        colsample_bylevel=0.8,
        l2_leaf_reg=1.0,
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',
    )
    model.fit(X_train, y_train)
    return model


def train_tabnet(X_train, y_train):
    """
    TabNet 분류기 학습 (비교 모델, attention-based deep learning).
    TabNet은 스케일링 필수이므로 StandardScaler를 함께 반환.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = TabNetClassifier(
        n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=42, verbose=0,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_model(model, X_test, y_test, le, model_name,
                   scaler=None):
    """모델 평가 결과를 출력하고 지표를 반환한다."""
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)

    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return {
        'model': model,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
    }


def train_and_evaluate_all(csv_path: str, save_dir: str = 'models'):
    """
    전체 파이프라인: 데이터 로드 → 3개 모델 학습 → 비교 평가 → 모델 저장.
    """
    print("데이터셋 로드 중...")
    X, y, le = load_dataset(csv_path)
    print(f"  총 표본: {len(y)}")
    for cls_name in CLASSES:
        cls_idx = le.transform([cls_name])[0]
        count = np.sum(y == cls_idx)
        print(f"  {cls_name}: {count} ({count/len(y)*100:.1f}%)")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"\n  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    results = {}

    # ── XGBoost ──
    print("\n[1/3] XGBoost 학습 중...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, le, 'XGBoost')
    results['XGBoost']['model'] = xgb_model

    # ── Random Forest ──
    print("\n[2/3] Random Forest 학습 중...")
    rf_model = train_random_forest(X_train, y_train)
    results['RandomForest'] = evaluate_model(rf_model, X_test, y_test, le, 'Random Forest')
    results['RandomForest']['model'] = rf_model

    # ── MLP ──
    print("\n[3/3] MLP 학습 중...")
    mlp_model, scaler = train_mlp(X_train, y_train)
    results['MLP'] = evaluate_model(
        mlp_model, X_test, y_test, le, 'MLP', scaler=scaler
    )
    results['MLP']['model'] = mlp_model
    results['MLP']['scaler'] = scaler

    # ── 요약 비교 ──
    print(f"\n{'='*60}")
    print("  성능 요약 (논문 표3)")
    print(f"{'='*60}")
    print(f"  {'모델':<20s} {'정밀도':>8s} {'재현율':>8s} {'F1':>8s}")
    print(f"  {'-'*46}")
    for name in ['XGBoost', 'RandomForest', 'MLP']:
        r = results[name]
        print(f"  {name:<20s} {r['precision']:>7.1%} {r['recall']:>7.1%} {r['f1']:>7.1%}")

    # ── 모델 저장 ──
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(save_dir, 'xgb_classifier.pkl'))
    joblib.dump(rf_model, os.path.join(save_dir, 'rf_classifier.pkl'))
    joblib.dump(mlp_model, os.path.join(save_dir, 'mlp_classifier.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'mlp_scaler.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
    print(f"\n  모델 저장 완료: {save_dir}/")

    return results, X_train, X_val, X_test, y_train, y_val, y_test, le


# ─── 단독 실행 ───
if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    train_and_evaluate_all(csv_path, save_dir='models')
