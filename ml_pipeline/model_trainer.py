"""
model_trainer.py
Paper Section 3.4.3 / Chapter 4: Training and comparative evaluation of 6 models.

- XGBoost: primary classifier (gradient boosting ensemble)
- Random Forest: baseline model (bagging ensemble, 100 trees)
- MLP: baseline model (4 hidden layers [10-64-32-16-5])
- LightGBM: baseline model (gradient boosting, leaf-wise)
- CatBoost: baseline model (gradient boosting, ordered boosting)
- TabNet: baseline model (attention-based deep learning)
- 60/20/20 train/val/test split
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

# --- Class definitions ---
CLASSES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def load_dataset(csv_path: str):
    """
    Load a CSV dataset.

    CSV format:
      F1_entropy, F2_chi_square, ..., F10_high_confidence_key, label
      label: KEY / IV / CIPHERTEXT / PLAINTEXT / NON_CRYPTO

    Returns:
        X: (n_samples, 10) feature array
        y: (n_samples,) encoded label array
        le: LabelEncoder instance
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    return X, y, le


def split_dataset(X, y, random_state=42):
    """60/20/20 split (stratified)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def grid_search_xgboost(X_train, y_train):
    """
    Paper Table hp_search: XGBoost hyperparameter grid search.
    5x6x4x3x3x3x3x3 = 29,160 combinations, 3-fold CV.
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
    print(f"\n  GridSearchCV best parameters: {gs.best_params_}")
    print(f"  GridSearchCV best F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier (primary classifier)."""
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
    """Train Random Forest classifier (baseline model, 100 trees)."""
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
    Train MLP classifier (baseline model).
    Paper architecture: [10 -> 64 -> 32 -> 16 -> 5]
    MLP requires feature scaling, so a StandardScaler is returned alongside the model.
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
    """Train LightGBM classifier (baseline model, leaf-wise gradient boosting)."""
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
    """Train CatBoost classifier (primary classifier, ordered boosting + balanced weights)."""
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
    Train TabNet classifier (baseline model, attention-based deep learning).
    TabNet requires feature scaling, so a StandardScaler is returned alongside the model.
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
    """Print model evaluation results and return metrics."""
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
    Full pipeline: load data -> train 3 models -> comparative evaluation -> save models.
    """
    print("Loading dataset...")
    X, y, le = load_dataset(csv_path)
    print(f"  Total samples: {len(y)}")
    for cls_name in CLASSES:
        cls_idx = le.transform([cls_name])[0]
        count = np.sum(y == cls_idx)
        print(f"  {cls_name}: {count} ({count/len(y)*100:.1f}%)")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"\n  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    results = {}

    # -- XGBoost --
    print("\n[1/3] Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, le, 'XGBoost')
    results['XGBoost']['model'] = xgb_model

    # -- Random Forest --
    print("\n[2/3] Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    results['RandomForest'] = evaluate_model(rf_model, X_test, y_test, le, 'Random Forest')
    results['RandomForest']['model'] = rf_model

    # -- MLP --
    print("\n[3/3] Training MLP...")
    mlp_model, scaler = train_mlp(X_train, y_train)
    results['MLP'] = evaluate_model(
        mlp_model, X_test, y_test, le, 'MLP', scaler=scaler
    )
    results['MLP']['model'] = mlp_model
    results['MLP']['scaler'] = scaler

    # -- Summary comparison --
    print(f"\n{'='*60}")
    print("  Performance Summary (Paper Table 3)")
    print(f"{'='*60}")
    print(f"  {'Model':<20s} {'Precision':>8s} {'Recall':>8s} {'F1':>8s}")
    print(f"  {'-'*46}")
    for name in ['XGBoost', 'RandomForest', 'MLP']:
        r = results[name]
        print(f"  {name:<20s} {r['precision']:>7.1%} {r['recall']:>7.1%} {r['f1']:>7.1%}")

    # -- Save models --
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(save_dir, 'xgb_classifier.pkl'))
    joblib.dump(rf_model, os.path.join(save_dir, 'rf_classifier.pkl'))
    joblib.dump(mlp_model, os.path.join(save_dir, 'mlp_classifier.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'mlp_scaler.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
    print(f"\n  Models saved to: {save_dir}/")

    return results, X_train, X_val, X_test, y_train, y_val, y_test, le


# --- Standalone execution ---
if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    train_and_evaluate_all(csv_path, save_dir='models')
