"""
hybrid_ensemble.py
Paper Section 3.4.5: Hybrid Ensemble Technique.

Combines ML probability predictions with rule-based heuristics
to improve detection performance on borderline cases.

Three-path decision logic:
  Path 1: ML confidence >= 90% -> Adopt ML result
  Path 2: ML confidence 70-90% -> Compare ML and heuristic; adopt if they agree
  Path 3: ML confidence < 70% -> Heuristic rule-based decision
"""

import numpy as np
from typing import Tuple

from heuristic_scorer import (
    compute_heuristic_score,
    heuristic_to_class,
    check_short_block_threshold,
    CLASS_NAMES,
)


class HybridEnsemble:
    """Hybrid ensemble classifier."""

    def __init__(self, ml_model, tau_high: float = 0.90, tau_low: float = 0.70,
                 sensitivity: str = 'normal'):
        """
        Args:
            ml_model: A trained ML model (e.g., XGBoost) that supports predict_proba()
            tau_high: High-confidence threshold (default 90%)
            tau_low: Low-confidence threshold (default 70%)
            sensitivity: Detection sensitivity ('high' / 'normal' / 'low')
        """
        self.ml_model = ml_model
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.sensitivity = sensitivity

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Hybrid ensemble prediction.

        Args:
            X: (n_samples, 10) feature vector array

        Returns:
            predictions: (n_samples,) predicted class indices
            confidences: (n_samples,) confidence scores (0.0-1.0)
            paths: decision path taken by each sample ('ml_high' / 'ml_agree' /
                   'ml_disagree' / 'heuristic')
        """
        proba = self.ml_model.predict_proba(X)  # (n, 5)

        predictions = []
        confidences = []
        paths = []

        for i in range(len(X)):
            features = X[i]
            ml_class = int(np.argmax(proba[i]))
            ml_conf = float(proba[i][ml_class])

            h_score = compute_heuristic_score(features)
            h_class = heuristic_to_class(h_score, features)

            if ml_conf >= self.tau_high:
                # ── Path 1: High confidence -> Adopt ML result directly ──
                predictions.append(ml_class)
                confidences.append(ml_conf)
                paths.append('ml_high')

            elif ml_conf >= self.tau_low:
                # ── Path 2: Medium confidence -> Compare ML and heuristic ──
                if ml_class == h_class:
                    predictions.append(ml_class)
                    confidences.append(ml_conf)
                    paths.append('ml_agree')
                else:
                    # Disagreement: keep ML class, reduce confidence
                    predictions.append(ml_class)
                    confidences.append(0.50)
                    paths.append('ml_disagree')

            else:
                # ── Path 3: Low confidence -> Heuristic rule-based decision ──
                predictions.append(h_class)
                confidences.append(h_score / 100.0)
                paths.append('heuristic')

        return (
            np.array(predictions),
            np.array(confidences),
            paths,
        )

    def predict_with_threshold(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Final classification with detection threshold applied.
        Includes short block correction.

        Returns:
            final_predictions: changed to NON_CRYPTO(4) if below threshold
            confidences: confidence scores
        """
        predictions, confidences, paths = self.predict(X)

        for i in range(len(X)):
            features = X[i]
            score = compute_heuristic_score(features)

            # Short block check (8-15 bytes)
            if not check_short_block_threshold(features, score, self.sensitivity):
                predictions[i] = 4  # NON_CRYPTO
                confidences[i] = min(confidences[i], 0.30)

        return predictions, confidences

    def predict_detail(self, X: np.ndarray) -> list:
        """
        Returns detailed prediction results as a list of dictionaries (for debugging/reporting).
        """
        proba = self.ml_model.predict_proba(X)
        predictions, confidences, paths = self.predict(X)

        details = []
        for i in range(len(X)):
            features = X[i]
            h_score = compute_heuristic_score(features)

            details.append({
                'index': i,
                'ml_class': CLASS_NAMES[int(np.argmax(proba[i]))],
                'ml_confidence': float(np.max(proba[i])),
                'ml_proba': {CLASS_NAMES[j]: float(proba[i][j]) for j in range(5)},
                'heuristic_score': h_score,
                'heuristic_class': CLASS_NAMES[heuristic_to_class(h_score, features)],
                'final_class': CLASS_NAMES[predictions[i]],
                'final_confidence': float(confidences[i]),
                'decision_path': paths[i],
                'features': {
                    'entropy': features[0],
                    'chi2': features[1],
                    'length': features[2],
                    'is_key_len': bool(features[3]),
                    'is_iv_len': bool(features[4]),
                    'memory_region': int(features[5]),
                    'change_count': int(features[6]),
                    'pattern': int(features[7]),
                },
            })

        return details


# ─── Test when run standalone ───
if __name__ == '__main__':
    import joblib
    import os

    model_path = 'models/xgb_classifier.pkl'
    if not os.path.exists(model_path):
        print(f"Trained model not found: {model_path}")
        print("Please run model_trainer.py first.")
        exit(1)

    xgb_model = joblib.load(model_path)
    ensemble = HybridEnsemble(xgb_model, sensitivity='normal')

    # Test data (manually generated)
    test_features = np.array([
        # AES-256 key: high entropy, standard length, DLL
        [7.9, 250.0, 32.0, 1.0, 0.0, 1.0, 18.0, 3.0, 39.5, 1.0],
        # IV: medium entropy, IV length, heap
        [7.2, 0.0, 16.0, 1.0, 1.0, 2.0, 0.0, 0.0, 28.8, 0.0],
        # Plaintext: low entropy
        [4.2, 800.0, 64.0, 1.0, 0.0, 2.0, 0.0, 0.0, 17.6, 0.0],
    ])

    details = ensemble.predict_detail(test_features)
    for d in details:
        print(f"\n--- Sample {d['index']} ---")
        print(f"  ML:         {d['ml_class']} ({d['ml_confidence']:.1%})")
        print(f"  Heuristic:  {d['heuristic_class']} (score {d['heuristic_score']:.0f})")
        print(f"  Final decision:  {d['final_class']} ({d['final_confidence']:.1%})")
        print(f"  Path:       {d['decision_path']}")
