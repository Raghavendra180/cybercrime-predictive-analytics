"""
Trains the withdrawal-likelihood classifier used by the dashboard's
Predictive Analytics Engine tab.

Run from anywhere:  python ml/train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cybercrime_data.csv"
MODEL_PATH = Path(__file__).resolve().parent / "cybercrime_model.pkl"

FEATURES = [
    "amount",
    "complaint_hour",
    "withdrawal_hour",
    "distance_km",
]


def main():
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data["withdrawal_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("================================")
    print("CYBERCRIME PREDICTION MODEL")
    print("================================")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC:        {auc:.3f}")

    for feat, importance in sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat:18} {importance:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
