"""XGBoost classifier with time-ordered cross-validation and class weights."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from Analysis.model_data import get_data


def _weights(y):
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    return np.array([len(y) / (len(classes) * freq[value]) for value in y])


def main():
    X_train, X_test, y_train, y_test, _ = get_data()
    model = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=500,
                          learning_rate=0.03, max_depth=5, min_child_weight=5,
                          subsample=0.75, colsample_bytree=0.7, eval_metric="mlogloss",
                          random_state=42, n_jobs=-1)
    splitter = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, valid_idx in splitter.split(X_train):
        candidate = XGBClassifier(**model.get_params())
        candidate.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], sample_weight=_weights(y_train.iloc[train_idx]))
        scores.append(accuracy_score(y_train.iloc[valid_idx], candidate.predict(X_train.iloc[valid_idx])))
    print("Time-series CV accuracy:", np.mean(scores))
    model.fit(X_train, y_train, sample_weight=_weights(y_train))
    pred = model.predict(X_test)
    print("Holdout accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=["Loss", "Draw", "Win"]))


if __name__ == "__main__":
    main()
