
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from dataset_generator import generate_dataset
from feature_extraction import FEATURE_NAMES

MODELS_DIR = "models"


def evaluate(model, X_test, y_test):
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0
    latency_ms_per_url = (elapsed / len(X_test)) * 1000

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "false_positive_rate": fpr,
        "inference_latency_ms": latency_ms_per_url,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    print("Generating dataset...")
    df = generate_dataset(n=9000, seed=42)
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    fitted = {}
    for name, clf in candidates.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        fitted[name] = clf
        results[name] = evaluate(clf, X_test, y_test)
        print(f"  {name}: F1={results[name]['f1']:.4f}  Acc={results[name]['accuracy']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best_model, f"{MODELS_DIR}/model.pkl")

    metrics_out = {
        "best_model": best_name,
        "all_results": results,
        "feature_names": FEATURE_NAMES,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

     importances = dict(zip(FEATURE_NAMES, best_model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    with open(f"{MODELS_DIR}/feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    stats = {}
    for feat in FEATURE_NAMES:
        stats[feat] = {
            "legit_mean": float(df.loc[df.label == 0, feat].mean()),
            "legit_std": float(df.loc[df.label == 0, feat].std() + 1e-6),
            "phish_mean": float(df.loc[df.label == 1, feat].mean()),
            "phish_std": float(df.loc[df.label == 1, feat].std() + 1e-6),
        }
    with open(f"{MODELS_DIR}/train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved model + metrics + feature importances + train stats to models/")
    print(json.dumps(results[best_name], indent=2))


if __name__ == "__main__":
    main()
