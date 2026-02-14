#!/usr/bin/env python3
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IN_PATH = Path("datasets/classification/bank-full.csv")
OUT_DIR = Path("reports/models_baseline")
MODELS_DIR = OUT_DIR / "models"

def build_and_train(in_path=IN_PATH, out_dir=OUT_DIR, models_dir=MODELS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep=";")
    df["y_bin"] = (df["y"] == "yes").astype(int)

    numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    cat_features = [c for c in df.columns if c not in numeric_features + ["y", "y_bin"]]

    X = df[numeric_features + cat_features]
    y = df["y_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = StandardScaler()
    try:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", cat_transformer, cat_features),
        ],
        remainder="drop",
    )

    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42),
    }

    results = {}

    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        print("Training", name)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, models_dir / f"{name}.joblib")

        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            try:
                y_proba = pipe.decision_function(X_test)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            except Exception:
                y_proba = np.zeros_like(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else None
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc) if auc is not None else None,
            "confusion_matrix": cm,
        }

        if auc is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(out_dir / f"roc_{name}.png", dpi=150)
            plt.close()

    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Training complete. Models and results saved to", out_dir)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    build_and_train()
