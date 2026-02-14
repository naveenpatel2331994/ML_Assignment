#!/usr/bin/env python3
"""Train and evaluate six classifiers on the UCI Bank Marketing dataset.

Models:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest
- XGBoost

Saves models to `reports/models_all/models/` and metrics to
`reports/models_all/results_all.json`. ROC plots are saved per model.
"""
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
                             matthews_corrcoef, precision_score, recall_score, 
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

XGBOOST_AVAILABLE = False
XGBOOST_ERROR = None
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBClassifier = None
    XGBOOST_ERROR = str(e)[:150]


IN_PATH = Path("datasets/classification/bank-full.csv")
OUT_DIR = Path("reports/models_all")
MODELS_DIR = OUT_DIR / "models"


def interpret_mcc(mcc_value):
    """Interpret MCC score on a scale of -1 to +1."""
    if mcc_value >= 0.8:
        return "Excellent agreement (0.8+)"
    elif mcc_value >= 0.6:
        return "Strong agreement (0.6-0.8)"
    elif mcc_value >= 0.4:
        return "Moderate agreement (0.4-0.6)"
    elif mcc_value >= 0.2:
        return "Fair agreement (0.2-0.4)"
    elif mcc_value >= 0:
        return "Slight agreement (0-0.2)"
    else:
        return "Poor/Inverse agreement (<0)"


def build_preprocessor(df, numeric_features):
    # OneHotEncoder difference across sklearn versions handled later
    try:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = StandardScaler()

    cat_features = [c for c in df.columns if c not in numeric_features + ["y", "y_bin"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", cat_transformer, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor, cat_features


def train_and_eval():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH, sep=';')
    df['y_bin'] = (df['y'] == 'yes').astype(int)

    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    preprocessor, cat_features = build_preprocessor(df, numeric_features)

    X = df[numeric_features + cat_features]
    y = df['y_bin']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'gaussian_nb': GaussianNB(),
        'random_forest': RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42),
    }

    if XGBOOST_AVAILABLE:
        try:
            models['xgboost'] = XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                n_estimators=200, 
                random_state=42, 
                verbosity=0
            )
        except Exception as e:
            print(f"Warning: could not instantiate XGBoost: {e}")
    else:
        print(f"\n⚠️  XGBoost not available. Reason: {XGBOOST_ERROR}")
        print("To fix on macOS: brew install libomp")
        print("Then reinstall xgboost: pip install --user xgboost\n")

    results = {}

    for name, clf in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")

        # Predictions and probabilities
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, 'predict_proba'):
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
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else None
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'mcc': float(mcc),
            'roc_auc': float(auc) if auc is not None else None,
            'confusion_matrix': cm,
        }

        # ROC plot
        if auc is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(OUT_DIR / f'roc_{name}.png', dpi=150)
            plt.close()

    # Save results
    with open(OUT_DIR / 'results_all.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nAll models trained. Results written to', OUT_DIR)
    print('\n' + '='*70)
    print('RESULTS SUMMARY WITH MCC INTERPRETATION')
    print('='*70)
    print(json.dumps(results, indent=2))
    
    # Add MCC interpretation
    print('\n' + '='*70)
    print('MCC SCORE INTERPRETATION')
    print('='*70)
    print("(MCC: Matthews Correlation Coefficient, range -1 to +1)\n")
    
    sorted_by_mcc = sorted(results.items(), key=lambda x: x[1]['mcc'], reverse=True)
    for name, metrics in sorted_by_mcc:
        mcc_val = metrics['mcc']
        interp = interpret_mcc(mcc_val)
        print(f"{name:25} MCC: {mcc_val:7.4f} — {interp}")


if __name__ == '__main__':
    train_and_eval()
