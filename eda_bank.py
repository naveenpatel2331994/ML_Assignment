#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

IN_PATH = Path("datasets/classification/bank-full.csv")
OUT_DIR = Path("reports/eda_bank")

def run_eda(in_path=IN_PATH, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading:", in_path)
    df = pd.read_csv(in_path, sep=";")

    # splits
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # numeric summary
    num_summary = df[numeric_cols].describe().T
    num_summary.to_csv(out_dir / "numeric_summary.csv")

    # categorical counts
    for c in cat_cols:
        vc = df[c].value_counts()
        vc.to_csv(out_dir / f"cat_counts_{c}.csv")

    # correlation
    corr = df[numeric_cols].corr()
    corr.to_csv(out_dir / "correlation_matrix.csv")

    # plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_heatmap.png", dpi=150)
    plt.close()

    # numeric histograms
    df[numeric_cols].hist(bins=30, figsize=(12, 8))
    plt.suptitle("Numeric Histograms")
    plt.tight_layout()
    plt.savefig(out_dir / "numeric_histograms.png", dpi=150)
    plt.close()

    # sample categorical barplots
    for c in ["job", "education", "marital", "poutcome", "contact"]:
        if c in df.columns:
            vc = df[c].value_counts().head(15)
            plt.figure(figsize=(8, 4))
            sns.barplot(x=vc.values, y=vc.index, palette="viridis")
            plt.title(f"Top values for {c}")
            plt.xlabel("Count")
            plt.tight_layout()
            plt.savefig(out_dir / f"bar_{c}.png", dpi=150)
            plt.close()

    # target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df, palette="pastel")
    plt.title("Target Distribution (y)")
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close()

    summary = {
        "shape": df.shape,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "target_counts": df["y"].value_counts().to_dict(),
        "top_job": df["job"].value_counts().idxmax() if "job" in df.columns else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("EDA outputs written to", out_dir)
    print("Summary:", summary)

if __name__ == "__main__":
    run_eda()
