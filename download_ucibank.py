#!/usr/bin/env python3
import os
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
OUT_DIR = Path("datasets/classification")

def download_and_unzip(url=UCI_URL, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "bank.zip"
    print(f"Downloading {url} -> {zip_path} ...")
    urlretrieve(url, zip_path)
    print("Download complete. Extracting...")
    with ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"Extracted to {out_dir}. Files:")
    for p in sorted(out_dir.iterdir()):
        print(" ", p.name)

if __name__ == "__main__":
    download_and_unzip()
