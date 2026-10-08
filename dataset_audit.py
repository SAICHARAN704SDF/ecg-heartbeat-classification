import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/charankailasa/Downloads/archive")

files = {
    "MIT-BIH Train": "mitbih_train.csv",
    "MIT-BIH Test": "mitbih_test.csv",
    "PTB Normal": "ptbdb_normal.csv",
    "PTB Abnormal": "ptbdb_abnormal.csv",
}

print("=" * 70)
print("ECG DATASET AUDIT")
print("=" * 70)

for name, filename in files.items():
    path = DATA_DIR / filename

    df = pd.read_csv(path, header=None)

    print(f"\n{name}")
    print("-" * 50)
    print("Shape:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("\nColumn count:", len(df.columns))

    # Last column = label
    labels = df.iloc[:, -1]

    print("Labels:")
    print(labels.value_counts().sort_index())

    print("\nLabel percentages:")
    print((labels.value_counts(normalize=True).sort_index() * 100).round(3))

    # Signal statistics
    X = df.iloc[:, :-1].values

    print("\nSignal statistics:")
    print("Min:", X.min())
    print("Max:", X.max())
    print("Mean:", X.mean())
    print("Std:", X.std())

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)