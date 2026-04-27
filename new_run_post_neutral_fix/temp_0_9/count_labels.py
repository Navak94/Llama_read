import pandas as pd
import os

FILES = [
    "migration_mode_1.csv",
    "migration_mode_2.csv",
    "migration_mode_3.csv",
    "migration_mode_4.csv"
]

LABELS = ["push","pull","mixed","neutral","unclear"]

results = []

for file in FILES:

    if not os.path.exists(file):
        print(f"Missing {file}")
        continue

    df = pd.read_csv(file)

    counts = df["pressure_final"].value_counts()

    row = {"file":file}

    total = len(df)

    for label in LABELS:
        row[label] = counts.get(label,0)

    row["total"] = total

    results.append(row)


summary = pd.DataFrame(results)

print("\nLabel counts:\n")
print(summary)

summary.to_csv("label_summary.csv",index=False)

print("\nSaved label_summary.csv")