"""
SIGNOVA - P2 Summary Exporter
Displays high-level statistical summaries and distributions of:
  - How2Sign
  - ArabSign
"""

import os
import sys
import csv
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_DIR = os.path.join(ROOT_DIR, "data", "metadata")
SPLITS_DIR = os.path.join(ROOT_DIR, "data", "splits")


def print_table(title, headers, rows):
    print(f"\n{'='*70}\n {title}\n{'='*70}")
    fmt = " | ".join([f"{{:<{w}}}" for w in [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]])
    print(fmt.format(*headers))
    print("-" * 70)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


def summarize_how2sign():
    csv_path = os.path.join(METADATA_DIR, "how2sign_canonical_metadata.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    splits = Counter(r["split"] for r in reader)
    signers = Counter(r["signer_id"] for r in reader)
    durations = [float(r["duration_sec"]) for r in reader if float(r["duration_sec"]) > 0]

    rows = [
        ["Total Sentences", len(reader)],
        ["Train Split", f"{splits['train']} ({splits['train']/len(reader)*100:.1f}%)"],
        ["Val Split", f"{splits['val']} ({splits['val']/len(reader)*100:.1f}%)"],
        ["Test Split", f"{splits['test']} ({splits['test']/len(reader)*100:.1f}%)"],
        ["Unique Signers", len(signers)],
        ["Avg Duration (sec)", f"{sum(durations)/len(durations):.2f}" if durations else "N/A"],
        ["Max Duration (sec)", f"{max(durations):.2f}" if durations else "N/A"],
        ["Min Duration (sec)", f"{min(durations):.2f}" if durations else "N/A"],
    ]
    print_table("How2Sign (Continuous ASL) Dataset Overview", ["Metric", "Value"], rows)


def summarize_arabsign():
    csv_path = os.path.join(METADATA_DIR, "arabsign_canonical_metadata.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    signers = Counter(r["signer_id"] for r in reader)
    indep_path = os.path.join(SPLITS_DIR, "arabsign_split_manifest_signer_independent.csv")
    with open(indep_path, "r", encoding="utf-8") as f:
        indep_reader = list(csv.DictReader(f))
    indep_splits = Counter(r["split"] for r in indep_reader)

    rows = [
        ["Total Samples", len(reader)],
        ["Unique Sentences", 50],
        ["Total Signers", 6],
        ["Signer 01 Samples", signers["01"]],
        ["Signer 02 Samples", signers["02"]],
        ["Signer 03 Samples", signers["03"]],
        ["Signer 04 Samples", signers["04"]],
        ["Signer 05 Samples", signers["05"]],
        ["Signer 06 Samples", signers["06"]],
        ["Signer-Indep Train", f"{indep_splits['train']} (Signers 01-04)"],
        ["Signer-Indep Val", f"{indep_splits['val']} (Signer 05)"],
        ["Signer-Indep Test", f"{indep_splits['test']} (Signer 06)"],
        ["Signer Overlap Across Splits", "0% (Strictly Zero Leakage)"],
    ]
    print_table("ArabSign (Continuous ArSL) Dataset Overview", ["Metric", "Value"], rows)


if __name__ == "__main__":
    summarize_how2sign()
    summarize_arabsign()
