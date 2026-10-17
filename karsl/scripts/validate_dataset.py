from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate():
    meta = pd.read_csv(ROOT / "data/metadata/karsl_metadata.csv", low_memory=False)
    raw = pd.read_csv(ROOT / "data/splits/karsl_split_manifest.csv", low_memory=False)
    key = pd.read_csv(ROOT / "data/metadata/keypoints_manifest.csv", low_memory=False)
    quality = pd.read_csv(ROOT / "data/reports/keypoints_quality.csv", low_memory=False)

    checks = {
        "Unique metadata IDs": meta.sample_id.is_unique,
        "Original sample count": len(meta) == 75514,
        "Raw benchmark size": raw.sample_id.nunique() == 75095,
        "Three raw folds": raw.fold.nunique() == 3,
        "Raw sample coverage": all(
            len(g) == 75095 and g.sample_id.is_unique for _, g in raw.groupby("fold")
        ),
        "Raw signer independence": all(
            g.groupby("signer_id").split.nunique().max() == 1 for _, g in raw.groupby("fold")
        ),
        "Raw class coverage": all(
            g.sign_id.nunique() == 500 for _, g in raw.groupby(["fold", "split"])
        ),
        "Keypoint benchmark size": key.sample_id.nunique() == 8793,
        "Three keypoint folds": key.fold.nunique() == 3,
        "Keypoint class coverage": all(
            g.class_label.nunique() == 62 for _, g in key.groupby(["fold", "split"])
        ),
        "Keypoint signer independence": all(
            g.groupby("signer_id").split.nunique().max() == 1 for _, g in key.groupby("fold")
        ),
        "Keypoint quality": (
            len(quality) == 8793
            and quality.sample_id.is_unique
            and quality.status.eq("PASS").all()
            and set(quality.sample_id) == set(key.sample_id)
        ),
        "Portable keypoint paths": all(
            not key[c].astype(str).str.startswith(("/", "C:\\")).any()
            for c in ["lh_path", "pose_path", "rh_path"]
        ),
    }

    return checks


if __name__ == "__main__":
    results = validate()
    print("=== DATASET VALIDATION ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("Overall:", "PASS" if all(results.values()) else "FAIL")
    if not all(results.values()):
        raise SystemExit(1)
