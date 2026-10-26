import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


SHAPE = (32, 150)
SPLITS = ("train", "validation", "test")
AUDIT_COLUMNS = [
    "pose_file", "sign_id", "signer_id", "sample_id", "split",
    "previously_evaluated_test", "status", "review_flags", "error_type",
    "error", "size_bytes", "raw_sha256", "feature_sha256", "frames",
    "visible_hand_fraction",
]


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_preprocessing(repo):
    path = Path(repo) / "karsl/ml/src/preprocessing.py"
    specification = importlib.util.spec_from_file_location("signova_frozen_preprocessing", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if (module.NUM_FRAMES, module.INPUT_DIM) != SHAPE:
        raise ValueError("The frozen preprocessing shape changed.")
    return module, path


def integer_column(frame, name):
    values = pd.to_numeric(frame[name], errors="raise")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Invalid numeric column: {name}")
    if not values.eq(np.floor(values)).all():
        raise ValueError(f"Non-integer identifiers: {name}")
    frame[name] = values.astype("int64")


def load_manifest(metadata_path, folds_path, fold, historical_test_cache):
    metadata = pd.DataFrame(json.loads(Path(metadata_path).read_text(encoding="utf-8-sig")))
    folds = pd.read_csv(folds_path)
    folds = folds.loc[folds["fold"].eq(fold)].copy()
    keys = ["pose_file", "sign_id", "signer_id", "sample_id"]
    for frame in (metadata, folds):
        if not set(keys).issubset(frame.columns) or frame[keys].isna().any().any():
            raise ValueError("Missing source or manifest identifiers.")
        if frame["pose_file"].duplicated().any() or frame.empty:
            raise ValueError("Empty or duplicated source/manifest paths.")
        for column in keys[1:]:
            integer_column(frame, column)
    joined = folds[keys].merge(metadata[keys], on="pose_file", how="outer", suffixes=("_fold", "_source"), indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError("Source and frozen manifest contain different samples.")
    for column in keys[1:]:
        if not joined[f"{column}_fold"].eq(joined[f"{column}_source"]).all():
            raise ValueError(f"Frozen/source identifier mismatch: {column}")
    if set(folds["split"]) != set(SPLITS):
        raise ValueError("Expected train, validation and test splits.")
    signers = {split: set(folds.loc[folds["split"].eq(split), "signer_id"]) for split in SPLITS}
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if signers[split] & signers[other]:
                raise ValueError(f"Signer overlap: {split}/{other}")
    for relative in folds["pose_file"]:
        path = PurePosixPath(str(relative))
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "pose_dir" or path.suffix != ".pose":
            raise ValueError(f"Invalid relative Pose path: {relative}")
    integer_column(metadata, "num_frames")
    rows = folds[keys + ["split"]].merge(metadata[["pose_file", "num_frames"]], on="pose_file", validate="one_to_one")
    rows = rows.sort_values(["split", "sign_id", "signer_id", "sample_id", "pose_file"], kind="stable").reset_index(drop=True)
    old_paths = set()
    if historical_test_cache is not None:
        with np.load(historical_test_cache, allow_pickle=False) as cache:
            old_paths = set(cache["paths"].astype(str).tolist())
        if not old_paths.issubset(set(rows.loc[rows["split"].eq("test"), "pose_file"])):
            raise ValueError("Historical test samples do not match the frozen test split.")
    rows["previously_evaluated_test"] = rows["split"].eq("test") & rows["pose_file"].isin(old_paths)
    return rows


def verify_parity(reference_cache, pose_root, preprocess, decode, version):
    with np.load(reference_cache, allow_pickle=False) as cache:
        if str(cache["preprocessing_version"].item()) != version or not cache["done"].all():
            raise ValueError("Reference feature cache is incomplete or incompatible.")
        features = cache["X"]
        paths = cache["paths"].astype(str)
        indices = np.unique(np.linspace(0, len(paths) - 1, min(8, len(paths)), dtype=int))
        records = []
        for index in indices:
            raw = (pose_root / paths[index]).read_bytes()
            actual = preprocess(decode(raw))
            np.testing.assert_allclose(actual, features[index], atol=1e-6, rtol=1e-6)
            records.append({"pose_file": paths[index], "maximum_absolute_difference": float(np.max(np.abs(actual - features[index])))})
    return records


def process_sample(row, root, preprocess, decode):
    audit = {key: row[key] for key in ("pose_file", "sign_id", "signer_id", "sample_id", "split", "previously_evaluated_test")}
    audit.update(status="FAILED", review_flags="", error_type="", error="", size_bytes=0, raw_sha256="", feature_sha256="", frames=0, visible_hand_fraction=0.0)
    try:
        raw = (root / row["pose_file"]).read_bytes()
        audit["size_bytes"] = len(raw)
        audit["raw_sha256"] = digest_bytes(raw)
        if not raw:
            raise ValueError("Empty Pose file")
        pose = decode(raw)
        frames = len(pose.body.data)
        audit["frames"] = frames
        if frames != row["num_frames"]:
            raise ValueError(f"Frame count mismatch: observed {frames}, metadata {row['num_frames']}")
        features = preprocess(pose)
        if features.shape != SHAPE or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("Invalid preprocessed feature tensor")
        coordinates = np.asarray(pose.body.data)[:, 0, 501:543, :2]
        confidence = np.asarray(pose.body.confidence)[:, 0, 501:543]
        visible = (confidence > 0.5) & np.isfinite(coordinates).all(axis=2)
        audit["visible_hand_fraction"] = float(visible.mean())
        audit["status"] = "PASS" if visible.any() else "REVIEW"
        audit["review_flags"] = "" if visible.any() else "no_visible_hand_landmarks"
        audit["feature_sha256"] = digest_bytes(features.tobytes(order="C"))
        return audit, features
    except (MemoryError, RecursionError):
        raise
    except Exception as error:
        audit["error_type"] = type(error).__name__
        audit["error"] = str(error)[:500]
        return audit, None


def write_shard(path, results, fingerprint, version, split):
    audits = [audit for audit, features in results]
    prepared = [(audit, features) for audit, features in results if features is not None]
    features = np.stack([features for audit, features in prepared]) if prepared else np.empty((0, *SHAPE), dtype=np.float32)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, X=features,
            labels=np.asarray([audit["sign_id"] for audit, features in prepared], dtype=np.int64),
            paths=np.asarray([audit["pose_file"] for audit, features in prepared], dtype=str),
            preprocessing_version=np.asarray(version), split=np.asarray(split),
            fingerprint=np.asarray(fingerprint), audit_json=np.asarray(json.dumps(audits, ensure_ascii=False)),
        )
    temporary.replace(path)


def read_shard(path, rows, fingerprint, version, split):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"].item()) != fingerprint or str(saved["preprocessing_version"].item()) != version or str(saved["split"].item()) != split:
            raise ValueError(f"Shard identity mismatch: {path.name}")
        audits = json.loads(str(saved["audit_json"].item()))
        if len(audits) != len(rows):
            raise ValueError(f"Shard row count mismatch: {path.name}")
        for audit, row in zip(audits, rows):
            for key in ("pose_file", "sign_id", "signer_id", "sample_id", "split", "previously_evaluated_test"):
                if audit[key] != row[key]:
                    raise ValueError(f"Shard row mismatch: {path.name}/{key}")
            if audit["status"] not in {"PASS", "REVIEW", "FAILED"}:
                raise ValueError("Unknown QC status")
        prepared = [audit for audit in audits if audit["status"] != "FAILED"]
        features = saved["X"]
        if features.shape != (len(prepared), *SHAPE) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError(f"Invalid cached features: {path.name}")
        if saved["paths"].astype(str).tolist() != [audit["pose_file"] for audit in prepared] or saved["labels"].tolist() != [audit["sign_id"] for audit in prepared]:
            raise ValueError(f"Cached labels/paths mismatch: {path.name}")
        for audit, feature in zip(prepared, features):
            if digest_bytes(feature.tobytes(order="C")) != audit["feature_sha256"]:
                raise ValueError(f"Feature checksum mismatch: {path.name}")
    return audits


def prepare_shards(rows, root, output, preprocess, decode, fingerprint, version, chunk_size, workers, retry_failed):
    all_audits = []
    shard_records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split in SPLITS:
            subset = rows.loc[rows["split"].eq(split)].to_dict("records")
            directory = output / split
            directory.mkdir(exist_ok=True)
            for start in range(0, len(subset), chunk_size):
                batch = subset[start:start + chunk_size]
                path = directory / f"part_{start // chunk_size:05d}.npz"
                audits = None
                reused = False
                if path.exists():
                    audits = read_shard(path, batch, fingerprint, version, split)
                    reused = not (retry_failed and any(audit["status"] == "FAILED" for audit in audits))
                if not reused:
                    results = list(executor.map(lambda row: process_sample(row, root, preprocess, decode), batch))
                    write_shard(path, results, fingerprint, version, split)
                    audits = read_shard(path, batch, fingerprint, version, split)
                all_audits.extend(audits)
                shard_records.append({"path": path.relative_to(output).as_posix(), "split": split, "expected_samples": len(batch), "prepared_samples": sum(audit["status"] != "FAILED" for audit in audits), "sha256": digest_file(path)})
                failed = sum(audit["status"] == "FAILED" for audit in all_audits)
                review = sum(audit["status"] == "REVIEW" for audit in all_audits)
                progress = {"processed_samples": len(all_audits), "expected_samples": len(rows), "failed_samples": failed, "review_samples": review, "last_shard": shard_records[-1]["path"]}
                write_json(output / "progress.json", progress)
                print(f"{split}: {min(start + chunk_size, len(subset))}/{len(subset)} | total={len(all_audits)}/{len(rows)} | failed={failed} | review={review} | {'reused' if reused else 'saved'}", flush=True)
    return pd.DataFrame(all_audits, columns=AUDIT_COLUMNS), shard_records


def duplicate_report(audit, digest_column):
    available = audit.loc[audit[digest_column].ne("")].copy()
    duplicates = available.loc[available[digest_column].duplicated(keep=False)].copy()
    duplicates["cross_split"] = duplicates.groupby(digest_column)["split"].transform("nunique").gt(1)
    duplicates["label_conflict"] = duplicates.groupby(digest_column)["sign_id"].transform("nunique").gt(1)
    return duplicates.sort_values([digest_column, "split", "pose_file"])


def main():
    parser = argparse.ArgumentParser()
    for argument in ("repo-root", "metadata", "folds", "download-report", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 4096 or not 1 <= args.workers <= 8:
        parser.error("Use chunk-size 1..4096 and workers 1..8.")
    from pose_format import Pose
    preprocess, preprocessing_path = load_preprocessing(args.repo_root)
    download = json.loads(args.download_report.read_text(encoding="utf-8"))
    if download["status"] != "complete" or download["dataset_handle"] != "tfmohamedyahia/kars502-pose-75/versions/1":
        raise ValueError("Expected the complete pinned version-1 download report.")
    pose_root = Path(download["pose_root"])
    if not (pose_root / "pose_dir").is_dir():
        raise FileNotFoundError("Pose data is not available in this runtime. Restore the dataset first.")
    private = args.repo_root / "karsl/ml/artifacts/private"
    reference = private / "pose_30_classes_features_v2.npz"
    historical_test = private / "pose_30_classes_test_v1.npz"
    rows = load_manifest(args.metadata, args.folds, args.fold, historical_test)
    classes = sorted(map(int, rows["sign_id"].unique()))
    if len(classes) != 502 or len(rows) != download["expected_files"]:
        raise ValueError("Unexpected full-dataset class coverage or sample count.")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1, "dataset_handle": download["dataset_handle"], "fold": args.fold,
        "metadata_sha256": digest_file(args.metadata), "folds_sha256": digest_file(args.folds),
        "preprocessing_sha256": digest_file(preprocessing_path), "preprocessing_version": preprocess.PREPROCESSING_VERSION,
        "script_sha256": digest_file(__file__), "reference_cache_sha256": digest_file(reference),
        "historical_test_cache_sha256": digest_file(historical_test), "chunk_size": args.chunk_size,
        "feature_shape": list(SHAPE), "classes": classes, "numpy_version": np.__version__,
        "pose_format_version": importlib.metadata.version("pose-format"),
    }
    fingerprint = digest_bytes(json.dumps(identity, sort_keys=True).encode())
    plan = {"fingerprint": fingerprint, "identity": identity, "expected_samples": len(rows), "label_semantics_reviewed": False, "quality_rules": {"FAILED": "Unreadable/empty data, metadata frame mismatch, invalid structure or shared preprocessing failure", "REVIEW": "Readable features but no hand landmark with finite x/y and confidence above 0.5 anywhere in the sequence", "PASS": "Technical preprocessing checks passed; this is not semantic label approval", "duplicates": "Reported for review; no automatic deletion or split reassignment"}}
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing feature cache belongs to different inputs/code/settings. Use a new output folder.")
    write_json(plan_path, plan)
    write_csv(args.output / "frozen_processing_manifest.csv", rows)
    parity = verify_parity(reference, pose_root, preprocess.preprocess_pose, Pose.read, preprocess.PREPROCESSING_VERSION)
    write_json(args.output / "preprocessing_parity.json", {"status": "passed", "samples": parity})
    print(f"Shared preprocessing parity: PASS ({len(parity)} reference samples)", flush=True)
    print(f"Samples: {len(rows)} | Classes: {len(classes)} | Workers: {args.workers} | Chunk size: {args.chunk_size}", flush=True)
    print(f"Persistent feature directory: {args.output}", flush=True)
    started = time.perf_counter()
    audit, shards = prepare_shards(rows, pose_root, args.output, preprocess.preprocess_pose, Pose.read, fingerprint, preprocess.PREPROCESSING_VERSION, args.chunk_size, args.workers, args.retry_failed)
    raw_duplicates = duplicate_report(audit, "raw_sha256")
    feature_duplicates = duplicate_report(audit, "feature_sha256")
    distribution = audit.groupby(["split", "sign_id", "status"]).size().unstack("status", fill_value=0).reset_index()
    for status in ("PASS", "REVIEW", "FAILED"):
        if status not in distribution:
            distribution[status] = 0
    write_csv(args.output / "sample_qc.csv", audit)
    write_csv(args.output / "class_split_qc.csv", distribution)
    write_csv(args.output / "raw_duplicates.csv", raw_duplicates)
    write_csv(args.output / "feature_duplicates.csv", feature_duplicates)
    write_json(args.output / "shards.json", {"fingerprint": fingerprint, "shards": shards})
    failed = int(audit["status"].eq("FAILED").sum())
    review = int(audit["status"].eq("REVIEW").sum())
    cross_split = int(raw_duplicates.loc[raw_duplicates["cross_split"], "raw_sha256"].nunique())
    label_conflicts = int(raw_duplicates.loc[raw_duplicates["label_conflict"], "raw_sha256"].nunique())
    report = {
        "status": "COMPLETED_WITH_REVIEW_ITEMS" if failed or review or cross_split or label_conflicts else "FEATURE_PREPARATION_COMPLETED",
        "expected_samples": len(rows), "processed_samples": len(audit), "prepared_samples": int(audit["status"].ne("FAILED").sum()),
        "technical_pass_samples": int(audit["status"].eq("PASS").sum()), "review_samples": review, "failed_samples": failed,
        "source_classes": len(classes), "prepared_classes": int(audit.loc[audit["status"].ne("FAILED"), "sign_id"].nunique()),
        "class_split_cells_without_pass_samples": int(distribution["PASS"].eq(0).sum()),
        "raw_duplicate_groups": int(raw_duplicates["raw_sha256"].nunique()), "raw_cross_split_duplicate_groups": cross_split,
        "raw_conflicting_label_groups": label_conflicts, "identical_feature_groups": int(feature_duplicates["feature_sha256"].nunique()),
        "historically_evaluated_test_samples": int(audit["previously_evaluated_test"].sum()),
        "feature_shape": list(SHAPE), "shards": len(shards), "fingerprint": fingerprint,
        "run_seconds_including_resumed_shard_checks": round(time.perf_counter() - started, 2),
        "python": platform.python_version(), "numpy": np.__version__, "pose_format": identity["pose_format_version"],
        "training_performed": False, "test_performance_evaluated": False, "labels_semantically_verified": False,
    }
    write_json(args.output / "preprocessing_report.json", report)
    print("\n=== FEATURE PREPARATION RESULT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("Report:", args.output / "preprocessing_report.json", flush=True)
    print("Status:", report["status"], flush=True)


if __name__ == "__main__":
    main()
