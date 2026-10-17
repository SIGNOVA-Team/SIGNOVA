"""
WLASL metadata cleaning.

Input : WLASL_v0.3.json (from https://github.com/dxli94/WLASL/tree/master/start_kit)
Output: out_dir/
    wlasl_clean.csv        full cleaned table (one row per video instance)
    wlasl100.csv ...       top-K gloss subsets (K = 100, 300, 1000, 2000) with label ids
    label_map.csv          gloss <-> label id (full 2000 classes)
    cleaning_report.txt    what was checked / removed

Usage:
    python clean_metadata.py --json WLASL_v0.3.json --out_dir clean_meta
    # optional: only keep rows whose video was actually downloaded
    python clean_metadata.py --json WLASL_v0.3.json --out_dir clean_meta --videos_dir videos
"""

import argparse
import json
import os

import pandas as pd

SUBSET_SIZES = (100, 300, 1000, 2000)


def load_flat(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    # The JSON is ordered by gloss; keep that order, it is what defines the top-K subsets.
    for gloss_rank, entry in enumerate(data):
        for inst in entry["instances"]:
            rows.append({"gloss": entry["gloss"], "gloss_rank": gloss_rank, **inst})
    return pd.DataFrame(rows)


def clean(df, log):
    n0 = len(df)
    log.append(f"Loaded {n0} instances, {df.gloss.nunique()} glosses")

    # 1. Normalise text fields
    df["gloss"] = df["gloss"].str.strip().str.lower()
    df["source"] = df["source"].str.strip().str.lower()
    df["url"] = df["url"].str.strip()
    df["video_id"] = df["video_id"].astype(str).str.strip()

    # 2. Drop rows with missing essentials
    essential = ["gloss", "url", "video_id", "split", "bbox", "frame_start", "frame_end"]
    bad = df[essential].isna().any(axis=1) | (df["url"] == "")
    log.append(f"Dropped {int(bad.sum())} rows with missing essential fields")
    df = df[~bad].copy()

    # 3. Validate split names
    valid_splits = {"train", "val", "test"}
    bad = ~df["split"].isin(valid_splits)
    log.append(f"Dropped {int(bad.sum())} rows with invalid split")
    df = df[~bad].copy()

    # 4. Bounding boxes: (xmin, ymin, xmax, ymax), must be 4 ints with positive area and non-negative origin
    bbox = pd.DataFrame(df["bbox"].tolist(), index=df.index, columns=["x1", "y1", "x2", "y2"])
    bad = (bbox.x2 <= bbox.x1) | (bbox.y2 <= bbox.y1) | (bbox.x1 < 0) | (bbox.y1 < 0)
    log.append(f"Dropped {int(bad.sum())} rows with invalid bbox")
    df = pd.concat([df.drop(columns="bbox"), bbox], axis=1)[~bad.reindex(df.index).values].copy()

    # 5. Frame range: frame_start is 1-indexed, frame_end == -1 means "until last frame"
    df["full_video"] = df["frame_end"] == -1
    bad = (df["frame_start"] < 1) | ((~df["full_video"]) & (df["frame_end"] <= df["frame_start"]))
    log.append(f"Dropped {int(bad.sum())} rows with invalid frame range")
    df = df[~bad].copy()
    df["start0"] = df["frame_start"] - 1  # 0-indexed start (inclusive)
    df["end0"] = df["frame_end"]  # 0-indexed end (exclusive); -1 => until the end

    # 6. Exact duplicates: same source video AND same segment AND same gloss
    key = ["gloss", "url", "frame_start", "frame_end"]
    dup = df.duplicated(subset=key, keep="first")
    log.append(f"Dropped {int(dup.sum())} exact duplicate instances (same gloss + url + segment)")
    df = df[~dup].copy()

    # 7. Same url + segment but DIFFERENT gloss = ambiguous label -> flag only (report, don't drop)
    amb = df.duplicated(subset=["url", "frame_start", "frame_end"], keep=False) & ~df.duplicated(
        subset=key, keep=False
    )
    df["ambiguous_label"] = amb
    log.append(
        f"Flagged {int(amb.sum())} rows sharing url+segment with a different gloss (column: ambiguous_label)"
    )

    # 8. Helpful flags
    df["is_youtube"] = df["url"].str.contains("youtube|youtu.be", regex=True)
    df["is_swf"] = df["url"].str.lower().str.endswith(".swf")

    # 9. Labels (gloss_rank is the order in the original JSON == popularity order)
    df = df.sort_values(["gloss_rank", "instance_id"]).reset_index(drop=True)
    return df


def split_report(df, log):
    log.append(
        "Split sizes: " + ", ".join(f"{k}={v}" for k, v in df["split"].value_counts().items())
    )
    tiny = df.groupby("gloss")["split"].nunique()
    missing = tiny[tiny < 3]
    log.append(f"Glosses missing at least one of train/val/test: {len(missing)}")
    leak = df.groupby("url")["split"].nunique()
    leak = leak[leak > 1]
    log.append(f"Source URLs appearing in more than one split (possible leakage): {len(leak)}")
    signer_leak = df.groupby("signer_id")["split"].nunique()
    log.append(
        f"Signers appearing in more than one split: {int((signer_leak > 1).sum())} of {len(signer_leak)}"
    )


def attach_availability(df, videos_dir, log):
    exists = df["video_id"].map(lambda v: os.path.exists(os.path.join(videos_dir, v + ".mp4")))
    df["video_available"] = exists
    log.append(f"Videos found in '{videos_dir}': {int(exists.sum())} / {len(df)}")
    return df[exists].copy()


def make_subsets(df, out_dir, log):
    for k in SUBSET_SIZES:
        sub = df[df["gloss_rank"] < k].copy()
        # re-index labels 0..K-1 within the subset
        sub["label"] = sub["gloss_rank"]
        cols = [
            "video_id",
            "gloss",
            "label",
            "split",
            "signer_id",
            "source",
            "variation_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "frame_start",
            "frame_end",
            "fps",
            "url",
            "is_youtube",
            "is_swf",
            "ambiguous_label",
        ]
        sub[cols].to_csv(os.path.join(out_dir, f"wlasl{k}.csv"), index=False)
        log.append(f"WLASL{k}: {len(sub)} instances, {sub.gloss.nunique()} glosses -> wlasl{k}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out_dir", default="clean_meta")
    ap.add_argument(
        "--videos_dir", default=None, help="keep only rows whose <video_id>.mp4 exists here"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = []

    df = load_flat(args.json)
    df = clean(df, log)
    split_report(df, log)
    if args.videos_dir:
        df = attach_availability(df, args.videos_dir, log)

    df.to_csv(os.path.join(args.out_dir, "wlasl_clean.csv"), index=False)
    df[["gloss_rank", "gloss"]].drop_duplicates().rename(columns={"gloss_rank": "label"}).to_csv(
        os.path.join(args.out_dir, "label_map.csv"), index=False
    )
    make_subsets(df, args.out_dir, log)
    log.append(f"Final: {len(df)} instances, {df.gloss.nunique()} glosses")

    with open(os.path.join(args.out_dir, "cleaning_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
