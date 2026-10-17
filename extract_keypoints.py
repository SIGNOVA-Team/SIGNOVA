"""
WLASL -> keypoints. For every row: download the video, crop it with the row's bbox,
cut it to [frame_start, frame_end], run MediaPipe Holistic, save a .npy file,
then DELETE the video. Only the small .npy files stay on disk.

Install once (versions matter, newer mediapipe removed the API used here):
  py -m pip install mediapipe==0.10.21 "numpy<2" opencv-python==4.11.0.86 pandas requests yt-dlp

Run (run the 2000 file once; 100/300/1000 are produced from it automatically):
  py extract_keypoints.py --csv checked2000/wlasl2000_ok.csv --out_dir kp2000

Each .npy has shape (T, 258) float32 per frame:
  pose 33 x (x, y, z, visibility) = 132
  left hand 21 x (x, y, z)        = 63
  right hand 21 x (x, y, z)       = 63
Coordinates are normalised (0-1) relative to the bbox crop. Undetected parts are zeros.

Outputs in out_dir:
  keypoints/<video_id>.npy
  keypoints_log.csv                 status per row (resume-friendly)
  <name>_kp_verified.csv            rows with a usable keypoint file (+ kp_path, n_frames, hand_frac)
  <name>_{train,val,test}_kp.csv
  wlasl{100,300,1000,2000}_{train,val,test}_kp.csv   subsets by label (when the csv has 'label')
  preview/                          a few cropped first frames to eyeball that the bbox is right
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}
FEATURES = 258
_HOL = None


# ---------------------------------------------------------------- downloading
def is_youtube(url):
    return "youtube.com" in url or "youtu.be" in url


def get_web(url, path):
    r = requests.get(url, headers=UA, stream=True, timeout=30, allow_redirects=True)
    if r.status_code != 200:
        return f"http_{r.status_code}"
    if "text/html" in r.headers.get("Content-Type", "").lower():
        return "html_page"
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    return None


def get_youtube(url, path, height):
    fmt = f"best[height<={height}][ext=mp4]/best[height<={height}]/worst"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        fmt,
        "--no-playlist",
        "-q",
        "--socket-timeout",
        "30",
        "-o",
        path,
        url,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return None if p.returncode == 0 else "yt_dlp_failed"


# ----------------------------------------------------------------- keypoints
def holistic():
    global _HOL
    if _HOL is None:
        import mediapipe as mp

        _HOL = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )
    else:
        _HOL.reset()  # forget tracking state from the previous clip
    return _HOL


def frame_vec(res):
    pose = np.zeros((33, 4), np.float32)
    if res.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark],
            np.float32,
        )

    def hand(h):
        if h:
            return np.array([[lm.x, lm.y, lm.z] for lm in h.landmark], np.float32)
        return np.zeros((21, 3), np.float32)

    vec = np.concatenate(
        [
            pose.ravel(),
            hand(res.left_hand_landmarks).ravel(),
            hand(res.right_hand_landmarks).ravel(),
        ]
    )
    has_hand = bool(res.left_hand_landmarks or res.right_hand_landmarks)
    return vec, has_hand


def crop_box(frame, bbox):
    if bbox is None:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:  # bbox does not fit this video: use the full frame
        return frame
    return frame[y1:y2, x1:x2]


def extract_clip(path, start0, end0, bbox, min_frames, preview_path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0, "cannot_open"
    hol = holistic()
    for _ in range(start0):
        if not cap.grab():
            break
    vecs, hands, i = [], 0, start0
    while end0 < 0 or i < end0:
        ok, frame = cap.read()
        if not ok:
            break
        frame = crop_box(frame, bbox)
        if preview_path and not vecs:
            cv2.imwrite(preview_path, frame)
        vec, has_hand = frame_vec(hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        vecs.append(vec)
        hands += has_hand
        i += 1
    cap.release()
    if len(vecs) < min_frames:
        return None, 0.0, "too_few_frames"
    return np.stack(vecs), hands / len(vecs), None


# --------------------------------------------------------------------- worker
def do_url(url, rows, kp_dir, preview_dir, height, use_bbox, min_frames):
    """Download one url, extract every row that uses it, delete the video."""
    out = []
    tmp = tempfile.mkdtemp(prefix="wlasl_")
    path = os.path.join(tmp, hashlib.sha1(url.encode()).hexdigest()[:16] + ".mp4")
    try:
        err = get_youtube(url, path, height) if is_youtube(url) else get_web(url, path)
    except Exception as e:
        err = type(e).__name__
    if err or not os.path.exists(path) or os.path.getsize(path) == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return [(r["video_id"], f"failed:{err or 'empty'}", 0, 0.0) for r in rows]
    for r in rows:
        try:
            prev = os.path.join(preview_dir, r["video_id"] + ".jpg") if r["preview"] else None
            arr, hand_frac, err = extract_clip(
                path, r["start0"], r["end0"], r["bbox"] if use_bbox else None, min_frames, prev
            )
            if err:
                out.append((r["video_id"], f"failed:{err}", 0, 0.0))
                continue
            np.save(os.path.join(kp_dir, r["video_id"] + ".npy"), arr)
            out.append((r["video_id"], "ok", len(arr), round(hand_frac, 3)))
        except Exception as e:
            out.append((r["video_id"], f"failed:{type(e).__name__}", 0, 0.0))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# ----------------------------------------------------------------------- main
def save_log(log, path):
    pd.DataFrame(log.values(), columns=["video_id", "status", "n_frames", "hand_frac"]).to_csv(
        path, index=False
    )


def run(groups, workers, args, kp_dir, preview_dir, log, log_path):
    if not groups:
        return
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = [
            ex.submit(
                do_url,
                u,
                rows,
                kp_dir,
                preview_dir,
                args.max_height,
                not args.no_bbox,
                args.min_frames,
            )
            for u, rows in groups.items()
        ]
        for f in as_completed(futs):
            for rec in f.result():
                log[rec[0]] = rec
            done += 1
            if done % 25 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} links")
                save_log(log, log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="e.g. checked2000/wlasl2000_ok.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=4, help="parallel processes for normal links")
    ap.add_argument("--yt_workers", type=int, default=2, help="keep low, YouTube blocks heavy use")
    ap.add_argument(
        "--max_height",
        type=int,
        default=480,
        help="max YouTube video height (lower = less data, but check previews)",
    )
    ap.add_argument("--no_bbox", action="store_true", help="do not crop with the bbox columns")
    ap.add_argument("--min_frames", type=int, default=5)
    ap.add_argument("--retry_failed", action="store_true")
    ap.add_argument(
        "--limit", type=int, default=0, help="only process the first N links (for a quick test)"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"video_id": str})
    url_col = "video_url" if "video_url" in df.columns else "url"
    kp_dir = os.path.join(args.out_dir, "keypoints")
    preview_dir = os.path.join(args.out_dir, "preview")
    os.makedirs(kp_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "keypoints_log.csv")

    log = {}
    if os.path.exists(log_path):
        old = pd.read_csv(log_path, dtype={"video_id": str}).fillna("")
        log = {r.video_id: tuple(r) for r in old.itertuples(index=False)}

    def need(vid):
        if vid not in log:
            return True
        status = log[vid][1]
        if status == "ok":
            return not os.path.exists(os.path.join(kp_dir, vid + ".npy"))
        return args.retry_failed

    has_box = all(c in df.columns for c in ("x1", "y1", "x2", "y2"))
    n_prev = 0
    groups = {}
    for r in df.itertuples(index=False):
        vid = str(r.video_id)
        if not need(vid):
            continue
        fs = getattr(r, "frame_start", 1)
        fe = getattr(r, "frame_end", -1)
        fs = 1 if pd.isna(fs) else int(fs)
        fe = -1 if pd.isna(fe) else int(fe)
        item = {
            "video_id": vid,
            "start0": max(0, fs - 1),
            "end0": fe,
            "bbox": (r.x1, r.y1, r.x2, r.y2) if has_box else None,
            "preview": n_prev < 20,
        }
        n_prev += 1
        groups.setdefault(str(getattr(r, url_col)), []).append(item)

    urls = list(groups)
    if args.limit:
        urls = urls[: args.limit]
        groups = {u: groups[u] for u in urls}
    web = {u: g for u, g in groups.items() if not is_youtube(u)}
    yt = {u: g for u, g in groups.items() if is_youtube(u)}
    print(f"{len(df)} rows, {len(groups)} links to process ({len(web)} web, {len(yt)} youtube)")

    log_path_local = log_path
    print("web:")
    run(web, args.workers, args, kp_dir, preview_dir, log, log_path_local)
    print("youtube:")
    run(yt, args.yt_workers, args, kp_dir, preview_dir, log, log_path_local)
    save_log(log, log_path)

    # ---- final tables
    lg = pd.DataFrame(log.values(), columns=["video_id", "kp_status", "n_frames", "hand_frac"])
    m = df.merge(lg, on="video_id", how="left")
    m["kp_status"] = m["kp_status"].fillna("not_processed")
    m["kp_path"] = m["video_id"].map(lambda v: os.path.join(kp_dir, v + ".npy"))
    good = m[m["kp_status"] == "ok"].copy()
    name = os.path.splitext(os.path.basename(args.csv))[0].replace("_ok", "")
    good.to_csv(os.path.join(args.out_dir, f"{name}_kp_verified.csv"), index=False)

    print("\n--- summary ---")
    print(f"rows in: {len(m)}   keypoints ok: {len(good)}   removed: {len(m) - len(good)}")
    print("failure reasons:")
    print(m[m.kp_status != "ok"]["kp_status"].value_counts().to_string())
    if len(good):
        print(
            f"mean fraction of frames with a detected hand: {good.hand_frac.astype(float).mean():.2f}"
        )

    if "split" in m.columns:
        total = m.gloss.nunique()
        for s in ("train", "val", "test"):
            part = good[good.split == s]
            part.to_csv(os.path.join(args.out_dir, f"{name}_{s}_kp.csv"), index=False)
            print(f"{s}: {len(part)} rows, glosses present {part.gloss.nunique()} / {total}")
        if "label" in good.columns:
            for k in (100, 300, 1000, 2000):
                if k > m.gloss.nunique():
                    continue
                sub = good[good.label < k]
                for s in ("train", "val", "test"):
                    sub[sub.split == s].to_csv(
                        os.path.join(args.out_dir, f"wlasl{k}_{s}_kp.csv"), index=False
                    )
                print(
                    f"WLASL{k}: "
                    + ", ".join(
                        f"{s} {int((sub.split == s).sum())} rows / {sub[sub.split == s].gloss.nunique()} glosses"
                        for s in ("train", "val", "test")
                    )
                )


if __name__ == "__main__":
    main()
