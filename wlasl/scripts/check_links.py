"""
Check which WLASL video links still work, WITHOUT downloading the videos, and
produce filtered CSVs that contain only the working ones.

  - normal links (mp4 / swf ...): HEAD request, or a 1-byte ranged GET if HEAD is refused
  - YouTube links: `yt-dlp --simulate` (asks YouTube if the video is available, downloads nothing)

Install once:   pip install requests yt-dlp pandas
Run (example):  python check_links.py --csv wlasl100/wlasl100_all.csv --out_dir checked100

Outputs in out_dir:
  url_status.csv            one row per unique url: ok / dead / unchecked  (+ detail)
  <name>_ok.csv             rows whose link is NOT dead  (use this one for the model)
  <name>_dead.csv           rows whose link is dead
  <name>_{train,val,test}_ok.csv   the same, split by train/val/test
  summary.txt               counts + glosses that lost examples

The script can be stopped and re-run: urls already in url_status.csv are not checked again.
"""

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (WLASL link checker)"}


def is_youtube(url):
    return "youtube.com" in url or "youtu.be" in url


def check_http(url, timeout):
    """Return (status, detail). status in ok / dead."""
    last = ""
    for _attempt in range(2):  # retry once so a network hiccup is not counted as a dead link
        try:
            r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if r.status_code in (403, 405, 501) or r.status_code >= 500:
                # many servers refuse HEAD: ask for the first byte only
                h = dict(HEADERS, Range="bytes=0-0")
                r = requests.get(url, headers=h, timeout=timeout, allow_redirects=True, stream=True)
                r.close()
            if r.status_code in (200, 206):
                return "ok", str(r.status_code)
            last = f"http {r.status_code}"
            if r.status_code in (404, 410):  # clearly gone, no need to retry
                return "dead", last
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(1)
    return "dead", last


def check_youtube(url, timeout):
    if shutil.which("yt-dlp") is None:
        return "unchecked", "yt-dlp not installed"
    try:
        p = subprocess.run(
            ["yt-dlp", "--simulate", "--quiet", "--no-warnings", url],
            capture_output=True,
            text=True,
            timeout=timeout * 4,
        )
    except subprocess.TimeoutExpired:
        return "unchecked", "timeout"
    if p.returncode == 0:
        return "ok", "available"
    err = (p.stderr or "").strip().splitlines()
    msg = err[-1][:120] if err else "error"
    low = msg.lower()
    if any(
        s in low
        for s in (
            "unavailable",
            "private",
            "removed",
            "terminated",
            "not available",
            "does not exist",
        )
    ):
        return "dead", msg
    return "unchecked", msg  # e.g. rate limit / bot check: not proof the video is gone


def check_one(args):
    url, timeout = args
    if is_youtube(url):
        s, d = check_youtube(url, timeout)
    else:
        s, d = check_http(url, timeout)
    return url, s, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=16, help="parallel checks for normal links")
    ap.add_argument(
        "--yt_workers", type=int, default=3, help="parallel checks for YouTube (keep low)"
    )
    ap.add_argument("--timeout", type=int, default=15)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv, dtype={"video_id": str})
    col = "video_url" if "video_url" in df.columns else "url"
    name = os.path.splitext(os.path.basename(a.csv))[0].replace("_all", "")

    status_path = os.path.join(a.out_dir, "url_status.csv")
    done = (
        pd.read_csv(status_path)
        if os.path.exists(status_path)
        else pd.DataFrame(columns=["url", "status", "detail"])
    )
    # re-check "unchecked" ones on re-runs; keep final ok/dead answers
    done = done[done.status.isin(["ok", "dead"])]
    todo = [u for u in df[col].unique() if u not in set(done.url)]
    web = [(u, a.timeout) for u in todo if not is_youtube(u)]
    yt = [(u, a.timeout) for u in todo if is_youtube(u)]
    print(
        f"{len(df[col].unique())} unique links, {len(todo)} to check ({len(web)} web, {len(yt)} youtube)"
    )

    results = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(check_one, web), 1):
            results.append(r)
            if i % 500 == 0:
                print(f"  web {i}/{len(web)}")
    with ThreadPoolExecutor(a.yt_workers) as ex:
        for i, r in enumerate(ex.map(check_one, yt), 1):
            results.append(r)
            if i % 100 == 0:
                print(f"  youtube {i}/{len(yt)}")

    status = pd.concat(
        [done, pd.DataFrame(results, columns=["url", "status", "detail"])], ignore_index=True
    )
    status.to_csv(status_path, index=False)

    m = df.merge(
        status.rename(columns={"url": col, "status": "link_status", "detail": "link_detail"}),
        on=col,
        how="left",
    )
    m["link_status"] = m["link_status"].fillna("unchecked")
    keep = m[m.link_status != "dead"]  # ok + unchecked (not proven dead)
    dead = m[m.link_status == "dead"]
    keep.to_csv(os.path.join(a.out_dir, f"{name}_ok.csv"), index=False)
    dead.to_csv(os.path.join(a.out_dir, f"{name}_dead.csv"), index=False)
    for s in ("train", "val", "test"):
        keep[keep.split == s].to_csv(os.path.join(a.out_dir, f"{name}_{s}_ok.csv"), index=False)

    lines = [
        f"rows total: {len(m)}",
        f"link status counts: {m.link_status.value_counts().to_dict()}",
        f"kept (ok + unchecked): {len(keep)}   removed (dead): {len(dead)}",
        f"kept per split: {keep.split.value_counts().to_dict()}",
    ]
    before = m.groupby("gloss").size()
    after = keep.groupby("gloss").size().reindex(before.index).fillna(0)
    lost_all = before[after == 0]
    lines.append(f"glosses left with 0 examples: {len(lost_all)}")
    for s in ("val", "test"):
        has = keep[keep.split == s].gloss.nunique()
        lines.append(f"glosses present in {s}: {has} / {m.gloss.nunique()}")
    txt = "\n".join(lines)
    open(os.path.join(a.out_dir, "summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
