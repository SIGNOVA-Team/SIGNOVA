# WLASL2000 Link Check Report

Preliminary availability check of the WLASL2000 video links, produced with `check_links.py`. WLASL100/300/1000 are subsets of WLASL2000 (same `video_id`s, selected by `label`), so only the 2000 set was checked and is reported here.

## What was checked

- **Web links (14,937 unique):** HTTP `HEAD` request (or a 1-byte ranged `GET` if `HEAD` is refused). Status 200/206 counts as `ok`; 404/410 counts as `dead`.
- **YouTube links (3,250 unique):** `yt-dlp --simulate` (asks YouTube whether the video is available, downloads nothing). Private, removed or unavailable videos count as `dead`.
- Links that could not be decided (rate limit, timeout) were marked `unchecked` and re-checked by re-running the script. The first run left 1,560 rows (all YouTube) `unchecked`; the second run re-checked 1,395 unique links and resolved all of them.

## Results

| Metric | Value |
|---|---|
| Unique links checked | 18,187 |
| Rows before the check | 20,976 |
| Dead rows removed | 7,191 (34.3%) |
| Rows kept | 13,785 |
| Train / Val / Test | 9,211 / 2,456 / 2,118 |
| Glosses with 0 examples overall | 0 |
| Glosses present in val | 1,697 / 2,000 |
| Glosses present in test | 1,649 / 2,000 |

## Important limitations

1. **`ok` means "the link responded at check time", not "the video downloads and opens".** Some `ok` links still fail when downloaded (a deleted-video page returned with status 200, a redirect to an error page, a YouTube video blocked in the downloader's country, `.swf` files that OpenCV cannot read).
2. **Val/test coverage is incomplete.** Because dead links were removed, about 300 glosses have no example in val and about 350 have none in test. Accuracy is therefore computed only over the glosses that are present.
3. **Links can die after the check.** Videos should be processed as soon as possible.

## Next step: keypoint extraction

Run `extract_keypoints.py` on `checked2000/wlasl2000_ok.csv`. It downloads each video, crops it with the row's bbox, trims it to `frame_start`/`frame_end`, extracts MediaPipe Holistic keypoints (pose + both hands, 258 values per frame), saves one `.npy` per row and deletes the video. It writes `wlasl{100,300,1000,2000}_{train,val,test}_kp.csv`, which are the splits the model should use. The numbers above will drop slightly after this step, and gloss coverage should be recomputed on those files.

## Notes for training

- Each row has `x1,y1,x2,y2` (bounding box) and `frame_start`/`frame_end` (1-indexed start; `-1` means until the last frame).
- Files in `checked2000/`: `wlasl2000_ok.csv` and its `train/val/test` splits, `wlasl2000_dead.csv`, per-link status in `url_status.csv`, and `summary.txt`.
