"""Compute aggregate PTR (Particle Tracking Ratio) for tracker output CSVs.

PTR = STABLE / (STABLE + SWAP + LOST + MISSING), aggregated over all
(GT-particle, frame-transition) pairs.  Per-transition outcomes:
  STABLE  : same tracker_id maintained across N -> N+1
  SWAP    : different tracker_id in N+1
  LOST    : tracked in N, untracked in N+1
  MISSING : GT untracked in N at all
PTR counts SWAP/LOST/MISSING as failures (stricter than FFCAR, which
only penalises SWAP).  This module REUSES the canonical implementation
in motion_success_histogram.py so the definition stays identical to the
prior session's FINAL_REPORT numbers.

Assignment tolerance defaults to 5.0 px to match the evaluator's
MAX_MATCH_DIST used for all Phase-1/Phase-2 HOTA/AssA numbers.

Usage:
  # single cell:
  python -m compute_ptr --tracks <tracks.csv> --video PRIMARY --step 4
  # batch over a results tree (auto-detects video/step from path):
  python -m compute_ptr --batch <root_dir> --out <ptr_metrics.csv>
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_TRACKERS_DIR = _THIS.parent
_V2_DIR = _TRACKERS_DIR.parent.parent
if str(_TRACKERS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACKERS_DIR))

from stage2_evaluate_calibrated import load_calibration, GT_BY_BASE  # noqa: E402
from motion_success_histogram import (  # noqa: E402
    video_base, load_gt_pixel, assign_tracker_ids_per_frame, classify_transitions,
)

_CAL = None


def _cal():
    global _CAL
    if _CAL is None:
        _CAL = load_calibration()
    return _CAL


def compute_ptr(tracks_csv, video, step, tol_px=5.0):
    """Return dict with PTR + the four outcome counts for one tracker CSV.

    Uses the phase2c calibration transform for `video`'s base, the same
    GT->tracker Hungarian assignment as the histogram tool, and aggregates
    every (particle, transition) outcome into a single PTR.
    """
    base = video_base(video)
    cal = _cal()
    crow = cal.get(base)
    if crow is None:
        return None
    scale = float(crow["scale"]); ox = float(crow["ox"]); oy = float(crow["oy"])
    H = float(crow["H_video"])
    gt_path = GT_BY_BASE.get(base, "blob_data 3.txt")
    gt_by_frame = load_gt_pixel(gt_path, scale, ox, oy, H)
    if not gt_by_frame:
        return None
    df = pd.read_csv(tracks_csv)
    # radius for motion-in-diameters (PTR aggregate doesn't depend on it,
    # but classify_transitions needs a value); use median tracker r or 8px.
    radius_px = float(df["r"].median()) if "r" in df.columns and len(df) else 8.0
    assignments = assign_tracker_ids_per_frame(df, gt_by_frame, step, tol_px)
    transitions = classify_transitions(assignments, gt_by_frame, radius_px, step)
    counts = {"STABLE": 0, "SWAP": 0, "LOST": 0, "MISSING": 0}
    for t in transitions:
        counts[t["outcome"]] += 1
    total = sum(counts.values())
    ptr = counts["STABLE"] / total if total else float("nan")
    ffcar_like = (counts["STABLE"] / (counts["STABLE"] + counts["SWAP"])
                  if (counts["STABLE"] + counts["SWAP"]) else float("nan"))
    return {"video": video, "step": step, "ptr": ptr, "ffcar_like": ffcar_like,
            "n_transitions": total, **counts}


def _detect_video_step(path: str):
    """Infer (video, step) from a results path like .../CPTV_PKT0_PRIMARY_step4/..."""
    m = re.search(r"(PRIMARY|K0\d_[a-z_]+?|K0\d)_step(\d+)", path)
    if m:
        return m.group(1), int(m.group(2))
    # phase2 layout: .../<cond>/PRIMARY_step4_rep1/tracks.csv
    m = re.search(r"/([A-Za-z0-9_]+)_step(\d+)(?:_rep\d+)?/tracks\.csv", path.replace("\\", "/"))
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batch", default=None, help="root dir to glob */tracks.csv under")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol-px", type=float, default=5.0)
    args = ap.parse_args()

    rows = []
    if args.tracks:
        v = args.video; s = args.step
        if v is None or s is None:
            dv, ds = _detect_video_step(args.tracks)
            v = v or dv; s = s or ds
        r = compute_ptr(args.tracks, v, s, tol_px=args.tol_px)
        if r: rows.append({"tracks": args.tracks, **r})
    elif args.batch:
        for f in sorted(glob.glob(str(Path(args.batch) / "**" / "tracks.csv"), recursive=True)):
            v, s = _detect_video_step(f)
            if v is None:
                continue
            try:
                r = compute_ptr(f, v, s, tol_px=args.tol_px)
            except Exception as e:
                print(f"  ERR {f}: {e}")
                continue
            if r:
                rows.append({"tracks": f, **r})
                print(f"  {v:<24} step{s:<2}  PTR={r['ptr']:.4f}  "
                      f"(STABLE={r['STABLE']} SWAP={r['SWAP']} LOST={r['LOST']} MISSING={r['MISSING']})")
    else:
        ap.error("need --tracks or --batch")

    if args.out and rows:
        keys = ["tracks", "video", "step", "ptr", "ffcar_like", "n_transitions",
                "STABLE", "SWAP", "LOST", "MISSING"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nwrote {args.out} ({len(rows)} rows)")
    elif rows and not args.batch:
        for r in rows:
            print(f"PTR={r['ptr']:.4f} ffcar_like={r['ffcar_like']:.4f} "
                  f"n={r['n_transitions']} STABLE={r['STABLE']} SWAP={r['SWAP']} "
                  f"LOST={r['LOST']} MISSING={r['MISSING']}")


if __name__ == "__main__":
    main()
