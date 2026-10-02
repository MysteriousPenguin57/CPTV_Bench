"""Phase 6e: SCALE-FAIR eval for the RES_B_4K cells of the P6e all-tracker grid.

Same transform as phase5_res_scalefair_eval (halve tracker coords -> virtual
1072-space == 10 px tolerance at native scale; PTR via --tol-px 10 on native
coords) but over data/results/p6_baselines/runs_res/*. Writes
runs_res/b4k_scalefair_metrics.csv.

Usage (from v2/src): python -m trackers.phase6e_b4k_scalefair
[Phase 6g] --runs-root runs_p6g scores another root's RES_B_4K cells and
writes <root>/b4k_scalefair_metrics.csv (default runs_res = unchanged).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

_V2 = Path(__file__).resolve().parent.parent.parent
CAL_CSV = (_V2 / "detection_benchmark" / "results" / "calibration_res_scaling"
           / "calibration_transforms_by_video.csv")
_ap = argparse.ArgumentParser()
_ap.add_argument("--runs-root", default="runs_res")
_args, _ = _ap.parse_known_args()
RUNS = _V2 / "data" / "results" / "p6_baselines" / _args.runs_root
VIDEO = "RES_B_4K"

os.environ["CPTV_CALIB_CSV"] = str(CAL_CSV)
from trackers.phase5_eval_primary import eval_video  # noqa: E402
from trackers.compute_ptr import compute_ptr  # noqa: E402
from trackers.phase5_res_scalefair_eval import halve_tracks  # noqa: E402


def main():
    with open(CAL_CSV, newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["video"] == VIDEO)
    half = dict(row)
    for k in ("scale", "ox", "oy", "scale_init", "ox_init", "oy_init"):
        half[k] = str(float(row[k]) / 2.0)
    half["H_video"] = str(float(row["H_video"]) / 2.0)
    cal = {VIDEO: half}
    rows_out = []
    for cell in sorted(glob.glob(str(RUNS / "*" / f"{VIDEO}_step*_rep*"))):
        cell = Path(cell)
        tcsv = cell / "tracks.csv"
        if not tcsv.exists() or tcsv.stat().st_size < 10_000:
            continue
        step = int(cell.name.split("_step")[1].split("_")[0])
        rep = int(cell.name.split("_rep")[1])
        arm = cell.parent.name
        halved = cell / "tracks_halved.csv"
        halve_tracks(tcsv, halved)
        m, _ = eval_video(halved, cell / "eval_scalefair", cal, VIDEO, step=step)
        try:
            pr = compute_ptr(str(tcsv), VIDEO, step, tol_px=10.0)
        except Exception as e:
            print(f"  PTR failed {cell.name}: {e}", flush=True)
            pr = None
        try:
            halved.unlink()  # regenerable; keep disk lean
        except OSError:
            pass
        print(f"[b4k-sf] {arm:<13} s{step} rep{rep} HOTA={m.get('hota')} "
              f"PTR@10={(pr['ptr'] if pr else None)}", flush=True)
        rows_out.append({"condition": arm, "video": VIDEO, "step": step, "rep": rep,
                         **{k: m.get(k) for k in ("hota", "assa", "f1", "idf1", "recall")},
                         "ptr10": (pr["ptr"] if pr else None),
                         "swap10": (pr["SWAP"] if pr else None)})
    out = RUNS / "b4k_scalefair_metrics.csv"
    if rows_out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"[b4k-sf] wrote {out} ({len(rows_out)} rows)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
