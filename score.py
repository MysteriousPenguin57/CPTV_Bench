"""Score one tracks.csv on one CPTV_bench cell with the UNMODIFIED v2 evaluator snapshot.

    python score.py --cell PRIMARY_s4 --tracks path/to/tracks.csv [--out-dir score_out]

tracks.csv: columns frame,track_id,x,y,r,missed; frame = PROCESSED index of that cell's step (as written by
every P6 adapter); only missed == 0 rows are scored. Prints one line and appends it to <out-dir>/scores.csv.

What it does, exactly as the published pipeline did it (master_table_v4):
- `5px` cells (all but RES_B_4K): phase3_evaluator.main for one cell. Frames are remapped x step
  (phase3_evaluator._remap), evaluator.py runs with the cell's Phase-2C calibration converted to its forced
  transform (phase3_evaluator._run_eval), PTR = compute_ptr(tracks, video, step) at 5 px.
- `scalefair` cells (RES_B_4K): phase6e_b4k_scalefair.main for one cell. Track x, y, r and the calibration
  (scale, ox, oy, H) are halved, then remapped and evaluated as above; PTR = compute_ptr(native tracks, tol_px=10).
  phase6e imports through the `trackers` package (whose __init__ loads the research trackers), so the four lines
  of halving are repeated here instead of importing phase5_res_scalefair_eval.halve_tracks; the formatting is identical.
Only two things are adapted, at run time, in this wrapper: CPTV_CALIB_CSV points at the pack's calibration file,
and the modules' hard-coded interpreter path (PYTHON) is replaced by the running interpreter.
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent
SNAP = PACK / "v2_snapshot"


def load_cell(name):
    with open(PACK / "cells.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["cell"] == name:
                return r
    raise SystemExit(f"unknown cell {name!r}; see cells.csv (e.g. PRIMARY_s4)")


def halve(src, dst):
    """phase5_res_scalefair_eval.halve_tracks, line for line."""
    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        rd = csv.DictReader(fin)
        wr = csv.DictWriter(fout, fieldnames=rd.fieldnames or [])
        wr.writeheader()
        for r in rd:
            for k in ("x", "y", "r"):
                if k in r and r[k] not in (None, ""):
                    r[k] = f"{float(r[k]) / 2.0:.4f}"
            wr.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default="score_out")
    ap.add_argument("--no-ptr", action="store_true")
    a = ap.parse_args()
    c = load_cell(a.cell)
    video, step = c["video"], int(c["step"])

    # The calibration is read when stage2_evaluate_calibrated is imported: set it first.
    os.environ["CPTV_CALIB_CSV"] = str(SNAP / "detection_benchmark" / "results" / c["calib_group"]
                                       / "calibration_transforms_by_video.csv")
    sys.path.insert(0, str(SNAP / "src" / "trackers"))
    import stage2_evaluate_calibrated as s2
    import phase3_evaluator as p3
    from compute_ptr import compute_ptr
    s2.PYTHON = p3.PYTHON = sys.executable
    for m in (s2, p3):
        assert Path(m.__file__).resolve().is_relative_to(SNAP), m.__file__   # never a v2 fallback

    cal = s2.load_calibration()
    row = dict(cal[c["base"]])
    work = Path(a.out_dir) / a.cell
    if work.exists():
        shutil.rmtree(work)                      # the evaluator helpers cache eval.log; always start clean
    work.mkdir(parents=True)
    tracks = Path(a.tracks).resolve()
    scored = tracks
    if c["eval_mode"] == "scalefair":
        for k in ("scale", "ox", "oy", "scale_init", "ox_init", "oy_init"):
            row[k] = str(float(row[k]) / 2.0)
        row["H_video"] = str(float(row["H_video"]) / 2.0)
        scored = work / "tracks_halved.csv"
        halve(tracks, scored)
    txt_max = float(row["viewport_y_max"]); scale = float(row["scale"])
    ox = float(row["ox"]); oy = float(row["oy"]); H = float(row["H_video"])
    remapped = work / "tracks_remapped.csv"
    p3._remap(scored, remapped, step)
    m = p3._run_eval(remapped, work / "eval", c["gt"], txt_max,
                     [f"--eval-scale={scale}", f"--eval-offset-x={ox}", f"--eval-offset-y={H - oy - txt_max * scale}"])
    out = {"cell": a.cell, "eval_mode": c["eval_mode"],
           **{k: m.get(k) for k in ("hota", "assa", "f1", "idf1", "recall", "fragmentation")}}
    if not a.no_ptr:
        pr = compute_ptr(str(tracks), video, step, tol_px=10.0 if c["eval_mode"] == "scalefair" else 5.0)
        out.update({"ptr": pr["ptr"] if pr else None, **({k: pr[k] for k in ("STABLE", "SWAP", "LOST", "MISSING")} if pr else {})})
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()), flush=True)
    dst = Path(a.out_dir) / "scores.csv"
    new = not dst.exists()
    with open(dst, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out)); new and w.writeheader(); w.writerow(out)
    return 0 if m else 1


if __name__ == "__main__":
    sys.exit(main())
