"""res_scaling: SCALE-FAIR eval for the RES_B_4K cells.

At 27 px diameter the detector localizes every particle (F1@10px 0.999) but
the px-fixed 5 px evaluator tolerance floors HOTA for BOTH linkers (~0.1-0.16,
F1 0.20). Fair comparison at matched RELATIVE tolerance = halve all tracker
coordinates and evaluate in the virtual 1072-space (exactly equivalent to a
10 px tolerance at native scale): px'=px/2 -> H'=H/2, s'=s/2, ox'=ox/2,
oy'=oy/2 under the y-flip transform py = H - (y*s + oy). PTR scale-fair via
compute_ptr --tol-px 10 on the NATIVE coords.

Writes eval to <cell>/eval_scalefair/ and prints a table.
Usage (from v2/src): python -m trackers.phase5_res_scalefair_eval
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

_V2 = Path(__file__).resolve().parent.parent.parent
CAL_CSV = (_V2 / "detection_benchmark" / "results" / "calibration_res_scaling"
           / "calibration_transforms_by_video.csv")
RUNS = _V2 / "data" / "results" / "p5" / "res_scaling" / "runs"
VIDEO = "RES_B_4K"

os.environ["CPTV_CALIB_CSV"] = str(CAL_CSV)
from trackers.phase5_eval_primary import eval_video  # noqa: E402
from trackers.compute_ptr import compute_ptr  # noqa: E402


def halve_tracks(src: Path, dst: Path):
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
    with open(CAL_CSV, newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["video"] == VIDEO)
    half = dict(row)
    for k in ("scale", "ox", "oy", "scale_init", "ox_init", "oy_init"):
        half[k] = str(float(row[k]) / 2.0)
    half["H_video"] = str(float(row["H_video"]) / 2.0)
    cal = {VIDEO: half}
    print(f"{'cell':<28}{'HOTA':>8}{'AssA':>8}{'F1':>8}{'PTR@10':>9}{'SWAP':>7}")
    rows_out = []
    for cell in sorted(glob.glob(str(RUNS / "*" / f"{VIDEO}_step*_rep*"))):
        cell = Path(cell)
        tcsv = cell / "tracks.csv"
        if not tcsv.exists():
            continue
        step = int(cell.name.split("_step")[1].split("_")[0])
        linker = cell.parent.name
        halved = cell / "tracks_halved.csv"
        halve_tracks(tcsv, halved)
        m, st_ = eval_video(halved, cell / "eval_scalefair", cal, VIDEO, step=step)
        try:
            pr = compute_ptr(str(tcsv), VIDEO, step, tol_px=10.0)
        except Exception as e:
            print(f"  PTR failed {cell.name}: {e}")
            pr = None
        name = f"{linker}/{cell.name}"
        print(f"{name:<28}{m.get('hota', float('nan')):>8.4f}{m.get('assa', float('nan')):>8.4f}"
              f"{m.get('f1', float('nan')):>8.4f}"
              f"{(pr['ptr'] if pr else float('nan')):>9.4f}"
              f"{(pr['SWAP'] if pr else -1):>7}")
        rows_out.append({"linker": linker, "cell": cell.name, "step": step,
                         **{k: m.get(k) for k in ("hota", "assa", "f1", "idf1", "recall")},
                         "ptr10": (pr["ptr"] if pr else None),
                         "swap10": (pr["SWAP"] if pr else None)})
    out = RUNS.parent / "b4k_scalefair_metrics.csv"
    if rows_out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
