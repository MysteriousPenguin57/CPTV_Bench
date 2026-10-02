"""Phase 5 speed — PRIMARY HOTA eval helper.

Wraps evaluator.py with the Phase-2C PRIMARY calibration (same transform
conversion as stage2_evaluate_calibrated.evaluate_one) so any tracks.csv on the
PRIMARY video can be scored to HOTA / recall / F1 / IDF1 / MOTA. Used for the
lever-2 (--kernel-recovery-gate) accuracy-within-sigma decision: recovery ON vs
OFF at w_kernel=0 must not move HOTA beyond the run-to-run spread.

Run from v2/src/:
    "C:/Users/Kavee/anaconda3/python.exe" -m trackers.phase5_eval_primary \
        <cell_dir> [<cell_dir> ...]        # evals each cell's tracks.csv
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from trackers.stage2_evaluate_calibrated import (load_calibration, _run_one_eval,
                                                 GT_BY_BASE)

_KEYS = ("hota", "recall", "f1", "idf1", "mota", "fragmentation")


def eval_video(tracks_csv, eval_dir, cal, base, force=False, step=1):
    """Score one tracks.csv for calibrated `base` (PRIMARY / K0x / M0x) -> metrics
    dict (hota, recall, f1, ...). Same Phase-2C transform conversion as
    stage2_evaluate_calibrated.evaluate_one. `cal` must contain `base` (load the
    matching calibration CSV via CPTV_CALIB_CSV for Group-M).

    WARNING (gs_disp CORRECTION #1, 2026-07-02): for a run made with
    --frame-step N>1 you MUST pass step=N. tracks.csv frame indices are the
    PROCESSED index while GT is indexed by SOURCE frame; without the remap the
    evaluator compares tracker frame N*i against GT frame i and HOTA floors at
    ~0.23 regardless of config. Every speed_gs s4/s8 number (bakeoff, wksweep,
    gate-d sweep, krec A/B) was produced by this helper WITHOUT the remap and
    is an eval artifact. phase3_evaluator has always remapped; prefer it for
    grids. NB results are cached in eval_dir/eval.log -- purge the dir if it
    was ever populated by an unremapped call."""
    tracks_csv = Path(tracks_csv)
    if int(step) > 1:
        import csv as _csv
        remapped = tracks_csv.parent / f"tracks_remapped_s{int(step)}.csv"
        with open(tracks_csv, newline="") as fin, open(remapped, "w", newline="") as fout:
            rd = _csv.DictReader(fin)
            wr = _csv.DictWriter(fout, fieldnames=rd.fieldnames or [])
            wr.writeheader()
            for r in rd:
                try:
                    r["frame"] = str(int(r["frame"]) * int(step))
                except (KeyError, ValueError):
                    pass
                wr.writerow(r)
        tracks_csv = remapped
    row = cal[base]
    txt_max = float(row["viewport_y_max"])
    scale = float(row["scale"])
    ox = float(row["ox"]); oy = float(row["oy"]); H = float(row["H_video"])
    eval_ox = ox
    eval_oy = H - oy - txt_max * scale
    res = _run_one_eval(
        tracks_csv, eval_dir, GT_BY_BASE[base], txt_max,
        extra_args=[f"--eval-scale={scale}", f"--eval-offset-x={eval_ox}",
                    f"--eval-offset-y={eval_oy}"],
        force=force)
    return res.get("metrics", {}), res.get("status", "?")


def eval_primary(tracks_csv, eval_dir, cal, force=False):
    """Score one PRIMARY tracks.csv -> metrics dict (hota, recall, f1, ...)."""
    return eval_video(tracks_csv, eval_dir, cal, "PRIMARY", force=force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", help="cell dirs containing tracks.csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cal = load_calibration()
    rows = []
    for c in a.cells:
        cell = Path(c)
        tcsv = cell / "tracks.csv"
        if not tcsv.exists():
            print(f"{cell.name:<40} MISSING tracks.csv"); continue
        m, st = eval_primary(tcsv, cell / "eval_phase2c", cal, force=a.force)
        rows.append((cell, m))
        print(f"{str(cell.relative_to(cell.parents[2])):<46} "
              f"HOTA={m.get('hota','?')} R={m.get('recall','?')} "
              f"F1={m.get('f1','?')} IDF1={m.get('idf1','?')} [{st}]")
    # if multiple, summarise HOTA mean/std
    hotas = [m.get("hota") for _, m in rows if m.get("hota") is not None]
    if len(hotas) > 1:
        print(f"\nHOTA n={len(hotas)} mean={statistics.mean(hotas):.4f} "
              f"std={statistics.pstdev(hotas):.4f} "
              f"min={min(hotas):.4f} max={max(hotas):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
