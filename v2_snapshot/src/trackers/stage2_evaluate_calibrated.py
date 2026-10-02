#!/usr/bin/env python3
"""stage2_evaluate_calibrated.py - Evaluator wrapper that passes per-video
calibration to evaluator.py.

Reads Phase 2C calibration CSV
(detection_benchmark/results/calibration_phase2c/calibration_transforms_by_video.csv)
to obtain per-video viewport bounds, then invokes evaluator.py with the
correct positional args:

    evaluator.py <tracks.csv> <eval_out.csv> <gt_path>
                 <txt_max> <vp_min> <vp_max> <vp_res>
                 <max_match_dist> <vp_y_min> <vp_y_max> <vp_res_y>

For Phase 3a noise variants (K01_clean_grey, K05_realistic, ...), the
base video's calibration applies directly (same physics, same viewport).
We strip the suffix to find the base entry in the Phase 2C CSV.

Phase 3a PRIMARY variants (PRIMARY_clean_grey, PRIMARY_clean_black, ...)
fall back to the original PRIMARY calibration with the canonical
``blob_data 3.txt`` GT.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_V2_DIR = _THIS.parent.parent.parent
EVALUATOR = _V2_DIR / "evaluation" / "evaluator.py"
PYTHON = r"C:\Users\Kavee\anaconda3\python.exe"
# Default = Phase-2C source-of-truth. Override via CPTV_CALIB_CSV to point at a
# merged CSV (e.g. phase2c + Group-M rows) without editing this path; unset =
# byte-identical legacy behaviour.
CALIB_CSV = Path(os.environ.get(
    "CPTV_CALIB_CSV",
    str(_V2_DIR / "detection_benchmark" / "results" / "calibration_phase2c"
        / "calibration_transforms_by_video.csv")))

# Healthy O outputs are 4-9MB; below 1MB indicates a partial run.
MIN_VALID_BYTES = 1_000_000


def _video_base(video):
    """Return the K-base ID for a Phase 3a noise variant."""
    m = re.match(r"^(K\d+|PRIMARY)(?:_.*)?$", video)
    return m.group(1) if m else video


def load_calibration():
    """Return dict {base_video: row} keyed on K01/K05/K06/PRIMARY."""
    cal = {}
    with open(CALIB_CSV, newline="") as f:
        for row in csv.DictReader(f):
            cal[row["video"]] = row
    return cal


# Per-video GT paths (mirrors detection_benchmark/strategy_combination_benchmark
# and parallel_runner.py).
GT_BY_BASE = {
    "PRIMARY": "blob_data 3.txt",
    "K01":     "group_k/K01/viewport_blobs_centered_data.txt",
    "K05":     "group_k/K05/viewport_swirl_centered_data.txt",
    "K06":     "group_k/K06/viewport_funnel_centered_data.txt",
    # [Phase 6i Part C] corpus completion — additive entries only. L-videos
    # are re-renders sharing the K GT (detection_benchmark GT_FILE_MAP).
    "K02":     "group_k/K02/viewport_blobs_corner_data.txt",
    "K03":     "group_k/K03/viewport_blobs_slit_data.txt",
    "K04":     "group_k/K04/viewport_blobs_centered3_data.txt",
    "L01":     "group_k/K01/viewport_blobs_centered_data.txt",
    "L02":     "group_k/K02/viewport_blobs_corner_data.txt",
    "L03":     "group_k/K03/viewport_blobs_slit_data.txt",
    # Group-M (Phase 5 Stage 4). Keyed by full video stem (phase3_evaluator's
    # _video_base passes non-K/PRIMARY names through unchanged). GT paths are
    # relative to v2/data, so one level up into ``LAMMPS downloads/``.
    "M01_drum":        "../LAMMPS downloads/test_suite_output/M01_drum/M01_drum_data.txt",
    "M02_chute_25deg": "../LAMMPS downloads/test_suite_output/M02_chute_25deg/M02_chute_25deg_data.txt",
    "M03_chute_35deg": "../LAMMPS downloads/test_suite_output/M03_chute_35deg/M03_chute_35deg_data.txt",
    "M04_chute_50deg": "../LAMMPS downloads/test_suite_output/M04_chute_50deg/M04_chute_50deg_data.txt",
    "M05_hopper":      "../LAMMPS downloads/test_suite_output/M05_hopper/M05_hopper_data.txt",
    "M06_chute_bump":  "../LAMMPS downloads/test_suite_output/M06_chute_bump/M06_chute_bump_data.txt",
    # Phase 5 resolution/scale corpus (res_scaling). B_1080/B_4K share the
    # ds=1 sim; per-video copies keep each video dir self-contained.
    "RES_A_1080": "../LAMMPS downloads/test_suite_output/RES_A_1080/RES_A_1080_data.txt",
    "RES_A_4K":   "../LAMMPS downloads/test_suite_output/RES_A_4K/RES_A_4K_data.txt",
    "RES_B_1080": "../LAMMPS downloads/test_suite_output/RES_B_1080/RES_B_1080_data.txt",
    "RES_B_4K":   "../LAMMPS downloads/test_suite_output/RES_B_4K/RES_B_4K_data.txt",
}


METRIC_PATTERNS = [
    ("recall",      re.compile(r"^\s*Recall:\s+([\d.]+)\s*$")),
    ("precision",   re.compile(r"^\s*Precision:\s+([\d.]+)\s*$")),
    ("fragmentation", re.compile(r"^\s*Fragmentation ratio:\s+([\d.]+)x")),
    ("mota",        re.compile(r"^MOTA.*?:\s+([\-\d.]+)")),
    ("idf1",        re.compile(r"^\s*IDF1:\s+([\d.]+)")),
    ("continuity",  re.compile(r"^\s*Continuity rate:\s+([\d.]+)")),
    ("f1",          re.compile(r"^\s*F1-Score:\s+([\d.]+)\s*$")),
    # Phase 1 additions — HOTA / AssA come out in the "  HOTA:  0.xxxx (xx.xx%)"
    # block; FFCAR is reported as a stand-alone "FFCAR  0.xxxx" line in the
    # summary table.
    ("hota",        re.compile(r"^\s*HOTA:\s+([\d.]+)\s*\(")),
    ("assa",        re.compile(r"^\s*AssA:\s+([\d.]+)\s*\(")),
    ("ffcar",       re.compile(r"^\s*FFCAR\s+([\d.]+)")),
]


def scrape_metrics(log_path):
    out = {}
    if not Path(log_path).exists():
        return out
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            for name, pat in METRIC_PATTERNS:
                m = pat.match(line)
                if m and name not in out:
                    try:
                        out[name] = float(m.group(1))
                    except ValueError:
                        pass
    return out


def _run_one_eval(tracks_csv, eval_dir, gt_path, txt_max, extra_args=None,
                   force=False):
    """Run evaluator.py once into eval_dir, return scraped metrics.

    extra_args: list of additional CLI flags (e.g. for the Phase 2C override).
    Returns dict with status + metrics + eval_dir/eval_csv/log paths.
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = eval_dir / "evaluation.csv"
    log_path = eval_dir / "eval.log"
    if not force and log_path.exists() and log_path.stat().st_size > 1000:
        cached = scrape_metrics(log_path)
        if cached:
            return {"status": "cached", "metrics": cached, "log": str(log_path)}

    cmd = [PYTHON, str(EVALUATOR), str(tracks_csv), str(eval_csv),
           gt_path, str(txt_max)]
    if extra_args:
        cmd.extend(extra_args)
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=str(_V2_DIR / "data"))
    metrics = scrape_metrics(log_path)
    return {"status": "ok" if r.returncode == 0 else f"eval_exit_{r.returncode}",
            "metrics": metrics, "log": str(log_path)}


def evaluate_one(task_dir, calibration, force=False):
    """Run BOTH evaluator paths (Phase 2C override + auto-fit) so we can
    compare. Both write into separate subdirs: eval_phase2c/ and
    eval_autofit/ . Sanity rule (per user direction):

      - Both agree     -> trust the number
      - phase2c >> autofit -> auto-fit drifted (the failure case we
                              observed on K01_clean_grey / PRIMARY_clean_grey)
      - autofit >> phase2c -> auto-fit found a "wrong but recall-maximising"
                              transform that fits to FPs; phase2c is the
                              honest number
      - both low       -> tracker genuinely struggles on that variant
                          (e.g. NCC threshold too strict for low-contrast
                          grey background)
    """
    task_dir = Path(task_dir)
    tracks_csv = task_dir / "tracks.csv"
    if not tracks_csv.exists():
        return {"task": task_dir.name, "status": "missing_tracks_csv"}
    if tracks_csv.stat().st_size < MIN_VALID_BYTES:
        return {"task": task_dir.name, "status": "partial",
                "size": tracks_csv.stat().st_size}

    parts = task_dir.name.split("_", 1)
    if len(parts) != 2:
        return {"task": task_dir.name, "status": "bad_name"}
    tracker, video = parts
    base = _video_base(video)
    if base not in calibration:
        return {"task": task_dir.name, "status": f"no_cal_for_{base}"}
    cal_row = calibration[base]
    if base not in GT_BY_BASE:
        return {"task": task_dir.name, "status": f"no_gt_for_{base}"}

    gt_path = GT_BY_BASE[base]
    txt_max = float(cal_row["viewport_y_max"])
    scale = float(cal_row["scale"])
    ox_p2c = float(cal_row["ox"])
    oy_p2c = float(cal_row["oy"])
    H_video = float(cal_row["H_video"])

    # Convert Phase 2C transform to evaluator.py's transform.
    #
    # Phase 2C  (gt_viewport.apply_gt_transform_yflip):
    #     px = x_sim * scale + ox_p2c
    #     py = H_video - (y_sim * scale + oy_p2c)
    #
    # evaluator.py (transform_txt_coords with y-flip in sim space):
    #     px = x_sim * scale + offset_x_eval
    #     py = (TXT_MAX - y_sim) * scale + offset_y_eval
    #
    # Equating:
    #     offset_x_eval = ox_p2c
    #     offset_y_eval = H_video - oy_p2c - TXT_MAX * scale
    eval_ox = ox_p2c
    eval_oy = H_video - oy_p2c - txt_max * scale

    print(f"[eval-cal] {task_dir.name} -> gt={gt_path} "
          f"p2c->eval=(scale={scale:.4f}, ox={eval_ox:.4f}, oy={eval_oy:.4f})",
          flush=True)

    # 1) Phase 2C override (converted to evaluator coordinate convention)
    p2c = _run_one_eval(
        tracks_csv, task_dir / "eval_phase2c",
        gt_path, txt_max,
        extra_args=[f"--eval-scale={scale}",
                    f"--eval-offset-x={eval_ox}",
                    f"--eval-offset-y={eval_oy}"],
        force=force,
    )
    # 2) Auto-fit
    af = _run_one_eval(
        tracks_csv, task_dir / "eval_autofit",
        gt_path, txt_max,
        extra_args=None,
        force=force,
    )

    return {
        "task": task_dir.name, "tracker": tracker, "video": video,
        "base": base,
        "phase2c": p2c, "autofit": af,
        "status": "ok",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default=str(
        _V2_DIR / "data" / "results" / "phase3e_full_pipeline" / "stage2_subset"))
    p.add_argument("--force", action="store_true",
                   help="Force re-evaluation even if cached eval.log exists")
    p.add_argument("--csv-out", default=None,
                   help="Path for the metrics CSV (default: <root>/stage2_metrics_calibrated.csv)")
    args = p.parse_args()

    out_root = Path(args.output_root)
    cal = load_calibration()
    print(f"[eval-cal] Calibration loaded for {len(cal)} base videos: "
          f"{sorted(cal.keys())[:6]}...")

    rows = []
    task_dirs = sorted(d for d in out_root.iterdir()
                        if d.is_dir() and "_" in d.name)
    print(f"[eval-cal] Scanning {len(task_dirs)} task dirs")

    for tdir in task_dirs:
        info = evaluate_one(tdir, cal, force=args.force)
        if info["status"] != "ok":
            print(f"  {tdir.name:<35} {info['status']}")
            rows.append(info)
            continue
        p2c = info["phase2c"].get("metrics", {})
        af = info["autofit"].get("metrics", {})
        print(f"  {tdir.name:<35} "
              f"P2C R={p2c.get('recall', '?')} F1={p2c.get('f1', '?')} | "
              f"AUTO R={af.get('recall', '?')} F1={af.get('f1', '?')}")
        rows.append(info)

    csv_path = Path(args.csv_out) if args.csv_out else (
        out_root / "stage2_metrics_calibrated.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tracker", "video", "base", "status",
            "p2c_recall", "p2c_precision", "p2c_f1",
            "p2c_fragmentation", "p2c_mota", "p2c_idf1", "p2c_continuity",
            "autofit_recall", "autofit_precision", "autofit_f1",
            "autofit_fragmentation", "autofit_mota", "autofit_idf1",
            "autofit_continuity",
            "diag",
        ])
        for r in rows:
            if r["status"] != "ok":
                w.writerow([r.get("tracker", ""), r.get("video", ""),
                            r.get("base", ""), r["status"]] + [""] * 14
                           + [r["status"]])
                continue
            p2c = r["phase2c"].get("metrics", {})
            af = r["autofit"].get("metrics", {})
            # Diagnostic interpretation per user spec.
            p2c_R = p2c.get("recall")
            af_R = af.get("recall")
            diag = "ok"
            if p2c_R is not None and af_R is not None:
                delta = af_R - p2c_R
                if abs(delta) < 0.02:
                    diag = "agree"
                elif delta > 0.10:
                    diag = "autofit_inflated"
                elif delta < -0.10:
                    diag = "autofit_drifted"
                else:
                    diag = f"diff={delta:+.3f}"
                if (p2c_R is not None and p2c_R < 0.5
                        and af_R is not None and af_R < 0.5):
                    diag = "both_low_tracker_struggles"
            w.writerow([
                r.get("tracker", ""), r.get("video", ""),
                r.get("base", ""), r["status"],
                p2c.get("recall", ""), p2c.get("precision", ""),
                p2c.get("f1", ""), p2c.get("fragmentation", ""),
                p2c.get("mota", ""), p2c.get("idf1", ""),
                p2c.get("continuity", ""),
                af.get("recall", ""), af.get("precision", ""),
                af.get("f1", ""), af.get("fragmentation", ""),
                af.get("mota", ""), af.get("idf1", ""),
                af.get("continuity", ""),
                diag,
            ])
    print(f"\n[eval-cal] Wrote {csv_path}")
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[eval-cal] {n_ok}/{len(rows)} tasks evaluated (both passes)")


if __name__ == "__main__":
    main()
