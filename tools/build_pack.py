"""Build the CPTV_bench data pack from the local v2 research tree (T019). Run on the machine that has v2.

Copies, never modifies:
  v2_snapshot/evaluation/evaluator.py and v2_snapshot/src/trackers/<evaluator modules>
  v2_snapshot/data/... and v2_snapshot/LAMMPS downloads/... : the GT files GT_BY_BASE maps (same relative layout,
      because the copied modules resolve GT as <root>/data/<GT_BY_BASE path>)
  v2_snapshot/detection_benchmark/results/<group>/calibration_transforms_by_video.csv (4 groups)
and writes cells.csv (the 71 master_table_v4 cells) and PROVENANCE.md (source path, v2 commit, sha256 per file).
detections_col/ is produced separately by v2 tools/add_colour.py --all --out-dir <pack>/detections_col (T013).

Usage: python tools/build_pack.py --v2 <v2 root> --p7 <v2 worktree with tools/p7_cells.py>
"""
import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
SNAP = PACK / "v2_snapshot"
EVAL_MODULES = ["phase3_evaluator", "stage2_evaluate_calibrated", "motion_success_histogram", "compute_ptr",
                "phase6e_b4k_scalefair", "gt_radii", "phase5_eval_primary", "phase5_res_scalefair_eval"]
CAL_GROUPS = {"primary": "calibration_phase2c", "groupm": "calibration_groupm_v2",
              "res": "calibration_res_scaling", "p6i": "calibration_p6i"}
SCALEFAIR = {"RES_B_4K"}       # master_table_v4 scores RES_B_4K with the scale-fair eval (phase6e_b4k_scalefair)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy(src, rel, prov, v2):
    dst = SNAP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    prov.append((rel.as_posix(), Path(src).relative_to(v2).as_posix(), sha(dst), dst.stat().st_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--p7", required=True, help="v2 worktree holding tools/p7_cells.py (T013)")
    a = ap.parse_args()
    v2, p7 = Path(a.v2), Path(a.p7)
    sys.path.insert(0, str(p7 / "tools"))
    sys.path.insert(0, str(v2 / "src" / "trackers"))
    import p7_cells
    from stage2_evaluate_calibrated import GT_BY_BASE
    from motion_success_histogram import video_base

    prov = []
    copy(v2 / "evaluation" / "evaluator.py", Path("evaluation/evaluator.py"), prov, v2)
    for m in EVAL_MODULES:
        copy(v2 / "src" / "trackers" / f"{m}.py", Path("src/trackers") / f"{m}.py", prov, v2)
    for rel in sorted(set(GT_BY_BASE.values())):
        src = (v2 / "data" / rel).resolve()
        copy(src, Path("data") / rel if not rel.startswith("../") else Path(rel[3:]), prov, v2)
    for g in CAL_GROUPS.values():
        rel = Path("detection_benchmark/results") / g / "calibration_transforms_by_video.csv"
        copy(v2 / rel, rel, prov, v2)

    import pandas as pd
    mt = pd.read_csv(p7_cells.MASTER)
    ship = mt[mt.arm == "capauto"].set_index(["video", "step"])
    rows = []
    for c in p7_cells.resolve_cells():
        base = video_base(c["video"])
        gt = GT_BY_BASE[base]
        s = ship.loc[(c["video"], c["step"])]
        rows.append({
            "cell": f"{c['video']}_s{c['step']}", "video": c["video"], "step": c["step"], "base": base,
            "detections": f"detections_col/{c['det_csv'].name}",
            "gt": gt, "calib_group": CAL_GROUPS[c["corpus"]],
            "eval_mode": "scalefair" if c["video"] in SCALEFAIR else "5px",
            "d_per_frame": s["d_per_frame"], "ship_hota": s["hota"], "ship_ptr": s["ptr"],
            "source_detections": Path(c["det_csv"]).relative_to(v2).as_posix(),
        })
    with open(PACK / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    commit = subprocess.run(["git", "-C", str(v2), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(v2), "status", "--porcelain", "--"] + [r[1] for r in prov],
                           capture_output=True, text=True).stdout.strip()
    lines = ["# Provenance of the copied files", "",
             f"Copied unmodified from the v2 research repository at commit `{commit}` "
             f"(local working tree; `git status` of the copied paths: {'clean' if not dirty else 'NOT clean: ' + dirty}).",
             "GT files under `LAMMPS downloads/` and `data/` are untracked research data; their sha256 below is the record.",
             "", "| pack path (under v2_snapshot/) | v2 source path | bytes | sha256 |", "|---|---|---|---|"]
    lines += [f"| `{d}` | `{s}` | {n} | `{h}` |" for d, s, h, n in prov]
    (PACK / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"copied {len(prov)} files; {len(rows)} cells -> cells.csv; PROVENANCE.md written")


if __name__ == "__main__":
    main()
