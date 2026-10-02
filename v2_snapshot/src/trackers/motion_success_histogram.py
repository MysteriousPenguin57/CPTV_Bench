#!/usr/bin/env python3
"""motion_success_histogram.py - Per-frame-transition tracking success
binned by GT particle motion (in particle diameters).

For each (frame transition N -> N+1, GT particle with assignment in
frame N):
  - Find tracker_id assigned to this GT in frame N (Hungarian on
    pos-distance, 6.44 px tolerance)
  - Find tracker_id assigned in frame N+1 (same matching)
  - Compute motion in pixels = || GT_pos(N+1) - GT_pos(N) ||
  - Compute motion in diameters = motion_px / (2 * radius_px)
  - Classify outcome:
      STABLE: same tracker_id in both frames
      SWAP:   different tracker_id in N+1 (different track followed it)
      LOST:   no tracker_id in N+1 (track died/dropped this particle)

Output: histogram (CSV + ASCII bar chart) of fraction
{stable, swap, lost} by motion-in-diameters bin.

Usage:
  python motion_success_histogram.py --task <task_dir>
  python motion_success_histogram.py --root <results_root>  # all tasks
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

_THIS = Path(__file__).resolve()
_TRACKERS_DIR = _THIS.parent
_V2_DIR = _TRACKERS_DIR.parent.parent
if str(_TRACKERS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACKERS_DIR))

from stage2_evaluate_calibrated import load_calibration, GT_BY_BASE
from gt_radii import extract_radii_for_base


def video_base(video):
    # P6J degraded-detection variants carry a __p<pp>s<seed> suffix
    # (additive: no pre-P6J name contains "__p").
    m = re.match(r"^(.+?)__p\d+s\d+$", video)
    if m:
        video = m.group(1)
    if video.startswith("PRIMARY"):
        return "PRIMARY"
    m = re.match(r"^(K\d+)", video)
    return m.group(1) if m else video


def parse_step(name):
    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else 1


def load_gt_pixel(gt_path, scale, ox, oy, H_video):
    """Returns dict: source_frame -> list of (atom_id, x_px, y_px)."""
    full = _V2_DIR / "data" / gt_path
    by_frame = defaultdict(list)
    if not full.exists():
        return by_frame
    with open(full) as f:
        in_atoms = False
        cur_idx = -1
        for line in f:
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True; cur_idx += 1; continue
            if line.startswith("ITEM: TIMESTEP") or line.startswith("ITEM: NUMBER"):
                in_atoms = False; continue
            if not in_atoms: continue
            parts = line.split()
            if len(parts) < 5: continue
            try:
                aid = int(parts[0])
                if int(parts[1]) != 3: continue
                xs, ys = float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue
            by_frame[cur_idx].append((aid, xs * scale + ox, H_video - (ys * scale + oy)))
    return by_frame


def assign_tracker_ids_per_frame(tracks_df, gt_by_frame, step, tol_px,
                                  legacy_offset=False):
    """For each processed frame, Hungarian-match GT to tracker rows
    within tol_px. Returns dict: source_frame -> {gt_atom_id: tracker_id}.

    legacy_offset=True: pre-fix data where source_frame = proc*step + (step-1).
    """
    out = {}
    if 'missed' in tracks_df.columns:
        active = tracks_df[tracks_df['missed'] == 0]
    else:
        active = tracks_df
    offset = (step - 1) if legacy_offset else 0
    for proc_frame, grp in active.groupby('frame'):
        sf = proc_frame * step + offset
        if sf not in gt_by_frame:
            continue
        gt = gt_by_frame[sf]
        if not gt:
            continue
        gt_xy = np.array([[x, y] for _, x, y in gt])
        gt_ids = [aid for aid, _, _ in gt]
        tr_xy = grp[['x', 'y']].values
        tr_ids = grp['track_id'].values
        if len(tr_xy) == 0:
            continue
        # Distance matrix (GT x tracker)
        cost = np.linalg.norm(gt_xy[:, None, :] - tr_xy[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = {}
        for r, c in zip(row, col):
            if cost[r, c] <= tol_px:
                mapping[gt_ids[r]] = int(tr_ids[c])
        out[sf] = mapping
    return out


def classify_transitions(assignments_by_frame, gt_by_frame, radius_px,
                          step, legacy_offset=False, count_missing=True):
    """For consecutive processed frames, classify each GT particle's
    outcome and compute motion in diameters AND local density.

    Local density per particle = distance to nearest GT neighbour
    in particle diameters (smaller = denser). Computed in frame N.

    Returns list of dicts: {motion_diam, motion_px, outcome,
                             nn_dist_diam, n_neighbours_3d}
    """
    sorted_sf = sorted(assignments_by_frame.keys())
    diameter_px = 2.0 * radius_px
    transitions = []
    for i in range(len(sorted_sf) - 1):
        sfA = sorted_sf[i]
        sfB = sorted_sf[i + 1]
        if sfB - sfA != step:
            continue
        gtA_list = gt_by_frame.get(sfA, [])
        gtA = {aid: (x, y) for aid, x, y in gtA_list}
        gtB = {aid: (x, y) for aid, x, y in gt_by_frame.get(sfB, [])}
        mapA = assignments_by_frame[sfA]
        mapB = assignments_by_frame.get(sfB, {})
        if not gtA_list:
            continue
        # KDTree on frame-A GT positions for density queries
        gtA_xy = np.array([[x, y] for _, x, y in gtA_list])
        gtA_ids = [aid for aid, _, _ in gtA_list]
        gtA_id_to_idx = {a: i for i, a in enumerate(gtA_ids)}
        tree_A = cKDTree(gtA_xy)
        # Phase 3l v2: iterate over ALL gtA particles (not just mapA)
        # to capture MISSING (untracked in N at all). PTR includes
        # MISSING + LOST + SWAP as failures; FFCAR only counts SWAP.
        gt_ids_in_both = [aid for aid in gtA if aid in gtB]
        for gt_id in gt_ids_in_both:
            tr_a = mapA.get(gt_id, None)
            tr_b = mapB.get(gt_id, None)
            xa, ya = gtA[gt_id]
            xb, yb = gtB[gt_id]
            motion_px = float(np.hypot(xb - xa, yb - ya))
            motion_diam = motion_px / diameter_px
            # Density: distance to nearest neighbour (excluding self)
            d2, _ = tree_A.query(np.array([xa, ya]), k=2)
            nn_px = float(d2[1]) if len(d2) > 1 else float('inf')
            nn_diam = nn_px / diameter_px
            n_3d = len(tree_A.query_ball_point([xa, ya], 3.0 * diameter_px)) - 1
            if tr_a is None:
                outcome = "MISSING"  # GT not tracked in N at all
            elif tr_b is None:
                outcome = "LOST"     # GT was tracked in N, dropped in N+1
            elif tr_b == tr_a:
                outcome = "STABLE"   # Same track ID maintained
            else:
                outcome = "SWAP"     # Different track ID
            transitions.append({
                "motion_px": motion_px,
                "motion_diam": motion_diam,
                "outcome": outcome,
                "nn_dist_diam": nn_diam,
                "n_neighbours_3d": n_3d,
            })
    return transitions


def histogram(transitions, bins, density_strata=False):
    """Bin by motion_diam, count outcomes per bin.

    If density_strata=True, also stratify by local density quartile
    (computed on nn_dist_diam: low quartile = closest neighbour =
    densest, high quartile = sparsest).
    """
    if not transitions:
        return None
    df = pd.DataFrame(transitions)
    df['bin'] = pd.cut(df['motion_diam'], bins=bins, include_lowest=True)
    if density_strata and 'nn_dist_diam' in df.columns:
        # Quartiles of nearest-neighbour distance
        try:
            df['density_q'] = pd.qcut(df['nn_dist_diam'], q=4,
                                      labels=['Q1_densest', 'Q2', 'Q3', 'Q4_sparsest'],
                                      duplicates='drop')
            grp = df.groupby(['bin', 'density_q'],
                              observed=False)['outcome'].value_counts().unstack(fill_value=0)
        except Exception:
            grp = df.groupby('bin', observed=False)['outcome'].value_counts().unstack(fill_value=0)
    else:
        grp = df.groupby('bin', observed=False)['outcome'].value_counts().unstack(fill_value=0)
    for col in ['STABLE', 'SWAP', 'LOST', 'MISSING']:
        if col not in grp.columns: grp[col] = 0
    grp['total'] = grp[['STABLE', 'SWAP', 'LOST', 'MISSING']].sum(axis=1)
    grp['stable_rate'] = grp['STABLE'] / grp['total'].replace(0, np.nan)
    grp['swap_rate'] = grp['SWAP'] / grp['total']
    grp['lost_rate'] = grp['LOST'] / grp['total']
    grp['missing_rate'] = grp['MISSING'] / grp['total']
    # PTR (Particle Tracking Rate) = STABLE / (STABLE+SWAP+LOST+MISSING)
    # FFCAR-equivalent here = STABLE / (STABLE+SWAP) (excludes LOST+MISSING)
    grp['PTR'] = grp['stable_rate']  # alias for clarity
    return grp


def analyse_task(task_dir, tol_px=6.44, legacy_offset=False):
    name = task_dir.name
    # Parse video name out of the dir name: <tracker>_<video>_step<N>[_extras]
    m = re.match(r"^[^_]+_(.*?)_step\d+", name)
    if not m:
        return None
    video = m.group(1)
    base = video_base(video)
    cal = load_calibration()
    if base not in cal:
        return None
    info = extract_radii_for_base(base, cal)
    if info is None:
        return None
    cr = cal[base]
    scale = float(cr["scale"]); ox = float(cr["ox"])
    oy = float(cr["oy"]); H = float(cr["H_video"])
    gt_path = GT_BY_BASE.get(base, "blob_data 3.txt")

    tracks_csv = task_dir / "tracks.csv"
    if not tracks_csv.exists() or tracks_csv.stat().st_size < 100_000:
        return None
    df = pd.read_csv(tracks_csv)
    step = parse_step(name)
    gt_by_frame = load_gt_pixel(gt_path, scale, ox, oy, H)
    if not gt_by_frame:
        return None
    assignments = assign_tracker_ids_per_frame(df, gt_by_frame, step, tol_px,
                                                legacy_offset=legacy_offset)
    transitions = classify_transitions(assignments, gt_by_frame,
                                        info['mean_px'], step,
                                        legacy_offset=legacy_offset)
    return {
        "task": name, "video": video, "step": step,
        "radius_px": info['mean_px'],
        "n_transitions": len(transitions),
        "transitions": transitions,
    }


def print_histogram(grp, label):
    print(f"\n=== {label} ===")
    print(f"{'bin':>14} {'n':>6} {'stable%':>8} {'swap%':>7} {'lost%':>7}  bar (stable | swap | lost)")
    print("-" * 100)
    for bin_, row in grp.iterrows():
        n = int(row['total'])
        if n == 0: continue
        sr = row['stable_rate'] * 100
        wr = row['swap_rate'] * 100
        lr = row['lost_rate'] * 100
        bar_len = 40
        s_ch = int(round(sr / 100 * bar_len))
        w_ch = int(round(wr / 100 * bar_len))
        l_ch = bar_len - s_ch - w_ch
        bar = "S" * s_ch + "x" * w_ch + "." * l_ch
        print(f"{str(bin_):>14} {n:>6} {sr:>7.1f}% {wr:>6.1f}% {lr:>6.1f}%  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="Single task dir to analyse")
    ap.add_argument("--root", help="Results root dir; analyse all tasks within")
    ap.add_argument("--bins", default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,10.0",
                    help="Comma-separated motion-diameter bin edges")
    ap.add_argument("--out-csv", help="Write per-task histogram CSV here")
    ap.add_argument("--tol", type=float, default=6.44,
                    help="Position-match tolerance (px)")
    ap.add_argument("--legacy-offset", action="store_true",
                    help="Use source_frame=proc*step+(step-1) for pre-fix data")
    args = ap.parse_args()

    bins = [float(x) for x in args.bins.split(",")]

    tasks = []
    if args.task:
        tasks.append(Path(args.task))
    elif args.root:
        for d in sorted(Path(args.root).iterdir()):
            if d.is_dir() and (d / "tracks.csv").exists():
                tasks.append(d)
    else:
        print("Need --task or --root"); return 1

    rows = []
    rows_density = []
    for tdir in tasks:
        result = analyse_task(tdir, tol_px=args.tol,
                               legacy_offset=args.legacy_offset)
        if result is None or result['n_transitions'] == 0:
            continue
        # Plain histogram
        grp = histogram(result['transitions'], bins, density_strata=False)
        if grp is None: continue
        print_histogram(grp,
            f"{result['task']} (r={result['radius_px']:.2f}px, "
            f"step={result['step']}, n={result['n_transitions']} transitions)")
        for bin_, r in grp.iterrows():
            rows.append({
                "task": result['task'], "video": result['video'],
                "step": result['step'], "radius_px": result['radius_px'],
                "motion_bin_diameters": str(bin_),
                "n": int(r['total']),
                "stable_pct": float(r['stable_rate'] * 100) if r['total'] > 0 else 0,
                "swap_pct": float(r['swap_rate'] * 100) if r['total'] > 0 else 0,
                "lost_pct": float(r['lost_rate'] * 100) if r['total'] > 0 else 0,
            })
        # Density-stratified histogram (motion x density quartile)
        grp_d = histogram(result['transitions'], bins, density_strata=True)
        if grp_d is None or grp_d.empty: continue
        for idx, r in grp_d.iterrows():
            if isinstance(idx, tuple):
                bin_, dq = idx
            else:
                bin_, dq = idx, "all"
            rows_density.append({
                "task": result['task'], "video": result['video'],
                "step": result['step'], "radius_px": result['radius_px'],
                "motion_bin_diameters": str(bin_),
                "density_quartile": str(dq),
                "n": int(r['total']),
                "stable_pct": float(r['stable_rate'] * 100) if r['total'] > 0 else 0,
                "swap_pct": float(r['swap_rate'] * 100) if r['total'] > 0 else 0,
                "lost_pct": float(r['lost_rate'] * 100) if r['total'] > 0 else 0,
            })

    if args.out_csv and rows:
        out = Path(args.out_csv)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {out} ({len(rows)} bin-rows)")
    if args.out_csv and rows_density:
        out_d = Path(args.out_csv).with_name(
            Path(args.out_csv).stem + "_density.csv")
        with open(out_d, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_density[0].keys()))
            w.writeheader(); w.writerows(rows_density)
        print(f"Wrote {out_d} ({len(rows_density)} bin-rows, density-stratified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
