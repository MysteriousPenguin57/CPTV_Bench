"""gt_radii.py - Extract per-video GT particle radius (px) from each video's
LAMMPS dump file. Returns video_base -> mean radius in pixels.

Used by the evaluator (tolerance = 1.1 * radius_px) and by reports.
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
_TRACKERS_DIR = _THIS.parent
_V2_DIR = _TRACKERS_DIR.parent.parent

if str(_TRACKERS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACKERS_DIR))

from stage2_evaluate_calibrated import load_calibration, GT_BY_BASE


def extract_radii_for_base(base, cal):
    """Returns (mean_radius_px, min_radius_px, max_radius_px, n_atoms)."""
    if base not in cal:
        return None
    gt_path = _V2_DIR / "data" / GT_BY_BASE.get(base, "blob_data 3.txt")
    if not gt_path.exists():
        return None
    cr = cal[base]
    scale = float(cr["scale"])

    diams = []
    with open(gt_path) as f:
        in_atoms = False
        for line in f:
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True; continue
            if line.startswith("ITEM: TIMESTEP") or line.startswith("ITEM: NUMBER"):
                in_atoms = False; continue
            if not in_atoms: continue
            parts = line.split()
            if len(parts) < 5: continue
            try:
                if int(parts[1]) != 3: continue
                diam = float(parts[4])
            except (ValueError, IndexError):
                continue
            diams.append(diam)
    if not diams:
        return None
    import numpy as np
    diams = np.array(diams)
    radii_px = diams * scale / 2.0
    return {
        "n": len(radii_px),
        "mean_px": float(radii_px.mean()),
        "min_px": float(radii_px.min()),
        "max_px": float(radii_px.max()),
        "scale": scale,
        "tolerance_px": 1.1 * float(radii_px.mean()),
    }


def all_radii():
    """Return dict: video_base -> radius info dict."""
    cal = load_calibration()
    out = {}
    for base in GT_BY_BASE.keys():
        info = extract_radii_for_base(base, cal)
        if info is not None:
            out[base] = info
    return out


def tolerance_for_video(video_name):
    """Get 1.1*radius tolerance for a video. video_name can be PRIMARY,
    K01_realistic, K05_anything, etc."""
    base = video_name
    if not video_name.startswith("PRIMARY"):
        import re
        m = re.match(r"^(K\d+)", video_name)
        if m: base = m.group(1)
    info = extract_radii_for_base(base, load_calibration())
    if info is None:
        return 12.0  # fallback
    return info["tolerance_px"]


if __name__ == "__main__":
    radii = all_radii()
    print(f"{'video':<10} {'n_atoms':>9} {'r_min':>7} {'r_mean':>7} {'r_max':>7} {'tol(1.1xr)':>11}")
    print("-" * 60)
    for base, info in sorted(radii.items()):
        print(f"{base:<10} {info['n']:>9} {info['min_px']:>7.2f} {info['mean_px']:>7.2f} "
              f"{info['max_px']:>7.2f} {info['tolerance_px']:>11.2f}")
