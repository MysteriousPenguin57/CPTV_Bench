"""Phase 3 — evaluator for the constellation/kernel-study grid.

Walks <root>/{condition}/{video}_step{N}_rep{R}/tracks.csv and runs the
IDENTICAL calibrated evaluator path used for Phases 1/2 (phase2c transform
+ frame-step remap + 5.0 px tolerance), then computes PTR + raw
STABLE/SWAP/LOST/MISSING per cell via compute_ptr (RAW proc-frame CSV +
real step — never the remapped CSV).

Emits <root>/phase3_metrics.csv (one row per cell, HOTA/AssA/F1/... +
PTR + transition counts) and prints mean±std per (condition, video, step).
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

PYTHON = r"C:\Users\Kavee\anaconda3\python.exe"
_THIS = Path(__file__).resolve()
_TRACKERS_DIR = _THIS.parent
_V2_DIR = _TRACKERS_DIR.parent.parent
EVALUATOR = _V2_DIR / "evaluation" / "evaluator.py"

if str(_TRACKERS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACKERS_DIR))
from stage2_evaluate_calibrated import load_calibration, GT_BY_BASE, scrape_metrics  # noqa
from compute_ptr import compute_ptr  # noqa

MIN_VALID_BYTES = 50_000
# slit-scene videos (K03/L03) are legitimately near-empty (GT only from
# ~frame 281), so their step-8 tracks are ~36 KB; the frame-tail check below
# is the real completeness test -- the byte floor drops to this secondary
# guard whenever the tail check can run (P6i wave 3).
MIN_VALID_BYTES_SPARSE = 5_000
N_SOURCE_FRAMES = 401


def _is_complete(tracks_csv: Path, step: int) -> bool:
    if not tracks_csv.exists() or tracks_csv.stat().st_size < MIN_VALID_BYTES_SPARSE:
        return False
    exp = (N_SOURCE_FRAMES + step - 1) // step
    try:
        with open(tracks_csv, "rb") as f:
            f.seek(0, 2); size = f.tell(); back = min(size, 4096)
            f.seek(size - back)
            tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
        return int(tail[-1].split(",")[0]) >= exp - 2
    except Exception:
        return False


def _video_base(video: str) -> str:
    # P6J degraded-detection variants carry a __p<pp>s<seed> suffix
    # (e.g. M02_chute_25deg__p20s1); strip it to resolve calibration + GT.
    # Additive: no pre-P6J cell name contains "__p".
    m = re.match(r"^(.+?)__p\d+s\d+$", video)
    if m:
        video = m.group(1)
    if video.startswith("PRIMARY"):
        return "PRIMARY"
    m = re.match(r"^(K\d+).*", video)
    return m.group(1) if m else video


def _remap(in_csv: Path, out_csv: Path, step: int):
    if step <= 1:
        out_csv.write_bytes(in_csv.read_bytes()); return
    with open(in_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        rd = csv.DictReader(fin); fn = rd.fieldnames or []
        wr = csv.DictWriter(fout, fieldnames=fn); wr.writeheader()
        for r in rd:
            try: r["frame"] = str(int(r["frame"]) * step)
            except (KeyError, ValueError): pass
            wr.writerow(r)


def _run_eval(tracks_csv: Path, eval_dir: Path, gt: str, txt_max: float, extra: list):
    eval_dir.mkdir(parents=True, exist_ok=True)
    logp = eval_dir / "eval.log"
    if logp.exists() and logp.stat().st_size > 1000:
        m = scrape_metrics(logp)
        if m: return m
    cmd = [PYTHON, str(EVALUATOR), str(tracks_csv), str(eval_dir / "evaluation.csv"),
           gt, str(txt_max)] + extra
    with open(logp, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(_V2_DIR / "data"))
    return scrape_metrics(logp)


def _parse(name: str):
    m = re.match(r"^(.+)_step(\d+)_rep(\d+)$", name)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_V2_DIR / "data" / "results" / "p3_constellations" / "runs"))
    ap.add_argument("--no-ptr", action="store_true", help="skip the PTR pass (faster)")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[p3-eval] {root} missing"); return 1
    cal = load_calibration()
    rows = []
    for cond_dir in sorted(root.iterdir()):
        if not cond_dir.is_dir(): continue
        condition = cond_dir.name
        for cell in sorted(cond_dir.iterdir()):
            if not cell.is_dir(): continue
            p = _parse(cell.name)
            if not p: continue
            video, step, rep = p
            tcsv = cell / "tracks.csv"
            if not _is_complete(tcsv, step):
                if tcsv.exists():
                    print(f"[p3-eval] skip {condition}/{cell.name} — INCOMPLETE (truncated)")
                continue
            base = _video_base(video)
            crow = cal.get(base)
            if crow is None: continue
            txt_max = float(crow["viewport_y_max"]); scale = float(crow["scale"])
            ox = float(crow["ox"]); oy = float(crow["oy"]); H = float(crow["H_video"])
            eval_ox = ox; eval_oy = H - oy - txt_max * scale
            gt = GT_BY_BASE.get(base, "blob_data 3.txt")
            remapped = cell / "tracks_remapped.csv"
            _remap(tcsv, remapped, step)
            m = _run_eval(remapped, cell / "eval_phase2c", gt, txt_max,
                          [f"--eval-scale={scale}", f"--eval-offset-x={eval_ox}",
                           f"--eval-offset-y={eval_oy}"])
            row = {"condition": condition, "video": video, "step": step, "rep": rep,
                   "hota": m.get("hota"), "assa": m.get("assa"), "f1": m.get("f1"),
                   "idf1": m.get("idf1"), "recall": m.get("recall"),
                   "fragmentation": m.get("fragmentation"),
                   "ptr": None, "stable": None, "swap": None, "lost": None, "missing": None}
            if not args.no_ptr:
                try:
                    pr = compute_ptr(str(tcsv), video, step)  # RAW CSV + real step
                    if pr:
                        row.update({"ptr": f"{pr['ptr']:.6f}", "stable": pr["STABLE"],
                                    "swap": pr["SWAP"], "lost": pr["LOST"],
                                    "missing": pr["MISSING"]})
                except Exception as e:
                    print(f"[p3-eval] PTR failed for {condition}/{cell.name}: {e}")
            rows.append(row)
            print(f"[p3-eval] {condition:<18} {video:<22} step={step:<2} rep={rep}  "
                  f"HOTA={m.get('hota','?')} AssA={m.get('assa','?')} F1={m.get('f1','?')} "
                  f"PTR={row['ptr'] or '?'} SWAP={row['swap'] if row['swap'] is not None else '?'}")

    out = root / "phase3_metrics.csv"
    cols = ["condition", "video", "step", "rep", "hota", "assa", "f1", "idf1",
            "recall", "fragmentation", "ptr", "stable", "swap", "lost", "missing"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"\n[p3-eval] wrote {out} ({len(rows)} rows)")

    def fnum(x):
        try: return float(x)
        except (TypeError, ValueError): return None
    agg = {}
    for r in rows:
        key = (r["condition"], r["video"], r["step"])
        agg.setdefault(key, {"hota": [], "assa": [], "ptr": [], "swap": []})
        for k in ("hota", "assa", "ptr", "swap"):
            v = fnum(r[k])
            if v is not None: agg[key][k].append(v)
    print("\n=== Phase-3 mean±std per condition × video × step ===")
    print(f"{'condition':<18} {'video':<22} {'step':>4} {'n':>2} {'HOTA':>16} {'AssA':>16} {'PTR':>16} {'SWAP':>12}")
    for (cond, vid, step) in sorted(agg.keys(), key=lambda x: (x[0], x[1], x[2])):
        d = agg[(cond, vid, step)]
        def ms(vals, fmt="{:.4f}"):
            if not vals: return "  -  "
            mu = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return f"{fmt.format(mu)}+/-{fmt.format(sd)}"
        print(f"{cond:<18} {vid:<22} {step:>4} {len(d['hota']):>2} "
              f"{ms(d['hota']):>16} {ms(d['assa']):>16} {ms(d['ptr']):>16} "
              f"{ms(d['swap'], '{:.0f}'):>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
