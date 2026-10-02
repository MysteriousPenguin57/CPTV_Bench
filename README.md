# CPTV_bench: detection-matched tracking benchmark pack

Everything needed to score a particle tracker on the 71 synthetic flow-displacement cells of the CPTV
paper (`master_table_v4`), **without the videos**: the shared detections (with a 3-bin colour label), the
ground truth, the calibration transforms and an unmodified snapshot of the evaluator. Synthetic data only
(LAMMPS simulations rendered to video); no real footage.

## Layout

```
cells.csv                 the 71 cells: cell, video, step, base, detections, gt, calib_group, eval_mode,
                          d_per_frame, ship_hota, ship_ptr (CPTV ship = capauto), source_detections
detections_col/           <video>_step<k>_detections.csv : frame,x,y,r,score,colour
score.py                  score one tracks.csv on one cell (thin wrapper, see below)
v2_snapshot/              UNMODIFIED copies, in the research repo's own layout (the copied modules locate
                          GT and the evaluator relative to their own path, so the layout must stay):
  evaluation/evaluator.py                     HOTA / AssA / F1 / IDF1 evaluator (positional-argument CLI)
  src/trackers/*.py                           phase3_evaluator, stage2_evaluate_calibrated, compute_ptr,
                                              motion_success_histogram, gt_radii, phase5_eval_primary,
                                              phase5_res_scalefair_eval, phase6e_b4k_scalefair
  data/blob_data 3.txt, data/group_k/...      ground truth (LAMMPS dumps)  <- "gt"
  LAMMPS downloads/test_suite_output/<video>/<video>_data.txt   ground truth of the M and RES videos
  detection_benchmark/results/<group>/calibration_transforms_by_video.csv   <- "calibration"
                          groups: calibration_phase2c (PRIMARY), calibration_groupm_v2 (M02-M06),
                          calibration_res_scaling (RES_*), calibration_p6i (K01-K06, L01-L03)
tools/build_pack.py       how the pack was built from the research tree (needs that tree; not needed to score)
PROVENANCE.md             source path, v2 commit and sha256 of every copied file
```

## Detections

`detections_col/<video>_step<k>_detections.csv`, one row per detected particle:

| column | meaning |
|---|---|
| `frame` | **processed** frame index of that cell (0, 1, ...; source frame = frame x step) |
| `x`, `y`, `r` | centre and radius, pixels (origin top-left) |
| `score` | detector confidence |
| `colour` | 0/1/2: argmax of the median BGR in a 3x3 window at the centre (the label every colour-using baseline gets) |

These are the exact Cellpose-SAM detections every tracker in the paper consumed (detection-matched protocol), plus
`colour`. A tracker must link **these** detections and must not run its own detector.

## Scoring

Requirements: Python >= 3.10, `pip install -r requirements.txt` (numpy, pandas, scipy, matplotlib, seaborn).

```
python score.py --cell PRIMARY_s4 --tracks my_tracks.csv [--out-dir score_out]
```

`my_tracks.csv` must have the columns `frame,track_id,x,y,r,missed` (frame = the cell's processed index, as in
the detections; only rows with `missed == 0` are scored). Output, one line per call, also appended to
`<out-dir>/scores.csv`:

| column | meaning |
|---|---|
| `cell`, `eval_mode` | the cell; `5px` (5 px match tolerance) or `scalefair` (RES_B_4K: coordinates halved, i.e. 10 px at native scale) |
| `hota`, `assa`, `f1`, `idf1`, `recall`, `fragmentation` | evaluator.py metrics |
| `ptr` | per-transition identity retention, `STABLE / (STABLE + SWAP + LOST + MISSING)` (5 px; 10 px for scalefair) |
| `STABLE`, `SWAP`, `LOST`, `MISSING` | transition counts behind `ptr` |

The wrapper reproduces the published pipeline (`phase3_evaluator` for 5 px cells, `phase6e_b4k_scalefair` for
RES_B_4K). It changes nothing in the copied files; at run time it only points `CPTV_CALIB_CSV` at the pack's
calibration file and replaces their hard-coded Windows interpreter path with the running interpreter.
Check: scoring the published CPTV tracks reproduces `ship_hota` / `ship_ptr` in `cells.csv` to 4 decimals
(verified for PRIMARY_s4, M02_chute_25deg_s4, K03_s1, RES_B_4K_s4).

## Notes

- Cell names: `<video>_s<step>`, see `cells.csv`. K03/L03 (slit) have no particles before source frame 281.
- Runtime: about 1-3 min per cell on one CPU core (evaluator.py).
- Licence: research data and code from the CPTV project (BSD-3-Clause, as the CPTV repository).
