# AUV Detection-Tracking Method Notes

This note records the implementation-facing details needed by the paper so the claims remain reproducible.

## ST-DFM Usage Protocol

`STDFM` is a spatio-temporal deformable feature module inserted in `experiments/models/yolo26_stdfm.yaml`.
It projects the current feature map, predicts offsets and attention weights, samples the current and cached
historical features with `grid_sample`, and fuses them through normalized attention.

Training defaults to single-frame behavior because shuffled image batches do not have reliable temporal continuity.
Static validation should also run with the temporal cache disabled, so unrelated validation images cannot contaminate
each other. Video inference and tracking enable the cache, which is the setting used to support the paper claim that
temporal context repairs weak or distorted sonar observations.

The ablation script enforces this protocol:

- static validation: `STDFM.set_temporal_enabled(False)` unless `--val-use-stdfm-cache` is explicitly passed;
- video tracking: STDFM models run with temporal cache enabled;
- cache reset: every mode switch clears stale temporal features.

## EKF-CTRV State And Transition

The EKF state is:

`X = [x, y, v, theta, omega, w, h, vw, vh]^T`

where `(x, y)` is the box center, `v` is translational speed, `theta` is heading, `omega` is yaw rate,
`(w, h)` is box size, and `(vw, vh)` are size change rates.

For `|omega| > eps`, CTRV prediction is:

`x' = x + v / omega * (sin(theta + omega * dt) - sin(theta))`

`y' = y + v / omega * (-cos(theta + omega * dt) + cos(theta))`

`theta' = theta + omega * dt`

`w' = w + vw * dt`, `h' = h + vh * dt`, while `v`, `omega`, `vw`, and `vh` remain constant.

For `omega -> 0`, the model degenerates to straight-line motion:

`x' = x + v * cos(theta) * dt`

`y' = y + v * sin(theta) * dt`

The analytic transition Jacobian is implemented in `ultralytics/trackers/modules/ekf_ctrv.py`.
Use `python experiments/check_ekf_ctrv_jacobian.py` to compare it against finite differences.

The observation model is:

`z = [x, y, w, h]^T`

with a linear observation matrix selecting state indices `0, 1, 5, 6`. Process and measurement noise scale with the
observed target height to adapt uncertainty to sonar target size.

## Tracking Ablation Protocol

The default ablation suite now contains six groups:

- `baseline_yolo26`: YOLO26 + ByteTrack;
- `ablation_stdfm`: YOLO26 + STDFM + ByteTrack;
- `ablation_ekf`: YOLO26 + EKF-CTRV;
- `ablation_joint_iou`: YOLO26 + EKF-CTRV + bow/body joint IoU;
- `ablation_trend_conf`: YOLO26 + EKF-CTRV + bounded docking-trend confidence;
- `ours_stdfm_ekf_iou_trend`: YOLO26 + STDFM + EKF-CTRV + joint IoU + trend confidence.

When `--track-source` and `--mot-gt-file` are provided, the runner exports MOT-format predictions and reports
`MOTA`, `MOTP`, `IDF1`, `IDP`, `IDR`, tracking precision/recall/F1, `FP`, `FN`, `IDS`, `Frag`, `MT`, `PT`, `ML`,
`FAF`, object/track counts, frame count, and protocol FPS. It also records model complexity fields:
`params`, `gflops`, `infer_ms`, and `model_fps`.

Metric meanings:

- `MOTA`: overall tracking accuracy, penalizing `FN`, `FP`, and `IDS`;
- `MOTP`: mean matched-box IoU, reflecting localization quality after association;
- `IDF1`, `IDP`, `IDR`: identity consistency metrics based on matched IDs;
- `TrackPrecision`, `TrackRecall`, `TrackF1`: detection-association precision, recall, and F1 under the IoU threshold;
- `IDS`: identity switches, and `Frag`: recovered-after-lost trajectory fragments;
- `MT`, `PT`, `ML`: mostly tracked, partially tracked, and mostly lost ground-truth trajectories;
- `FAF`: false alarms per frame;
- `GT`, `Pred`, `Matches`, `GTTracks`, `PredTracks`, `Frames`: scale/context counters for interpreting the rates.

`eval_mot.py` remains a lightweight project evaluator based on greedy IoU matching. For final publication, either
state this protocol explicitly or replace the evaluator with an official MOTChallenge-compatible evaluator.
