# Spatio-Temporal Transformer and Non-linear Extended Kalman Filter

Robust underwater acoustic target detection and tracking for AUV autonomous docking.

This repository is built on Ultralytics YOLO26 and adds a paper-oriented AUV detection-tracking framework with:

- **ST-DFM**: spatio-temporal deformable feature fusion for weak or distorted sonar observations.
- **EKF-CTRV**: extended Kalman filtering with a constant-turn-rate-and-velocity motion model for nonlinear AUV maneuvers.
- **AUVTrack**: an end-to-end tracking pipeline combining EKF-CTRV, bow/body joint IoU association, and docking-trend confidence calibration.
- **Experiment tooling**: six-way ablation, MOT-style tracking metrics, model complexity reporting, and Jacobian validation.

## Main Additions

Important project files:

- `experiments/models/yolo26_stdfm.yaml`: YOLO26 + ST-DFM model config.
- `experiments/ablation_runner.py`: detection, tracking, complexity, and ablation runner.
- `experiments/eval_mot.py`: lightweight MOT-style evaluator.
- `experiments/check_ekf_ctrv_jacobian.py`: analytic-vs-numeric EKF transition Jacobian check.
- `ultralytics/nn/modules/transformer.py`: ST-DFM implementation.
- `ultralytics/trackers/auv_tracker.py`: AUV tracker integration.
- `ultralytics/trackers/modules/ekf_ctrv.py`: EKF-CTRV filter.
- `ultralytics/trackers/modules/iou_association.py`: bow/body joint IoU prior.
- `ultralytics/trackers/modules/trend_confidence.py`: bounded docking-trend confidence calibration.
- `docs/auv_detection_tracking_method.md`: implementation-facing method notes for the paper.

## Quick Start

Install dependencies in your Python environment:

```bash
pip install -e .
```

Run a 10-epoch smoke test for the full method:

```bash
python experiments/ablation_runner.py --epochs 10 --run-ours --data experiments/datasets/auv_sonar.yaml --profile-runs 5
```

Run the full six-way ablation:

```bash
python experiments/ablation_runner.py --epochs 100 --data experiments/datasets/auv_sonar.yaml --profile-runs 20
```

If you have video input and MOT-format ground truth, add tracking metrics:

```bash
python experiments/ablation_runner.py \
  --epochs 100 \
  --data experiments/datasets/auv_sonar.yaml \
  --track-source <video-or-frame-folder> \
  --mot-gt-file <gt.txt> \
  --profile-runs 20
```

Check the EKF-CTRV transition Jacobian:

```bash
python experiments/check_ekf_ctrv_jacobian.py
```

## Dataset

The default dataset config is:

```text
experiments/datasets/auv_sonar.yaml
```

It expects YOLO-format labels with two classes:

- `0: bow`
- `1: body`

Update the `path` field in that YAML file before training on a new machine.

## Outputs

Training and evaluation outputs are written under:

```text
runs/ablation/
```

Large artifacts such as `runs/`, `.pt` weights, images, videos, and caches are ignored by Git.

## Notes

This repository intentionally keeps the Ultralytics codebase structure so the custom modules can reuse existing YOLO training, validation, and tracking interfaces. The project-specific method description is in `docs/auv_detection_tracking_method.md`.

## License

This project inherits the upstream Ultralytics AGPL-3.0 license unless otherwise specified.
