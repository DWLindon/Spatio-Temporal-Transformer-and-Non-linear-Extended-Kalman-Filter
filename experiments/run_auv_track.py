import argparse
from pathlib import Path

try:
    from local_ultralytics import use_local_ultralytics
except ModuleNotFoundError:
    from experiments.local_ultralytics import use_local_ultralytics

use_local_ultralytics()

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run AUV tracking with YOLOv26 + custom auvtrack.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv26 weights, e.g. runs/detect/train/weights/best.pt")
    parser.add_argument("--source", type=str, required=True, help="Video/image/folder source path")
    parser.add_argument(
        "--tracker",
        type=str,
        default="ultralytics/cfg/trackers/auvtrack.yaml",
        help="Tracker yaml path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="", help="Device id, e.g. 0 or cpu")
    parser.add_argument("--save", action="store_true", help="Save tracking video")
    parser.add_argument("--project", type=str, default="runs/track", help="Output project directory")
    parser.add_argument("--name", type=str, default="auvtrack", help="Output run name")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        project=str(project),
        name=args.name,
        persist=True,
    )


if __name__ == "__main__":
    main()
