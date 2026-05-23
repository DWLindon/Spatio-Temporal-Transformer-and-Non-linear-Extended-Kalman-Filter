import argparse
from pathlib import Path

try:
    from local_ultralytics import use_local_ultralytics
except ModuleNotFoundError:
    from experiments.local_ultralytics import use_local_ultralytics

use_local_ultralytics()

from ultralytics import YOLO


def export_track_to_mot(
    model_path: str,
    source: str,
    tracker: str,
    out_txt: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "",
    save_vis: bool = False,
    project: str = "runs/track",
    name: str = "mot_export",
):
    model = YOLO(model_path)
    out_path = Path(out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    lines = []
    results_iter = model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=save_vis,
        project=project,
        name=name,
        persist=True,
        stream=True,
    )
    for res in results_iter:
        frame_id += 1
        boxes = getattr(res, "boxes", None)
        if boxes is None or not len(boxes) or not boxes.is_track:
            continue

        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
        xywh = boxes.xywh.cpu().tolist()
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(xywh)
        clss = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [-1] * len(xywh)
        for tid, b, score, cls in zip(ids, xywh, confs, clss):
            x, y, w, h = [float(v) for v in b]
            left = x - w / 2.0
            top = y - h / 2.0
            lines.append(f"{frame_id},{int(tid)},{left:.6f},{top:.6f},{w:.6f},{h:.6f},{float(score):.6f},{int(cls)},-1,-1")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Export Ultralytics track results to MOT txt.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--tracker", type=str, default="ultralytics/cfg/trackers/auvtrack.yaml")
    p.add_argument("--out-txt", type=str, required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--project", type=str, default="runs/track")
    p.add_argument("--name", type=str, default="mot_export")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_track_to_mot(
        model_path=args.model,
        source=args.source,
        tracker=args.tracker,
        out_txt=args.out_txt,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_vis=args.save_vis,
        project=args.project,
        name=args.name,
    )
