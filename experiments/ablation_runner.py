import argparse
import csv
import time
from pathlib import Path

import torch

try:
    from local_ultralytics import use_local_ultralytics
except ModuleNotFoundError:
    from experiments.local_ultralytics import use_local_ultralytics

REPO_ROOT = use_local_ultralytics()

from ultralytics import YOLO
from ultralytics.nn.modules import STDFM
from ultralytics.utils.torch_utils import get_flops, get_num_params, select_device

try:
    from eval_mot import evaluate_mot
    from export_track_mot import export_track_to_mot
except ModuleNotFoundError:
    from experiments.eval_mot import evaluate_mot
    from experiments.export_track_mot import export_track_to_mot


def parse_args():
    p = argparse.ArgumentParser(description="Run AUV ablation experiments with detection, tracking, and complexity metrics.")
    p.add_argument("--data", type=str, default="experiments/datasets/auv_sonar.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/ablation")
    p.add_argument("--run-baseline", action="store_true")
    p.add_argument("--run-stdfm", action="store_true")
    p.add_argument("--run-ekf", action="store_true")
    p.add_argument("--run-imm-ukf", action="store_true")
    p.add_argument("--run-joint-iou", action="store_true")
    p.add_argument("--run-trend", action="store_true")
    p.add_argument("--run-ours", action="store_true")
    p.add_argument("--baseline-model", type=str, default="ultralytics/cfg/models/26/yolo26.yaml")
    p.add_argument("--stdfm-model", type=str, default="experiments/models/yolo26_stdfm.yaml")
    p.add_argument("--tracker-baseline", type=str, default="ultralytics/cfg/trackers/bytetrack.yaml")
    p.add_argument("--tracker-ekf", type=str, default="ultralytics/cfg/trackers/auvtrack_ekf.yaml")
    p.add_argument("--tracker-imm-ukf", type=str, default="ultralytics/cfg/trackers/auvtrack_imm_ukf.yaml")
    p.add_argument("--tracker-joint-iou", type=str, default="ultralytics/cfg/trackers/auvtrack_joint_iou.yaml")
    p.add_argument("--tracker-trend", type=str, default="ultralytics/cfg/trackers/auvtrack_trend.yaml")
    p.add_argument("--tracker-ours", type=str, default="ultralytics/cfg/trackers/auvtrack.yaml")
    p.add_argument("--track-source", type=str, default="", help="Optional video/image/folder for tracking ablation")
    p.add_argument("--track-conf", type=float, default=0.25)
    p.add_argument("--track-iou", type=float, default=0.45)
    p.add_argument("--save-track", action="store_true")
    p.add_argument("--summary-file", type=str, default="runs/ablation/summary.csv")
    p.add_argument("--mot-gt-file", type=str, default="", help="Optional GT MOT txt for tracking metrics")
    p.add_argument("--mot-pred-dir", type=str, default="", help="Directory for exported MOT prediction txt files")
    p.add_argument("--mot-iou-thr", type=float, default=0.5)
    p.add_argument("--mot-fps", type=float, default=30.0)
    p.add_argument("--val-use-stdfm-cache", action="store_true", help="Use STDFM temporal cache during static val")
    p.add_argument("--profile-warmup", type=int, default=5)
    p.add_argument("--profile-runs", type=int, default=20)
    return p.parse_args()


def resolve_repo_path(path: str) -> str:
    """Resolve a user path against the repository root unless it is already absolute or empty."""
    if not path:
        return path
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def normalize_args(args):
    """Normalize relative paths so Ultralytics does not nest them under its default runs directory."""
    args.data = resolve_repo_path(args.data)
    args.project = resolve_repo_path(args.project)
    args.summary_file = resolve_repo_path(args.summary_file)
    args.baseline_model = resolve_repo_path(args.baseline_model)
    args.stdfm_model = resolve_repo_path(args.stdfm_model)
    args.tracker_baseline = resolve_repo_path(args.tracker_baseline)
    args.tracker_ekf = resolve_repo_path(args.tracker_ekf)
    args.tracker_imm_ukf = resolve_repo_path(args.tracker_imm_ukf)
    args.tracker_joint_iou = resolve_repo_path(args.tracker_joint_iou)
    args.tracker_trend = resolve_repo_path(args.tracker_trend)
    args.tracker_ours = resolve_repo_path(args.tracker_ours)
    args.track_source = resolve_repo_path(args.track_source)
    args.mot_gt_file = resolve_repo_path(args.mot_gt_file)
    args.mot_pred_dir = resolve_repo_path(args.mot_pred_dir)
    return args


def configure_stdfm(model, temporal_enabled: bool) -> None:
    """Set all STDFM modules to static or temporal mode and clear stale cache."""
    for module in model.model.modules():
        if isinstance(module, STDFM):
            module.set_temporal_enabled(temporal_enabled)
            module.reset_cache()


def train_one(model_cfg: str, args, name: str):
    model = YOLO(model_cfg)
    configure_stdfm(model, temporal_enabled=False)
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=name,
    )
    return Path(train_results.save_dir) / "weights" / "best.pt"


def val_one(weights_path: str, args):
    model = YOLO(str(weights_path))
    configure_stdfm(model, temporal_enabled=args.val_use_stdfm_cache)
    val_results = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)
    configure_stdfm(model, temporal_enabled=False)
    rd = getattr(val_results, "results_dict", {}) or {}
    return {
        "map50": rd.get("metrics/mAP50(B)", ""),
        "map50_95": rd.get("metrics/mAP50-95(B)", ""),
        "precision": rd.get("metrics/precision(B)", ""),
        "recall": rd.get("metrics/recall(B)", ""),
    }


def profile_one(weights_path: str, args, use_temporal: bool):
    """Return Params, FLOPs, and raw model forward latency for paper complexity analysis."""
    model = YOLO(str(weights_path))
    configure_stdfm(model, temporal_enabled=use_temporal)
    device = select_device(args.device or "cpu")
    net = model.model.to(device).eval()

    try:
        params = get_num_params(net)
        flops = get_flops(net, imgsz=args.imgsz)
    except Exception:
        params, flops = "", ""

    if args.profile_runs <= 0:
        return {"params": params, "gflops": flops, "infer_ms": "", "model_fps": ""}

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        for _ in range(max(0, args.profile_warmup)):
            configure_stdfm(model, temporal_enabled=use_temporal)
            _ = net(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.profile_runs):
            _ = net(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    infer_ms = elapsed * 1000.0 / max(1, args.profile_runs)
    fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
    return {"params": params, "gflops": flops, "infer_ms": infer_ms, "model_fps": fps}


def track_one(weights_path: str, tracker_cfg: str, args, name: str, use_temporal: bool):
    if not args.track_source:
        return "", ""
    model = YOLO(str(weights_path))
    configure_stdfm(model, temporal_enabled=use_temporal)
    track_results = model.track(
        source=args.track_source,
        tracker=tracker_cfg,
        conf=args.track_conf,
        iou=args.track_iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save_track,
        project=str(Path(args.project) / "track"),
        name=name,
        persist=True,
    )

    pred_txt = ""
    if args.mot_pred_dir:
        pred_txt_path = Path(args.mot_pred_dir) / f"{name}.txt"
        pred_txt = export_track_to_mot(
            model_path=str(weights_path),
            source=args.track_source,
            tracker=tracker_cfg,
            out_txt=str(pred_txt_path),
            conf=args.track_conf,
            iou=args.track_iou,
            imgsz=args.imgsz,
            device=args.device,
            save_vis=False,
            project=str(Path(args.project) / "track"),
            name=f"{name}_mot_export",
        )
    if track_results and hasattr(track_results[0], "save_dir"):
        return str(track_results[0].save_dir), pred_txt
    return "", pred_txt


def append_summary(summary_file: Path, row: dict):
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "model_cfg",
        "weights",
        "tracker_cfg",
        "stdfm_temporal_in_val",
        "stdfm_temporal_in_track",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "track_run_dir",
        "MOTA",
        "MOTP",
        "IDF1",
        "IDP",
        "IDR",
        "TrackPrecision",
        "TrackRecall",
        "TrackF1",
        "FP",
        "FN",
        "FPS",
        "IDS",
        "Frag",
        "MT",
        "PT",
        "ML",
        "FAF",
        "GT",
        "Pred",
        "Matches",
        "GTTracks",
        "PredTracks",
        "Frames",
        "params",
        "gflops",
        "infer_ms",
        "model_fps",
    ]
    if summary_file.exists():
        with summary_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields != fields:
            with summary_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({k: old_row.get(k, "") for k in fields})

    write_header = not summary_file.exists()
    with summary_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def run_experiment(exp_name: str, model_cfg: str, tracker_cfg: str, args, summary_file: Path, use_stdfm_track: bool):
    weights = train_one(model_cfg, args, exp_name)
    val_metrics = val_one(weights, args)
    complexity = profile_one(weights, args, use_temporal=use_stdfm_track)
    track_dir, pred_txt = track_one(weights, tracker_cfg, args, exp_name, use_temporal=use_stdfm_track)

    mot_metrics = {}
    if args.mot_gt_file and pred_txt:
        pred_file = Path(pred_txt)
        if pred_file.exists():
            mot_metrics = evaluate_mot(args.mot_gt_file, str(pred_file), iou_thr=args.mot_iou_thr, fps=args.mot_fps)

    append_summary(
        summary_file,
        {
            "experiment": exp_name,
            "model_cfg": model_cfg,
            "weights": str(weights),
            "tracker_cfg": tracker_cfg,
            "stdfm_temporal_in_val": bool(args.val_use_stdfm_cache),
            "stdfm_temporal_in_track": bool(use_stdfm_track),
            **val_metrics,
            "track_run_dir": track_dir,
            "MOTA": mot_metrics.get("MOTA", ""),
            "MOTP": mot_metrics.get("MOTP", ""),
            "IDF1": mot_metrics.get("IDF1", ""),
            "IDP": mot_metrics.get("IDP", ""),
            "IDR": mot_metrics.get("IDR", ""),
            "TrackPrecision": mot_metrics.get("TrackPrecision", ""),
            "TrackRecall": mot_metrics.get("TrackRecall", ""),
            "TrackF1": mot_metrics.get("TrackF1", ""),
            "FP": mot_metrics.get("FP", ""),
            "FN": mot_metrics.get("FN", ""),
            "FPS": mot_metrics.get("FPS", ""),
            "IDS": mot_metrics.get("IDS", ""),
            "Frag": mot_metrics.get("Frag", ""),
            "MT": mot_metrics.get("MT", ""),
            "PT": mot_metrics.get("PT", ""),
            "ML": mot_metrics.get("ML", ""),
            "FAF": mot_metrics.get("FAF", ""),
            "GT": mot_metrics.get("GT", ""),
            "Pred": mot_metrics.get("Pred", ""),
            "Matches": mot_metrics.get("Matches", ""),
            "GTTracks": mot_metrics.get("GTTracks", ""),
            "PredTracks": mot_metrics.get("PredTracks", ""),
            "Frames": mot_metrics.get("Frames", ""),
            **complexity,
        },
    )


def main():
    args = normalize_args(parse_args())
    Path(args.project).mkdir(parents=True, exist_ok=True)
    summary_file = Path(args.summary_file)
    if args.mot_gt_file and args.track_source and not args.mot_pred_dir:
        args.mot_pred_dir = str(Path(args.project) / "mot_preds")

    if not any(
        [
            args.run_baseline,
            args.run_stdfm,
            args.run_ekf,
            args.run_imm_ukf,
            args.run_joint_iou,
            args.run_trend,
            args.run_ours,
        ]
    ):
        args.run_baseline = True
        args.run_stdfm = True
        args.run_ekf = True
        args.run_imm_ukf = True
        args.run_joint_iou = True
        args.run_trend = True
        args.run_ours = True

    if args.run_baseline:
        run_experiment("baseline_yolo26", args.baseline_model, args.tracker_baseline, args, summary_file, False)
    if args.run_stdfm:
        run_experiment("ablation_stdfm", args.stdfm_model, args.tracker_baseline, args, summary_file, True)
    if args.run_ekf:
        run_experiment("ablation_ekf", args.baseline_model, args.tracker_ekf, args, summary_file, False)
    if args.run_imm_ukf:
        run_experiment("ablation_imm_ukf", args.baseline_model, args.tracker_imm_ukf, args, summary_file, False)
    if args.run_joint_iou:
        run_experiment("ablation_joint_iou", args.baseline_model, args.tracker_joint_iou, args, summary_file, False)
    if args.run_trend:
        run_experiment("ablation_trend_conf", args.baseline_model, args.tracker_trend, args, summary_file, False)
    if args.run_ours:
        run_experiment("ours_stdfm_imm_ukf_iou_trend", args.stdfm_model, args.tracker_ours, args, summary_file, True)


if __name__ == "__main__":
    main()
