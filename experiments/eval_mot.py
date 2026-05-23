import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_mot_txt(path: str):
    frames = defaultdict(list)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            frames[frame].append({"id": tid, "bbox": np.array([x, y, w, h], dtype=np.float32), "conf": conf})
    return frames


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-12
    return inter / union


def _greedy_match(gt_objs, pr_objs, iou_thr=0.5):
    pairs = []
    for gi, g in enumerate(gt_objs):
        for pi, p in enumerate(pr_objs):
            iou = _iou_xywh(g["bbox"], p["bbox"])
            if iou >= iou_thr:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches, used_g, used_p


def evaluate_mot(gt_path: str, pred_path: str, iou_thr: float = 0.5, fps: float = 30.0):
    """Evaluate MOT-style metrics with project-local greedy IoU matching.

    This is a lightweight engineering evaluator. For final paper submission, report this protocol explicitly or
    replace it with an official MOTChallenge-compatible evaluator.
    """
    gt = _load_mot_txt(gt_path)
    pr = _load_mot_txt(pred_path)
    frames = sorted(set(gt.keys()) | set(pr.keys()))

    total_gt = 0
    total_pred = 0
    fp = fn = ids = 0
    idtp = idfp = idfn = 0
    total_iou = 0.0
    match_count = 0

    # gt_id -> last pred_id mapping for IDS counting
    last_match = {}
    gt_total = defaultdict(int)
    gt_matched = defaultdict(int)
    gt_last_seen_matched = defaultdict(bool)
    gt_fragments = defaultdict(int)
    pred_ids = set()

    start_frame = frames[0] if frames else 1
    end_frame = frames[-1] if frames else 1
    num_frames = max(1, end_frame - start_frame + 1)

    for frame in frames:
        gt_objs = gt.get(frame, [])
        pr_objs = pr.get(frame, [])
        total_gt += len(gt_objs)
        total_pred += len(pr_objs)
        pred_ids.update(p["id"] for p in pr_objs)

        matches, used_g, used_p = _greedy_match(gt_objs, pr_objs, iou_thr=iou_thr)
        fn += len(gt_objs) - len(used_g)
        fp += len(pr_objs) - len(used_p)
        match_count += len(matches)
        total_iou += sum(iou for *_rest, iou in matches)

        # IDF1 components (simple approximation with per-frame matching)
        idtp += len(matches)
        idfn += len(gt_objs) - len(used_g)
        idfp += len(pr_objs) - len(used_p)

        matched_gt_ids = set()
        for gi, pi, _ in matches:
            gt_id = gt_objs[gi]["id"]
            pr_id = pr_objs[pi]["id"]
            matched_gt_ids.add(gt_id)
            if gt_id in last_match and last_match[gt_id] != pr_id:
                ids += 1
            last_match[gt_id] = pr_id

        for gi, g in enumerate(gt_objs):
            gt_id = g["id"]
            is_matched = gi in used_g
            gt_total[gt_id] += 1
            gt_matched[gt_id] += int(is_matched)
            if is_matched and gt_id in gt_last_seen_matched and not gt_last_seen_matched[gt_id]:
                gt_fragments[gt_id] += 1
            gt_last_seen_matched[gt_id] = is_matched

    mota = 1.0 - ((fn + fp + ids) / total_gt) if total_gt > 0 else 0.0
    motp = total_iou / match_count if match_count > 0 else 0.0
    idf1 = (2.0 * idtp) / (2.0 * idtp + idfp + idfn + 1e-12)
    precision = match_count / (match_count + fp + 1e-12)
    recall = match_count / (match_count + fn + 1e-12)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    idp = idtp / (idtp + idfp + 1e-12)
    idr = idtp / (idtp + idfn + 1e-12)
    faf = fp / num_frames

    gt_track_count = len(gt_total)
    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0
    for gt_id, total in gt_total.items():
        coverage = gt_matched[gt_id] / max(1, total)
        if coverage >= 0.8:
            mostly_tracked += 1
        elif coverage <= 0.2:
            mostly_lost += 1
        else:
            partially_tracked += 1

    fps_est = fps if fps > 0 else num_frames / max(1e-12, num_frames / 30.0)
    return {
        "MOTA": mota * 100.0,
        "MOTP": motp * 100.0,
        "IDF1": idf1 * 100.0,
        "IDP": idp * 100.0,
        "IDR": idr * 100.0,
        "TrackPrecision": precision * 100.0,
        "TrackRecall": recall * 100.0,
        "TrackF1": f1 * 100.0,
        "FP": float(fp),
        "FN": float(fn),
        "IDS": float(ids),
        "Frag": float(sum(gt_fragments.values())),
        "MT": float(mostly_tracked),
        "PT": float(partially_tracked),
        "ML": float(mostly_lost),
        "FAF": float(faf),
        "GT": float(total_gt),
        "Pred": float(total_pred),
        "Matches": float(match_count),
        "GTTracks": float(gt_track_count),
        "PredTracks": float(len(pred_ids)),
        "Frames": float(num_frames),
        "FPS": float(fps_est),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MOT metrics from GT and prediction MOT txt files.")
    p.add_argument("--gt", type=str, required=True, help="GT MOT txt path")
    p.add_argument("--pred", type=str, required=True, help="Prediction MOT txt path")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    m = evaluate_mot(args.gt, args.pred, iou_thr=args.iou_thr, fps=args.fps)
    for k, v in m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
