from __future__ import annotations

from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .modules import EKFCTRVFilter, IMMUKFFilter, JointIOUAssociator, TrendConfidenceCalibrator
from .utils import matching


class AUVSTrack(STrack):
    """STrack variant with AUV-specific nonlinear motion state space."""

    shared_kalman = EKFCTRVFilter()

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        return self.tlwh_to_xywh(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        if len(self.mean) == 10:
            x, y, _v, _a, _theta, _omega, w, h, _vw, _vh = self.mean
        else:
            x, y, _v, _theta, _omega, w, h, _vw, _vh = self.mean
        return np.asarray([x - w / 2, y - h / 2, max(1e-3, w), max(1e-3, h)], dtype=np.float32)

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[AUVSTrack]):
        if len(stracks) <= 0:
            return
        for st in stracks:
            st.mean, st.covariance = st.kalman_filter.predict(st.mean, st.covariance)

    @staticmethod
    def multi_gmc(stracks: list[AUVSTrack], H: np.ndarray = np.eye(2, 3)):
        if stracks:
            for st in stracks:
                if st.mean is None:
                    continue
                R = H[:2, :2]
                t = H[:2, 2]
                st.mean[:2] = R.dot(st.mean[:2]) + t


class AUVByteTracker(BYTETracker):
    """BYTETracker extension for AUV tracking with paper-aligned postprocessing."""

    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args=args, frame_rate=frame_rate)
        self.motion_filter = str(getattr(args, "motion_filter", "imm_ukf")).lower()
        self.iou_associator = JointIOUAssociator(
            enabled=bool(getattr(args, "use_joint_iou", False)),
            bow_class_id=int(getattr(args, "bow_class_id", 0)),
            body_class_id=int(getattr(args, "body_class_id", 1)),
            lambda1=float(getattr(args, "lambda1", 0.12)),
            lambda2=float(getattr(args, "lambda2", 0.08)),
            lambda_tol=float(getattr(args, "lambda_tol", 0.6)),
        )
        self.trend_calibrator = TrendConfidenceCalibrator(
            enabled=bool(getattr(args, "use_trend_confidence", False)),
            k=float(getattr(args, "trend_k", 0.1)),
            max_abs_delta=float(getattr(args, "trend_max_delta", 0.25)),
            sonar_origin_x=float(getattr(args, "sonar_origin_x", 0.0)),
            sonar_origin_y=float(getattr(args, "sonar_origin_y", 0.0)),
        )

    def get_kalmanfilter(self):
        motion_filter = str(getattr(self.args, "motion_filter", "imm_ukf")).lower()
        if motion_filter in {"ekf", "ekf_ctrv", "ctrv"}:
            return EKFCTRVFilter()
        if motion_filter in {"imm", "ukf", "imm_ukf", "imm-ukf"}:
            return IMMUKFFilter()
        raise ValueError(f"Unsupported AUV motion_filter: {motion_filter}")

    def init_track(self, results, img: np.ndarray | None = None) -> list[AUVSTrack]:
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [AUVSTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def multi_predict(self, tracks: list[AUVSTrack]):
        AUVSTrack.multi_predict(tracks)

    def _filter_by_joint_iou(self, results):
        if len(results) == 0 or not self.iou_associator.enabled:
            return results
        keep = self.iou_associator.filter_indices(results.xyxy, results.cls)
        return results[keep]

    def _apply_trend_confidence(self):
        if not self.trend_calibrator.enabled:
            return
        for track in self.tracked_stracks:
            if not hasattr(track, "_trend_d0"):
                self.trend_calibrator.initialize_track(track)
            else:
                self.trend_calibrator.update_track_score(track)

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        results = self._filter_by_joint_iou(results)
        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep)
        unconfirmed = []
        tracked_stracks: list[AUVSTrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            if self.trend_calibrator.enabled:
                self.trend_calibrator.initialize_track(track)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        self._apply_trend_confidence()
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
