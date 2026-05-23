from __future__ import annotations

import numpy as np


class JointIOUAssociator:
    """AUV bow-body joint IoU association filter.

    This module implements the post-detection physical prior described in the previous paper:
    - bow/body area ratio consistency
    - bow inside/intersection with body consistency
    """

    def __init__(
        self,
        enabled: bool = True,
        bow_class_id: int = 0,
        body_class_id: int = 1,
        lambda1: float = 0.12,
        lambda2: float = 0.08,
        lambda_tol: float = 0.6,
    ):
        self.enabled = enabled
        self.bow_class_id = bow_class_id
        self.body_class_id = body_class_id
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_tol = lambda_tol

    @staticmethod
    def _area(xyxy: np.ndarray) -> float:
        w = max(0.0, float(xyxy[2] - xyxy[0]))
        h = max(0.0, float(xyxy[3] - xyxy[1]))
        return w * h

    @staticmethod
    def _inter_area(a: np.ndarray, b: np.ndarray) -> float:
        ix1 = max(float(a[0]), float(b[0]))
        iy1 = max(float(a[1]), float(b[1]))
        ix2 = min(float(a[2]), float(b[2]))
        iy2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        return iw * ih

    def _is_valid_pair(self, bow_xyxy: np.ndarray, body_xyxy: np.ndarray) -> bool:
        body_area = self._area(body_xyxy)
        if body_area <= 1e-6:
            return False

        bow_area = self._area(bow_xyxy)
        inter = self._inter_area(bow_xyxy, body_xyxy)

        r1 = bow_area / body_area
        r2 = inter / body_area
        l1_ok = abs(r1 - self.lambda1) <= max(1e-6, self.lambda1 * self.lambda_tol)
        l2_ok = abs(r2 - self.lambda2) <= max(1e-6, self.lambda2 * self.lambda_tol)
        return l1_ok and l2_ok

    def filter_indices(self, boxes_xyxy: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Return boolean keep mask for detections."""
        n = len(boxes_xyxy)
        if not self.enabled or n == 0:
            return np.ones((n,), dtype=bool)

        bow_ids = np.where(classes.astype(int) == self.bow_class_id)[0]
        body_ids = np.where(classes.astype(int) == self.body_class_id)[0]
        if len(bow_ids) == 0 or len(body_ids) == 0:
            return np.ones((n,), dtype=bool)

        keep = np.ones((n,), dtype=bool)
        valid_bows = set()
        valid_bodies = set()
        for bi in bow_ids:
            bow = boxes_xyxy[bi]
            for boi in body_ids:
                body = boxes_xyxy[boi]
                if self._is_valid_pair(bow, body):
                    valid_bows.add(int(bi))
                    valid_bodies.add(int(boi))

        for bi in bow_ids:
            if int(bi) not in valid_bows:
                keep[bi] = False
        for boi in body_ids:
            if int(boi) not in valid_bodies:
                keep[boi] = False
        return keep
