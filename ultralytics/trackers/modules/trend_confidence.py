from __future__ import annotations

import math


class TrendConfidenceCalibrator:
    """Bounded confidence calibration based on docking trend prior."""

    def __init__(
        self,
        enabled: bool = True,
        k: float = 0.1,
        max_abs_delta: float = 0.25,
        sonar_origin_x: float = 0.0,
        sonar_origin_y: float = 0.0,
    ):
        self.enabled = enabled
        self.k = k
        self.max_abs_delta = max_abs_delta
        self.sonar_origin_x = sonar_origin_x
        self.sonar_origin_y = sonar_origin_y

    def _dist(self, x: float, y: float) -> float:
        dx = float(x) - self.sonar_origin_x
        dy = float(y) - self.sonar_origin_y
        return math.sqrt(dx * dx + dy * dy)

    def initialize_track(self, track) -> None:
        if not self.enabled:
            return
        x, y = track.xywh[:2]
        track._trend_t = 0
        track._trend_d0 = self._dist(float(x), float(y))

    def update_track_score(self, track) -> None:
        if not self.enabled:
            return
        if not hasattr(track, "_trend_d0"):
            self.initialize_track(track)
            return

        track._trend_t += 1
        x, y = track.xywh[:2]
        dt = self._dist(float(x), float(y))
        d0 = float(track._trend_d0)
        t = float(track._trend_t)

        # Bounded time weight avoids the exponential blow-up caused by long tracks while preserving the
        # docking prior: objects moving toward the sonar origin receive a positive confidence correction.
        trend = (d0 - dt) / max(d0, 1.0)
        delta = self.k * math.tanh(t / 10.0) * math.tanh(trend)
        delta = max(-self.max_abs_delta, min(self.max_abs_delta, delta))
        track.score = float(max(0.0, min(1.0, float(track.score) + delta)))
