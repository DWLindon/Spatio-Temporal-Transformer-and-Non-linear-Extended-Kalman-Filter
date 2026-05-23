from __future__ import annotations

import numpy as np


class EKFCTRVFilter:
    """EKF-CTRV filter with 9D state: [x, y, v, theta, omega, w, h, vw, vh]."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self._eps = 1e-5

    def initiate(self, measurement: np.ndarray):
        """Initialize from measurement [x, y, w, h]."""
        x, y, w, h = [float(v) for v in measurement]
        mean = np.array([x, y, 0.0, 0.0, 0.0, w, h, 0.0, 0.0], dtype=np.float32)
        std = np.array(
            [
                max(1.0, 0.05 * h),
                max(1.0, 0.05 * h),
                1.0,
                0.5,
                0.3,
                max(1.0, 0.05 * w),
                max(1.0, 0.05 * h),
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def _predict_state(self, mean: np.ndarray):
        x, y, v, theta, omega, w, h, vw, vh = [float(z) for z in mean]
        dt = self.dt

        if abs(omega) < self._eps:
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
        else:
            x_new = x + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            y_new = y + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))

        v_new = v
        theta_new = theta + omega * dt
        omega_new = omega
        w_new = max(1e-3, w + vw * dt)
        h_new = max(1e-3, h + vh * dt)
        vw_new = vw
        vh_new = vh
        return np.array([x_new, y_new, v_new, theta_new, omega_new, w_new, h_new, vw_new, vh_new], dtype=np.float32)

    def _jacobian(self, mean: np.ndarray):
        x, y, v, theta, omega, w, h, vw, vh = [float(z) for z in mean]
        dt = self.dt
        F = np.eye(9, dtype=np.float32)

        if abs(omega) < self._eps:
            F[0, 2] = np.cos(theta) * dt
            F[0, 3] = -v * np.sin(theta) * dt
            F[1, 2] = np.sin(theta) * dt
            F[1, 3] = v * np.cos(theta) * dt
            F[3, 4] = dt
        else:
            s1 = np.sin(theta + omega * dt)
            s0 = np.sin(theta)
            c1 = np.cos(theta + omega * dt)
            c0 = np.cos(theta)
            F[0, 2] = (s1 - s0) / omega
            F[0, 3] = v * (c1 - c0) / omega
            F[0, 4] = (v * dt * c1) / omega - (v * (s1 - s0)) / (omega * omega)
            F[1, 2] = (-c1 + c0) / omega
            F[1, 3] = v * (s1 - s0) / omega
            F[1, 4] = (v * dt * s1) / omega - (v * (-c1 + c0)) / (omega * omega)
            F[3, 4] = dt

        F[5, 7] = dt
        F[6, 8] = dt
        return F

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        F = self._jacobian(mean)
        mean_pred = self._predict_state(mean)

        h = max(1.0, float(mean[6]))
        q = np.array([0.05 * h, 0.05 * h, 0.6, 0.15, 0.1, 0.05 * h, 0.05 * h, 0.6, 0.6], dtype=np.float32)
        Q = np.diag(np.square(q))

        covariance_pred = F @ covariance @ F.T + Q
        return mean_pred, covariance_pred

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray):
        out_means, out_covs = [], []
        for mean, cov in zip(means, covariances):
            pm, pc = self.predict(mean, cov)
            out_means.append(pm)
            out_covs.append(pc)
        return np.asarray(out_means), np.asarray(out_covs)

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        # measurement z = [x, y, w, h]
        H = np.zeros((4, 9), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 5] = 1.0
        H[3, 6] = 1.0

        h = max(1.0, float(mean[6]))
        r = np.array([0.05 * h, 0.05 * h, 0.07 * h, 0.07 * h], dtype=np.float32)
        R = np.diag(np.square(r))

        mean_proj = H @ mean
        cov_proj = H @ covariance @ H.T + R
        return mean_proj, cov_proj, H

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        z = np.asarray(measurement, dtype=np.float32)
        z_mean, S, H = self.project(mean, covariance)

        K = covariance @ H.T @ np.linalg.inv(S)
        innovation = z - z_mean
        new_mean = mean + K @ innovation
        I = np.eye(covariance.shape[0], dtype=np.float32)
        new_cov = (I - K @ H) @ covariance
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        z_mean, S, _H = self.project(mean, covariance)
        if only_position:
            z_mean = z_mean[:2]
            S = S[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - z_mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric != "maha":
            raise ValueError("Invalid distance metric")

        S_inv = np.linalg.inv(S)
        return np.einsum("ij,jk,ik->i", d, S_inv, d)
