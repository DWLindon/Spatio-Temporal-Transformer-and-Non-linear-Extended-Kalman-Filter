from __future__ import annotations

import numpy as np


class IMMUKFFilter:
    """IMM-UKF filter with CV, CA, and CTRA motion models.

    Common 10D state:
        [x, y, v, a, theta, omega, w, h, vw, vh]

    The filter stores per-model UKF states in the covariance object while exposing the mixed mean as the public track
    state. This keeps compatibility with the existing ByteTrack/STRack interface.
    """

    model_names = ("cv", "ca", "ctra")

    def __init__(self, dt: float = 1.0, alpha: float = 0.7, beta: float = 2.0, kappa: float = 0.0):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.dim_x = 10
        self.dim_z = 4
        self._eps = 1e-6
        self.transition = np.array(
            [
                [0.90, 0.07, 0.03],
                [0.07, 0.88, 0.05],
                [0.04, 0.08, 0.88],
            ],
            dtype=np.float64,
        )

    def initiate(self, measurement: np.ndarray):
        """Initialize from measurement [x, y, w, h]."""
        x, y, w, h = [float(v) for v in measurement]
        mean = np.array([x, y, 0.0, 0.0, 0.0, 0.0, w, h, 0.0, 0.0], dtype=np.float64)
        std = np.array(
            [
                max(1.0, 0.05 * h),
                max(1.0, 0.05 * h),
                1.0,
                0.7,
                0.5,
                0.3,
                max(1.0, 0.05 * w),
                max(1.0, 0.05 * h),
                1.0,
                1.0,
            ],
            dtype=np.float64,
        )
        cov = np.diag(np.square(std))
        imm_state = {
            "means": np.repeat(mean[None, :], len(self.model_names), axis=0),
            "covariances": np.repeat(cov[None, :, :], len(self.model_names), axis=0),
            "probs": np.array([0.45, 0.35, 0.20], dtype=np.float64),
        }
        mean, mixed_cov = self._combine_estimate(imm_state)
        imm_state["mixed_covariance"] = mixed_cov
        return mean.astype(np.float32), imm_state

    def _as_state10(self, mean: np.ndarray) -> np.ndarray:
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        if mean.size == self.dim_x:
            return mean.copy()
        if mean.size == 9:
            x, y, v, theta, omega, w, h, vw, vh = mean
            return np.array([x, y, v, 0.0, theta, omega, w, h, vw, vh], dtype=np.float64)
        raise ValueError(f"IMMUKFFilter expects 9D legacy or 10D state, got {mean.size}D")

    def _as_imm_state(self, mean: np.ndarray, covariance) -> dict:
        if isinstance(covariance, dict) and {"means", "covariances", "probs"}.issubset(covariance):
            return {
                "means": np.asarray(covariance["means"], dtype=np.float64),
                "covariances": np.asarray(covariance["covariances"], dtype=np.float64),
                "probs": self._normalize(np.asarray(covariance["probs"], dtype=np.float64)),
            }
        mean10 = self._as_state10(mean)
        if isinstance(covariance, np.ndarray) and covariance.shape == (self.dim_x, self.dim_x):
            cov10 = covariance.astype(np.float64)
        else:
            cov10 = np.eye(self.dim_x, dtype=np.float64)
            if isinstance(covariance, np.ndarray):
                n = min(covariance.shape[0], self.dim_x)
                cov10[:n, :n] = covariance[:n, :n]
        return {
            "means": np.repeat(mean10[None, :], len(self.model_names), axis=0),
            "covariances": np.repeat(self._symmetrize(cov10)[None, :, :], len(self.model_names), axis=0),
            "probs": np.ones(len(self.model_names), dtype=np.float64) / len(self.model_names),
        }

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(x, dtype=np.float64), 1e-12)
        return x / np.sum(x)

    @staticmethod
    def _symmetrize(cov: np.ndarray) -> np.ndarray:
        return 0.5 * (cov + cov.T)

    def _ensure_spd(self, cov: np.ndarray) -> np.ndarray:
        cov = self._symmetrize(cov)
        jitter = 1e-6
        eye = np.eye(cov.shape[0], dtype=np.float64)
        for _ in range(6):
            try:
                np.linalg.cholesky(cov + jitter * eye)
                return cov + jitter * eye
            except np.linalg.LinAlgError:
                jitter *= 10.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _weights(self, n: int):
        lam = self.alpha * self.alpha * (n + self.kappa) - n
        c = n + lam
        wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        wc = wm.copy()
        wm[0] = lam / c
        wc[0] = lam / c + (1.0 - self.alpha * self.alpha + self.beta)
        return wm, wc, c

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        mean = np.asarray(mean, dtype=np.float64)
        cov = self._ensure_spd(cov)
        n = mean.size
        _wm, _wc, c = self._weights(n)
        chol = np.linalg.cholesky(c * cov)
        points = [mean]
        for i in range(n):
            points.append(mean + chol[:, i])
            points.append(mean - chol[:, i])
        return np.asarray(points, dtype=np.float64)

    def _process_noise(self, mean: np.ndarray, model: str) -> np.ndarray:
        h = max(1.0, float(mean[7]))
        w = max(1.0, float(mean[6]))
        q = np.array([0.04 * h, 0.04 * h, 0.45, 0.45, 0.10, 0.12, 0.04 * w, 0.04 * h, 0.45, 0.45])
        if model == "cv":
            q[3] = 0.90
            q[5] = 0.24
        elif model == "ca":
            q[3] = 0.35
            q[5] = 0.20
        else:  # ctra
            q[3] = 0.50
            q[5] = 0.10
            q[4] = 0.14
        return np.diag(np.square(q))

    def _measurement_noise(self, mean: np.ndarray) -> np.ndarray:
        h = max(1.0, float(mean[7]))
        w = max(1.0, float(mean[6]))
        r = np.array([0.05 * h, 0.05 * h, 0.07 * w, 0.07 * h], dtype=np.float64)
        return np.diag(np.square(r))

    def _fx(self, state: np.ndarray, model: str) -> np.ndarray:
        x, y, v, a, theta, omega, w, h, vw, vh = [float(z) for z in state]
        dt = self.dt
        if model == "cv":
            a = 0.0
            omega = 0.0
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
        elif model == "ca":
            omega = 0.0
            distance = v * dt + 0.5 * a * dt * dt
            x += distance * np.cos(theta)
            y += distance * np.sin(theta)
            v += a * dt
        else:
            steps = 4
            sub_dt = dt / steps
            for _ in range(steps):
                distance = v * sub_dt + 0.5 * a * sub_dt * sub_dt
                x += distance * np.cos(theta)
                y += distance * np.sin(theta)
                v += a * sub_dt
                theta += omega * sub_dt
        w = max(1e-3, w + vw * dt)
        h = max(1e-3, h + vh * dt)
        return np.array([x, y, v, a, theta, omega, w, h, vw, vh], dtype=np.float64)

    @staticmethod
    def _hx(state: np.ndarray) -> np.ndarray:
        return np.asarray([state[0], state[1], state[6], state[7]], dtype=np.float64)

    def _ukf_predict(self, mean: np.ndarray, cov: np.ndarray, model: str):
        sigma = self._sigma_points(mean, cov)
        propagated = np.asarray([self._fx(point, model) for point in sigma], dtype=np.float64)
        wm, wc, _c = self._weights(mean.size)
        mean_pred = np.sum(wm[:, None] * propagated, axis=0)
        cov_pred = np.zeros_like(cov, dtype=np.float64)
        for i, point in enumerate(propagated):
            d = point - mean_pred
            cov_pred += wc[i] * np.outer(d, d)
        cov_pred += self._process_noise(mean_pred, model)
        return mean_pred, self._ensure_spd(cov_pred)

    def _ukf_update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        sigma = self._sigma_points(mean, cov)
        z_sigma = np.asarray([self._hx(point) for point in sigma], dtype=np.float64)
        wm, wc, _c = self._weights(mean.size)
        z_mean = np.sum(wm[:, None] * z_sigma, axis=0)
        S = np.zeros((self.dim_z, self.dim_z), dtype=np.float64)
        Pxz = np.zeros((self.dim_x, self.dim_z), dtype=np.float64)
        for i, point in enumerate(sigma):
            dz = z_sigma[i] - z_mean
            dx = point - mean
            S += wc[i] * np.outer(dz, dz)
            Pxz += wc[i] * np.outer(dx, dz)
        S += self._measurement_noise(mean)
        S = self._ensure_spd(S)
        K = Pxz @ np.linalg.inv(S)
        innovation = np.asarray(measurement, dtype=np.float64) - z_mean
        mean_new = mean + K @ innovation
        cov_new = self._ensure_spd(cov - K @ S @ K.T)
        likelihood = self._gaussian_likelihood(innovation, S)
        return mean_new, cov_new, likelihood

    @staticmethod
    def _gaussian_likelihood(innovation: np.ndarray, cov: np.ndarray) -> float:
        cov = 0.5 * (cov + cov.T)
        try:
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return 1e-12
            maha = float(innovation.T @ np.linalg.inv(cov) @ innovation)
            log_like = -0.5 * (maha + logdet + innovation.size * np.log(2.0 * np.pi))
            return max(1e-12, float(np.exp(np.clip(log_like, -80.0, 20.0))))
        except np.linalg.LinAlgError:
            return 1e-12

    def _mix_models(self, imm_state: dict) -> dict:
        means = np.asarray(imm_state["means"], dtype=np.float64)
        covs = np.asarray(imm_state["covariances"], dtype=np.float64)
        probs = self._normalize(imm_state["probs"])
        mixed_probs = self.transition.T @ probs
        mixed_probs = self._normalize(mixed_probs)
        mixed_means, mixed_covs = [], []
        for j in range(len(self.model_names)):
            denom = max(mixed_probs[j], 1e-12)
            weights = self.transition[:, j] * probs / denom
            mean_j = np.sum(weights[:, None] * means, axis=0)
            cov_j = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
            for i in range(len(self.model_names)):
                d = means[i] - mean_j
                cov_j += weights[i] * (covs[i] + np.outer(d, d))
            mixed_means.append(mean_j)
            mixed_covs.append(self._ensure_spd(cov_j))
        return {"means": np.asarray(mixed_means), "covariances": np.asarray(mixed_covs), "probs": mixed_probs}

    def _combine_estimate(self, imm_state: dict):
        means = np.asarray(imm_state["means"], dtype=np.float64)
        covs = np.asarray(imm_state["covariances"], dtype=np.float64)
        probs = self._normalize(imm_state["probs"])
        mean = np.sum(probs[:, None] * means, axis=0)
        cov = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        for i in range(len(self.model_names)):
            d = means[i] - mean
            cov += probs[i] * (covs[i] + np.outer(d, d))
        return mean, self._ensure_spd(cov)

    def predict(self, mean: np.ndarray, covariance):
        imm_state = self._mix_models(self._as_imm_state(mean, covariance))
        pred_means, pred_covs = [], []
        for model, model_mean, model_cov in zip(self.model_names, imm_state["means"], imm_state["covariances"]):
            pm, pc = self._ukf_predict(model_mean, model_cov, model)
            pred_means.append(pm)
            pred_covs.append(pc)
        pred_state = {"means": np.asarray(pred_means), "covariances": np.asarray(pred_covs), "probs": imm_state["probs"]}
        mixed_mean, mixed_cov = self._combine_estimate(pred_state)
        pred_state["mixed_covariance"] = mixed_cov
        return mixed_mean.astype(np.float32), pred_state

    def multi_predict(self, means: np.ndarray, covariances: list):
        out_means, out_covs = [], []
        for mean, cov in zip(means, covariances):
            pm, pc = self.predict(mean, cov)
            out_means.append(pm)
            out_covs.append(pc)
        return np.asarray(out_means), out_covs

    def update(self, mean: np.ndarray, covariance, measurement: np.ndarray):
        imm_state = self._as_imm_state(mean, covariance)
        upd_means, upd_covs, likelihoods = [], [], []
        for model_mean, model_cov in zip(imm_state["means"], imm_state["covariances"]):
            um, uc, like = self._ukf_update(model_mean, model_cov, measurement)
            upd_means.append(um)
            upd_covs.append(uc)
            likelihoods.append(like)
        probs = self._normalize(imm_state["probs"] * np.asarray(likelihoods, dtype=np.float64))
        upd_state = {"means": np.asarray(upd_means), "covariances": np.asarray(upd_covs), "probs": probs}
        mixed_mean, mixed_cov = self._combine_estimate(upd_state)
        upd_state["mixed_covariance"] = mixed_cov
        return mixed_mean.astype(np.float32), upd_state

    def project(self, mean: np.ndarray, covariance):
        imm_state = self._as_imm_state(mean, covariance)
        mixed_mean, mixed_cov = self._combine_estimate(imm_state)
        z_mean = self._hx(mixed_mean)
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 6] = 1.0
        H[3, 7] = 1.0
        S = H @ mixed_cov @ H.T + self._measurement_noise(mixed_mean)
        return z_mean.astype(np.float32), self._ensure_spd(S).astype(np.float32), H.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance,
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
