import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.trackers.modules.ekf_ctrv import EKFCTRVFilter


def finite_difference_jacobian(filter_: EKFCTRVFilter, state: np.ndarray, eps: float) -> np.ndarray:
    f0 = filter_._predict_state(state)
    jac = np.zeros((len(f0), len(state)), dtype=np.float64)
    for i in range(len(state)):
        step = np.zeros_like(state, dtype=np.float64)
        step[i] = eps
        fp = filter_._predict_state((state + step).astype(np.float32)).astype(np.float64)
        fm = filter_._predict_state((state - step).astype(np.float32)).astype(np.float64)
        jac[:, i] = (fp - fm) / (2.0 * eps)
    return jac


def parse_args():
    p = argparse.ArgumentParser(description="Check EKF-CTRV analytic transition Jacobian against finite differences.")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--omega", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=5e-2)
    return p.parse_args()


def main():
    args = parse_args()
    filter_ = EKFCTRVFilter(dt=args.dt)
    state = np.array([120.0, 80.0, 6.5, 0.7, args.omega, 42.0, 18.0, 0.2, -0.1], dtype=np.float32)
    analytic = filter_._jacobian(state).astype(np.float64)
    numeric = finite_difference_jacobian(filter_, state.astype(np.float64), args.eps)
    diff = np.abs(analytic - numeric)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    print(f"max_abs_error: {max_err:.6f}")
    print(f"mean_abs_error: {mean_err:.6f}")
    if max_err > args.tol:
        raise SystemExit(f"Jacobian check failed: max_abs_error {max_err:.6f} > tol {args.tol:.6f}")


if __name__ == "__main__":
    main()
