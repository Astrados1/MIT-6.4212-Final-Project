"""Pure-numpy helpers for probe perturbations. Quaternions are (w, x, y, z), as in Isaac Lab."""

from __future__ import annotations

import numpy as np


def quat_multiply(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)])


def displaced_pose(pos, quat, dx: float, dy: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift in the world xy-plane and rotate about the world z-axis through the object origin."""
    new_pos = np.asarray(pos, dtype=float) + np.array([dx, dy, 0.0])
    new_quat = quat_multiply(yaw_quat(np.deg2rad(yaw_deg)), np.asarray(quat, dtype=float))
    return new_pos, new_quat / np.linalg.norm(new_quat)


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Rodrigues' formula."""
    rotvec = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def perturb_camera_pose(world_from_cam, trans_m: float, rot_deg: float, rng: np.random.Generator):
    """Apply a calibration-style error expressed in the camera frame.

    The translation and rotation axes are random unit vectors drawn from `rng` (so the
    error direction is reproducible from the seed). Returns (perturbed 4x4, details dict).
    """
    t_dir = random_unit_vector(rng)
    r_axis = random_unit_vector(rng)
    delta = np.eye(4)
    delta[:3, :3] = rotvec_to_matrix(r_axis * np.deg2rad(rot_deg))
    delta[:3, 3] = t_dir * trans_m
    out = np.asarray(world_from_cam, dtype=np.float64) @ delta
    details = {
        "trans_m": trans_m,
        "trans_dir_cam": t_dir.tolist(),
        "rot_deg": rot_deg,
        "rot_axis_cam": r_axis.tolist(),
    }
    return out.astype(np.float32), details


def corrupt_depth(depth, noise_std_m: float, dropout_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise on valid pixels, then zero out a random fraction (TiPToP treats 0 as invalid)."""
    d = np.array(depth, dtype=np.float32, copy=True)
    valid = np.isfinite(d) & (d > 0)
    if noise_std_m > 0:
        d[valid] += rng.normal(scale=noise_std_m, size=int(valid.sum())).astype(np.float32)
        d[valid & (d < 0)] = 0.0
    if dropout_frac > 0:
        d[rng.random(d.shape) < dropout_frac] = 0.0
    return d
