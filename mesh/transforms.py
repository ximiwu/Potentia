from typing import Tuple

import numpy as np
import taichi as ti


@ti.kernel
def _rotate_kernel(positions: ti.template(),
                   rotation_matrix: ti.template(),
                   out_positions: ti.template()):
    for i in range(positions.shape[0]):
        out_positions[i] = rotation_matrix @ positions[i]


@ti.kernel
def _scale_kernel(positions: ti.template(),
                  scaling_vector: ti.template(),
                  out_positions: ti.template()):
    for i in range(positions.shape[0]):
        out_positions[i] = positions[i] * scaling_vector


def rotate_positions(positions: ti.Field, rotation_angles: Tuple[float, float, float]) -> ti.Field:
    rx, ry, rz = rotation_angles

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]], dtype=np.float32)
    R_np = (Rz @ Ry @ Rx).astype(np.float32)

    R_ti = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
    R_ti.from_numpy(R_np)

    out = ti.Vector.field(3, dtype=ti.f32, shape=positions.shape[0])
    _rotate_kernel(positions, R_ti[None], out)
    return out


def scale_positions(positions: ti.Field, scaling_factors: Tuple[float, float, float]) -> ti.Field:
    scaling_vector = ti.Vector(list(scaling_factors), dt=ti.f32)
    out = ti.Vector.field(3, dtype=ti.f32, shape=positions.shape[0])
    _scale_kernel(positions, scaling_vector, out)
    return out


def rotate_numpy(vertices: np.ndarray, rotation_angles: Tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = rotation_angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]], dtype=np.float32)
    R_np = (Rz @ Ry @ Rx).astype(np.float32)
    return vertices.astype(np.float32, copy=False) @ R_np.T


def scale_numpy(vertices: np.ndarray, scaling_factors: Tuple[float, float, float]) -> np.ndarray:
    return vertices.astype(np.float32, copy=False) * np.array(scaling_factors, dtype=np.float32)


def apply_transform_numpy(vertices: np.ndarray,
                          translation: Tuple[float, float, float],
                          rotation: Tuple[float, float, float],
                          scale: Tuple[float, float, float]) -> np.ndarray:
    v = vertices.astype(np.float32, copy=False)
    t = np.asarray(translation, dtype=np.float32)
    r = np.asarray(rotation, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)

    if not np.allclose(s, (1.0, 1.0, 1.0)):
        v = scale_numpy(v, s)
    if not np.allclose(r, (0.0, 0.0, 0.0)):
        v = rotate_numpy(v, r)
    if not np.allclose(t, (0.0, 0.0, 0.0)):
        v = v + t
    return v


