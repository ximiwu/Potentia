import taichi as ti
from typing import Tuple
import os
import numpy as np

from .base import ISimulationData


@ti.data_oriented
class MassPointData(ISimulationData):
    """
    Simulation data container for a collection of 3D mass points.
    DoFs are the 3D positions of the points.
    """
    def __init__(self, max_point_num: int, max_degree: int = 16):
        if max_point_num <= 0:
            raise ValueError("max_point_num must be positive.")
        if max_degree <= 0:
            raise ValueError("max_degree must be positive.")

        self._max_point_num = max_point_num
        self._current_num_dofs = 0
        self._is_primary_buffer_a = True # True: positions is primary, False: predicted_positions is primary
        self._max_degree = int(max_degree)

        # Use a Struct.field to define an AoS layout
        self.particle_data = ti.Struct.field({
            "positions": ti.math.vec3,
            "predicted_positions": ti.math.vec3,
            "record_positions": ti.math.vec3,
            "velocities": ti.math.vec3,
            "masses": ti.f32,
            "inv_masses": ti.f32
        }, shape=(self._max_point_num,))

        # Global vertex adjacency in CSR form (directed i→j), aligned with TriMesh semantics
        self.vertex_adj_offsets = ti.field(dtype=ti.i32, shape=self._max_point_num + 1)
        self.vertex_adj_info = ti.Struct.field({
            "vertex_adj_indices": ti.i32,
            "vertex_adj_cotan_weights": ti.f32,
        }, shape=self._max_point_num * self._max_degree)

        # Global write pointer for CSR adjacency (number of used entries in vertex_adj_info)
        self.vertex_adj_next = ti.field(dtype=ti.i32, shape=())
        self.vertex_adj_next[None] = 0

    def get_dofs(self) -> ti.Field:
        if self._is_primary_buffer_a:
            return self.particle_data.positions
        else:
            return self.particle_data.predicted_positions

    def get_velocities(self) -> ti.Field:
        return self.particle_data.velocities

    def get_inv_masses(self) -> ti.Field:
        return self.particle_data.inv_masses

    def get_masses(self) -> ti.Field:
        return self.particle_data.masses

    def get_predicted_dofs(self) -> ti.Field:
        if self._is_primary_buffer_a:
            return self.particle_data.predicted_positions
        else:
            return self.particle_data.positions

    def get_record_dofs(self) -> ti.Field:
        return self.particle_data.record_positions

    def record_predicted_dofs(self):
        self._record_dofs_kernel(self.get_predicted_dofs(), self.get_record_dofs(), self.get_max_num_dofs())

    @ti.kernel
    def _record_dofs_kernel(self, source: ti.template(), dest: ti.template(), dim: int):
        for i in range(dim):
            dest[i] = source[i]



    def get_num_dofs(self) -> int:
        return self._current_num_dofs

    def get_max_num_dofs(self) -> int:
        return self._max_point_num

    def allocate_dofs(self, num_dofs: int) -> int:
        if self._current_num_dofs + num_dofs > self._max_point_num:
            raise RuntimeError(
                f"Cannot allocate {num_dofs} DoFs. "
                f"Available: {self._max_point_num - self._current_num_dofs}, "
                f"Total capacity: {self._max_point_num}."
            )
        
        offset = self._current_num_dofs
        self._current_num_dofs += num_dofs
        return offset

    def swap_buffers(self) -> None:
        self._is_primary_buffer_a = not self._is_primary_buffer_a

    def get_vertex_adjacency(self) -> Tuple[ti.Field, ti.Field]:
        """
        Returns global CSR vertex adjacency fields with directed semantics (i→j):
        - offsets: length = num_vertices + 1
        - info: Struct field with (vertex_adj_indices: i32, vertex_adj_cotan_weights: f32)
        """
        return self.vertex_adj_offsets, self.vertex_adj_info

    def get_vertex_adjacency_write_ptr(self) -> ti.Field:
        """Returns the scalar field that stores the global write pointer for CSR info array."""
        return self.vertex_adj_next

    def get_max_degree(self) -> int:
        return self._max_degree

    def reset_vertex_adjacency(self) -> None:
        """Resets the global CSR write pointer to zero (offsets are not cleared)."""
        self.vertex_adj_next[None] = 0

    def resume_from_record(self, base_dir: str, frame_idx: int) -> None:
        """
        从记录的会话目录恢复到指定帧：覆盖当前 primary DoF 与 velocities 的前 N 项（N=get_num_dofs()）。
        目录结构要求：
          - {base_dir}/dofs/dofs_{frame_idx:06d}.npy
          - {base_dir}/velocities/velocities_{frame_idx:06d}.npy
        两者形状均为 (N, 3)，N 必须等于 get_num_dofs()。
        """
        if not isinstance(base_dir, str):
            raise TypeError("base_dir 必须为 str")
        if not isinstance(frame_idx, int):
            raise TypeError("frame_idx 必须为 int")

        n = int(self.get_num_dofs())
        if n <= 0:
            raise RuntimeError("当前数据容器尚未分配 DoFs（get_num_dofs()==0），无法恢复。")

        dofs_path = os.path.join(base_dir, "dofs", f"dofs_{frame_idx:06d}.npy")
        vels_path = os.path.join(base_dir, "velocities", f"velocities_{frame_idx:06d}.npy")

        if not os.path.isfile(dofs_path):
            raise FileNotFoundError(f"未找到 DoF 文件: {dofs_path}")
        if not os.path.isfile(vels_path):
            raise FileNotFoundError(f"未找到 Velocity 文件: {vels_path}")

        dofs_np = np.load(dofs_path)
        vels_np = np.load(vels_path)

        if not (hasattr(dofs_np, "ndim") and dofs_np.ndim == 2 and dofs_np.shape[1] == 3):
            raise ValueError(f"dofs 文件形状非法，期望 (N,3)，实际 {getattr(dofs_np, 'shape', None)}")
        if not (hasattr(vels_np, "ndim") and vels_np.ndim == 2 and vels_np.shape[1] == 3):
            raise ValueError(f"velocities 文件形状非法，期望 (N,3)，实际 {getattr(vels_np, 'shape', None)}")

        if dofs_np.shape[0] != n or vels_np.shape[0] != n:
            raise ValueError(f"记录帧的 N 与当前 get_num_dofs() 不一致: 文件N={dofs_np.shape[0]},{vels_np.shape[0]} 当前N={n}")

        if dofs_np.dtype != np.float32:
            dofs_np = dofs_np.astype(np.float32, copy=False)
        if vels_np.dtype != np.float32:
            vels_np = vels_np.astype(np.float32, copy=False)

        self._resume_copy_kernel(self.get_dofs(), self.get_velocities(), dofs_np, vels_np, n)

        try:
            ti.sync()
        except Exception:
            pass

    @ti.kernel
    def _resume_copy_kernel(self,
                            dofs: ti.template(),
                            vels: ti.template(),
                            dofs_np: ti.types.ndarray(dtype=ti.f32, ndim=2),
                            vels_np: ti.types.ndarray(dtype=ti.f32, ndim=2),
                            n: ti.i32):
        for i in range(n):
            dofs[i] = ti.math.vec3(dofs_np[i, 0], dofs_np[i, 1], dofs_np[i, 2])
            vels[i] = ti.math.vec3(vels_np[i, 0], vels_np[i, 1], vels_np[i, 2])


    def set_predict_dof(self, npy_path: str) -> None:
        """
        从 .npy 文件加载 (N, 3) 的 float32 数组，并覆盖当前 predicted DoFs 的前 N 项，
        其中 N = get_num_dofs()。
        """
        if not isinstance(npy_path, str):
            raise TypeError("npy_path 必须为 str")

        n = int(self.get_num_dofs())
        if n <= 0:
            raise RuntimeError("当前数据容器尚未分配 DoFs（get_num_dofs()==0），无法设置 predicted DoFs。")

        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"未找到 .npy 文件: {npy_path}")

        arr = np.load(npy_path)
        if not (hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[1] == 3):
            raise ValueError(f".npy 文件形状非法，期望 (N,3)，实际 {getattr(arr, 'shape', None)}")
        if arr.shape[0] != n:
            raise ValueError(f".npy 的 N 与当前 get_num_dofs() 不一致: 文件N={arr.shape[0]} 当前N={n}")

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        self._copy_predicted_from_np(self.get_predicted_dofs(), arr, n)

        try:
            ti.sync()
        except Exception:
            pass

    @ti.kernel
    def _copy_predicted_from_np(self,
                                predicted: ti.template(),
                                arr: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                n: ti.i32):
        for i in range(n):
            predicted[i] = ti.math.vec3(arr[i, 0], arr[i, 1], arr[i, 2])

