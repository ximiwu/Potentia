import taichi as ti
from typing import List, Optional

from data.base import ISimulationData
from objects.base import IMeshObject
from .base import IForce


@ti.data_oriented
class ConstantForce(IForce):
    """
    对给定对象的指定局部顶点集合，施加统一的常量力（单位：N）。
    - 仅对可动点生效（inv_mass > 1e-6）。
    - 通过对象的 data_offset 将局部索引映射为全局索引后缓存。
    """

    def __init__(
        self,
        obj: IMeshObject,
        force: ti.math.vec3,
        local_indices: Optional[List[int]] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> None:
        required_capabilities = ["get_data_offset", "get_num_dofs"]
        for name in required_capabilities:
            if not hasattr(obj, name):
                raise TypeError(f"Argument 'obj' must implement capability: {name}")

        self.force = force
        offset = obj.get_data_offset()
        num_dofs = obj.get_num_dofs()

        # 选择来源：优先使用非空列表；否则使用闭区间 [start_idx, end_idx]
        has_list = local_indices is not None and len(local_indices) > 0
        has_range = (start_idx is not None) or (end_idx is not None)

        if has_list and has_range:
            raise ValueError("Provide either local_indices or (start_idx, end_idx), not both.")

        if has_list:
            self._num_targets = len(local_indices) if local_indices is not None else 0
            if self._num_targets == 0:
                self.target_indices = ti.field(dtype=ti.i32, shape=1)
                self.target_indices[0] = -1
                return
            self.target_indices = ti.field(dtype=ti.i32, shape=self._num_targets)
            for k, li in enumerate(local_indices if local_indices is not None else []):
                if li < 0 or li >= num_dofs:
                    raise IndexError(f"local index {li} out of range [0, {num_dofs - 1}]")
                self.target_indices[k] = offset + li
        else:
            if start_idx is None or end_idx is None:
                raise ValueError("When local_indices is not provided, start_idx and end_idx must be given.")
            if start_idx < 0 or end_idx < 0:
                raise IndexError("start_idx and end_idx must be non-negative.")
            if start_idx > end_idx:
                raise ValueError("start_idx must be <= end_idx.")
            if end_idx >= num_dofs:
                raise IndexError(f"end_idx {end_idx} out of range [0, {num_dofs - 1}]")

            self._num_targets = end_idx - start_idx + 1
            if self._num_targets <= 0:
                self.target_indices = ti.field(dtype=ti.i32, shape=1)
                self.target_indices[0] = -1
                return
            self.target_indices = ti.field(dtype=ti.i32, shape=self._num_targets)
            for k in range(self._num_targets):
                li = start_idx + k
                self.target_indices[k] = offset + li

    def add_force_to_vector(self, data: ISimulationData, force_vector: ti.Field) -> None:
        """
        仅对缓存的目标全局索引累加常量力。

        Args:
            data (ISimulationData): 全局模拟数据容器（仅使用 inv_masses）。
            force_vector (ti.Field): 全局力向量（Vector.field(3, f32)）。
        """
        if self._num_targets == 0:
            return
        self._apply_force_kernel(data.get_inv_masses(), force_vector)

    @ti.kernel
    def _apply_force_kernel(self, inv_masses: ti.template(), force_vector: ti.template()):
        for k in range(self.target_indices.shape[0]):
            i = self.target_indices[k]
            if i >= 0 and inv_masses[i] > 1e-6:
                force_vector[i] += self.force


