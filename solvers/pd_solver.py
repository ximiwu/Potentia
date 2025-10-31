from typing import Optional

import numpy as np
import taichi as ti

from energies.global_energy_container import GlobalEnergyContainer
from .base import ISolver

from data.base import ISimulationData


@ti.data_oriented
class PDSolver(ISolver):
    """
    Projective Dynamics 求解器：
    - 手动触发 LHS 重建与 LLT 分解并缓存
    - 每帧初始化 RHS 初值并进行 K 次全局线性求解
    - 固定点（inv_mass == 0 / mass == -1）跳过写回
    """

    def __init__(self, data: "ISimulationData", iterations: int = 3, ordering: str = "AMD") -> None:
        self.iterations: int = int(iterations)
        self.ordering: str = ordering

        self._container = GlobalEnergyContainer.get_instance()

        self._lhs: Optional[ti.linalg.SparseMatrix] = None
        self._solver: Optional[ti.linalg.SparseSolver] = None
        self._n: Optional[int] = None

        self._capacity: int = data.get_max_num_dofs()

        self._rhs_init = ti.Vector.field(3, dtype=ti.f32, shape=self._capacity)
        self._rhs = ti.Vector.field(3, dtype=ti.f32, shape=self._capacity)

    def build_lhs(self, data: "ISimulationData", dt: float) -> None:
        """
        构建并分解 LHS：
        - 由 GlobalEnergyContainer 装配 LHS（N×N 标量稀疏矩阵）
        - 使用 LLT（Cholesky）进行 analyze + factorize，并缓存求解器
        """
        n = int(data.get_num_dofs())
        lhs = self._container.compute_pd_lhs_mat(data, dt)

        solver = ti.linalg.SparseSolver(solver_type="LLT", ordering=self.ordering)
        solver.analyze_pattern(lhs)
        solver.factorize(lhs)

        self._lhs = lhs
        self._solver = solver
        self._n = n

        if self._capacity < n:
            raise RuntimeError(f"PDSolver: capacity {self._capacity} < required n {n} in build_lhs().")

    def initialize_rhs_init(self, data: "ISimulationData", dt: float) -> None:
        """
        初始化每帧不变的 RHS 初值
        """
        self._container.compute_pd_rhs_init_vec(data, self._rhs_init, dt)

    def solve(self, data: "ISimulationData", dt: float) -> None:
        if self._solver is None or self._n is None:
            raise RuntimeError("PDSolver: 尚未构建/分解 LHS。请先调用 rebuild_lhs(data, dt)。")

        n_now = int(data.get_num_dofs())
        if n_now != self._n:
            raise RuntimeError(f"PDSolver: DoF 数量变化（cached={self._n}, now={n_now}）。请手动重建 LHS。")

        # 每帧初始化 RHS 初值
        self.initialize_rhs_init(data, dt)

        assert self._rhs is not None and self._rhs_init is not None

        # 临时标量右手边与解向量（逐分量求解）
        bx = ti.ndarray(dtype=ti.f32, shape=(self._n,))
        by = ti.ndarray(dtype=ti.f32, shape=(self._n,))
        bz = ti.ndarray(dtype=ti.f32, shape=(self._n,))

        for _ in range(self.iterations):
            # 构建/累加本次迭代的 RHS
            self._container.compute_pd_rhs_vec(data, self._rhs, self._rhs_init)

            # 切分到 3 个标量向量
            self._split_vec3_to_scalars(self._rhs, self._n, bx, by, bz)

            # 逐分量求解
            solx = self._solver.solve(bx)  # type: ignore[union-attr]
            soly = self._solver.solve(by)  # type: ignore[union-attr]
            solz = self._solver.solve(bz)  # type: ignore[union-attr]

            # 写回到 q_predict，固定点（inv_mass == 0）跳过
            self._write_solution_to_qpredict(solx, soly, solz, data.get_predicted_dofs(), data.get_inv_masses(), self._n)


    @ti.kernel
    def _split_vec3_to_scalars(self,
                               src: ti.template(),
                               n: ti.i32,
                               bx: ti.types.ndarray(dtype=ti.f32, ndim=1),
                               by: ti.types.ndarray(dtype=ti.f32, ndim=1),
                               bz: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            bx[i] = src[i][0]
            by[i] = src[i][1]
            bz[i] = src[i][2]

    @ti.kernel
    def _write_solution_to_qpredict(self,
                                    x: ti.types.ndarray(dtype=ti.f32, ndim=1),
                                    y: ti.types.ndarray(dtype=ti.f32, ndim=1),
                                    z: ti.types.ndarray(dtype=ti.f32, ndim=1),
                                    q_predict: ti.template(),
                                    inv_masses: ti.template(),
                                    n: ti.i32):
        for i in range(n):
            if inv_masses[i] != 0.0:
                q_predict[i] = ti.Vector([x[i], y[i], z[i]])


