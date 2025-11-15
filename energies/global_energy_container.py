from typing import Any, Dict

import numpy as np
import taichi as ti

from data.base import ISimulationData
from .base import IGlobalEnergyContainer, IPotentialEnergy


@ti.data_oriented
class GlobalEnergyContainer(IGlobalEnergyContainer):
    _instance = None

    # 固定容量上限：PD 中， A 的行、列上限（列与 v_indices 无关，支持能量端自定义列到全局顶点的映射）
    PD_A_ROW_MAX: int = 8
    PD_A_COL_MAX: int = 16

    @classmethod
    def get_instance(cls):
        """
        Returns the singleton instance of the GlobalEnergyContainer.
        The instance is created at the module level to be accessible by Taichi kernels.
        """
        if cls._instance is None:
            # This branch should ideally not be taken if the module is imported correctly,
            # as the instance is created at the end of the file.
            # It's here as a safeguard.
            cls._instance = GlobalEnergyContainer()
        return cls._instance

    def __init__(self,
                 max_constraints: int = 200000,
                 v_indices_size: int = 4,
                 params_size: int = 8):
        
        if GlobalEnergyContainer._instance is not None:
            raise RuntimeError("Error: Attempting to re-instantiate a singleton class.")
        
        self.max_constraints = max_constraints
        self.v_indices_size = v_indices_size
        self.params_size = params_size

        self.GenericConstraint = ti.types.struct(
            constraint_type=ti.i32,
            v_indices=ti.types.vector(self.v_indices_size, ti.i32),
            params=ti.types.vector(self.params_size, ti.f32),
            lambda_=ti.f32,
        )

        self.constraints = self.GenericConstraint.field()
        self.num_active_constraints = ti.field(dtype=ti.i32, shape=())
        self.num_static_constraints = ti.field(dtype=ti.i32, shape=())
        
        # self.root = ti.root.dynamic(ti.i, self.max_constraints, chunk_size=1024)
        self.root = ti.root.dense(ti.i, self.max_constraints)
        self.root.place(self.constraints)

        self.registered_energies: Dict[int, IPotentialEnergy] = {}

        self._energy_acc = ti.field(dtype=ti.f32, shape=())
        self._loss_acc = ti.field(dtype=ti.f32, shape=())
        
        # Set the class instance variable
        GlobalEnergyContainer._instance = self

    def register_energy(self, energy: IPotentialEnergy):
        type_id = energy.get_type_id()
        if type_id in self.registered_energies:
            print(f"Warning: Overwriting registered energy for type_id {type_id}")
        self.registered_energies[type_id] = energy

    def reserve_constraints(self, num_to_add: int, is_static: bool = True) -> int:
        is_static_int = 1 if is_static else 0
        return self._reserve_constraints_kernel(num_to_add, is_static_int)

    @ti.kernel
    def _reserve_constraints_kernel(self, num_to_add: int, is_static: ti.i32) -> int:
        start_idx = ti.atomic_add(self.num_active_constraints[None], num_to_add)
        if is_static == 1:
            ti.atomic_add(self.num_static_constraints[None], num_to_add)
        return start_idx

    def clear_constraints(self):
        self.num_active_constraints[None] = 0
        self.num_static_constraints[None] = 0

    def clear_dynamic_constraints(self):
        self.num_active_constraints[None] = self.num_static_constraints[None]

    @ti.func
    def add_one_constraint(self,
                           constraint_idx: int,
                           constraint_type: int,
                           v_indices_vec: ti.template(),
                           params_vec: ti.template()):
        if constraint_idx < self.max_constraints:
            self.constraints[constraint_idx].constraint_type = constraint_type
            
            for k in ti.static(range(self.v_indices_size)):
                if k < v_indices_vec.n:
                    self.constraints[constraint_idx].v_indices[k] = v_indices_vec[k]
                else:
                    self.constraints[constraint_idx].v_indices[k] = -1

            for k in ti.static(range(self.params_size)):
                if k < params_vec.n:
                    self.constraints[constraint_idx].params[k] = params_vec[k]
                else:
                    self.constraints[constraint_idx].params[k] = 0.0
            self.constraints[constraint_idx].lambda_ = 0.0

    def get_num_constraints(self) -> int:
        return self.num_active_constraints[None]

    def compute_gradient(self, data: ISimulationData, out_grad: ti.template()):
        out_grad.fill(0)
        q = data.get_predicted_dofs()
        self._compute_gradient_kernel(q, out_grad)

    @ti.kernel
    def _compute_gradient_kernel(self, q: ti.template(), out_grad: ti.template()):
        for i in range(self.num_active_constraints[None]):
            constraint = self.constraints[i]
            
            for type_id in ti.static(list(self.registered_energies.keys())):
                if constraint.constraint_type == type_id:
                    self.registered_energies[type_id].compute_gradient_func(constraint, q, out_grad)

    def compute_energy(self, data: ISimulationData) -> ti.f32:
        self._energy_acc[None] = 0.0
        q = data.get_predicted_dofs()
        self._compute_energy_kernel(q, self._energy_acc)
        return self._energy_acc[None]

    def compute_loss(self, data: ISimulationData, x: ti.template(), y:ti.template(), dt: float) -> ti.f32:
        """
        g(x) = 1/2 (x - y)^T M (x - y) + dt^2 E(x)
        - x: predicted DoFs
        - y: initial predicted DoFs
        - M: diagonal from masses
        - E(x): total energy computed via compute_energy
        """
        self._loss_acc[None] = 0.0
        masses = data.get_masses()
        n = data.get_num_dofs()

        # accumulate 1/2 (x - y)^T M (x - y)
        self._accumulate_mass_quadratic_term(x, y, masses, n, self._loss_acc)

        # add dt^2 * E(x)
        self._energy_acc[None] = 0.0
        self._compute_energy_kernel(x, self._energy_acc)
        self._loss_acc[None] = self._loss_acc[None] + (dt * dt) * self._energy_acc[None]
        return self._loss_acc[None]

    @ti.kernel
    def _compute_energy_kernel(self, q: ti.template(), total_energy: ti.template()):
        # The loop is parallelized by Taichi's default scheduler.
        for i in range(self.num_active_constraints[None]):
            constraint = self.constraints[i]
            energy = 0.0
            for type_id in ti.static(list(self.registered_energies.keys())):
                if constraint.constraint_type == type_id:
                    energy = self.registered_energies[type_id].compute_energy_func(constraint, q)
            ti.atomic_add(total_energy[None], energy)

    @ti.kernel
    def _accumulate_mass_quadratic_term(self,
                                        x: ti.template(),
                                        y: ti.template(),
                                        masses: ti.template(),
                                        n: ti.i32,
                                        out_acc: ti.template()):
        for i in range(n):
            m = masses[i]
            if m >= 1e-6:
                d = x[i] - y[i]
                ti.atomic_add(out_acc[None], 0.5 * m * d.dot(d))

    def compute_hessian(self, data: ISimulationData, out_hessian_builder: Any):
        q = data.get_predicted_dofs()
        self._compute_hessian_kernel(q, out_hessian_builder)

    @ti.kernel
    def _compute_hessian_kernel(self,
                                q: ti.template(),
                                out_builder: ti.types.sparse_matrix_builder()):
        for idx in range(self.num_active_constraints[None]):
            c = self.constraints[idx]
            for type_id in ti.static(list(self.registered_energies.keys())):
                if c.constraint_type == type_id:
                    self.registered_energies[type_id].assemble_hessian_to_builder_func(c, q, out_builder)

    def compute_hessian_abs_eig(self, data: ISimulationData, out_hessian_builder: Any):
        """
        装配 “绝对值特征值投影” 后的 Hessian。
        """
        q = data.get_predicted_dofs()
        self._compute_hessian_abs_eig_kernel(q, out_hessian_builder)

    @ti.kernel
    def _compute_hessian_abs_eig_kernel(self,
                                        q: ti.template(),
                                        out_builder: ti.types.sparse_matrix_builder()):
        for idx in range(self.num_active_constraints[None]):
            c = self.constraints[idx]
            for type_id in ti.static(list(self.registered_energies.keys())):
                if c.constraint_type == type_id:
                    self.registered_energies[type_id].assemble_hessian_abs_eig_to_builder_func(c, q, out_builder)

    def compute_pd_rhs_init_vec(self, data: ISimulationData, out_vec: ti.template(), dt: float):
        """
        计算rhs vec每次迭代都相同的部分
        """
        q_predict = data.get_predicted_dofs()
        masses = data.get_masses()
        n = data.get_num_dofs()
        self._compute_pd_rhs_init_vec_kernel(q_predict, masses, n, dt, out_vec)

    @ti.kernel
    def _compute_pd_rhs_init_vec_kernel(self,
                                        q_predict: ti.template(),
                                        masses: ti.template(),
                                        n: ti.i32,
                                        dt: ti.f32,
                                        out_vec: ti.template()):
        inv_dt2 = 1.0 / (dt * dt)
        for i in range(n):
            m = masses[i]
            if m > 0.0:
                out_vec[i] = (m * inv_dt2) * q_predict[i]
            else:
                out_vec[i] = ti.Vector([0.0, 0.0, 0.0])

    

    
    def compute_pd_rhs_vec(self, data: ISimulationData, out_vec: ti.template(), init_vec: ti.template()):
        """
        启动 kernel 计算各能量的 PD 局部项并装配到全局右手边向量。

        约定 out_vec 为一个 ti.Vector.field(3, ...)，
        能量内部通过 atomic_add 对其进行原地累加。
        """
        # 先将 out_vec 写成 init_vec
        qn = data.get_num_dofs()
        self._copy_vec_kernel(init_vec, qn, out_vec)

        q_predict = data.get_predicted_dofs()
        # 从 ISimulationData 获取 CSR 顶点邻接与与其对齐的 cotan 权重
        offsets, info = data.get_vertex_adjacency()

        self._compute_pd_rhs_vec_kernel(
            q_predict,
            offsets,
            info.vertex_adj_indices,
            info.vertex_adj_cotan_weights,
            out_vec
        )

    @ti.kernel
    def _copy_vec_kernel(self, src: ti.template(), n: ti.i32, dst: ti.template()):
        for i in range(n):
            dst[i] = src[i]

    @ti.kernel
    def _compute_pd_rhs_vec_kernel(self,
                                   q_predict: ti.template(),
                                   vertex_adj_offsets: ti.template(),
                                   vertex_adj_indices: ti.template(),
                                   vertex_adj_cotan_weights: ti.template(),
                                   out_vec: ti.template()):
        for i in range(self.num_active_constraints[None]):
            c = self.constraints[i]
            for type_id in ti.static(list(self.registered_energies.keys())):
                if c.constraint_type == type_id:
                    self.registered_energies[type_id].compute_pd_rhs_vec_func(
                        c,
                        q_predict,
                        vertex_adj_offsets,
                        vertex_adj_indices,
                        vertex_adj_cotan_weights,
                        out_vec
                    )

    @ti.kernel
    def _fill_mass_lhs(self,
                        masses: ti.template(),
                        n: ti.i32,
                        inv_dt2: ti.f32,
                        out_builder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            m = masses[i]
            if m > 0.0:
                out_builder[i, i] += m * inv_dt2

    def compute_pd_lhs_mat(self, 
                           data: ISimulationData, 
                           dt: float) -> ti.linalg.SparseMatrix:
        """
        装配并返回 PD 的 LHS（N×N 标量稀疏矩阵）。
        - 单一 SparseMatrixBuilder：写入质量对角 M/dt^2，并在一个 kernel 中并行装配所有约束的 S^T A^T A S 贡献
        """

        n = data.get_num_dofs()
        masses = data.get_masses()

        # 估计 triplets：质量对角 n + 每个约束至多 PD_A_COL_MAX^2 个条目
        num_constraints = int(self.num_active_constraints[None])
        max_tris = n + num_constraints * (GlobalEnergyContainer.PD_A_COL_MAX * GlobalEnergyContainer.PD_A_COL_MAX)

        builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=max_tris)

        # 写入质量对角（M/dt^2）
        inv_dt2 = 1.0 / (dt * dt)
        self._fill_mass_lhs(masses, n, inv_dt2, builder)

        # 从 ISimulationData 获取 CSR 顶点邻接与与其对齐的 cotan 权重
        offsets, info = data.get_vertex_adjacency()
        self._accumulate_pd_lhs_kernel(
            offsets,
            info.vertex_adj_indices,
            info.vertex_adj_cotan_weights,
            builder,
        )

        return builder.build()

    @ti.kernel
    def _accumulate_pd_lhs_kernel(self,
                                  vertex_adj_offsets: ti.template(),
                                  vertex_adj_indices: ti.template(),
                                  vertex_adj_cotan_weights: ti.template(),
                                  out_builder: ti.types.sparse_matrix_builder()):
        for idx in range(self.num_active_constraints[None]):
            c = self.constraints[idx]

            for type_id in ti.static(list(self.registered_energies.keys())):
                if c.constraint_type == type_id:
                    # 局部缓冲：A_buf[PD_A_ROW_MAX, PD_A_COL_MAX]，cols_buf[PD_A_COL_MAX]
                    A_buf = ti.Matrix.zero(ti.f32, GlobalEnergyContainer.PD_A_ROW_MAX, GlobalEnergyContainer.PD_A_COL_MAX)
                    cols_buf = ti.Vector.zero(ti.i32, GlobalEnergyContainer.PD_A_COL_MAX)

                    used_cols, stiffness = self.registered_energies[type_id].fill_pd_A_and_cols(
                        c,
                        vertex_adj_offsets,
                        vertex_adj_indices,
                        vertex_adj_cotan_weights,
                        A_buf,
                        cols_buf,
                    )

                    # K_local = A^T A
                    for a in range(used_cols):
                        ia = cols_buf[a]
                        for b in range(used_cols):
                            ib = cols_buf[b]
                            val = 0.0
                            for r in range(GlobalEnergyContainer.PD_A_ROW_MAX):
                                val += A_buf[r, a] * A_buf[r, b]
                            out_builder[ia, ib] += val * stiffness



    @ti.func
    def compute_one_constraint_gradient_func(self, i: int, q: ti.template()):
        constraint = self.constraints[i]
        
        C = 0.0
        grads = ti.Matrix.zero(ti.f32, self.v_indices_size, 3)
        num_vertices = 0

        for type_id in ti.static(list(self.registered_energies.keys())):
            if constraint.constraint_type == type_id:
                C, num_vertices = self.registered_energies[type_id].compute_constraint_gradient_func(constraint, q, grads)
        return C, grads, num_vertices

    def compute_loss_gradient_fro_norm(self, data: ISimulationData, x: ti.template(), y: ti.template(), dt: float) -> ti.f32:
        capacity = int(data.get_max_num_dofs())
        if not hasattr(self, "_tmp_grad"):
            self._tmp_grad = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        elif int(self._tmp_grad.shape[0]) != capacity:
            self._tmp_grad = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._tmp_grad.fill(0)
        self._compute_gradient_kernel(x, self._tmp_grad)
        n = int(data.get_num_dofs())
        return self._grad_loss_l2_norm_kernel(self._tmp_grad, data.get_masses(), x, y, n, float(dt))

    def compute_loss_hessian_fro_norm(self, data: ISimulationData, dt: float) -> ti.f32:
        n = int(data.get_num_dofs())
        dim = 3 * n
        num_constraints = int(self.num_active_constraints[None])
        max_blocks_per_c = self.v_indices_size * self.v_indices_size
        max_triplets = max(1, num_constraints * 9 * max_blocks_per_c)
        he_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=max_triplets)
        self.compute_hessian(data, he_builder)
        He = he_builder.build()
        mass_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=dim)
        self._fill_mass_diag_for_hessian_kernel(data.get_masses(), n, mass_builder)
        M = mass_builder.build()
        H = M + (He * (float(dt) * float(dt)))

        e = ti.ndarray(dtype=ti.f32, shape=(dim,))
        y = ti.ndarray(dtype=ti.f32, shape=(dim,))
        acc = 0.0
        for i in range(dim):
            self._set_one_hot_nd(e, i)
            y = H @ e
            acc += float(self._l2_norm_nd(y)) ** 2
        return float(np.sqrt(acc))

    @ti.kernel
    def _grad_loss_l2_norm_kernel(self,
                                  grad_e: ti.template(),
                                  masses: ti.template(),
                                  x: ti.template(),
                                  y: ti.template(),
                                  n: ti.i32,
                                  dt: ti.f32) -> ti.f32:
        acc = 0.0
        for i in range(n):
            g0 = masses[i] * (x[i] - y[i])
            g = g0 + grad_e[i] * (dt * dt)
            acc += g.dot(g)
        return ti.sqrt(acc)

    @ti.kernel
    def _fill_mass_diag_for_hessian_kernel(self,
                                           masses: ti.template(),
                                           n: ti.i32,
                                           builder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            m = masses[i]
            base = 3 * i
            for c in ti.static(range(3)):
                builder[base + c, base + c] += m

    @ti.kernel
    def _set_one_hot_nd(self, e: ti.types.ndarray(dtype=ti.f32, ndim=1), idx: ti.i32):
        for i in range(e.shape[0]):
            e[i] = 1.0 if i == idx else 0.0

    @ti.kernel
    def _l2_norm_nd(self, x: ti.types.ndarray(dtype=ti.f32, ndim=1)) -> ti.f32:
        acc = 0.0
        for i in range(x.shape[0]):
            v = x[i]
            acc += v * v
        return ti.sqrt(acc)
