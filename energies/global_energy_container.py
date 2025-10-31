from typing import Any, Dict

import numpy as np
import taichi as ti

from data.base import ISimulationData
from .base import IGlobalEnergyContainer, IPotentialEnergy


@ti.data_oriented
class GlobalEnergyContainer(IGlobalEnergyContainer):
    _instance = None

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
        
        self.root = ti.root.dynamic(ti.i, self.max_constraints, chunk_size=1024)
        # self.root = ti.root.dense(ti.i, 100000)
        self.root.place(self.constraints)

        self.registered_energies: Dict[int, IPotentialEnergy] = {}
        
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
        total_energy = ti.field(dtype=ti.f32, shape=())
        total_energy[None] = 0.0
        q = data.get_predicted_dofs()
        self._compute_energy_kernel(q, total_energy)
        return total_energy[None]

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

    def compute_hessian(self, data: ISimulationData, out_hessian_builder: Any):
        # This is a placeholder for the hessian computation.
        # A concrete kernel would be needed here.
        print("Warning: GlobalEnergyContainer.compute_hessian is not implemented.")
        pass

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
        - 写入质量对角 M/dt^2（Python 侧构造）
        - 遍历所有已注册能量，若其实现了 compute_pd_lhs_mat_func(data)，则将其返回稀疏矩阵相加
        """

        n = data.get_num_dofs()
        masses = data.get_masses()

        # 1) 质量对角（M/dt^2） —— 在 kernel 内填充 builder
        inv_dt2 = 1.0 / (dt * dt)
        mass_builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n)
        self._fill_mass_lhs(masses, n, inv_dt2, mass_builder)
        lhs = mass_builder.build()

        # 2) 遍历每个约束，按类型分发到对应能量，累加“单约束”贡献矩阵
        num_constraints = self.num_active_constraints[None]
        for i in range(num_constraints):
            c = self.constraints[i]
            type_id = int(c.constraint_type)
            energy = self.registered_energies.get(type_id, None)
            if energy is None:
                continue
            try:
                contrib = energy.compute_pd_lhs_mat_func(c, data)
            except NotImplementedError:
                contrib = None
            except Exception as e:
                print(f"Warning: {energy.__class__.__name__}.compute_pd_lhs_mat_func 约束 {i} 失败：{e}")
                contrib = None

            if contrib is not None:
                lhs = lhs + contrib

        return lhs



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
