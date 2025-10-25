from typing import Any, Dict

import numpy as np
import taichi as ti

from data.base import ISimulationData
from energies.base import IGlobalEnergyContainer, IPotentialEnergy


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
        
        # self.root = ti.root.dynamic(ti.i, self.max_constraints, chunk_size=1024)
        self.root = ti.root.dense(ti.i, 100000)
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

    def gather_projection(self, data: ISimulationData, **kwargs):
        # This is a placeholder for the gather projection step.
        # A concrete kernel would be needed here, likely with a more specific signature.
        print("Warning: GlobalEnergyContainer.gather_projection is not implemented.")
        pass

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
