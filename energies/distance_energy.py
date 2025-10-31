import taichi as ti
import numpy as np
from typing import TYPE_CHECKING, Tuple

from .base import PotentialEnergy
from .global_energy_container import GlobalEnergyContainer


@ti.data_oriented
class DistanceEnergy(PotentialEnergy):
    """
    Defines the computation for distance constraints.
    This class is stateless and operates on data stored in the global container.
    """
    TYPE_ID = 0

    def __init__(self):
        super().__init__()
        # Cache the global container instance for use inside Taichi funcs via ti.static
        self._container = GlobalEnergyContainer.get_instance()

    @ti.func
    def add_one_constraint_func(self,
                                container: ti.template(),
                                constraint_idx: int,
                                p1_idx: int,
                                p2_idx: int,
                                rest_dist: ti.f32,
                                stiffness: ti.f32):
        v_indices = ti.Vector([p1_idx, p2_idx])
        params = ti.Vector([rest_dist, stiffness])
        container.add_one_constraint(
            constraint_idx,
            self.TYPE_ID,
            v_indices,
            params
        )

    @ti.func
    def compute_constraint_gradient_func(self, constraint: ti.template(), q: ti.template(), grads: ti.template()):
        idx1, idx2 = constraint.v_indices[0], constraint.v_indices[1]
        p1 = q[idx1]
        p2 = q[idx2]
        
        rest_dist = constraint.params[0]

        current_distance_vec = p1 - p2
        current_distance_norm = current_distance_vec.norm()

        C = current_distance_norm - rest_dist

        if current_distance_norm > 1e-9:
            segment_direction = current_distance_vec / current_distance_norm
            grad1 = segment_direction
            grad2 = -segment_direction
            grads[0, :] = grad1
            grads[1, :] = grad2
            
        return C, 2 # Return num_vertices = 2

    @ti.func
    def compute_energy_func(self, constraint: ti.template(), q: ti.template()) -> ti.f32:
        idx1, idx2 = constraint.v_indices[0], constraint.v_indices[1]
        p1 = q[idx1]
        p2 = q[idx2]
        
        rest_dist = constraint.params[0]
        stiffness = constraint.params[1]
        
        C = (p1 - p2).norm() - rest_dist
        energy = 0.5 * stiffness * C * C
        return energy
