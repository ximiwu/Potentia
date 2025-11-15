from math import dist
import taichi as ti
import numpy as np

from .base import PotentialEnergy
from .global_energy_container import GlobalEnergyContainer
from data.base import ISimulationData


@ti.data_oriented
class AttachmentEnergy(PotentialEnergy):

    TYPE_ID = 4


    def __init__(self):
        super().__init__()
        # Cache the global container instance for use inside Taichi funcs via ti.static
        self._container = GlobalEnergyContainer.get_instance()

    @ti.func
    def add_one_constraint_func(self,
                                container: ti.template(),
                                constraint_idx: int,
                                idx: ti.i32,
                                attach_pos: ti.types.vector(3, ti.f32),
                                stiffness: ti.f32,
                                ):
        v_indices = ti.Vector([idx])

        params = ti.Vector([stiffness, attach_pos[0], attach_pos[1], attach_pos[2]])
        
        container.add_one_constraint(
            constraint_idx,
            self.TYPE_ID,
            v_indices,
            params
        )

    @ti.func
    def compute_energy_func(self, constraint: ti.template(), q: ti.template()) -> ti.f32:
        """
        E = 1/2 * k * (||q[i] - attach_pos||)^2
        """
        idx = constraint.v_indices[0]
        k_stiff = constraint.params[0]
        attach_pos = ti.Vector([constraint.params[1], constraint.params[2], constraint.params[3]])

        r = q[idx] - attach_pos
        return 0.5 * k_stiff * r.dot(r)

    @ti.func
    def compute_gradient_func(self,
                              constraint: ti.template(),
                              q: ti.template(),
                              out_grad: ti.template()):
        """
        attachment spring gradient: k*(current_length)*current_direction
        """
        idx = constraint.v_indices[0]
        k_stiff = constraint.params[0]
        attach_pos = ti.Vector([constraint.params[1], constraint.params[2], constraint.params[3]])

        r = q[idx] - attach_pos
        
        g = k_stiff * r

        for c in ti.static(range(3)):
            ti.atomic_add(out_grad[idx][c], g[c])

    @ti.func
    def compute_pd_rhs_vec_func(self, constraint: ti.template(), 
                                q_predict: ti.template(), 
                                vertex_adj_offsets: ti.template(), 
                                vertex_adj_indices: ti.template(), 
                                vertex_adj_cotan_weights: ti.template(), 
                                out_rhs: ti.template()):
        
        idx = constraint.v_indices[0]

        stiffness = constraint.params[0]
        
        p2 = ti.Vector([constraint.params[1], constraint.params[2], constraint.params[3]])

        for k in ti.static(range(3)):
            ti.atomic_add(out_rhs[idx][k], p2[k] * stiffness)


    @ti.func
    def fill_pd_A_and_cols(self,
                           constraint: ti.template(),
                           vertex_adj_offsets: ti.template(),
                           vertex_adj_indices: ti.template(),
                           vertex_adj_cotan_weights: ti.template(),
                           A_out: ti.template(),
                           cols_out: ti.template()):
        idx = constraint.v_indices[0]
        k_stiff = constraint.params[0]


        A_out[0, 0] = 1

        cols_out[0] = idx

        return 1, k_stiff

            

    @ti.func
    def assemble_hessian_to_builder_func(self,
                                         constraint: ti.template(),
                                         q: ti.template(),
                                         out_builder: ti.template()):
        idx = constraint.v_indices[0]
        base = 3 * idx
        k_stiff = constraint.params[0]
        for d in ti.static(range(3)):
            out_builder[base + d, base + d] += k_stiff

    @ti.func
    def compute_hessian_block_ij_func(self,
                                      constraint: ti.template(),
                                      q: ti.template(),
                                      a: ti.i32,
                                      b: ti.i32,
                                      out_builder: ti.template()) -> ti.types.matrix(3, 3, ti.f32):
        H = ti.Matrix.zero(ti.f32, 3, 3)
        if a == 0 and b == 0:
            k_stiff = constraint.params[0]
            for u in ti.static(range(3)):
                H[u, u] = k_stiff
        return H


