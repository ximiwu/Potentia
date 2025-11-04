import taichi as ti
import numpy as np

from .base import PotentialEnergy
from .global_energy_container import GlobalEnergyContainer
from data.base import ISimulationData


@ti.data_oriented
class PDStrainEnergy(PotentialEnergy):
    """
    Defines the local computation step for pd strain.
    This class is stateless and operates on data stored in the global container.
    """
    TYPE_ID = 2

    def __init__(self):
        super().__init__()
        # Cache the global container instance for use inside Taichi funcs via ti.static
        self._container = GlobalEnergyContainer.get_instance()

    @ti.func
    def add_one_constraint_func(self,
                                container: ti.template(),
                                constraint_idx: int,
                                v_indices: ti.types.vector(3, ti.i32),
                                local_edge : ti.types.vector(4, ti.f32),
                                surface_area : ti.f32,
                                stiffness: ti.f32,
                                singular_min: ti.f32,
                                singular_max: ti.f32

                                ):


        #inverse
        denom = 1.0 / (local_edge[0] * local_edge[3] - local_edge[2] * local_edge[1])
        params = ti.Vector([stiffness, local_edge[3] * denom , -local_edge[1] * denom, -local_edge[2] * denom, local_edge[0] * denom, surface_area, singular_min, singular_max])
        
        container.add_one_constraint(
            constraint_idx,
            self.TYPE_ID,
            v_indices,
            params
        )

    @ti.func
    def compute_pd_rhs_vec_func(self, constraint: ti.template(), 
                                q_predict: ti.template(), 
                                vertex_adj_offsets: ti.template(), 
                                vertex_adj_indices: ti.template(), 
                                vertex_adj_cotan_weights: ti.template(), 
                                out_rhs: ti.template()):
        
        a_idx = constraint.v_indices[0]
        b_idx = constraint.v_indices[1]
        c_idx = constraint.v_indices[2]

        stiffness = constraint.params[0]

        a = constraint.params[1]
        b = constraint.params[3]
        c = constraint.params[2]
        d = constraint.params[4]

        surface_area = constraint.params[5]

        singular_min = constraint.params[6]
        singular_max = constraint.params[7]

        ab = q_predict[a_idx] - q_predict[b_idx]
        ac = q_predict[a_idx] - q_predict[c_idx]

        X_f_X_g = ti.Matrix([
                            [ab[0] * a + ac[0] * c, ab[0] * b + ac[0] * d, 0.0],
                            [ab[1] * a + ac[1] * c, ab[1] * b + ac[1] * d, 0.0],
                            [ab[2] * a + ac[2] * c, ab[2] * b + ac[2] * d, 0.0]
        ])

        U, S, V = ti.svd(X_f_X_g)

        S[0, 0] = ti.math.clamp(S[0, 0], singular_min, singular_max)
        S[1, 1] = ti.math.clamp(S[1, 1], singular_min, singular_max)
        S[2, 2] = 0.0

        T = (U @ S @ V.transpose()) * stiffness * surface_area



        for k in ti.static(range(3)):
            ti.atomic_add(out_rhs[a_idx][k], T[k, 0] * (a + c) + T[k, 1] * (b + d))
            ti.atomic_add(out_rhs[b_idx][k], T[k, 0] * (-a) + T[k, 1] * (-b))
            ti.atomic_add(out_rhs[c_idx][k], T[k, 0] * (-c) + T[k, 1] * (-d))


    @ti.func
    def fill_pd_A_and_cols(self,
                           constraint: ti.template(),
                           vertex_adj_offsets: ti.template(),
                           vertex_adj_indices: ti.template(),
                           vertex_adj_cotan_weights: ti.template(),
                           A_out: ti.template(),
                           cols_out: ti.template()):

        a_idx = constraint.v_indices[0]
        b_idx = constraint.v_indices[1]
        c_idx = constraint.v_indices[2]

        stiffness = constraint.params[0]

        a = constraint.params[1]
        b = constraint.params[3]
        c = constraint.params[2]
        d = constraint.params[4]

        surface_area = constraint.params[5]

        cols_out[0] = a_idx
        cols_out[1] = b_idx
        cols_out[2] = c_idx

        A_out[0, 0] = a + c
        A_out[0, 1] = -a
        A_out[0, 2] = -c
        A_out[1, 0] = b + d
        A_out[1, 1] = -b
        A_out[1, 2] = -d

        return 3, stiffness * surface_area
