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


            


    @ti.kernel
    def _fill_builders(self,
                        a_idx : int,
                        b_idx : int,
                        c_idx : int,
                        a: ti.f32,
                        b: ti.f32,
                        c: ti.f32,
                        d: ti.f32,
                        S: ti.types.sparse_matrix_builder(),
                        A: ti.types.sparse_matrix_builder()):

        S[0, a_idx] += 1.0
        S[1, b_idx] += 1.0
        S[2, c_idx] += 1.0
        A[0, 0] += a + c
        A[0, 1] += -a
        A[0, 2] += -c
        A[1, 0] += b + d
        A[1, 1] += -b
        A[1, 2] += -d
        

    def compute_pd_lhs_mat_func(
                                self, 
                                constraint: ti.template,
                                data: ISimulationData,):


        a_idx = constraint.v_indices[0]
        b_idx = constraint.v_indices[1]
        c_idx = constraint.v_indices[2]

        stiffness = constraint.params[0]

        Xg_a = constraint.params[1]
        Xg_b = constraint.params[3]
        Xg_c = constraint.params[2]
        Xg_d = constraint.params[4]

        surface_area = constraint.params[5]


        points_num = data.get_num_dofs()

        # 在 kernel 中填充 S/A builders
        S_builder = ti.linalg.SparseMatrixBuilder(3, points_num, max_num_triplets=3)
        A_builder = ti.linalg.SparseMatrixBuilder(2, 3, max_num_triplets=6)
        self._fill_builders(
                            a_idx,
                            b_idx,
                            c_idx,
                            Xg_a,
                            Xg_b,
                            Xg_c,
                            Xg_d,
                            S_builder,
                            A_builder)

        S = S_builder.build()
        A = A_builder.build()

        S_T = S.transpose()
        A_T = A.transpose()

        result = stiffness * surface_area * (S_T @ A_T @ A @ S)

        return result

