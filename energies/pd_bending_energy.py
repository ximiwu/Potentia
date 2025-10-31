import taichi as ti
import numpy as np

from .base import PotentialEnergy
from .global_energy_container import GlobalEnergyContainer
from data.base import ISimulationData


@ti.data_oriented
class PDBendingEnergy(PotentialEnergy):
    """
    Defines the local computation step for pd bending.
    This class is stateless and operates on data stored in the global container.
    """
    TYPE_ID = 1

    def __init__(self):
        super().__init__()
        # Cache the global container instance for use inside Taichi funcs via ti.static
        self._container = GlobalEnergyContainer.get_instance()

    @ti.func
    def compute_vertex_laplace(self,
                               center_idx: int,
                               dof: ti.template(),
                               vertex_adj_offsets: ti.template(),
                               vertex_adj_indices: ti.template(),
                               vertex_adj_cotan_weights: ti.template()) -> ti.types.vector(3, ti.f32):
        p_i = dof[center_idx]
        lap = ti.Vector([0.0, 0.0, 0.0])
        start = vertex_adj_offsets[center_idx]
        end = vertex_adj_offsets[center_idx + 1]
        for ptr in range(start, end):
            j = vertex_adj_indices[ptr]
            c = vertex_adj_cotan_weights[ptr]
            lap += c * (dof[j] - p_i)
        return lap

    @ti.func
    def add_one_constraint_func(self,
                                container: ti.template(),
                                constraint_idx: int,
                                idx: int,
                                stiffness: ti.f32,
                                voronoi_area: ti.f32,
                                dof: ti.template(),
                                vertex_adj_offsets: ti.template(),
                                vertex_adj_indices: ti.template(),
                                vertex_adj_cotan_weights: ti.template(),
                                ):
        v_indices = ti.Vector([idx])
        rest_lap = self.compute_vertex_laplace(
            idx,
            dof,
            vertex_adj_offsets,
            vertex_adj_indices,
            vertex_adj_cotan_weights,
        )
        params = ti.Vector([stiffness, rest_lap.norm(), voronoi_area])
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
        
        center_vertex_idx = constraint.v_indices[0]
        stiffness = constraint.params[0]
        rest_laplace = constraint.params[1]
        voronoi_area = constraint.params[2]

    
        # Compute current Laplacian at center vertex using q_predict
        cur_lap = self.compute_vertex_laplace(
            center_vertex_idx,
            q_predict,
            vertex_adj_offsets,
            vertex_adj_indices,
            vertex_adj_cotan_weights,
        )

        lap_norm = cur_lap.norm()
        R_vg = cur_lap / ti.max(lap_norm, 1e-6) * rest_laplace * stiffness * voronoi_area
        # if(cur_lap.norm() < 1e-6):
        #     print("pd_bending_energy: bad current_laplace.norm()")

        start = vertex_adj_offsets[center_vertex_idx]
        end = vertex_adj_offsets[center_vertex_idx + 1]

        accum_c = 0.0

        for ptr in range(start, end):
            j = vertex_adj_indices[ptr]
            c = vertex_adj_cotan_weights[ptr]
            accum_c += c

            for k in ti.static(range(3)):
                ti.atomic_add(out_rhs[j][k], R_vg[k] * c)
        
        for k in ti.static(range(3)):
            ti.atomic_add(out_rhs[center_vertex_idx][k], R_vg[k] * (-accum_c))
        

    @ti.kernel
    def _fill_bending_builders(self,
                               center: ti.i32,
                               offsets: ti.template(),
                               indices: ti.template(),
                               cotan: ti.template(),
                               S: ti.types.sparse_matrix_builder(),
                               A: ti.types.sparse_matrix_builder()):
        start = offsets[center]
        end = offsets[center + 1]
        accum_c = 0.0

        ti.loop_config(serialize=True)
        for ptr in range(start, end):
            accum_c += cotan[ptr]

        S[0, center] += 1.0
        A[0, 0] += -accum_c
        for ptr in range(start, end):
            row = 1 + (ptr - start)
            j = indices[ptr]
            c = cotan[ptr]
            S[row, j] += 1.0
            A[0, row] += c

    def compute_pd_lhs_mat_func(
                                self, 
                                constraint: ti.template,
                                data: ISimulationData,):


        center_vertex_idx = constraint.v_indices[0]
        stiffness = constraint.params[0]
        rest_laplace = constraint.params[1]
        voronoi_area = constraint.params[2]

        vertex_adj_offsets, vertex_adj_info = data.get_vertex_adjacency()
        vertex_adj_indices = vertex_adj_info.vertex_adj_indices
        vertex_adj_cotan_weights = vertex_adj_info.vertex_adj_cotan_weights

        points_num = data.get_num_dofs()

        # 仅用于确定 builder 尺寸：在 Python 侧读取 offsets
        offsets_np = vertex_adj_offsets.to_numpy()
        start = offsets_np[center_vertex_idx]
        end = offsets_np[center_vertex_idx + 1]
        adj_num = end - start

        # 在 kernel 中填充 S/A builders
        S_builder = ti.linalg.SparseMatrixBuilder(adj_num + 1, points_num, max_num_triplets=adj_num + 1)
        A_builder = ti.linalg.SparseMatrixBuilder(1, adj_num + 1, max_num_triplets=adj_num + 1)
        self._fill_bending_builders(center_vertex_idx,
                                    vertex_adj_offsets,
                                    vertex_adj_indices,
                                    vertex_adj_cotan_weights,
                                    S_builder,
                                    A_builder)

        S = S_builder.build()
        A = A_builder.build()

        S_T = S.transpose()
        A_T = A.transpose()

        result = stiffness * voronoi_area * (S_T @ A_T @ A @ S)

        return result

