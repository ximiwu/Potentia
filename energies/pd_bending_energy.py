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


    @ti.func
    def fill_pd_A_and_cols(self,
                           constraint: ti.template(),
                           vertex_adj_offsets: ti.template(),
                           vertex_adj_indices: ti.template(),
                           vertex_adj_cotan_weights: ti.template(),
                           A_out: ti.template(),
                           cols_out: ti.template()):

        center = constraint.v_indices[0]
        k_stiff = constraint.params[0]
        voronoi_area = constraint.params[2]

        start = vertex_adj_offsets[center]
        end = vertex_adj_offsets[center + 1]
        accum_c = 0.0

        ti.loop_config(serialize=True)
        for ptr in range(start, end):
            accum_c += vertex_adj_cotan_weights[ptr]

        cols_out[0] = center
        A_out[0, 0] += -accum_c
        col_count = 1
        for ptr in range(start, end):
            col_count += 1
            col = 1 + (ptr - start)
            j = vertex_adj_indices[ptr]
            c = vertex_adj_cotan_weights[ptr]
            cols_out[col] = j
            A_out[0, col] += c
        
        return col_count, k_stiff * voronoi_area
                           
        