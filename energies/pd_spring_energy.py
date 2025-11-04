from math import dist
import taichi as ti
import numpy as np

from .base import PotentialEnergy
from .global_energy_container import GlobalEnergyContainer
from data.base import ISimulationData


@ti.data_oriented
class PDSpringEnergy(PotentialEnergy):
    """
    Defines the local computation step for pd spring (Fast Simulation of Mass-Spring Systems - Tiantian Liu).
    This class is stateless and operates on data stored in the global container.
    """
    TYPE_ID = 3

    def __init__(self):
        super().__init__()
        # Cache the global container instance for use inside Taichi funcs via ti.static
        self._container = GlobalEnergyContainer.get_instance()

    @ti.func
    def add_one_constraint_func(self,
                                container: ti.template(),
                                constraint_idx: int,
                                v_indices: ti.types.vector(2, ti.i32),
                                distance: ti.f32,
                                stiffness: ti.f32,
                                ):



        params = ti.Vector([stiffness, distance])
        
        container.add_one_constraint(
            constraint_idx,
            self.TYPE_ID,
            v_indices,
            params
        )

    @ti.func
    def compute_energy_func(self, constraint: ti.template(), q: ti.template()) -> ti.f32:
        """
        真实弹簧势能：E = 1/2 * k * (||p1 - p2|| - L)^2
        """
        i = constraint.v_indices[0]
        j = constraint.v_indices[1]
        k_stiff = constraint.params[0]
        rest_len = constraint.params[1]

        r = q[i] - q[j]
        s = r.norm()
        dl = s - rest_len
        return 0.5 * k_stiff * dl * dl

    @ti.func
    def compute_gradient_func(self,
                              constraint: ti.template(),
                              q: ti.template(),
                              out_grad: ti.template()):
        """
        ∇E 对应两个端点的梯度：
        g1 = k * (1 - L/s) * r, g2 = -g1
        """
        i = constraint.v_indices[0]
        j = constraint.v_indices[1]
        k_stiff = constraint.params[0]
        rest_len = constraint.params[1]

        r = q[i] - q[j]
        s = r.norm()
        s_safe = ti.max(s, 1e-6)
        coeff = k_stiff * (1.0 - rest_len / s_safe)
        g1 = coeff * r
        g2 = -g1

        for c in ti.static(range(3)):
            ti.atomic_add(out_grad[i][c], g1[c])
            ti.atomic_add(out_grad[j][c], g2[c])

    @ti.func
    def compute_pd_rhs_vec_func(self, constraint: ti.template(), 
                                q_predict: ti.template(), 
                                vertex_adj_offsets: ti.template(), 
                                vertex_adj_indices: ti.template(), 
                                vertex_adj_cotan_weights: ti.template(), 
                                out_rhs: ti.template()):
        
        p1_idx = constraint.v_indices[0]
        p2_idx = constraint.v_indices[1]

        stiffness = constraint.params[0]

        distance = constraint.params[1]

        p1 = q_predict[p1_idx]
        p2 = q_predict[p2_idx]

        dir = p1 - p2
        norm = max(dir.norm(), 1e-6)
        d = dir / norm * distance * stiffness

        for k in ti.static(range(3)):
            ti.atomic_add(out_rhs[p1_idx][k], d[k])
            ti.atomic_add(out_rhs[p2_idx][k], -d[k])


    @ti.func
    def fill_pd_A_and_cols(self,
                           constraint: ti.template(),
                           vertex_adj_offsets: ti.template(),
                           vertex_adj_indices: ti.template(),
                           vertex_adj_cotan_weights: ti.template(),
                           A_out: ti.template(),
                           cols_out: ti.template()):
        # 两端点
        p1_idx = constraint.v_indices[0]
        p2_idx = constraint.v_indices[1]
        k_stiff = constraint.params[0]

        # A 为 1×2： [ 1, -1 ]

        A_out[0, 0] = 1
        A_out[0, 1] = -1

        # 列映射：
        cols_out[0] = p1_idx
        cols_out[1] = p2_idx

        return 2, k_stiff

            

    @ti.func
    def compute_hessian_block_ij_func(self,
                                      constraint: ti.template(),
                                      q: ti.template(),
                                      a: ti.i32,
                                      b: ti.i32) -> ti.types.matrix(3, 3, ti.f32):
        # 二元弹簧：local 0 -> p1, local 1 -> p2
        H = ti.Matrix.zero(ti.f32, 3, 3)
        if a < 2 and b < 2:
            i = constraint.v_indices[0]
            j = constraint.v_indices[1]
            k_stiff = constraint.params[0]
            rest_len = constraint.params[1]

            r = q[i] - q[j]
            s = r.norm()
            s_safe = ti.max(s, 1e-6)

            a_diag = k_stiff * (1.0 - rest_len / s_safe)
            b_rrt = k_stiff * (rest_len / (s_safe * s_safe * s_safe))

            Hblk = ti.Matrix.zero(ti.f32, 3, 3)
            for u in ti.static(range(3)):
                for v in ti.static(range(3)):
                    Hblk[u, v] = (a_diag if u == v else 0.0) + b_rrt * r[u] * r[v]

            sgn = 1.0
            if a != b:
                sgn = -1.0
            for u in ti.static(range(3)):
                for v in ti.static(range(3)):
                    H[u, v] = sgn * Hblk[u, v]
        return H

