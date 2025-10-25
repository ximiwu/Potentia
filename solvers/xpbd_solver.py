from typing import TYPE_CHECKING, List

import taichi as ti

from energies.global_energy_container import GlobalEnergyContainer
from solvers.base import ISolver

if TYPE_CHECKING:
    from data.base import ISimulationData




@ti.data_oriented
class XPBDSolver(ISolver):
    """
    An implementation of the Extended Position Based Dynamics (XPBD) solver.

    This solver iterates through constraints provided by a global energy container
    and updates particle positions to satisfy these constraints in a compliant manner.
    It uses Lagrange multipliers (`lambda`) which are preserved across time steps within the container.
    """

    def __init__(self, iterations: int = 10):
        """
        Initializes the XPBD solver.

        Args:
            iterations (int): The number of sub-steps (iterations) to perform per time step.
        """
        self.iterations = iterations
        self.energy_container = GlobalEnergyContainer.get_instance()

        



    def solve(
        self,
        data: "ISimulationData",
        dt: float,
    ) -> None:
        """
        Performs the XPBD solve.

        Args:
            data (ISimulationData): The simulation data container.
            dt (float): The time step duration.
        """


        num_constraints = self.energy_container.get_num_constraints()
        if num_constraints > 0:
            self._substep(
                self.iterations,
                num_constraints,
                data.get_predicted_dofs(),
                data.get_inv_masses(),
                dt,
                )

    @ti.kernel
    def _substep(self, 
        iterations: ti.i32,
        num_constraints: ti.i32,
        q: ti.template(),
        inv_masses: ti.template(),
        dt: ti.f32,
    ):
        """
        The core XPBD update loop, executed as a Taichi kernel.
        """
        ti.loop_config(serialize=True)
        for _ in range(iterations):
            ti.loop_config(serialize=True)
            for c in range(num_constraints):
                C, grads, num_vertices = self.energy_container.compute_one_constraint_gradient_func(c, q)
                
                constraint = self.energy_container.constraints[c]

                weighted_grad_sqr_norm = 0.0
                for i in range(num_vertices):
                    v_idx = constraint.v_indices[i]
                    if v_idx >= 0:
                        w_i = inv_masses[v_idx]
                        grad_i = grads[i, :]
                        weighted_grad_sqr_norm += w_i * grad_i.dot(grad_i)
                
                # Convention: stiffness is stored in params[1]
                stiffness = constraint.params[1]
                alpha = 1.0 / (stiffness * dt * dt)

                denominator = weighted_grad_sqr_norm + alpha

                if denominator < 1e-9:
                    continue
                
                # Compute the change in lambda
                current_lambda = constraint.lambda_
                delta_lambda = (-C - alpha * current_lambda) / denominator
                self.energy_container.constraints[c].lambda_ += delta_lambda

                # Update positions based on the change in lambda
                for i in range(num_vertices):
                    v_idx = constraint.v_indices[i]
                    if v_idx >= 0:
                        w_i = inv_masses[v_idx]
                        grad_i = grads[i, :]
                        q[v_idx] += w_i * grad_i * delta_lambda