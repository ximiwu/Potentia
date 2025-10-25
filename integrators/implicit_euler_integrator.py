
import abc
from typing import List

import taichi as ti

from data.base import ISimulationData
from forces.base import IForce
from integrators.base import IIntegrator


@ti.data_oriented
class ImplicitEulerIntegrator(IIntegrator):
    def __init__(self):
        super().__init__()
        self.force_vector = None

    def predict(self, data: ISimulationData, forces: List[IForce], dt: float) -> None:
        if self.force_vector is None or self.force_vector.shape[0] != data.get_num_dofs():
            self.force_vector = ti.Vector.field(3, dtype=ti.f32, shape=data.get_num_dofs())
        
        self.force_vector.fill(0.0)
        for force in forces:
            force.add_force_to_vector(data, self.force_vector)

        self._predict_kernel(data.get_dofs(), data.get_velocities(), data.get_inv_masses(), self.force_vector, data.get_predicted_dofs(), dt)


    @ti.kernel
    def _predict_kernel(self,
                        positions: ti.template(),
                        velocities: ti.template(),
                        inv_masses: ti.template(),
                        forces: ti.template(),
                        out_predicted_positions: ti.template(),
                        dt: ti.f32):
        for i in positions:
            if inv_masses[i] > 0.0:
                out_predicted_positions[i] = positions[i] + dt * velocities[i] + dt * dt * inv_masses[i] * forces[i]
            else:
                out_predicted_positions[i] = positions[i]
    
    def update_state(self, data: ISimulationData, dt: float) -> None:
        self._update_state_kernel(data.get_velocities(), data.get_dofs(), data.get_predicted_dofs(), dt)
        
        # Finalize the state update by swapping the roles of the position buffers.
        data.swap_buffers()

    @ti.kernel
    def _update_state_kernel(self,
                             velocities: ti.template(),
                             positions: ti.template(), # These are p_n (positions from the previous step)
                             predicted_positions: ti.template(), # These are p_{n+1} (new positions from the solver)
                             dt: ti.f32):
        for i in velocities:
            velocities[i] = (predicted_positions[i] - positions[i]) / dt
