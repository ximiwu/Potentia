import abc
from typing import List

import taichi as ti

from data.base import ISimulationData
from forces.base import IForce


@ti.data_oriented
class IIntegrator(abc.ABC):
    """Manages time integration, inertial term calculation, and state updates."""

    @abc.abstractmethod
    def predict(self, data: ISimulationData, forces: List[IForce], dt: float) -> None:
        """
        Computes the predicted positions s_n and stores them in data.get_predicted_dofs().
        """
        pass

    @abc.abstractmethod
    def update_state(self, data: ISimulationData, dt: float) -> None:
        """
        Updates both velocities and positions.
        - Computes v_{n+1} using the original positions p_n (from data.get_dofs())
          and the new positions p_{n+1} (from data.get_predicted_dofs()).
        - Copies the new positions p_{n+1} from data.get_predicted_dofs() into
          data.get_dofs() to finalize the current time step.
        """
        pass
