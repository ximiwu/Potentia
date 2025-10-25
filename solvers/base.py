import abc
from typing import List

import taichi as ti

from data.base import ISimulationData
from energies.base import IPotentialEnergy
from integrators.base import IIntegrator


@ti.data_oriented
class ISolver(abc.ABC):
    """Implements an optimization strategy to minimize total energy."""

    @abc.abstractmethod
    def solve(self, data: ISimulationData, dt: float) -> None:
        """
        Gets predicted positions s_n from data.get_predicted_dofs(), applies an 
        optimization algorithm based on the constraints in the energy_container, 
        and writes the resulting new positions p_{n+1} back to data.get_predicted_dofs().
        """
        pass
