import abc
from typing import List

import taichi as ti

from data.base import ISimulationData


@ti.data_oriented
class IForce(abc.ABC):
    """
    An abstract base class for all force fields.
    Forces are responsible for modifying velocities based on positions.
    """
    @abc.abstractmethod
    def add_force_to_vector(self, data: ISimulationData, force_vector: ti.Field):
        """
        Computes the force and adds its contribution to the provided force_vector.

        Args:
            data (ISimulationData): The simulation data container with current states.
            force_vector (ti.Field): The global force vector to accumulate forces into.
        """
        pass
