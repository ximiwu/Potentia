import abc
from typing import List

import taichi as ti

from forces.base import IForce
from objects.base import ISimulationObject


@ti.data_oriented
class ISimulationWorld(abc.ABC):
    """
    Interface for the main simulation world, providing methods for high-level
    control that GUI or other modules can interact with.
    """

    @abc.abstractmethod
    def add_object(self, obj: ISimulationObject):
        """Adds a simulation object to the world."""
        pass

    @abc.abstractmethod
    def add_force(self, force: IForce):
        """Adds an external force to the world."""
        pass

    @abc.abstractmethod
    def step(self, dt: float):
        """Advances the simulation by one time step."""
        pass
