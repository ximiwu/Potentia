import abc
from typing import List, Optional

import taichi as ti

from forces.base import IForce
from actuators.base import IVertexActuator
from objects.base import ISimulationObject
from recorders.base import IRecorder


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
    def add_actuator(self, actuator: IVertexActuator) -> None:
        """Adds a vertex actuator that will be applied each frame (predict→solve)."""
        pass

    @abc.abstractmethod
    def get_recorder(self) -> Optional[IRecorder]:
        """Returns the recorder if available; otherwise None."""
        pass

    @abc.abstractmethod
    def step(self, dt: float):
        """Advances the simulation by one time step."""
        pass
