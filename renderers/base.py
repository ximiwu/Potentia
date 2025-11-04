import abc
from typing import List

import taichi as ti

from data.base import ISimulationData
from objects.base import ISimulationObject



@ti.data_oriented
class IRenderer(abc.ABC):
    """Decoupled visualization module."""

    @abc.abstractmethod
    def render(self, data: ISimulationData, objects: List[ISimulationObject]):
        """Renders the current state of the simulation."""
        pass

    @abc.abstractmethod
    def present(self) -> None:
        """Presents the frame (swap buffers)."""
        pass

    @abc.abstractmethod
    def is_window_running(self) -> bool:
        """Returns True to continue the simulation loop."""
        pass
