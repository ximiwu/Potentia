import abc
from typing import Any, Optional, Tuple

import taichi as ti

from data.base import ISimulationData
from energies.base import IPotentialEnergy
from mesh.base import IMesh


class ISimulationObject(abc.ABC):
    """
    An abstract base class for any object participating in the simulation.
    It represents a set of degrees of freedom (DoFs) within the global simulation data.
    """
    @abc.abstractmethod
    def __init__(self, data: ISimulationData):
        """
        In the implementation, degrees of freedom must be allocated from the `data` container.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_data_offset(self) -> int:
        """Returns the starting index in ISimulationData."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_dofs(self) -> int:
        """Returns the number of DoFs this object occupies."""
        raise NotImplementedError


class IMeshObject(ISimulationObject):
    """
    An abstract base class for simulation objects that are based on a static mesh.
    It binds an IMesh to a slice of ISimulationData.
    """
    @abc.abstractmethod
    def __init__(self, mesh: IMesh, data: ISimulationData):
        """
        In the implementation, degrees of freedom must be allocated from the `data` container,
        and the rest positions of the `mesh` must be copied into the allocated slice.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_mesh(self) -> IMesh:
        raise NotImplementedError

    @abc.abstractmethod
    def get_color(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """
        Returns a tuple of colors for rendering.

        Returns:
            Tuple[face_color, edge_color, vertex_color]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_mass(self, local_index: int, mass: float) -> None:
        """
        Set mass for a single vertex specified by its local index within this object.

        Behavior:
            - If mass == -1, set inv_mass to 0 and mass to -1 (pinned vertex).
            - Otherwise, set inv_mass = 1 / mass and mass = mass.
        """
        raise NotImplementedError