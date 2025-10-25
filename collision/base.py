import abc
from typing import List

import taichi as ti

from data.base import ISimulationData
from energies.base import IPotentialEnergy


@ti.data_oriented
class ICollisionHandler(abc.ABC):
    """Detects collisions and dynamically generates temporary IPotentialEnergy terms."""

    @abc.abstractmethod
    def detect_and_create_potentials(self, data: ISimulationData, q_predict: ti.Field) -> List[IPotentialEnergy]:
        """Returns a (possibly empty) list of collision energy/constraint terms."""
        pass
