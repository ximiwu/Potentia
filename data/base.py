import abc
from typing import List

import taichi as ti


@ti.data_oriented
class ISimulationData(abc.ABC):
    """Stores global, dynamic simulation state."""

    @abc.abstractmethod
    def get_dofs(self) -> ti.Field:
        """Returns global degrees of freedom (e.g., positions)."""
        pass

    @abc.abstractmethod
    def get_velocities(self) -> ti.Field:
        """Returns global velocities."""
        pass

    @abc.abstractmethod
    def get_inv_masses(self) -> ti.Field:
        """Returns global inverse masses."""
        pass

    @abc.abstractmethod
    def get_masses(self) -> ti.Field:
        """Returns global masses."""
        pass

    @abc.abstractmethod
    def get_predicted_dofs(self) -> ti.Field:
        """Returns global predicted degrees of freedom (e.g., predicted positions)."""
        pass

    @abc.abstractmethod
    def get_num_dofs(self) -> int:
        """Returns the total number of degrees of freedom."""
        pass

    @abc.abstractmethod
    def allocate_dofs(self, num_dofs: int) -> int:
        """
        Allocates space for a number of DoFs and returns the starting offset.

        Args:
            num_dofs (int): The number of degrees of freedom to allocate.

        Returns:
            int: The starting index (offset) of the allocated block.
        """
        pass

    @abc.abstractmethod
    def swap_buffers(self) -> None:
        """
        Swaps the roles of the primary DoF buffer and the predicted DoF buffer.
        This is a fast, zero-copy operation used to finalize a time step.
        """
        pass