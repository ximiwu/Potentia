import abc
from typing import List, Type

import taichi as ti
from taichi.math import *


@ti.data_oriented
class IEdgeDataProvider(abc.ABC):
    """Capability interface for meshes with edge topology."""

    @abc.abstractmethod
    def get_edge_indices(self) -> ti.Field:
        """Returns edge indices."""
        pass


@ti.data_oriented
class ISurfaceDataProvider(abc.ABC):
    """Capability interface for meshes with surface (triangle) topology."""

    @abc.abstractmethod
    def get_surface_indices(self) -> ti.Field:
        """Returns surface (triangle) indices."""
        pass


@ti.data_oriented
class ITetDataProvider(abc.ABC):
    """Capability interface for meshes with tetrahedral topology."""

    @abc.abstractmethod
    def get_tet_indices(self) -> ti.Field:
        """Returns tetrahedral indices."""
        pass


@ti.data_oriented
class IMesh(abc.ABC):
    """Stores the static, read-only rest geometry and topology of an object."""

    @abc.abstractmethod
    def get_rest_positions(self) -> ti.Field:
        """Returns rest positions in local coordinates."""
        pass

    @abc.abstractmethod
    def get_num_vertices(self) -> int:
        """Returns the number of vertices in this mesh."""
        pass

    def compute_rest_shape_matrices(self) -> ti.Field:
        """(Optional) Computes rest shape matrices Dm^-1."""
        raise NotImplementedError

    def compute_laplacian_weights(self) -> ti.linalg.SparseMatrix:
        """
        (Optional) Computes Laplacian weights for energies like uniform Laplacian energy.
        """
        raise NotImplementedError
