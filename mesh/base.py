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
class IVertexAdjacencyProvider(abc.ABC):
    """Capability interface for meshes exposing per-vertex adjacency in CSR layout."""

    @abc.abstractmethod
    def get_vertex_adjacency_offsets(self) -> ti.Field:
        """Returns CSR offsets of vertex adjacency (length V+1)."""
        pass

    @abc.abstractmethod
    def get_vertex_adjacency_indices(self) -> ti.Field:
        """Returns CSR indices of vertex adjacency (flattened neighbor vertex indices)."""
        pass


@ti.data_oriented
class ICotanWeightProvider(abc.ABC):
    """Capability interface for meshes exposing CSR-aligned cotan weights.

    The returned field must be parallel to the vertex adjacency indices (CSR),
    i.e., for every directed neighbor pair i→j in the adjacency, there is one
    corresponding weight entry at the same index.
    """

    @abc.abstractmethod
    def get_vertex_adjacency_cotan_weights(self) -> ti.Field:
        """Returns CSR-aligned cotangent weights for vertex adjacency."""
        pass


@ti.data_oriented
class IVertexAreaProvider(abc.ABC):
    """Capability interface for meshes exposing per-vertex mixed Voronoi areas.

    The area definition follows the Meyer mixed Voronoi region:
    - For obtuse triangles: the obtuse vertex receives 1/2 triangle area, the
      other two vertices receive 1/4 triangle area each.
    - For acute triangles: use circumcenter-based formula with cotangents.
    """

    @abc.abstractmethod
    def get_vertex_mixed_voronoi_areas(self) -> ti.Field:
        """Returns per-vertex mixed Voronoi areas as a Taichi field of length V."""
        pass


@ti.data_oriented
class ISurfaceAreaProvider(abc.ABC):
    """Capability interface for meshes exposing per-face surface areas."""

    @abc.abstractmethod
    def get_surface_areas(self) -> ti.Field:
        """Returns per-triangle absolute areas as a Taichi field of length F."""
        pass


@ti.data_oriented
class ISurfaceLocalEdge2DProvider(abc.ABC):
    """Capability interface for meshes exposing per-face local 2D edge vectors.

    Each triangle face contributes one vec4 packed as:
    [(a−b).x2d, (a−b).y2d, (a−c).x2d, (a−c).y2d],
    where the 2D components are computed in the per-face orthonormal basis
    e1 = normalize(b − a), n = normalize((b − a) × (c − a)), e2 = normalize(n × e1).
    """

    @abc.abstractmethod
    def get_surface_local_edge_2x2(self) -> ti.Field:
        """Returns a Taichi field of shape F with vec4 entries per face."""
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
