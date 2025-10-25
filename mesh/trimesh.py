from typing import Tuple, Type

import numpy as np
import taichi as ti
from taichi.math import vec3

try:
    import trimesh
except ImportError:
    trimesh = None

from mesh.base import IEdgeDataProvider, IMesh, ISurfaceDataProvider
from mesh.mesh import Mesh
from mesh.transforms import apply_transform_numpy


@ti.data_oriented
class TriMesh(Mesh, ISurfaceDataProvider, IEdgeDataProvider):
    """
    A concrete mesh class for triangle-based surface meshes.
    Provides factory methods to load from OBJ files or create procedural primitives.
    """
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray, materialize: bool = False):
        super().__init__(vertices)
        # Accept array-like faces/edges; check capabilities and coerce
        if not (hasattr(faces, 'ndim') and hasattr(faces, 'shape')):
            try:
                faces = np.asarray(faces, dtype=np.int32)
            except Exception as e:
                raise TypeError("Faces must be array-like of shape (m, 3)") from e
        if not (hasattr(edges, 'ndim') and hasattr(edges, 'shape')):
            try:
                edges = np.asarray(edges, dtype=np.int32)
            except Exception as e:
                raise TypeError("Edges must be array-like of shape (k, 2)") from e
        
        self._num_faces = faces.shape[0]
        self._num_edges = edges.shape[0]

        self.edge_indices = ti.field(dtype=ti.i32, shape=self._num_edges * 2)
        self.surface_indices = ti.field(dtype=ti.i32, shape=self._num_faces * 3)

        # Ensure dtypes on host copies
        self._faces_host = np.asarray(faces, dtype=np.int32)
        self._edges_host = np.asarray(edges, dtype=np.int32)

        if materialize:
            self.materialize()

    def materialize(self):
        super().materialize()
        self.edge_indices.from_numpy(self._edges_host.astype(np.int32).flatten())
        self.surface_indices.from_numpy(self._faces_host.astype(np.int32).flatten())


    def get_surface_indices(self) -> ti.Field:
        return self.surface_indices

    def get_edge_indices(self) -> ti.Field:
        return self.edge_indices

    @staticmethod
    def _load_from_obj(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [Protected Helper] Loads OBJ data only, without creating an instance.
        This can be reused by subclasses.
        """
        if trimesh is None:
            raise ImportError("The 'trimesh' library is required to load OBJ files. Please install it using 'pip install trimesh'.")
        
        mesh = trimesh.load(filepath, force='mesh')
        return mesh.vertices, mesh.faces, mesh.edges

    @classmethod
    def from_obj(
        cls: Type['TriMesh'],
        filepath: str,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> 'TriMesh':
        """
        [Public Factory] Creates a TriMesh instance from an OBJ file with transformations.
        The transformations are baked into the rest positions of the mesh.
        """
        vertices_np, faces_np, edges_np = cls._load_from_obj(filepath)

        # Apply transformations directly on numpy arrays (Scale -> Rotate -> Translate)
        final_vertices_np = apply_transform_numpy(vertices_np, translation, rotation, scale)

        return cls(vertices=final_vertices_np, faces=faces_np, edges=edges_np)

    @staticmethod
    def _create_cube_data(
        extents: Tuple[float, float, float],
        subdivisions: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [Protected Helper] Generates vertex and face data for a cube.
        """
        if trimesh is None:
            raise ImportError("The 'trimesh' library is required to create primitives. Please install it using 'pip install trimesh'.")
        
        mesh = trimesh.creation.box(extents=np.array(extents))
        if subdivisions > 0:
            mesh = mesh.subdivide(subdivisions)
        
        return mesh.vertices, mesh.faces, mesh.edges

    @classmethod
    def create_cube(
        cls: Type['TriMesh'],
        extents: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        subdivisions: int = 0
    ) -> 'TriMesh':
        """
        [Public Factory] Creates a TriMesh instance representing a cube with transformations.
        """
        vertices_np, faces_np, edges_np = cls._create_cube_data(extents, subdivisions)

        final_vertices_np = apply_transform_numpy(vertices_np, translation, rotation, scale)

        return cls(vertices=final_vertices_np, faces=faces_np, edges=edges_np)

    @staticmethod
    def _create_sphere_data(radius: float, center: vec3, subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [Protected Helper] Generates vertex and face data for a sphere.
        """
        if trimesh is None:
            raise ImportError("The 'trimesh' library is required to create primitives. Please install it using 'pip install trimesh'.")
        
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius, center=np.array([center[0], center[1], center[2]]))
        return mesh.vertices, mesh.faces, mesh.edges

    @classmethod
    def create_sphere(
        cls: Type['TriMesh'],
        radius: float = 0.5,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        subdivisions: int = 3,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> 'TriMesh':
        """
        [Public Factory] Creates a TriMesh instance representing a sphere with transformations.
        """
        # Note: 'center' acts as an initial translation baked by trimesh library.
        # The final translation will be added to this.
        total_translation = np.array(translation, dtype=np.float32) + np.array(center, dtype=np.float32)
        
        vertices_np, faces_np, edges_np = cls._create_sphere_data(radius, vec3(0.0, 0.0, 0.0), subdivisions)

        final_vertices_np = apply_transform_numpy(vertices_np, total_translation, rotation, scale)

        return cls(vertices=final_vertices_np, faces=faces_np, edges=edges_np)
