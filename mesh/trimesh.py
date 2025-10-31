from typing import Tuple, Type

import numpy as np
import taichi as ti
from taichi.math import vec3

try:
    import trimesh
except ImportError:
    trimesh = None

from .base import (
    IEdgeDataProvider,
    ISurfaceDataProvider,
    IVertexAdjacencyProvider,
    ICotanWeightProvider,
    IVertexAreaProvider,
    ISurfaceAreaProvider,
    ISurfaceLocalEdge2DProvider,
)

from .mesh import Mesh

from mesh.transforms import apply_transform_numpy


@ti.data_oriented
class TriMesh(
    Mesh,
    ISurfaceDataProvider,
    IEdgeDataProvider,
    IVertexAdjacencyProvider,
    ICotanWeightProvider,
    IVertexAreaProvider,
    ISurfaceAreaProvider,
    ISurfaceLocalEdge2DProvider,
):
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

        # Build CSR vertex adjacency using trimesh's vertex_neighbors
        if trimesh is None:
            raise ImportError("The 'trimesh' library is required to build vertex adjacency. Please install it using 'pip install trimesh'.")
        V = self._rest_positions_host.shape[0]
        tm = trimesh.Trimesh(vertices=self._rest_positions_host, faces=self._faces_host, process=False)
        neighbors = tm.vertex_neighbors  # list of sets/lists per vertex

        # Normalize neighbors: unique + sorted, drop self-loop if any
        lengths = np.fromiter((len(neighbors[i]) for i in range(V)), dtype=np.int32, count=V)
        # Precompute offsets
        self._vertex_adj_offsets_host = np.zeros(V + 1, dtype=np.int32)
        np.cumsum(lengths, out=self._vertex_adj_offsets_host[1:])

        # Fill indices
        total_neighbors = int(self._vertex_adj_offsets_host[-1])
        self._vertex_adj_indices_host = np.empty(total_neighbors, dtype=np.int32)
        write_ptr = 0
        for i in range(V):
            nbrs_i = np.asarray(list(neighbors[i]), dtype=np.int32)
            if nbrs_i.size > 0:
                # remove self if present, unique + sort
                nbrs_i = nbrs_i[nbrs_i != i]
                if nbrs_i.size > 1:
                    nbrs_i = np.unique(nbrs_i)
                nbrs_i.sort()
                count_i = nbrs_i.size
                self._vertex_adj_indices_host[write_ptr:write_ptr + count_i] = nbrs_i
                write_ptr += count_i

        # Compute per-vertex mixed Voronoi area (Meyer mixed area) and per-face area
        eps_area = 1e-12
        positions = self._rest_positions_host
        faces = self._faces_host
        A_vertex = np.zeros(V, dtype=np.float64)
        self._surface_areas_host = np.zeros(self._num_faces, dtype=np.float32)
        # Host buffer for per-face local 2D edge vectors (vec4)
        self._surface_local_edge_2x2_host = np.zeros((self._num_faces, 4), dtype=np.float32)

        # Accumulate per-edge cotangent sums (unordered edge -> sum of opposite cots)
        edge_cot_sum: dict[tuple[int, int], float] = {}
        neg_cot_count = 0
        neg_cot_examples: list[tuple[int, int, float]] = []
        degenerate_face_count = 0

        local2d_zero_count = 0
        for f_idx in range(faces.shape[0]):
            a, b, c = int(faces[f_idx, 0]), int(faces[f_idx, 1]), int(faces[f_idx, 2])
            pa = positions[a].astype(np.float64)
            pb = positions[b].astype(np.float64)
            pc = positions[c].astype(np.float64)

            # Common geometric quantities
            eab = pb - pa
            eac = pc - pa
            ebc = pc - pb
            eba = pa - pb
            eca = pa - pc
            ecb = pb - pc

            cross_pa = np.cross(eab, eac)
            cross_len = float(np.linalg.norm(cross_pa))
            tri_area = 0.5 * cross_len
            self._surface_areas_host[f_idx] = np.float32(tri_area)

            if cross_len <= eps_area:
                degenerate_face_count += 1
                # No area and no reliable cot contributions
                # local 2D edges remain zeros
                local2d_zero_count += 1
                continue

            # Per-face local orthonormal basis
            ab_len = float(np.linalg.norm(eab))  # |b - a|
            if ab_len <= eps_area:
                local2d_zero_count += 1
            else:
                e1 = eab / ab_len
                n_unit = cross_pa / cross_len
                e2 = np.cross(n_unit, e1)
                e2_len = float(np.linalg.norm(e2))
                if e2_len <= eps_area:
                    local2d_zero_count += 1
                else:
                    e2 = e2 / e2_len
                    # Project (b-a) and (c-a) onto (e1, e2)
                    u1 = -float(np.dot(eab, e1))
                    v1 = -float(np.dot(eab, e2))
                    u2 = -float(np.dot(eac, e1))
                    v2 = -float(np.dot(eac, e2))
                    self._surface_local_edge_2x2_host[f_idx, 0] = np.float32(u1)
                    self._surface_local_edge_2x2_host[f_idx, 1] = np.float32(v1)
                    self._surface_local_edge_2x2_host[f_idx, 2] = np.float32(u2)
                    self._surface_local_edge_2x2_host[f_idx, 3] = np.float32(v2)

            # Obtuse test via dot signs (no need for actual angles)
            dot_a = float(np.dot(eab, eac))
            dot_b = float(np.dot(ebc, eba))
            dot_c = float(np.dot(eca, ecb))
            is_obtuse_a = dot_a < 0.0
            is_obtuse_b = dot_b < 0.0
            is_obtuse_c = dot_c < 0.0

            # Cotangents for angles at vertices a, b, c
            denom = cross_len  # same for the three angles of this triangle
            cot_a = dot_a / denom
            cot_b = dot_b / denom
            cot_c = dot_c / denom

            if is_obtuse_a:
                A_vertex[a] += 0.5 * tri_area
                A_vertex[b] += 0.25 * tri_area
                A_vertex[c] += 0.25 * tri_area
            elif is_obtuse_b:
                A_vertex[b] += 0.5 * tri_area
                A_vertex[c] += 0.25 * tri_area
                A_vertex[a] += 0.25 * tri_area
            elif is_obtuse_c:
                A_vertex[c] += 0.5 * tri_area
                A_vertex[a] += 0.25 * tri_area
                A_vertex[b] += 0.25 * tri_area
            else:
                # Acute triangle: use circumcenter (Meyer) mixed Voronoi area
                sab2 = float(np.dot(eab, eab))  # |a-b|^2
                sac2 = float(np.dot(eac, eac))  # |a-c|^2
                sbc2 = float(np.dot(ebc, ebc))  # |b-c|^2

                A_vertex[a] += (sab2 * cot_c + sac2 * cot_b) / 8.0
                A_vertex[b] += (sbc2 * cot_a + sab2 * cot_c) / 8.0
                A_vertex[c] += (sac2 * cot_b + sbc2 * cot_a) / 8.0

            # Edge (b,c) opposite a
            e_bc = (b, c) if b < c else (c, b)
            edge_cot_sum[e_bc] = edge_cot_sum.get(e_bc, 0.0) + cot_a
            if cot_a < 0.0 and len(neg_cot_examples) < 8:
                neg_cot_examples.append((e_bc[0], e_bc[1], cot_a))
            neg_cot_count += int(cot_a < 0.0)

            # Edge (c,a) opposite b
            e_ca = (c, a) if c < a else (a, c)
            edge_cot_sum[e_ca] = edge_cot_sum.get(e_ca, 0.0) + cot_b
            if cot_b < 0.0 and len(neg_cot_examples) < 8:
                neg_cot_examples.append((e_ca[0], e_ca[1], cot_b))
            neg_cot_count += int(cot_b < 0.0)

            # Edge (a,b) opposite c
            e_ab = (a, b) if a < b else (b, a)
            edge_cot_sum[e_ab] = edge_cot_sum.get(e_ab, 0.0) + cot_c
            if cot_c < 0.0 and len(neg_cot_examples) < 8:
                neg_cot_examples.append((e_ab[0], e_ab[1], cot_c))
            neg_cot_count += int(cot_c < 0.0)

        # Allocate Taichi fields for adjacency
        self.vertex_adj_offsets = ti.field(dtype=ti.i32, shape=V + 1)
        self.vertex_adj_indices = ti.field(dtype=ti.i32, shape=total_neighbors)
        self.vertex_adj_cotan_weights = ti.field(dtype=ti.f32, shape=total_neighbors)

        # Build CSR-aligned cotan weights (directed i→j), normalized by A_i
        area_clamp_count = 0
        neg_weight_count = 0
        neg_weight_examples: list[tuple[int, int, float]] = []
        weights_host = np.zeros(total_neighbors, dtype=np.float32)

        # Convert A_vertex to float64 safe division then cast
        for i in range(V):
            start = int(self._vertex_adj_offsets_host[i])
            end = int(self._vertex_adj_offsets_host[i + 1])
            Ai = float(A_vertex[i])
            denom_area = Ai if Ai > eps_area else eps_area
            if Ai <= eps_area:
                area_clamp_count += (end - start)
            for ptr in range(start, end):
                j = int(self._vertex_adj_indices_host[ptr])
                e = (i, j) if i < j else (j, i)
                cot_sum = float(edge_cot_sum.get(e, 0.0))
                w = 0.5 * cot_sum / denom_area
                weights_host[ptr] = np.float32(w)
                if w < 0.0 and len(neg_weight_examples) < 8:
                    neg_weight_examples.append((i, j, w))
                neg_weight_count += int(w < 0.0)

        self._vertex_adj_cotan_weights_host = weights_host

        # Prepare host arrays for areas (vertex-wise mixed Voronoi)
        self._vertex_mixed_voronoi_areas_host = A_vertex.astype(np.float32)

        # Allocate Taichi fields for areas
        self.vertex_mixed_voronoi_areas = ti.field(dtype=ti.f32, shape=V)
        self.surface_areas = ti.field(dtype=ti.f32, shape=self._num_faces)
        # Allocate Taichi field for per-face local 2D edge vectors (vec4)
        self.surface_local_edge_2x2 = ti.Vector.field(4, dtype=ti.f32, shape=self._num_faces)

        if materialize:
            self.materialize()

        # Aggregated warnings
        if degenerate_face_count > 0:
            print(f"[TriMesh] Warning: {degenerate_face_count} degenerate faces detected (area≈0); cot set to 0.")
        if neg_cot_count > 0:
            print(f"[TriMesh] Warning: {neg_cot_count} negative cot contributions. Examples: {neg_cot_examples[:5]}")
        if area_clamp_count > 0:
            print(f"[TriMesh] Warning: {area_clamp_count} adjacency entries used clamped vertex area (<= {eps_area}).")
        if neg_weight_count > 0:
            print(f"[TriMesh] Warning: {neg_weight_count} negative cotan weights. Examples: {neg_weight_examples[:5]}")
        if local2d_zero_count > 0:
            print(f"[TriMesh] Warning: {local2d_zero_count} faces wrote local 2D edge vectors as zeros (degenerate basis).")

    def materialize(self):
        super().materialize()
        self.edge_indices.from_numpy(self._edges_host.astype(np.int32).flatten())
        self.surface_indices.from_numpy(self._faces_host.astype(np.int32).flatten())
        # Materialize adjacency
        self.vertex_adj_offsets.from_numpy(self._vertex_adj_offsets_host)
        if self.vertex_adj_indices.shape[0] > 0:
            self.vertex_adj_indices.from_numpy(self._vertex_adj_indices_host)
        if self.vertex_adj_cotan_weights.shape[0] > 0:
            self.vertex_adj_cotan_weights.from_numpy(self._vertex_adj_cotan_weights_host)
        # Materialize areas
        self.vertex_mixed_voronoi_areas.from_numpy(self._vertex_mixed_voronoi_areas_host)
        self.surface_areas.from_numpy(self._surface_areas_host)
        # Materialize per-face local 2D edge vectors
        self.surface_local_edge_2x2.from_numpy(self._surface_local_edge_2x2_host)



    def get_surface_indices(self) -> ti.Field:
        return self.surface_indices

    def get_edge_indices(self) -> ti.Field:
        return self.edge_indices

    def get_vertex_adjacency_offsets(self) -> ti.Field:
        return self.vertex_adj_offsets

    def get_vertex_adjacency_indices(self) -> ti.Field:
        return self.vertex_adj_indices

    def get_vertex_adjacency_cotan_weights(self) -> ti.Field:
        return self.vertex_adj_cotan_weights

    def get_vertex_mixed_voronoi_areas(self) -> ti.Field:
        return self.vertex_mixed_voronoi_areas

    def get_surface_areas(self) -> ti.Field:
        return self.surface_areas

    def get_surface_local_edge_2x2(self) -> ti.Field:
        return self.surface_local_edge_2x2

    @staticmethod
    def _load_from_obj(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [Protected Helper] Loads OBJ data only, without creating an instance.
        This can be reused by subclasses.
        """
        if trimesh is None:
            raise ImportError("The 'trimesh' library is required to load OBJ files. Please install it using 'pip install trimesh'.")
        
        mesh = trimesh.load(filepath, force='mesh')
        return mesh.vertices, mesh.faces, mesh.edges_unique

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
        
        return mesh.vertices, mesh.faces, mesh.edges_unique

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
        return mesh.vertices, mesh.faces, mesh.edges_unique

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
