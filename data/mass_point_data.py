import taichi as ti
from typing import Tuple

from .base import ISimulationData


@ti.data_oriented
class MassPointData(ISimulationData):
    """
    Simulation data container for a collection of 3D mass points.
    DoFs are the 3D positions of the points.
    """
    def __init__(self, max_point_num: int, max_degree: int = 16):
        if max_point_num <= 0:
            raise ValueError("max_point_num must be positive.")
        if max_degree <= 0:
            raise ValueError("max_degree must be positive.")

        self._max_point_num = max_point_num
        self._current_num_dofs = 0
        self._is_primary_buffer_a = True # True: positions is primary, False: predicted_positions is primary
        self._max_degree = int(max_degree)

        # Use a Struct.field to define an AoS layout
        self.particle_data = ti.Struct.field({
            "positions": ti.math.vec3,
            "predicted_positions": ti.math.vec3,
            "velocities": ti.math.vec3,
            "masses": ti.f32,
            "inv_masses": ti.f32
        }, shape=(self._max_point_num,))

        # Global vertex adjacency in CSR form (directed i→j), aligned with TriMesh semantics
        self.vertex_adj_offsets = ti.field(dtype=ti.i32, shape=self._max_point_num + 1)
        self.vertex_adj_info = ti.Struct.field({
            "vertex_adj_indices": ti.i32,
            "vertex_adj_cotan_weights": ti.f32,
        }, shape=self._max_point_num * self._max_degree)

        # Global write pointer for CSR adjacency (number of used entries in vertex_adj_info)
        self.vertex_adj_next = ti.field(dtype=ti.i32, shape=())
        self.vertex_adj_next[None] = 0

    def get_dofs(self) -> ti.Field:
        if self._is_primary_buffer_a:
            return self.particle_data.positions
        else:
            return self.particle_data.predicted_positions

    def get_velocities(self) -> ti.Field:
        return self.particle_data.velocities

    def get_inv_masses(self) -> ti.Field:
        return self.particle_data.inv_masses

    def get_masses(self) -> ti.Field:
        return self.particle_data.masses

    def get_predicted_dofs(self) -> ti.Field:
        if self._is_primary_buffer_a:
            return self.particle_data.predicted_positions
        else:
            return self.particle_data.positions

    def get_num_dofs(self) -> int:
        return self._current_num_dofs

    def get_max_num_dofs(self) -> int:
        return self._max_point_num

    def allocate_dofs(self, num_dofs: int) -> int:
        if self._current_num_dofs + num_dofs > self._max_point_num:
            raise RuntimeError(
                f"Cannot allocate {num_dofs} DoFs. "
                f"Available: {self._max_point_num - self._current_num_dofs}, "
                f"Total capacity: {self._max_point_num}."
            )
        
        offset = self._current_num_dofs
        self._current_num_dofs += num_dofs
        return offset

    def swap_buffers(self) -> None:
        self._is_primary_buffer_a = not self._is_primary_buffer_a

    def get_vertex_adjacency(self) -> Tuple[ti.Field, ti.Field]:
        """
        Returns global CSR vertex adjacency fields with directed semantics (i→j):
        - offsets: length = num_vertices + 1
        - info: Struct field with (vertex_adj_indices: i32, vertex_adj_cotan_weights: f32)
        """
        return self.vertex_adj_offsets, self.vertex_adj_info

    def get_vertex_adjacency_write_ptr(self) -> ti.Field:
        """Returns the scalar field that stores the global write pointer for CSR info array."""
        return self.vertex_adj_next

    def get_max_degree(self) -> int:
        return self._max_degree

    def reset_vertex_adjacency(self) -> None:
        """Resets the global CSR write pointer to zero (offsets are not cleared)."""
        self.vertex_adj_next[None] = 0
