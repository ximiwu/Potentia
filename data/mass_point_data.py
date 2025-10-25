import taichi as ti

from data.base import ISimulationData


@ti.data_oriented
class MassPointData(ISimulationData):
    """
    Simulation data container for a collection of 3D mass points.
    DoFs are the 3D positions of the points.
    """
    def __init__(self, max_point_num: int):
        if max_point_num <= 0:
            raise ValueError("max_point_num must be positive.")

        self._max_point_num = max_point_num
        self._current_num_dofs = 0
        self._is_primary_buffer_a = True # True: positions is primary, False: predicted_positions is primary

        # Use a Struct.field to define an AoS layout
        self.particle_data = ti.Struct.field({
            "positions": ti.math.vec3,
            "predicted_positions": ti.math.vec3,
            "velocities": ti.math.vec3,
            "masses": ti.f32,
            "inv_masses": ti.f32
        }, shape=(self._max_point_num,))

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
