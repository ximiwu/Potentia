from typing import List, Optional, Tuple

import taichi as ti
from taichi.math import vec3

from data.base import ISimulationData
from energies.base import IPotentialEnergy
from energies.distance_energy import DistanceEnergy
from energies.global_energy_container import GlobalEnergyContainer
from mesh.base import IEdgeDataProvider, IMesh
from mesh.transforms import rotate_positions, scale_positions
from objects.base import IMeshObject

@ti.data_oriented
class MeshObject(IMeshObject):
    """
    A concrete implementation of IMeshObject.

    This class binds a static mesh (IMesh) to a dynamic slice of simulation data
    (ISimulationData) and serves as a factory for creating potential energy terms
    that act upon that data.
    """

    def __init__(
        self,
        mesh: IMesh,
        data: ISimulationData,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        face_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        edge_color: Optional[Tuple[float, float, float]] = None,
        vertex_color: Optional[Tuple[float, float, float]] = None,
        mass: float = 1.0,
    ):
        # Interface adherence via capability checks rather than isinstance
        required_mesh_methods = ['get_rest_positions', 'get_num_vertices']
        for method_name in required_mesh_methods:
            if not hasattr(mesh, method_name):
                raise TypeError(f"Argument 'mesh' must implement IMesh capability '{method_name}'.")
        required_data_methods = [
            'allocate_dofs', 'get_dofs', 'get_inv_masses', 'get_masses'
        ]
        for method_name in required_data_methods:
            if not hasattr(data, method_name):
                raise TypeError(f"Argument 'data' must implement ISimulationData capability '{method_name}'.")

        # Cache DistanceEnergy singleton for use inside Taichi kernels via ti.static
        self._distance_energy = DistanceEnergy.get_instance()
        self._energy_container = GlobalEnergyContainer.get_instance()

        self._mesh = mesh
        self._data = data
        self._mass = mass
        self._face_color = face_color
        self._edge_color = edge_color
        self._vertex_color = vertex_color
        
        # Automatically allocate space from the simulation data container
        self._data_offset = self._data.allocate_dofs(self._mesh.get_num_vertices())

        # --- Apply transformations ---
        # The transformation pipeline is Scale -> Rotate -> Translate
        transformed_positions = self._mesh.get_rest_positions()

        # 1. Scale
        if scale != (1.0, 1.0, 1.0):
            transformed_positions = scale_positions(transformed_positions, scale)
        
        # 2. Rotate
        if rotation != (0.0, 0.0, 0.0):
            transformed_positions = rotate_positions(transformed_positions, rotation)
        
        # self._initialize_positions_kernel()
        self._initialize_positions_kernel(
            transformed_positions,
            self._data.get_dofs(),
            self._data.get_inv_masses(),
            self._data.get_masses(),
            self._data_offset,
            vec3(*translation),
            self._mass,
        )


    @ti.kernel
    def _set_mass_kernel(
        self,
        inv_masses: ti.template(),
        masses: ti.template(),
        idx: ti.i32,
        inv_mass_val: ti.f32,
        mass_val: ti.f32,
    ):
        inv_masses[idx] = inv_mass_val
        masses[idx] = mass_val


    @ti.kernel
    def _initialize_positions_kernel(
        self,
        mesh_positions: ti.template(),
        sim_data_dofs: ti.template(),
        inv_masses: ti.template(),
        masses: ti.template(),
        offset: ti.i32,
        translation: ti.types.vector(3, ti.f32),
        mass: ti.f32,
    ):
        for i in range(mesh_positions.shape[0]):
            sim_data_dofs[offset + i] = mesh_positions[i] + translation
            inv_masses[offset + i] = 1.0 / mass if mass != 0.0 else -1.0
            masses[offset + i] = mass



    @ti.kernel
    def _add_distance_constraints_kernel(
        self,
        dofs: ti.template(),
        mesh_edge_indices: ti.template(),
        data_offset: ti.i32,
        stiffness: ti.f32,
        start_idx: ti.i32,
        distance_energy: ti.template(),
        energy_container: ti.template(),
        num_edges: ti.i32
    ):
        """
        Taichi kernel to compute rest lengths from the current deformed shape and
        add distance constraints directly to the global energy container.
        """
        for i in range(num_edges):
            idx1_local = mesh_edge_indices[2 * i + 0]
            idx2_local = mesh_edge_indices[2 * i + 1]

            p1_idx_global = idx1_local + data_offset
            p2_idx_global = idx2_local + data_offset

            # Calculate rest distance from the CURRENT positions in the global dofs array
            p1_current = dofs[p1_idx_global]
            p2_current = dofs[p2_idx_global]
            rest_dist = (p1_current - p2_current).norm()
            
            constraint_idx = start_idx + i
            distance_energy.add_one_constraint_func(
                energy_container,
                constraint_idx,
                p1_idx_global,
                p2_idx_global,
                rest_dist,
                stiffness
            )

    def get_color(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        return self._face_color, self._edge_color, self._vertex_color

    def get_mesh(self) -> IMesh:
        return self._mesh

    def get_data_offset(self) -> int:
        return self._data_offset

    def get_num_dofs(self) -> int:
        return self._mesh.get_num_vertices()

    def set_mass(self, local_index: int, mass: float) -> None:
        if local_index < 0 or local_index >= self._mesh.get_num_vertices():
            raise IndexError(f"local_index {local_index} out of range [0, {self._mesh.get_num_vertices()-1}]")

        global_idx = self._data_offset + local_index

        if mass == -1.0:
            # Pinned vertex per spec: inv_mass = 0, mass = -1
            inv_mass_val = 0.0
            mass_val = -1.0
        else:
            if mass <= 0.0:
                raise ValueError("mass must be positive, or -1 for pinned vertex")
            inv_mass_val = 1.0 / float(mass)
            mass_val = float(mass)

        self._set_mass_kernel(
            self._data.get_inv_masses(),
            self._data.get_masses(),
            global_idx,
            inv_mass_val,
            mass_val,
        )

    def add_xpbd_distance_energy(self, stiffness: float):
        """
        Factory method to create and add PBD-style distance constraints (edge springs)
        to the global DistanceEnergy singleton.

        This method requires the associated mesh to provide edge connectivity data.

        Args:
            stiffness (float): The stiffness parameter for this batch of constraints.
        """
        # --- Type and Capability Checking ---
        if not hasattr(self._mesh, 'get_edge_indices'):
            raise AttributeError("The mesh does not provide edge data (IEdgeDataProvider). Cannot create distance energy.")

        # --- Constraint Preparation ---
        edge_indices = self._mesh.get_edge_indices()
        assert edge_indices.shape[0] % 2 == 0
        num_edges = edge_indices.shape[0] // 2
        if num_edges == 0:
            return

        start_idx = self._energy_container.reserve_constraints(num_edges)

        self._add_distance_constraints_kernel(
            self._data.get_dofs(),
            edge_indices,
            self._data_offset,
            stiffness,
            start_idx,
            self._distance_energy,
            self._energy_container,
            num_edges,

        )

    def add_pd_strain_energy(self, youngs_modulus: float, poissons_ratio: float):
        """
        Factory method to create a Projective Dynamics-style strain energy.

        This energy requires the associated mesh to provide tetrahedral connectivity data.

        Args:
            youngs_modulus (float): The Young's modulus (stiffness) of the material.
            poissons_ratio (float): The Poisson's ratio of the material.
        """
        if not hasattr(self._mesh, 'get_tet_indices'):
            raise AttributeError("The mesh does not provide tetrahedral data (ITetDataProvider). Cannot create strain energy.")

        # Placeholder for PDStrainEnergy instantiation
        # energy = PDStrainEnergy(self, youngs_modulus, poissons_ratio)
        # self._energies.append(energy)
        print(f"INFO: Placeholder for creating and adding PDStrainEnergy with E={youngs_modulus}, nu={poissons_ratio}.")
