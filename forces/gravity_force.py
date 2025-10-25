import taichi as ti

from data.base import ISimulationData
from forces.base import IForce


@ti.data_oriented
class GravityForce(IForce):
    """
    A simple force field that applies a constant gravitational force.
    """
    def __init__(self, gravity: ti.math.vec3 = ti.math.vec3(0, -9.8, 0)):
        """
        Initializes the gravity force.

        Args:
            gravity (ti.math.vec3): The gravity vector.
        """
        self.gravity = gravity

    def add_force_to_vector(self, data: ISimulationData, force_vector: ti.Field):
        """
        Adds the gravitational force to the force vector for all particles with mass.

        Args:
            data (ISimulationData): The simulation data.
            force_vector (ti.Field): The vector to accumulate forces into.
        """
        self.apply_gravity_kernel(
            data.get_inv_masses(),
            force_vector
        )

    @ti.kernel
    def apply_gravity_kernel(
        self,
        inv_masses: ti.template(),
        force_vector: ti.template()
    ):
        """
        Taichi kernel to apply gravity to each degree of freedom.

        Args:
            inv_masses (ti.template()): The inverse masses of the particles.
            force_vector (ti.template()): The global force vector.
        """
        for i in force_vector:
            if inv_masses[i] > 1e-6:  # Check for non-static particles
                mass = 1.0 / inv_masses[i]
                gravity_force = self.gravity * mass
                force_vector[i] += gravity_force
