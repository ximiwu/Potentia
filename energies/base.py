import abc
from typing import Any, Dict

import taichi as ti

from data.base import ISimulationData


@ti.data_oriented
class IPotentialEnergy(abc.ABC):
    """
    Abstract base class for a specific type of potential energy.
    This class is responsible for two things:
    1. Defining the computation logic for its energy (energy, gradient, etc.) via *_func methods that operate on a single constraint.
    2. Adding its specific constraint data to a global container via add_constraints.
    It does not store any simulation data itself.
    """

    @classmethod
    @abc.abstractmethod
    def get_type_id(cls) -> int:
        """Returns the unique integer ID for this energy type."""
        pass

    def add_one_constraint_func(self):
        """
        Adds one constraint to the global energy container at a given index.
        This should be implemented by subclasses as a @ti.func.
        The signature is specific to each energy type and must be known by the caller.
        Example for a distance constraint:
        @ti.func
        def add_one_constraint_func(self, constraint_idx: int, p1_idx: int, p2_idx: int, rest_dist: ti.f32, stiffness: ti.f32):
            ...
        """
        pass

    # The following methods are Taichi functions (@ti.func) that will be called
    # from within a Taichi kernel in the global container, dispatched via ti.static.
    # They operate on a single constraint. A concrete class must implement the ones it needs.

    def compute_energy_func(self, constraint: ti.template(), data: ISimulationData) -> ti.f32:
        """
        Computes the energy for a single constraint.
        This should be implemented as a @ti.func.
        """
        return 0.0

    def compute_gradient_func(self, constraint: ti.template(), data: ISimulationData, out_grad: ti.template()):
        """
        Computes the gradient for a single constraint and adds it to the output.
        This should be implemented as a @ti.func.
        """
        pass

    def compute_hessian_func(self, constraint: ti.template(), data: ISimulationData, out_hessian_builder: Any):
        """
        Computes the Hessian for a single constraint and adds it to the output.
        This should be implemented as a @ti.func.
        """
        pass

    def compute_constraint_gradient_func(self, constraint: ti.template(), q: ti.template()):
        """
        Computes the constraint value C and gradient nabla_C for a single constraint.
        Used for constraint-based solvers.
        Should return a tuple containing:
        1. C (ti.f32): The scalar constraint value.
        2. grads (ti.Matrix): A matrix of gradients for its vertices (e.g., 4x3).
        3. num_vertices (int): The number of active vertices for this constraint.
        """
        pass

    def gather_projection_func(self, constraint: ti.template(), data: ISimulationData, **kwargs):
        """
        Performs local-global gather step for a single constraint for Projective Dynamics.
        This should be implemented as a @ti.func.
        """
        pass


@ti.data_oriented
class IGlobalEnergyContainer(abc.ABC):
    """
    Abstract base class for a global container that manages and computes all potential energy terms.
    This interface defines the methods that a solver might require. It operates on the entire set of constraints.
    """

    @abc.abstractmethod
    def compute_energy(self, data: ISimulationData) -> ti.f32:
        """(Optional) Computes the total energy E(q) from all constraints."""
        pass

    @abc.abstractmethod
    def compute_gradient(self, data: ISimulationData, out_grad: ti.Field):
        """(Optional) Computes the total gradient ∇E(q) from all constraints."""
        pass

    @abc.abstractmethod
    def compute_hessian(self, data: ISimulationData, out_hessian_builder: Any):
        """(Optional) Computes the total Hessian ∇²E(q) from all constraints."""
        pass

    @abc.abstractmethod
    def get_num_constraints(self) -> int:
        """(Optional) Returns the total number of active constraints in the container."""
        pass

    @abc.abstractmethod
    def compute_one_constraint_gradient_func(self, i: int, q: ti.template()):
        """
        (Optional, @ti.func) Computes the constraint value C and gradient nabla_C for a single constraint at index i.
        This is intended to be called from within another Taichi kernel (typically from a solver).
        Should return a tuple: (C, grads_matrix, num_vertices).
        """
        pass


    @abc.abstractmethod
    def gather_projection(self, data: ISimulationData, **kwargs):
        """(Optional) Performs local-global gather step for Projective Dynamics for all constraints."""
        pass

    @abc.abstractmethod
    def register_energy(self, energy: IPotentialEnergy):
        """Registers a potential energy instance, making its compute functions available in kernels."""
        pass

    @abc.abstractmethod
    def reserve_constraints(self, num_to_add: int, is_static: bool = True) -> int:
        """Reserves a block of constraints and returns the starting index."""
        pass

    @abc.abstractmethod
    def clear_constraints(self):
        """Clears all constraints, including static ones."""
        pass

    @abc.abstractmethod
    def clear_dynamic_constraints(self):
        """Clears only the dynamic (temporary) constraints, keeping the static ones."""
        pass


