from typing import List, Optional

import taichi as ti

from collision.base import ICollisionHandler
from data.base import ISimulationData
from energies.base import IPotentialEnergy
from energies.global_energy_container import GlobalEnergyContainer
from forces.base import IForce
from integrators.base import IIntegrator
from objects.base import ISimulationObject
from renderers.base import IRenderer
from solvers.base import ISolver
from world.base import ISimulationWorld
from controller.base import IInputHandler


class XPBDSimulationWorld(ISimulationWorld):
    def __init__(
            self,
            data: ISimulationData,
            solver: ISolver,
            integrator: IIntegrator,
            collision_handler: ICollisionHandler,
            renderer: IRenderer,
            energies: List[IPotentialEnergy],
            input_handler: Optional[IInputHandler] = None,
    ):
        self.data = data
        self.solver = solver
        self.integrator = integrator
        self.collision_handler = collision_handler
        self.renderer = renderer
        self.energies = energies
        self.input_handler = input_handler
        self.objects: List[ISimulationObject] = []
        self.forces: List[IForce] = []
        

        self.energy_container = GlobalEnergyContainer.get_instance()
        for energy in self.energies:
            self.energy_container.register_energy(energy)

    def add_object(self, obj: ISimulationObject):
        self.objects.append(obj)

    def add_force(self, force: IForce):
        self.forces.append(force)

    def step(self, dt: float):

        if self.input_handler is not None:
            self.input_handler.handle_inputs(self, self.renderer, dt)

        if self.input_handler is not None and self.input_handler.is_paused():
            # 暂停时仅绘制 UI 与渲染，不做任何物理与约束更新
            self.input_handler.draw_ui(self, self.renderer)
            self.renderer.render(self.data, self.objects)
            return

        self.energy_container.clear_dynamic_constraints()

        # Collision detection will be added here in the future.

        self.integrator.predict(self.data, self.forces, dt)
        self.solver.solve(self.data, dt)
        self.integrator.update_state(self.data, dt)

        if self.input_handler is not None:
            # 物理更新完成后、渲染前绘制 UI
            self.input_handler.draw_ui(self, self.renderer)

        self.renderer.render(self.data, self.objects)
