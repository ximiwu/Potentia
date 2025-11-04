from typing import List, Optional

import taichi as ti

from collision.base import ICollisionHandler
from data.base import ISimulationData
from energies.base import IPotentialEnergy
from energies.global_energy_container import GlobalEnergyContainer
from forces.base import IForce
from integrators.base import IIntegrator
from objects.base import ISimulationObject
from solvers.base import ISolver
from .base import ISimulationWorld
from controller.base import IInputHandler
from renderers.base import IRenderer
from recorders.base import IRecorder
from actuators.base import IVertexActuator


class SimulationWorld(ISimulationWorld):
    def __init__(
            self,
            data: ISimulationData,
            solver: ISolver,
            integrator: IIntegrator,
            collision_handler: ICollisionHandler,
            renderer: IRenderer,
            energies: List[IPotentialEnergy],
            input_handler: Optional[IInputHandler] = None,
            recorder: Optional[IRecorder] = None,
            actuators: Optional[List[IVertexActuator]] = None,
    ):
        self.data = data
        self.solver = solver
        self.integrator = integrator
        self.collision_handler = collision_handler
        self.renderer = renderer
        self.energies = energies
        self.input_handler = input_handler
        self.recorder = recorder
        self.objects: List[ISimulationObject] = []
        self.forces: List[IForce] = []
        self.actuators: List[IVertexActuator] = []
        

        self.energy_container = GlobalEnergyContainer.get_instance()
        for energy in self.energies:
            self.energy_container.register_energy(energy)

        if actuators is not None:
            for a in actuators:
                self.actuators.append(a)

    def add_object(self, obj: ISimulationObject):
        self.objects.append(obj)

    def add_force(self, force: IForce):
        self.forces.append(force)

    def add_actuator(self, actuator: IVertexActuator) -> None:
        self.actuators.append(actuator)

    def get_recorder(self) -> Optional[IRecorder]:
        return self.recorder

    def step(self, dt: float):

        if self.input_handler is not None:
            self.input_handler.handle_inputs(self, self.renderer, dt)

        if self.input_handler is not None and self.input_handler.is_paused():
            # 暂停时仅绘制 UI 与渲染，不做任何物理与约束更新
            self.input_handler.draw_ui(self, self.renderer)
            self.renderer.render(self.data, self.objects)
            if self.recorder is not None:
                self.recorder.on_frame_end(self.renderer, True)
            self.renderer.present()
            return

        self.energy_container.clear_dynamic_constraints()

        # Collision detection will be added here in the future.

        self.integrator.predict(self.data, self.forces, dt)

        # 在 predict 与 solve 之间应用所有 Actuator（写入 predicted_dofs）
        for actuator in self.actuators:
            actuator.apply(self.data, dt)

        self.data.record_predicted_dofs()

        # 可选能力：在 predict+actuator 后、求解前保存 predicted_dofs 及其 loss
        if self.recorder is not None and hasattr(self.recorder, "on_predict_end"):
            try:
                self.recorder.on_predict_end(self.data, dt)
            except Exception as e:
                print(f"[World] 调用 recorder.on_predict_end 失败: {e}")

        self.solver.solve(self.data, dt)

        self.integrator.update_state(self.data, dt)
        
        # 可选能力：在求解完成后、更新状态之前持久化 DoF/速度 与 loss
        if self.recorder is not None and hasattr(self.recorder, "on_solve_end"):
            try:
                self.recorder.on_solve_end(self.data, dt)
            except Exception as e:
                print(f"[World] 调用 recorder.on_solve_end 失败: {e}")

        print(GlobalEnergyContainer.get_instance().compute_loss(data=self.data, x=self.data.get_dofs(), y=self.data.get_record_dofs(), dt=dt))


        if self.input_handler is not None:
            # 物理更新完成后、渲染前绘制 UI
            self.input_handler.draw_ui(self, self.renderer)

        self.renderer.render(self.data, self.objects)
        if self.recorder is not None:
            self.recorder.on_frame_end(self.renderer, False)
        self.renderer.present()
