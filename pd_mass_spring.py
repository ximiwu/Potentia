import abc
from typing import List, Dict, Any
import json

import taichi as ti
import taichi.math as tm
from taichi.ui import camera

from collision.base import ICollisionHandler
from data.base import ISimulationData
from energies.base import IGlobalEnergyContainer
from data.mass_point_data import MassPointData
from energies.pd_spring_energy import PDSpringEnergy
from forces.gravity_force import GravityForce
from integrators.implicit_euler_integrator import ImplicitEulerIntegrator
from mesh.trimesh import TriMesh
from objects.mesh_object import MeshObject
from renderers.mesh_renderer import MeshRenderer
from solvers.pd_solver import PDSolver
from world.simulation_world import SimulationWorld
from controller.fp_input_handler import FPInputHandler
from recorders.frame_recorder import FrameRecorder, RecordingMode
from actuators.pingpong_move_actuator import PingPongMoveActuator
from forces.constant_force import ConstantForce


# 定义一个简单的碰撞处理器，在这个示例中它什么也不做
class DummyCollisionHandler(ICollisionHandler):
    def detect_and_create_potentials(self, data: ISimulationData, q_predict: ti.Field) -> List:
        return []



ti.init(arch=ti.cuda)

# 1. 组装仿真世界的各个模块
# 为网格对象增加自由度上限
sim_data = MassPointData(max_point_num=20000)

energies_to_register = [PDSpringEnergy.get_instance()]
# 关键：PD 需要把 data 传入求解器构造函数
solver = PDSolver(data=sim_data, iterations=1000)
integrator = ImplicitEulerIntegrator()
# renderer = MeshRenderer(title="PD Mesh Simulation", fps_limit=1000, camera_lookat=(0.846, 0.216, -0.583), camera_pos=(1.651, 0.767, -1.879))
# collision_handler = DummyCollisionHandler()

# input_handler = FPInputHandler()
# input_handler.set_paused_state(True)

# recorder = FrameRecorder(output_dir="captures/pd", mode=RecordingMode.RUNNING_ONLY, make_video=True, fps=30)

# world = SimulationWorld(data=sim_data, 
#                             solver=solver, 
#                             integrator=integrator, 
#                             collision_handler=collision_handler, 
#                             renderer=renderer,
#                             energies=energies_to_register,
#                             input_handler=input_handler,
#                             recorder=recorder)

# # 2. 添加外力
# gravity = GravityForce(gravity=tm.vec3(0.0, -9.8, 0.0))
# world.add_force(gravity)

# # 3. 创建并添加仿真对象



# cloth_mesh = TriMesh.from_obj("models/cloth81x81.obj", rotation=(0, 0, 0))

# cloth_mesh.materialize()

# obj1 = MeshObject(mass=1 / 441.0, mesh=cloth_mesh, data=sim_data, translation=(0.0, 0.0, 0.0), face_color=(0.0, 179/256.0, 209/256.0))
# obj1.add_pd_spring_energy(1200)


# world.add_object(obj1)


# act0 = PingPongMoveActuator(
#     obj=obj1,
#     local_vertex_indices=[6560],
#     pos1=(0, 0.5, 0.5),
#     pos2=(0.666, 0.5, 0.5),
#     move_duration=1.166,
#     wait_duration=0.1,
# )
# world.add_actuator(act0)

# act1 = PingPongMoveActuator(
#     obj=obj1,
#     local_vertex_indices=[80],
#     pos1=(0, 0.5, -0.5),
#     pos2=(0.666, 0.5, -0.5),
#     move_duration=1.166,
#     wait_duration=0.1,
# )
# world.add_actuator(act1)



renderer = MeshRenderer(title="PD Mesh Simulation", fps_limit=1000, camera_lookat=(0.5, -0.3, 0), camera_pos=(4, -0.3, 0))

collision_handler = DummyCollisionHandler()

input_handler = FPInputHandler()
input_handler.set_paused_state(True)

recorder = FrameRecorder(output_dir="captures/pd", mode=RecordingMode.RUNNING_ONLY, make_video=True, fps=30)

world = SimulationWorld(data=sim_data, 
                            solver=solver, 
                            integrator=integrator, 
                            collision_handler=collision_handler, 
                            renderer=renderer,
                            energies=energies_to_register,
                            input_handler=input_handler,
                            recorder=recorder)

# 2. 添加外力
# gravity = GravityForce(gravity=tm.vec3(0.0, -9.8, 0.0))
# world.add_force(gravity)

# 3. 创建并添加仿真对象


cloth_mesh = TriMesh.from_obj("models/cloth10x10.obj", rotation=(0, 0, 0))

cloth_mesh.materialize()

obj1 = MeshObject(mass=1 / 441.0, mesh=cloth_mesh, data=sim_data, translation=(0.0, 0.0, 0.0), face_color=(0.0, 179/256.0, 209/256.0))
obj1.add_pd_spring_energy(1200)


world.add_object(obj1)

const_fr = ConstantForce(obj=obj1, local_indices=[9,8,7,6,5,4,3,2,1,0], force=ti.math.vec3(0.0, 0.0, -10.0))
const_fl = ConstantForce(obj=obj1, local_indices=[99,98,97,96,95,94,93,92,91,90], force=ti.math.vec3(0.0, 0.0, 10.0))
world.add_force(const_fr)
world.add_force(const_fl)

# sim_data.get_dofs()[0] = ti.Vector([0, -1, -1], dt=ti.f32)
# sim_data.get_dofs()[1] = ti.Vector([0, 1, -1], dt=ti.f32)
# sim_data.get_dofs()[2] = ti.Vector([0, -1, 1], dt=ti.f32)
# sim_data.get_dofs()[3] = ti.Vector([0, 1, 1], dt=ti.f32)




# 关键：PD 需要在进入主循环前构建并分解一次 LHS（若 DoF/约束 或 dt 改变需重新调用）
dt = 1.0 / 30.0
solver.build_lhs(sim_data, dt)



# 4. 运行仿真主循环（录制由 UI Start/Stop 控制，不在此处自动 start/stop）
frame = 0
while renderer.is_window_running():
    world.step(dt=dt)
    frame += 1
