import abc
from typing import List, Dict, Any
import json

import taichi as ti
import taichi.math as tm
from taichi.ui import camera

from collision.base import ICollisionHandler
from data.base import ISimulationData
from energies import global_energy_container
from energies.global_energy_container import GlobalEnergyContainer
from data.mass_point_data import MassPointData
from energies.pd_spring_energy import PDSpringEnergy
from energies.pd_strain_energy import PDStrainEnergy
from energies.pd_bending_energy import PDBendingEnergy  
from energies.attachment_energy import AttachmentEnergy
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
from actuators.pingpong_move_attachment_constraint_actuator import PingPongMoveAttachmentConstraintActuator
from forces.constant_force import ConstantForce


# 定义一个简单的碰撞处理器，在这个示例中它什么也不做
class DummyCollisionHandler(ICollisionHandler):
    def detect_and_create_potentials(self, data: ISimulationData, q_predict: ti.Field) -> List:
        return []



ti.init(arch=ti.cuda)

# 1. 组装仿真世界的各个模块
# 为网格对象增加自由度上限
sim_data = MassPointData(max_point_num=20000)

energies_to_register = [PDSpringEnergy.get_instance(), AttachmentEnergy.get_instance()]
# 关键：PD 需要把 data 传入求解器构造函数
solver = PDSolver(data=sim_data, iterations=1000)
integrator = ImplicitEulerIntegrator()


#侧视
renderer = MeshRenderer(title="PD Mesh Simulation", fps_limit=1000, camera_lookat=(0.846, 0.216, -0.583), camera_pos=(1.651, 0.767, -1.879))

#正视
# renderer = MeshRenderer(title="PD Mesh Simulation", fps_limit=1000, camera_lookat=(0.5, -0.3, 0), camera_pos=(2, -0.3, 0))

#正视 - 2x
# renderer = MeshRenderer(title="PD Mesh Simulation", fps_limit=1000, camera_lookat=(0.5, -0.3, 0), camera_pos=(4, -0.3, 0))


collision_handler = DummyCollisionHandler()

input_handler = FPInputHandler()
input_handler.set_paused_state(True)

recorder = FrameRecorder(output_dir="captures/pd", mode=RecordingMode.RUNNING_ONLY, make_video=True, fps=30, iter_plot=False, solve_info=False)

world = SimulationWorld(data=sim_data, 
                            solver=solver, 
                            integrator=integrator, 
                            collision_handler=collision_handler, 
                            renderer=renderer,
                            energies=energies_to_register,
                            input_handler=input_handler,
                            recorder=recorder)

# 2. 添加外力
gravity = GravityForce(gravity=tm.vec3(0.0, -100, 0.0))
world.add_force(gravity)

# 3. 创建并添加仿真对象


cloth_mesh = TriMesh.from_obj("models/cloth21x21.obj", rotation=(0, 0, 0))

cloth_mesh.materialize()

obj1 = MeshObject(mass=1 / (21.0 * 21.0), mesh=cloth_mesh, data=sim_data, translation=(0.0, 0.0, 0.0), face_color=(0.0, 179/256.0, 209/256.0))
obj1.add_pd_spring_energy(80, single_diag=True)
# obj1.add_pd_strain_energy(120, 0.99, 1.01)
# obj1.add_pd_bending_energy(0.001)

print(GlobalEnergyContainer.get_instance().get_num_constraints())
# print(GlobalEnergyContainer.get_instance().get_constraints())


world.add_object(obj1)

# const_fr = ConstantForce(obj=obj1, local_indices=[9,8,7,6,5,4,3,2,1,0], force=ti.math.vec3(0.0, 0.0, -10.0))
# const_fl = ConstantForce(obj=obj1, local_indices=[99,98,97,96,95,94,93,92,91,90], force=ti.math.vec3(0.0, 0.0, 10.0))
# world.add_force(const_fr)
# world.add_force(const_fl)


# act0 = PingPongMoveActuator(
#     obj=obj1,
#     local_vertex_indices=[6560],
#     pos1=(0, 0.5, 0.49),
#     pos2=(0.666, 0.5, 0.49),
#     move_duration=1.166,
#     wait_duration=0.1,
# )
# world.add_actuator(act0)

# act1 = PingPongMoveActuator(
#     obj=obj1,
#     local_vertex_indices=[80],
#     pos1=(0, 0.5, -0.49),
#     pos2=(0.666, 0.5, -0.49),
#     move_duration=1.166,
#     wait_duration=0.1,
# )
# world.add_actuator(act1)

# act0 = PingPongMoveAttachmentConstraintActuator(
#     obj=obj1,
#     idx=6560,
#     pos1=(0, 0.5, 0.49),
#     pos2=(0.666, 0.5, 0.49),
#     move_duration=1.166,
#     wait_duration=0.1,
#     stiffness=240,

# )
# world.add_actuator(act0)

# act1 = PingPongMoveAttachmentConstraintActuator(
#     obj=obj1,
#     idx=80,
#     pos1=(0, 0.5, -0.49),
#     pos2=(0.666, 0.5, -0.49),
#     move_duration=1.166,
#     wait_duration=0.1,
#     stiffness=240,
# )
# world.add_actuator(act1)


act0 = PingPongMoveAttachmentConstraintActuator(
    obj=obj1,
    idx=440,
    pos1=(5, 5, 5),
    pos2=(5, 5, 5),
    move_duration=1,
    wait_duration=1000000000000,
    stiffness=120,

)
world.add_actuator(act0)

act1 = PingPongMoveAttachmentConstraintActuator(
    obj=obj1,
    idx=20,
    pos1=(5, 5, -5),
    pos2=(5, 5, -5),
    move_duration=1,
    wait_duration=1000000000000,
    stiffness=120,
)
world.add_actuator(act1)




# 将所有 DoF 的坐标统一缩放为 k 倍
# k = 2.0

# @ti.kernel
# def scale_all_dofs_kernel(dofs: ti.template(), dim: int, scale: ti.f32):
#     for i in range(dim):
#         dofs[i] *= scale

# scale_all_dofs_kernel(sim_data.get_dofs(), sim_data.get_num_dofs(), k)




# 关键：PD 需要在进入主循环前构建并分解一次 LHS（若 DoF/约束 或 dt 改变需重新调用）
dt = 1.0 / 30.0
solver.build_lhs(sim_data, dt)


# 可选：在第一次 world.step 前从录制的会话恢复（需要用户自行保证对象添加顺序与数量一致）
# def set_init_frame(idx: int):
#     sim_data.resume_from_record("captures/newton_pcg", idx)
#     recorder.set_init_frame(idx)
    
#     act0.set_init_frame(idx)
#     act1.set_init_frame(idx)

# set_init_frame(39)


# 4. 运行仿真主循环（录制由 UI Start/Stop 控制，不在此处自动 start/stop）
frame = 0
while renderer.is_window_running():
    world.step(dt=dt)
    frame += 1

# 在窗口关闭后自动停止录制并合成视频（若开启）
# try:
#     recorder.stop()
# except Exception:
#     pass
