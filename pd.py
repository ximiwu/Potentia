import abc
from typing import List

import taichi as ti
import taichi.math as tm

from collision.base import ICollisionHandler
from data.base import ISimulationData
from data.mass_point_data import MassPointData
from energies.pd_bending_energy import PDBendingEnergy
from energies.pd_strain_energy import PDStrainEnergy
from forces.gravity_force import GravityForce
from integrators.implicit_euler_integrator import ImplicitEulerIntegrator
from mesh.trimesh import TriMesh
from objects.mesh_object import MeshObject
from renderers.mesh_renderer import MeshRenderer
from solvers.pd_solver import PDSolver
from world.simulation_world import SimulationWorld
from controller.fp_input_handler import FPInputHandler

# 定义一个简单的碰撞处理器，在这个示例中它什么也不做
class DummyCollisionHandler(ICollisionHandler):
    def detect_and_create_potentials(self, data: ISimulationData, q_predict: ti.Field) -> List:
        return []

def main():
    ti.init(arch=ti.cuda)

    # 1. 组装仿真世界的各个模块
    # 为网格对象增加自由度上限
    sim_data = MassPointData(max_point_num=10000)

    energies_to_register = [PDBendingEnergy.get_instance(), PDStrainEnergy.get_instance()]
    # 关键：PD 需要把 data 传入求解器构造函数
    solver = PDSolver(data=sim_data, iterations=5)
    integrator = ImplicitEulerIntegrator()
    renderer = MeshRenderer(title="PD Mesh Simulation")
    collision_handler = DummyCollisionHandler()

    input_handler = FPInputHandler()

    world = SimulationWorld(data=sim_data, 
                                solver=solver, 
                                integrator=integrator, 
                                collision_handler=collision_handler, 
                                renderer=renderer,
                                energies=energies_to_register,
                                input_handler=input_handler)

    # 2. 添加外力
    # gravity = GravityForce(gravity=tm.vec3(0.0, -9.8, 0.0))
    # world.add_force(gravity)

    # 3. 创建并添加仿真对象
    cube_mesh = TriMesh.create_cube()
    sphere_mesh = TriMesh.create_sphere(radius=0.2, subdivisions=2)
    cloth_mesh = TriMesh.from_obj("models/plane_30x30.obj")

    cube_mesh.materialize()
    sphere_mesh.materialize()
    cloth_mesh.materialize()

    cube_obj1 = MeshObject(mesh=cube_mesh, data=sim_data, translation=(1.0, 1.0, 1.0), face_color=(0.9, 0.3, 0.3), rotation=(30, 15, 60))
    # cube_obj1.add_pd_bending_energy(1e2)
    cube_obj1.add_pd_strain_energy(1e2, 1.0, 1.0)
    # cube_obj1.set_mass(0, -1.0)
    world.add_object(cube_obj1)

    # 关键：PD 需要在进入主循环前构建并分解一次 LHS（若 DoF/约束 或 dt 改变需重新调用）
    dt = 1.0 / 60.0
    solver.build_lhs(sim_data, dt)

    # 4. 运行仿真主循环
    frame = 0
    while renderer.is_window_running():
        world.step(dt=dt)
        frame += 1

if __name__ == "__main__":
    main()