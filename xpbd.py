import abc
from typing import List

import taichi as ti
import taichi.math as tm

from collision.base import ICollisionHandler
from data.base import ISimulationData
from data.mass_point_data import MassPointData
from energies.distance_energy import DistanceEnergy
from forces.gravity_force import GravityForce
from integrators.implicit_euler_integrator import ImplicitEulerIntegrator
from mesh.trimesh import TriMesh
from objects.mesh_object import MeshObject
from renderers.mesh_renderer import MeshRenderer
from solvers.xpbd_solver import XPBDSolver
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

    energies_to_register = [DistanceEnergy.get_instance()]
    solver = XPBDSolver(iterations=20)
    integrator = ImplicitEulerIntegrator()
    renderer = MeshRenderer(title="XPBD Mesh Simulation")
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
    gravity = GravityForce(gravity=tm.vec3(0.0, -9.8, 0.0))
    world.add_force(gravity)


    # 3. 创建并添加仿真对象（球体）
    
    cube_mesh = TriMesh.create_cube()
    sphere_mesh = TriMesh.create_sphere(radius=0.2, subdivisions=2)
    cloth_mesh = TriMesh.from_obj("models/plane_30x30.obj")

    cube_mesh.materialize()
    sphere_mesh.materialize()
    cloth_mesh.materialize()


    cube_obj1 = MeshObject(mesh=cube_mesh, data=sim_data, translation=(0.0, 2.0, 0.0), face_color=(0.9, 0.3, 0.3))
    cube_obj1.add_xpbd_distance_energy(stiffness=1e4)
    cube_obj1.set_mass(0, -1.0)
    world.add_object(cube_obj1)




    # sphere_obj2 = MeshObject(mesh=sphere_mesh, data=sim_data, translation=(0.5, 2.5, 0.0), face_color=(0.3, 0.9, 0.3))
    # sphere_obj2.add_xpbd_distance_energy(stiffness=1e6)
    # sphere_obj2.set_mass(0, -1.0)
    # world.add_object(sphere_obj2)

    # cloth_obj = MeshObject(mesh=cloth_mesh, data=sim_data, translation=(0, 2.5, 0.0), face_color=(0.3, 0.9, 0.3))
    # cloth_obj.add_xpbd_distance_energy(stiffness=1e6)
    # cloth_obj.set_mass(0, -1.0)
    # world.add_object(cloth_obj)


    # 4. 运行仿真主循环
    frame = 0
    while renderer.is_window_running():
        world.step(dt=1.0 / 60.0)
        frame += 1

if __name__ == "__main__":
    main()
