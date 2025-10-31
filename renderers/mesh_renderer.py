from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import taichi as ti

from data.base import ISimulationData
from mesh.base import IMesh, ISurfaceDataProvider
from objects.base import IMeshObject, ISimulationObject
from .base import IRenderer





@ti.data_oriented
class MeshRenderer(IRenderer):
    """
    A renderer that draws simulation objects based on their mesh topology.

    It checks for surface (face), edge, and vertex data and renders them
    accordingly. The visibility of each component is controlled by the
    color tuple provided by the simulation object.
    """
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "Energy Minimization Framework",
        background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        vertex_radius: float = 0.01,
        edge_width: float = 1.0
    ):
        self.window = ti.ui.Window(title, (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 1.5, 3)
        self.camera.lookat(0.0, 1.0, 0)
        self.camera.up(0.0, 1.0, 0)

        self._background_color = background_color
        self._vertex_radius = vertex_radius
        self._edge_width = edge_width

    def set_background_color(self, color: Tuple[float, float, float]):
        """Sets the background color of the scene."""
        self._background_color = color

    def set_vertex_radius(self, radius: float):
        """Sets the radius for rendering vertices."""
        self._vertex_radius = radius

    def set_edge_width(self, width: float):
        """Sets the width for rendering edges."""
        self._edge_width = width

    def render(
        self,
        data: 'ISimulationData',
        objects: List['ISimulationObject']
    ) -> None:
        # self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        positions = data.get_dofs()

        for obj in objects:
            if not hasattr(obj, 'get_mesh'):
                continue

            mesh = obj.get_mesh()
            offset = obj.get_data_offset()
            num_vertices = mesh.get_num_vertices()

            face_color, edge_color, vertex_color = obj.get_color()

            # Render faces if the mesh provides surface data and a face color is specified
            if face_color is not None and hasattr(mesh, 'get_surface_indices'):
                surface_indices = mesh.get_surface_indices()
                self.scene.mesh(
                    positions,
                    indices=surface_indices,
                    color=face_color,
                    vertex_offset=offset,
                    vertex_count=num_vertices,
                    two_sided=True
                )

            # Render edges if the mesh provides edge data and an edge color is specified
            if edge_color is not None and hasattr(mesh, 'get_edge_indices'):
                edge_indices = mesh.get_edge_indices()
                self.scene.lines(
                    positions,
                    indices=edge_indices,
                    color=edge_color,
                    width=self._edge_width,
                    vertex_offset=offset,
                    vertex_count=num_vertices
                )

            # Render vertices if a vertex color is specified
            if vertex_color is not None:
                self.scene.particles(
                    positions,
                    radius=self._vertex_radius,
                    color=vertex_color,
                    vertex_offset=offset,
                    vertex_count=num_vertices
                )
        
        self.canvas.set_background_color(self._background_color)
        self.canvas.scene(self.scene)
        self.window.show()

    def is_window_running(self) -> bool:
        return self.window.running
