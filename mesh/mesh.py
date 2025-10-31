import abc
from typing import Any, Dict, Tuple

import numpy as np
import taichi as ti

from .base import IMesh





class Mesh(IMesh, abc.ABC):
    def __init__(self, vertices: np.ndarray, materialize: bool = False):
        # Prefer capability checks over type checks; accept array-like inputs
        if not (hasattr(vertices, 'ndim') and hasattr(vertices, 'shape') and getattr(vertices, 'ndim', -1) == 2 and getattr(vertices, 'shape', (0, 0))[1] == 3):
            try:
                vertices = np.asarray(vertices, dtype=np.float32)
            except Exception as e:
                raise ValueError("Vertices must be an array-like of shape (n, 3)") from e
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("Vertices must be an array-like of shape (n, 3)")
        # Ensure dtype is float32 and materialize to numpy
        if getattr(vertices, 'dtype', None) != np.float32:
            vertices = np.asarray(vertices, dtype=np.float32)

        self._rest_positions = ti.Vector.field(3, dtype=ti.f32, shape=vertices.shape[0])
        self._rest_positions_host = vertices

        if materialize:
            self.materialize()

    def materialize(self):
        self._rest_positions.from_numpy(self._rest_positions_host)

    def get_rest_positions(self) -> ti.Field:
        return self._rest_positions

    def get_num_vertices(self) -> int:
        return self._rest_positions.shape[0]

    # Transformation helpers were removed from this class.
