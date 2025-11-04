import abc
from typing import List

import taichi as ti

from data.base import ISimulationData


class IVertexActuator(abc.ABC):
    """
    顶点操控器接口：用于在 predict→solve 之间对部分顶点进行人为驱动。

    约束：
    - 仅修改 `ISimulationData.get_predicted_dofs()`，不直接写 `get_dofs()` 或速度。
    - 不依赖具体 World/Renderer 实现。
    """

    @abc.abstractmethod
    def apply(self, data: ISimulationData, dt: float) -> None:
        """对本次帧进行应用（写入 predicted_dofs）。"""
        raise NotImplementedError


