import abc
from typing import Optional

import taichi as ti

from renderers.base import IRenderer
from world.base import ISimulationWorld


class IInputHandler(abc.ABC):
    """
    统一处理用户的键盘/鼠标输入与基础 UI 操作（遵循 @gui.mdc 的 API 约束）。

    约束：
    - 不直接进行场景绘制；不调用 window.show()。
    - 不依赖具体渲染器实现细节；仅依赖 IRenderer 抽象。
    - 暂停语义：当 is_paused() 为 True 时，world.step 仅渲染不更新物理。
    """

    @abc.abstractmethod
    def handle_inputs(self, world: ISimulationWorld, renderer: IRenderer, dt: float) -> None:
        """读取输入状态（键盘/鼠标）并据此修改高层状态，如切换暂停、触发重置等。"""
        pass

    @abc.abstractmethod
    def is_paused(self) -> bool:
        """返回当前是否处于“暂停”状态。"""
        pass

    def draw_ui(self, world: ISimulationWorld, renderer: IRenderer) -> None:
        """(可选) 构建即时 UI（例如重置/暂停按钮）。调用方需保证在 window.show() 之前被调用。"""
        return None


