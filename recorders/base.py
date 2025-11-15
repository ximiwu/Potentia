import abc
from enum import Enum
from typing import Optional, Callable

from renderers.base import IRenderer
from data.base import ISimulationData


class RecordingMode(Enum):
    DISABLED = 0
    RUNNING_ONLY = 1
    ALWAYS = 2


class IRecorder(abc.ABC):
    @abc.abstractmethod
    def set_mode(self, mode: RecordingMode) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def set_init_frame(self, init_frame: int) -> None:
        """设置起始帧编号（默认 -1，首帧为 init_frame + 1）。"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_mode(self) -> RecordingMode:
        raise NotImplementedError

    @abc.abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_frame_end(self, renderer: IRenderer, is_paused: bool) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_solve_end(self, data: ISimulationData, dt: Optional[float] = None) -> None:
        """在一次求解结束后回调，用于持久化状态或统计数据。"""
        raise NotImplementedError

    @abc.abstractmethod
    def on_predict_end(self, data: ISimulationData, dt: float) -> None:
        """在预测阶段结束后回调，用于持久化 predicted_dofs 等。"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_iteration_callback(self) -> Optional[Callable[[int, ISimulationData, float], None]]:
        """返回 Solver 迭代阶段的回调函数，签名为 (iter_idx, data, dt)。若不需要记录则返回 None。"""
        raise NotImplementedError


