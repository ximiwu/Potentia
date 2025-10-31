import abc
from enum import Enum
from typing import Optional

from renderers.base import IRenderer


class RecordingMode(Enum):
    DISABLED = 0
    RUNNING_ONLY = 1
    ALWAYS = 2


class IRecorder(abc.ABC):
    @abc.abstractmethod
    def set_mode(self, mode: RecordingMode) -> None:
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


