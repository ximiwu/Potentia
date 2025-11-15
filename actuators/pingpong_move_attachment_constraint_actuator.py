from typing import Tuple

import taichi as ti

from data.base import ISimulationData
from objects.base import IMeshObject
from energies.global_energy_container import GlobalEnergyContainer
from .base import IVertexActuator


@ti.data_oriented
class PingPongMoveAttachmentConstraintActuator(IVertexActuator):
    """
    将单个局部顶点的附件约束目标点在 pos1 与 pos2 之间按时间往返移动：
    - 前进：用 move_duration 秒从 pos1 → pos2
    - 等待：在 pos2 停 wait_duration 秒
    - 返回：用 move_duration 秒从 pos2 → pos1
    - 等待：在 pos1 停 wait_duration 秒
    - 如此循环往复

    本类会在首次构造时调用 `obj.add_attachment_energy(stiffness, idx)` 添加固定点约束，
    并记录返回的约束索引；在 `apply` 中通过写 GlobalEnergyContainer 中该约束的 `params[1..3]`
    来更新固定点位置。
    """

    def __init__(
        self,
        obj: IMeshObject,
        idx: int,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
        move_duration: float,
        wait_duration: float,
        stiffness: float,
        start_in_forward: bool = True,
    ):
        required_obj_methods = ['add_attachment_energy']
        for method_name in required_obj_methods:
            if not hasattr(obj, method_name):
                raise TypeError(f"obj 必须实现 IMeshObject 能力 '{method_name}'。")

        if move_duration <= 0.0:
            raise ValueError('move_duration 必须为正数')
        if wait_duration < 0.0:
            raise ValueError('wait_duration 不能为负数')

        self._obj = obj
        self._move_duration = float(move_duration)
        self._wait_duration = float(wait_duration)

        self._pos1 = ti.Vector([float(pos1[0]), float(pos1[1]), float(pos1[2])])
        self._pos2 = ti.Vector([float(pos2[0]), float(pos2[1]), float(pos2[2])])

        self._container = GlobalEnergyContainer.get_instance()
        self._constraint_idx = int(self._obj.add_attachment_energy(float(stiffness), int(idx)))

        self._state = 0 if start_in_forward else 2
        self._t_in_state = 0.0

    def apply(self, data: ISimulationData, dt: float) -> None:
        steps_to_skip = int(getattr(self, "_init_frame", -1))
        if not getattr(self, "_init_skip_done", False):
            if steps_to_skip > 0:
                for _ in range(steps_to_skip):
                    self._advance_state_only(float(dt))
            self._init_skip_done = True

        t = self._t_in_state + float(dt)
        state = self._state

        if state == 0:  # move pos1 -> pos2
            alpha = t / self._move_duration
            if alpha >= 1.0:
                alpha = 1.0
                state = 1
                t = t - self._move_duration
            target = (1.0 - alpha) * self._pos1 + alpha * self._pos2
            self._write_attachment_pos_kernel(self._container, self._constraint_idx, target)

        elif state == 1:  # wait at pos2
            alpha = 1.0
            if t >= self._wait_duration:
                state = 2
                t = t - self._wait_duration
            target = (1.0 - alpha) * self._pos1 + alpha * self._pos2
            self._write_attachment_pos_kernel(self._container, self._constraint_idx, target)

        elif state == 2:  # move pos2 -> pos1
            alpha = t / self._move_duration
            if alpha >= 1.0:
                alpha = 1.0
                state = 3
                t = t - self._move_duration
            target = (1.0 - alpha) * self._pos2 + alpha * self._pos1
            self._write_attachment_pos_kernel(self._container, self._constraint_idx, target)

        else:  # state == 3, wait at pos1
            alpha = 1.0
            if t >= self._wait_duration:
                state = 0
                t = t - self._wait_duration
            target = (1.0 - alpha) * self._pos2 + alpha * self._pos1
            self._write_attachment_pos_kernel(self._container, self._constraint_idx, target)

        self._state = state
        self._t_in_state = t

    def set_init_frame(self, init_frame: int) -> None:
        self._init_frame = int(init_frame)
        self._init_skip_done = False

    def _advance_state_only(self, dt: float) -> None:
        t = self._t_in_state + float(dt)
        state = self._state

        if state == 0:
            alpha = t / self._move_duration
            if alpha >= 1.0:
                state = 1
                t = t - self._move_duration

        elif state == 1:
            if t >= self._wait_duration:
                state = 2
                t = t - self._wait_duration

        elif state == 2:
            alpha = t / self._move_duration
            if alpha >= 1.0:
                state = 3
                t = t - self._move_duration

        else:
            if t >= self._wait_duration:
                state = 0
                t = t - self._wait_duration

        self._state = state
        self._t_in_state = t

    @ti.kernel
    def _write_attachment_pos_kernel(
        self,
        container: ti.template(),
        constraint_idx: ti.i32,
        target: ti.types.vector(3, ti.f32),
    ):
        container.constraints[constraint_idx].params[1] = target[0]
        container.constraints[constraint_idx].params[2] = target[1]
        container.constraints[constraint_idx].params[3] = target[2]