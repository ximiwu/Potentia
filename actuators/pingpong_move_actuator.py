from typing import List, Tuple, Union

import taichi as ti

from data.base import ISimulationData
from objects.base import IMeshObject
from .base import IVertexActuator


@ti.data_oriented
class PingPongMoveActuator(IVertexActuator):
    """
    将一组局部顶点在 pos1 与 pos2 之间按时间往返移动：
    - 前进：用 move_duration 秒从 pos1 → pos2
    - 等待：在 pos2 停 wait_duration 秒
    - 返回：用 move_duration 秒从 pos2 → pos1
    - 等待：在 pos1 停 wait_duration 秒
    - 如此循环往复

    本类会在首次构造时自动将这些顶点 pinned（set_mass(local_index, -1.0)）。
    """

    def __init__(
        self,
        obj: IMeshObject,
        local_vertex_indices: List[int],
        pos1: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
        pos2: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
        move_duration: float,
        wait_duration: float,
        start_in_forward: bool = True,
    ):
        # 能力检查（接口隔离：仅检查方法存在）
        required_obj_methods = [
            'get_data_offset', 'get_mesh', 'get_num_dofs', 'set_mass'
        ]
        for method_name in required_obj_methods:
            if not hasattr(obj, method_name):
                raise TypeError(f"obj 必须实现 IMeshObject 能力 '{method_name}'。")

        if move_duration <= 0.0:
            raise ValueError('move_duration 必须为正数')
        if wait_duration < 0.0:
            raise ValueError('wait_duration 不能为负数')
        if len(local_vertex_indices) == 0:
            raise ValueError('local_vertex_indices 不能为空')

        self._obj = obj
        self._num_vertices = int(len(local_vertex_indices))
        self._move_duration = float(move_duration)
        self._wait_duration = float(wait_duration)

        # 计算全局索引
        data_offset = int(self._obj.get_data_offset())
        global_indices_py: List[int] = [data_offset + int(li) for li in local_vertex_indices]

        # Taichi fields
        self._global_indices = ti.field(dtype=ti.i32, shape=self._num_vertices)
        self._pos1 = ti.Vector.field(3, dtype=ti.f32, shape=self._num_vertices)
        self._pos2 = ti.Vector.field(3, dtype=ti.f32, shape=self._num_vertices)

        # 写入索引
        for i, g in enumerate(global_indices_py):
            self._global_indices[i] = g

        # 广播/写入 pos1/pos2
        def expand_positions(
            base: Union[Tuple[float, float, float], List[Tuple[float, float, float]]]
        ) -> List[Tuple[float, float, float]]:
            if isinstance(base, tuple):
                return [base for _ in range(self._num_vertices)]
            return list(base)

        pos1_list = expand_positions(pos1)
        pos2_list = expand_positions(pos2)
        if len(pos1_list) != self._num_vertices or len(pos2_list) != self._num_vertices:
            raise ValueError('pos1/pos2 的长度必须与 local_vertex_indices 相同，或为单个三元组用于广播')
        for i in range(self._num_vertices):
            p1 = pos1_list[i]
            p2 = pos2_list[i]
            self._pos1[i] = ti.Vector([float(p1[0]), float(p1[1]), float(p1[2])])
            self._pos2[i] = ti.Vector([float(p2[0]), float(p2[1]), float(p2[2])])

        # 自动 pinned
        for li in local_vertex_indices:
            self._obj.set_mass(int(li), -1.0)

        # 状态机：0=move_forward, 1=wait_at_pos2, 2=move_backward, 3=wait_at_pos1
        self._state = 0 if start_in_forward else 2
        self._t_in_state = 0.0

    def apply(self, data: ISimulationData, dt: float) -> None:
        # 一次性快进：将内部状态先推进 (_init_frame + 1) 个 dt（不写 predicted_dofs）
        steps_to_skip = int(getattr(self, "_init_frame", -1))
        if not getattr(self, "_init_skip_done", False):
            if steps_to_skip > 0:
                for _ in range(steps_to_skip):
                    self._advance_state_only(float(dt))
            self._init_skip_done = True

        # 推进状态机
        t = self._t_in_state + float(dt)
        state = self._state

        if state == 0:  # move pos1 -> pos2
            alpha = t / self._move_duration
            if alpha >= 1.0:
                alpha = 1.0
                state = 1
                t = t - self._move_duration
            self._write_targets_kernel(self._global_indices, self._pos1, self._pos2, data.get_predicted_dofs(), float(alpha), 1)

        elif state == 1:  # wait at pos2
            alpha = 1.0
            if t >= self._wait_duration:
                state = 2
                t = t - self._wait_duration
            self._write_targets_kernel(self._global_indices, self._pos1, self._pos2, data.get_predicted_dofs(), float(alpha), 1)

        elif state == 2:  # move pos2 -> pos1
            alpha = t / self._move_duration
            if alpha >= 1.0:
                alpha = 1.0
                state = 3
                t = t - self._move_duration
            self._write_targets_kernel(self._global_indices, self._pos1, self._pos2, data.get_predicted_dofs(), float(alpha), 0)

        else:  # state == 3, wait at pos1
            alpha = 1.0
            if t >= self._wait_duration:
                state = 0
                t = t - self._wait_duration
            self._write_targets_kernel(self._global_indices, self._pos1, self._pos2, data.get_predicted_dofs(), float(alpha), 0)

        self._state = state
        self._t_in_state = t

    # 帧编号设置（接口统一）：当前不影响运动，仅存储，供将来按帧驱动时使用
    def set_init_frame(self, init_frame: int) -> None:
        self._init_frame = int(init_frame)
        # 标记未执行过快进，等待首次 apply 时进行
        self._init_skip_done = False

    def _advance_state_only(self, dt: float) -> None:
        """仅推进内部状态机，不写 predicted_dofs。"""
        t = self._t_in_state + float(dt)
        state = self._state

        if state == 0:  # move pos1 -> pos2
            alpha = t / self._move_duration
            if alpha >= 1.0:
                state = 1
                t = t - self._move_duration

        elif state == 1:  # wait at pos2
            if t >= self._wait_duration:
                state = 2
                t = t - self._wait_duration

        elif state == 2:  # move pos2 -> pos1
            alpha = t / self._move_duration
            if alpha >= 1.0:
                state = 3
                t = t - self._move_duration

        else:  # state == 3, wait at pos1
            if t >= self._wait_duration:
                state = 0
                t = t - self._wait_duration

        self._state = state
        self._t_in_state = t

    @ti.kernel
    def _write_targets_kernel(
        self,
        global_indices: ti.template(),
        pos1: ti.template(),
        pos2: ti.template(),
        predicted: ti.template(),
        alpha: ti.f32,
        forward: ti.i32,
    ):
        for i in range(global_indices.shape[0]):
            g = global_indices[i]
            if forward == 1:
                target = (1.0 - alpha) * pos1[i] + alpha * pos2[i]
                predicted[g] = target
            else:
                target = (1.0 - alpha) * pos2[i] + alpha * pos1[i]
                predicted[g] = target


