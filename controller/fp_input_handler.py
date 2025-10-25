from typing import Optional
import math
import time

import taichi as ti

from controller.base import IInputHandler
from renderers.base import IRenderer
from world.base import ISimulationWorld


class FPInputHandler(IInputHandler):
    """
    第一人称(游戏风格)输入处理器：
    - WASD：平面移动相机（前后左右）
    - E/Q：升高/降低相机的 y 轴位置
    - 鼠标：由渲染器在 render(...) 中通过 camera.track_user_inputs 处理视角（保持现有逻辑）

    该实现不直接访问 window/show，仅通过 IRenderer 抽象协作；暂停语义遵循 IInputHandler 规范。
    """

    def __init__(self, move_speed: float = 1.0, yaw_speed: float = 2.0, pitch_speed: float = 2.0, hold_mouse_button: int = ti.ui.RMB):
        self._paused: bool = False
        self._move_speed: float = move_speed
        self._yaw_speed: float = yaw_speed
        self._pitch_speed: float = pitch_speed
        self._hold_mouse_button: int = hold_mouse_button
        self._last_mouse_x: Optional[float] = None
        self._last_mouse_y: Optional[float] = None
        self._last_time_ns: Optional[int] = None
        self._yaw: Optional[float] = None
        self._pitch: Optional[float] = None

        self._space_was_pressed: bool = False

    def handle_inputs(self, world: ISimulationWorld, renderer: IRenderer, dt: float) -> None:
        # 可用键位（字母键使用字符，遵循 GGUI/ImGui 键位规范）
        W = 'w'
        A = 'a'
        S = 's'
        D = 'd'
        E = 'e'
        Q = 'q'
        SPACE = ti.ui.SPACE

        # 尝试从渲染器获取 window 能力（不强求，若无则降级为无操作）
        window = getattr(renderer, 'window', None)
        camera = getattr(renderer, 'camera', None)
        if window is None or camera is None:
            return

        # 切换暂停（Space）- 使用按键状态的边沿检测，避免消费事件队列影响 GUI
        space_pressed_now = window.is_pressed(SPACE)
        if space_pressed_now and not self._space_was_pressed:
            self._paused = not self._paused
        self._space_was_pressed = space_pressed_now

        # 若暂停，仅允许 UI 和渲染，不进行移动
        if self._paused:
            return

        # 相机当前位置与朝向
        move = ti.Vector([0.0, 0.0, 0.0])
        speed = self._move_speed * dt

        # 基于窗口按键状态累积移动方向
        if window.is_pressed(W):  # forward (look direction projected on xz)
            move.z += 1.0
        if window.is_pressed(S):  # backward
            move.z -= 1.0
        if window.is_pressed(A):  # left
            move.x += 1.0
        if window.is_pressed(D):  # right
            move.x -= 1.0
        if window.is_pressed(E):  # up
            move.y += 1.0
        if window.is_pressed(Q):  # down
            move.y -= 1.0

        # 规范化平面分量，避免斜向加速
        planar_len = (move.x ** 2 + move.z ** 2) ** 0.5
        if planar_len > 1e-6:
            move.x /= planar_len
            move.z /= planar_len

        # 应用速度
        move *= speed

        # 读取/初始化相机缓存参数
        if not hasattr(renderer, '_camera_pos'):
            renderer._camera_pos = [0.0, 1.5, 3.0]
        if not hasattr(renderer, '_camera_lookat'):
            renderer._camera_lookat = [0.0, 1.0, 0.0]
        if not hasattr(renderer, '_camera_up'):
            renderer._camera_up = [0.0, 1.0, 0.0]

        # 视线方向（用于移动基和鼠标旋转）
        dir_x = renderer._camera_lookat[0] - renderer._camera_pos[0]
        dir_y = renderer._camera_lookat[1] - renderer._camera_pos[1]
        dir_z = renderer._camera_lookat[2] - renderer._camera_pos[2]
        # 投影到 xz 平面，构造前向与右向
        forward_x = dir_x
        forward_z = dir_z
        forward_len = (forward_x ** 2 + forward_z ** 2) ** 0.5
        if forward_len < 1e-6:
            forward_x, forward_z = 0.0, -1.0
            forward_len = 1.0
        forward_x /= forward_len
        forward_z /= forward_len
        right_x = forward_z
        right_z = -forward_x

        # 计算最终位移（x/z 使用相机局部前右基，y 直接用世界 y）
        delta_x = move.x * right_x + move.z * forward_x
        delta_z = move.x * right_z + move.z * forward_z
        delta_y = move.y

        renderer._camera_pos[0] += delta_x
        renderer._camera_pos[1] += delta_y
        renderer._camera_pos[2] += delta_z

        renderer._camera_lookat[0] += delta_x
        renderer._camera_lookat[1] += delta_y
        renderer._camera_lookat[2] += delta_z

        # 鼠标 yaw/pitch 旋转（按住指定按键时生效）
        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
        now_ns = time.perf_counter_ns()
        if self._last_time_ns is None:
            self._last_time_ns = now_ns
        time_elapsed = (now_ns - self._last_time_ns) * 1e-9
        self._last_time_ns = now_ns

        # 初始化 yaw/pitch（从当前前向推导）
        if self._yaw is None or self._pitch is None:
            fx = renderer._camera_lookat[0] - renderer._camera_pos[0]
            fy = renderer._camera_lookat[1] - renderer._camera_pos[1]
            fz = renderer._camera_lookat[2] - renderer._camera_pos[2]
            fl = (fx * fx + fy * fy + fz * fz) ** 0.5
            if fl < 1e-6:
                fx, fy, fz = 0.0, 0.0, -1.0
                fl = 1.0
            fx /= fl
            fy /= fl
            fz /= fl
            self._yaw = math.atan2(fx, fz)
            self._pitch = math.asin(max(-1.0, min(1.0, fy)))

        if (self._hold_mouse_button is None) or window.is_pressed(self._hold_mouse_button):
            if (self._last_mouse_x is None) or (self._last_mouse_y is None):
                self._last_mouse_x, self._last_mouse_y = curr_mouse_x, curr_mouse_y
            dx = curr_mouse_x - self._last_mouse_x
            dy = curr_mouse_y - self._last_mouse_y

            self._yaw -= dx * self._yaw_speed * time_elapsed * 60.0
            self._pitch += dy * self._pitch_speed * time_elapsed * 60.0

            pitch_limit = math.pi * 0.5 * 0.99
            if self._pitch > pitch_limit:
                self._pitch = pitch_limit
            elif self._pitch < -pitch_limit:
                self._pitch = -pitch_limit

        # 由 yaw/pitch 生成新的前向，并更新 lookat
        cos_pitch = math.cos(self._pitch)
        front_x = math.sin(self._yaw) * cos_pitch
        front_y = math.sin(self._pitch)
        front_z = math.cos(self._yaw) * cos_pitch
        renderer._camera_lookat[0] = renderer._camera_pos[0] + front_x
        renderer._camera_lookat[1] = renderer._camera_pos[1] + front_y
        renderer._camera_lookat[2] = renderer._camera_pos[2] + front_z

        # 写回 Taichi Camera
        camera.position(renderer._camera_pos[0], renderer._camera_pos[1], renderer._camera_pos[2])
        camera.lookat(renderer._camera_lookat[0], renderer._camera_lookat[1], renderer._camera_lookat[2])
        camera.up(renderer._camera_up[0], renderer._camera_up[1], renderer._camera_up[2])
        self._last_mouse_x, self._last_mouse_y = curr_mouse_x, curr_mouse_y

    def is_paused(self) -> bool:
        return self._paused

    def draw_ui(self, world: ISimulationWorld, renderer: IRenderer) -> None:
        # 通过渲染器窗口的 GUI 提供最小 UI：暂停/继续、重置按钮
        # window = getattr(renderer, 'window', None)
        # if window is None:
        #     return
        # gui = window.get_gui()
        # with gui.sub_window("Control", x=10, y=10, width=220, height=80):
        #     if gui.button("Pause/Resume (Space)"):
        #         print("test")
        #         self._paused = not self._paused
        #     if gui.button("Reset"):
        #         # 若世界实现了 reset 接口，可调用；否则忽略
        #         if hasattr(world, 'reset') and callable(getattr(world, 'reset')):
        #             world.reset()
        pass

