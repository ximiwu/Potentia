from typing import Optional
import math
import time

import taichi as ti
import tkinter as tk

from controller.base import IInputHandler
from renderers.base import IRenderer
from world.base import ISimulationWorld
from recorders.base import RecordingMode


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

        # 外部控制面板（Tkinter）相关句柄
        self._tk_root: Optional[tk.Tk] = None
        self._tk_state_label: Optional[tk.Label] = None
        self._tk_pos_label: Optional[tk.Label] = None
        self._tk_look_label: Optional[tk.Label] = None
        self._tk_yaw_label: Optional[tk.Label] = None
        self._tk_mode_var: Optional[tk.StringVar] = None

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

        # # 由 yaw/pitch 生成新的前向，并更新 lookat
        # cos_pitch = math.cos(self._pitch)
        # front_x = math.sin(self._yaw) * cos_pitch
        # front_y = math.sin(self._pitch)
        # front_z = math.cos(self._yaw) * cos_pitch
        # renderer._camera_lookat[0] = renderer._camera_pos[0] + front_x
        # renderer._camera_lookat[1] = renderer._camera_pos[1] + front_y
        # renderer._camera_lookat[2] = renderer._camera_pos[2] + front_z

        # 仅在旋转时更新 lookat，并保留原始距离
        if (self._hold_mouse_button is None) or window.is_pressed(self._hold_mouse_button):
            cos_pitch = math.cos(self._pitch)
            front_x = math.sin(self._yaw) * cos_pitch
            front_y = math.sin(self._pitch)
            front_z = math.cos(self._yaw) * cos_pitch
            ox = renderer._camera_lookat[0] - renderer._camera_pos[0]
            oy = renderer._camera_lookat[1] - renderer._camera_pos[1]
            oz = renderer._camera_lookat[2] - renderer._camera_pos[2]
            orig_len = (ox*ox + oy*oy + oz*oz) ** 0.5
            if orig_len < 1e-6:
                orig_len = 1.0
            renderer._camera_lookat[0] = renderer._camera_pos[0] + front_x * orig_len
            renderer._camera_lookat[1] = renderer._camera_pos[1] + front_y * orig_len
            renderer._camera_lookat[2] = renderer._camera_pos[2] + front_z * orig_len

        # 写回 Taichi Camera
        camera.position(renderer._camera_pos[0], renderer._camera_pos[1], renderer._camera_pos[2])
        camera.lookat(renderer._camera_lookat[0], renderer._camera_lookat[1], renderer._camera_lookat[2])
        camera.up(renderer._camera_up[0], renderer._camera_up[1], renderer._camera_up[2])
        self._last_mouse_x, self._last_mouse_y = curr_mouse_x, curr_mouse_y

    def is_paused(self) -> bool:
        return self._paused

    def set_paused_state(self, paused: bool):
        self._paused = paused

    def draw_ui(self, world: ISimulationWorld, renderer: IRenderer) -> None:
        # 使用 Tkinter 作为独立控制面板（避免 GGUI 多窗口的 ImGui 后端冲突）
        if (self._tk_root is None) or (not self._tk_root.winfo_exists()):
            root = tk.Tk()
            root.title("Control Panel")
            # 尝试把窗口放在渲染窗口右侧（无法获取确切位置时使用保守偏移）
            try:
                x_offset = 1400
                y_offset = 80
                w = 1280
                if hasattr(renderer, 'window') and hasattr(renderer.window, 'get_window_shape'):
                    try:
                        w_shape = renderer.window.get_window_shape()
                        if isinstance(w_shape, (tuple, list)) and len(w_shape) >= 2:
                            w = int(w_shape[0])
                    except Exception:
                        pass
                x_offset = w + 220
                root.geometry(f"360x440+{x_offset}+{y_offset}")
            except Exception:
                try:
                    root.geometry("360x440+1500+80")
                except Exception:
                    pass

            # 关闭时仅销毁引用，允许后续自动重建
            def _on_close():
                try:
                    root.destroy()
                finally:
                    self._tk_root = None

            root.protocol("WM_DELETE_WINDOW", _on_close)

            # Simulation
            sim_frame = tk.LabelFrame(root, text="Simulation")
            sim_frame.pack(fill="x", padx=8, pady=8)
            self._tk_state_label = tk.Label(sim_frame, text="State: Unknown")
            self._tk_state_label.pack(anchor="w", padx=8, pady=4)

            def _toggle_pause():
                self._paused = not self._paused

            tk.Button(sim_frame, text="Pause/Resume (Space)", command=_toggle_pause).pack(anchor="w", padx=8, pady=4)

            # Camera
            cam_frame = tk.LabelFrame(root, text="Camera")
            cam_frame.pack(fill="x", padx=8, pady=8)
            self._tk_pos_label = tk.Label(cam_frame, text="pos: N/A")
            self._tk_pos_label.pack(anchor="w", padx=8, pady=4)
            self._tk_look_label = tk.Label(cam_frame, text="lookat: N/A")
            self._tk_look_label.pack(anchor="w", padx=8, pady=4)
            self._tk_yaw_label = tk.Label(cam_frame, text="yaw/pitch: N/A")
            self._tk_yaw_label.pack(anchor="w", padx=8, pady=4)

            # Recorder
            rec_frame = tk.LabelFrame(root, text="Recorder")
            rec_frame.pack(fill="x", padx=8, pady=8)

            self._tk_mode_var = tk.StringVar(value="DISABLED")

            def _apply_mode():
                recorder_local = None
                if hasattr(world, 'get_recorder') and callable(getattr(world, 'get_recorder')):
                    recorder_local = world.get_recorder()
                if recorder_local is None:
                    return
                name = self._tk_mode_var.get()
                try:
                    recorder_local.set_mode(RecordingMode[name])
                except Exception:
                    pass

            modes = ["DISABLED", "RUNNING_ONLY", "ALWAYS"]
            for m in modes:
                tk.Radiobutton(rec_frame, text=m, variable=self._tk_mode_var, value=m, command=_apply_mode).pack(anchor="w", padx=8)

            def _start_rec():
                recorder_local = None
                if hasattr(world, 'get_recorder') and callable(getattr(world, 'get_recorder')):
                    recorder_local = world.get_recorder()
                if recorder_local is None:
                    return
                try:
                    recorder_local.start()
                except Exception:
                    pass

            def _stop_rec():
                recorder_local = None
                if hasattr(world, 'get_recorder') and callable(getattr(world, 'get_recorder')):
                    recorder_local = world.get_recorder()
                if recorder_local is None:
                    return
                try:
                    recorder_local.stop()
                except Exception:
                    pass

            btns = tk.Frame(rec_frame)
            btns.pack(fill="x", padx=8, pady=4)
            tk.Button(btns, text="Start Recording", command=_start_rec).pack(side="left")
            tk.Button(btns, text="Stop Recording", command=_stop_rec).pack(side="left", padx=8)

            self._tk_root = root

        # 每帧更新显示文本
        if self._tk_root is not None and self._tk_root.winfo_exists():
            if self._tk_state_label is not None:
                self._tk_state_label.config(text=f"State: {'Paused' if self._paused else 'Running'}")

            pos = getattr(renderer, '_camera_pos', None)
            look = getattr(renderer, '_camera_lookat', None)
            if self._tk_pos_label is not None:
                if pos is not None and len(pos) == 3:
                    self._tk_pos_label.config(text=f"pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                else:
                    self._tk_pos_label.config(text="pos: N/A")
            if self._tk_look_label is not None:
                if look is not None and len(look) == 3:
                    self._tk_look_label.config(text=f"lookat: ({look[0]:.3f}, {look[1]:.3f}, {look[2]:.3f})")
                else:
                    self._tk_look_label.config(text="lookat: N/A")

            if self._tk_yaw_label is not None:
                if self._yaw is not None and self._pitch is not None:
                    yaw_deg = self._yaw * 180.0 / math.pi
                    pitch_deg = self._pitch * 180.0 / math.pi
                    self._tk_yaw_label.config(text=f"yaw: {yaw_deg:.1f}°, pitch: {pitch_deg:.1f}°")
                elif pos is not None and look is not None:
                    fx = look[0] - pos[0]
                    fy = look[1] - pos[1]
                    fz = look[2] - pos[2]
                    fl = (fx * fx + fy * fy + fz * fz) ** 0.5
                    if fl > 1e-6:
                        fx /= fl
                        fy /= fl
                        fz /= fl
                        yaw = math.atan2(fx, fz)
                        pitch = math.asin(max(-1.0, min(1.0, fy)))
                        self._tk_yaw_label.config(text=f"yaw: {yaw * 180.0 / math.pi:.1f}°, pitch: {pitch * 180.0 / math.pi:.1f}°")
                    else:
                        self._tk_yaw_label.config(text="yaw/pitch: N/A")
                else:
                    self._tk_yaw_label.config(text="yaw/pitch: N/A")

            try:
                self._tk_root.update_idletasks()
                self._tk_root.update()
            except Exception:
                # 若用户强制关闭窗口，容错为下次重建
                self._tk_root = None

