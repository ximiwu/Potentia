import os
import subprocess
import imageio_ffmpeg
from typing import Optional

import numpy as np
import taichi as ti

from renderers.base import IRenderer
from data.base import ISimulationData
from .base import IRecorder, RecordingMode
from energies.global_energy_container import GlobalEnergyContainer


class FrameRecorder(IRecorder):
    def __init__(
        self,
        output_dir: str,
        mode: RecordingMode = RecordingMode.DISABLED,
        make_video: bool = True,
        fps: int = 60,
        prefix: str = "frame",
        start_index: int = 0,
        ffmpeg_preset: str = "veryfast",
        ffmpeg_crf: int = 18,
    ) -> None:
        self._base_output_dir: str = output_dir
        self._session_dir: Optional[str] = None
        self._session_seq: int = 0
        self._mode: RecordingMode = mode
        self._make_video: bool = make_video
        self._fps: int = fps
        self._prefix: str = prefix
        self._index: int = start_index
        self._ffmpeg_preset: str = ffmpeg_preset
        self._ffmpeg_crf: int = ffmpeg_crf

        self._started: bool = False


    def set_mode(self, mode: RecordingMode) -> None:
        self._mode = mode

    def get_mode(self) -> RecordingMode:
        return self._mode

    def start(self) -> None:
        # 为本次录制挑选一个不与现有目录冲突的会话目录
        self._session_dir = self._compute_next_session_dir()
        os.makedirs(self._session_dir, exist_ok=True)
        # 为本次会话创建固定子目录结构
        self._png_dir = os.path.join(self._session_dir, "png")
        self._mp4_dir = os.path.join(self._session_dir, "mp4")
        self._dofs_dir = os.path.join(self._session_dir, "dofs")
        self._vels_dir = os.path.join(self._session_dir, "velocities")
        self._pred_dir = os.path.join(self._session_dir, "predicted_dofs")
        os.makedirs(self._png_dir, exist_ok=True)
        os.makedirs(self._mp4_dir, exist_ok=True)
        os.makedirs(self._dofs_dir, exist_ok=True)
        os.makedirs(self._vels_dir, exist_ok=True)
        os.makedirs(self._pred_dir, exist_ok=True)
        self._index = 0
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        if self._make_video and self._session_dir is not None:
            
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg is not None:
                input_pattern = os.path.join("png", f"{self._prefix}_%06d.png")
                output_mp4 = os.path.join("mp4", "video.mp4")
                cmd = [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(self._fps),
                    "-i",
                    input_pattern,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    self._ffmpeg_preset,
                    "-crf",
                    str(self._ffmpeg_crf),
                    output_mp4,
                ]
                try:
                    subprocess.run(cmd, cwd=self._session_dir, check=True)
                    print(f"[Recorder] MP4 合成完成: {os.path.join(self._session_dir, output_mp4)}")
                except Exception as e:
                    print(f"[Recorder] ffmpeg 合成失败: {e}")
            else:
                print("[Recorder] 未检测到 ffmpeg，可执行文件未找到，已跳过视频合成，仅保留 PNG 序列。")

        self._started = False
        # 下次 start() 时将创建新目录

    def on_frame_end(self, renderer: IRenderer, is_paused: bool) -> None:
        if self._mode == RecordingMode.DISABLED:
            return
        if self._mode == RecordingMode.RUNNING_ONLY and is_paused:
            return
        if not self._started:
            return

        window = getattr(renderer, "window", None)
        if window is None:
            print("[Recorder] 渲染器不具备 window 能力，无法保存帧。")
            return

        if self._session_dir is None:
            # 若出现未 start 就进入此分支（理论不应发生），安全返回
            return

        filename = f"{self._prefix}_{self._index:06d}.png"
        # 优先写入 png 子目录；兼容不存在时回退
        base_dir = getattr(self, "_png_dir", None) or self._session_dir
        path = os.path.join(base_dir, filename)
        try:
            window.save_image(path)
            self._index += 1
        except Exception as e:
            print(f"[Recorder] 保存帧失败: {e}")

    def on_solve_end(self, data: ISimulationData, dt: Optional[float] = None) -> None:
        if self._mode == RecordingMode.DISABLED:
            return
        if not self._started or self._session_dir is None:
            return

        # 确保会话子目录存在（若外部调用 start 失败或被清理）
        dofs_dir = getattr(self, "_dofs_dir", None) or os.path.join(self._session_dir, "dofs")
        vels_dir = getattr(self, "_vels_dir", None) or os.path.join(self._session_dir, "velocities")
        os.makedirs(dofs_dir, exist_ok=True)
        os.makedirs(vels_dir, exist_ok=True)

        try:
            ti.sync()
        except Exception:
            # 在某些后端上 sync 可能为空操作，忽略即可
            pass

        n = int(data.get_num_dofs())
        try:
            dofs_np = data.get_predicted_dofs().to_numpy()[:n]
            vels_np = data.get_velocities().to_numpy()[:n]
        except Exception as e:
            print(f"[Recorder] 导出 DoF/速度失败: {e}")
            return

        dofs_path = os.path.join(dofs_dir, f"dofs_{self._index:06d}.npy")
        vels_path = os.path.join(vels_dir, f"velocities_{self._index:06d}.npy")
        try:
            np.save(dofs_path, dofs_np)
            np.save(vels_path, vels_np)
        except Exception as e:
            print(f"[Recorder] 保存 DoF/速度文件失败: {e}")

        # 计算并保存 dofs 的 loss（基于当前 predicted_dofs）
        if dt is not None:
            try:
                container = GlobalEnergyContainer.get_instance()
                loss_val = float(container.compute_loss(data, data.get_dofs(), data.get_record_dofs(), float(dt)))
                loss_path = os.path.join(dofs_dir, f"loss_{self._index:06d}.npy")
                np.save(loss_path, np.array(loss_val, dtype=np.float32))
            except Exception as e:
                print(f"[Recorder] 计算/保存 dofs loss 失败: {e}")

    def on_predict_end(self, data: ISimulationData, dt: float) -> None:
        if self._mode == RecordingMode.DISABLED:
            return
        if not self._started or self._session_dir is None:
            return

        # 保存 predicted_dofs
        pred_dir = getattr(self, "_pred_dir", None) or os.path.join(self._session_dir, "predicted_dofs")
        os.makedirs(pred_dir, exist_ok=True)

        try:
            ti.sync()
        except Exception:
            pass

        n = int(data.get_num_dofs())
        try:
            pred_np = data.get_predicted_dofs().to_numpy()[:n]
        except Exception as e:
            print(f"[Recorder] 导出 predicted_dofs 失败: {e}")
            return

        pred_path = os.path.join(pred_dir, f"predicted_dofs_{self._index:06d}.npy")
        try:
            np.save(pred_path, pred_np)
        except Exception as e:
            print(f"[Recorder] 保存 predicted_dofs 文件失败: {e}")

        # 计算并保存 predicted_dofs 的 loss
        try:
            container = GlobalEnergyContainer.get_instance()
            loss_val = float(container.compute_loss(data, data.get_predicted_dofs(), data.get_record_dofs(), float(dt)))
            loss_path = os.path.join(pred_dir, f"loss_{self._index:06d}.npy")
            np.save(loss_path, np.array(loss_val, dtype=np.float32))
        except Exception as e:
            print(f"[Recorder] 计算/保存 predicted_dofs loss 失败: {e}")

    def _compute_next_session_dir(self) -> str:
        base = self._base_output_dir
        if not os.path.exists(base):
            return base
        # 尝试 base_001, base_002, ...
        idx = 1
        while True:
            candidate = f"{base}_{idx:03d}"
            if not os.path.exists(candidate):
                return candidate
            idx += 1


