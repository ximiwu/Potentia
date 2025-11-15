import os
import subprocess
import shutil
import imageio_ffmpeg
import imageio
import csv
from typing import Optional, Callable

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

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
        iter_plot: bool = False,
        solve_info: bool = False,
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
        self.iter_callback: bool = iter_plot
        self.record_solve_info: bool = solve_info

        self._started: bool = False
        # 起始帧编号控制：默认 -1，首帧为 _init_frame + 1（即 0）
        self._init_frame: int = -1
        # 迭代曲线缓存将在 start() 中初始化，避免在 __init__ 中写入状态
        self._iter_indices = []
        self._iter_losses = []


    def set_mode(self, mode: RecordingMode) -> None:
        self._mode = mode

    def set_init_frame(self, init_frame: int) -> None:
        self._init_frame = int(init_frame)

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
        # 迭代曲线缓存（避免在 __init__ 中写入新状态）
        self._iter_indices = []
        self._iter_losses = []
        # 使用 init_frame 控制首帧编号（默认 0）
        self._index = int(self._init_frame) + 1
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        if self._make_video and self._session_dir is not None:
            input_pattern = os.path.join("png", f"{self._prefix}_%06d.png")
            output_rel = os.path.join("mp4", "video.mp4")

            # 统计可用帧，若没有则直接提示并跳过
            png_dir = getattr(self, "_png_dir", None) or os.path.join(self._session_dir, "png")
            try:
                png_files = sorted([n for n in os.listdir(png_dir) if n.lower().endswith(".png")])
            except Exception:
                png_files = []
            if len(png_files) == 0:
                print("[Recorder] 未检测到 PNG 帧，跳过视频合成。")
            else:
                ffmpeg_exe = None
                try:
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = None
                if ffmpeg_exe is None:
                    ffmpeg_exe = shutil.which("ffmpeg")

                if ffmpeg_exe is not None:
                    cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-framerate",
                        str(self._fps),
                        "-start_number",
                        "0",
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
                        output_rel,
                    ]
                    try:
                        subprocess.run(cmd, cwd=self._session_dir, check=True)
                        print(f"[Recorder] MP4 合成完成: {os.path.join(self._session_dir, output_rel)}")
                    except Exception as e:
                        print(f"[Recorder] ffmpeg 合成失败: {e}. 尝试使用 imageio 回退方案。")
                        # ffmpeg 失败时回退到 imageio 写入器
                        try:
                            writer = imageio.get_writer(
                                os.path.join(self._session_dir, output_rel),
                                fps=self._fps,
                                codec="libx264",
                                format="ffmpeg",
                                pixelformat="yuv420p",
                                output_params=["-preset", self._ffmpeg_preset, "-crf", str(self._ffmpeg_crf)],
                            )
                            for name in png_files:
                                frame_path = os.path.join(png_dir, name)
                                img = imageio.imread(frame_path)
                                writer.append_data(img)
                            writer.close()
                            print(f"[Recorder] MP4 合成完成(回退): {os.path.join(self._session_dir, output_rel)}")
                        except Exception as e2:
                            print(f"[Recorder] imageio 合成失败: {e2}")
                else:
                    # 无 ffmpeg 时直接使用 imageio 写入器
                    try:
                        writer = imageio.get_writer(
                            os.path.join(self._session_dir, output_rel),
                            fps=self._fps,
                            codec="libx264",
                            format="ffmpeg",
                            pixelformat="yuv420p",
                            output_params=["-preset", self._ffmpeg_preset, "-crf", str(self._ffmpeg_crf)],
                        )
                        for name in png_files:
                            frame_path = os.path.join(png_dir, name)
                            img = imageio.imread(frame_path)
                            writer.append_data(img)
                        writer.close()
                        print(f"[Recorder] MP4 合成完成(无 ffmpeg，使用 imageio): {os.path.join(self._session_dir, output_rel)}")
                    except Exception as e:
                        print(f"[Recorder] 无法合成 MP4（缺少 ffmpeg 且 imageio 失败）: {e}")

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
            dofs_np = data.get_dofs().to_numpy()[:n]
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

        # 绘制 y=loss, x=iteration 的折线图
        try:
            if self.iter_callback and hasattr(self, "_iter_losses") and len(self._iter_losses) > 0:
                plots_dir = os.path.join(self._session_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
                ax.plot(self._iter_indices, self._iter_losses, color="tab:blue", linewidth=1.5)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.set_title(f"Loss vs Iteration (frame {self._index:06d})")
                ax.grid(True, linestyle="--", alpha=0.4)
                fig.tight_layout()

                plot_path = os.path.join(plots_dir, f"loss_iter_{self._index:06d}.png")
                fig.savefig(plot_path)
                x_array = np.array(self._iter_indices, dtype=np.int32)
                y_array = np.array(self._iter_losses, dtype=np.float32)
                x_path = os.path.join(plots_dir, f"x_{self._index:06d}.npy")
                y_path = os.path.join(plots_dir, f"y_{self._index:06d}.npy")
                np.save(x_path, x_array)
                np.save(y_path, y_array)
                plt.close(fig)
                print(f"[Recorder] 迭代损失折线图已保存: {plot_path}")
                print(f"[Recorder] 迭代数据已保存: x={x_path}, y={y_path}")
            # 清理迭代缓存，避免跨帧污染
            self._iter_indices = []
            self._iter_losses = []
        except Exception as e:
            print(f"[Recorder] 保存迭代损失折线图失败: {e}")

        try:
            if self.record_solve_info and dt is not None:
                container = GlobalEnergyContainer.get_instance()
                grad_norm = float(container.compute_loss_gradient_fro_norm(
                    data,
                    data.get_dofs(),
                    data.get_record_dofs(),
                    float(dt),
                ))
                hess_fro = float(container.compute_loss_hessian_fro_norm(data, float(dt)))
                loss_val = float(container.compute_loss(data, data.get_dofs(), data.get_record_dofs(), float(dt)))
                csv_path = os.path.join(self._session_dir, "solve_info.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["frame_idx", "loss", "grad_fro_norm", "hessian_fro_norm"])
                    w.writerow([
                        int(self._index),
                        f"{loss_val:.9g}",
                        f"{grad_norm:.9g}",
                        f"{hess_fro:.9g}",
                    ])
        except Exception as e:
            print(f"[Recorder] 统计求解信息失败: {e}")

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
            # 将本次 predicted_dofs 的损失记录到迭代缓存
            if self.iter_callback:
                self._iter_losses.append(loss_val)
                self._iter_indices.append(0)
        except Exception as e:
            print(f"[Recorder] 计算/保存 predicted_dofs loss 失败: {e}")
        
        print("predicted_dofs loss: ", loss_val)

    def get_iteration_callback(self) -> Optional[Callable[[int, ISimulationData, float], None]]:
        if not self.iter_callback or self._mode == RecordingMode.DISABLED:
            return None
        return self._on_solver_iteration

    def _on_solver_iteration(self, iter_idx: int, data: ISimulationData, dt: float, loss: float = None) -> None:
        # 计算当前迭代的 loss 并缓存
        try:
            if loss is None:
                container = GlobalEnergyContainer.get_instance()
                loss = float(container.compute_loss(
                    data,
                    data.get_predicted_dofs(),
                    data.get_record_dofs(),
                    float(dt)
                ))
        except Exception as e:
            print(f"[Recorder] 迭代损失计算失败: {e}")
            return

        self._iter_indices.append(int(iter_idx) + 1)
        self._iter_losses.append(loss)

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


