import os
import subprocess
import imageio_ffmpeg
from typing import Optional

from renderers.base import IRenderer
from .base import IRecorder, RecordingMode


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
        self._output_dir: str = output_dir
        self._mode: RecordingMode = mode
        self._make_video: bool = make_video
        self._fps: int = fps
        self._prefix: str = prefix
        self._index: int = start_index
        self._ffmpeg_preset: str = ffmpeg_preset
        self._ffmpeg_crf: int = ffmpeg_crf

        self._started: bool = False

        os.makedirs(self._output_dir, exist_ok=True)

    def set_mode(self, mode: RecordingMode) -> None:
        self._mode = mode

    def get_mode(self) -> RecordingMode:
        return self._mode

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        if self._make_video:
            
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg is not None:
                input_pattern = f"{self._prefix}_%06d.png"
                output_mp4 = "video.mp4"
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
                    subprocess.run(cmd, cwd=self._output_dir, check=True)
                    print(f"[Recorder] MP4 合成完成: {os.path.join(self._output_dir, output_mp4)}")
                except Exception as e:
                    print(f"[Recorder] ffmpeg 合成失败: {e}")
            else:
                print("[Recorder] 未检测到 ffmpeg，可执行文件未找到，已跳过视频合成，仅保留 PNG 序列。")

        self._started = False

    def on_frame_end(self, renderer: IRenderer, is_paused: bool) -> None:
        if self._mode == RecordingMode.DISABLED:
            return
        if self._mode == RecordingMode.RUNNING_ONLY and is_paused:
            return

        window = getattr(renderer, "window", None)
        if window is None:
            print("[Recorder] 渲染器不具备 window 能力，无法保存帧。")
            return

        if not self._started:
            # 若未显式 start，则自动开始，便于仅代码配置场景
            self._started = True

        filename = f"{self._prefix}_{self._index:06d}.png"
        path = os.path.join(self._output_dir, filename)
        try:
            window.save_image(path)
            self._index += 1
        except Exception as e:
            print(f"[Recorder] 保存帧失败: {e}")


