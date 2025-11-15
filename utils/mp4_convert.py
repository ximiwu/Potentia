import os
import re
import sys
import threading
import shutil
from typing import Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as e:
    raise RuntimeError('需要 tkinter 以运行该工具') from e

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError('需要安装 Pillow: pip install Pillow') from e

try:
    import imageio
except Exception as e:
    raise RuntimeError('需要安装 imageio: pip install imageio imageio-ffmpeg') from e

try:
    import imageio_ffmpeg  # 提供 ffmpeg 可执行文件
    IMAGEIO_FFMPEG_AVAILABLE = True
except Exception:
    IMAGEIO_FFMPEG_AVAILABLE = False

try:
    import numpy as np
except Exception as e:
    raise RuntimeError('需要安装 numpy: pip install numpy') from e


MISSING_MODE_DUPLICATE = 'duplicate'
MISSING_MODE_SKIP = 'skip'


def has_ffmpeg() -> bool:
    # 优先使用 imageio-ffmpeg 提供的内置 ffmpeg
    if IMAGEIO_FFMPEG_AVAILABLE:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            return os.path.isfile(exe)
        except Exception:
            pass
    # 回退到系统 PATH 中的 ffmpeg
    return shutil.which('ffmpeg') is not None


def parse_frames(folder: str) -> Tuple[Dict[int, str], Optional[int], Optional[int]]:
    """解析目录中的 frame_*.png 文件，返回 idx->path 映射与最小/最大 idx。

    仅匹配形如 frame_000040.png 的文件名，其中数字部分宽度不限。
    """
    mapping: Dict[int, str] = {}
    pattern = re.compile(r'^frame_(\d+)\.png$', re.IGNORECASE)
    try:
        for name in os.listdir(folder):
            m = pattern.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            mapping[idx] = os.path.join(folder, name)
    except Exception as e:
        raise RuntimeError(f'读取目录失败: {e}')

    if not mapping:
        return {}, None, None
    idxs = sorted(mapping.keys())
    return mapping, idxs[0], idxs[-1]


def build_frame_sequence(mapping: Dict[int, str], min_idx: int, max_idx: int, missing_mode: str) -> List[str]:
    """根据缺帧策略构建待编码的图像路径序列。"""
    if missing_mode == MISSING_MODE_SKIP:
        return [mapping[i] for i in sorted(mapping.keys())]

    if missing_mode == MISSING_MODE_DUPLICATE:
        seq: List[str] = []
        prev_path: Optional[str] = None
        for i in range(min_idx, max_idx + 1):
            if i in mapping:
                prev_path = mapping[i]
                seq.append(prev_path)
            else:
                # 缺帧用前一帧补齐；若前一帧不存在（理应不会发生，因为从min_idx开始），则跳过
                if prev_path is not None:
                    seq.append(prev_path)
                else:
                    # 安全降级，实际不会触发
                    continue
        return seq

    raise ValueError('未知的缺帧策略')


def convert_with_imageio(paths: List[str], fps: int, output_path: str) -> None:
    """使用 imageio 的 ffmpeg 插件编码为 mp4。

    - 将所有帧统一为 RGB，尺寸以首帧为基准，必要时缩放。
    - 使用 libx264 + yuv420p，设置 faststart，兼容主流播放器。
    """
    if not paths:
        raise RuntimeError('没有可转换的帧')

    first = Image.open(paths[0]).convert('RGB')
    width, height = first.size
    # yuv420p 需要偶数尺寸，必要时下调 1 像素
    target_w = width - (width % 2)
    target_h = height - (height % 2)
    if target_w <= 0 or target_h <= 0:
        raise RuntimeError(f'无效尺寸 {width}x{height}')
    del first

    # 使用 imageio 的 ffmpeg 写入器
    # 传递 ffmpeg 参数以获得更好的质量与兼容性
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        format='ffmpeg',
        pixelformat='yuv420p',
        macro_block_size=None,
        output_params=['-movflags', '+faststart', '-preset', 'fast', '-crf', '18'],
    )

    try:
        for p in paths:
            img = Image.open(p).convert('RGB')
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.BILINEAR)
            frame = np.asarray(img)
            writer.append_data(frame)
    finally:
        writer.close()


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('MP4 Convert - frame_*.png')
        self.folder: Optional[str] = None

        # 缺帧策略
        self.missing_mode = tk.StringVar(value=MISSING_MODE_DUPLICATE)

        # FPS 与输出名
        self.fps_var = tk.StringVar(value='30')
        self.output_name_var = tk.StringVar(value='converted.mp4')

        # 状态
        self.status_var = tk.StringVar(value='请选择一个包含 frame_*.png 的文件夹')

        self._build_ui()

    def _build_ui(self) -> None:
        frm = tk.Frame(self.root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # 目录选择
        tk.Label(frm, text='目标文件夹:').grid(row=0, column=0, sticky='w')
        self.folder_label = tk.Label(frm, text='(未选择)', anchor='w')
        self.folder_label.grid(row=0, column=1, sticky='w')
        tk.Button(frm, text='选择文件夹', command=self._choose_folder).grid(row=0, column=2, padx=8)

        # 缺帧策略
        tk.Label(frm, text='缺帧处理:').grid(row=1, column=0, sticky='w')
        tk.Radiobutton(frm, text='用前一帧补齐（重复）', variable=self.missing_mode, value=MISSING_MODE_DUPLICATE).grid(row=1, column=1, sticky='w')
        tk.Radiobutton(frm, text='跳过缺帧', variable=self.missing_mode, value=MISSING_MODE_SKIP).grid(row=1, column=2, sticky='w')

        # FPS
        tk.Label(frm, text='帧率 (FPS):').grid(row=2, column=0, sticky='w')
        tk.Entry(frm, textvariable=self.fps_var, width=8).grid(row=2, column=1, sticky='w')

        # 输出名
        tk.Label(frm, text='输出文件名:').grid(row=3, column=0, sticky='w')
        tk.Entry(frm, textvariable=self.output_name_var, width=24).grid(row=3, column=1, sticky='w')

        # 开始按钮与状态
        tk.Button(frm, text='开始转换', command=self._start_convert).grid(row=4, column=0, pady=10)
        tk.Label(frm, textvariable=self.status_var, fg='#555').grid(row=4, column=1, columnspan=2, sticky='w')

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title='选择包含 frame_*.png 的文件夹')
        if folder:
            self.folder = folder
            self.folder_label.config(text=folder)
            self.status_var.set('已选择文件夹，准备转换')

    def _start_convert(self) -> None:
        if not self.folder:
            messagebox.showwarning('提示', '请先选择文件夹')
            return
        if not has_ffmpeg():
            messagebox.showerror('错误', '未检测到 ffmpeg（imageio-ffmpeg 或系统 ffmpeg）。请执行 pip install imageio-ffmpeg，或配置系统 PATH')
            return

        try:
            fps = int(self.fps_var.get().strip())
        except Exception:
            messagebox.showwarning('提示', '帧率应为整数，例如 30')
            return

        output_name = self.output_name_var.get().strip()
        if not output_name:
            messagebox.showwarning('提示', '请填写输出文件名，例如 converted.mp4')
            return
        if not output_name.lower().endswith('.mp4'):
            output_name += '.mp4'

        output_path = os.path.join(self.folder, output_name)

        # 启动后台线程，避免界面卡死
        t = threading.Thread(target=self._convert_worker, args=(self.folder, fps, output_path, self.missing_mode.get()), daemon=True)
        t.start()
        self.status_var.set('正在转换，请稍候...')

    def _convert_worker(self, folder: str, fps: int, output_path: str, missing_mode: str) -> None:
        try:
            mapping, min_idx, max_idx = parse_frames(folder)
            if not mapping:
                self._set_status('未找到 frame_*.png 文件')
                return
            assert min_idx is not None and max_idx is not None

            paths = build_frame_sequence(mapping, min_idx, max_idx, missing_mode)

            # 统计缺帧数量
            total_expected = (max_idx - min_idx + 1) if missing_mode == MISSING_MODE_DUPLICATE else len(mapping)
            total_actual = len(paths)
            missing_count = total_expected - len(mapping)

            convert_with_imageio(paths, fps, output_path)
            msg = f'转换完成: {output_path}\n总帧: {total_actual}，原始帧: {len(mapping)}，缺帧: {max(missing_count, 0)}（策略: {"重复" if missing_mode==MISSING_MODE_DUPLICATE else "跳过"}）'
            self._set_status('转换完成')
            messagebox.showinfo('完成', msg)
        except Exception as e:
            self._set_status('转换失败')
            messagebox.showerror('错误', f'转换失败: {e}')

    def _set_status(self, text: str) -> None:
        # 在主线程更新状态文本
        def _update() -> None:
            self.status_var.set(text)
        try:
            self.root.after(0, _update)
        except Exception:
            pass


def main() -> None:
    app = App()
    app.root.mainloop()


if __name__ == '__main__':
    main()