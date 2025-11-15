import os
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Callable

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.axes import Axes

# 全局字体设置为 Arial（优先使用），并确保负号正常显示
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.unicode_minus"] = False

# 曲线参数默认值（便于集中修改）
DEFAULT_LINEWIDTH: float = 1.4
DEFAULT_DRAW_POINTS: bool = True
DEFAULT_MARKER_SIZE: float = 1.0
DEFAULT_MARKER_EDGE_WIDTH: float = 6.5

# 图像尺寸与分辨率（预览与保存分离，保存使用高DPI）
FIG_SIZE: Tuple[float, float] = (8.0, 6.0)
PREVIEW_DPI: int = 100
# 目标：保存图片放大10倍仍肉眼无损，因此将保存DPI提升为预览DPI的10倍
SAVE_SCALE: int = 10
SAVE_DPI: int = PREVIEW_DPI * SAVE_SCALE


@dataclass
class CurveConfig:
    folder: str
    name: str
    color: str
    linewidth: float
    draw_points: bool
    marker_size: float
    marker_edge_width: float
    start_index: int
    end_index: int


class CurveColumn:
    def __init__(self, parent: tk.Widget, index: int) -> None:
        self.parent = parent
        self.index = index

        self.folder_var = tk.StringVar(value="")
        self.name_var = tk.StringVar(value=f"曲线{index}")
        self.color_var = tk.StringVar(value=self._default_color(index))
        self.linewidth_var = tk.DoubleVar(value=DEFAULT_LINEWIDTH)
        self.draw_points_var = tk.BooleanVar(value=DEFAULT_DRAW_POINTS)
        self.marker_size_var = tk.DoubleVar(value=DEFAULT_MARKER_SIZE)
        self.marker_edge_width_var = tk.DoubleVar(value=DEFAULT_MARKER_EDGE_WIDTH)
        # 曲线1采用与其他曲线不同的初始范围：从0到末尾
        if index == 1:
            self.start_index_var = tk.IntVar(value=0)
            self.end_index_var = tk.IntVar(value=-1)
        else:
            self.start_index_var = tk.IntVar(value=-1)
            self.end_index_var = tk.IntVar(value=-1)

        self.frame = ttk.LabelFrame(parent, text=f"曲线 {index}", padding=(6, 6, 6, 6))
        self._build_ui()

    def _default_color(self, i: int) -> str:
        default_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        return default_colors[(i - 1) % len(default_colors)]

    def _build_ui(self) -> None:
        folder_entry = ttk.Entry(self.frame, textvariable=self.folder_var, width=26)
        folder_entry.grid(row=0, column=0, sticky="we", padx=2, pady=2)
        select_btn = ttk.Button(self.frame, text="选择文件夹", command=self._on_select_folder)
        select_btn.grid(row=0, column=1, sticky="we", padx=2, pady=2)

        ttk.Label(self.frame, text="图例名称").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        name_entry = ttk.Entry(self.frame, textvariable=self.name_var, width=20)
        name_entry.grid(row=1, column=1, sticky="we", padx=2, pady=2)

        ttk.Label(self.frame, text="线颜色").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        color_frame = ttk.Frame(self.frame)
        color_frame.grid(row=2, column=1, sticky="we", padx=2, pady=2)
        color_entry = ttk.Entry(color_frame, textvariable=self.color_var, width=14)
        color_entry.pack(side="left", fill="x", expand=True)
        color_btn = ttk.Button(color_frame, text="选色", command=self._on_choose_color)
        color_btn.pack(side="right")

        ttk.Label(self.frame, text="线宽").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        linewidth_spin = ttk.Spinbox(
            self.frame, from_=0.1, to=20.0, increment=0.1, textvariable=self.linewidth_var, width=8
        )
        linewidth_spin.grid(row=3, column=1, sticky="we", padx=2, pady=2)

        draw_points_cb = ttk.Checkbutton(self.frame, text="画点", variable=self.draw_points_var)
        draw_points_cb.grid(row=4, column=0, sticky="w", padx=2, pady=2, columnspan=2)

        ttk.Label(self.frame, text="点大小").grid(row=5, column=0, sticky="w", padx=2, pady=2)
        marker_size_spin = ttk.Spinbox(
            self.frame, from_=1.0, to=50.0, increment=1.0, textvariable=self.marker_size_var, width=8
        )
        marker_size_spin.grid(row=5, column=1, sticky="we", padx=2, pady=2)

        ttk.Label(self.frame, text="点边宽").grid(row=6, column=0, sticky="w", padx=2, pady=2)
        marker_edge_spin = ttk.Spinbox(
            self.frame, from_=0.0, to=10.0, increment=0.5, textvariable=self.marker_edge_width_var, width=8
        )
        marker_edge_spin.grid(row=6, column=1, sticky="we", padx=2, pady=2)

        # 有效点范围（-1, -1 表示全部有效）
        ttk.Label(self.frame, text="有效范围").grid(row=7, column=0, sticky="w", padx=2, pady=2)
        range_frame = ttk.Frame(self.frame)
        range_frame.grid(row=7, column=1, sticky="we", padx=2, pady=2)
        ttk.Label(range_frame, text="起始").pack(side="left")
        start_spin = ttk.Spinbox(range_frame, from_=-1, to=1_000_000, increment=1,
                                 textvariable=self.start_index_var, width=8)
        start_spin.pack(side="left", padx=2)
        ttk.Label(range_frame, text="结束").pack(side="left")
        end_spin = ttk.Spinbox(range_frame, from_=-1, to=1_000_000, increment=1,
                               textvariable=self.end_index_var, width=8)
        end_spin.pack(side="left", padx=2)

        clear_btn = ttk.Button(self.frame, text="清空本列", command=self._clear_column)
        clear_btn.grid(row=8, column=0, columnspan=2, sticky="we", padx=2, pady=4)

        # 参数应用到全体按钮（除图例名称、线颜色）
        apply_btn = ttk.Button(self.frame, text="参数应用到全体", command=self._on_apply_params_to_all)
        apply_btn.grid(row=9, column=0, columnspan=2, sticky="we", padx=2, pady=2)

    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def _on_select_folder(self) -> None:
        folder = filedialog.askdirectory(title=f"选择曲线 {self.index} 的数据文件夹")
        if folder:
            self.folder_var.set(folder)

    def _on_choose_color(self) -> None:
        color_tuple = colorchooser.askcolor(title=f"选择曲线 {self.index} 颜色", initialcolor=self.color_var.get())
        if color_tuple and color_tuple[1]:
            self.color_var.set(color_tuple[1])

    def _clear_column(self) -> None:
        self.folder_var.set("")
        self.name_var.set(f"line{self.index}")
        self.color_var.set(self._default_color(self.index))
        self.linewidth_var.set(DEFAULT_LINEWIDTH)
        self.draw_points_var.set(DEFAULT_DRAW_POINTS)
        self.marker_size_var.set(DEFAULT_MARKER_SIZE)
        self.marker_edge_width_var.set(DEFAULT_MARKER_EDGE_WIDTH)
        # 保持曲线1的默认范围与初始化一致；其他曲线重置为（-1, -1）
        if self.index == 1:
            self.start_index_var.set(0)
            self.end_index_var.set(-1)
        else:
            self.start_index_var.set(-1)
            self.end_index_var.set(-1)

    def get_config(self) -> CurveConfig:
        return CurveConfig(
            folder=self.folder_var.get().strip(),
            name=self.name_var.get().strip() or f"line{self.index}",
            color=self.color_var.get().strip() or self._default_color(self.index),
            linewidth=float(self.linewidth_var.get()),
            draw_points=bool(self.draw_points_var.get()),
            marker_size=float(self.marker_size_var.get()),
            marker_edge_width=float(self.marker_edge_width_var.get()),
            start_index=int(self.start_index_var.get()),
            end_index=int(self.end_index_var.get()),
        )

    def bind_on_change(self, callback: Callable[[], None]) -> None:
        def _notify(*_: object) -> None:
            callback()
        # 绑定变量变化事件以自动刷新预览
        self.folder_var.trace_add("write", _notify)
        self.name_var.trace_add("write", _notify)
        self.color_var.trace_add("write", _notify)
        self.linewidth_var.trace_add("write", _notify)
        self.draw_points_var.trace_add("write", _notify)
        self.marker_size_var.trace_add("write", _notify)
        self.marker_edge_width_var.trace_add("write", _notify)
        self.start_index_var.trace_add("write", _notify)
        self.end_index_var.trace_add("write", _notify)

    def _on_apply_params_to_all(self) -> None:
        # 回调由外部 PlotApp 设置
        if hasattr(self, "apply_params_to_all_cb") and callable(self.apply_params_to_all_cb):
            self.apply_params_to_all_cb(self)


class PlotApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Numpy 曲线绘制器（10列，共有编号批量保存）")
        self.root.geometry("1500x800")

        self.main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_pane.pack(fill="both", expand=True)

        # 全局图表设置
        self.title_var = tk.StringVar(value="graph")
        self.xlabel_var = tk.StringVar(value="x")
        self.ylabel_var = tk.StringVar(value="y")

        self.left_container = ttk.Frame(self.main_pane)
        self._build_left_ui(self.left_container)

        self.right_container = ttk.Frame(self.main_pane)
        self._build_right_ui(self.right_container)

        self.main_pane.add(self.left_container, weight=1)
        self.main_pane.add(self.right_container, weight=3)

        # 输出目录：utils/plot/
        self.output_dir = os.path.join(os.path.dirname(__file__), "plot")
        os.makedirs(self.output_dir, exist_ok=True)

        # 预览的最后编号
        self.last_preview_index: Optional[int] = None

    def _build_left_ui(self, parent: tk.Widget) -> None:
        top_bar = ttk.Frame(parent)
        top_bar.pack(fill="x", padx=6, pady=6)
        ttk.Button(top_bar, text="扫描共有编号", command=self.scan_common_indices).pack(side="left", padx=4)
        ttk.Button(top_bar, text="绘制并保存所有共有编号", command=self.plot_and_save_all).pack(side="left", padx=4)
        ttk.Button(top_bar, text="清空画布", command=self.clear_plot).pack(side="left", padx=4)
        ttk.Button(top_bar, text="保存当前预览图片", command=self.save_current_preview).pack(side="left", padx=4)

        self.common_info_var = tk.StringVar(value="共有编号：未扫描")
        ttk.Label(top_bar, textvariable=self.common_info_var).pack(side="left", padx=12)

        # 全局标题与坐标轴名称设置
        settings_bar = ttk.Frame(parent)
        settings_bar.pack(fill="x", padx=6, pady=0)
        ttk.Label(settings_bar, text="标题").pack(side="left", padx=(0, 4))
        title_entry = ttk.Entry(settings_bar, textvariable=self.title_var, width=20)
        title_entry.pack(side="left")
        ttk.Label(settings_bar, text="X轴").pack(side="left", padx=(12, 4))
        xlabel_entry = ttk.Entry(settings_bar, textvariable=self.xlabel_var, width=12)
        xlabel_entry.pack(side="left")
        ttk.Label(settings_bar, text="Y轴").pack(side="left", padx=(12, 4))
        ylabel_entry = ttk.Entry(settings_bar, textvariable=self.ylabel_var, width=12)
        ylabel_entry.pack(side="left")
        # 绑定变化刷新
        def _notify_labels(*_: object) -> None:
            self.refresh_preview()
        self.title_var.trace_add("write", _notify_labels)
        self.xlabel_var.trace_add("write", _notify_labels)
        self.ylabel_var.trace_add("write", _notify_labels)

        # 刻度（比例）选择：linear 或 log（第一行）
        self.x_scale_var = tk.StringVar(value="linear")
        self.y_scale_var = tk.StringVar(value="linear")
        ttk.Label(settings_bar, text="X刻度").pack(side="left", padx=(12, 4))
        xscale_cb = ttk.Combobox(settings_bar, textvariable=self.x_scale_var, values=("linear", "log"), state="readonly", width=8)
        xscale_cb.pack(side="left")
        ttk.Label(settings_bar, text="Y刻度").pack(side="left", padx=(12, 4))
        yscale_cb = ttk.Combobox(settings_bar, textvariable=self.y_scale_var, values=("linear", "log"), state="readonly", width=8)
        yscale_cb.pack(side="left")
        def _notify_scale(*_: object) -> None:
            self.refresh_preview()
        self.x_scale_var.trace_add("write", _notify_scale)
        self.y_scale_var.trace_add("write", _notify_scale)

        # 第二行：轴范围设置（-1 表示自动）
        settings_bar2 = ttk.Frame(parent)
        settings_bar2.pack(fill="x", padx=6, pady=(4, 0))
        self.x_range_min_var = tk.DoubleVar(value=-1.0)
        self.x_range_max_var = tk.DoubleVar(value=-1.0)
        self.y_range_min_var = tk.DoubleVar(value=-1.0)
        self.y_range_max_var = tk.DoubleVar(value=-1.0)

        ttk.Label(settings_bar2, text="X范围").pack(side="left", padx=(0, 4))
        x_min_spin = ttk.Spinbox(settings_bar2, from_=-1.0, to=1e12, increment=0.1, textvariable=self.x_range_min_var, width=10)
        x_min_spin.pack(side="left")
        ttk.Label(settings_bar2, text="到").pack(side="left", padx=(4, 4))
        x_max_spin = ttk.Spinbox(settings_bar2, from_=-1.0, to=1e12, increment=0.1, textvariable=self.x_range_max_var, width=10)
        x_max_spin.pack(side="left")

        ttk.Label(settings_bar2, text="Y范围").pack(side="left", padx=(12, 4))
        y_min_spin = ttk.Spinbox(settings_bar2, from_=-1.0, to=1e12, increment=0.1, textvariable=self.y_range_min_var, width=10)
        y_min_spin.pack(side="left")
        ttk.Label(settings_bar2, text="到").pack(side="left", padx=(4, 4))
        y_max_spin = ttk.Spinbox(settings_bar2, from_=-1.0, to=1e12, increment=0.1, textvariable=self.y_range_max_var, width=10)
        y_max_spin.pack(side="left")

        def _notify_ranges(*_: object) -> None:
            self.refresh_preview()
        self.x_range_min_var.trace_add("write", _notify_ranges)
        self.x_range_max_var.trace_add("write", _notify_ranges)
        self.y_range_min_var.trace_add("write", _notify_ranges)
        self.y_range_max_var.trace_add("write", _notify_ranges)

        # 可滚动的纵向容器
        scroll_container = ttk.Frame(parent)
        scroll_container.pack(fill="both", expand=True, padx=6, pady=6)

        canvas = tk.Canvas(scroll_container, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        inner_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        def _on_inner_configure(event: tk.Event) -> None:
            # 更新滚动区域
            canvas.configure(scrollregion=canvas.bbox("all"))
            # 同步内部宽度到画布宽度，避免出现水平滚动
            canvas_width = canvas.winfo_width()
            canvas.itemconfigure(canvas_window, width=canvas_width)

        inner_frame.bind("<Configure>", _on_inner_configure)

        # 鼠标滚轮支持（Windows）
        def _on_mousewheel(event: tk.Event) -> None:
            delta = int(-1 * (event.delta / 120))
            canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._left_canvas = canvas

        # 10 个 UI 纵向排列
        self.columns: List[CurveColumn] = []
        for i in range(10):
            col = CurveColumn(inner_frame, index=i + 1)
            # 为每列提供“参数应用到全体”的回调
            setattr(col, "apply_params_to_all_cb", self.apply_params_to_all)
            self.columns.append(col)
            col.grid(row=i, column=0, sticky="we", padx=4, pady=4)
            # 绑定自动刷新
            col.bind_on_change(self.on_column_changed)

        inner_frame.columnconfigure(0, weight=1)

    def _build_right_ui(self, parent: tk.Widget) -> None:
        fig = Figure(figsize=FIG_SIZE, dpi=PREVIEW_DPI)
        self.ax = fig.add_subplot(111)
        title, xlabel, ylabel = self._get_labels()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run(self) -> None:
        self.root.mainloop()

    def _collect_selected_configs(self) -> List[CurveConfig]:
        configs: List[CurveConfig] = []
        for col in self.columns:
            cfg = col.get_config()
            if cfg.folder:
                configs.append(cfg)
        return configs

    def on_column_changed(self) -> None:
        # 左侧任一配置变化时自动刷新右侧预览，但不进行保存
        self.refresh_preview()

    def clear_plot(self) -> None:
        self.ax.clear()
        title, xlabel, ylabel = self._get_labels()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self._apply_axis_scale(self.ax)
        self._apply_axis_limits(self.ax)
        self.canvas.draw_idle()

    def apply_params_to_all(self, source_col: CurveColumn) -> None:
        """将源列的线参数应用到其他列（不包含图例名称与线颜色）。"""
        for target in self.columns:
            if target is source_col:
                continue
            # 复制线的参数
            target.linewidth_var.set(float(source_col.linewidth_var.get()))
            target.draw_points_var.set(bool(source_col.draw_points_var.get()))
            target.marker_size_var.set(float(source_col.marker_size_var.get()))
            target.marker_edge_width_var.set(float(source_col.marker_edge_width_var.get()))
            target.start_index_var.set(int(source_col.start_index_var.get()))
            target.end_index_var.set(int(source_col.end_index_var.get()))
        # 应用后刷新预览
        self.refresh_preview()

    def save_current_preview(self) -> None:
        if self.last_preview_index is None:
            messagebox.showinfo("无预览", "当前没有预览内容可保存。请先绘制并保存共有编号后再试。")
            return
        file_path = filedialog.asksaveasfilename(
            title="保存当前预览图片",
            initialfile=f"{self.last_preview_index:06d}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All Files", "*.*")],
        )
        if file_path:
            try:
                self.canvas.figure.savefig(file_path, dpi=SAVE_DPI, bbox_inches="tight")
                messagebox.showinfo("保存成功", f"图片已保存到：\n{file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片失败：\n{e}")

    def _list_indices_in_folder(self, folder: str) -> Set[int]:
        """扫描一个文件夹中同时存在 x_####.npy 与 y_####.npy 的编号集合。"""
        xs: Set[int] = set()
        ys: Set[int] = set()
        try:
            for name in os.listdir(folder):
                m = re.match(r"^(x|y)_(\d+)\.npy$", name)
                if not m:
                    continue
                prefix, digits = m.group(1), m.group(2)
                try:
                    idx = int(digits)
                except ValueError:
                    continue
                if prefix == "x":
                    xs.add(idx)
                else:
                    ys.add(idx)
            return xs & ys
        except Exception as e:
            messagebox.showerror("扫描失败", f"扫描文件夹失败：\n{folder}\n错误：{e}")
            return set()

    def scan_common_indices(self) -> None:
        configs = self._collect_selected_configs()
        if not configs:
            self.common_info_var.set("共有编号：未选择文件夹")
            messagebox.showinfo("提示", "请至少选择一个文件夹。")
            return

        # 交集：所有选中文件夹里同时存在 x/y 的编号
        common: Optional[Set[int]] = None
        details: List[str] = []
        for cfg in configs:
            indices = self._list_indices_in_folder(cfg.folder)
            details.append(f"{os.path.basename(cfg.folder)}: {len(indices)}")
            if common is None:
                common = set(indices)
            else:
                common &= indices

        common = common or set()
        count = len(common)
        self.common_indices_sorted: List[int] = sorted(common)
        self.common_info_var.set(f"共有编号：{count} 个 | " + ", ".join(details))
        if count == 0:
            messagebox.showinfo("结果", "没有发现所有选中文件夹共同拥有的编号。")
        # 扫描后尝试刷新预览
        self.refresh_preview()

    def _load_xy(self, folder: str, index_: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x_path = os.path.join(folder, f"x_{index_:06d}.npy")
        y_path = os.path.join(folder, f"y_{index_:06d}.npy")
        if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
            return None, None
        try:
            x = np.load(x_path, allow_pickle=False)
            y = np.load(y_path, allow_pickle=False)
            return x, y
        except Exception as e:
            messagebox.showerror("读取失败", f"读取失败：\n{x_path}\n{y_path}\n错误：{e}")
            return None, None

    def plot_and_save_all(self) -> None:
        # 若未扫描，先扫描
        if not hasattr(self, "common_indices_sorted"):
            self.scan_common_indices()
        indices = getattr(self, "common_indices_sorted", [])
        configs = self._collect_selected_configs()

        if not indices:
            messagebox.showinfo("提示", "没有共有编号可绘制。请检查已选文件夹是否存在匹配的 x/y 文件。")
            return
        if not configs:
            messagebox.showinfo("提示", "未选择任何文件夹。")
            return

        saved_files: List[str] = []
        last_index: Optional[int] = None

        for idx in indices:
            # 新建一张独立图用于保存（预览DPI用于布局，保存时传入高DPI）
            fig = Figure(figsize=FIG_SIZE, dpi=PREVIEW_DPI)
            ax = fig.add_subplot(111)
            title, xlabel, ylabel = self._get_labels()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)

            plotted_any = False
            for cfg in configs:
                x, y = self._load_xy(cfg.folder, idx)
                if x is None or y is None:
                    # 理论上不会发生，因为已取交集，但仍防御
                    continue
                x_arr = np.asarray(x).flatten()
                y_arr = np.asarray(y).flatten()
                if x_arr.shape[0] != y_arr.shape[0]:
                    # 维度不匹配则跳过该曲线
                    continue
                x_arr, y_arr = self._apply_range(x_arr, y_arr, cfg.start_index, cfg.end_index)
                if x_arr.shape[0] == 0:
                    continue
                marker = "o" if cfg.draw_points else None
                ax.plot(
                    x_arr,
                    y_arr,
                    label=cfg.name,
                    color=cfg.color,
                    linewidth=cfg.linewidth,
                    marker=marker,
                    markersize=cfg.marker_size if cfg.draw_points else None,
                    markeredgewidth=cfg.marker_edge_width if cfg.draw_points else None,
                )
                plotted_any = True

            if plotted_any:
                ax.legend(loc="best", prop={"family": "Arial"})
            # 应用刻度与轴范围设置到保存图像
            self._apply_axis_scale(ax)
            self._apply_axis_limits(ax)
            # 保存到 utils/plot 目录，按编号命名
            out_path = os.path.join(self.output_dir, f"{idx:06d}.png")
            try:
                fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
                saved_files.append(out_path)
                last_index = idx
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片失败（编号 {idx:06d}）：\n{e}")

        # 预览最后一个编号到右侧画布
        if last_index is not None:
            self.preview_index(last_index)
            messagebox.showinfo(
                "完成",
                f"已保存 {len(saved_files)} 张图片到：\n{self.output_dir}\n示例：\n" + ("\n".join(saved_files[:5]) + ("\n..." if len(saved_files) > 5 else "")),
            )

    def preview_index(self, index_: int) -> None:
        self.clear_plot()
        configs = self._collect_selected_configs()
        plotted_any = False
        for cfg in configs:
            x, y = self._load_xy(cfg.folder, index_)
            if x is None or y is None:
                continue
            x_arr = np.asarray(x).flatten()
            y_arr = np.asarray(y).flatten()
            if x_arr.shape[0] != y_arr.shape[0]:
                continue
            x_arr, y_arr = self._apply_range(x_arr, y_arr, cfg.start_index, cfg.end_index)
            if x_arr.shape[0] == 0:
                continue
            marker = "o" if cfg.draw_points else None
            self.ax.plot(
                x_arr,
                y_arr,
                label=cfg.name,
                color=cfg.color,
                linewidth=cfg.linewidth,
                marker=marker,
                markersize=cfg.marker_size if cfg.draw_points else None,
                markeredgewidth=cfg.marker_edge_width if cfg.draw_points else None,
            )
            plotted_any = True

        if plotted_any:
            self.ax.legend(loc="best", prop={"family": "Arial"})
        # 标题与坐标轴名称来自全局设置
        title, xlabel, ylabel = self._get_labels()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        # 应用刻度与轴范围设置
        self._apply_axis_scale(self.ax)
        self._apply_axis_limits(self.ax)
        # 立即绘制以避免需额外 UI 事件（如滚动）才触发刷新
        self.canvas.draw()
        self.last_preview_index = index_

    def refresh_preview(self) -> None:
        # 优先使用现有的预览编号；如果没有，则使用共有编号中的最后一个
        if self.last_preview_index is not None:
            self.preview_index(self.last_preview_index)
        elif hasattr(self, "common_indices_sorted") and self.common_indices_sorted:
            self.preview_index(self.common_indices_sorted[-1])

    def _get_labels(self) -> Tuple[str, str, str]:
        title = (self.title_var.get() or "").strip() or "graph"
        xlabel = (self.xlabel_var.get() or "").strip() or "x"
        ylabel = (self.ylabel_var.get() or "").strip() or "y"
        return title, xlabel, ylabel

    def _get_axis_limits(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """返回 (xmin, xmax, ymin, ymax)，其中 -1 表示自动（返回 None）。"""
        def norm(v: float) -> Optional[float]:
            # 精确等于 -1 时视为自动
            return None if v == -1.0 else float(v)

        xmin = norm(float(self.x_range_min_var.get()))
        xmax = norm(float(self.x_range_max_var.get()))
        ymin = norm(float(self.y_range_min_var.get()))
        ymax = norm(float(self.y_range_max_var.get()))
        return xmin, xmax, ymin, ymax

    def _get_axis_scales(self) -> Tuple[str, str]:
        """返回 x/y 刻度类型（"linear" 或 "log"）。非法值回落为 "linear"。"""
        xscale = (self.x_scale_var.get() or "linear").strip().lower()
        yscale = (self.y_scale_var.get() or "linear").strip().lower()
        if xscale not in ("linear", "log"):
            xscale = "linear"
        if yscale not in ("linear", "log"):
            yscale = "linear"
        return xscale, yscale

    def _apply_axis_scale(self, ax: Axes) -> None:
        """应用 x/y 刻度到坐标轴。若失败则回退线性。"""
        xscale, yscale = self._get_axis_scales()
        try:
            ax.set_xscale(xscale)
        except Exception:
            ax.set_xscale("linear")
        try:
            ax.set_yscale(yscale)
        except Exception:
            ax.set_yscale("linear")

    def _apply_axis_limits(self, ax: Axes) -> None:
        """根据设置应用轴范围；为 None 的边界保持自动缩放。"""
        xmin, xmax, ymin, ymax = self._get_axis_limits()
        xscale, yscale = self._get_axis_scales()
        # log 刻度下需保证边界为正
        if xscale == "log":
            if xmin is not None and xmin <= 0:
                xmin = 1e-12
            if xmax is not None and xmax <= 0:
                xmax = 1.0
        if yscale == "log":
            if ymin is not None and ymin <= 0:
                ymin = 1e-12
            if ymax is not None and ymax <= 0:
                ymax = 1.0
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin if xmin is not None else None, right=xmax if xmax is not None else None)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin if ymin is not None else None, top=ymax if ymax is not None else None)

    def _apply_range(self, x_arr: np.ndarray, y_arr: np.ndarray, start_index: int, end_index: int) -> Tuple[np.ndarray, np.ndarray]:
        n = int(x_arr.shape[0])
        si = 0 if start_index < 0 else start_index
        ei = (n - 1) if end_index < 0 else end_index
        if si < 0:
            si = 0
        if ei >= n:
            ei = n - 1
        if si > ei:
            return x_arr[:0], y_arr[:0]
        return x_arr[si:ei + 1], y_arr[si:ei + 1]


if __name__ == "__main__":
    app = PlotApp()
    app.run()