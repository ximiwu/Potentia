#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tkinter 小工具：计算 assessed_y 与 gt 的相对误差

支持两种模式：
- init_relative_error: (assessed_y - gt) / (init_y - gt)
- relative_error:     (assessed_y - gt) / gt

gt 来源三选一：
- 手动输入标量
- 从 assessed_y 选择一个 idx 作为标量
- 从 gt_y 文件（.npy，一维，长度与 assessed_y 一致）逐元素作为向量

选项：
- abs：是否对结果取绝对值
- eps：对分母进行钳制，逐元素 denom = max(abs(denom), eps)

保存：
- 将结果保存为 .npy，长度与 assessed_y 一致

注意：
- 分母进行钳制：逐元素将 |denom| 与 eps 取最大值以避免除零。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from typing import Literal, Optional, Union

import numpy as np


def _load_npy_1d(path: str) -> np.ndarray:
    """加载一维 .npy 文件并转换为 float64。

    Args:
        path: 文件路径。
    Returns:
        一维 ndarray，dtype=float64。
    Raises:
        ValueError: 若数组维度不是 1。
    """
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"文件 {path} 的维度为 {arr.ndim}，请提供一维 .npy。")
    return arr.astype(np.float64)


def _clamp_denominator_scalar(denom: float, eps: float) -> float:
    """分母钳制（标量）：返回 max(abs(denom), eps)。"""
    return max(abs(denom), eps)


def _clamp_denominator_array(denom: np.ndarray, eps: float) -> np.ndarray:
    """分母钳制（向量）：逐元素返回 max(abs(denom), eps)。"""
    return np.maximum(np.abs(denom), eps)


def _compute_init_relative_error(
    assessed: np.ndarray,
    gt: Union[float, np.ndarray],
    init_y: float,
    use_abs: bool,
    eps: float,
) -> np.ndarray:
    """计算 (assessed - gt) / (init_y - gt)，带分母钳制。

    Args:
        assessed: 一维评估数组。
        gt: 标量或与 assessed 等长的向量。
        init_y: 标量初始值（来自 assessed 的某个 idx）。
        use_abs: 是否对结果取绝对值。
        eps: 分母钳制下界。
    Returns:
        一维 ndarray 结果。
    """
    numer = assessed - gt
    denom = init_y - gt
    if isinstance(denom, np.ndarray):
        denom_safe = _clamp_denominator_array(denom, eps)
    else:
        denom_safe = _clamp_denominator_scalar(float(denom), eps)
    result = numer / denom_safe
    if use_abs:
        result = np.abs(result)
    return result


def _compute_relative_error(
    assessed: np.ndarray,
    gt: Union[float, np.ndarray],
    use_abs: bool,
    eps: float,
) -> np.ndarray:
    """计算 (assessed - gt) / gt，带分母钳制。

    Args:
        assessed: 一维评估数组。
        gt: 标量或与 assessed 等长的向量。
        use_abs: 是否对结果取绝对值。
        eps: 分母钳制下界。
    Returns:
        一维 ndarray 结果。
    """
    numer = assessed - gt
    denom = gt
    if isinstance(denom, np.ndarray):
        denom_safe = _clamp_denominator_array(denom, eps)
    else:
        denom_safe = _clamp_denominator_scalar(float(denom), eps)
    result = numer / denom_safe
    if use_abs:
        result = np.abs(result)
    return result


def y_analyzer_main() -> None:
    """启动 Tkinter UI。"""
    root = tk.Tk()
    root.title("Y Analyzer")

    # --------------------
    # 状态数据（闭包变量）
    # --------------------
    assessed_y_path: Optional[str] = None
    assessed_y: Optional[np.ndarray] = None

    gt_y_path: Optional[str] = None
    gt_y_arr: Optional[np.ndarray] = None

    # Tk 变量
    abs_var = tk.BooleanVar(value=False)
    mode_var = tk.StringVar(value="init_relative_error")  # 两个模式之一
    init_idx_var = tk.IntVar(value=0)  # init_y 默认取 assessed_y[0]

    # gt 来源：三选一
    gt_source_var = tk.StringVar(value="assessed_idx_scalar")
    gt_manual_var = tk.StringVar(value="0.0")
    gt_idx_var = tk.IntVar(value=0)  # 默认加载后设为最后一个 idx
    eps_var = tk.StringVar(value="1e-12")

    # 结果缓存
    result_array: Optional[np.ndarray] = None

    # --------------------
    # 读取工具（不参与计算）状态
    # --------------------
    reader_y_path: Optional[str] = None
    reader_y_arr: Optional[np.ndarray] = None
    reader_idx_var = tk.IntVar(value=0)
    reader_path_label_var = tk.StringVar(value="读取 y: 未选择")
    reader_info_var = tk.StringVar(value="")
    read_value_var = tk.StringVar(value="")

    # --------------------
    # 工具函数
    # --------------------
    def set_status(msg: str) -> None:
        status_var.set(msg)

    def on_select_assessed_y() -> None:
        nonlocal assessed_y_path, assessed_y
        path = filedialog.askopenfilename(
            title="选择 assessed_y .npy 文件",
            filetypes=[("NumPy 文件", "*.npy"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            arr = _load_npy_1d(path)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载 {path}\n{e}")
            return
        assessed_y_path = path
        assessed_y = arr
        assessed_path_label_var.set(f"assessed_y: {path}")
        assessed_info_var.set(f"长度={arr.size}, dtype={arr.dtype}")

        # 更新 init_idx / gt_idx 范围
        init_idx_spin.config(from_=0, to=max(0, arr.size - 1))
        gt_idx_spin.config(from_=0, to=max(0, arr.size - 1))
        init_idx_var.set(0)
        gt_idx_var.set(max(0, arr.size - 1))  # gt 默认取最大 idx
        set_status("assessed_y 已加载。")

    def on_select_gt_y() -> None:
        nonlocal gt_y_path, gt_y_arr
        path = filedialog.askopenfilename(
            title="选择 gt_y .npy 文件（向量）",
            filetypes=[("NumPy 文件", "*.npy"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            arr = _load_npy_1d(path)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载 {path}\n{e}")
            return
        gt_y_path = path
        gt_y_arr = arr
        gt_path_label_var.set(f"gt_y: {path}")
        gt_info_var.set(f"长度={arr.size}, dtype={arr.dtype}")
        set_status("gt_y 已加载。")

    def current_init_y() -> float:
        if assessed_y is None:
            raise RuntimeError("请先加载 assessed_y 文件。")
        idx = int(init_idx_var.get())
        if idx < 0 or idx >= assessed_y.size:
            raise IndexError(f"init_idx={idx} 超出范围 [0, {assessed_y.size - 1}]")
        return float(assessed_y[idx])

    def current_gt_value_or_vector() -> Union[float, np.ndarray]:
        src = gt_source_var.get()
        if src == "manual_scalar":
            try:
                v = float(gt_manual_var.get().strip())
            except Exception:
                raise ValueError(f"gt 手动输入无法解析为标量：{gt_manual_var.get()}")
            return v
        elif src == "assessed_idx_scalar":
            if assessed_y is None:
                raise RuntimeError("请先加载 assessed_y 文件。")
            idx = int(gt_idx_var.get())
            if idx < 0 or idx >= assessed_y.size:
                raise IndexError(f"gt_idx={idx} 超出范围 [0, {assessed_y.size - 1}]")
            return float(assessed_y[idx])
        elif src == "gt_file_vector":
            if gt_y_arr is None:
                raise RuntimeError("请选择并加载 gt_y 文件。")
            if assessed_y is not None and gt_y_arr.size != assessed_y.size:
                raise ValueError(
                    f"gt_y 长度 {gt_y_arr.size} 与 assessed_y 长度 {assessed_y.size} 不一致。"
                )
            return gt_y_arr
        else:
            raise RuntimeError(f"未知 gt 来源：{src}")

    def recompute() -> None:
        nonlocal result_array
        if assessed_y is None:
            messagebox.showerror("错误", "请先选择 assessed_y 文件。")
            return
        # 解析 eps
        try:
            eps = float(eps_var.get().strip())
        except Exception:
            messagebox.showerror("错误", f"eps 无法解析为浮点数：{eps_var.get()}")
            return
        if eps <= 0:
            messagebox.showwarning("警告", f"eps={eps} 非正，将使用默认 1e-12")
            eps = 1e-12
        try:
            gt = current_gt_value_or_vector()
            mode = mode_var.get()
            init_y = current_init_y()
            if mode == "init_relative_error":
                res = _compute_init_relative_error(assessed_y, gt, init_y, abs_var.get(), eps)
            elif mode == "relative_error":
                res = _compute_relative_error(assessed_y, gt, abs_var.get(), eps)
            else:
                messagebox.showerror("错误", f"未知模式：{mode}")
                return
        except Exception as e:
            messagebox.showerror("计算失败", f"{e}")
            return

        result_array = res
        result_info_var.set(
            f"已计算：模式={mode}, abs={abs_var.get()}, eps={eps}, size={res.size}"
        )
        set_status("计算完成。")

    def save_result() -> None:
        if result_array is None:
            messagebox.showwarning("提示", "请先进行计算，再保存结果。")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果为 .npy",
            defaultextension=".npy",
            filetypes=[("NumPy 文件", "*.npy"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            np.save(path, result_array)
        except Exception as e:
            messagebox.showerror("保存失败", f"{e}")
            return
        set_status(f"已保存结果到 {path}")

    def open_section(section_name: Literal["init", "rel"]) -> None:
        # 折叠区：只展开一个
        if section_name == "init":
            mode_var.set("init_relative_error")
            init_section_content.grid()
            init_section_header.configure(text="▼ init_relative_error")
            rel_section_content.grid_remove()
            rel_section_header.configure(text="▶ relative_error")
        else:
            mode_var.set("relative_error")
            rel_section_content.grid()
            rel_section_header.configure(text="▼ relative_error")
            init_section_content.grid_remove()
            init_section_header.configure(text="▶ init_relative_error")

    def update_gt_source_visibility() -> None:
        src = gt_source_var.get()
        manual_frame.grid_remove()
        assessed_idx_frame.grid_remove()
        gt_file_frame.grid_remove()
        if src == "manual_scalar":
            manual_frame.grid()
        elif src == "assessed_idx_scalar":
            assessed_idx_frame.grid()
        elif src == "gt_file_vector":
            gt_file_frame.grid()

    # 读取工具：选择文件与读取值
    def on_select_reader_y() -> None:
        nonlocal reader_y_path, reader_y_arr
        path = filedialog.askopenfilename(
            title="选择用于读取的 y .npy 文件",
            filetypes=[("NumPy 文件", "*.npy"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            arr = _load_npy_1d(path)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载 {path}\n{e}")
            return
        reader_y_path = path
        reader_y_arr = arr
        reader_path_label_var.set(f"读取 y: {path}")
        reader_info_var.set(f"长度={arr.size}, dtype={arr.dtype}")
        reader_idx_spin.config(from_=0, to=max(0, arr.size - 1))
        reader_idx_var.set(0)
        read_value_var.set("")
        set_status("读取工具：y 已加载。")

    def on_read_value() -> None:
        if reader_y_arr is None:
            messagebox.showerror("错误", "请先在读取工具中选择 y 文件。")
            return
        idx = int(reader_idx_var.get())
        if idx < 0 or idx >= reader_y_arr.size:
            messagebox.showerror("错误", f"读取 idx={idx} 超出范围 [0, {reader_y_arr.size - 1}]")
            return
        val = float(reader_y_arr[idx])
        read_value_var.set(f"值：{val:.6g}")
        set_status("读取工具：已读取指定索引的值。")

    # --------------------
    # 布局
    # --------------------
    container = ttk.Frame(root, padding=10)
    container.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # assessed_y 选择
    assessed_frame = ttk.LabelFrame(container, text="assessed_y（必选，一维 .npy）")
    assessed_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    assessed_frame.columnconfigure(1, weight=1)

    assessed_path_label_var = tk.StringVar(value="assessed_y: 未选择")
    ttk.Label(assessed_frame, textvariable=assessed_path_label_var).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    assessed_info_var = tk.StringVar(value="")
    ttk.Label(assessed_frame, textvariable=assessed_info_var).grid(
        row=1, column=0, columnspan=2, sticky="w"
    )

    select_assessed_btn = ttk.Button(
        assessed_frame, text="选择 assessed_y 文件", command=on_select_assessed_y
    )
    select_assessed_btn.grid(row=2, column=0, sticky="w", padx=2, pady=2)

    ttk.Label(assessed_frame, text="init_y 取 assessed_y 的 idx:").grid(
        row=3, column=0, sticky="w"
    )
    init_idx_spin = ttk.Spinbox(
        assessed_frame, textvariable=init_idx_var, from_=0, to=0, width=8
    )
    init_idx_spin.grid(row=3, column=1, sticky="w")

    # gt 来源
    gt_frame = ttk.LabelFrame(container, text="gt 来源（三选一）")
    gt_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    for i in range(3):
        gt_frame.columnconfigure(i, weight=1)

    rb_manual = ttk.Radiobutton(
        gt_frame,
        text="手动标量",
        value="manual_scalar",
        variable=gt_source_var,
        command=update_gt_source_visibility,
    )
    rb_manual.grid(row=0, column=0, sticky="w")

    rb_assessed_idx = ttk.Radiobutton(
        gt_frame,
        text="从 assessed_y 选 idx（标量）",
        value="assessed_idx_scalar",
        variable=gt_source_var,
        command=update_gt_source_visibility,
    )
    rb_assessed_idx.grid(row=0, column=1, sticky="w")

    rb_gt_file = ttk.Radiobutton(
        gt_frame,
        text="从 gt_y 文件（向量）",
        value="gt_file_vector",
        variable=gt_source_var,
        command=update_gt_source_visibility,
    )
    rb_gt_file.grid(row=0, column=2, sticky="w")

    manual_frame = ttk.Frame(gt_frame)
    manual_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
    ttk.Label(manual_frame, text="gt（标量）：").grid(row=0, column=0, sticky="w")
    gt_entry = ttk.Entry(manual_frame, textvariable=gt_manual_var, width=12)
    gt_entry.grid(row=0, column=1, sticky="w")

    assessed_idx_frame = ttk.Frame(gt_frame)
    assessed_idx_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
    ttk.Label(assessed_idx_frame, text="gt 取 assessed_y 的 idx：").grid(
        row=0, column=0, sticky="w"
    )
    gt_idx_spin = ttk.Spinbox(
        assessed_idx_frame, textvariable=gt_idx_var, from_=0, to=0, width=8
    )
    gt_idx_spin.grid(row=0, column=1, sticky="w")

    gt_file_frame = ttk.Frame(gt_frame)
    gt_file_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
    gt_path_label_var = tk.StringVar(value="gt_y: 未选择")
    ttk.Label(gt_file_frame, textvariable=gt_path_label_var).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    gt_info_var = tk.StringVar(value="")
    ttk.Label(gt_file_frame, textvariable=gt_info_var).grid(
        row=1, column=0, columnspan=2, sticky="w"
    )
    select_gt_btn = ttk.Button(gt_file_frame, text="选择 gt_y 文件", command=on_select_gt_y)
    select_gt_btn.grid(row=2, column=0, sticky="w")

    # 模式折叠栏
    mode_frame = ttk.LabelFrame(container, text="模式选择（折叠，单选）")
    mode_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
    mode_frame.columnconfigure(0, weight=1)

    init_section_header = ttk.Button(
        mode_frame, text="▼ init_relative_error", command=lambda: open_section("init")
    )
    init_section_header.grid(row=0, column=0, sticky="ew")
    init_section_content = ttk.Frame(mode_frame)
    init_section_content.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    ttk.Label(
        init_section_content, text="公式：(assessed_y - gt) / (init_y - gt)"
    ).grid(row=0, column=0, sticky="w")

    rel_section_header = ttk.Button(
        mode_frame, text="▶ relative_error", command=lambda: open_section("rel")
    )
    rel_section_header.grid(row=2, column=0, sticky="ew")
    rel_section_content = ttk.Frame(mode_frame)
    rel_section_content.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
    ttk.Label(rel_section_content, text="公式：(assessed_y - gt) / gt").grid(
        row=0, column=0, sticky="w"
    )
    # 默认展开 init，折叠 rel
    rel_section_content.grid_remove()

    # 选项
    options_frame = ttk.LabelFrame(container, text="选项")
    options_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
    ttk.Checkbutton(options_frame, text="abs（对结果取绝对值）", variable=abs_var).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(options_frame, text="eps：").grid(row=0, column=1, sticky="e")
    eps_entry = ttk.Entry(options_frame, textvariable=eps_var, width=12)
    eps_entry.grid(row=0, column=2, sticky="w")

    # 操作区
    action_frame = ttk.Frame(container)
    action_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
    compute_btn = ttk.Button(action_frame, text="计算", command=recompute)
    compute_btn.grid(row=0, column=0, sticky="w")
    save_btn = ttk.Button(action_frame, text="保存结果为 .npy", command=save_result)
    save_btn.grid(row=0, column=1, sticky="w", padx=5)

    result_info_var = tk.StringVar(value="")
    ttk.Label(container, textvariable=result_info_var).grid(
        row=5, column=0, sticky="w", padx=5
    )
    container.rowconfigure(5, weight=0)

    # 读取工具（不参与计算）
    reader_frame = ttk.LabelFrame(container, text="读取工具（不参与计算）")
    reader_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
    reader_frame.columnconfigure(1, weight=1)

    ttk.Label(reader_frame, textvariable=reader_path_label_var).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(reader_frame, textvariable=reader_info_var).grid(
        row=1, column=0, columnspan=2, sticky="w"
    )
    ttk.Button(reader_frame, text="选择读取用 y 文件", command=on_select_reader_y).grid(
        row=2, column=0, sticky="w", padx=2, pady=2
    )
    ttk.Label(reader_frame, text="读取 idx：").grid(row=3, column=0, sticky="w")
    reader_idx_spin = ttk.Spinbox(
        reader_frame, textvariable=reader_idx_var, from_=0, to=0, width=8
    )
    reader_idx_spin.grid(row=3, column=1, sticky="w")
    ttk.Button(reader_frame, text="读取值", command=on_read_value).grid(
        row=4, column=0, sticky="w"
    )
    ttk.Label(reader_frame, textvariable=read_value_var).grid(
        row=4, column=1, sticky="w"
    )

    # 状态栏
    status_var = tk.StringVar(value="就绪。")
    status_bar = ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w")
    status_bar.grid(row=1, column=0, sticky="ew")

    # 初始可见性
    update_gt_source_visibility()

    root.mainloop()


if __name__ == "__main__":
    y_analyzer_main()