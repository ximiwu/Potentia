import argparse
import csv
import json
import os
from typing import Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def _require_scalar_loss(session_dir: str, subdir: str, frame_idx: int) -> float:
    """
    Strictly load scalar loss from <session>/<subdir>/loss_XXXXXX.npy.
    Raises FileNotFoundError/ValueError on failure.
    """
    path = os.path.join(session_dir, subdir, f"loss_{frame_idx:06d}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Loss file not found: {path}")
    arr = np.load(path)
    try:
        return float(np.asarray(arr).reshape(()))
    except Exception as e:
        raise ValueError(f"Loss file malformed: {path}") from e


def _require_arrays_for_session(session_dir: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strictly load (dofs, predicted_dofs) for a frame from a session directory.
    Paths: dofs/dofs_XXXXXX.npy and predicted_dofs/predicted_dofs_XXXXXX.npy
    """
    dofs_path = os.path.join(session_dir, "dofs", f"dofs_{frame_idx:06d}.npy")
    pred_path = os.path.join(session_dir, "predicted_dofs", f"predicted_dofs_{frame_idx:06d}.npy")
    if not os.path.isfile(dofs_path):
        raise FileNotFoundError(f"DoFs file not found: {dofs_path}")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Predicted DoFs file not found: {pred_path}")
    dofs = np.load(dofs_path)
    pred = np.load(pred_path)
    return dofs, pred


def analyze_sessions(assessed_dir: str, gt_dir: str, start: int, end: int, out_csv: str, error_mode: str = "baseline", eps: float = 1e-12) -> None:
    """
    Compute per-frame error using two session directories:
    - assessed_dir provides g(x0) from predicted_dofs/loss_*.npy and g(xi) from dofs/loss_*.npy
    - gt_dir provides g(x*) from dofs/loss_*.npy
    Strict mode: any missing file triggers an immediate error.

    error_mode:
      - "baseline": error = (assessed_optimized - gt_optimized) / max(|assessed_init - gt_optimized|, eps)
      - "relative": error = (assessed_optimized - gt_optimized) / max(|gt_optimized|, eps)
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "assessed_init_loss", "gt_init_loss",
            "assessed_optimized_loss", "gt_optimized_loss",
            "error",
            # "assessed_dofs", "assessed_predicted_dofs",
            # "gt_dofs", "gt_predicted_dofs",
        ])

        for i in range(start, end + 1):
            assessed_init_loss = _require_scalar_loss(assessed_dir, "predicted_dofs", i)
            assessed_optimized_loss = _require_scalar_loss(assessed_dir, "dofs", i)
            gt_optimized_loss = _require_scalar_loss(gt_dir, "dofs", i)
            gt_init_loss = _require_scalar_loss(gt_dir, "predicted_dofs", i)

            if error_mode == "baseline":
                denom = assessed_init_loss - gt_optimized_loss
                denom = max(abs(denom), eps)
            elif error_mode == "relative":
                denom = max(abs(gt_optimized_loss), eps)
            else:
                raise ValueError(f"Unknown error_mode: {error_mode}")
            error = (assessed_optimized_loss - gt_optimized_loss) / denom

            a_dofs, a_pred = _require_arrays_for_session(assessed_dir, i)
            g_dofs, g_pred = _require_arrays_for_session(gt_dir, i)

            a_dofs_json = json.dumps(a_dofs.tolist(), separators=(",", ":"))
            a_pred_json = json.dumps(a_pred.tolist(), separators=(",", ":"))
            g_dofs_json = json.dumps(g_dofs.tolist(), separators=(",", ":"))
            g_pred_json = json.dumps(g_pred.tolist(), separators=(",", ":"))

            writer.writerow([
                i,
                assessed_init_loss, gt_init_loss,
                assessed_optimized_loss, gt_optimized_loss,
                error,
                # a_dofs_json, a_pred_json,
                # g_dofs_json, g_pred_json,
            ])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze session losses and export error CSV.")
    parser.add_argument("--assessed", required=True, help="Session dir providing g(x0) from predicted_dofs and g(xi) from dofs")
    parser.add_argument("--gt", required=True, help="Session dir providing g(x*) from dofs")
    parser.add_argument("--start", type=int, required=True, help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End frame index (inclusive)")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    parser.add_argument("--error-mode", choices=["baseline", "relative"], default="baseline",
                        help="Error definition: baseline=(assessed_optimized-gt_optimized)/|assessed_init-gt_optimized|, "
                             "relative=(assessed_optimized-gt_optimized)/|gt_optimized|")
    return parser.parse_args()


def _get_captures_dir() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    cap = os.path.join(root, "captures")
    return cap if os.path.isdir(cap) else os.getcwd()


def _run_gui() -> None:
    root = tk.Tk()
    root.title("Analyze Sessions - Error CSV")

    padx = 8
    pady = 6

    assessed_var = tk.StringVar()
    gt_var = tk.StringVar()
    start_var = tk.StringVar(value="0")
    end_var = tk.StringVar(value="0")
    mode_var = tk.StringVar(value="baseline")

    def browse_dir(target_var: tk.StringVar) -> None:
        init_dir = _get_captures_dir()
        path = filedialog.askdirectory(initialdir=init_dir, title="Select session directory")
        if path:
            target_var.set(path)

    def on_start() -> None:
        assessed = assessed_var.get().strip()
        gt = gt_var.get().strip()
        try:
            start = int(start_var.get().strip())
            end = int(end_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Start/End must be integers")
            return
        if start > end:
            messagebox.showerror("Error", "Start must be <= End")
            return
        if not assessed or not os.path.isdir(assessed):
            messagebox.showerror("Error", "Please select a valid assessed session directory")
            return
        if not gt or not os.path.isdir(gt):
            messagebox.showerror("Error", "Please select a valid ground-truth (gt) session directory")
            return
        out_csv = os.path.join(assessed, "analysis.csv")
        try:
            analyze_sessions(assessed, gt, start, end, out_csv, error_mode=mode_var.get())
            messagebox.showinfo("Done", f"CSV saved to:\n{out_csv}")
        except Exception as e:
            messagebox.showerror("Failed", str(e))

    # Layout
    frm = ttk.Frame(root)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Assessed
    ttk.Label(frm, text="Assessed session (g(x0), g(xi)):").grid(row=0, column=0, sticky="w", padx=padx, pady=pady)
    assessed_entry = ttk.Entry(frm, textvariable=assessed_var, width=64)
    assessed_entry.grid(row=0, column=1, sticky="we", padx=padx, pady=pady)
    ttk.Button(frm, text="Browse...", command=lambda: browse_dir(assessed_var)).grid(row=0, column=2, padx=padx, pady=pady)

    # GT
    ttk.Label(frm, text="GT session (g(x*)):").grid(row=1, column=0, sticky="w", padx=padx, pady=pady)
    gt_entry = ttk.Entry(frm, textvariable=gt_var, width=64)
    gt_entry.grid(row=1, column=1, sticky="we", padx=padx, pady=pady)
    ttk.Button(frm, text="Browse...", command=lambda: browse_dir(gt_var)).grid(row=1, column=2, padx=padx, pady=pady)

    # Start / End
    ttk.Label(frm, text="Start frame:").grid(row=2, column=0, sticky="w", padx=padx, pady=pady)
    start_entry = ttk.Entry(frm, textvariable=start_var, width=12)
    start_entry.grid(row=2, column=1, sticky="w", padx=padx, pady=pady)

    ttk.Label(frm, text="End frame:").grid(row=3, column=0, sticky="w", padx=padx, pady=pady)
    end_entry = ttk.Entry(frm, textvariable=end_var, width=12)
    end_entry.grid(row=3, column=1, sticky="w", padx=padx, pady=pady)

    # Error mode
    ttk.Label(frm, text="Error mode:").grid(row=4, column=0, sticky="w", padx=padx, pady=pady)
    mode_combo = ttk.Combobox(frm, textvariable=mode_var, values=("baseline", "relative"), state="readonly", width=16)
    mode_combo.grid(row=4, column=1, sticky="w", padx=padx, pady=pady)
    mode_combo.current(0)

    # Start button
    ttk.Button(frm, text="Start", command=on_start).grid(row=5, column=0, columnspan=3, pady=12)

    for c in range(3):
        frm.columnconfigure(c, weight=1 if c == 1 else 0)

    # Pre-fill initialdir for convenience
    default_dir = _get_captures_dir()
    assessed_var.set(default_dir)
    gt_var.set(default_dir)

    root.mainloop()


def _maybe_run_cli() -> bool:
    import sys
    # If arguments provided, use CLI; else open GUI
    if len(sys.argv) > 1:
        args = _parse_args()
        analyze_sessions(args.assessed, args.gt, args.start, args.end, args.out, error_mode=args.error_mode)
        return True
    return False


if __name__ == "__main__":
    if not _maybe_run_cli():
        _run_gui()


