from typing import Optional, Callable
import numpy as np
import taichi as ti
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

from data.base import ISimulationData
from energies.global_energy_container import GlobalEnergyContainer
from .base import ISolver


@ti.data_oriented
class ScipyNewtonCGSolver(ISolver):
    """
    使用 SciPy Newton-CG（共轭梯度近似牛顿）最小化
    g(x) = 1/2 (x - y)^T M (x - y) + dt^2 E(x)

    - 梯度与赫塞装配与 newton_pcg_solver 一致，直接复用容器与 Taichi kernels
    - 方向与线搜索交给 scipy.optimize.minimize(method='Newton-CG')
    """

    def __init__(
        self,
        data: ISimulationData,
        max_iterations: int = 20,
        hessian_projection: str = "none",
        use_scipy_fd: bool = False,
        float64: bool = False,
    ) -> None:
        self.max_iterations: int = int(max_iterations)
        self.hessian_projection: str = hessian_projection
        self.use_scipy_fd: bool = bool(use_scipy_fd)
        self.use_float64: bool = bool(float64)

        self._container = GlobalEnergyContainer.get_instance()

        capacity = int(data.get_max_num_dofs())
        self._y = data.get_record_dofs()
        self._grad_e = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._grad_total = ti.Vector.field(3, dtype=ti.f32, shape=capacity)

        self._tmp_scalar = ti.field(dtype=ti.f32, shape=())

    def solve(self, data: ISimulationData, dt: float,
              iteration_callback: Optional[Callable[[int, ISimulationData, float], None]] = None) -> None:

        n = int(data.get_num_dofs())
        if n == 0:
            return

        h2 = dt * dt
        dim = 3 * n

        # 初始 x0 = flatten(q_predict)
        x0_nd = ti.ndarray(dtype=ti.f32, shape=(dim,))
        self._flatten_vec3_to_scalars(data.get_predicted_dofs(), n, x0_nd)
        dtype_np = np.float64 if self.use_float64 else np.float32
        x0 = x0_nd.to_numpy().astype(dtype_np)

        # 基于 inv_mass 与阈值 eps 构建自由度掩码
        mask_nd = ti.ndarray(dtype=ti.i32, shape=(dim,))
        eps = np.float32(1e-8)
        self._build_free_mask(data.get_inv_masses(), n, eps, mask_nd)
        free_mask = mask_nd.to_numpy().astype(bool)

        # 写入 q，仅更新自由度（保持固定点不动）
        def write_q_from_flat(x_flat_np: np.ndarray) -> None:
            x32 = np.asarray(x_flat_np, dtype=np.float32)
            x_nd = ti.ndarray(dtype=ti.f32, shape=(dim,))
            x_nd.from_numpy(x32)
            self._write_flat_masked(
                data.get_predicted_dofs(), data.get_inv_masses(), n, eps, x_nd
            )

        # 目标函数：复用容器 loss（与 newton_pcg_solver 一致）
        def fun(x: np.ndarray) -> float:
            write_q_from_flat(x)
            return self._objective(data, dt, self._y)

        # 梯度：G = M(x - y) + h2 ∇E(x)，固定自由度置零
        def jac(x: np.ndarray) -> np.ndarray:
            write_q_from_flat(x)
            self._container.compute_gradient(data, self._grad_e)
            self._compute_inertial_grad(
                data.get_predicted_dofs(), self._y, data.get_masses(), n, self._grad_total
            )
            self._axpy_inplace(self._grad_total, self._grad_e, h2, n)  # grad_total += h2 * grad_e

            g_nd = ti.ndarray(dtype=ti.f32, shape=(dim,))
            self._flatten_vec3_to_scalars(self._grad_total, n, g_nd)
            g = g_nd.to_numpy().astype(dtype_np)
            g[~free_mask] = 0.0
            return g

        # 赫塞乘子：Hp = (M + h2 E''(x)) · p，仅在自由度子空间内
        def hessp(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            write_q_from_flat(x)

            # 组装 Hessian（与 newton_pcg_solver 相同）
            max_blocks_per_c = self._container.v_indices_size * self._container.v_indices_size
            num_constraints = int(self._container.get_num_constraints())
            max_triplets = max(1, num_constraints * 9 * max_blocks_per_c)

            he_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=max_triplets)
            if self.hessian_projection == "abs_eig":
                self._container.compute_hessian_abs_eig(data, he_builder)
            else:
                self._container.compute_hessian(data, he_builder)
            He = he_builder.build()

            mass_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=dim)
            self._fill_mass_diag(data.get_masses(), n, mass_builder)
            M = mass_builder.build()

            H = M + h2 * He

            p_nd = ti.ndarray(dtype=ti.f32, shape=(dim,))
            p_nd.from_numpy(np.asarray(p, dtype=np.float32))
            Hp_nd = H @ p_nd
            Hp = Hp_nd.to_numpy().astype(dtype_np)
            Hp[~free_mask] = 0.0
            return Hp

        # SciPy 的迭代回调：在每次迭代后调用以记录当前 loss
        iter_idx: int = 0
        def scipy_callback(xk: np.ndarray) -> None:
            nonlocal iter_idx
            try:
                # 将当前迭代解写回并计算 loss
                write_q_from_flat(xk)
                loss_val: float = self._objective(data, dt, self._y)
                if iteration_callback is not None:
                    try:
                        iteration_callback(iter_idx, data, dt, loss_val)
                    except Exception as e:
                        print(f"[ScipyNewtonCGSolver] 迭代回调失败: {e}")
            except Exception as e:
                print(f"[ScipyNewtonCGSolver] SciPy 回调执行失败: {e}")
            finally:
                iter_idx += 1

        # 运行 SciPy Newton-CG
        try:
            if self.use_scipy_fd:

                fprime = lambda x: approx_fprime(x, fun, 0.0001)
                res = minimize(
                    fun,
                    x0,
                    method="Newton-CG",
                    jac=fprime,
                    hess='3-point',
                    callback=scipy_callback,
                    options={"xtol": 1e-8, "disp": False, "maxiter": self.max_iterations},
                )
            else:
                # 使用已有的解析梯度与赫塞乘子（推荐）
                res = minimize(
                    fun,
                    x0,
                    method="Newton-CG",
                    jac=jac,
                    hessp=hessp,
                    callback=scipy_callback,
                    options={"xtol": 1e-8, "disp": False, "maxiter": self.max_iterations},
                )
            print("scipy Newton-CG start")
            x_opt = np.asarray(res.x, dtype=np.float32)
            x_opt_nd = ti.ndarray(dtype=ti.f32, shape=(dim,))
            x_opt_nd.from_numpy(x_opt)
            self._write_flat_masked(
                data.get_predicted_dofs(), data.get_inv_masses(), n, eps, x_opt_nd
            )
        except Exception as e:
            print(f"[ScipyNewtonCGSolver] SciPy 优化失败: {e}")

    # ------------------------ kernels & helpers ------------------------

    @ti.kernel
    def _compute_inertial_grad(
        self,
        q: ti.template(),
        y: ti.template(),
        masses: ti.template(),
        n: ti.i32,
        out_grad: ti.template(),
    ):
        for i in range(n):
            m = masses[i]
            if m > 0.0:
                out_grad[i] = m * (q[i] - y[i])
            else:
                out_grad[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _axpy_inplace(self, dst: ti.template(), src: ti.template(), a: ti.f32, n: ti.i32):
        for i in range(n):
            dst[i] += a * src[i]

    @ti.kernel
    def _flatten_vec3_to_scalars(
        self, src: ti.template(), n: ti.i32, out_arr: ti.types.ndarray(dtype=ti.f32, ndim=1)
    ):
        for i in range(n):
            base = 3 * i
            out_arr[base + 0] = src[i][0]
            out_arr[base + 1] = src[i][1]
            out_arr[base + 2] = src[i][2]

    @ti.kernel
    def _fill_mass_diag(
        self, masses: ti.template(), n: ti.i32, builder: ti.types.sparse_matrix_builder()
    ):
        for i in range(n):
            m = masses[i]
            base = 3 * i
            for c in ti.static(range(3)):
                builder[base + c, base + c] += m

    @ti.kernel
    def _write_flat_unmasked(
        self,
        q: ti.template(),
        n: ti.i32,
        x_flat: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        
        for i in range(n):
            base = 3 * i
            q[i][0] = x_flat[base + 0]
            q[i][1] = x_flat[base + 1]
            q[i][2] = x_flat[base + 2]

    @ti.kernel
    def _write_flat_masked(
        self,
        q: ti.template(),
        inv_masses: ti.template(),
        n: ti.i32,
        eps: ti.f32,
        x_flat: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            base = 3 * i
            q[i][0] = x_flat[base + 0]
            q[i][1] = x_flat[base + 1]
            q[i][2] = x_flat[base + 2]

    @ti.kernel
    def _build_free_mask(
        self,
        inv_masses: ti.template(),
        n: ti.i32,
        eps: ti.f32,
        out_mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(n):
            val = 1
            base = 3 * i
            out_mask[base + 0] = val
            out_mask[base + 1] = val
            out_mask[base + 2] = val


    def _objective(self, data: ISimulationData, dt: float, y: ti.Field) -> float:
        return float(self._container.compute_loss(data, data.get_predicted_dofs(), y, dt))