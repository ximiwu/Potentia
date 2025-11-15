from typing import Optional, Callable
import taichi as ti
import math

from data.base import ISimulationData
from energies.global_energy_container import GlobalEnergyContainer
from .base import ISolver


@ti.data_oriented
class NewtonPCGSolver(ISolver):


    def __init__(self, 
                 data: ISimulationData, 
                 max_iterations: int = 20, 
                 hessian_projection: str = "none") -> None:
        self.max_iterations: int = int(max_iterations)
        self.hessian_projection: str = hessian_projection

        self._container = GlobalEnergyContainer.get_instance()

        capacity = int(data.get_max_num_dofs())
        self._y = data.get_record_dofs()
        self._grad_e = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._grad_total = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._q_backup = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._delta_prev: Optional[ti.types.ndarray] = None

        # 临时标量缓存
        self._tmp_scalar = ti.field(dtype=ti.f32, shape=())

    def solve(self, data: ISimulationData, dt: float,
              iteration_callback: Optional[Callable[[int, ISimulationData, float], None]] = None) -> None:
        n = int(data.get_num_dofs())
        if n == 0:
            return

        h2 = dt * dt
        dim = 3 * n
        if self._delta_prev is None or self._delta_prev.shape[0] != dim:
            self._delta_prev = ti.ndarray(dtype=ti.f32, shape=(dim,))
            self._fill_const_nd(self._delta_prev, 0.0)

        num_constraints = int(self._container.get_num_constraints())


        history_loss = self._objective(data, dt, self._y)
        g_old = history_loss
        g_new = history_loss
        g0_norm = 0.0

        for iter_idx in range(self.max_iterations):
            g_old = g_new
            # print(f"Newton_iter{iter_idx}")
            if iter_idx % 50 == 0:
                # print(f"Newton_iter{iter_idx}")
                print(f"history_loss: {history_loss}")
                print(f"delta_loss: {abs(history_loss - g_new)}")
                if iter_idx != 0:
                    if abs(history_loss - g_new) < 1e-8:
                        break
                    history_loss = g_new

            # 1) 梯度：G = M(x-y) + h^2 E'(x)
            self._container.compute_gradient(data, self._grad_e)
            self._compute_inertial_grad(data.get_predicted_dofs(), self._y, data.get_masses(), n, self._grad_total)
            self._axpy_inplace(self._grad_total, self._grad_e, h2, n)  # grad_total += h2 * grad_e

            gk_norm = self._vec_field_l2_norm(self._grad_total, n)
            if iter_idx == 0:
                g0_norm = gk_norm

            grad_inf = self._grad_inf_norm(self._grad_total, n)
            if grad_inf < 1e-12:
                break

            # Inexact Newton: 动态计算 PCG 容差 (Eisenstat–Walker)
            eta_max = 0.5
            eta_min = 1e-5
            c = 0.5
            alpha_ew = 1
            eta_k = min(eta_max, max(eta_min, c * (gk_norm / (g0_norm + 1e-9)) ** alpha_ew))

            # 2) 赫塞：H = M + h^2 E''(x)
            # 估算 triplets：每约束至多 v_indices_size^2 个 3x3 块，每块 9 个 triplets
            max_blocks_per_c = self._container.v_indices_size * self._container.v_indices_size
            max_triplets = max(1, num_constraints * 9 * max_blocks_per_c)
            he_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=max_triplets)
            if self.hessian_projection == "abs_eig":
                self._container.compute_hessian_abs_eig(data, he_builder)
            else:
                self._container.compute_hessian(data, he_builder)
            He = he_builder.build()  # E''(x)
            # 构建质量矩阵 M 的稀疏对角
            mass_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=dim)
            self._fill_mass_diag(data.get_masses(), n, mass_builder)
            M = mass_builder.build()
            # 显式组合：H = M + h^2 E''(x)
            H = M + h2 * He

            # 3) 方向：PCG 解 H Δ = -G
            b = ti.ndarray(dtype=ti.f32, shape=(dim,))
            self._flatten_vec3_to_scalars(self._grad_total, n, b)
            self._scale_array_(b, -1.0)  # b = -G

            delta = self._solve_with_pcg(H, b, dim, data.get_masses(), n, rtol=eta_k)
            self._copy_nd(self._delta_prev, delta)  # 保存解用于下一次热启动

            # 4) 线搜索（Armijo 回溯）
            gdotp = float(self._dot_grad_with_delta(self._grad_total, n, delta))
            sd_ready = False

            if gdotp >= 0.0:
                print("gdotp >= 0.0，退化为梯度下降")
                gdotp = self._make_sd_direction(data, n, dim, 1e-6, delta)
                sd_ready = True
                if gdotp >= 0.0:
                    if iteration_callback is not None:
                        iteration_callback(iter_idx, data, dt, g_old)
                    print("警告：梯度下降法仍然不是下降方向")
                    break


            alpha = 1.0
            accepted = False
            self._copy_vec3_field(self._q_backup, data.get_predicted_dofs(), n)

            for _ls in range(12):
                self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, alpha)
                g_new = self._objective(data, dt, self._y)
                if g_new <= g_old + 1e-8 * alpha * gdotp:
                # if g_new < g_old:
                    accepted = True
                    break
                # revert
                self._copy_vec3_field(data.get_predicted_dofs(), self._q_backup, n)
                alpha *= 0.5


            if not accepted:
                # 退化到梯度下降（Jacobi 预条件的最速下降），并进行回溯线搜索
                print("Newton 步长未接受，退化到梯度下降")
                if not sd_ready:
                    gdotp = self._make_sd_direction(data, n, dim, 1e-6, delta)
                else:
                    gdotp = float(self._dot_grad_with_delta(self._grad_total, n, delta))
                if gdotp >= 0.0:
                    print("警告：梯度下降法仍然不是下降方向")
                    if iteration_callback is not None:
                        iteration_callback(iter_idx, data, dt, g_old)
                    break

                alpha_sd = 1.0
                accepted_sd = False
                self._copy_vec3_field(self._q_backup, data.get_predicted_dofs(), n)
                for _ls_sd in range(12):
                    self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, alpha_sd)
                    g_new = self._objective(data, dt, self._y)
                    if g_new <= g_old + 1e-8 * alpha_sd * gdotp:
                        accepted_sd = True
                        break
                    # revert
                    self._copy_vec3_field(data.get_predicted_dofs(), self._q_backup, n)
                    alpha_sd *= 0.5



                if not accepted_sd:
                    if iteration_callback is not None:
                        iteration_callback(iter_idx, data, dt, g_old)
                    print("梯度下降也未接受，停止迭代")
                    break

            if iteration_callback is not None:
                iteration_callback(iter_idx, data, dt, g_new)
            # 步长很小也可提前结束
            # if float(self._l2_norm(delta) * alpha) < 1e-12:
            #     break
        
    def _make_sd_direction(self,
                           data: ISimulationData,
                           n: int,
                           dim: int,
                           mu: float,
                           delta: ti.types.ndarray(dtype=ti.f32, ndim=1)) -> float:
        g_flat = ti.ndarray(dtype=ti.f32, shape=(dim,))
        self._flatten_vec3_to_scalars(self._grad_total, n, g_flat)
        diag_inv = ti.ndarray(dtype=ti.f32, shape=(dim,))
        self._fill_jacobi_inv_diag(data.get_masses(), n, mu, diag_inv)
        self._hadamard_scale_nd(delta, g_flat, diag_inv)
        self._scale_nd(delta, -1.0)
        return float(self._dot_grad_with_delta(self._grad_total, n, delta))
    # ------------------------ kernels & helpers ------------------------


    @ti.kernel
    def _copy_vec3_field(self,
                         dst: ti.template(),
                         src: ti.template(),
                         n: ti.i32):
        for i in range(n):
            dst[i] = src[i]

    @ti.kernel
    def _compute_inertial_grad(self,
                               q: ti.template(),
                               y: ti.template(),
                               masses: ti.template(),
                               n: ti.i32,
                               out_grad: ti.template()):
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
    def _flatten_vec3_to_scalars(self, src: ti.template(), n: ti.i32, out_arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            base = 3 * i
            out_arr[base + 0] = src[i][0]
            out_arr[base + 1] = src[i][1]
            out_arr[base + 2] = src[i][2]

    @ti.kernel
    def _scale_array_(self, arr: ti.types.ndarray(dtype=ti.f32, ndim=1), s: ti.f32):
        for i in range(arr.shape[0]):
            arr[i] = s * arr[i]

    @ti.kernel
    def _fill_mass_diag(self,
                        masses: ti.template(),
                        n: ti.i32,
                        builder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            m = masses[i]
            base = 3 * i
            for c in ti.static(range(3)):
                builder[base + c, base + c] += m

    @ti.kernel
    def _apply_step(self,
                    q: ti.template(),
                    inv_masses: ti.template(),
                    n: ti.i32,
                    delta: ti.types.ndarray(dtype=ti.f32, ndim=1),
                    alpha: ti.f32):
        for i in range(n):
            # if inv_masses[i] >= 1e-6:
            base = 3 * i
            q[i][0] += alpha * delta[base + 0]
            q[i][1] += alpha * delta[base + 1]
            q[i][2] += alpha * delta[base + 2]

    @ti.kernel
    def _grad_inf_norm(self, g: ti.template(), n: ti.i32) -> ti.f32:
        mx = 0.0
        for i in range(n):
            for c in ti.static(range(3)):
                val = ti.abs(g[i][c])
                if val > mx:
                    mx = val
        return mx

    @ti.kernel
    def _l2_norm(self, arr: ti.types.ndarray(dtype=ti.f32, ndim=1)) -> ti.f32:
        acc = 0.0
        for i in range(arr.shape[0]):
            v = arr[i]
            acc += v * v
        return ti.sqrt(acc)


    @ti.kernel
    def _vec_field_l2_norm(self, g: ti.template(), n: ti.i32) -> ti.f32:
        acc = 0.0
        for i in range(n):
            acc += g[i].dot(g[i])
        return ti.sqrt(acc)


    def _objective(self, data: ISimulationData, dt: float, y: ti.Field) -> float:
        return float(self._container.compute_loss(data, data.get_predicted_dofs(), y, dt))


    @ti.kernel
    def _dot_grad_with_delta(self,
                              g: ti.template(),
                              n: ti.i32,
                              delta: ti.types.ndarray(dtype=ti.f32, ndim=1)) -> ti.f32:
        acc = 0.0
        for i in range(n):
            base = 3 * i
            acc += g[i][0] * delta[base + 0]
            acc += g[i][1] * delta[base + 1]
            acc += g[i][2] * delta[base + 2]
        return acc



    # ------------------------ PCG helpers ------------------------

    def _solve_with_pcg(self,
                        H: ti.linalg.SparseMatrix,
                        b: ti.types.ndarray(dtype=ti.f32, ndim=1),
                        dim: int,
                        masses: ti.Field,
                        n: int,
                        mu: float = 1e-6,
                        max_iter: Optional[int] = None,
                        rtol: float = 1e-6) -> ti.types.ndarray:
        if max_iter is None:
            max_iter = 200

        # 分配工作向量
        x = ti.ndarray(dtype=ti.f32, shape=(dim,))
        r = ti.ndarray(dtype=ti.f32, shape=(dim,))
        p = ti.ndarray(dtype=ti.f32, shape=(dim,))
        z = ti.ndarray(dtype=ti.f32, shape=(dim,))
        Ap = ti.ndarray(dtype=ti.f32, shape=(dim,))
        diag_inv = ti.ndarray(dtype=ti.f32, shape=(dim,))

        # 预条件：Jacobi (diag(M)+mu)^{-1}
        self._fill_jacobi_inv_diag(masses, n, mu, diag_inv)

        # 初始化
        self._copy_nd(x, self._delta_prev)  # 热启动
        # r = b - Ax
        r_init = H @ x
        self._axpy_nd(r_init, mu, x)
        self._copy_nd(r, b)
        self._axpy_nd(r, -1.0, r_init)
        self._hadamard_scale_nd(z, r, diag_inv)  # z = M^{-1} r
        self._copy_nd(p, z)

        rr_old = float(self._dot_nd(r, z))
        b_norm = float(math.sqrt(self._dot_nd(b, b)))
        if b_norm == 0.0:
            return x

        # PCG 主循环
        for iter_idx in range(max_iter):
            # if iter_idx % 5 == 0:
            #     print(f"PCG_iter{iter_idx}")
            Ap = H @ p
            self._axpy_nd(Ap, mu, p)  # Ap += mu * p

            denom = float(self._dot_nd(p, Ap))
            if abs(denom) < 1e-7:
                break

            alpha = rr_old / denom

            self._axpy_nd(x, alpha, p)      # x += alpha p
            self._axpy_nd(r, -alpha, Ap)    # r -= alpha Ap

            rel = float(math.sqrt(self._dot_nd(r, r))) / b_norm
            if rel < rtol:
                break

            self._hadamard_scale_nd(z, r, diag_inv)  # z = M^{-1} r
            rr_new = float(self._dot_nd(r, z))
            if rr_old == 0.0:
                break
            if abs(rr_new - rr_old) / rr_old < 1e-3:
                break
            beta = rr_new / rr_old
            rr_old = rr_new

            self._scale_nd(p, beta)
            self._axpy_nd(p, 1.0, z)  # p = z + beta p

        return x

    @ti.kernel
    def _fill_jacobi_inv_diag(self,
                              masses: ti.template(),
                              n: ti.i32,
                              mu: ti.f32,
                              out_diag_inv: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            base = 3 * i
            v = masses[i] + mu
            inv = 1.0 / v
            out_diag_inv[base + 0] = inv
            out_diag_inv[base + 1] = inv
            out_diag_inv[base + 2] = inv


    @ti.kernel
    def _dot_nd(self,
                a: ti.types.ndarray(dtype=ti.f32, ndim=1),
                b: ti.types.ndarray(dtype=ti.f32, ndim=1)) -> ti.f32:
        acc = 0.0
        for i in range(a.shape[0]):
            acc += a[i] * b[i]
        return acc

    @ti.kernel
    def _axpy_nd(self,
                 y: ti.types.ndarray(dtype=ti.f32, ndim=1),
                 a: ti.f32,
                 x: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(y.shape[0]):
            y[i] += a * x[i]

    @ti.kernel
    def _copy_nd(self,
                 dst: ti.types.ndarray(dtype=ti.f32, ndim=1),
                 src: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(dst.shape[0]):
            dst[i] = src[i]

    @ti.kernel
    def _scale_nd(self,
                  x: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  s: ti.f32):
        for i in range(x.shape[0]):
            x[i] = s * x[i]

    @ti.kernel
    def _fill_const_nd(self,
                       x: ti.types.ndarray(dtype=ti.f32, ndim=1),
                       v: ti.f32):
        for i in range(x.shape[0]):
            x[i] = v

    @ti.kernel
    def _hadamard_scale_nd(self,
                           dst: ti.types.ndarray(dtype=ti.f32, ndim=1),
                           src: ti.types.ndarray(dtype=ti.f32, ndim=1),
                           weights: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(dst.shape[0]):
            dst[i] = src[i] * weights[i]


