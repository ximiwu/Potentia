from typing import Optional
import taichi as ti
import math

from data.base import ISimulationData
from energies.global_energy_container import GlobalEnergyContainer
from .base import ISolver


@ti.data_oriented
class NewtonPCGSolver(ISolver):


    def __init__(self, data: ISimulationData, max_iterations: int = 20, ordering: str = "AMD") -> None:
        self.max_iterations: int = int(max_iterations)
        self.ordering: str = ordering

        self._container = GlobalEnergyContainer.get_instance()

        capacity = int(data.get_max_num_dofs())
        self._y = data.get_record_dofs()
        self._grad_e = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._grad_total = ti.Vector.field(3, dtype=ti.f32, shape=capacity)

        # 临时标量缓存
        self._tmp_scalar = ti.field(dtype=ti.f32, shape=())

    def solve(self, data: ISimulationData, dt: float) -> None:
        n = int(data.get_num_dofs())
        if n == 0:
            return

        h2 = dt * dt
        dim = 3 * n
        num_constraints = int(self._container.get_num_constraints())

        for iter_idx in range(self.max_iterations):

            print(f"Newton_iter{iter_idx}")

            # 1) 梯度：G = M(x-y) + h^2 E'(x)
            self._container.compute_gradient(data, self._grad_e)
            self._compute_inertial_grad(data.get_predicted_dofs(), self._y, data.get_masses(), n, self._grad_total)
            self._axpy_inplace(self._grad_total, self._grad_e, h2, n)  # grad_total += h2 * grad_e

            grad_inf = self._grad_inf_norm(self._grad_total, n)
            if grad_inf < 1e-12:
                break

            # 2) 赫塞：H = M + h^2 E''(x)
            # 估算 triplets：每约束至多 v_indices_size^2 个 3x3 块，每块 9 个 triplets
            max_blocks_per_c = self._container.v_indices_size * self._container.v_indices_size
            max_triplets = max(1, num_constraints * 9 * max_blocks_per_c)
            he_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=max_triplets)
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

            delta = self._solve_with_pcg(H, b, dim, data.get_masses(), n)

            # 4) 线搜索（Armijo 回溯）
            g_old = self._objective(data, dt, self._y)
            gdotp = float(self._dot_grad_with_delta(self._grad_total, n, delta))

            if gdotp >= 0.0:
                print("gdotp >= 0.0")
                g_flat = ti.ndarray(dtype=ti.f32, shape=(dim,))
                self._flatten_vec3_to_scalars(self._grad_total, n, g_flat)
                diag_inv = ti.ndarray(dtype=ti.f32, shape=(dim,))
                self._fill_jacobi_inv_diag(data.get_masses(), n, 1e-6, diag_inv)
                self._hadamard_scale_nd(delta, g_flat, diag_inv)  # delta = (M+mu I)^{-1} G
                self._scale_nd(delta, -1.0)                       # delta = -(M+mu I)^{-1} G
                gdotp = float(self._dot_grad_with_delta(self._grad_total, n, delta))


            alpha = 1.0
            accepted = False

            for _ls in range(12):
                self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, alpha)
                g_new = self._objective(data, dt, self._y)
                if g_new <= g_old + 1e-4 * alpha * gdotp:
                # if g_new < g_old:
                    accepted = True
                    g_old = g_new
                    break
                # revert
                # print("reject")
                self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, -alpha)
                alpha *= 0.5

            if not accepted:
                # alpha = 2.0 ** -12
                # self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, alpha)
                print("not accepted")
                break

            # 步长很小也可提前结束
            # if float(self._l2_norm(delta) * alpha) < 1e-12:
            #     break

    # ------------------------ kernels & helpers ------------------------


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
            if inv_masses[i] != 0.0:
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
            max_iter = min(200, max(50, int(4 * math.sqrt(max(1, dim)))))

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
        self._fill_const_nd(x, 0.0)
        self._copy_nd(r, b)
        self._hadamard_scale_nd(z, r, diag_inv)  # z = M^{-1} r
        self._copy_nd(p, z)

        rr_old = float(self._dot_nd(r, z))
        b_norm = float(math.sqrt(self._dot_nd(b, b)))
        if b_norm == 0.0:
            return x

        # PCG 主循环
        for iter_idx in range(max_iter):
            print(f"PCG_iter{iter_idx}")
            # Ap = (H + mu I)·p
            Ap = H @ p
            self._axpy_nd(Ap, mu, p)  # Ap += mu * p

            denom = float(self._dot_nd(p, Ap))
            if abs(denom) < 1e-20:
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


