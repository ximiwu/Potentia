from typing import Optional
import taichi as ti

from data.base import ISimulationData
from energies.global_energy_container import GlobalEnergyContainer
from .base import ISolver


@ti.data_oriented
class NewtonSolver(ISolver):
    """
    牛顿法：最小化 g(x) = 1/2 (x - y)^T M (x - y) + h^2 E(x)
    """

    def __init__(self, data: ISimulationData, max_iterations: int = 20, ordering: str = "AMD") -> None:
        self.max_iterations: int = int(max_iterations)
        self.ordering: str = ordering

        self._container = GlobalEnergyContainer.get_instance()

        capacity = int(data.get_max_num_dofs())
        self._y = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._grad_e = ti.Vector.field(3, dtype=ti.f32, shape=capacity)
        self._grad_total = ti.Vector.field(3, dtype=ti.f32, shape=capacity)

        # 临时标量缓存
        self._tmp_scalar = ti.field(dtype=ti.f32, shape=())

    def solve(self, data: ISimulationData, dt: float) -> None:
        n = int(data.get_num_dofs())
        if n == 0:
            return

        h2 = dt * dt

        # 固定 “惯性参考” y = s_n
        self._copy_vec3(data.get_predicted_dofs(), n, self._y)

        # 预构建一些缓冲
        dim = 3 * n
        num_constraints = int(self._container.get_num_constraints())

        # 牛顿迭代
        for _ in range(self.max_iterations):
            # 1) 梯度：G = M(x-y) + h^2 E'(x)
            self._container.compute_gradient(data, self._grad_e)
            self._compute_inertial_grad(data.get_predicted_dofs(), self._y, data.get_masses(), n, self._grad_total)
            self._axpy_inplace(self._grad_total, self._grad_e, h2, n)  # grad_total += h2 * grad_e

            grad_inf = self._grad_inf_norm(self._grad_total, n)
            if grad_inf < 1e-5:
                break

            # 2) 赫塞：H = M + h^2 E''(x)
            # 估算 triplets：每约束至多 v_indices_size^2 个 3x3 块，每块 9 个 triplets
            max_blocks_per_c = self._container.v_indices_size * self._container.v_indices_size
            max_triplets = max(1, num_constraints * 9 * max_blocks_per_c)
            he_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=max_triplets)
            self._container.compute_hessian(data, he_builder)
            He = he_builder.build()

            mass_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=dim)
            self._fill_mass_diag(data.get_masses(), n, mass_builder)
            M = mass_builder.build()

            H = M + (He * h2)

            # 3) 方向：解 H Δ = -G，对角修正 + LLT
            b = ti.ndarray(dtype=ti.f32, shape=(dim,))
            self._flatten_vec3_to_scalars(self._grad_total, n, b)
            self._scale_array_(b, -1.0)  # b = -G

            delta = self._solve_with_correction(H, b, dim)

            # 4) 线搜索（Armijo 回溯）
            g_old = self._objective(data, dt, self._y)
            gdotp = float(self._dot_grad_with_delta(self._grad_total, n, delta))
            alpha = 1.0
            accepted = False
            for _ls in range(12):
                self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, alpha)
                g_new = self._objective(data, dt, self._y)
                if g_new <= g_old + 1e-4 * alpha * gdotp:
                    accepted = True
                    g_old = g_new
                    break
                # revert
                self._apply_step(data.get_predicted_dofs(), data.get_inv_masses(), n, delta, -alpha)
                alpha *= 0.5

            if not accepted:
                # 无法接受任何步长，结束
                break

            # 步长很小也可提前结束
            if float(self._l2_norm(delta)) < 1e-8:
                break

    # ------------------------ kernels & helpers ------------------------

    @ti.kernel
    def _copy_vec3(self, src: ti.template(), n: ti.i32, dst: ti.template()):
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

    def _solve_with_correction(self,
                               H: ti.linalg.SparseMatrix,
                               b: ti.types.ndarray(dtype=ti.f32, ndim=1),
                               dim: int) -> Optional[ti.types.ndarray]:
        mu = 1e-6
        diag_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=dim)
        self._fill_diag(diag_builder, dim, mu)
        H_mu = H + diag_builder.build()

        solver = ti.linalg.SparseSolver(solver_type="LLT", ordering=self.ordering)
        solver.analyze_pattern(H_mu)
        solver.factorize(H_mu)
        x = solver.solve(b)
        return x


    @ti.kernel
    def _fill_diag(self, builder: ti.types.sparse_matrix_builder(), dim: ti.i32, mu: ti.f32):
        for i in range(dim):
            builder[i, i] += mu

    def _objective(self, data: ISimulationData, dt: float, y: ti.Field) -> float:
        h2 = dt * dt
        e = float(self._container.compute_energy(data))
        return 0.5 * float(self._inertial_quadratic(data.get_predicted_dofs(), y, data.get_masses(), int(data.get_num_dofs()))) + h2 * e

    @ti.kernel
    def _inertial_quadratic(self,
                            q: ti.template(),
                            y: ti.template(),
                            masses: ti.template(),
                            n: ti.i32) -> ti.f32:
        acc = 0.0
        for i in range(n):
            m = masses[i]
            if m > 0.0:
                d = q[i] - y[i]
                acc += m * d.dot(d)
        return acc

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


