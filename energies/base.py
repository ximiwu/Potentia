import abc
from typing import Any, Dict, Optional, Type, TypeVar

import taichi as ti

from data.base import ISimulationData

T_PE = TypeVar("T_PE", bound="PotentialEnergy")

@ti.data_oriented
class IPotentialEnergy(abc.ABC):
    """
    Abstract base class for a specific type of potential energy.
    This class is responsible for two things:
    1. Defining the computation logic for its energy (energy, gradient, etc.) via *_func methods that operate on a single constraint.
    2. Adding its specific constraint data to a global container via add_constraints.
    It does not store any simulation data itself.
    """

    @classmethod
    @abc.abstractmethod
    def get_type_id(cls) -> int:
        """Returns the unique integer ID for this energy type."""
        pass

    def add_one_constraint_func(self):
        """
        Adds one constraint to the global energy container at a given index.
        This should be implemented by subclasses as a @ti.func.
        The signature is specific to each energy type and must be known by the caller.
        Example for a distance constraint:
        @ti.func
        def add_one_constraint_func(self, constraint_idx: int, p1_idx: int, p2_idx: int, rest_dist: ti.f32, stiffness: ti.f32):
            ...
        """
        pass

    # The following methods are Taichi functions (@ti.func) that will be called
    # from within a Taichi kernel in the global container, dispatched via ti.static.
    # They operate on a single constraint. A concrete class must implement the ones it needs.
    def compute_energy_func(self, constraint: ti.template(), data: ISimulationData) -> ti.f32:
        """
        Computes the energy for a single constraint.
        This should be implemented as a @ti.func.
        """
        return 0.0
    def compute_gradient_func(self, constraint: ti.template(), data: ISimulationData, out_grad: ti.template()):
        """
        Computes the gradient for a single constraint and adds it to the output.
        This should be implemented as a @ti.func.
        """
        pass
    def compute_hessian_func(self, constraint: ti.template(), data: ISimulationData, out_hessian_builder: Any):
        """
        Computes the Hessian for a single constraint and adds it to the output.
        This should be implemented as a @ti.func.
        """
        pass
    def compute_constraint_gradient_func(self, constraint: ti.template(), q: ti.template()):
        """
        Computes the constraint value C and gradient nabla_C for a single constraint.
        Used for constraint-based solvers.
        Should return a tuple containing:
        1. C (ti.f32): The scalar constraint value.
        2. grads (ti.Matrix): A matrix of gradients for its vertices (e.g., 4x3).
        3. num_vertices (int): The number of active vertices for this constraint.
        """
        pass
    def compute_pd_rhs_vec_func(self,
                                constraint: ti.template(),
                                q_predict: ti.template(),
                                vertex_adj_offsets: ti.template(),
                                vertex_adj_indices: ti.template(),
                                vertex_adj_cotan_weights: ti.template(),
                                out_rhs: ti.template()):
        """
        (@ti.func) 为单个约束计算 PD 的局部结果并将其贡献累加到全局右手边向量。

        约定：
        - q_predict: data.get_predicted_dofs()，长度为全局 DoF 数，元素为 vec3。
        - vertex_adj_*: 从 ISimulationData.get_vertex_adjacency() 获取的 CSR 邻接与与其对齐的 cotan 权重。
        - out_rhs: 右手边累加缓冲，推荐为 Struct.field({"x","y","z"})，能量内部对 out_rhs[...] 的 x/y/z 分量使用 atomic_add 原地累加。
        """
        pass

    def compute_pd_lhs_mat_func(self, constraint: ti.template(), data: ISimulationData) -> ti.linalg.SparseMatrix:
        """
        （可选）返回“单个约束”的 PD LHS 稀疏矩阵贡献（ti.linalg.SparseMatrix）。
        容器会遍历所有约束，并按类型分发调用本函数，再将返回矩阵累加。
        """
        raise NotImplementedError


@ti.data_oriented
class PotentialEnergy(IPotentialEnergy, abc.ABC):
    """
    抽象能量基类：统一提供
    - 单例获取 get_instance()
    - 类型 ID 访问 get_type_id()（要求子类定义 TYPE_ID:int）
    - 二次实例化防护（要求子类 __init__ 调用 super().__init__()）
    """

    _instance: Optional["PotentialEnergy"] = None
    TYPE_ID: int = -1  # 子类必须覆盖为非负整数

    @classmethod
    def get_instance(cls: Type[T_PE]) -> T_PE:
        if getattr(cls, "_instance", None) is None:
            cls._instance = cls()  # type: ignore[misc]
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def get_type_id(cls) -> int:
        type_id = getattr(cls, "TYPE_ID", None)
        if not isinstance(type_id, int) or type_id < 0:
            raise RuntimeError(f"{cls.__name__}.TYPE_ID 未定义或非法（需为非负整数）。")
        return type_id

    def __init__(self) -> None:
        # 防止直接多次 new 子类（单例约束）
        if getattr(self.__class__, "_instance", None) is not None:
            raise RuntimeError(f"Error: Attempting to re-instantiate singleton {self.__class__.__name__}.")
        # 这里不要缓存 GlobalEnergyContainer，避免 base 与 container 的循环依赖
        self.__class__._instance = self

@ti.data_oriented
class IGlobalEnergyContainer(abc.ABC):
    """
    Abstract base class for a global container that manages and computes all potential energy terms.
    This interface defines the methods that a solver might require. It operates on the entire set of constraints.
    """

    @abc.abstractmethod
    def compute_energy(self, data: ISimulationData) -> ti.f32:
        """(Optional) Computes the total energy E(q) from all constraints."""
        pass

    @abc.abstractmethod
    def compute_gradient(self, data: ISimulationData, out_grad: ti.Field):
        """(Optional) Computes the total gradient ∇E(q) from all constraints."""
        pass

    @abc.abstractmethod
    def compute_hessian(self, data: ISimulationData, out_hessian_builder: Any):
        """(Optional) Computes the total Hessian ∇²E(q) from all constraints."""
        pass

    @abc.abstractmethod
    def get_num_constraints(self) -> int:
        """(Optional) Returns the total number of active constraints in the container."""
        pass

    @abc.abstractmethod
    def compute_one_constraint_gradient_func(self, i: int, q: ti.template()):
        """
        (Optional, @ti.func) Computes the constraint value C and gradient nabla_C for a single constraint at index i.
        This is intended to be called from within another Taichi kernel (typically from a solver).
        Should return a tuple: (C, grads_matrix, num_vertices).
        """
        pass


    @abc.abstractmethod
    def compute_pd_rhs_init_vec(self, data: ISimulationData, out_vec: ti.Field, dt: float) -> None:
        """
        计算每次迭代不变的 PD 右手边初始化向量：
        out[i] = masses[i] / (dt * dt) * q_predict[i]
        """
        pass

    
    @abc.abstractmethod
    def compute_pd_rhs_vec(self, data: ISimulationData, out_vec: ti.Field, init_vec: ti.Field) -> None:
        """
        以 init_vec 作为起始值填充 out_vec，然后启动 kernel 对各个约束计算局部项并累加到 out_vec。
        """
        pass

    @abc.abstractmethod
    def compute_pd_lhs_mat(self, data: ISimulationData, dt: float) -> ti.linalg.SparseMatrix:
        """
        装配并返回 PD 的 LHS：N×N 标量稀疏矩阵（ti.linalg.SparseMatrix）。
        要求在该方法内写入质量对角 M/dt^2，并遍历所有约束，按约束类型分发到对应能量的
        compute_pd_lhs_mat_func(constraint, data) 获取单约束贡献矩阵并相加。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def register_energy(self, energy: IPotentialEnergy):
        """Registers a potential energy instance, making its compute functions available in kernels."""
        pass

    @abc.abstractmethod
    def reserve_constraints(self, num_to_add: int, is_static: bool = True) -> int:
        """Reserves a block of constraints and returns the starting index."""
        pass

    @abc.abstractmethod
    def clear_constraints(self):
        """Clears all constraints, including static ones."""
        pass

    @abc.abstractmethod
    def clear_dynamic_constraints(self):
        """Clears only the dynamic (temporary) constraints, keeping the static ones."""
        pass

    @classmethod
    @abc.abstractmethod
    def get_instance(self):
        """Clears only the dynamic (temporary) constraints, keeping the static ones."""
        pass
