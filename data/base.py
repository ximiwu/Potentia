import abc
from typing import List

import taichi as ti


@ti.data_oriented
class ISimulationData(abc.ABC):
    """Stores global, dynamic simulation state."""

    @abc.abstractmethod
    def get_dofs(self) -> ti.Field:
        """Returns global degrees of freedom (e.g., positions)."""
        pass

    @abc.abstractmethod
    def get_velocities(self) -> ti.Field:
        """Returns global velocities."""
        pass

    @abc.abstractmethod
    def get_inv_masses(self) -> ti.Field:
        """Returns global inverse masses."""
        pass

    @abc.abstractmethod
    def get_masses(self) -> ti.Field:
        """Returns global masses."""
        pass

    @abc.abstractmethod
    def get_predicted_dofs(self) -> ti.Field:
        """Returns global predicted degrees of freedom (e.g., predicted positions)."""
        pass

    @abc.abstractmethod
    def get_record_dofs(self) -> ti.Field:
        pass

    @abc.abstractmethod
    def record_predicted_dofs(self):
        pass

    

    @abc.abstractmethod
    def set_predict_dof(self, npy_path: str) -> None:
        """
        从 .npy 文件加载形状为 (N, 3) 的数组并覆盖当前 predicted DoFs 的前 N 项，
        其中 N = get_num_dofs()。实现需进行以下校验：
          - 路径存在并可读
          - 形状严格等于 (N, 3)
          - dtype 可转换为 float32
        任一校验失败应抛出异常。
        """
        pass

    

    @abc.abstractmethod
    def get_num_dofs(self) -> int:
        """Returns the total number of degrees of freedom."""
        pass

    @abc.abstractmethod
    def get_max_num_dofs(self) -> int:
        """Returns the maximum number of degrees of freedom."""
        pass

    @abc.abstractmethod
    def allocate_dofs(self, num_dofs: int) -> int:
        """
        Allocates space for a number of DoFs and returns the starting offset.

        Args:
            num_dofs (int): The number of degrees of freedom to allocate.

        Returns:
            int: The starting index (offset) of the allocated block.
        """
        pass

    @abc.abstractmethod
    def swap_buffers(self) -> None:
        """
        Swaps the roles of the primary DoF buffer and the predicted DoF buffer.
        This is a fast, zero-copy operation used to finalize a time step.
        """
        pass

    @abc.abstractmethod
    def resume_from_record(self, base_dir: str, frame_idx: int) -> None:
        """
        Load DoFs and velocities from a recorded session directory for a given frame,
        and overwrite the first N entries of current primary DoF buffer and velocities,
        where N = get_num_dofs().

        Expected layout inside base_dir:
          - dofs/dofs_{frame_idx:06d}.npy         (shape: [N, 3], dtype: float32 preferred)
          - velocities/velocities_{frame_idx:06d}.npy (shape: [N, 3], dtype: float32 preferred)

        Implementations should validate:
          - files exist
          - shapes are (N, 3) and N matches get_num_dofs()
          - types are convertible to float32

        Any validation failure should raise an exception.
        """
        pass