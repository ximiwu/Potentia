# import unittest
# import numpy as np
# import taichi as ti

# from mesh.trimesh import TriMesh, trimesh as _trimesh_mod


# ti.init(arch=ti.cpu)


# class TestTriMeshLocal2DEdges(unittest.TestCase):
#     def test_right_triangle_xy(self):
#         if _trimesh_mod is None:
#             self.skipTest("trimesh not installed; skip TriMesh adjacency-dependent test")

#         vertices = np.array([
#             [0.0, 0.0, 0.0],  # a
#             [1.0, 0.0, 0.0],  # b
#             [0.0, 1.0, 0.0],  # c
#         ], dtype=np.float32)
#         faces = np.array([[0, 1, 2]], dtype=np.int32)
#         edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)

#         mesh = TriMesh(vertices=vertices, faces=faces, edges=edges, materialize=True)
#         local_vec4 = mesh.get_surface_local_edge_2x2().to_numpy()

#         expected = np.array([[-1.0, 0.0, 0.0, -1.0]], dtype=np.float32)
#         np.testing.assert_allclose(local_vec4, expected, rtol=1e-6, atol=1e-6)


# unittest.main()

import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)

# ----------------------------------------------------------------------------
# 这是我们实现的核心函数
# ----------------------------------------------------------------------------
@ti.func
def svd_3x2(A):
    """
    计算一个 3x2 矩阵 A 的 "经济" SVD。
    
    返回:
        U_hat (3x2): 左奇异向量
        S_vec (2x1): 奇异值向量 [s1, s2]
        VT_hat (2x2): 右奇异向量的转置
    """
    
    # 2. 对 3x3 矩阵执行 SVD
    U_full, S_full_vec, VT_full = ti.svd(A)
    
    # 3. 提取 "经济" SVD 的组件
    
    # U_hat 是 U_full 的前两列
    U_hat = ti.Matrix.cols([U_full.get_column(0), U_full.get_column(1)])
    
    # S_vec 是奇异值向量的前两个值
    S_vec = ti.Vector([S_full_vec[0], S_full_vec[1]])
    
    # VT_hat 是 VT_full 的左上角 2x2 子矩阵
    VT_hat = ti.Matrix([
        [VT_full[0, 0], VT_full[0, 1]],
        [VT_full[1, 0], VT_full[1, 1]]
    ])
    
    return U_hat, S_vec, VT_hat

# ----------------------------------------------------------------------------
# 内核：用于测试我们的 svd_3x2 函数
# ----------------------------------------------------------------------------
@ti.kernel
def test_svd_3x2():
    # 定义一个 3x2 矩阵
    A = ti.Matrix([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [5.0, 6.0, 0.0]
    ])
    print(A[2, 1])

    U, S, VT = ti.svd(A)

    print(U@S@VT.transpose())
    
    print("--- Taichi 3x2 SVD (通过 3x3 填充) ---")
    print("A (3x2) =\n", A)
    print("U_hat (3x2) =\n", U)
    print("S_vec (2x1) =\n", S)
    print("VT_hat (2x2) =\n", VT)

# ----------------------------------------------------------------------------
# NumPy 验证
# ----------------------------------------------------------------------------
def numpy_verification():
    A_np = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [5.0, 6.0, 0.0]
    ])
    
    # NumPy 默认执行 "经济" SVD
    U_np, S_np, VT_np = np.linalg.svd(A_np, full_matrices=False)
    
    print("\n--- NumPy (对照组) ---")
    print("U_hat (3x2) =\n", U_np)
    print("S_vec (2x1) =\n", S_np)
    print("VT_hat (2x2) =\n", VT_np)
    
    # 注意：SVD 的 U 和 V 的某些列/行可能会有正负号反转
    # (例如 U 的第一列和 VT 的第一行同时反号)
    # 只要重构的矩阵 A 是正确的，并且奇异值 S 匹配，就说明分解是正确的。
    A_recon_np = U_np @ np.diag(S_np) @ VT_np
    print("Reconstructed A =\n", A_recon_np)


# --- 运行 ---
test_svd_3x2()
numpy_verification()