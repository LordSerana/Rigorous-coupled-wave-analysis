import numpy as np
import scipy.linalg as la
import warnings
from scipy import sparse
from scipy.sparse import linalg as spla

def solve_ill_conditioned_system(GF_matrix, GF_col, method='tsvd', **kwargs):
    """
    解决病态线性系统的稳定方法
    
    参数:
        GF_matrix: 系数矩阵 (m*n)
        GF_col: 右侧向量 (m*1)
        method: 求解方法
            'tsvd': 截断SVD
            'tikhonov': Tikhonov正则化
            'ridge': 岭回归
            'qr': QR分解
            'svd_reg': SVD正则化
            'lasso': Lasso正则化
            'elastic': Elastic Net
        **kwargs: 方法特定的参数
    
    返回:
        Amplitude: 解向量
    """
    m, n = GF_matrix.shape
    
    # 计算原始矩阵的条件数（用于诊断）
    try:
        cond_number = np.linalg.cond(GF_matrix)
        print(f"矩阵条件数: {cond_number:.2e}")
    except:
        print("无法计算条件数（可能太大或矩阵奇异）")
    
    # 方法选择
    if method == 'tsvd':
        return solve_with_tsvd(GF_matrix, GF_col, **kwargs)
    elif method == 'tikhonov':
        return solve_with_tikhonov(GF_matrix, GF_col, **kwargs)
    elif method == 'ridge':
        return solve_with_ridge(GF_matrix, GF_col, **kwargs)
    elif method == 'qr':
        return solve_with_qr(GF_matrix, GF_col, **kwargs)
    elif method == 'svd_reg':
        return solve_with_svd_regularized(GF_matrix, GF_col, **kwargs)
    elif method == 'lasso':
        return solve_with_lasso(GF_matrix, GF_col, **kwargs)
    elif method == 'elastic':
        return solve_with_elastic_net(GF_matrix, GF_col, **kwargs)
    else:
        raise ValueError(f"未知方法: {method}")

def solve_with_tsvd(GF_matrix, GF_col, truncation_ratio=0.01, min_singular_values=5):
    """
    使用截断SVD求解病态系统
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        truncation_ratio: 截断比例（保留奇异值大于最大奇异值的比例）
        min_singular_values: 最小保留的奇异值数量
    
    返回:
        解向量
    """
    # 执行SVD
    U, s, Vt = la.svd(GF_matrix, full_matrices=False)
    
    print(f"奇异值范围: {s  [0]:.2e} 到 {s[-1]:.2e}")
    print(f"条件数（基于奇异值）: {s  [0]/s[-1]:.2e}")
    
    # 确定截断阈值
    max_s = s  [0]
    threshold = max_s * truncation_ratio
    
    # 保留大于阈值的奇异值，但至少保留min_singular_values个
    mask = s>threshold
    num_kept = max(np.sum(mask), min_singular_values)
    
    # 如果保留的数量太少，调整阈值
    if num_kept<min_singular_values:
        # 保留前min_singular_values个最大的奇异值
        mask = np.zeros_like(s, dtype=bool)
        mask[:min_singular_values] = True
        num_kept = min_singular_values
    
    print(f"保留 {num_kept}/{len(s)} 个奇异值")
    print(f"截断阈值: {threshold:.2e}")
    
    # 计算截断的伪逆
    s_inv = np.zeros_like(s)
    s_inv[:num_kept] = 1.0 / s[:num_kept]
    s_inv = s_inv.reshape(-1, 1)
    
    # 计算解
    Amplitude = Vt.T @ (s_inv * (U.T @ GF_col))
    Amplitude = Amplitude.flatten()
    
    # 计算残差
    residual = np.linalg.norm(GF_matrix @ Amplitude - GF_col.flatten())
    print(f"截断SVD残差: {residual:.2e}")
    
    return Amplitude

def solve_with_tikhonov(GF_matrix, GF_col, alpha=1e-10, method='svd'):
    """
    使用Tikhonov正则化(岭回归)求解
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        alpha: 正则化参数
        method: 计算方法 ('svd', 'cholesky', 'lsqr')
    
    返回:
        解向量
    """
    m, n = GF_matrix.shape
    
    if method == 'svd':
        # 使用SVD的Tikhonov正则化
        U, s, Vt = la.svd(GF_matrix, full_matrices=False)
        
        # 计算正则化的奇异值倒数
        s_reg = s / (s**2 + alpha**2)
        s_reg = s_reg.reshape(-1, 1)
        
        Amplitude = Vt.T @ (s_reg * (U.T @ GF_col))
        Amplitude = Amplitude.flatten()
        
    elif method == 'cholesky':
        # 使用Cholesky分解
        # 解决 (A^T A + αI) x = A^T b
        ATA = GF_matrix.T @ GF_matrix
        ATb = GF_matrix.T @ GF_col
        
        # 添加正则化项
        ATA_reg = ATA + alpha * np.eye(n)
        
        # Cholesky分解
        try:
            L = la.cholesky(ATA_reg, lower=True)
            y = la.solve_triangular(L, ATb.flatten(), lower=True)
            Amplitude = la.solve_triangular(L.T, y, lower=False)
        except la.LinAlgError:
            # 如果Cholesky失败，使用LU分解
            Amplitude = la.solve(ATA_reg, ATb.flatten())
    
    elif method == 'lsqr':
        # 使用迭代方法
        Amplitude, istop, itn, r1norm, r2norm = spla.lsqr(
            GF_matrix, GF_col.flatten(),
            damp=alpha,
            atol=1e-10,
            btol=1e-10,
            conlim=1e8,
            iter_lim=1000
        )[0:5]
    
    else:
        raise ValueError(f"未知的Tikhonov方法: {method}")
    
    # 计算残差和正则化项
    residual = np.linalg.norm(GF_matrix @ Amplitude - GF_col.flatten())
    reg_term = alpha * np.linalg.norm(Amplitude)
    print(f"Tikhonov残差: {residual:.2e}, 正则化项: {reg_term:.2e}")
    
    return Amplitude

def solve_with_ridge(GF_matrix, GF_col, alpha=1e-8):
    """
    岭回归(Tikhonov正则化的特例)
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        alpha: 正则化参数
    
    返回:
        解向量
    """
    m, n = GF_matrix.shape
    
    # 岭回归解: (A^T A + αI)^(-1) A^T b
    ATA = GF_matrix.T @ GF_matrix
    ATb = GF_matrix.T @ GF_col
    
    # 添加正则化
    ATA_reg = ATA + alpha * np.eye(n)
    
    # 使用更稳定的求解器
    try:
        # 尝试Cholesky
        L = la.cholesky(ATA_reg, lower=True)
        y = la.solve_triangular(L, ATb.flatten(), lower=True)
        Amplitude = la.solve_triangular(L.T, y, lower=False)
    except la.LinAlgError:
        # 如果失败，使用SVD
        U, s, Vt = la.svd(GF_matrix, full_matrices=False)
        s_reg = s / (s**2 + alpha**2)
        s_reg = s_reg.reshape(-1, 1)
        Amplitude = Vt.T @ (s_reg * (U.T @ GF_col))
        Amplitude = Amplitude.flatten()
    
    return Amplitude

def solve_with_qr(GF_matrix, GF_col, use_pivoting=True):
    """
    使用QR分解求解(带列主元选择)
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        use_pivoting: 是否使用列主元
    
    返回:
        解向量
    """
    if use_pivoting:
        # 带列主元的QR分解
        Q, R, P = la.qr(GF_matrix, pivoting=True, mode='economic')
        
        # 解 R y = Q^T b
        y = la.solve_triangular(R, Q.T @ GF_col)
        
        # 恢复原始顺序
        Amplitude = np.zeros_like(y)
        Amplitude[P] = y.flatten()
    else:
        # 标准QR分解
        Q, R = la.qr(GF_matrix, mode='economic')
        Amplitude = la.solve_triangular(R, Q.T @ GF_col).flatten()
    
    return Amplitude

def solve_with_svd_regularized(GF_matrix, GF_col, alpha=1e-10, beta=1e-12):
    """
    使用SVD和双重正则化
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        alpha: 主正则化参数
        beta: 辅助正则化参数（用于处理极小的奇异值）
    
    返回:
        解向量
    """
    U, s, Vt = la.svd(GF_matrix, full_matrices=False)
    
    # 双重正则化：避免除以极小的数
    s_reg = s / (s**2 + alpha**2 + beta**2 / s**2)
    s_reg = s_reg.reshape(-1, 1)
    
    Amplitude = Vt.T @ (s_reg * (U.T @ GF_col))
    Amplitude = Amplitude.flatten()
    
    return Amplitude

def solve_with_lasso(GF_matrix, GF_col, alpha=1e-6, max_iter=1000):
    """
    使用Lasso正则化(L1正则化)
    注意:对于病态问题,L1正则化可能更稳定
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        alpha: 正则化参数
        max_iter: 最大迭代次数
    
    返回:
        解向量
    """
    from sklearn.linear_model import Lasso
    
    # 使用sklearn的Lasso
    lasso = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=max_iter,
        tol=1e-10,
        selection='random'  # 随机选择可以提高稳定性
    )
    
    lasso.fit(GF_matrix, GF_col.flatten())
    Amplitude = lasso.coef_
    
    # 计算稀疏度
    sparsity = np.sum(np.abs(Amplitude)<1e-10) / len(Amplitude) * 100
    print(f"Lasso解稀疏度: {sparsity:.1f}%")
    
    return Amplitude

def solve_with_elastic_net(GF_matrix, GF_col, alpha=1e-6, l1_ratio=0.5):
    """
    使用Elastic Net正则化(L1和L2混合)
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        alpha: 正则化参数
        l1_ratio: L1正则化比例(0-1)
    
    返回:
        解向量
    """
    from sklearn.linear_model import ElasticNet
    
    elastic = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=5000,
        tol=1e-10
    )
    
    elastic.fit(GF_matrix, GF_col.flatten())
    Amplitude = elastic.coef_
    
    return Amplitude

def iterative_refinement(A, b, x0, max_iter=10, tol=1e-12):
    """
    迭代精化：提高解的精度
    
    参数:
        A: 系数矩阵
        b: 右侧向量
        x0: 初始解
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        精化后的解
    """
    x = x0.copy()
    
    for i in range(max_iter):
        # 计算残差（使用高精度）
        r = b.flatten() - A @ x
        
        # 检查收敛
        norm_r = np.linalg.norm(r)
        if norm_r<tol:
            print(f"迭代精化在 {i+1} 次迭代后收敛")
            break
        
        # 求解修正量
        # 使用更稳定的方法求解修正方程
        try:
            # 尝试使用QR分解
            dx = la.lstsq(A, r)  [0]
        except:
            # 如果失败，使用SVD
            U, s, Vt = la.svd(A, full_matrices=False)
            s_inv = np.zeros_like(s)
            mask = s>1e-10 * s  [0]
            s_inv[mask] = 1.0 / s[mask]
            dx = Vt.T @ (s_inv.reshape(-1, 1) * (U.T @ r))
        
        # 更新解
        x += dx
        
        print(f"迭代 {i+1}: 残差范数 = {norm_r:.2e}")
    
    return x

def precondition_system(GF_matrix, GF_col, method='diagonal'):
    """
    预处理系统以改善条件数
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        method: 预处理方法
            'diagonal': 对角缩放
            'column': 列缩放
            'row': 行缩放
            'svd': SVD预处理
    
    返回:
        预处理后的矩阵和向量
    """
    A = GF_matrix.copy()
    b = GF_col.copy()
    
    if method == 'diagonal':
        # 对角预处理：平衡矩阵的行和列
        # 行缩放
        row_norms = np.linalg.norm(A, axis=1)
        row_norms[row_norms == 0] = 1
        D1 = np.diag(1.0 / row_norms)
        A = D1 @ A
        b = D1 @ b
        
        # 列缩放
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0] = 1
        D2 = np.diag(1.0 / col_norms)
        A = A @ D2
        
        return A, b, D1, D2
        
    elif method == 'column':
        # 列缩放：使每列的范数为1
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0] = 1
        D = np.diag(1.0 / col_norms)
        A = A @ D
        
        return A, b, None, D
        
    elif method == 'row':
        # 行缩放：使每行的范数为1
        row_norms = np.linalg.norm(A, axis=1)
        row_norms[row_norms == 0] = 1
        D = np.diag(1.0 / row_norms)
        A = D @ A
        b = D @ b
        
        return A, b, D, None
        
    elif method == 'svd':
        # SVD预处理
        U, s, Vt = la.svd(A, full_matrices=False)
        
        # 缩放奇异值
        s_scaled = s / s  [0]  # 归一化到[0, 1]
        
        # 构建预处理矩阵
        S_inv = np.diag(1.0 / np.sqrt(s_scaled + 1e-12))
        P1 = U @ S_inv
        P2 = Vt.T @ S_inv
        
        A_precond = P1.T @ A @ P2
        b_precond = P1.T @ b
        
        return A_precond, b_precond, P1, P2
    
    else:
        raise ValueError(f"未知的预处理方法: {method}")

def solve_with_preconditioning(GF_matrix, GF_col, precond_method='diagonal', solver_method='tsvd'):
    """
    使用预处理和求解的组合方法
    
    参数:
        GF_matrix: 系数矩阵
        GF_col: 右侧向量
        precond_method: 预处理方法
        solver_method: 求解方法
    
    返回:
        解向量
    """
    # 预处理
    A_precond, b_precond, P1, P2 = precondition_system(GF_matrix, GF_col, precond_method)
    
    # 计算预处理后的条件数
    try:
        cond_precond = np.linalg.cond(A_precond)
        print(f"预处理后条件数: {cond_precond:.2e}")
    except:
        print("无法计算预处理后的条件数")
    
    # 求解预处理后的系统
    if solver_method == 'tsvd':
        x_precond = solve_with_tsvd(A_precond, b_precond)
    elif solver_method == 'tikhonov':
        x_precond = solve_with_tikhonov(A_precond, b_precond)
    elif solver_method == 'qr':
        x_precond = solve_with_qr(A_precond, b_precond)
    else:
        x_precond = la.lstsq(A_precond, b_precond)  [0]
    
    # 恢复原始解