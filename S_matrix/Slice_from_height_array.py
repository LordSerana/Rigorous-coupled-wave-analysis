import numpy as np

def slice_from_height_array(x, s, T, z_i, z_i1):
    """
    根据高度数组 s 计算第 i 层的 fill_factor 和 offset。
    约定：材料存在于 z > s(x) 的区域。
    """
    # 材料在本层内的法向厚度（沿z方向）：
    # 若 s(x) < z_i   -> 整个切片都是材料，长度 = dz
    # 若 z_i < s(x) < z_i1 -> 材料从 s(x) 到 z_i1，长度 = z_i1 - s(x)
    # 若 s(x) >= z_i1 -> 本层无材料
    length = np.maximum(0, z_i1 - np.maximum(s, z_i))
    
    fill_factor = np.mean(length) / (z_i1 - z_i)
    
    if fill_factor > 0:
        # 计算材料区域的质心，用于确定offset
        x_norm = x / T          # 归一化到 [0,1)
        weights = length
        center = np.sum(x_norm * weights) / np.sum(weights)
        # 通常RCWA中offset定义为相对单元中心的偏移
        offset = center - 0.5
    else:
        offset = 0.0
    
    return fill_factor, offset