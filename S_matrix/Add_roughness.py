import numpy as np

def Add_roughness(s, T, Nx, rms, corr_length, seed=None):
    '''
    RMS=1.25*Ra,具有单位
    corr_length=0.05~0.3*T
    '''
    rng = np.random.default_rng(seed)
    # 生成带有一定空间相关性的随机扰动
    if corr_length is None or corr_length <= 0:
        rough = rng.normal(0, rms, Nx)
    else:
        # 使用傅里叶滤波生成高斯相关的粗糙度
        dx = T / Nx
        k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi   # 空间角频率
        S = np.exp(-(k * corr_length / 2.0) ** 2)  # 高斯功率谱
        S[0] = 0  # 去掉均值，保证平均高度不变
        amp = np.sqrt(S)
        phases = rng.uniform(0, 2*np.pi, Nx)
        z = amp * np.exp(1j * phases)
        rough = np.fft.ifft(z).real
        # 归一化到指定RMS
        rough = rough / np.std(rough) * rms
    return s + rough