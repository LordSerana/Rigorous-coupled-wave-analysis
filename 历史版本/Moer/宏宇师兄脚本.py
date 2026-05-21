import numpy as np
import matplotlib.pyplot as plt

# 参数设置
wavelength = 810e-9  # 波长 (m)
d1 = 20e-6  # 主光栅周期 (m)
a1 = 10e-6  # 主光栅栅齿宽度 (m)
delta_x = 2e-6  # 主光栅位移 (m)
d2 = 20e-6  # 倾斜光栅周期 (m)
a2 = 10e-6  # 倾斜光栅栅齿宽度 (m)
theta = np.deg2rad(5)  # 倾斜角 (弧度)
z0 = 2e-3  # 两光栅之间的距离 (m)
z_T = 2 * d1 ** 2 / wavelength  # 泰伯距离 (m)
k = 2 * np.pi / wavelength  # 波数

# 观察屏坐标范围
x_prime = np.linspace(-200e-6, 200e-6, 500)  # x' 方向 (m)
y_prime = np.linspace(-200e-6, 200e-6, 500)  # y' 方向 (m)
X_prime, Y_prime = np.meshgrid(x_prime, y_prime)  # 网格坐标


# 计算光场 U(x', y')
def calculate_U(x_prime, y_prime, delta_x):
    U = np.zeros_like(X_prime, dtype=complex)  # 初始化光场
    #在观察屏上每一点处计算光场
    # 求和的级数范围（增加级数范围）
    n_max = 1 # 主光栅级数
    m_max = 1  # 倾斜光栅级数
    for n in range(-n_max, n_max + 1):
        for m in range(-m_max, m_max + 1):
            # 计算主光栅的 sinc 函数
            sinc1 = np.sinc(n * a1 / d1)

            # 计算倾斜光栅的 sinc 函数
            sinc2 = np.sinc(m * a2 / d2)

            # 计算位移项的相位
            displacement_phase = np.exp(-1j * 2 * np.pi * n * delta_x / d1)

            # 计算倾斜光栅的相位
            tilt_phase = np.exp(1j * 2 * np.pi * m * (X_prime * np.cos(theta) + Y_prime * np.sin(theta)) / d2)

            # 计算相位项（去掉二次相位项）
            phase = np.exp(1j * (2 * np.pi * n / d1) * X_prime) * \
                    np.exp(1j * k / ( (z_T + z0)) * (X_prime ** 2 + Y_prime ** 2))

            # 累加光场
            # U += (a1 / d1) * (a2 / d2) * sinc1 * sinc2 * displacement_phase * tilt_phase * phase
            U+=(a1/d1)*sinc1*displacement_phase*phase

    # 乘以常数项
    U *= (2 * np.pi) / (1j * wavelength * k)
    return U


# 计算光栅位移前的光场
U_before = calculate_U(X_prime, Y_prime, delta_x=0)  # 位移前
I_before = np.abs(U_before) ** 2

# 计算光栅位移后的光场
U_after = calculate_U(X_prime, Y_prime, delta_x=10e-6)  # 位移后
I_after = np.abs(U_after) ** 2

# 绘制光栅位移前的光场分布图
plt.figure(figsize=(8, 8))
# plt.imshow(I_before, extent=[-200, 200, -200, 200], cmap='gray')
plt.imshow(I_before,cmap='gray')
# plt.colorbar(label="Intensity (a.u.)")
# plt.xlabel("x' (µm)")
# plt.ylabel("y' (µm)")
# plt.title("Intensity Distribution (Before Displacement)")
plt.show()

# 绘制光栅位移后的光场分布图
plt.figure(figsize=(8, 8))
plt.imshow(I_after, extent=[-200, 200, -200, 200], cmap='gray')
plt.colorbar(label="Intensity (a.u.)")
plt.xlabel("x' (µm)")
plt.ylabel("y' (µm)")
plt.title("Intensity Distribution (After Displacement)")
plt.show()

