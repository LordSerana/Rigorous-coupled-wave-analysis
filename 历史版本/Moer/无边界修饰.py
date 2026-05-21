import numpy as np
import cmath  # 用于复数运算
import matplotlib.pyplot as plt
import pandas as pd  # 用于创建和保存表格
import os

def sinc(x):
    """
    定义 sinc 函数
    :param x: 输入值
    :return: sinc(x)
    """
    if x == 0:
        return 1.0  # sinc(0) = 1
    return np.sin(np.pi * x) / (np.pi * x)

def calculate_complex_amplitude(lambda_, z1, d, b, x, n_max):
    """
    计算复振幅分布
    :param lambda_: 波长 λ
    :param z1: 参数 z1
    :param d: 参数 d
    :param b: 参数 b
    :param x: 参数 x
    :param n_max: 求和的上下限，n_max 是正无穷的近似值
    :return: 复振幅分布
    """
    result = 0.0 + 0.0j  # 初始化结果为复数

    # 外层指数项
    outer_exp = cmath.exp(1j * (2 * np.pi / lambda_) * z1)

    # 求和部分
    for n in range(-n_max, n_max + 1):
        # 第一项：(0.867) * (b/d) * sinc(nb/d)
        if n == 0:
            term1 = (0.867) * (b / 1) * 1.0  # sinc(0) = 1
        else:
            term1 = (0.867) * (b / 1) * sinc(n * b / d)

        # 第二项：(0.0324) * (d / (-i2πn)) * [exp(-i2πn + iπnb/d) - exp(-iπnb/d)]
        if n == 0:
            # 当 n = 0 时，第二项的分母为零，需要单独处理
            term2 = (0.0324) * (d-b)
        else:
            term2 = (0.0324) * (d / (-1j * 2 * np.pi * n)) * (
                cmath.exp(-1j * 2 * np.pi * n + 1j * np.pi * n * b / d) -
                cmath.exp(-1j * np.pi * n * b / d)
            )

        # 求和项
        sum_term = (1 / d) * (term1 + term2) * cmath.exp(1j * (2 * np.pi * n * x) / d) * cmath.exp(
            -1j * (np.pi * lambda_ * z1 * n**2) / (d**2)
        )

        result += sum_term

    # 最终复振幅
    final_complex_amplitude = outer_exp * result
    return final_complex_amplitude

def calculate_intensity(lambda_, z1, d, b, x_values, n_max):
    """
    计算光强分布
    :param lambda_: 波长 λ
    :param z1: 参数 z1
    :param d: 参数 d
    :param b: 参数 b
    :param x_values: x 的取值数组
    :param n_max: 求和的上下限，n_max 是正无穷的近似值
    :return: 光强分布数组
    """
    intensity_values = []

    # 遍历 x 值
    for x in x_values:
        # 计算复振幅
        complex_amplitude = calculate_complex_amplitude(lambda_, z1, d, b, x, n_max)
        # 计算光强（复振幅的模的平方）
        intensity = np.abs(complex_amplitude) ** 2
        intensity_values.append(intensity)

    return np.array(intensity_values)  # 将列表转换为 NumPy 数组

def save_to_excel_with_index(df, base_filename):
    """
    保存数据框为 Excel 文件，如果文件名已存在，则在后面添加序号。
    :param df: 要保存的数据框
    :param base_filename: 基础文件名（不带序号）
    :return: 最终保存的文件名
    """
    index = 1
    filename = base_filename  # 初始文件名

    # 检查文件名是否已存在，如果存在则添加序号
    while os.path.exists(filename):
        filename = f"{os.path.splitext(base_filename)[0]}_{index}.xlsx"
        index += 1

    # 保存文件
    df.to_excel(filename, index=False)
    print(f"文件已保存为: {filename}")
    return filename

# 设定参数
lambda_ = 810e-9  # 波长lambda in meters
d = 20e-6        # 参数 d in meters,标尺光栅栅距
b = 10e-6        # 参数 b in meters,指示光栅栅距
z1 = 2 * d**2 / lambda_       # 参数 z1
n_max =55   # 求和的近似上限
x_values = np.linspace(-0.00001, 0.00001, 1000)  # 从 -0.00001 到 0.00001，生成 1000 个点

# 计算光强分布
intensity_values = calculate_intensity(lambda_, z1, d, b, x_values, n_max)

# -----------------*************************------------------
# 创建数据框
data = {
    "x (m)": x_values,
    "Intensity Distribution": intensity_values
}

df = pd.DataFrame(data)

# 保存为 xlsx 文件
save_to_excel_with_index(df, "normalized_intensity_distribution.xlsx")

# -----------------*************************------------------
# 不进行归一化的输出
# 创建灰度图数据
# 将光强分布扩展为二维数组，用于灰度图
intensity_2d = np.tile(intensity_values, (100, 1))  # 重复 100 行

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

# 绘制光强分布曲线
ax1.plot(x_values, intensity_values, label="Intensity Distribution", color='blue')
ax1.set_xlabel("x (m)")
ax1.set_ylabel("Intensity")
ax1.set_title("Light Intensity Distribution")
ax1.legend()
ax1.grid()

# 绘制灰度图
ax2.imshow(intensity_2d, cmap='gray', extent=[x_values.min(), x_values.max(), 0, 1], aspect='auto')
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Intensity")
ax2.set_title("Intensity Grayscale Map")
ax2.yaxis.set_ticks([])  # 隐藏y轴刻度

# 调整布局
plt.tight_layout()
plt.show()


'''
# -----------------*************************------------------
# 归一化光强
# 设置接收面接收到的总光强为1，一共被分成了total_intensity份，计算每个x对应的intensity_values所分到的光强份数
P = 1.0  # 接收面接收到的光的总功率，单位：W
total_intensity = np.sum(intensity_values)  # 计算总光强
intensity_values_normalized = intensity_values * (P / total_intensity) # 归一化


# 创建灰度图数据
# 将光强分布扩展为二维数组，用于灰度图
intensity_2d = np.tile(intensity_values_normalized, (100, 1))  # 重复 100 行

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

# 绘制光强分布曲线
ax1.plot(x_values, intensity_values_normalized, label="Intensity Distribution", color='blue')
ax1.set_xlabel("x (m)")
ax1.set_ylabel("Intensity (W/m²)")
ax1.set_title("normalized_Light Intensity Distribution")
ax1.legend()
ax1.grid()

# 绘制灰度图
ax2.imshow(intensity_2d, cmap='gray', extent=[x_values.min(), x_values.max(), 0, 1], aspect='auto')
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Intensity")
ax2.set_title("Intensity Grayscale Map")
ax2.yaxis.set_ticks([])  # 隐藏y轴刻度

# 调整布局
plt.tight_layout()
plt.show()
'''