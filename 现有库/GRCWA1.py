import numpy as np
import grcwa
import matplotlib.pyplot as plt

# 定义光栅参数
period = 4.0          # 光栅周期 (μm)
wavelength = 0.6328     # 波长 (μm)
theta = 0.0           # 入射角 (度，0为垂直入射)
pol = 'TE'            # 极化方式 ('TE' 或 'TM')

# 初始化RCWA对象
obj = grcwa.obj(
    nG=11,            # 傅里叶谐波数（越大越精确，但计算量增加）
    L1=[period,0],        # x方向周期
    L2=[0,1e-3],          # y方向周期（设为极小值表示1D光栅）
    freq=1,
    theta=0,
    phi=0
)

Np=1#光栅层数量
Nx=100
Ny=100
thick0=10
thickp=[2,10]
thickn=10
# 材料折射率
ep0 = 1.0
ep_si = 11.7+1e-6j          # 硅的折射率（1550 nm波长）

# 设置材料参数
obj.Add_LayerUniform(thick0,ep0)
for i in range(Np):
    obj.Add_LayerGrid(thickp[i],Nx,Ny)
obj.Add_LayerUniform(thickn,ep0)
obj.Init_Setup()

a=0.5#归一化设置后，指的是占周期的比值
x0=np.linspace(0,1.,Nx)
y0=np.linspace(0,1.,Ny)
x,y=np.meshgrid(x0,y0,indexing='ij')
epgrid1=np.ones((Nx,Ny))*ep0
ind=np.abs(x-.5)<a/2
epgrid1[ind]=ep_si
epgrid2=np.ones((Nx,Ny))*ep_si
epgrid=np.concatenate((epgrid1.flatten(),epgrid2.flatten()))
obj.GridLayer_geteps(epgrid)

planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
# 设置入射波参数
obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order=0)

n_orders=6
# 提取衍射效率
# R, T = obj.RT_Solve(normalize=1)  # 反射和透射效率
# print(f"反射效率: {R}")
# print(f"透射效率: {T}")

# 提取各级衍射效率
orders = range(0,n_orders)
Ri,Ti=obj.RT_Solve(normalize=1,byorder=1)

# 绘制柱状图
plt.figure(figsize=(10, 4))
plt.plot(orders,Ri,label='Reflection')
plt.plot(orders,Ti,label='Transmission')
plt.xlabel('Diffraction Order')
plt.ylabel('Efficiency')
plt.title('Diffraction Efficiency of a 1D Grating')
plt.legend()
plt.show()