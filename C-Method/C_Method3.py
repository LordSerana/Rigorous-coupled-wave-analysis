import matplotlib.pyplot as plt
import numpy as np
from Grating import Triangular
from Compute import Compute
'''
采用封装式的代码结构
此文件就是main文件
'''

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

##################设定仿真常数区域##########################################
Constant={}
n1=1
n2=1.3152+7.609j
Constant['n1']=n1
Constant['n2']=n2
Constant['thetai']=np.radians(1e-4)
Constant['n_Tr']=2*40+1
Constant['wavelength']=632.8*1e-9
Constant['gx']=4*1e-6#结构x方向上的周期
#set Accuracy
Constant['cut']=0#是否对变换后的傅里叶级数进行去除小数处理
Constant['accuracy']=1e-9
Constant['kVecImagMin']=1e-10
R_effi=[]
Abs_error=[]
Rela_error=[]
##########################################################################

##########以下为三角光栅常数设定##############
grating=Triangular(4*1e-6,30,1)
a,a_diff=grating.profile()
Constant['a']=a#光栅表面轮廓函数
Constant['diff_a']=a_diff#光栅表面轮廓的导数
#############################################

##########任意偏振态,为TE、TM偏振态的组合###############################
alpha=0
alpha=np.radians(alpha)
a=np.cos(alpha)#TM模式的分量
b=np.sin(alpha)#TE模式的分量
if a!=0:
    Polarization='TM'
    etaR_TM,etaT_TM=Compute(n1,n2,Polarization,Constant)
if b!=0:
    Polarization='TE'
    etaR_TE,etaT_TE=Compute(n1,n2,Polarization,Constant)
if a==0:
    etaR_TM=np.zeros_like(etaR_TE)
if b==0:
    etaR_TE=np.zeros_like(etaR_TM)
polar=a**2*etaR_TM+b**2*etaR_TE
real_Ray1_idx=Constant['real_Ray1_idx']
x=np.linspace(min(real_Ray1_idx),max(real_Ray1_idx),max(real_Ray1_idx)-min(real_Ray1_idx)+1,dtype=int)
plt.plot(x,polar,label='Reflection')
plt.legend()
plt.xlabel("Diffraction order")
plt.ylabel("Diffraction efficiency")
plt.show()
print(polar)
print("sum:"+str(sum(polar)))