import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Set_polarization import Set_Polarization
from S_matrix.Layer import Layer
from S_matrix.Compute import Compute
import matplotlib.pyplot as plt
from S_matrix.Slice import Slice
from S_matrix.Grating import Rectangular
from openpyxl import load_workbook

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
############################设定仿真设备层#################################
layers=[
    Layer(n=1,t=1*1e-6),
    Layer(n=0.77+6.4692j,t=2*1e-6,fill_factor=0.5),
    Layer(n=0.77+6.4692j,t=4*1e-6)
    ]
###########################设定仿真常数################################
grating=Rectangular(T=4*1e-6,fill_factor=0.5,depth=2*1e-6)
# Constant=Set_Polarization(thetai=0,phi=0,wavelength=632.8*1e-9,pTE=1,pTM=0,
#                           m=50,Nx=2**10,accuracy=1e-9,grating=grating,n=20,layers=layers)
########################数据的输出######################################
file_path='C:/Users/123/Desktop/矩形01仿真对比数据.xlsx'
wb=load_workbook(file_path)
ws=wb.active
start_row=2
start_col=6
count=0
save_interval=5
for wavelength in np.arange(300,701,1):
    wavelength=wavelength*1e-9
    Constant=Set_Polarization(thetai=-10,phi=0,wavelength=wavelength,pTE=1,pTM=0,
                            m=100,Nx=2**10,accuracy=1e-9,grating=grating,n=20,layers=layers)
    layers=Slice(layers,grating,Constant)
    Constant=Compute(Constant,layers)
    R_effi=Constant['R_effi']
    temp=np.where(Constant['Ref_set']==0)[0][0]#0级光在R_effi中的位置
    R0=R_effi[temp]
    R1=R_effi[temp-1]
    ws.cell(row=start_row,column=start_col,value=R0)
    ws.cell(row=start_row,column=start_col+1,value=R1)
    count+=2
    if count%save_interval==0:
        wb.save(file_path)
        print(f"已保存{count}个数据")
    start_row+=1
wb.save(file_path)