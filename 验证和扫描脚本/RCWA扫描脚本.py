import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Layer import Layer
from S_matrix.Grating import Sinusoidal,Triangular,Blazed
from S_matrix.Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix
from S_matrix.Layer_mode import layer_mode
from S_matrix.Star import Star
from S_matrix.Build_scatter_side import build_scatter_side
from S_matrix.CalcEffi import calcEffi
from S_matrix.Plot_Effi import Plot_Effi
from openpyxl import load_workbook
from S_matrix.Set_polarization import Set_Polarization
from S_matrix.Slice import Slice
from S_matrix.Compute import Compute

file_path='C:/Users/123/Desktop/三角光栅2微米扫描数据.xlsx'
wb=load_workbook(file_path)
ws=wb.active
start_col=2
start_row=2#行
counter=0
save_interval=5
for fill_factor in np.arange(start=0.7,stop=1,step=0.1):
    for angle in np.arange(start=5,stop=45.5,step=0.5):
        layers=[
            Layer(n=1,t=1*1e-6),
            Layer(n=1.4482+7.5367j,t=2*1e-6,fill_factor=fill_factor),
            Layer(n=1.4482+7.5367j,t=4*1e-6)
            ]
        # grating=Blazed(4*1e-6,angle=angle,fill_factor=fill_factor,n=1)
        grating=Triangular(T=2*1e-6,base_angle=angle,fill_factor=fill_factor)
        Constant=Set_Polarization(thetai=0,phi=0,n1=1,n2=1.4482+7.5367j,wavelength=632.8*1e-9,
        pTE=1,pTM=0,m=20,Nx=2**10,accuracy=1e-9,grating=grating,n=50)

        layers=Slice(layers,grating,Constant)
        Constant=Compute(Constant,layers)
        R3=Constant['R_effi'][0]
        ws.cell(row=start_row,column=start_col,value=R3)
        counter+=1
        start_row+=1
        if counter%save_interval==0:
            wb.save(file_path)
            print(f"已保存{counter}个数据")
    start_col+=1
    start_row=2
wb.save(file_path)