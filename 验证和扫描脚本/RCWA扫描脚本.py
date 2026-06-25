import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Layer import Layer
from S_matrix.Grating import Sinusoidal,Triangular,Blazed
from openpyxl import load_workbook
from S_matrix.Set_polarization import Set_Polarization
from S_matrix.Slice import Slice
from S_matrix.Compute import Compute

file_path='C:/Users/123/Desktop/正弦光栅结构优化.xlsx'
wb=load_workbook(file_path)
ws=wb.active
start_col=2
start_row=2#行
counter=0
save_interval=5
for depth in np.arange(start=0.5*1e-6,stop=4.01*1e-6,step=0.01*1e-6):
    layers=[
        Layer(n=1,t=1*1e-6),
        Layer(n=1.4482+7.5367j,t=depth,fill_factor=1),
        Layer(n=1.4482+7.5367j,t=4*1e-6),
        ]
        # grating=Blazed(2*1e-6,angle=angle,fill_factor=fill_factor,n=1)
    # grating=Triangular(T=4*1e-6,base_angle=36,fill_factor=0.9)
    grating=Sinusoidal(4*1e-6,1,depth)
    Constant=Set_Polarization(thetai=0,phi=0,wavelength=632.8*1e-9,pTE=1,pTM=0,m=20,Nx=2**10,accuracy=1e-9,
                              grating=grating,n=50,layers=layers)
    layers=Slice(layers,grating,Constant)
    Constant=Compute(Constant,layers)
    temp=np.where(Constant['Ref_set']==0)[0][0]
    R0=Constant['R_effi'][temp]
    R1=Constant['R_effi'][temp+1]
    ws.cell(row=start_row,column=start_col,value=R0)
    ws.cell(row=start_row,column=start_col+1,value=R1)
    counter+=1
    start_row+=1
    if counter%save_interval==0:
        wb.save(file_path)
        print(f"已保存{counter}个数据")
    # start_col+=1
    # start_row=2
wb.save(file_path)