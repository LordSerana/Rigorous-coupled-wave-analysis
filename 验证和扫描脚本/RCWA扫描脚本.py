import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Layer import Layer
from S_matrix.Grating import Sinusoidal,Triangular,Blazed
from openpyxl import load_workbook
from S_matrix.Set_polarization import Set_Polarization
from S_matrix.Slice import Slice
from S_matrix.Compute import Compute
from S_matrix.CheckConvergence import CheckConvergence

file_path='C:/Users/123/Desktop/三角光栅倾角对切片数的影响.xlsx'
wb=load_workbook(file_path)
ws=wb.active
start_col=2
start_row=2#行
counter=0
save_interval=5
for angle in np.arange(start=3,stop=50.5,step=0.5):
    # grating=Blazed(1.667*1e-6,angle=11.1,fill_factor=1,n=1)
    Result_last=0
    grating=Triangular(4*1e-6,angle,1)
    for n in range(5,100,1):
        layers=[
            Layer(n=1,t=1*1e-6),
            Layer(n=1.4482+6.3329j,fill_factor=grating.fill_factor),
            Layer(n=1.4482+6.3329j,t=4*1e-6),
            ]
        Constant=Set_Polarization(thetai=0,phi=0,wavelength=632.8*1e-9,pTE=1,pTM=0,m=50,Nx=2**10,accuracy=1e-9,
                                grating=grating,n=n,layers=layers)
        layers=Slice(layers,grating,Constant)
        Constant=Compute(Constant,layers)
        Result_new=Constant['R_effi']
        if n==5:
            Result_last=Result_new
            continue
        else:
            Error=CheckConvergence(Result_last=Result_last,Result_new=Result_new)
            Result_last=Result_new
            print(f"angle={angle},n={n},Error={Error}")
            if Error<0.01:
                ws.cell(row=start_row,column=start_col,value=n)
                break
    # temp=np.where(Constant['Ref_set']==0)[0][0]
    # R1=Constant['R_effi'][temp+1]
    # ws.cell(row=start_row,column=start_col,value=R1)
    counter+=1
    start_row+=1
    if counter%save_interval==0:
        wb.save(file_path)
        print(f"已保存{counter}个数据")
    wb.save(file_path)