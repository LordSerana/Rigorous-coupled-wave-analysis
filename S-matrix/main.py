import numpy as np
from Set_polarization import Set_Polarization
from Layer import Layer

###########################设定仿真常数################################
thetai=np.radians(0)#入射角thetai
phi=np.radians(0)#入射角phi
wavelength=632.8*1e-9
n1=1
n2=1.4482+7.5367j
pTM=0
pTE=1
Constant=Set_Polarization(thetai,phi,wavelength,n1,pTM,pTE)
m=15
Constant['n_Tr']=2*m+1
Constant['mx']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
Constant['my']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
Constant['period']=4*1e-6
Constant['Nx']=2**10
Constant['n2']=n2
Constant['accuracy']=1e-9
Constant['error']=0.001#相对误差
R_effi=[]
Abs_error=[]
Rela_error=[]
############################设定仿真设备层#################################
layers=[
    Layer(n=1,t=1*1e-6),
    Layer(n=1.4482+7.5367j,t=1.8*1e-6,fill_factor=0.5),
    Layer(n=1.4482+7.5367j,t=4*1e-6)
    ]