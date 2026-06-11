import numpy as np
import sys
sys.path.append('E:/Project/python')
from S_matrix.Set_polarization import Set_Polarization
from S_matrix.Grating import Rectangular

def Calculate_diffraction_angle(thetai,phi,wavelength,period,n1):
    kinc=n1*np.array([np.sin(thetai)*np.cos(phi),np.sin(thetai)*np.sin(phi),np.cos(thetai)])
    n_Tr=30
    accuracy=1e-10
    mx=np.arange(-n_Tr//2,n_Tr//2+1)
    k0=2*np.pi/wavelength
    kx=np.diag(kinc[0]-2*np.pi*mx/k0/period)
    ky=np.zeros_like(kx)
    temp=n_Tr//2
    kzref=np.zeros((2*temp+1,2*temp+1),dtype=complex)
    for i in range(2*temp+1):
        kzref[i,i]=np.sqrt(n1**2-kx[i,i]**2-ky[i,i]**2,dtype=complex)
    real_mask=np.abs(np.imag(np.diag(kzref)))<accuracy
    real_set=mx[real_mask]
    return real_set

if __name__=='__main__':
    grating=Rectangular(T=4*1e-6,fill_factor=0.5,depth=2*1e-6)
    Constant=Set_Polarization(thetai=-71.66,phi=0,n1=1,n2=1.4482+7.5367j,wavelength=632.8*1e-9,
                              pTE=1,pTM=0,m=20,Nx=2**10,accuracy=1e-9,grating=grating,n=20,Rough=False)
    Ref_set=Constant['Ref_set']
    thetam=np.arange(2*len(Ref_set),dtype=float).reshape((len(Ref_set),2))
    i=0
    for m in Ref_set:
        thetam[i,0]=m
        temp=np.asin(m*Constant['wavelength']/Constant['period']-np.sin(Constant['thetai']))
        temp=np.rad2deg(temp)
        thetam[i,1]=np.round(temp,2)
        i+=1
    print(thetam)