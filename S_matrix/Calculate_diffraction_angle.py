import numpy as np
import sys
sys.path.append('E:/Project/python')
from S_matrix.Set_polarization import Set_Polarization

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
    thetai=-71.66
    Constant={}
    Constant['n1']=1
    Constant=Set_Polarization(thetai,0,632.8*1e-9,0,1,Constant)
    Constant['period']=4*1e-6
    Constant['n_Tr']=2*30+1
    Constant['accuracy']=1e-10
    Constant['mx']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
    kinc=Constant['kinc']
    kx=np.diag(kinc[0]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])
    temp=Constant['n_Tr']//2
    ky=np.zeros((2*temp+1,2*temp+1))
    Constant['ky']=ky
    kzref=np.zeros((2*temp+1,2*temp+1),dtype=complex)
    for i in range(2*temp+1):
        kzref[i,i]=np.sqrt(Constant['n1']**2-kx[i,i]**2-ky[i,i]**2,dtype=complex)
    Constant['kz']=kzref
    real_mask=np.abs(np.imag(np.diag(Constant['kz'])))<Constant['accuracy']
    real_set=Constant['mx'][real_mask]
    real_set_ind=np.where(real_mask)
    Constant['real_set']=real_set
    thetam=np.arange(2*len(real_set),dtype=float).reshape((len(real_set),2))
    i=0
    for m in real_set:
        thetam[i,0]=m
        temp=np.asin(m*Constant['wavelength']/Constant['period']-np.sin(thetai))
        temp=np.rad2deg(temp)
        thetam[i,1]=np.round(temp,2)
        i+=1
    print(thetam)