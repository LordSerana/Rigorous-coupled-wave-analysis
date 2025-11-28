import numpy as np
from S_matrix.Set_polarization import Set_Polarization
from C_Method.Grating import Triangular
from C_Method.F_series_gen import F_series_gen
from C_Method.Toeplitze import Toeplitz

def layer_mode(layer,Constant):
    #计算介电常数卷积矩阵
    Nx=Constant['Nx']
    m=Constant['n_Tr']//2
    epsilon=np.ones(Nx,dtype=complex)*layer.n**2
    temp=int(layer.fill_factor*Nx/2)
    q0=int(Nx/2)
    epsilon[q0-temp:q0+temp+1]=Constant['n1']**2
    epsilon_recip=1/epsilon            
    fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
    fourier_coeffi_recip=np.fft.fftshift(np.fft.fft(epsilon_recip,axis=0)/epsilon.shape[0])
    E=Toeplitz(fourier_coeffi,2*m+1)
    E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    E_recip_inv=np.linalg.inv(E_recip)
    return E,E_recip_inv

def Compute(Constant,layers,plot=False):
    kinc=Constant['kinc']
    kx=np.diag(kinc[0]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])
    Constant['kx']=kx
    # temp=Constant['n_Tr']//2
    ky=np.diag(kinc[1]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])
    Constant['ky']=ky
    # kzref=np.zeros((2*temp+1,2*temp+1),dtype=complex)
    # for i in range(2*temp+1):
    #     kzref[i,i]=np.sqrt(Constant['n1']**2-kx[i,i]**2,dtype=complex)
    # Constant['kzref']=-kzref
    nDim=Constant['n_Tr']
    gamma=np.sqrt(kx**2+ky**2,dtype=complex)
    kmz1=np.sqrt(Constant['omiga']**2*Constant['e1']-gamma**2)
    kmz2=np.sqrt(Constant['omiga']**2*Constant['e2']-gamma**2)
    ###构造M矩阵
    a_diff_vec=F_series_gen(Constant['a'],10,Constant['period'],nDim)
    c=a_diff_vec/np.sqrt(1+a_diff_vec**2,dtype=complex)
    c=Toeplitz(c,nDim)
    s=1/np.sqrt(1+a_diff_vec**2,dtype=complex)
    s=Toeplitz(s,nDim)
    E,E_recip_inv=layer_mode(layers,Constant)
    A=c@E_recip_inv@c+s@E@s
    B=s@E@c-c@E_recip_inv@s
    C=c@E@s-s@E_recip_inv@c
    D=s@E_recip_inv@s+c@E@c
    M11=-kx@np.linalg.inv(D)@C

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
Constant['e1']=Constant['n1']**2
Constant['e2']=Constant['n2']**2
Constant['c']=299792458
Constant['omiga']=2*np.pi*Constant['c']/Constant['wavelength']
Constant['accuracy']=1e-9
Constant['error']=0.001#相对误差
R_effi=[]
Abs_error=[]
Rela_error=[]
#######################################################################
grating=Triangular(4*1e-6,30,1)
a,a_diff=grating.profile()
Constant['a']=a
Constant['diff_a']=a_diff
