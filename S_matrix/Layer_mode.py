import numpy as np
from Toeplitz import Toeplitz

def layer_mode(layer,Constant,case='FFT'):
    #####从介电常数序列计算Toeplitze矩阵
    Nx=Constant['Nx']
    m=Constant['n_Tr']//2
    if case=='FFT':
        epsilon=np.ones(Nx,dtype=complex)*layer.n**2
        temp=int(layer.fill_factor*Nx/2)
        q0=int(Nx/2)
        epsilon[q0-temp:q0+temp+1]=Constant['n1']**2
        epsilon_recip=1/epsilon            
        fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
        fourier_coeffi_recip=np.fft.fftshift(np.fft.fft(epsilon_recip,axis=0)/epsilon.shape[0])
        E=Toeplitz(fourier_coeffi,2*m+1)
        E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    else:
        fourier_coeffi=np.zeros(4*m+1,dtype=complex)
        fourier_coeffi_recip=np.zeros(4*m+1,dtype=complex)
        temp=Constant['n1']**2-layer.n**2
        temp2=(1/Constant['n1'])**2-(1/layer.n)**2
        for i in range(-2*m,2*m+1,1):
            if i!=0:
                fourier_coeffi[i+2*m]=temp*np.sin(np.pi*i*layer.fill_factor)/np.pi/i
                fourier_coeffi_recip[i+2*m]=temp2*np.sin(np.pi*i*layer.fill_factor)/np.pi/i
            else:
                fourier_coeffi[i+2*m]=Constant['n1']**2*layer.fill_factor+layer.n**2*(1-layer.fill_factor)
                fourier_coeffi_recip[i+2*m]=(1/Constant['n1'])**2*layer.fill_factor+(1/layer.n)**2*(1-layer.fill_factor)
        E=Toeplitz(fourier_coeffi,2*m+1)
        E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    #####计算本层的模态
    E_inv=np.linalg.inv(E)
    E_recip_inv=np.linalg.inv(E_recip)
    kx=Constant['kx']
    ky=Constant['ky']
    P=np.block([[kx@E_inv@ky,np.eye(2*m+1)-kx@E_inv@kx],[ky@E_inv@ky-np.eye(2*m+1),-ky@E_inv@kx]])
    Q=np.block([[kx@ky,E-kx@kx],[ky@ky-E_recip_inv,-ky@kx]])
    omiga2=P@Q
    LAM2,W=np.linalg.eig(omiga2)
    LAM=np.sqrt(LAM2,dtype=complex)
    LAM = -np.abs(LAM.real) + 1j*np.abs(LAM.imag)
    #验证模态长度
    temp=LAM.real**2+LAM.imag**2
    V=Q@W@np.diag(1/LAM)
    X=np.exp(LAM*Constant['k0']*layer.t)
    X=np.diag(X)
    #####构建散射矩阵S
    W0=Constant['W0']
    V0=Constant['V0']
    A=np.linalg.inv(W)@W0+np.linalg.inv(V)@V0
    B=np.linalg.inv(W)@W0-np.linalg.inv(V)@V0
    S11=S22=np.linalg.inv(A-X@B@np.linalg.inv(A)@X@B)@(X@B@np.linalg.inv(A)@X@A-B)
    S12=S21=np.linalg.inv(A-X@B@np.linalg.inv(A)@X@B)@X@(A-B@np.linalg.inv(A)@B)
    S=np.block([[S11,S12],[S21,S22]])
    return S