import numpy as np
from S_matrix.F_series_gen import F_series_gen
from S_matrix.Toeplitz import Toeplitz

'''
针对RCWA计算Non_lamellar结构的不准确的情况,采用微分法计算
DM法和RCWA法在许多方面类似,不同的是两者一个采用Eigenvalue求解,一个采用求解ODE方程组
'''

def Construct_M_matrix(layer,n,Constant):
    kx=Constant['kx']
    ky=Constant['ky']
    Nx=Constant['Nx']
    x=np.linspace(0,Constant['period'],Nx)
    dx=Constant['period']/Nx
    a=np.zeros(Nx)
    width=int(Nx*layer.fill_factor/2)
    q0=Nx//2
    a[q0-width:q0+width+1]=Constant['a'](x[q0-width:q0+width+1])
    if n!=1:
        #切片层的顶部高度相同
        temp=int((n-1)/Constant['n']*Constant['fill_factor']*Nx//2)
        a[q0-temp:q0+temp+1]=a[q0-temp-1]
    ######根据a数组,求a数组的导数
    # temp1=a[:-1]
    # temp2=a[1:]
    # a_diff=(temp2-temp1)/dx
    if Constant['name']=="Rectangular":
        '''
        对矩形光栅,其导数处处设置为无穷大,则结果正确
        '''
        a_diff=np.zeros_like(a)
        NX=-1*np.ones_like(a_diff)
        NZ=np.zeros_like(a_diff)
        temp=F_series_gen(NX**2,Constant['n_Tr'])
        NX2=Toeplitz(temp,Constant['n_Tr'])
        temp=F_series_gen(NZ**2,Constant['n_Tr'])
        NZ2=Toeplitz(temp,Constant['n_Tr'])
        temp=F_series_gen(NX*NZ,Constant['n_Tr'])
        NXNZ=Toeplitz(temp,Constant['n_Tr'])
    else:
        a_diff=np.gradient(a,dx)
        # a_diff[np.where(a_diff==0)]=1
        NX=np.zeros_like(a_diff)
        NX[np.where(a_diff>0)]=np.sin(np.radians(30))
        NX[np.where(a_diff<0)]=np.sin(np.radians(30))
        NX[np.where(a_diff==0)]=1
        # NX=-a_diff/np.sqrt(1+a_diff**2)#为了与上面的Nx做区分,此处表示的是在x方向的偏导数
        temp=F_series_gen(NX**2,Constant['n_Tr'])
        NX2=Toeplitz(temp,Constant['n_Tr'])
        NZ=np.zeros_like(a_diff)
        NZ[np.where(a_diff>0)]=-np.cos(np.radians(30))
        NZ[np.where(a_diff<0)]=np.cos(np.radians(30))
        NZ[np.where(a_diff==0)]=0
        # NZ=1/np.sqrt(1+a_diff**2)#在z方向的偏导数
        temp=F_series_gen(NZ**2,Constant['n_Tr'])
        NZ2=Toeplitz(temp,Constant['n_Tr'])
        temp=F_series_gen(NX*NZ,Constant['n_Tr'])
        NXNZ=Toeplitz(temp,Constant['n_Tr'])
    E,E_recip_inv=layer_mode(layer,Constant)
    # A=E@(c**2)+E_recip_inv@(s**2)
    A=E@NZ2+E_recip_inv@NX2
    B=-(E-E_recip_inv)@NXNZ
    C=-(E-E_recip_inv)@NXNZ
    # D=E@(s**2)+E_recip_inv@(c**2)
    D=E@NX2+E_recip_inv@NZ2
    D_inv=np.linalg.inv(D)
    I=np.eye(kx.shape[0])
    M11=-1j*kx@D_inv@C
    M12=np.zeros_like(M11)
    M13=kx@D_inv@ky
    M14=I-kx@D_inv@kx
    M21=-1j*ky@D_inv@C
    M22=np.zeros_like(M11)
    M23=-I+ky@D_inv@ky
    M24=-ky@D_inv@kx
    M31=kx@ky
    M32=E-kx@kx
    M33=np.zeros_like(M11)
    M34=np.zeros_like(M11)
    M41=ky@ky-A+B@D_inv@C
    M42=-ky@kx
    M43=1j*B@D_inv@ky
    M44=-1j*B@D_inv@kx
    M=np.block([[M11,M12,M13,M14],[M21,M22,M23,M24],[M31,M32,M33,M34],[M41,M42,M43,M44]])
    return M