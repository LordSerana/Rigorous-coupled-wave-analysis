import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Set_polarization import Set_Polarization
from C_Method.Grating import Triangular
from C_Method.Toeplitze import Toeplitz
from S_matrix.Layer import Layer
from S_matrix.Star import Star
from S_matrix.CalcEffi import calcEffi
from S_matrix.Plot_Effi import Plot_Effi
import matplotlib.pyplot as plt
from S_matrix.F_series_gen import F_series_gen
# from S_matrix.Calculate_Poynting import Calculate_Poynting
from S_matrix.Build_scatter_side import build_scatter_side
# from S_matrix.Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

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

def Calculate_Poynting(Eigenvector,Eigenvalue):
    '''
    本函数用于计算本征模态的坡印廷矢量,根据计算结果的正负进行排序
    '''
    num=np.shape(Eigenvector)[1]
    Sz=np.zeros(num,dtype=float)
    block_size=int(num/4)
    for i in range(num):
        temp=Eigenvector[:,i]
        Ex=temp[:block_size]
        Ey=temp[block_size:2*block_size]
        Hx=temp[2*block_size:3*block_size]
        Hy=temp[3*block_size:]
        Sz[i]=np.real(np.dot(1j*Ex,np.conj(Hy))-np.dot(1j*Ey,np.conj(Hx)))
    #首先对Poynting向量初步排序，正的排在前半，负的排在后半
    tol=1e-12
    forward=[]
    backward=[]
    for i in range(num):
        if Sz[i]>tol:
            forward.append(i)
        elif Sz[i]<-tol:
            backward.append(i)
        else:
            if np.real(Eigenvalue[i])<0:
                forward.append(i)
            else:
                backward.append(i)
    new_ind=np.array(forward+backward,dtype=int)
    Eigenvalue=Eigenvalue[new_ind]
    Eigenvector=Eigenvector[:,new_ind]
    half=num//2
    if len(forward)!=half or len(backward)!=half:
        raise ValueError(
            f"Poynting分类错误:forward={len(forward)},backward={len(backward)},应各为{half}"
        )
    ###############内部排序,按照虚部的降序排序
    half=Eigenvalue.shape[0]//2
    forward=np.argsort(-Eigenvalue[:half].imag)
    backward=np.argsort(-abs(Eigenvalue[half:].imag))+half
    new_ind=np.concatenate([forward,backward])
    Eigenvalue=Eigenvalue[new_ind]
    Eigenvalue[forward]=-abs(Eigenvalue[forward].real)+1j*abs(Eigenvalue[forward].imag)
    Eigenvalue[backward]=abs(Eigenvalue[backward].real)-1j*abs(Eigenvalue[backward].imag)
    Eigenvector=Eigenvector[:,new_ind]
    return Eigenvector,Eigenvalue

def Calculate_Gap(kx,ky,Constant):
    I=np.eye(kx.shape[0])
    W=np.eye(2*kx.shape[0])
    omega=np.block([[kx@ky,I+ky@ky],[-(I+kx@kx),-kx@ky]])#到底用1还是I？
    V=-1j*omega
    LAM=-1j*Constant['kzref']
    LAM=np.concatenate([LAM,LAM])
    Eigenvalue=np.concatenate([LAM,-LAM])
    Eigenvector=np.block([[W,W],[V,-V]])
    # zero=np.zeros_like(omega)
    # M=np.block([[zero,omega],[omega,zero]])
    #是否需要排序呢？
    # LAM,W=np.linalg.eig(M)
    # Eigenvector,Eigenvalue=Calculate_Poynting(Eigenvector,Eigenvalue)
    half=np.shape(Eigenvector[0,:])[0]//2
    V_g_E_P=Eigenvector[:half,:half]
    V_g_E_N=Eigenvector[:half,half:]
    V_g_H_P=Eigenvector[half:,:half]
    V_g_H_N=Eigenvector[half:,half:]
    Constant['V_g_E_P']=V_g_E_P
    Constant['V_g_E_N']=V_g_E_N
    Constant['V_g_H_P']=V_g_H_P
    Constant['V_g_H_N']=V_g_H_N
    # LAM,W=homogeneous_isotropic_matrix(1,1,kx,ky)
    # W,LAM=Calculate_Poynting(W,LAM)
    # temp=int(W.shape[0]/2)
    # W0=W[:temp,:temp]
    # Constant['W0']=W0
    # Constant['W']=W
    # Constant['V_g_E_P']=W[:temp,:temp]
    # Constant['V_g_E_N']=W[:temp,temp:]
    # Constant['V_g_H_P']=W[temp:,:temp]
    # Constant['V_g_H_N']=W[temp:,temp:]
    return Constant

def Calculate_Ref(kx,ky,layers,Constant):
    #按计划，使用解析式替代矩阵计算，如Calculate_Gap所做的那样
    E,E_recip_inv=layer_mode(layers[0],Constant)
    nDim=Constant['n_Tr']
    c=np.zeros((nDim,nDim))
    s=np.eye(nDim)
    A=c@E_recip_inv@c+s@E@s
    B=s@E@c-c@E_recip_inv@s
    C=c@E@s-s@E_recip_inv@c
    D=s@E_recip_inv@s+c@E@c
    D_inv=np.linalg.inv(D)
    M11=-kx@D_inv@C
    M12=np.zeros_like(M11)
    M13=-kx@D_inv@ky
    M14=np.ones_like(M11)+kx@D_inv@kx
    M21=-ky@D_inv@C
    M22=np.zeros_like(M11)
    M23=-(ky@D_inv@ky+np.ones_like(M11))
    M24=ky@D_inv@kx
    M31=-kx@ky
    M32=kx@kx+E
    M33=np.zeros_like(M11)
    M34=np.zeros_like(M11)
    M41=-kx@ky-A+B@D_inv@C
    M42=ky@kx
    M43=B@D_inv@ky
    M44=-B@D_inv@kx
    M=np.block([[M11,M12,M13,M14],[M21,M22,M23,M24],[M31,M32,M33,M34],[M41,M42,M43,M44]])
    LAM,W=np.linalg.eig(M)
    Eigenvector,Eigenvalue=Calculate_Poynting(W,LAM)
    V_g_E_P=Constant['V_g_E_P']
    V_g_E_N=Constant['V_g_E_N']
    V_g_H_P=Constant['V_g_H_P']
    V_g_H_N=Constant['V_g_H_N']
    temp1=np.block([[V_g_E_P,V_g_E_N],[V_g_H_P,V_g_H_N]])
    temp=np.linalg.solve(temp1,Eigenvector)
    half=np.shape(temp)[0]//2
    A=temp[:half,:half]
    B=temp[:half,half:]
    C=temp[half:,:half]
    D=temp[half:,half:]
    D_inv=np.linalg.inv(D)
    S11=-D_inv@C
    S12=D_inv
    S21=A-B@D_inv@C
    S22=B@D_inv
    S_ref=np.block([[S11,S12],[S21,S22]])
    # Sref,W,LAM=build_scatter_side(Constant['e1'],1,kx,ky,Constant['W'])
    # S_ref=np.block([[Sref[0],Sref[1]],[Sref[2],Sref[3]]])
    return S_ref

def Calculate_trn(kx,ky,layers,Constant):
    E,E_recip_inv=layer_mode(layers[-1],Constant)
    c=np.zeros((Constant['n_Tr'],Constant['n_Tr']))
    s=np.eye((Constant['n_Tr']))
    A=c@E_recip_inv@c+s@E@s
    B=s@E@c-c@E_recip_inv@s
    C=c@E@s-s@E_recip_inv@c
    D=s@E_recip_inv@s+c@E@c
    D_inv=np.linalg.inv(D)
    M11=-kx@D_inv@C
    M12=np.zeros_like(M11)
    M13=-kx@D_inv@ky
    M14=np.ones_like(M11)+kx@D_inv@kx
    M21=-ky@D_inv@C
    M22=np.zeros_like(M11)
    M23=-(ky@D_inv@ky+np.ones_like(M11))
    M24=ky@D_inv@kx
    M31=-kx@ky
    M32=kx@kx+E
    M33=np.zeros_like(M11)
    M34=np.zeros_like(M11)
    M41=-kx@ky-A+B@D_inv@C
    M42=ky@kx
    M43=B@D_inv@ky
    M44=-B@D_inv@kx
    M=np.block([[M11,M12,M13,M14],[M21,M22,M23,M24],[M31,M32,M33,M34],[M41,M42,M43,M44]])
    LAM,W=np.linalg.eig(M)
    Eigenvector,Eigenvalue=Calculate_Poynting(W,LAM)
    V_g_E_P=Constant['V_g_E_P']
    V_g_E_N=Constant['V_g_E_N']
    V_g_H_P=Constant['V_g_H_P']
    V_g_H_N=Constant['V_g_H_N']
    temp1=np.block([[V_g_E_P,V_g_E_N],[V_g_H_P,V_g_H_N]])
    temp=np.linalg.solve(temp1,Eigenvector)
    half=np.shape(temp)[0]//2
    A=temp[:half,:half]
    B=temp[:half,half:]
    C=temp[half:,:half]
    D=temp[half:,half:]
    A_inv=np.linalg.inv(A)
    S11=C@A_inv
    S12=D-C@A_inv@B
    S21=A_inv
    S22=-A_inv@B
    S_trn=np.block([[S11,S12],[S21,S22]])
    # Strn,W,LAM=build_scatter_side(Constant['e2'],1,kx,ky,Constant['W'],transmission_side=True)
    # W,LAM=Calculate_Poynting(W,LAM)
    # S_trn=np.block([[Strn[0],Strn[1]],[Strn[2],Strn[3]]])
    return S_trn

def Compute(Constant,layers,plot=False):
    kinc=Constant['kinc']
    kx=np.diag(kinc[0]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])#已经归一化
    Constant['kx']=kx
    temp=Constant['n_Tr']//2
    if Constant['dimension']==1:
        ky=np.zeros_like(kx)
        ky[temp,temp]=kinc[1]
    else:
        ky=np.diag(kinc[1]-2*np.pi*Constant['my']/Constant['k0']/Constant['period'])#已经归一化
    Constant['ky']=ky
    kzref=np.zeros((2*temp+1,2*temp+1),dtype=complex)
    for i in range(2*temp+1):
        val=Constant['n1']**2-kx[i,i]**2-ky[i,i]**2
        kz=np.sqrt(val+0j)
        if np.imag(kz)<0:
            kz=-kz
        kzref[i,i]=kz
    Constant['kzref']=-kzref
    nDim=Constant['n_Tr']
    ###构造M矩阵
    x=np.linspace(0,Constant['period'],2**10)
    temp=Constant['diff_a']
    diff_a=temp(x)
    c=diff_a/np.sqrt(1+diff_a*diff_a,dtype=complex)#遗漏了一步傅里叶变换
    temp=F_series_gen(c,nDim)
    c=Toeplitz(temp,nDim)
    s=1/np.sqrt(1+diff_a*diff_a,dtype=complex)
    temp=F_series_gen(s,nDim)
    s=Toeplitz(temp,nDim)
    #############计算间隙介质的散射矩阵
    # LAM,W=homogeneous_isotropic_matrix(1,1,kx,ky)
    # W,LAM=Calculate_Poynting(W,LAM)
    # half=W.shape[0]//2
    # Constant['W']=W
    # Constant['V_g_E_P']=W[:half,:half]
    # Constant['V_g_E_N']=W[:half,half:]
    # Constant['V_g_H_P']=W[half:,:half]
    # Constant['V_g_H_N']=W[half:,half:]
    Constant=Calculate_Gap(kx,ky,Constant)
    V_g_E_P=Constant['V_g_E_P']
    V_g_E_N=Constant['V_g_E_N']
    V_g_H_P=Constant['V_g_H_P']
    V_g_H_N=Constant['V_g_H_N']
    temp1=np.block([[V_g_E_P,V_g_E_N],[V_g_H_P,V_g_H_N]])#Gap_medium的散射矩阵
    zero=np.zeros((2*nDim,2*nDim))
    S_global=np.block([[zero,np.eye(2*nDim)],[np.eye(2*nDim),zero]])
    S_ref=Calculate_Ref(kx,ky,layers,Constant)
    S_global=Star(S_global,S_ref)
    for i in layers[1:-1]:
        E,E_recip_inv=layer_mode(i,Constant)
        A=c@E_recip_inv@c+s@E@s
        B=s@E@c-c@E_recip_inv@s
        C=c@E@s-s@E_recip_inv@c
        D=s@E_recip_inv@s+c@E@c
        D_inv=np.linalg.inv(D)
        M11=-kx@D_inv@C
        M12=np.zeros_like(M11)
        M13=-kx@D_inv@ky
        M14=np.ones_like(M11)+kx@D_inv@kx
        M21=-ky@D_inv@C
        M22=np.zeros_like(M11)
        M23=-(ky@D_inv@ky+np.ones_like(M11))
        M24=ky@D_inv@kx
        M31=-kx@ky
        M32=kx@kx+E
        M33=np.zeros_like(M11)
        M34=np.zeros_like(M11)
        M41=-kx@ky-A+B@D_inv@C
        M42=ky@kx
        M43=B@D_inv@ky
        M44=-B@D_inv@kx
        M=np.block([[M11,M12,M13,M14],[M21,M22,M23,M24],[M31,M32,M33,M34],[M41,M42,M43,M44]])
        #########计算EigenVector和Eigenvalue，并进行排序
        LAM,W=np.linalg.eig(M)
        Eigenvector,Eigenvalue=Calculate_Poynting(W,LAM)
        #########构造S矩阵
        temp=np.linalg.solve(Eigenvector,temp1)
        half=np.shape(temp)[0]//2
        A=temp[:half,:half]
        B=temp[:half,half:]
        C=temp[half:,:half]
        D=temp[half:,half:]
        A_inv=np.linalg.inv(A)
        D_inv=np.linalg.inv(D)
        half=np.shape(Eigenvalue)[0]//2
        X_P=np.diag(np.exp(Eigenvalue[:half]*Constant['k0']*Constant['depth']))
        X_N=np.diag(np.exp(-Eigenvalue[half:]*Constant['k0']*Constant['depth']))
        S11=np.linalg.solve(D-X_N@C@A_inv@X_P@B,X_N@C@A_inv@X_P@A-C)
        temp=X_N@(D-C@A_inv@B)
        S12=np.linalg.solve(D-X_N@C@A_inv@X_P@B,temp)
        temp=X_P@(A-B@D_inv@C)
        S21=np.linalg.solve(A-X_P@B@D_inv@X_N@C,temp)
        S22=np.linalg.solve(A-X_P@B@D_inv@X_N@C,X_P@B@D_inv@X_N@D-B)
        S=np.block([[S11,S12],[S21,S22]])
        S_global=Star(S_global,S)
    S_trn=Calculate_trn(kx,ky,layers,Constant)
    S_global=Star(S_global,S_trn)
    R_effi,T_effi=calcEffi(Constant['p'],Constant,S_global)
    Plot_Effi(R_effi,T_effi,Constant)

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
# R_effi=[]
Abs_error=[]
Rela_error=[]
#####################设定光栅参数#####################################
grating=Triangular(4*1e-6,30,1)
a,a_diff=grating.profile()
Constant['dimension']=1#光栅是一维光栅
Constant['a']=a
Constant['diff_a']=a_diff
Constant['depth']=Constant['period']/2*np.tan(np.radians(30))
#####################################################################
layers=[
    Layer(n=1+0.001j,t=1*1e-6),
    Layer(n=1.4482+7.5367j,t=1.8*1e-6,fill_factor=1),
    Layer(n=1.4482+7.5367j,t=4*1e-6)
    ]
Compute(Constant,layers)