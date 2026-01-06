import numpy as np
import sys
sys.path.append("E:/Project/Python")
# from S_matrix.Set_polarization import Set_Polarization
from C_Method.Toeplitze import Toeplitz
from S_matrix.Layer import Layer
from S_matrix.Star import Star
# from S_matrix.CalcEffi import calcEffi
from S_matrix.Plot_Effi import Plot_Effi
import matplotlib.pyplot as plt
from S_matrix.F_series_gen import F_series_gen
from S_matrix.Grating import Rectangular,Triangular

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
'''
该脚本用于验证切片操作和散射矩阵的级联的正确性
'''

def layer_mode(layer,Constant):
    #计算介电常数卷积矩阵
    Nx=Constant['Nx']
    m=Constant['n_Tr']//2
    epsilon=np.ones(Nx,dtype=complex)*Constant['e1']
    width=int(layer.fill_factor*Nx)
    epsilon[:width]=layer.n**2
    epsilon_recip=1/epsilon            
    fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
    fourier_coeffi_recip=np.fft.fftshift(np.fft.fft(epsilon_recip,axis=0)/epsilon.shape[0])
    E=Toeplitz(fourier_coeffi,2*m+1)
    E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    E_recip_inv=np.linalg.inv(E_recip)
    return E,E_recip_inv

def Set_Polarization(thetai,phi,wavelength,pTM,pTE,Constant):
    Constant['thetai']=thetai
    Constant['phi']=phi
    Constant['wavelength']=wavelength
    Constant['pTM']=pTM
    Constant['pTE']=pTE
    k0=2*np.pi/wavelength
    Constant['k0']=k0
    n=[0,0,1]
    kinc=Constant['n1']*np.array([np.sin(thetai)*np.cos(phi),np.sin(thetai)*np.sin(phi),np.cos(thetai)])
    Constant['kinc']=kinc
    if thetai==0:
        aTE=[0,1,0]
    else:
        aTE=np.cross(n,kinc)/np.linalg.norm(np.cross(n,kinc))
    aTM=np.cross(aTE,kinc)/np.linalg.norm(np.cross(aTE,kinc))
    Constant['eTE']=aTE
    Constant['eTM']=aTM
    p=np.dot(pTE,aTE)+np.dot(pTM,aTM)
    p=p/np.linalg.norm(p)
    Constant['p']=p
    return Constant

def CalcEffi(p,Constant,S_global):
    m=Constant['n_Tr']//2
    delta0=np.zeros(2*m+1)
    delta0[m]=1
    c_inc=np.concatenate((p[0]*delta0,p[1]*delta0))
    c_ref=S_global[:4*m+2,:4*m+2]@c_inc
    c_trn=S_global[4*m+2:,:4*m+2]@c_inc
    Ref_EM_field=Constant['Vref']@np.concatenate([np.zeros_like(c_ref),c_ref])
    # Ref_EM_field=Constant['Vref']@c_ref
    block_size=Ref_EM_field.shape[0]//4
    Ex_ref=Ref_EM_field[:block_size]
    Ey_ref=Ref_EM_field[block_size:2*block_size]
    Hx_ref=Ref_EM_field[2*block_size:3*block_size]
    Hy_ref=Ref_EM_field[3*block_size:]
    Sz_ref=np.real(1j*Ex_ref*np.conj(Hy_ref)-1j*Ey_ref*np.conj(Hx_ref))
    Trn_EM_field=Constant['Vtrn']@np.concatenate([c_trn,np.zeros_like(c_trn)])
    # Trn_EM_field=Constant['Vtrn']@c_trn
    Ex_trn=Trn_EM_field[:block_size]
    Ey_trn=Trn_EM_field[block_size:2*block_size]
    Hx_trn=Trn_EM_field[2*block_size:3*block_size]
    Hy_trn=Trn_EM_field[3*block_size:]
    Sz_trn=np.real(1j*Ex_trn*np.conj(Hy_trn)-1j*Ey_trn*np.conj(Hx_trn))
    Sz_inc=(p[0]**2+p[1]**2)*np.cos(Constant['thetai'])
    # print(np.sum(abs(Ex_ref)),np.sum(abs(Ey_ref)))
    R=abs(Sz_ref)/Sz_inc
    T=Sz_trn/Sz_inc
    return R,T

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
    half=num//2
    if len(forward)!=half or len(backward)!=half:
        # raise ValueError(
        #     f"Poynting分类错误:forward={len(forward)},backward={len(backward)},应各为{half}"
        # )
        forward=[]
        backward=[]
        for i in range(num):
            if Eigenvalue[i].imag>tol:
                forward.append(i)
            elif Eigenvalue[i].imag<-tol:
                backward.append(i)
            else:
                if Eigenvalue[i].real<-tol:
                    forward.append(i)
                else:
                    backward.append(i)
    new_ind=np.array(forward+backward,dtype=int)
    Eigenvalue=Eigenvalue[new_ind]
    Eigenvector=Eigenvector[:,new_ind]
    ###############内部排序,按照虚部的降序排序
    # half=Eigenvalue.shape[0]//2
    # forward=np.argsort(-Eigenvalue[:half].imag)
    # backward=np.argsort(-abs(Eigenvalue[half:].imag))+half
    # new_ind=np.concatenate([forward,backward])
    # Eigenvalue=Eigenvalue[new_ind]
    # Eigenvalue[forward]=-abs(Eigenvalue[forward].real)+1j*abs(Eigenvalue[forward].imag)
    # Eigenvalue[backward]=abs(Eigenvalue[backward].real)-1j*abs(Eigenvalue[backward].imag)
    # Eigenvector=Eigenvector[:,new_ind]
    return Eigenvector,Eigenvalue

def Calculate_Gap(kx,ky,Constant):
    I=np.eye(kx.shape[0])
    W=np.eye(2*kx.shape[0])
    omega=np.block([[kx@ky,I+ky@ky],[-(I+kx@kx),-kx@ky]])#到底用1还是I？用I
    V=-1j*omega
    LAM=-1j*Constant['kz']
    LAM=np.concatenate([LAM,LAM])
    Eigenvalue=np.concatenate([LAM,-LAM])
    Eigenvector=np.block([[W,W],[V,-V]])
    half=np.shape(Eigenvector[0,:])[0]//2
    V_g_E_P=Eigenvector[:half,:half]
    V_g_E_N=Eigenvector[:half,half:]
    V_g_H_P=Eigenvector[half:,:half]
    V_g_H_N=Eigenvector[half:,half:]
    Constant['V_g_E_P']=V_g_E_P
    Constant['V_g_E_N']=V_g_E_N
    Constant['V_g_H_P']=V_g_H_P
    Constant['V_g_H_N']=V_g_H_N
    Constant['W0']=V_g_E_P
    Constant['V0']=V
    return Constant

def Calculate_Ref(kx,ky,layers,Constant):
    #按计划，使用解析式替代矩阵计算，如Calculate_Gap所做的那样
    I=np.eye(kx.shape[0])
    W=np.eye(2*kx.shape[0])
    omega=np.block([[kx@ky,I+ky@ky],[-(I+kx@kx),-kx@ky]])#到底用1还是I？用I
    V=-1j*omega
    kz=Constant['kz']
    if kz.ndim==2:
        kz=np.diag(kz)
    LAM=1j*kz
    LAM=np.concatenate([LAM,LAM])
    Eigenvalue=np.concatenate([LAM,-LAM])
    Eigenvector=np.block([[W,W],[V,-V]])
    Constant['Vref']=Eigenvector
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
    return S_ref

def Calculate_trn(kx,ky,layers,Constant):
    ###使用解析式的形式
    I=np.eye(kx.shape[0])*Constant['e2']
    W=np.eye(2*kx.shape[0])
    omega=np.block([[kx@ky,I+ky@ky],[-(I+kx@kx),-kx@ky]])#到底用1还是I？用I
    kz=Constant['kz']
    if kz.ndim==2:
        kz=np.diag(kz)
    LAM=1j*kz
    LAM=np.concatenate([LAM,LAM])
    V=omega@np.linalg.inv(np.diag(LAM))
    Eigenvalue=np.concatenate([LAM,-LAM])
    Eigenvector=np.block([[W,W],[V,-V]])
    Constant['Vtrn']=Eigenvector
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
    return S_trn

def Slice(n,layers,Constant):
    '''
    n:切片数
    layers:传进仿真层,对中间层进行切片,并返回新的layers函数
    '''
    origin_FillFactor=layers[1].fill_factor
    depth=Constant['depth']/n#切片层的平均厚度
    layer0=layers[0]
    layer_last=layers[-1]
    layer_new=[]
    layer_new.append(layer0)
    for i in range(n):
        fill_factor=origin_FillFactor
        layer=Layer(n=Constant['n2'],t=depth,fill_factor=fill_factor)
        layer_new.append(layer)
    layer_new.append(layer_last)
    return layer_new

def Construct_M_matrix(layer,n,Constant):
    kx=Constant['kx']
    ky=Constant['ky']
    E,E_recip_inv=layer_mode(layer,Constant)
    E_inv=np.linalg.inv(E)
    I=np.eye(kx.shape[0])
    zero=np.zeros((kx.shape[0],kx.shape[0]))
    M11=zero
    M12=np.zeros_like(M11)
    M13=kx@E_inv@ky
    M14=I-kx@E_inv@kx
    M21=zero
    M22=np.zeros_like(M11)
    M23=ky@E_inv@ky-I
    M24=-ky@E_inv@kx
    M31=kx@ky
    M32=E-kx@kx
    M33=np.zeros_like(M11)
    M34=np.zeros_like(M11)
    M41=ky@ky-E_recip_inv
    M42=-ky@kx
    M43=zero
    M44=zero
    M=np.block([[M11,M12,M13,M14],[M21,M22,M23,M24],[M31,M32,M33,M34],[M41,M42,M43,M44]])
    return M

def Compute(Constant,layers,plot=False):
    ############前期计算的准备工作##############
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
    Constant['kz']=kzref
    nDim=Constant['n_Tr']
    #############计算间隙介质的散射矩阵##########
    Constant=Calculate_Gap(kx,ky,Constant)
    V_g_E_P=Constant['V_g_E_P']
    V_g_E_N=Constant['V_g_E_N']
    V_g_H_P=Constant['V_g_H_P']
    V_g_H_N=Constant['V_g_H_N']
    temp1=np.block([[V_g_E_P,V_g_E_N],[V_g_H_P,V_g_H_N]])#Gap_medium的散射矩阵
    ##############构建全局散射矩阵#############
    zero=np.zeros((2*nDim,2*nDim))
    S_global=np.block([[zero,np.eye(2*nDim)],[np.eye(2*nDim),zero]])
    S_ref=Calculate_Ref(kx,ky,layers,Constant)
    S_global=Star(S_global,S_ref)
    ############构造4*4M矩阵####################
    if Constant['name']!="Rectangular":
        layers=Slice(Constant['n'],layers,Constant)#光栅区域切片操作
    n=1
    for i in layers[1:-1]:
        M=Construct_M_matrix(i,n,Constant)
        n+=1
        #########计算EigenVector和Eigenvalue，并进行排序
        LAM,W=np.linalg.eig(M)
        Eigenvector,Eigenvalue=Calculate_Poynting(W,LAM)
        # Eigenvector=W
        # Eigenvalue=LAM
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
        X_P=np.diag(np.exp(Eigenvalue[:half]*Constant['k0']*i.t))
        X_N=np.diag(np.exp(-Eigenvalue[half:]*Constant['k0']*i.t))
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
    R_effi,T_effi=CalcEffi(Constant['p'],Constant,S_global)
    Constant['R_effi']=R_effi
    Plot_Effi(Constant,[])

#####################设定光栅参数#####################################
Constant={}
grating=Rectangular(4*1e-6,1,2*1e-6)
# grating=Triangular(4*1e-6,30,1)
Constant['period']=grating.T
Constant['fill_factor']=grating.fill_factor
a=grating.profile()
Constant['name']=grating.name
Constant['dimension']=1#光栅是一维光栅
Constant['a']=a#光栅轮廓函数
# Constant['diff_a']=a_diff#光栅表面轮廓的导数函数
# Constant['depth']=Constant['period']/2*np.tan(np.radians(30))
Constant['depth']=grating.depth
######################设定仿真层#####################################
layers=[
    Layer(n=1,t=1*1e-6),#光栅上方的自由空间,可以理解为空气层
    Layer(n=1.4482+7.5367j,t=2*1e-6,fill_factor=Constant['fill_factor']),#光栅层
    Layer(n=1.4482+7.5367j,t=4*1e-6)#光栅基底区域
    ]
Constant['n1']=layers[0].n
Constant['e1']=Constant['n1']**2
Constant['n2']=layers[-1].n
Constant['e2']=Constant['n2']**2
###########################设定仿真常数################################
thetai=np.radians(0)#入射角thetai
phi=np.radians(0)#入射角phi
wavelength=632.8*1e-9
pTM=1
pTE=1
Constant=Set_Polarization(thetai,phi,wavelength,pTM,pTE,Constant)
m=15
Constant['n_Tr']=2*m+1
Constant['n']=50#切片数
Constant['mx']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
Constant['my']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
Constant['Nx']=2**10
Constant['c']=299792458
Constant['omiga']=2*np.pi*Constant['c']/Constant['wavelength']
Constant['accuracy']=1e-9
Constant['error']=0.001#相对误差
# R_effi=[]
Abs_error=[]
Rela_error=[]
#####################开始计算########################################
Compute(Constant,layers)