import matplotlib.pyplot as plt
import numpy as np
from S_matrix.Grating import Triangular,Sinusoidal
from C_Method.SetConstantByPolar import setConstanByPola
from S_matrix.F_series_gen import F_series_gen
from C_Method.Toeplitze import Toeplitz
from C_Method.Eigen import Eigen
from C_Method.Plot_intensity import Plot_intensity
from C_Method.Plot_Effi import Plot_Effi

#一般说来，对A,B1,B2的归一化能够降低矩阵条件数,且最终计算结果不变
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
  
def GenerateFFieldsChand(a,b0,nDim,k0,d,m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2):
    LP_fun=np.exp(-1j*b0*a)#define fourier transform argument for positive fields
    F_in_0=F_series_gen(LP_fun,nDim,cut_small=False)#正场系数
    q0=len(F_in_0)//2
    F_in=F_in_0[q0-(nDim-1)//2:q0+(nDim-1)//2+1]
    FRN=np.zeros((nDim,len(real_Ray2_idx)),dtype=np.complex128)#F_mk_R_N
    FRP=np.zeros((nDim,len(real_Ray1_idx)),dtype=np.complex128)#F_mn_R_P
    if len(real_Ray1_idx)!=0:
        min_ray1=np.min(real_Ray1_idx)
        max_ray1=np.max(real_Ray1_idx)
        for M in range(min_ray1,max_ray1+1):
            LPn_fun=np.exp(-1j*SB1[M-m1]*a)
            FRP0=F_series_gen(LPn_fun,nDim,cut_small=False)
            q0=len(FRP0)//2
            part1=np.conj(FRP0[q0+m1-M:q0])
            part2=np.conj(FRP0[q0:q0+m2-M+1])
            FRP[:,M-min_ray1]=np.concatenate([part1,part2])
    else:
        FRP=np.zeros((nDim,len(real_Ray1_idx)))
    if len(real_Ray2_idx)!=0:
        min_ray2=np.min(real_Ray2_idx)
        max_ray2=np.max(real_Ray2_idx)
        for k in range(min_ray2,max_ray2+1):
            LNn_fun=np.exp(-1j*SB2[k]*a)
            FRN0=F_series_gen(LNn_fun,nDim,cut_small=False)
            q0=len(FRN0)//2
            for m in range(m1,m2+1):
                FRN[m,k]=FRN0[q0+m-k]
    else:
        FRN=np.zeros((nDim,len(real_Ray2_idx)))
    return F_in,FRN,FRP

def GenerateGFieldsChand(b0,a_mat,real_eig1p,real_eig2n,SB1,SB2,real_Ray1_idx,real_Ray2_idx,\
    m1,m2,nDim,imag_eig1p,imag_eig2n,FRP,FRN,F_in,imag_Vec1p,imag_Vec2n,A,eps1,eps2):
    G_RP=np.zeros((nDim,len(real_eig1p)),dtype=complex)
    G_RN=np.zeros((nDim,len(real_eig2n)),dtype=complex)
    G_in=np.zeros((nDim,1),dtype=complex)
    AUX_mat=np.eye(nDim)+a_mat@a_mat
    #计算G_RP(实特征值正场)
    if len(real_Ray1_idx)!=0:
        min_ray1=np.min(real_Ray1_idx)
        for M in range(nDim):
            for N in range(len(real_eig1p)):
                for K in range(nDim):
                    sb1_idx=N+min_ray1-m1
                    term=a_mat[M,K]*A[K]-AUX_mat[M,K]*SB1[sb1_idx]
                    G_RP[M,N]+=term*FRP[K,N]
    else:
        G_RP=np.zeros((nDim,len(real_eig1p)),dtype=complex)
    #计算G_RN(实特征值负场)
    if len(real_Ray2_idx)!=0:
        min_ray2=np.min(real_Ray2_idx)
        for M in range(nDim):
            for N in range(len(real_eig2n)):
                for K in range(nDim):
                    sb2_idx=N+min_ray2-m1
                    term=a_mat[M,K]*A[K]-AUX_mat[M,K]*SB2[sb2_idx]
                    G_RN[M,N]+=term*FRN[K,N]
    else:
        G_RN=np.zeros((nDim,len(real_eig2n)),dtype=complex)
    #计算G_in
    for M in range(nDim):
        for N in range(nDim):
            term=a_mat[M,N]*A[N]+AUX_mat[M,N]*b0
            G_in[M,0]+=term*F_in[N]
    #计算G_P,G_N(虚特征值场)
    G_P=np.zeros((nDim,len(imag_eig1p)),dtype=complex)
    G_N=np.zeros((nDim,len(imag_eig2n)),dtype=complex)
    for M in range(nDim):
        for N in range(len(imag_eig1p)):
            for K in range(nDim):
                term=a_mat[M,K]*A[K]-AUX_mat[M,K]*imag_eig1p[N]
                G_P[M,N]+=term*imag_Vec1p[K,N]
    for M in range(nDim):
        for N in range(len(imag_eig2n)):
            for K in range(nDim):
                term=a_mat[M,K]*A[K]-AUX_mat[M,K]*imag_eig2n[N]
                G_N[M,N]+=term*imag_Vec2n[K,N]
    #归一化处理
    G_RP=Constant['Z0']/Constant['k0']/eps1*G_RP
    G_RN=Constant['Z0']/Constant['k0']/eps2*G_RN
    G_in=Constant['Z0']/Constant['k0']/eps1*G_in
    G_P=Constant['Z0']/Constant['k0']/eps1*G_P
    G_N=Constant['Z0']/Constant['k0']/eps2*G_N
    return G_RP,G_RN,G_in,G_P,G_N

def Compute(n1,n2,polar,Constant):
    Constant=setConstanByPola(n1,n2,polar,Constant)
    k0=2*np.pi/Constant['wavelength']#波数
    K=2*np.pi/Constant['period']#倒易空间矢量
    nDim=Constant['n_Tr']#计算所用的总模数
    alpha0=Constant['n1']*k0*np.sin(Constant['thetai'])
    m1=int(-(nDim-1)/2)
    m2=int((nDim-1)/2)
    nDim=m2-m1+1
    A=(alpha0*np.ones((1,m2-m1+1))+K*np.linspace(m1,m2,nDim))#文献中的alpham
    A=A.flatten()
    B1=Constant['n1']**2*k0**2-A*A#beta1**2
    B1=B1.flatten()
    B2=Constant['n2']**2*k0**2-A*A#beta2**2,*k0**2
    B2=B2.flatten()
    SB1=np.sqrt(B1,dtype=complex)#eigenvalues of incident medium
    SB1_idx=(abs(np.imag(SB1))==0)&(np.real(SB1)>0)#indices of real propogation orders, in incident medium
    SB1_ind=np.arange(m1,m2+1)#indices of all modes
    real_Ray1_idx=SB1_ind[SB1_idx]#indices of real propogation numbers
    Constant['real_Ray1_idx']=real_Ray1_idx
    SB2=-np.sqrt(B2,dtype=complex)#eigenvalues of transmition medium
    SB2_idx2=(np.abs(np.imag(SB2))==0)&(np.real(SB2)<0)#indices of real propogation order
    SB2_ind2=np.arange(m1,m2+1)
    real_Ray2_idx=SB2_ind2[SB2_idx2]
    Constant['real_Ray2_idx']=real_Ray2_idx
    a_diff_vec=F_series_gen(a_diff,nDim)
    ##观察傅里叶频谱分量
    # temp_x=range(-nDim,nDim+1)
    # plt.plot(temp_x,a_diff_vec)
    # plt.show()
    #####
    a_mat=Toeplitz(a_diff_vec,nDim)
    V1,rho1,V2,rho2=Eigen(A,B1,B2,a_mat,nDim)
    eig1_p,vect1_p,_,_=SortEigenvalue(rho1,V1,100)
    _,_,eig2_n,vect2_n=SortEigenvalue(rho2,V2,100)
    # real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n=SortEigenvalueChand(V1,rho1,V2,rho2,Constant['accuracy'],nDim)
    real_eig1p=eig1_p[:len(Constant['real_Ray1_idx'])]
    real_eig2n=eig2_n[:len(Constant['real_Ray2_idx'])]
    imag_eig1p=eig1_p[len(Constant['real_Ray1_idx']):]
    imag_eig2n=eig2_n[len(Constant['real_Ray2_idx']):]
    imag_Vec1p=vect1_p[:Constant['n_Tr'],len(Constant['real_Ray1_idx']):]
    imag_Vec2n=vect2_n[:Constant['n_Tr'],len(Constant['real_Ray2_idx']):]
    #Assemble F-matrices
    b0=np.sqrt(B1[-m1])
    F_in,FRN,FRP=GenerateFFieldsChand(Constant['a'],b0,nDim,k0,Constant['period'],m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2)
    #Assemble G-matrices
    G_RP,G_RN,G_in,G_P,G_N=GenerateGFieldsChand(b0,a_mat,real_eig1p,real_eig2n,SB1,SB2,real_Ray1_idx,real_Ray2_idx,m1,m2,nDim,imag_eig1p,
    imag_eig2n,FRP,FRN,F_in,imag_Vec1p,imag_Vec2n,A,Constant['eps1'],Constant['eps2'])
    #Assemble matrix for matching boundary conditions and solve the linear system
    G_in=G_in.T
    G_in=G_in.flatten()
    MatBC=np.block([[FRP,imag_Vec1p,-FRN,-imag_Vec2n],[G_RP,G_P,-G_RN,-G_N]])
    MatBC=Perturbation(MatBC)
    VecBC=np.concatenate((-F_in,-G_in),axis=0)
    RVec=np.linalg.solve(MatBC,VecBC)
    # RVec=np.linalg.inv(MatBC)@VecBC
    #绘制衍射效率曲线
    etaR,etaT=Plot_intensity(Constant,RVec,real_Ray1_idx,real_Ray2_idx,B1,B2,m1,b0,nDim,False)
    etaR=etaR.flatten()
    etaT=etaT.flatten()
    return etaR,etaT

def SortEigenvalue(eig,vect,threshold):
    real_parts=np.real(eig)
    imag_parts=np.imag(eig)
    abs_real=np.abs(real_parts)
    abs_imag=np.abs(imag_parts)
    #=======判断特征值的主导性
    n=len(eig)
    is_real_dominant=(abs_real>threshold*abs_imag)
    is_imag_dominant=(abs_imag>threshold*abs_real)
    is_mixed=(~is_real_dominant)&(~is_imag_dominant)
    idxRealAndPositive=np.zeros(n,dtype=bool)
    idxRealAndNegative=np.zeros(n,dtype=bool)
    idxImagAndPositive=np.zeros(n,dtype=bool)
    idxImagAndNegative=np.zeros(n,dtype=bool)
    #===============实数主导,按实数正负分类
    real_dominant_indices=np.where(is_real_dominant)[0]
    for idx in real_dominant_indices:
        if real_parts[idx]>0:
            idxRealAndPositive[idx]=True
        else:
            idxRealAndNegative[idx]=True
    #================虚数主导
    imag_dominant_indices=np.where(is_imag_dominant)[0]
    for idx in imag_dominant_indices:
        if imag_parts[idx]>0:
            idxImagAndPositive[idx]=True
        else:
            idxImagAndNegative[idx]=True
    #================混合特征值,复数
    mixed_indices=np.where(is_mixed)[0]
    for idx in mixed_indices:
        if imag_parts[idx]>0:
            idxImagAndPositive[idx]=True
        else:
            idxImagAndNegative[idx]=True

    eig_real_p=eig[idxRealAndPositive]
    vect_real_p=vect[:,idxRealAndPositive]
    eig_real_n=eig[idxRealAndNegative]
    vect_real_n=vect[:,idxRealAndNegative]
    eig_imag_p=eig[idxImagAndPositive]
    vect_imag_p=vect[:,idxImagAndPositive]
    eig_imag_n=eig[idxImagAndNegative]
    vect_imag_n=vect[:,idxImagAndNegative]

    if len(eig_real_p)!=0:
        ind=np.argsort(eig_real_p.real)[::-1]
        eig_real_p=eig_real_p[ind]
        vect_real_p=vect_real_p[:,ind]
    if len(eig_imag_p)!=0:
        ind=np.argsort(eig_imag_p.imag)
        eig_imag_p=eig_imag_p[ind]
        vect_imag_p=vect_imag_p[:,ind]
    if len(eig_real_n)!=0:
        ind=np.argsort(eig_real_n.real)
        eig_real_n=eig_real_n[ind]
        vect_real_n=vect_real_n[:,ind]
    if len(eig_imag_n)!=0:
        ind=np.argsort(eig_imag_n.imag)[::-1]
        eig_imag_n=eig_imag_n[ind]
        vect_imag_n=vect_imag_n[:,ind]
    
    eig_p=np.block([eig_real_p,eig_imag_p])
    vect_p=np.block([vect_real_p,vect_imag_p])
    eig_n=np.block([eig_real_n,eig_imag_n])
    vect_n=np.block([vect_real_n,vect_imag_n])
    return eig_p,vect_p,eig_n,vect_n

def Perturbation(Matrix):
    temp=Matrix.astype(np.complex64)
    Matrix=temp.astype(np.complex128)
    return Matrix

##################设定仿真常数区域##########################################
Constant={}
n1=1
n2=1.4482+7.5367j
Constant['n1']=n1
Constant['n2']=n2
Constant['thetai']=np.radians(0)
Constant['n_Tr']=2*20+1
Constant['wavelength']=632.8*1e-9
Constant['k0']=2*np.pi/Constant['wavelength']
#set Accuracy
Constant['cut']=0#是否对变换后的傅里叶级数进行去除小数处理
Constant['accuracy']=1e-9
Constant['kVecImagMin']=1e-10
R_effi=[]
##########################################################################

##########以下为光栅常数设定##############
# grating=Triangular(4*1e-6,36,1)
grating=Sinusoidal(4*1e-6,1,2*1e-6)
Constant['period']=grating.T
a=grating.profile()
x=np.linspace(0,Constant['period'],2**10,endpoint=False)
Constant['a']=a(x)#光栅表面轮廓函数
# Constant['a']=Constant['k0']*Roughness(a(x),0.05,42)
dx=Constant['period']/len(x)
a_diff=np.gradient(Constant['a'],dx)
# plt.plot(x,Constant['a'])
# plt.plot(x,a_diff)
# plt.show()
#############################################

##########任意偏振态,为TE、TM偏振态的组合###############################
alpha=90
alpha=np.radians(alpha)
TM=np.cos(alpha)#TM模式的分量
TE=np.sin(alpha)#TE模式的分量
if TM<1e-10:
    TM=0
if TE<1e-10:
    TE=0
if TM!=0:
    Polarization='TM'
    etaR_TM,etaT_TM=Compute(n1,n2,Polarization,Constant)
if TE!=0:
    Polarization='TE'
    etaR_TE,etaT_TE=Compute(n1,n2,Polarization,Constant)
if TM==0:
    etaR_TM=np.zeros_like(etaR_TE)
if TE==0:
    etaR_TE=np.zeros_like(etaR_TM)
polar=TM**2*etaR_TM+TE**2*etaR_TE
Constant['R_effi']=polar
######################################################################
real_Ray1_idx=Constant['real_Ray1_idx']
Plot_Effi(Constant,[],False)