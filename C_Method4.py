import matplotlib.pyplot as plt
import numpy as np
from S_matrix.Grating import Triangular
from C_Method.SetConstantByPolar import setConstanByPola
import math
from S_matrix.F_series_gen import F_series_gen
from C_Method.Toeplitze import Toeplitz
from C_Method.Eigen import Eigen
from C_Method.SortEigenValue import SortEigenvalueChand
from C_Method.GenerateGFields import GenerateGFieldsChand
from C_Method.Plot_intensity import Plot_intensity

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

def GenerateFFieldsChand(a,b0,nDim,k0,d,m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2):
    LP_fun=np.exp(-1j*b0*a)#define fourier transform argument for positive fields
    F_in_0=F_series_gen(LP_fun,nDim)#正场系数
    q0=len(F_in_0)//2
    F_in=F_in_0[q0-(nDim-1)//2:q0+(nDim-1)//2+1]
    FRN=np.zeros((nDim,len(real_Ray2_idx)),dtype=np.complex128)#F_mk_R_N
    FRP=np.zeros((nDim,len(real_Ray1_idx)),dtype=np.complex128)#F_mn_R_P
    if len(real_Ray1_idx)!=0:
        min_ray1=np.min(real_Ray1_idx)
        max_ray1=np.max(real_Ray1_idx)
        for M in range(min_ray1,max_ray1+1):
            LPn_fun=np.exp(-1j*SB1[M-m1]*a)
            FRP0=F_series_gen(LPn_fun,nDim)
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
            FRN0=F_series_gen(LNn_fun,nDim)
            q0=len(FRN0)//2
            for m in range(m1,m2+1):
                FRN[m,k]=FRN0[q0+m-k]
    else:
        FRN=np.zeros((nDim,len(real_Ray2_idx)))
    return F_in,FRN,FRP

def Compute(n1,n2,polar,Constant):
    Constant=setConstanByPola(n1,n2,polar,Constant)
    k0=2*np.pi/Constant['wavelength']#波数
    K=2*np.pi/Constant['period']#倒易空间矢量
    nDim=Constant['n_Tr']#计算所用的总模数
    alpha0=Constant['n1']*k0*np.sin(Constant['thetai'])
    m1=int(-math.floor(alpha0/K)-(nDim-1)/2)
    m2=int(-math.floor(alpha0/K)+(nDim-1)/2)
    nDim=m2-m1+1
    A=(alpha0*np.ones((1,m2-m1+1))+K*np.linspace(m1,m2,nDim))/k0#文献中的alpham
    A=A.flatten()
    B1=Constant['n1']**2-A*A#beta1**2
    B1=B1.flatten()
    B2=Constant['n2']**2-A*A#beta2**2
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
    a_diff_vec=F_series_gen(a_diff,nDim)
    a_mat=Toeplitz(a_diff_vec,nDim)
    V1,rho1,V2,rho2=Eigen(A,B1,B2,a_mat,nDim)
    real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n=SortEigenvalueChand(V1,rho1,V2,rho2,Constant['accuracy'],nDim)
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
    VecBC=np.concatenate((-F_in,-G_in),axis=0)
    RVec=np.linalg.inv(MatBC)@VecBC
    #绘制衍射效率曲线
    etaR,etaT=Plot_intensity(Constant,RVec,real_Ray1_idx,real_Ray2_idx,B1,B2,m1,b0,nDim,False)
    etaR=etaR.flatten()
    etaT=etaT.flatten()
    return etaR,etaT

##################设定仿真常数区域##########################################
Constant={}
n1=1
n2=1.4482+7.5367j
Constant['n1']=n1
Constant['n2']=n2
Constant['thetai']=np.radians(1e-4)
Constant['n_Tr']=2*40+1
Constant['wavelength']=632.8*1e-9
Constant['k0']=2*np.pi/Constant['wavelength']
#set Accuracy
Constant['cut']=0#是否对变换后的傅里叶级数进行去除小数处理
Constant['accuracy']=1e-9
Constant['kVecImagMin']=1e-10
R_effi=[]
##########################################################################

##########以下为光栅常数设定##############
grating=Triangular(4*1e-6,30,1)
Constant['period']=grating.T
a=grating.profile()
x=np.linspace(0,Constant['period']*Constant['k0'],2**10)
Constant['a']=Constant['k0']*a(x/Constant['k0'])#光栅表面轮廓函数
a_diff=np.gradient(Constant['a'],Constant['period']*Constant['k0']/2**10)
#############################################

##########任意偏振态,为TE、TM偏振态的组合###############################
alpha=45
alpha=np.radians(alpha)
a=np.cos(alpha)#TM模式的分量
b=np.sin(alpha)#TE模式的分量
if a!=0:
    Polarization='TM'
    etaR_TM,etaT_TM=Compute(n1,n2,Polarization,Constant)
if b!=0:
    Polarization='TE'
    etaR_TE,etaT_TE=Compute(n1,n2,Polarization,Constant)
if a==0:
    etaR_TM=np.zeros_like(etaR_TE)
if b==0:
    etaR_TE=np.zeros_like(etaR_TM)
polar=a**2*etaR_TM+b**2*etaR_TE
######################################################################
real_Ray1_idx=Constant['real_Ray1_idx']
x=np.linspace(min(real_Ray1_idx),max(real_Ray1_idx),max(real_Ray1_idx)-min(real_Ray1_idx)+1,dtype=int)
plt.plot(x,polar,label='Reflection')
plt.legend()
plt.xlabel("Diffraction order")
plt.ylabel("Diffraction efficiency")
plt.show()
print(polar)
print("sum:"+str(sum(polar)))