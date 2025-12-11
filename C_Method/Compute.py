import numpy as np
import math
from SetConstantByPolar import setConstanByPola
from F_series_gen import F_series_gen
from Toeplitze import Toeplitz
from Eigen import Eigen
from SortEigenValue import SortEigenvalueChand
from GenerateFFields import GenerateFFieldsChand
from GenerateGFields import GenerateGFieldsChand
from Plot_intensity import Plot_intensity

def Compute(n1,n2,polar,Constant):
    Constant=setConstanByPola(n1,n2,polar,Constant)
    k0=2*np.pi/Constant['wavelength']#波数
    a_fun=lambda x:k0*Constant['a'](x/k0)
    a_diff_fun=lambda x:Constant['diff_a'](x/k0)
    K=2*np.pi/Constant['gx']#倒易空间矢量
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
    a_diff_vec=F_series_gen(a_diff_fun,10,k0*Constant['gx'],nDim)
    a_mat=Toeplitz(a_diff_vec,nDim)
    V1,rho1,V2,rho2=Eigen(A,B1,B2,a_mat,nDim)
    real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n=SortEigenvalueChand(V1,rho1,V2,rho2,Constant['accuracy'],nDim)
    #Assemble F-matrices
    b0=np.sqrt(B1[-m1])
    F_in,FRN,FRP=GenerateFFieldsChand(a_fun,b0,nDim,k0,Constant['gx'],m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2)
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