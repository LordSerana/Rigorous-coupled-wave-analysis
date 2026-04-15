import matplotlib.pyplot as plt
import numpy as np
from S_matrix.Grating import Triangular,Sinusoidal,Blazed
from C_Method.SetConstantByPolar import setConstanByPola
from S_matrix.F_series_gen import F_series_gen
from C_Method.Toeplitze import Toeplitz
from C_Method.Eigen import Eigen
from C_Method.Plot_intensity import Plot_intensity
from C_Method.Plot_Effi import Plot_Effi
from scipy.special import jv
from C_Method.SortEigenValue import SortEigenvalueChand as Sort

'''
对于正弦函数的特殊情况,特意重写为此文件
'''

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

def Bessel_fourier_coeffi(beta,depth,k0,m1,m2):
    '''
    计算exp(-j*beta*a(x))的Fourier系数
    a(x)=depth/2*(1+sin(Kx))
    '''
    m=np.arange(m1,m2+1)
    z=beta*k0*depth/2
    coeffi=np.exp(-1j*z)*((-1.0)**m)*jv(m,z)
    return coeffi
    
def GenerateFFieldsChand(a,b0,nDim,k0,d,m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2,Constant):
    # LP_fun=np.exp(-1j*b0*a)#define fourier transform argument for positive fields
    # F_in_0=F_series_gen(LP_fun,nDim)#正场系数
    # q0=len(F_in_0)//2
    # F_in=F_in_0[q0-(nDim-1)//2:q0+(nDim-1)//2+1]
    Fm0=Bessel_fourier_coeffi(b0,Constant['depth'],k0,m1,m2)
    Fmk=np.zeros((nDim,len(real_Ray2_idx)),dtype=np.complex128)#F_mk_R_N
    Fmn=np.zeros((nDim,len(real_Ray1_idx)),dtype=np.complex128)#F_mn_R_P
    if len(real_Ray1_idx)!=0:
        min_ray1=np.min(real_Ray1_idx)
        max_ray1=np.max(real_Ray1_idx)
        for M in range(min_ray1,max_ray1+1):
            # LPn_fun=np.exp(-1j*SB1[M-m1]*a)
            # FRP0=F_series_gen(LPn_fun,nDim)
            beta=SB1[M-m1]
            Fmn0=Bessel_fourier_coeffi(beta,Constant['depth'],k0,-nDim,nDim)
            # FRP[:,M-min_ray1]=Bessel_fourier_coeffi(beta,Constant['depth'],m1,m2)
            q0=len(Fmn0)//2
            part1=np.conj(Fmn0[q0+m1-M:q0])
            part2=np.conj(Fmn0[q0:q0+m2-M+1])
            Fmn[:,M-min_ray1]=np.concatenate([part1,part2])

    if len(real_Ray2_idx)!=0:
        min_ray2=np.min(real_Ray2_idx)
        max_ray2=np.max(real_Ray2_idx)
        for k in range(min_ray2,max_ray2+1):
            # LNn_fun=np.exp(-1j*SB2[k]*a)
            # FRN0=F_series_gen(LNn_fun,nDim)
            beta=SB2[k-m1]
            Fmk0=Bessel_fourier_coeffi(beta,Constant['depth'],k0,-nDim,nDim)
            q0=len(Fmk0)//2
            for m in range(m1,m2+1):
                Fmk[m-m1,k-min_ray2]=Fmk0[q0+m-k]
    return Fm0,Fmk,Fmn

def GenerateGFieldsChand(b0,a_mat,real_eig1p,real_eig2n,SB1,SB2,real_Ray1_idx,real_Ray2_idx,\
    m1,m2,nDim,imag_eig1p,imag_eig2n,Fmn,Fmk,Fm0,Fmq,Fmr,A,eps1,eps2):
    Gmn=np.zeros((nDim,len(real_Ray1_idx)),dtype=complex)
    Gmk=np.zeros((nDim,len(real_Ray2_idx)),dtype=complex)
    Gm0=np.zeros((nDim,1),dtype=complex)
    Gmq=np.zeros((nDim,len(imag_eig1p)),dtype=complex)
    Gmr=np.zeros((nDim,len(imag_eig2n)),dtype=complex)
    AUX_mat=np.eye(nDim)+a_mat@a_mat
    #计算Gmn(实特征值正场)
    if len(real_Ray1_idx)!=0:
        min_ray1=np.min(real_Ray1_idx)
        for m in range(nDim):
            for n in range(len(real_Ray1_idx)):
                sb1_idx=n+min_ray1-m1
                beta_n=SB1[sb1_idx]
                for s in range(nDim):
                    term=a_mat[m,s]*A[s]-AUX_mat[m,s]*beta_n
                    Gmn[m,n]+=term*Fmn[s,n]
    #计算Gmk(实特征值负场)
    if len(real_Ray2_idx)!=0:
        min_ray2=np.min(real_Ray2_idx)
        for m in range(nDim):
            for k in range(len(real_Ray2_idx)):
                sb2_idx=k+min_ray2-m1
                beta_k=SB2[sb2_idx]
                for s in range(nDim):
                    term=a_mat[m,s]*A[s]+AUX_mat[m,s]*beta_k
                    Gmk[m,k]+=term*Fmk[s,k]
    #计算Gm0
    for m in range(nDim):
        for s in range(nDim):
            term=a_mat[m,s]*A[s]+AUX_mat[m,s]*b0
            Gm0[m,0]+=term*Fm0[s]
    #计算Gmq,Gmr(虚特征值场)
    for m in range(nDim):
        for q in range(len(imag_eig1p)):
            rho_q=imag_eig1p[q]
            for s in range(nDim):
                term=a_mat[m,s]*A[s]-AUX_mat[m,s]*rho_q
                Gmq[m,q]+=term*Fmq[s,q]
    for m in range(nDim):
        for r in range(len(imag_eig2n)):
            rho_r=imag_eig2n[r]
            for s in range(nDim):
                term=a_mat[m,s]*A[s]-AUX_mat[m,s]*rho_r
                Gmr[m,r]+=term*Fmr[s,r]
    #归一化处理
    Gmn=(1/eps1)*Gmn
    Gmk=(1/eps2)*Gmk
    Gm0=(1/eps1)*Gm0
    Gmq=(1/eps1)*Gmq
    Gmr=(1/eps2)*Gmr
    return Gmn,Gmk,Gm0,Gmq,Gmr

def Compute(n1,n2,polar,Constant):
    Constant=setConstanByPola(n1,n2,polar,Constant)
    k0=2*np.pi/Constant['wavelength']#波数
    K=2*np.pi/Constant['period']#倒易空间矢量
    nDim=Constant['n_Tr']#计算所用的总模数
    alpha0=Constant['n1']*k0*np.sin(Constant['thetai'])
    m1=int(-(nDim-1)/2)
    m2=int((nDim-1)/2)
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
    Constant['real_Ray2_idx']=real_Ray2_idx
    ###对a_diff_vec采用解析解
    # a_diff_vec=F_series_gen(a_diff,nDim)
    a_diff_vec=np.zeros(2*nDim+1,dtype=complex)
    q0=nDim
    a_diff_vec[q0-1]=Constant['depth']*k0*np.pi/Constant['period']/2
    a_diff_vec[q0+1]=Constant['depth']*k0*np.pi/Constant['period']/2
    ##观察傅里叶频谱分量
    # temp_x=range(-nDim,nDim+1)
    # plt.plot(temp_x,a_diff_vec)
    # plt.show()
    #####
    a_mat=Toeplitz(a_diff_vec,nDim)
    V1,rho1,V2,rho2=Eigen(A,B1,B2,a_mat,nDim)
    # real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n=Sort(V1,rho1,V2,rho2,Constant['accuracy'],nDim)
    real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n=SortEigenvalueChand(V1,rho1,V2,rho2,Constant['accuracy'],Constant)
    Fmq=imag_Vec1p
    Fmr=imag_Vec2n
    #Assemble F-matrices
    b0=np.sqrt(B1[-m1])
    Fm0,Fmk,Fmn=GenerateFFieldsChand(Constant['a'],b0,nDim,k0,Constant['period'],m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2,Constant)
    #Assemble G-matrices
    Gmn,Gmk,Gm0,Gmq,Gmr=GenerateGFieldsChand(b0,a_mat,real_eig1p,real_eig2n,SB1,SB2,real_Ray1_idx,real_Ray2_idx,m1,m2,nDim,imag_eig1p,
    imag_eig2n,Fmn,Fmk,Fm0,Fmq,Fmr,A,Constant['eps1'],Constant['eps2'])
    #Assemble matrix for matching boundary conditions and solve the linear system
    Gm0=Gm0.T
    Gm0=Gm0.flatten()
    MatBC=np.block([[Fmn,Fmq,-Fmk,-Fmr],[Gmn,Gmq,-Gmk,-Gmr]])
    VecBC=np.concatenate((-Fm0,-Gm0),axis=0)
    RVec=np.linalg.solve(MatBC,VecBC)
    #绘制衍射效率曲线
    etaR,etaT=Plot_intensity(Constant,RVec,real_Ray1_idx,real_Ray2_idx,B1,B2,m1,b0,nDim,False)
    etaR=etaR.flatten()
    etaT=etaT.flatten()
    return etaR,etaT

def Roughness(a_func,Ra=0.0,seed=None):
    '''
    为原始光栅轮廓函数附加粗糙度
    参数:
    a_func:原始轮廓函数
    Ra:粗糙度幅度,使用μm作为单位
    seed:随机种子
    '''
    if seed is not None:
        np.random.seed(seed)
    required_std=Ra/np.sqrt(2/np.pi)
    if Ra>0:
        noise=np.random.normal(0,required_std,size=a_func.shape)
        return a_func+noise
    else:
        return a_func

def SortEigenvalueChand(V1,rho1,V2,rho2,tol,Constant):
    nDim=Constant['n_Tr']
    #=============处理D1区域================================
    n_prop1=len(Constant['real_Ray1_idx'])#传播级的数量
    prop1_mask=(np.abs(np.imag(rho1))<tol)&(np.real(rho1)>0)
    evan1_mask=(np.imag(rho1)>tol)
    real_eig1p=rho1[prop1_mask]
    imag_eig1p=rho1[evan1_mask]

    real_vec1p=V1[:,prop1_mask]
    imag_vec1p=V1[:,evan1_mask]
    ###传播模,按实部降序
    idx_prop1=np.argsort(-np.real(real_eig1p))
    real_eig1p=real_eig1p[idx_prop1]
    real_vec1p=real_vec1p[:,idx_prop1]
    ###倏逝模,按虚部升序
    idx_evan1=np.argsort(np.imag(imag_eig1p))
    imag_eig1p=imag_eig1p[idx_evan1]
    imag_vec1p=imag_vec1p[:nDim,:][:,idx_evan1]
    #=========处理D2区域================
    prop2_mask=(np.abs(np.imag(rho2))<tol)&(np.real(rho2)<0)
    evan2_mask=(np.imag(rho2)<-tol)
    real_eig2n=rho2[prop2_mask]
    imag_eig2n=rho2[evan2_mask]

    real_vec2n=V2[:,prop2_mask]
    imag_vec2n=V2[:,evan2_mask]
    ###传播模,按实部绝对值升序
    idx_prop2=np.argsort(np.abs(np.real(real_eig2n)))
    real_eig2n=real_eig2n[idx_prop2]
    real_vec2n=real_vec2n[:,idx_prop2]
    ###倏逝模,按虚部升序
    idx_evan2=np.argsort(np.abs(np.imag(imag_eig2n)))
    imag_eig2n=imag_eig2n[idx_evan2]
    imag_vec2n=imag_vec2n[:nDim,:][:,idx_evan2]
    return real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_vec1p,imag_vec2n

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
depth=2*1e-6
grating=Sinusoidal(4*1e-6,1,depth)
Constant['period']=grating.T
Constant['depth']=depth
a=grating.profile()
x=np.linspace(0,Constant['period'],2**10,endpoint=False)
Constant['a']=Constant['k0']*a(x)#光栅表面轮廓函数
dx=Constant['period']*Constant['k0']/len(x)
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