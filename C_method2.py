import numpy as np
import matplotlib.pyplot as plt
import math
#C1脚本计算结果不收敛，最大值达1e9。debug······
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

class Triangular():
    def __init__(self,T,base_angle,fill_factor):
        '''
        T:周期,base_angle:三角光栅底角,fill_factor:占空比,amplitude:光栅槽深
        n1:入射区域折射率,n2:透射区域折射率
        '''
        self.name="Triangular"
        self.T=T
        self.base_angle=np.radians(base_angle)
        self.fill_factor=fill_factor
        self.amplitude=T*fill_factor/2*np.tan(self.base_angle)
    
    def profile(self):
        '''
        m:快速傅里叶变换阶数
        '''
        a_fun=lambda x:(x*np.tan(self.base_angle))*(x<=self.T/2)+(self.amplitude-(x-self.T/2)*np.tan(self.base_angle))*(x>self.T/2)
        a_diff_fun=lambda x:(np.tan(self.base_angle))*(x<=self.T/2)+(-np.tan(self.base_angle))*(x>self.T/2)
        # plt.plot(x,y)
        # plt.xlabel("x")
        # plt.ylabel("Grating profile")
        # plt.show()
        return a_fun,a_diff_fun

def setConstanByPola(n1,n2,polar,Constant):
    '''
    d:光栅周期,h:槽深,profile:光栅槽型,n1、n2:入射、出射区域折射率,
    thetai:入射角,lam:波长,polar:极化状态,numTr:截断阶数,tol:误差等级
    '''
    eps0=8.8541878171e-12
    mu0=12.5663706141e-7
    mu1=1
    mu2=1
    eps1=n1**2/mu1
    eps2=n2**2/mu2
    if polar=='TE':
        mu0,eps0=-eps0,-mu0
        mu1,eps1=eps1,-mu1
        mu2,eps2=eps2,-mu2
    Z0=np.sqrt(mu0/eps0)
    Constant['n1']=n1
    Constant['n2']=n2
    Constant['eps0']=eps0
    Constant['eps1']=eps1
    Constant['eps2']=eps2
    Constant['mu0']=mu0
    Constant['mu1']=mu1
    Constant['mu2']=mu2
    Constant['Z0']=Z0
    return Constant

def Compute(eps1,eps2):
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
    imag_eig2n,FRP,FRN,F_in,imag_Vec1p,imag_Vec2n,A,eps1,eps2)
    #Assemble matrix for matching boundary conditions and solve the linear system
    G_in=G_in.T
    G_in=G_in.flatten()
    MatBC=np.block([[FRP,imag_Vec1p,-FRN,-imag_Vec2n],[G_RP,G_P,-G_RN,-G_N]])
    VecBC=np.concatenate((-F_in,-G_in),axis=0)
    RVec=np.linalg.inv(MatBC)@VecBC
    #绘制衍射效率曲线
    etaR,etaT=Plot_intensity(RVec,real_Ray1_idx,real_Ray2_idx,B1,B2,m1,b0,nDim,False)
    return etaR,etaT

def calculateAlphaBetaAndMNset(Constant):
    k=2*np.pi/Constant['gx']#inverse lattice vector
    k0=2*np.pi/Constant['wavelength']
    m1=-Constant['n_Tr']//2
    m2=-m1
    m_set=np.linspace(m1,m2,m2-m1)+1
    n_set=np.linspace(1-Constant['n_Tr'],Constant['n_Tr']-1,2*Constant['n_Tr']-1)
    alpha_m=k0*Constant['n1']*np.sin(Constant['thetai'])+k*m_set
    temp=(k0*Constant['n1'])**2
    beta1_m=np.diag(np.sqrt(np.eye(Constant['n_Tr'])*temp-np.diag(alpha_m**2)))
    temp=(k0*Constant['n2'])**2
    beta2_m=np.diag(np.sqrt(np.eye(Constant['n_Tr'])*temp-np.diag(alpha_m**2)))
    #寻找实数beta1_m索引
    real_mask=np.abs(np.imag(beta1_m))<Constant['accuracy']
    n1_set=m_set[real_mask]#实数解的集合
    n1_set_ind=np.where(real_mask)#其对应的索引
    real_mask2=np.abs(np.imag(beta2_m))<Constant['accuracy']
    n2_set=m_set[real_mask2]
    n2_set_ind=np.where(real_mask2)
    return m_set,n_set,alpha_m,beta1_m,beta2_m,n1_set,n1_set_ind,n2_set,n2_set_ind

def F_series_gen(a_fun,m,T,nDim,cut_small=True):
    '''
    profile:光栅轮廓函数,m:FFT级次,T:周期长度,nDim:模态数
    该函数返回-N/2~N/2级谐波
    '''
    N=2**m
    tol=1e-9
    x=np.linspace(0,T,N+1)
    f=a_fun(x)
    fhat=f
    fhat[0]=(f[0]+f[-1])/2
    fhat=fhat[:-1]
    fourier_coeffi=np.fft.fftshift(np.fft.fft(fhat)/fhat.shape[0])
    #减去系数中的小量
    if cut_small==True:
        ind_small_real=(np.abs(np.real(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_real]=1j*np.imag(fourier_coeffi[ind_small_real])
        ind_small_imag=(np.abs(np.imag(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_imag]=np.real(fourier_coeffi[ind_small_imag])
    q0=len(fourier_coeffi)//2
    return fourier_coeffi[q0-nDim:q0+nDim+1]

def Toeplitz(fourier_coeffi,nDim):
    '''
    构造Toeplitz矩阵
    '''
    A=np.zeros((nDim,nDim),dtype=complex)
    q0=len(fourier_coeffi)//2
    for i in range(nDim):
        for j in range(nDim):
            A[i,j]=fourier_coeffi[q0+i-j]
    return A
    
def Eigen(A,B1,B2,a_mat,nDim):
    '''
    求解特征值和特征向量
    '''
    IB1=np.linalg.solve(np.diag(B1),np.eye(nDim))
    IB2=np.linalg.solve(np.diag(B2),np.eye(nDim))
    AUX_mat=np.eye(nDim)+a_mat@a_mat
    #eigenvalue matrix from (12),incident medium
    ChandM1=np.block([[-IB1@(np.diag(A)@a_mat+a_mat@np.diag(A)),IB1@AUX_mat],[np.eye(nDim),np.zeros((nDim,nDim))]])
    #eigenvalue matrix from (12),transmission medium
    ChandM2=np.block([[-IB2@(np.diag(A)@a_mat+a_mat@np.diag(A)),IB2@AUX_mat],[np.eye(nDim),np.zeros((nDim,nDim))]])
    rho1,V1=np.linalg.eig(ChandM1)
    rho2,V2=np.linalg.eig(ChandM2)
    rho1=1/rho1#eigenvalue of ChandM1,计算的是原本的ρ值
    rho2=1/rho2#eigenvalue of ChandM2,计算的是原本的ρ值
    return V1,rho1,V2,rho2

def SortEigenvalueChand(V1,rho1,V2,rho2,tol,nDim):
    #分离实特征值，入射介质
    real_eig1p_ind=(abs(np.imag(rho1))<tol)&(np.real(rho1)>tol)
    real_eig1p=rho1[real_eig1p_ind]
    sort_idx1=np.argsort(-real_eig1p)
    real_eig1p=real_eig1p[sort_idx1]
    #分离实特征值，透射介质
    real_eig2n_ind=(abs(np.imag(rho2))<tol)&(np.real(rho2)<-tol)
    real_eig2n=rho2[real_eig2n_ind]
    sort_idx2=np.argsort(-real_eig2n)
    real_eig2n=real_eig2n[sort_idx2]
    #分离虚特征值，入射介质
    imag_eig1p_ind=(np.imag(rho1)>tol)
    imag_eig1p=rho1[imag_eig1p_ind]
    sort_idx3=np.argsort(np.abs(np.imag(imag_eig1p)))
    imag_eig1p=imag_eig1p[sort_idx3]
    #分离虚特征值，透射介质
    imag_eig2n_ind=(np.imag(rho2)<-tol)
    imag_eig2n=rho2[imag_eig2n_ind]
    sort_idx4=np.argsort(np.abs(np.imag(imag_eig2n)))
    imag_eig2n=imag_eig2n[sort_idx4]
    #提取并排序对应的特征向量
    #入射介质虚部特征向量
    s_imag_Vec1p=V1[:nDim,:][:,imag_eig1p_ind]
    imag_Vec1p=s_imag_Vec1p[:,sort_idx3]
    #透射介质虚部特征向量
    s_imag_Vec2n=V2[:nDim,:][:,imag_eig2n_ind]
    imag_Vec2n=s_imag_Vec2n[:,sort_idx4]
    return real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n

def GenerateFFieldsChand(a_fun,b0,nDim,k0,d,m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2):
    LP_fun=lambda x:np.exp(-1j*b0*a_fun(x))#define fourier transform argument for positive fields
    F_in_0=F_series_gen(LP_fun,10,k0*d,nDim)#正场系数
    q0=len(F_in_0)//2
    F_in=F_in_0[q0-(nDim-1)//2:q0+(nDim-1)//2+1]
    FRN=np.zeros((nDim,len(real_Ray2_idx)),dtype=np.complex128)#F_mk_R_N
    FRP=np.zeros((nDim,len(real_Ray1_idx)),dtype=np.complex128)#F_mn_R_P
    if len(real_Ray1_idx)!=0:
        min_ray1=np.min(real_Ray1_idx)
        max_ray1=np.max(real_Ray1_idx)
        for M in range(min_ray1,max_ray1+1):
            LPn_fun=lambda x:np.exp(-1j*SB1[M-m1]*a_fun(x))
            FRP0=F_series_gen(LPn_fun,12,k0*d,nDim)
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
            LNn_fun=np.exp(-1j*SB2[k]*a_fun)
            FRN0=F_series_gen(LNn_fun,nDim)
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
    #计算G_RP(实特征值负场)
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
    G_RP=(1/eps1)*G_RP
    G_RN=(1/eps2)*G_RN
    G_in=(1/eps1)*G_in
    G_P=(1/eps1)*G_P
    G_N=(1/eps2)*G_N
    return G_RP,G_RN,G_in,G_P,G_N

def Plot_intensity(RVec,real_Ray1_idx,real_Ray2_idx,B1,B2,m1,b0,nDim,plot):
    '''
    RVec:计算的最终结果，包含衍射效率信息;real_Ray1_idx:反射级的有效传播阶数
    real_Ray2_idx:透射级的有效传播阶数;B1:对应论文β1;B2:对应论文β2
    m1:左截断阶数,对应光栅衍射级即为-6级;b0:对应论文β0
    nDim:总有效衍射级;plot:是否绘图,输入True或False
    '''
    #calculate reflected order
    etaR=np.zeros((1,len(real_Ray1_idx)),dtype=float)
    etaT=np.zeros((1,len(real_Ray1_idx)),dtype=float)
    for i in range(min(real_Ray1_idx),max(real_Ray1_idx)+1):
        idx_etaR=i-min(real_Ray1_idx)
        idx_B1=i-m1
        etaR[0,idx_etaR]=np.sqrt(B1[idx_B1])/b0*np.abs(RVec[idx_etaR])**2
    #calaulate trans order
    try:
        min_ray2=np.min(real_Ray2_idx)
    except ValueError as e:
        if "zero-size array to reduction operation" in str(e):
            min_ray2=None
        else:
            raise
    if min_ray2==None:
        pass
    else:
        for i in range(min(real_Ray2_idx),max(real_Ray2_idx)+1):
            idx_etaT=i-min(real_Ray2_idx)
            idx_B2=i-m1
            idx_RVec=idx_etaT+nDim
            etaT[0,idx_etaT]=np.abs(Constant['eps1']/Constant['eps2'])*(np.sqrt(B2[idx_B2])/b0)*np.abs(RVec[idx_RVec])**2
    if plot:
        x=np.linspace(min(real_Ray1_idx),max(real_Ray1_idx),max(real_Ray1_idx)-min(real_Ray1_idx)+1,dtype=int)
        plt.plot(x,etaR[0],label='Reflection')
        if etaT.all()==0:
            pass
        else:
            plt.plot(x,etaT[0],label='Transmission')
        plt.legend()
        plt.xlabel("Diffraction order")
        plt.ylabel("Diffraction efficiency")
        plt.show()
    return etaR,etaT

##################设定仿真常数区域##########################################
n1=1
n2=1.4482+7.5367j
Constant={}
Constant['thetai']=np.radians(1e-4)
Constant['n_Tr']=2*15+1#谐波截断阶数
Constant['wavelength']=632.8*1e-9
Constant['gx']=4*1e-6#结构x方向上的周期
#set Accuracy
Constant['cut']=0#是否对变换后的傅里叶级数进行去除小数处理
Constant['accuracy']=1e-9
Constant['kVecImagMin']=1e-10
##########################################################################

##########以下为三角光栅常数设定
grating=Triangular(4*1e-6,30,1)
a,a_diff=grating.profile()
Constant['a']=a#光栅表面轮廓函数
Constant['diff_a']=a_diff#光栅表面轮廓的导数
##########开始计算
Polarization='TM'
Constant=setConstanByPola(n1,n2,Polarization,Constant)
etaR_TM,etaT_TM=Compute(Constant['eps1'],Constant['eps2'])
Polarization='TE'
Constant=setConstanByPola(n1,n2,Polarization,Constant)
etaR_TE,etaT_TE=Compute(Constant['eps1'],Constant['eps2'])
##########任意偏振态,为TE、TM偏振态的组合
a=0#TM模式的分量
b=1#TE模式的分量
polar=(a/(a+b))*etaR_TM+(b/(a+b))*etaR_TE
polar=polar.flatten()
real_Ray1_idx=Constant['real_Ray1_idx']
x=np.linspace(min(real_Ray1_idx),max(real_Ray1_idx),max(real_Ray1_idx)-min(real_Ray1_idx)+1,dtype=int)
plt.plot(x,polar,label='Reflection')
plt.legend()
plt.xlabel("Diffraction order")
plt.ylabel("Diffraction efficiency")
plt.show()
print(polar)
print("sum:"+str(sum(polar)))