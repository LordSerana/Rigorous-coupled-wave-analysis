import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
#本脚本使用Chandezon方法（即坐标变换法），求解非垂直光栅的衍射效率
#废案
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
    
    def profile(self,m):
        '''
        m:快速傅里叶变换阶数
        '''
        N=2**m
        x=np.linspace(0,self.T,N)
        y=np.zeros(N)
        temp=np.tan(self.base_angle)
        for i in range(N):
            if x[i]<=self.T/2:
                y[i]=x[i]*temp
            else:
                y[i]=self.amplitude-(x[i]-self.T/2)*np.tan(self.base_angle)
        y_diff=np.zeros(N)
        y_diff[:int(N//2)]=np.tan(self.base_angle)
        y_diff[int(N//2):]=-np.tan(self.base_angle)
        # plt.plot(x,y)
        # plt.xlabel("x")
        # plt.ylabel("Grating profile")
        # plt.show()
        return y,y_diff

def setConstanByPola(n1,n2,polar):
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
    Constant={'n1':n1,'n2':n2,'eps0':eps0,'eps1':eps1,'eps2':eps2,
    'mu0':mu0,'mu1':mu1,'mu2':mu2,'Z0':Z0}
    return Constant

def CutSmallArrayElements(A,accuracy):
    idx=abs(A)<accuracy
    A[idx]=0
    return A

def Compute():
    m_set,n_set,alpha_m,beta1_m,beta2_m,n1_set,n1_set_ind,n2_set,n2_set_ind=calculateAlphaBetaAndMNset(Constant)
    V1,rho1,V2,rho2=Eigen(alpha_m,beta1_m,beta2_m,Constant)
    L_mn_beta1,L_mk_beta2,L_m0_beta1=Compute_L_Arrays(m_set,Constant,beta1_m,beta2_m,n1_set,n1_set_ind,n2_set,n2_set_ind)
    eig1_p,vec1_p,temp1,temp2=SortEigenvalueChand(rho1,V1,Constant)
    temp1,temp2,eig2_m,vec2_m=SortEigenvalueChand(rho2,V2,Constant)
    #Assemble F-matrices
    F_mn_r_p=L_mn_beta1
    F_mk_r_m=L_mk_beta2
    F_m0_r_in=L_m0_beta1
    F_mq_p=vec1_p[:Constant['n_Tr'],len(n1_set):-1]
    F_mr_m=vec2_m[:Constant['n_Tr'],len(n2_set):-1]
    if Constant['cut']==1:
        F_mn_r_p=CutSmallArrayElements(F_mn_r_p,Constant['accuracy'])
        F_mk_r_m=CutSmallArrayElements(F_mk_r_m,Constant['accuracy'])
        F_m0_r_in=CutSmallArrayElements(F_m0_r_in,Constant['accuracy'])
        F_mq_p=CutSmallArrayElements(F_mq_p,Constant['accuracy'])
        F_mr_m=CutSmallArrayElements(F_mr_m,Constant['accuracy'])
    #Assemble G-matrices
    G_mn_r_p,G_mk_r_m,G_m0_r_in,G_mq_p,G_mr_m=GenerateGFieldsChand(m_set,n1_set,n1_set_ind,n2_set,n2_set_ind,alpha_m,\
    Constant['diff_a'],beta1_m,beta2_m,L_mn_beta1,L_mk_beta2,L_m0_beta1,eig1_p,F_mq_p,eig2_m,F_mr_m,Constant)
    #Assemble matrix for matching boundary conditions and solve the linear system
    GF_matrix=np.block([[F_mn_r_p,F_mq_p,-F_mk_r_m,-F_mr_m],[G_mn_r_p,G_mq_p,-G_mk_r_m,-G_mr_m]])
    GF_in=-np.concatenate([F_m0_r_in,G_m0_r_in])
    Amplitude=np.linalg.solve(GF_matrix,GF_in)
    #绘制衍射效率曲线
    R,T=Plot_intensity(Amplitude,n1_set,n1_set_ind,n2_set,n2_set_ind,beta1_m,beta2_m,Constant)
    return R,T

def calculateAlphaBetaAndMNset(Constant):
    k=2*np.pi/Constant['gx']#inverse lattice vector
    k0=2*np.pi/Constant['wavelength']
    m1=-int((Constant['n_Tr']-1)/2)
    m2=-m1
    m_set=np.linspace(m1,m2,m2-m1+1,dtype=int)
    n_set=np.linspace(1-Constant['n_Tr'],Constant['n_Tr']-1,2*Constant['n_Tr']-1,dtype=int)
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

def Compute_L_Arrays(m_set,Constant,beta1_m,beta2_m,n1_set,n1_set_ind,n2_set,n2_set_ind):
    #L_mn_beta1
    idx_m=0
    idx_n=0
    if n1_set.size!=0:
        L_mn_beta1=np.zeros((len(m_set),len(n1_set)))
        for m in range(m_set[0],m_set[-1]+1):
            for n in range(n1_set[0],n1_set[-1]+1):
                L_mn_beta1[idx_m,idx_n]=L_integrate(beta1_m[n1_set_ind[idx_n]],m-n,Constant)
                idx_n+=1
            idx_n=0
            idx_m+=1
    else:
        L_mn_beta1=[]
    #L_mk_beta2
    idx_m=0
    idx_k=0
    if n2_set.size!=0:
        L_mk_beta2=np.zeros((len(m_set),len(n2_set)))
        for m in range(m_set[0],m_set[-1]+1):
            for k in range(n2_set[0],n2_set[-1]+1):
                L_mk_beta2[idx_m,idx_k]=L_integrate(-beta2_m[n2_set_ind[idx_k]],m-k,Constant)
                idx_k+=1
            idx_k=0
            idx_m+=1
    else:
        L_mk_beta2=[]
    #L_m0_beta1
    idx=0
    L_m0_beta1=np.zeros(shape=len(m_set))
    for m in range(m_set[0],m_set[-1]+1):
        L_m0_beta1[idx]=L_integrate(-beta1_m[Constant['n_Tr']//2],m,Constant)
        idx+=1
    return L_mn_beta1,L_mk_beta2,L_m0_beta1
    
def L_integrate(gamma,m,Constant):
    #用于计算积分L
    FF=(1/Constant['gx'])*np.exp(1j*gamma*Constant['a']-1j*m*2*np.pi*Constant['x']/Constant['gx'])
    L=simpson(FF,Constant['x'])#simpson积分法
    return L

def F_series_gen(profile,nDim,cut_small=True):
    '''
    profile:光栅轮廓函数,m:FFT级次,T:周期长度,nDim:模态数
    该函数返回0~+nDim级谐波
    '''
    fourier_coeffi=np.fft.fft(profile,axis=0)/profile.shape[0]
    tol=1e-10
    #减去系数中的小量
    if cut_small==True:
        ind_small_real=(np.abs(np.real(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_real]=1j*np.imag(fourier_coeffi[ind_small_real])
        ind_small_imag=(np.abs(np.imag(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_imag]=np.real(fourier_coeffi[ind_small_imag])
    return fourier_coeffi[:nDim+1]

def Toeplitz(fourier_coeffi,nDim):
    '''
    构造Toeplitz矩阵
    '''
    A=np.zeros((nDim,nDim),dtype=complex)
    for i in range(nDim):
        for j in range(nDim):
            if j<i:
                A[i,j]=np.conj(fourier_coeffi[i-j])
            else:
                A[i,j]=fourier_coeffi[j-i]
    return A
    
def Eigen(alpha_m,beta1_m,beta2_m,Constant):
    '''
    求解特征值和特征向量
    '''
    a_diff=F_series_gen(Constant['diff_a'],Constant['n_Tr'])
    a_mat=Toeplitz(a_diff,Constant['n_Tr'])
    Mat11=-np.diag(1/beta1_m**2)@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m))
    Mat12=np.diag(1/beta1_m**2)@(np.eye(Constant['n_Tr'])+a_mat@a_mat)
    Mat13=np.eye(Constant['n_Tr'])
    Mat14=np.zeros((Constant['n_Tr'],Constant['n_Tr']))
    Mat1=np.block([[Mat11,Mat12],[Mat13,Mat14]])
    Mat21=-np.diag(1/beta2_m**2)@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m))
    Mat22=np.diag(1/beta2_m**2)@(np.eye(Constant['n_Tr'])+a_mat@a_mat)
    Mat23=np.eye(Constant['n_Tr'])
    Mat24=np.zeros((Constant['n_Tr'],Constant['n_Tr']))
    Mat2=np.block([[Mat21,Mat22],[Mat23,Mat24]])
    rho1,V1=np.linalg.eig(Mat1)
    rho2,V2=np.linalg.eig(Mat2)
    rho1=1/rho1#eigenvalue of ChandM1,计算的是原本的ρ值
    rho2=1/rho2#eigenvalue of ChandM2,计算的是原本的ρ值
    return V1,rho1,V2,rho2

def SortEigenvalueChand(eig,vec,Constant):
    tol=Constant['accuracy']
    idxRealAndPositive=abs(np.imag(eig)<tol)&(np.real(eig)>0)
    idxRealAndNegative=abs(np.imag(eig)<tol)&(np.real(eig)<=0)
    idxImagAndPositive=abs(np.imag(eig)>=tol)&(np.imag(eig)>0)
    idxImagAndNegative=abs(np.imag(eig)>=tol)&(np.imag(eig)<=0)
    eig_real_p=eig[idxRealAndPositive]
    vec_real_p=vec[:,idxRealAndPositive]
    eig_real_m=eig[idxRealAndNegative]
    vec_real_m=vec[:,idxRealAndNegative]
    eig_comp_p=eig[idxImagAndPositive]
    vec_comp_p=vec[:,idxImagAndPositive]
    eig_comp_m=eig[idxImagAndNegative]
    vec_comp_m=vec[:,idxImagAndNegative]
    #sort in descend order
    if ~eig_real_p.size==0:
        ind=np.argsort(-np.real(eig_real_p))
        eig_real_p=eig_real_p[ind]
        vec_real_p=vec_real_p[:,ind]
    if ~eig_comp_p.size==0:
        ind=np.argsort(np.imag(eig_comp_p))
        eig_comp_p=eig_comp_p[ind]
        vec_comp_p=vec_comp_p[:,ind]
    if ~eig_real_m.size==0:
        ind=np.argsort(np.real(eig_real_m))
        eig_real_m=eig_real_m[ind]
        vec_real_m=vec_real_m[:,ind]
    if ~eig_comp_p.size==0:
        ind=np.argsort(-np.imag(eig_comp_m))
        eig_comp_m=eig_comp_m[ind]
        vec_comp_m=vec_comp_m[:,ind]
    eig_p=np.concatenate([eig_real_p,eig_comp_p])
    vec_p=np.concatenate([vec_real_p,vec_comp_p],axis=1)
    eig_m=np.concatenate([eig_real_m,eig_comp_m])
    vec_m=np.concatenate([vec_real_m,vec_comp_m],axis=1)
    return eig_p,vec_p,eig_m,vec_m

def GenerateGFieldsChand(m_set,n1_set,n1_set_ind,n2_set,n2_set_ind,alpha_m,a_diff,beta1_m,beta2_m,\
    L_mn_beta1,L_mk_beta2,L_m0_beta1,eig1_p,F_mq_p,eig2_m,F_mr_m,Constant):
    a_mat=Toeplitz(a_diff,Constant['n_Tr'])
    a_sq=np.eye(Constant['n_Tr'])+a_mat@a_mat
    #G_mn_r_p
    if len(n1_set_ind)!=0:
        G_mn_r_p=np.zeros((len(m_set),len(n1_set)))
        for m in range(len(m_set)):
            for n in range(len(n1_set)):
                G_mn_s=np.zeros(len(m_set))
                for s in range(len(m_set)):
                    G_mn_s[s]=(a_mat[m,s]*alpha_m[s]-a_sq[m,s]*beta1_m[n1_set_ind[n]])*L_mn_beta1[s,n]
                G_mn_r_p[m,n]=(Constant['Z0']/(2*np.pi/Constant['wavelength'])/Constant['eps1'])*sum(G_mn_s)
    else:
        G_mn_r_p=[]
    #G_mk_r_m
    if len(n2_set_ind)!=0:
        G_mk_r_m=np.zeros((len(m_set),len(n2_set)))
        for m in range(m_set):
            for k in range(n2_set):
                G_mk_s=np.zeros(len(m_set))
                for s in range(m_set):
                    G_mk_s[s]=(a_mat[m,s]*alpha_m[s]+a_sq[m,s]*beta2_m[n2_set_ind[k]])*L_mk_beta2[s,k]
                G_mk_r_m[m,k]=(Constant['Z0']/(2*np.pi/Constant['wavelength'])/Constant['eps2'])*sum(G_mk_s)
    else:
        G_mk_r_m=[]
    #G_m0_r_in
    G_m0_r_in=np.zeros(len(m_set))
    for m in range(len(m_set)):
        G_m0_s=np.zeros(len(m_set))
        for s in range(len(m_set)):
            G_m0_s[s]=(a_mat[m,s]*alpha_m[s]+a_sq[m,s]*beta1_m[Constant['n_Tr']//2])*L_m0_beta1[s]
        G_m0_r_in[m]=(Constant['Z0']/(2*np.pi/Constant['wavelength'])/Constant['eps1'])*sum(G_m0_s)
    #G_mq_p
    if len(n1_set)<Constant['n_Tr']:
        G_mq_p=np.zeros(len(m_set))
        for m in range(len(m_set)):
            for q in range(len(Constant['n_Tr']-len(n1_set))):
                G_mq_s=np.zeros(len(m_set))
                for s in range(len(m_set)):
                    G_mq_s[s]=(a_mat[m,s]*alpha_m[s]-a_sq[m,s]*eig1_p[len(n1_set)+q])*F_mq_p[s,q]
                G_mq_p[m,q]=(Constant['Z0']/(2*np.pi/Constant['wavelength'])/Constant['eps1'])*sum(G_mq_s)
    else:
        G_mq_p=[]
    #G_mr_m
    if len(n2_set)<Constant['n_Tr']:
        G_mr_m=np.zeros((len(m_set),Constant['n_Tr']-len(n2_set)))
        for m in range(m_set):
            for r in range(Constant['n_Tr']-len(n2_set)):
                G_mr_s=np.zeros(len(m_set))
                for s in range(len(m_set)):
                    G_mr_s[s]=(a_mat[m,s]*alpha_m[s]-a_sq[m,s]*eig2_m[len(n2_set)+r])*F_mr_m[s,r]
                G_mr_m[m,r]=(Constant['Z0']/(2*np.pi/Constant['wavelength'])/Constant['eps2'])*sum(G_mr_s)
    else:
        G_mr_m=[]
    return G_mn_r_p,G_mk_r_m,G_m0_r_in,G_mq_p,G_mr_m

def Plot_intensity(Amplitude,n1_set,n1_set_ind,n2_set,n2_set_ind,beta1_m,beta2_m,Constant):
    '''
    RVec:计算的最终结果，包含衍射效率信息;real_Ray1_idx:反射级的有效传播阶数
    real_Ray2_idx:透射级的有效传播阶数;B1:对应论文β1;B2:对应论文β2
    m1:左截断阶数,对应光栅衍射级即为-6级;b0:对应论文β0
    nDim:总有效衍射级;plot:是否绘图,输入True或False
    '''
    #calculate reflected order
    R=np.zeros(len(n1_set))
    for i in range(len(n1_set)):
        R[i]=(beta1_m[n1_set_ind[i]]/beta1_m[Constant['n_Tr']//2])*abs(Amplitude[i])**2
    #calaulate trans order
    T=np.zeros(len(n1_set))
    if ~n2_set.size==0:
        for i in range(len(n2_set)):
            T[i]=(Constant['eps1']*beta2_m[n2_set_ind[i]]/(Constant['eps2']*beta1_m[Constant['n_Tr']//2]))*(abs(Amplitude[Constant['n_Tr']+i]))**2
    return R,T

##################设定仿真常数区域##########################################
Polarization='TE'
n1=1
n2=1.4482+7.5367j
Constant=setConstanByPola(n1,n2,Polarization)
Constant['thetai']=np.radians(1e-4)
Constant['n_Tr']=2*15+1#谐波截断阶数
Constant['wavelength']=632.8*1e-9
Constant['gx']=4#结构x方向上的周期
#set Accuracy
Constant['cut']=0#是否对变换后的傅里叶级数进行去除小数处理
Constant['accuracy']=1e-12
Constant['kVecImagMin']=1e-10
##########################################################################

##########以下为三角光栅常数设定
grating=Triangular(4*1e-6,30,1)
a,a_diff=grating.profile(10)
x=np.linspace(0,Constant['gx'],len(a))
Constant['x']=x
Constant['a']=a#光栅表面轮廓函数
Constant['diff_a']=a_diff#光栅表面轮廓的导数
R,T=Compute()
pass