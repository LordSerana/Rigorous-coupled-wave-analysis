import matplotlib.pyplot as plt
import numpy as np
from S_matrix.Grating import Triangular,Sinusoidal
from S_matrix.F_series_gen import F_series_gen
from C_Method.Toeplitz import Toeplitz

#一般说来，对A,B1,B2的归一化能够降低矩阵条件数,且最终计算结果不变
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

def SetConstant(n1,n2,polar,period,n_Tr,lam,thetai):
    '''
    d:光栅周期,h:槽深,profile:光栅槽型,n1、n2:入射、出射区域折射率,
    thetai:入射角,lam:波长,polar:极化状态,numTr:截断阶数,tol:误差等级
    '''
    eps0=8.8541878171*1e-12
    mu0=12.5663706141*1e-7
    mu1=1
    mu2=1
    eps1=n1**2/mu1
    eps2=n2**2/mu2
    if polar=='TE':
        mu0,eps0=-eps0,-mu0
        mu1,eps1=eps1,-mu1
        mu2,eps2=eps2,-mu2
    Z0=np.sqrt(mu0/eps0)
    Constant={}
    Constant['period']=period
    Constant['n1']=n1
    Constant['n2']=n2
    Constant['eps0']=eps0
    Constant['eps1']=eps1
    Constant['eps2']=eps2
    Constant['mu0']=mu0
    Constant['mu1']=mu1
    Constant['mu2']=mu2
    Constant['Z0']=Z0
    Constant['k0']=2*np.pi/lam
    Constant['K']=2*np.pi/period
    Constant['n_Tr']=n_Tr
    m1=int(-((n_Tr-1)/2))
    m2=int(-m1)
    m_set=np.linspace(m1,m2,m2-m1+1,dtype=int)
    n_set=np.linspace(1-n_Tr,n_Tr-1,2*n_Tr-1,dtype=int)
    Constant['m_set']=m_set
    Constant['n_set']=n_set
    alpha_m=n1*Constant['k0']*np.sin(thetai)+Constant['K']*m_set
    beta1_m=np.sqrt(np.diag(np.eye(n_Tr)*(n1*Constant['k0'])**2-np.diag(alpha_m**2)),dtype=np.complex128)
    beta2_m=np.sqrt(np.diag(np.eye(n_Tr)*(n2*Constant['k0'])**2-np.diag(alpha_m**2)),dtype=np.complex128)
    Constant['alpha_m']=alpha_m
    Constant['beta1_m']=beta1_m
    Constant['beta2_m']=beta2_m

    idx=np.where(np.abs(np.imag(beta1_m))<1e-10)
    n1_set=m_set[idx]
    n1_set_ind=idx
    Constant['n1_set']=n1_set
    Constant['n1_set_ind']=n1_set_ind[0]

    idx=np.where(np.abs(np.imag(beta2_m))<1e-10)
    n2_set=m_set[idx]
    n2_set_ind=idx
    Constant['n2_set']=n2_set
    Constant['n2_set_ind']=n2_set_ind[0]
    return Constant

def Eigen(Constant):
    a_mat=Constant['a_mat']
    alpha_m=Constant['alpha_m']
    beta1_m=Constant['beta1_m']
    beta2_m=Constant['beta2_m']
    IB1=np.linalg.solve(np.diag(beta1_m**2),np.eye(Constant['n_Tr'],dtype=np.complex128))
    IB2=np.linalg.solve(np.diag(beta2_m**2),np.eye(Constant['n_Tr'],dtype=np.complex128))
    AUX_mat=np.eye(Constant['n_Tr'],dtype=np.complex128)+a_mat@a_mat
    Matrix1=np.block([[-IB1@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m)),IB1@AUX_mat],
    [np.eye(Constant['n_Tr'],dtype=np.complex128),np.zeros((Constant['n_Tr'],Constant['n_Tr']),dtype=np.complex128)]])
    #==========使用微扰动技术
    # Matrix1=Perturbation(Matrix1)
    Matrix2=np.block([[-IB2@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m)),IB2@AUX_mat],
    [np.eye(Constant['n_Tr'],dtype=np.complex128),np.zeros((Constant['n_Tr'],Constant['n_Tr']),dtype=np.complex128)]])
    #============微扰动技术
    # Matrix2=Perturbation(Matrix2)
    eig1,vec1=np.linalg.eig(Matrix1)
    eig2,vec2=np.linalg.eig(Matrix2)
    eig1=1/eig1
    eig2=1/eig2
    return eig1,vec1,eig2,vec2

def GenerateFFieldsChand(Constant):
    m_set=Constant['m_set']
    n1_set=Constant['n1_set']
    n1_set_ind=Constant['n1_set_ind']
    n2_set=Constant['n2_set']
    n2_set_ind=Constant['n2_set_ind']
    beta1_m=Constant['beta1_m']
    beta2_m=Constant['beta2_m']
    m1=m_set[0]
    m2=m_set[-1]
    a=Constant['a']
    #===================Fmn===============
    if len(n1_set)!=0:
        Fmn=np.zeros((len(m_set),len(n1_set)),dtype=np.complex128)
        min_ray1=np.min(n1_set)
        max_ray1=np.max(n1_set)
        for M in range(min_ray1,max_ray1+1):
            LPn_fun=np.exp(1j*beta1_m[M-m1]*a)
            temp=F_series_gen(LPn_fun,Constant['n_Tr'],cut_small=False)
            q0=len(temp)//2
            part1=temp[q0+m1-M:q0]
            part2=temp[q0:q0+m2-M+1]
            Fmn[:,M-min_ray1]=np.concatenate([part1,part2])
    else:
        Fmn=np.zeros((len(m_set),len(n1_set)),dtype=np.complex128)
    #================Fmk=================
    if len(n2_set)!=0:
        Fmk=np.zeros((len(m_set),len(n2_set)),dtype=np.complex128)
        min_ray2=np.min(n2_set)
        max_ray2=np.max(n2_set)
        for k in range(min_ray2,max_ray2+1):
            LNn_fun=np.exp(-1j*beta2_m[k]*a)
            temp=F_series_gen(LNn_fun,Constant['n_Tr'],cut_small=False)
            q0=len(temp)//2
            for m in range(m1,m2+1):
                Fmk[m,k]=temp[q0+m-k]
    else:
        Fmk=np.zeros((len(m_set),len(n2_set)),dtype=np.complex128)
    #==============Fm0===================
    b0=beta1_m[-m1]
    LP_fun=np.exp(-1j*b0*a)#define fourier transform argument for positive fields
    F_in_0=F_series_gen(LP_fun,Constant['n_Tr'],cut_small=False)#正场系数
    nDim=Constant['n_Tr']
    q0=len(F_in_0)//2
    Fm0=F_in_0[q0-(nDim-1)//2:q0+(nDim-1)//2+1]
    #==============Fmq==================
    Fmq=Constant['vect1_p'][:Constant['n_Tr'],len(n1_set):]
    #==============Fmr==================
    Fmr=Constant['vect2_n'][:Constant['n_Tr'],len(n2_set):]
    #===================================
    Constant['Fmn']=Fmn
    Constant['Fmk']=Fmk
    Constant['Fm0']=Fm0
    Constant['Fmq']=Fmq
    Constant['Fmr']=Fmr
    return Fmn,Fmk,Fm0,Fmq,Fmr

def GenerateGFieldsChand(Constant):
    nDim=Constant['n_Tr']
    a_mat=Constant['a_mat']
    n1_set=Constant['n1_set']
    n2_set=Constant['n2_set']
    m_set=Constant['m_set']
    m1=m_set[0]
    m2=m_set[-1]
    alpha_m=Constant['alpha_m']
    beta1_m=Constant['beta1_m']
    beta2_m=Constant['beta2_m']
    Fmn=Constant['Fmn']
    Fmk=Constant['Fmk']
    Fm0=Constant['Fm0']
    Fmq=Constant['Fmq']
    Fmr=Constant['Fmr']
    eig1_p=Constant['eig1_p']
    eig2_n=Constant['eig2_n']
    AUX_mat=np.eye(nDim)+a_mat@a_mat
    #=================Gmn============================
    if len(n1_set)!=0:
        Gmn=np.zeros((nDim,len(Constant['n1_set'])),dtype=complex)
        min_ray1=np.min(n1_set)
        for M in range(nDim):
            for N in range(len(n1_set)):
                for K in range(nDim):
                    sb1_idx=N+min_ray1-m1
                    term=a_mat[M,K]*alpha_m[K]-AUX_mat[M,K]*beta1_m[sb1_idx]
                    Gmn[M,N]+=term*Fmn[K,N]
    else:
        Gmn=np.zeros((nDim,len(n1_set)),dtype=complex)
    #===============Gmk=============================
    if len(n2_set)!=0:
        Gmk=np.zeros((nDim,len(Constant['n2_set'])),dtype=complex)
        min_ray2=np.min(n2_set)
        for M in range(nDim):
            for N in range(len(n2_set)):
                for K in range(nDim):
                    sb2_idx=N+min_ray2-m1
                    term=a_mat[M,K]*alpha_m[K]-AUX_mat[M,K]*beta2_m[sb2_idx]
                    Gmk[M,N]+=term*Fmk[K,N]
    else:
        Gmk=np.zeros((nDim,len(n2_set)),dtype=complex)
    #=====================Gm0=======================
    b0=beta1_m[-m1]
    Gm0=np.zeros((nDim,1),dtype=complex)
    for M in range(nDim):
        for N in range(nDim):
            term=a_mat[M,N]*alpha_m[N]+AUX_mat[M,N]*b0
            Gm0[M,0]+=term*Fm0[N]
    #=====================Gmq=======================
    if len(n1_set)<Constant['n_Tr']:
        Gmq=np.zeros((nDim,nDim-len(n1_set)),dtype=complex)
        for M in range(nDim):
            for N in range(nDim-len(n1_set)):
                for K in range(nDim):
                    term=a_mat[M,K]*alpha_m[K]-AUX_mat[M,K]*eig1_p[len(n1_set)+N]
                    Gmq[M,N]+=term*Fmq[K,N]
    else:
        Gmq=np.zeros((nDim,len(Fmq)),dtype=complex)
    #===================Gmr==========================
    if len(n2_set)<Constant['n_Tr']:
        Gmr=np.zeros((nDim,nDim-len(n2_set)),dtype=complex)
        for M in range(nDim):
            for N in range(nDim-len(n2_set)):
                for K in range(nDim):
                    term=a_mat[M,K]*alpha_m[K]-AUX_mat[M,K]*eig2_n[len(n2_set)+N]
                    Gmr[M,N]+=term*Fmr[K,N]
    else:
        Gmr=np.zeros((len(m_set),Constant['n_Tr']-len(n2_set)),dtype=np.complex128)
    #==================================================
    #归一化处理
    Gmn=Constant['Z0']/Constant['k0']/Constant['eps1']*Gmn
    Gmk=Constant['Z0']/Constant['k0']/Constant['eps2']*Gmk
    Gm0=Constant['Z0']/Constant['k0']/Constant['eps1']*Gm0
    Gmq=Constant['Z0']/Constant['k0']/Constant['eps1']*Gmq
    Gmr=Constant['Z0']/Constant['k0']/Constant['eps2']*Gmr
    return Gmn,Gmk,Gm0,Gmq,Gmr

def Compute(Constant):
    nDim=Constant['n_Tr']
    a_diff_vec=F_series_gen(a_diff,nDim)
    ##观察傅里叶频谱分量
    # temp_x=range(-nDim,nDim+1)
    # plt.plot(temp_x,a_diff_vec)
    # plt.show()
    #####
    a_mat=Toeplitz(a_diff_vec,nDim)
    Constant['a_mat']=a_mat
    rho1,V1,rho2,V2=Eigen(Constant)
    eig1_p,vect1_p,_,_=SortEigenvalue(rho1,V1,100)
    _,_,eig2_n,vect2_n=SortEigenvalue(rho2,V2,100)
    Constant['eig1_p']=eig1_p
    Constant['eig2_n']=eig2_n
    Constant['vect1_p']=vect1_p
    Constant['vect2_n']=vect2_n
    #Assemble F-matrices
    Fmn,Fmk,Fm0,Fmq,Fmr=GenerateFFieldsChand(Constant)
    #Assemble G-matrices
    Gmn,Gmk,Gm0,Gmq,Gmr=GenerateGFieldsChand(Constant)
    #Assemble matrix for matching boundary conditions and solve the linear system
    Gm0=Gm0.T
    Gm0=Gm0.flatten()
    MatBC=np.block([[Fmn,Fmq,-Fmk,-Fmr],[Gmn,Gmq,-Gmk,-Gmr]])
    MatBC=Perturbation(MatBC)
    VecBC=np.concatenate((-Fm0,-Gm0),axis=0)
    RVec=np.linalg.solve(MatBC,VecBC)
    # RVec=np.linalg.inv(MatBC)@VecBC
    #绘制衍射效率曲线
    return RVec

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

def Plot_Effi(Constant,effi=False,error=False):
    # real_set=Calculate_diffraction_angle(np.rad2deg(Constant['thetai']),0,Constant['wavelength'],Constant['period'],1)
    real_set=Constant['n1_set']
    R_effi=Constant['R_effi']
    fig=plt.figure(figsize=(8,6))
    ##子图1：效率对比
    ax1=plt.subplot(2,2,(1,2))
    ax1.plot(real_set,Constant['R_effi'],'o-',linewidth=2,markersize=6,color='#1f77b4',label='代码反射效率',markerfacecolor='white',markeredgewidth=2)
    # plt.figure(1)
    # plt.plot(real_set,R_effi[real_set_ind],label='Reflection')
    # plt.plot(real_set,VirtualLab_R,label='VirtualLab')
    if len(effi)!=0:
        ax1.plot(real_set,effi,'s--',linewidth=2,markersize=6,color='#ff7f0e',label='VirtualLab 反射效率',markerfacecolor='white',markeredgewidth=2)
        # plt.plot(real_set,effi,label="VirtualLab_Effi")
    ax1.set_xlabel("衍射级次",fontsize=12,fontweight='bold')
    ax1.set_ylabel("衍射效率",fontsize=12,fontweight='bold')
    ax1.legend(loc='best',frameon=True,shadow=True)
    ax1.grid(True,alpha=0.3)
    # ax1.set_ylim(0,1)
    ###子图2：绝对误差
    ax2=plt.subplot(2,2,3)
    if effi is not None and error:
        abs_error=Constant['R_effi']-effi
        line1=ax2.plot(real_set,abs_error,'o-',linewidth=2,markersize=6,
                       color='#2ca02c',label='绝对误差',markerfacecolor='white',markeredgewidth=2)
        ax2.set_xlabel("衍射级次",fontsize=12,fontweight='bold')
        ax2.set_ylabel("绝对误差",fontsize=12,fontweight='bold',color='#2ca02c')
        ax2.set_title("绝对误差图",fontsize=14,fontweight='bold')
        ax2.tick_params(axis='y',labelcolor='#2ca02c')
        ax2.grid(True,alpha=0.3)
        ##标注最大值
        max_abs_error=np.max(np.abs(abs_error))
        ax2.text(0.05,0.95,f'最大绝对误差:{max_abs_error:.4f}',
                 transform=ax2.transAxes,fontsize=10,verticalalignment='top',
                 bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
    ##子图3:相对误差
    ax3=plt.subplot(2,2,4)
    if effi is not None and error:
        rel_error=(Constant['R_effi']-effi)/effi
        ax3_twin=ax3.twinx()
        line2=ax3_twin.plot(real_set,rel_error,'s--',linewidth=2,markersize=6,
                            color='#d62728',label='相对误差',markerfacecolor='white',markeredgewidth=2)
        ax3.set_xlabel("衍射级次",fontsize=12,fontweight='bold')
        ax3_twin.set_ylabel("相对误差",fontsize=12,fontweight='bold',color='#d62728')
        ax3.set_title("相对误差图",fontsize=14,fontweight='bold')
        ax3_twin.tick_params(axis='y',labelcolor='#d62728')
        ax3.grid(True,alpha=0.3)
        max_rel_error=np.max(np.abs(rel_error))
        ax3.text(0.05,0.95,f'最大相对误差:{max_rel_error:.2%}',
                 transform=ax3.transAxes,fontsize=10,verticalalignment='top',
                 bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
    plt.tight_layout()
    plt.show()
    # plt.xlabel('Diffraction order')
    # plt.ylabel('Diffraction efficiency')
    # plt.title("4微米周期矩形光栅")
    # plt.legend()
    # plt.show()
    print(R_effi)
    print("sum:"+str(sum(R_effi)))
    # if error!=False:
    #     plt.figure(2)
    #     plt.plot(real_set,Constant['R_effi']-effi,label="绝对误差")
    #     plt.plot(real_set,(Constant['R_effi']-effi)/effi,label="相对误差")
    #     plt.xlabel("Diffraction order")
    #     plt.ylabel("Relavent Error")
    #     plt.legend()
    #     plt.show()

#=======================指定材料等仿真参数=============================
n1=1
n2=1.4482+7.5367j
pol='TE'
n_Tr=2*20+1
lam=632.8*1e-9
thetai=np.radians(0)
cut=0
ImagMin=1e-9
accuracy=1e-10
# grating=Triangular(4*1e-6,36,1)
grating=Sinusoidal(4*1e-6,1,2*1e-6)
R_effi=[]
Constant=SetConstant(n1,n2,pol,grating.T,n_Tr,lam,thetai)
Constant['depth']=grating.depth
#===================================================================
a=grating.profile()
x=np.linspace(0,Constant['period'],2**10,endpoint=False)
Constant['a']=a(x)#光栅表面轮廓函数
dx=Constant['period']/len(x)
a_diff=np.gradient(Constant['a'],dx)
Constant['a_diff']=a_diff
# plt.plot(x,Constant['a'])
# plt.plot(x,a_diff)
# plt.show()
#=======================================================================
RVec=Compute(Constant)
R=np.zeros(len(Constant['n1_set']))
beta1_m=Constant['beta1_m']
n1_set_ind=Constant['n1_set_ind']
for i in range(len(Constant['n1_set'])):
    R[i]=np.real(beta1_m[n1_set_ind[i]]/beta1_m[int((Constant['n_Tr']-1)/2)])*(abs(RVec[i])**2)
T=np.zeros(len(Constant['n1_set']))
beta2_m=Constant['beta2_m']
n2_set=Constant['n2_set']
n2_set_ind=Constant['n2_set_ind']
if len(n2_set)!=0:
    for i in range(len(n2_set)):
        T[i]=(Constant['eps1']*beta2_m[n2_set_ind[i]]/(Constant['eps2']*beta1_m[(Constant['n_Tr']-1)/2]))*abs(RVec[Constant['n_Tr']+i])**2
Constant['R_effi']=R
Constant['T_effi']=T
# start_order=-6
# [print(f"{start_order+i} {val}") for i,val in enumerate(R)]
# print("sum R:{}".format(sum(R)))
Plot_Effi(Constant,[])