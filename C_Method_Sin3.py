import numpy as np
import sys
sys.path.append('E:/Project/python')
from S_matrix.Grating import Sinusoidal
from scipy import integrate
from scipy.linalg import toeplitz
import scipy.linalg as la

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
    beta1_m=np.sqrt(np.diag(np.eye(n_Tr)*(n1*Constant['k0'])**2-np.diag(alpha_m**2)),dtype=complex)
    beta2_m=np.sqrt(np.diag(np.eye(n_Tr)*(n2*Constant['k0'])**2-np.diag(alpha_m**2)),dtype=complex)
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

def Eigen(a_mat,alpha_m,beta1_m,beta2_m,Constant):
    IB1=np.linalg.solve(np.diag(beta1_m**2),np.eye(Constant['n_Tr']))
    IB2=np.linalg.solve(np.diag(beta2_m**2),np.eye(Constant['n_Tr']))
    AUX_mat=np.eye(Constant['n_Tr'])+a_mat@a_mat
    Matrix1=np.block([[-IB1@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m)),IB1@AUX_mat],[np.eye(Constant['n_Tr']),np.zeros((Constant['n_Tr'],Constant['n_Tr']))]])
    Matrix2=np.block([[-IB2@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m)),IB2@AUX_mat],[np.eye(Constant['n_Tr']),np.zeros((Constant['n_Tr'],Constant['n_Tr']))]])
    eig1,vec1=np.linalg.eig(Matrix1)
    eig2,vec2=np.linalg.eig(Matrix2)
    eig1=1/eig1
    eig2=1/eig2
    return eig1,vec1,eig2,vec2

def SortEigenvalue(eig,vect,accuracyImag,ratio_threshold):
    real_parts=np.real(eig)
    imag_parts=np.imag(eig)
    abs_real=np.abs(real_parts)
    abs_imag=np.abs(imag_parts)
    #=======判断特征值的主导性
    n=len(eig)
    is_real_dominant=(abs_real>ratio_threshold*abs_imag)
    is_imag_dominant=(abs_imag>ratio_threshold*abs_real)
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

def L_eval(gamma,m,Constant):
    def integrand(x):
        if isinstance(Constant['a(x)'],str):
            a_x_val=eval(Constant['a(x)'],{"np":np,"x":x})
        else:
            a_x_val=Constant['a(x)'](x)
        exponent=1j*gamma*a_x_val-1j*m*2*np.pi*x/Constant['period']
        return (1/Constant['period'])*np.exp(exponent)
    L,error=integrate.quad(integrand,0,Constant['period'],complex_func=True)
    return L

def GenerateFField(Constant):
    m_set=Constant['m_set']
    n1_set=Constant['n1_set']
    n1_set_ind=Constant['n1_set_ind']
    n2_set=Constant['n2_set']
    n2_set_ind=Constant['n2_set_ind']
    beta1_m=Constant['beta1_m']
    beta2_m=Constant['beta2_m']
    idx_m=0
    idx_n=0
    #=====Fmn=====
    if len(Constant['n1_set'])!=0:
        Fmn=np.zeros((len(m_set),len(n1_set)),dtype=complex)
        for m in range(m_set[0],m_set[-1]+1,1):
            for n in range(n1_set[0],n1_set[-1],1):
                Fmn[idx_m,idx_n]=L_eval(beta1_m[n1_set_ind[idx_n]],m-n,Constant)
                idx_n+=1
            idx_n=0
            idx_m+=1
    else:
        Fmn=np.zeros(len(m_set),len(n1_set))
    #======Fmk========
    idx_m=0
    idx_k=0
    if len(Constant['n2_set'])!=0:
        Fmk=np.zeros(len(m_set),len(n2_set))
        for m in range(m_set[0],m_set[-1]+1,1):
            for k in range(n2_set[0],n2_set[-1]+1,1):
                Fmk[idx_m,idx_k]=L_eval(-beta2_m[n2_set_ind[idx_k]],m-k,Constant)
                idx_k+=1
            idx_k=0
            idx_m+=1
    else:
        Fmk=np.zeros((len(m_set),len(n2_set)))
    #=====F_in_0===========
    idx=0
    Fm0=np.zeros((len(m_set),1),dtype=complex)
    for m in range(m_set[0],m_set[-1]+1,1):
        Fm0[idx,0]=L_eval(-beta1_m[int((Constant['n_Tr']+1)/2)],m,Constant)
        idx+=1
    #===========Fmq=================
    Fmq=vect1_p[:Constant['n_Tr'],len(n1_set):]
    #===========Fmr=================
    Fmr=vect2_n[:Constant['n_Tr'],len(n2_set):]
    Constant['Fmn']=Fmn
    Constant['Fmk']=Fmk
    Constant['Fm0']=Fm0
    Constant['Fmq']=Fmq
    Constant['Fmr']=Fmr
    return Fmn,Fmk,Fm0,Fmq,Fmr

def GenerateGField(a_mat,Constant,eig1_p,eig2_n):
    a_ms=np.eye(Constant['n_Tr'])+a_mat@a_mat
    n1_set=Constant['n1_set']
    n1_set_ind=Constant['n1_set_ind']
    n2_set=Constant['n2_set']
    n2_set_ind=Constant['n2_set_ind']
    m_set=Constant['m_set']
    alpha_m=Constant['alpha_m']
    beta1_m=Constant['beta1_m']
    beta2_m=Constant['beta2_m']
    Fmn=Constant['Fmn']
    Fmk=Constant['Fmk']
    Fm0=Constant['Fm0']
    Fmq=Constant['Fmq']
    Fmr=Constant['Fmr']
    #===========Gmn=========================
    if len(Constant['n1_set_ind'])!=0:
        Gmn=np.zeros((len(m_set),len(n1_set)),dtype=complex)
        for m in range(len(m_set)):
            for n in range(len(n1_set)):
                Gmn_s=np.zeros(len(m_set),dtype=complex)
                for s in range(len(m_set)):
                    Gmn_s[s]=(a_mat[m,s]*alpha_m[s]-a_ms[m,s]*beta1_m[n1_set_ind[n]])*Fmn[s,n]
                Gmn[m,n]=Constant['Z0']/Constant['k0']/Constant['eps1']*sum(Gmn_s)
    else:
        Gmn=np.zeros((len(m_set),len(n1_set)))
    #==========Gmk==========================
    if len(Constant['n2_set_ind'])!=0:
        Gmk=np.zeros((len(m_set),len(n2_set)),dtype=complex)
        for m in range(len(m_set)):
            for k in range(len(n2_set_ind)):
                Gmk_s=np.zeros((len(m_set),1))
                for s in range(len(m_set)):
                    Gmk_s[s,0]=(a_mat[m,s]*alpha_m[s]+a_ms[m,s]*beta2_m[n2_set_ind[k]])*Fmk[s,k]
                Gmk[m,k]=Constant['Z0']/Constant['k0']/Constant['eps2']*sum(Gmk_s)
    else:
        Gmk=np.zeros((len(m_set),len(n2_set)),dtype=complex)
    #=================Gm0=========================
    Gm0=np.zeros((len(m_set),1),dtype=complex)
    for m in range(len(m_set)):
        Gm0_s=np.zeros((len(m_set),1),dtype=complex)
        for s in range(len(m_set)):
            Gm0_s[s,0]=(a_mat[m,s]*alpha_m[s]+a_ms[m,s]*beta1_m[int((Constant['n_Tr']-1)/2)])*Fm0[s][0]
        Gm0[m,0]=Constant['Z0']/Constant['k0']/Constant['eps1']*sum(Gm0_s)
    #=================Gmq==========================
    if len(n1_set)<Constant['n_Tr']:
        Gmq=np.zeros((len(m_set),Constant['n_Tr']-len(n1_set)),dtype=complex)
        for m in range(len(m_set)):
            for q in range(Constant['n_Tr']-len(n1_set)):
                Gmq_s=np.zeros((len(m_set),1))
                for s in range(len(m_set)):
                    Gmq_s[s,0]=(a_mat[m,s]*alpha_m[s]-a_ms[m,s]*eig1_p[len(n1_set)+q])*Fmq[s,q]
                Gmq[m,q]=Constant['Z0']/Constant['k0']/Constant['eps1']*sum(Gmq_s)
    else:
        Gmq=np.zeros((len(m_set),Constant['n_Tr']-len(n1_set)),dtype=complex)
    #==================Gmr==========================
    if len(n2_set)<Constant['n_Tr']:
        Gmr=np.zeros((len(m_set),Constant['n_Tr']-len(n2_set)),dtype=complex)
        for m in range(len(m_set)):
            for r in range(Constant['n_Tr']-len(n2_set)):
                Gmr_s=np.zeros((len(m_set),1))
                for s in range(len(m_set)):
                    Gmr_s[s]=(a_mat[m,s]*alpha_m[s]-a_ms[m,s]*eig2_n[len(n2_set)+r])*Fmr[s,r]
                Gmr_s=Gmr_s.flatten()
                Gmr[m,r]=Constant['Z0']/Constant['k0']/Constant['eps2']*sum(Gmr_s)
    else:
        Gmr=np.zeros((len(m_set),Constant['n_Tr']-len(n2_set)),dtype=complex)
    return Gmn,Gmk,Gm0,Gmq,Gmr

def CutSmallElement(a,accuracy):
    idx=(np.abs(a)<accuracy)
    a[idx]=0
    return a

def ComputeAdiff(Constant):
    n_set=Constant['n_set']
    a_diff=np.zeros(len(n_set))
    i=0
    for n in range(n_set[0],n_set[-1]+1):
        F=lambda x:(1/Constant['period'])*np.exp(-1j*n*Constant['K']*x)*Constant['a_diff(x)'](x)
        a_diff[i],error=integrate.quad(F,0,Constant['period'])
        i+=1
    a_diff=CutSmallElement(a_diff,1e-10)
    return a_diff

####=============指定仿真材料等参数================
n1=1
n2=1.4482+7.5367j
pol='TE'
period=4*1e-6
n_Tr=2*45+1
lam=632.8*1e-9
thetai=np.radians(0)
depth=2*1e-6#光栅深度
ImagMin=1e-8
cut=1#需要修剪数据
accuracy=1e-10
#==================================================
Constant=SetConstant(n1,n2,pol,period,n_Tr,lam,thetai)
Constant['ImagMin']=ImagMin
grating=Sinusoidal(period,1,depth)
Constant['period']=period
a=grating.profile()
Constant['a(x)']=a
Constant['a_diff(x)']=grating.a_diff()
a_diff_vec=ComputeAdiff(Constant)
a_col=a_diff_vec[Constant['n_Tr']-1:]
a_row=a_diff_vec[Constant['n_Tr']-1::-1]
# x=np.linspace(0,Constant['period'],2**10,endpoint=False)
# Constant['a']=Constant['k0']*a(x)#光栅表面轮廓函数
# dx=Constant['period']*Constant['k0']/len(x)
# a_diff=np.gradient(Constant['a'],dx)
# a_diff_vec=F_series_gen(a_diff,n_Tr)
a_mat=toeplitz(a_col,a_row)
eig1,vect1,eig2,vect2=Eigen(a_mat,Constant['alpha_m'],Constant['beta1_m'],Constant['beta2_m'],Constant)
eig1_p,vect1_p,_,_=SortEigenvalue(eig1,vect1,Constant['ImagMin'],100)
_,_,eig2_n,vect2_n=SortEigenvalue(eig2,vect2,Constant['ImagMin'],100)
Constant['vect1_p']=vect1_p
Constant['vect2_n']=vect2_n
Fmn,Fmk,Fm0,Fmq,Fmr=GenerateFField(Constant)
if cut==1:
    Fmn=CutSmallElement(Fmn,accuracy)
    Fmk=CutSmallElement(Fmk,accuracy)
    Fm0=CutSmallElement(Fm0,accuracy)
    Fmq=CutSmallElement(Fmq,accuracy)
    Fmr=CutSmallElement(Fmr,accuracy)
Gmn,Gmk,Gm0,Gmq,Gmr=GenerateGField(a_mat,Constant,eig1_p,eig2_n)
GF_matrix=np.block([[Fmn,Fmq,-Fmk,-Fmr],[Gmn,Gmq,-Gmk,-Gmr]])
GF_col=-np.block([[Fm0],[Gm0]])
##==============固定碰到大条件数矩阵=====================
U,s,Vt=la.svd(GF_matrix,full_matrices=False)
threshold=np.max(s)*np.max(GF_matrix.shape)*np.finfo(float).eps
s_inv=np.zeros_like(s)
mask=(s>threshold)
s_inv[mask]=1.0/s[mask]
s_inv=s_inv.reshape(-1,1)
Amplitude=Vt.T@(s_inv*(U.T@GF_col))
Amplitude=Amplitude.flatten()
# Amplitude=np.linalg.solve(GF_matrix,GF_col)
#==============Reflection efficiency=================
R=np.zeros(len(Constant['n1_set']))
beta1_m=Constant['beta1_m']
n1_set_ind=Constant['n1_set_ind']
for i in range(len(Constant['n1_set'])):
    R[i]=beta1_m[n1_set_ind[i]]/beta1_m[int((Constant['n_Tr']-1)/2)]*(abs(Amplitude[i])**2)
#=============Transmission efficiency==============
T=np.zeros(len(Constant['n1_set']))
beta2_m=Constant['beta2_m']
n2_set=Constant['n2_set']
n2_set_ind=Constant['n2_set_ind']
if len(n2_set)!=0:
    for i in range(len(n2_set)):
        T[i]=(Constant['eps1']*beta2_m[n2_set_ind[i]]/(Constant['eps2']*beta1_m[(Constant['n_Tr']+1)/2]))*abs(Amplitude[Constant['n_Tr']+i])**2
start_order=-6
[print(f"{start_order+i} {val}") for i,val in enumerate(R)]
print("sum R:{}".format(sum(R)))