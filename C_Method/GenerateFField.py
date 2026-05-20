import numpy as np
import sys
sys.path.append("E:/Project/Python")
from C_Method.F_series_gen import F_series_gen

def GenerateFField(Constant):
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