import numpy as np

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