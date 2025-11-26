import numpy as np

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