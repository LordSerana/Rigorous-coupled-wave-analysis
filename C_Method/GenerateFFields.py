import numpy as np
from F_series_gen import F_series_gen

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