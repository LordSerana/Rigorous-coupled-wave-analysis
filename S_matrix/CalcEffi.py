import numpy as np

def calcEffi(p,Constant,S_global):
    m=Constant['n_Tr']//2
    kzref=Constant['kzref']
    kx=Constant['kx']
    ky=Constant['ky']
    kinc=Constant['kinc']
    delta0=np.zeros(2*m+1)
    delta0[m]=1
    esrc=np.concatenate((p[0]*delta0,p[1]*delta0))
    Wref=Wtrn=np.eye(4*m+2)
    csrc=np.linalg.inv(Wref)@esrc
    cref=S_global[:4*m+2,:4*m+2]@csrc
    ctrn=S_global[4*m+2:,:4*m+2]@csrc
    eref=Wref@cref
    rx=eref[:2*m+1]
    ry=eref[2*m+1:]
    rz=-np.linalg.inv(kzref)@(kx@rx+ky@ry)
    etrn=Wtrn@ctrn
    tx=etrn[:2*m+1]
    ty=etrn[2*m+1:]
    tz=-np.linalg.inv(kzref)@(kx@tx+ky@ty)
    r2=abs(rx)**2+abs(ry)**2+abs(rz)**2
    R=np.real(-kzref)/np.real(kinc[2])*r2
    R_effi=np.sum(R,axis=1)
    t2=abs(tx)**2+abs(ty)**2+abs(tz)**2
    T=np.real(-kzref)/np.real(kinc[2])*t2
    T_effi=np.sum(T,axis=1)
    return R_effi,T_effi