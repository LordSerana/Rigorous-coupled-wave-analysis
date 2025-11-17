import numpy as np
import matplotlib.pyplot as plt

def Plot_intensity(Constant,RVec,real_Ray1_idx,real_Ray2_idx,B1,B2,m1,b0,nDim,plot):
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