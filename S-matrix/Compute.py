import numpy as np
from Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix
from Layer_mode import layer_mode
from Build_scatter_side import build_scatter_side
from Star import Star
from CalcEffi import calcEffi
import matplotlib.pyplot as plt

def Compute(Constant,layers,plot=False):
    #####计算波矢k
    kinc=Constant['kinc']
    kx=np.diag(kinc[0]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])
    Constant['kx']=kx
    temp=Constant['n_Tr']//2
    ky=np.zeros((2*temp+1,2*temp+1))
    Constant['ky']=ky
    kzref=np.zeros((2*temp+1,2*temp+1),dtype=complex)
    for i in range(2*temp+1):
        kzref[i,i]=np.sqrt(Constant['n1']**2-kx[i,i]**2,dtype=complex)
    Constant['kzref']=-kzref
    #####计算自由空间本征模态
    LAM,W=homogeneous_isotropic_matrix(1,1,np.diag(kx),np.diag(ky))
    temp=int(W.shape[0]/2)
    W0=W[:temp,:temp]
    V0=W[temp:,:temp]
    Constant['W0']=W0
    Constant['V0']=V0
    #####构建全局散射矩阵
    temp=Constant['n_Tr']//2
    S_global=np.block(
        [[np.zeros((4*temp+2,4*temp+2),dtype=complex),np.eye(4*temp+2,dtype=complex)],
         [np.eye(4*temp+2,dtype=complex),np.zeros((4*temp+2,4*temp+2),dtype=complex)]])
    for i in layers[1:-1]:
        S=layer_mode(i,Constant,'formula')
        S_global=Star(S_global,S)
    #####计算反射侧、透射侧散射矩阵
    Ssub,Wsub,LAMsub=build_scatter_side(Constant['n2']**2,1,np.diag(kx),np.diag(ky),W,transmission_side=True)
    Ssub=np.block([[Ssub[0],Ssub[1]],[Ssub[2],Ssub[3]]])
    Sref,Wref,LAMref=build_scatter_side(1,1,np.diag(kx),np.diag(ky),W)
    Sref=np.block([[Sref[0],Sref[1]],[Sref[2],Sref[3]]])
    S_global=Star(S_global,Ssub)
    S_global=Star(Sref,S_global)
    #####全局散射矩阵计算完成,计算效率
    R_effi,T_effi=calcEffi(Constant['p'],Constant,S_global)
    #####标记正常传输级次
    real_mask=np.abs(np.imag(np.diag(Constant['kzref'])))<Constant['accuracy']
    real_set=Constant['mx'][real_mask]
    real_set_ind=np.where(real_mask)
    Constant['real_set']=real_set
    Constant['real_set_ind']=real_set_ind
    Constant['R_effi']=R_effi[real_set_ind]
    Constant['T_effi']=T_effi[real_set_ind]
    #####绘制效率曲线
    if plot==True:
    # VirtualLab_R=[0.0060481,0.013476,0.0063769,0.055142,0.021614,0.21037,0.22983,0.21037,0.021614,0.055142,0.0063769,0.013476,0.0060481]#Al,45°线偏振光,矩形
        plt.figure()
        plt.plot(real_set,R_effi[real_set_ind],label='Reflection')
        # plt.plot(real_set,VirtualLab_R,label='VirtualLab')
        plt.xlabel('Diffraction order')
        plt.ylabel('Diffraction efficiency')
        plt.title("4微米周期矩形光栅")
        plt.legend()
        plt.show()
        print(R_effi[real_set_ind])
        print("sum:"+str(sum(R_effi[real_set_ind])))
    return Constant