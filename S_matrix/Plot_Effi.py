import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

def Plot_Effi(Constant,effi=False):
    real_mask=np.abs(np.imag(np.diag(Constant['kzref'])))<Constant['accuracy']
    real_set=Constant['mx'][real_mask]
    real_set_ind=np.where(real_mask)
    Constant['real_set']=real_set
    Constant['real_set_ind']=real_set_ind
    R_effi=Constant['R_effi']
    T_effi=Constant['T_effi']
    Constant['R_effi']=R_effi[real_set_ind]
    Constant['T_effi']=T_effi[real_set_ind]
    plt.figure()
    plt.plot(real_set,R_effi[real_set_ind],label='Reflection')
    # plt.plot(real_set,VirtualLab_R,label='VirtualLab')
    if len(effi)!=0:
        plt.plot(real_set,effi,label="VirtualLab_Effi")
    plt.xlabel('Diffraction order')
    plt.ylabel('Diffraction efficiency')
    plt.title("4微米周期矩形光栅")
    plt.legend()
    plt.show()
    print(R_effi[real_set_ind])
    print("sum:"+str(sum(R_effi[real_set_ind])))