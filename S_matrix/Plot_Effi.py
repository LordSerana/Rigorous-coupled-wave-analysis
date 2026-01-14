import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif']=['SimHei','Arial Unicode MS','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

def Plot_Effi(Constant,effi=False,error=False):
    real_mask=np.abs(np.imag(np.diag(Constant['kz'])))<Constant['accuracy']
    real_set=Constant['mx'][real_mask]
    real_set_ind=np.where(real_mask)
    Constant['real_set']=real_set
    Constant['real_set_ind']=real_set_ind
    R_effi=Constant['R_effi']
    # T_effi=Constant['T_effi']
    Constant['R_effi']=R_effi[real_set_ind]
    # Constant['T_effi']=T_effi[real_set_ind]
    fig=plt.figure(figsize=(15,10))
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
    ax1.set_ylim(0,1)
    ###子图2：绝对误差
    ax2=plt.subplot(2,2,3)
    if effi is not None and error:
        abs_error=Constant['R_effi']-effi
    plt.xlabel('Diffraction order')
    plt.ylabel('Diffraction efficiency')
    plt.title("4微米周期矩形光栅")
    plt.legend()
    plt.show()
    print(R_effi[real_set_ind])
    print("sum:"+str(sum(R_effi[real_set_ind])))
    if error!=False:
        plt.figure(2)
        plt.plot(real_set,Constant['R_effi']-effi,label="绝对误差")
        plt.plot(real_set,(Constant['R_effi']-effi)/effi,label="相对误差")
        plt.xlabel("Diffraction order")
        plt.ylabel("Relavent Error")
        plt.legend()
        plt.show()