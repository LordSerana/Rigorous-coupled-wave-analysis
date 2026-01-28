import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif']=['SimHei','Arial Unicode MS','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题

def Plot_Effi(Constant,effi=False,error=False):
    real_set=Constant['real_Ray1_idx']
    R_effi=Constant['R_effi']
    fig=plt.figure(figsize=(8,6))
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
    # ax1.set_ylim(0,1)
    ###子图2：绝对误差
    ax2=plt.subplot(2,2,3)
    if effi is not None and error:
        abs_error=Constant['R_effi']-effi
        line1=ax2.plot(real_set,abs_error,'o-',linewidth=2,markersize=6,
                       color='#2ca02c',label='绝对误差',markerfacecolor='white',markeredgewidth=2)
        ax2.set_xlabel("衍射级次",fontsize=12,fontweight='bold')
        ax2.set_ylabel("绝对误差",fontsize=12,fontweight='bold',color='#2ca02c')
        ax2.set_title("绝对误差图",fontsize=14,fontweight='bold')
        ax2.tick_params(axis='y',labelcolor='#2ca02c')
        ax2.grid(True,alpha=0.3)
        ##标注最大值
        max_abs_error=np.max(np.abs(abs_error))
        ax2.text(0.05,0.95,f'最大绝对误差:{max_abs_error:.4f}',
                 transform=ax2.transAxes,fontsize=10,verticalalignment='top',
                 bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
    ##子图3:相对误差
    ax3=plt.subplot(2,2,4)
    if effi is not None and error:
        rel_error=(Constant['R_effi']-effi)/effi
        ax3_twin=ax3.twinx()
        line2=ax3_twin.plot(real_set,rel_error,'s--',linewidth=2,markersize=6,
                            color='#d62728',label='相对误差',markerfacecolor='white',markeredgewidth=2)
        ax3.set_xlabel("衍射级次",fontsize=12,fontweight='bold')
        ax3_twin.set_ylabel("相对误差",fontsize=12,fontweight='bold',color='#d62728')
        ax3.set_title("相对误差图",fontsize=14,fontweight='bold')
        ax3_twin.tick_params(axis='y',labelcolor='#d62728')
        ax3.grid(True,alpha=0.3)
        max_rel_error=np.max(np.abs(rel_error))
        ax3.text(0.05,0.95,f'最大相对误差:{max_rel_error:.2%}',
                 transform=ax3.transAxes,fontsize=10,verticalalignment='top',
                 bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
    plt.tight_layout()
    plt.show()
    # plt.xlabel('Diffraction order')
    # plt.ylabel('Diffraction efficiency')
    # plt.title("4微米周期矩形光栅")
    # plt.legend()
    # plt.show()
    print(R_effi)
    print("sum:"+str(sum(R_effi)))
    # if error!=False:
    #     plt.figure(2)
    #     plt.plot(real_set,Constant['R_effi']-effi,label="绝对误差")
    #     plt.plot(real_set,(Constant['R_effi']-effi)/effi,label="相对误差")
    #     plt.xlabel("Diffraction order")
    #     plt.ylabel("Relavent Error")
    #     plt.legend()
    #     plt.show()