import numpy as np
import cmath
from scipy import linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import quad
#本脚本求解的为TE偏振态
#对partial4脚本的改进，添加有损耗区域的计算方法

class Rectangular:
    '''
    T:光栅周期
    Amplitude:光栅齿的高度
    fill_factor:光栅脊部的占空比
    n_rd:光栅脊部折射率，使用的材料是Al
    n_gr:光栅槽部折射率，即空气
    '''
    j=cmath.sqrt(-1)
    T=0
    amplitude=0
    fill_factor=0
    name="Rectangular"
    n_rd=1.4482+j*7.5367
    n_gr=1.00027
    substrate_thickness=0*1e-6
    def __init__(self,T,amplitude,fill_factor):
        self.T=T
        self.amplitude=amplitude
        self.fill_factor=fill_factor
    
class Triangular:
    j=cmath.sqrt(-1)
    T=0
    base_angle=0
    fill_factor=0
    amplitude=0
    name="Triangular"
    n_rd=3.8827+j*0.019626
    n_gr=1.00027
    substrate_thickness=10*1e-6
    def __init__(self,T,base_angle,fill_factor):
        self.T=T
        self.base_angle=base_angle
        self.fill_factor=fill_factor
        self.amplitude=T*fill_factor/2*np.tan(base_angle)

def grating_fft(epsilon):
    '''
    epsilon:光栅区域的介电常数离散函数
    m:展开的傅里叶谐波数
    '''
    fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])#归一化
    return fourier_coeffi

def fourier_coeffi_complex(signal_func,T,N_max):
    '''
    signal_func:传入的待展开的函数表达式
    T:周期
    N_max:展开的最大谐波数
    '''
    f0=1/T
    coeffi_list=[]
    for i in range(-N_max,N_max+1):
        c_n=f0*quad(lambda t:signal_func(t)*np.exp(-j*2*np.pi*i*f0*t),0,T)[0]
        coeffi_list.append(c_n)
    coeffi_list=np.array(coeffi_list)
    return coeffi_list

def iteration(n_rd,n_gr,d,k0,kx,m,f,g,fill_factor):
    #对多层光栅切片迭代求解f1,g1
    E=np.zeros((2*m+1,2*m+1),dtype=np.float64)
    E=E.astype('complex')
    for i in range(2*m+1):
        for j in range(2*m+1):
            if i==j:
                E[i,j]=n_rd**2*fill_factor+n_gr**2*(1-fill_factor)
            else:
                k=i-j
                E[i,j]=(n_rd**2-n_gr**2)*np.sin(pi*k*fill_factor)/(pi*k)
    
    A=E-kx**2
    Q,W=LA.eig(A)#Q是A矩阵的特征值，W是A矩阵的特征向量
    Q=np.sqrt(Q)
    Q=np.diag(Q)
    for k in range(2*m+1):
        if Q[k,k].imag<0:
            Q[k,k]=np.conj(Q[k,k])
    # # for k in range(2*m+1):
    #     if n_rd**2>kx[k,k]**2:
    #         Q[k,k]=Q[k,k].real
    #     elif n_rd**2<=kx[k,k]**2:
    #         Q[k,k]=Q[k,k].imag
    # Q=Q.imag#对普通介质，Q取实部。对金属介质，Q取虚部
    V=W@Q
    X=np.diag(np.exp(-k0*d*np.diag(Q)))
    temp=np.block([[-W,f],[V,g]])
    temp2=np.concatenate((W@X,V@X),axis=0)
    ab=np.linalg.inv(temp)@temp2
    a=ab[:2*m+1,:]
    b=ab[2*m+1:,:]

    f_L=W@(np.eye(2*m+1)+X@a)
    g_L=V@(np.eye(2*m+1)-X@a)
    return f_L,g_L,b

def slicer(grating,k):
    '''
    grating:需要传入光栅类的实例
    k:切片数
    用于将光栅区域切片,传回切片的厚度序列，以及占空比序列。共两个返回参数
    '''
    thickness=grating.amplitude
    single_thickness=thickness/k
    # num=int(np.ceil(thickness/single_thickness))#应该向上取整
    d=np.ones((k+1,1))*single_thickness
    d[k,0]=grating.substrate_thickness
    fill_factor=np.ones((k+1,1))
    n_rd=np.ones((k+1,1))*grating.n_rd
    n_gr=np.ones((k+1,1))*grating.n_gr
    n_gr=n_gr.astype('complex')
    if grating.name=="Rectangular":
        fill_factor[:k,0]=grating.fill_factor
        n_gr[k:,0]=grating.n_rd
    else:
        fill_factor=np.zeros((k,1))
        for i in range(k):
            fill_factor[i,0]=grating.fill_factor*i/k
        # d[num-1,0]=thickness-(num-1)*single_thickness
    return d,fill_factor,n_rd,n_gr

def stable_inv_tikhonov(A,lambda_reg=1e-5):
    n=A.shape[0]
    A_reg=A+lambda_reg*np.eye(n)
    return np.linalg.inv(A_reg)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#下列部分为光栅的系数
grating=Rectangular(4*1e-6,2*1e-6,0.5)
m=6#傅里叶谐波数，指的是1~m次,这一项影响的是计算精度
j=cmath.sqrt(-1)
n1=1.00027#入射区域的折射率
n2=1.4482+j*7.5367#出射区域的折射率
thetai=0#入射角
pi=np.pi

d,fill_factor,n_rd,n_gr=slicer(grating,10)
#计算光栅介电常数矩阵E
# Nx=1024
# epsilon=np.ones((Nx,1),dtype=complex)
# for i in range(Nx):
#     if i <=Nx*fill_factor:
#         epsilon[i,0]=n_rd
#     else:
#         epsilon[i,0]=n1
# fourier_coeffi=grating_fft(epsilon)

#观察fft返回结果
# plt.figure()
# plt.plot(np.real(fourier_coeffi),linewidth=2)
# plt.show()

# E=np.zeros((2*m+1,2*m+1))
# E=E.astype('complex')
# p0=int(Nx/2)
# for i in range(2*m+1):
#     for j in range(2*m+1):
#         k=i-j
#         E[i,j]=fourier_coeffi[p0+k,0]

count=0
t=1#求解的波长分解数
T_effi=np.zeros((t,2*m+1),dtype=np.float64)#长度为10*(2m+1)，行代表波长范围，列代表衍射级次
R_effi=np.zeros((t,2*m+1),dtype=np.float64)
T_sum=np.zeros((t,1))#总的透射效率
R_sum=np.zeros((t,1))#总的反射效率

wavelengh=np.linspace(0.6328,0.6328,1)
for lam in wavelengh:
    print("当前求解波长为:{lam}".format(lam=lam))
    lam=lam*1e-6
    #lam0=632.8*1e-9#初始波长
    #对于衍射级次的数目，需要使用严谨的光栅方程考虑
    #判断衍射级次能否存在的方法，就是sin(thetam)<=1
    h=np.linspace(-m,m,2*m+1)#衍射级次
    #衍射角暂时没必要计算
    # thetam=[]#衍射角
    # h.append(0)
    # thetam.append(thetai)
    # order=1
    # k=0#正向反向搜索的标志
    #填充衍射级次和衍射角数组
    # while 1:
    #     #正向搜索
    #     temp=(n1*cmath.sin(thetai)+order*lam/period)/n1
    #     if abs(temp)<=1 and k==0:
    #         h.append(order)
    #         thetam.append(cmath.asin(temp)/pi*180)
    #         order+=1
    #     elif abs(temp)<=1 and k==1:
    #         h.append(order)
    #         thetam.append(cmath.asin(temp)/pi*180)
    #         order-=1
    #     elif abs(temp)>1 and k==0:
    #         order=-1
    #         k=1
    #     elif abs(temp)>1 and k==1:
    #         break
    # h=np.array(h)
    # thetam=np.array(thetam)
    k0=2*pi/lam
    k_xi=k0*(n1*np.sin(thetai)-h*(lam/grating.T))#(2m+1,1)大小
    KX=np.diag(k_xi/k0)

    kz1=np.zeros((2*m+1,1),dtype=complex)
    kz2=np.zeros((2*m+1,1),dtype=complex)
    #进一步改进。kz的实部必定为正，虚部必定为负
    #if的判断条件同样可以用于ql,m的取值
    for k in range(2*m+1):
        if k0**2*n1**2>k_xi[k]**2:
            kz1[k,0]=k0*np.sqrt(n1**2-(k_xi[k]/k0)**2)
            if kz1[k,0].imag<0:
                kz1[k,0]=np.conj(kz1[k,0])
        elif k0**2*n1**2<=k_xi[k]**2:
            kz1[k,0]=-j*k0*np.sqrt((k_xi[k]/k0)**2-n1**2)
            if kz1[k,0].imag<0:
                kz1[k,0]=np.conj(kz1[k,0])
    
    for k in range(2*m+1):
        if k0**2*n2**2>k_xi[k]**2:
            kz2[k,0]=k0*np.sqrt(n2**2-(k_xi[k]/k0)**2)
            if kz2[k,0].imag<0:
                kz2[k,0]=np.conj(kz2[k,0])
        elif k0**2*n2**2<=k_xi[k]**2:
            kz2[k,0]=-j*k0*np.sqrt((k_xi[k]/k0)**2-n2**2)
            if kz2[k,0].imag<0:
                kz2[k,0]=np.conj(kz2[k,0])

    kz1=kz1.reshape(-1)
    kz2=kz2.reshape(-1)
    kz1=np.diag(kz1)#(2m+1,2m+1)
    kz2=np.diag(kz2)
    # Z1=kz1/(n1**2*k0)
    # Z2=kz2/(n2**2*k0)
    Y1=kz1/k0
    Y2=kz2/k0
    delta_i0=np.zeros((len(k_xi),1))
    delta_i0[m]=1
    n_delta_i0=delta_i0*j*np.cos(thetai)*n1

    #构造T矩阵系数f,g
    f=np.eye(2*m+1)
    g=j*Y2
    inv_a_XL=np.eye(2*m+1)
    #求解R的系数
    for i in range(len(d)):
        [f,g,b]=iteration(n_rd[i,0],n_gr[i,0],d[i,0],k0,KX,m,f,g,fill_factor[i,0])
        inv_a_XL=inv_a_XL@b

    #试试高斯消去法
    T=np.linalg.inv(j*np.matmul(Y1,f)+g)
    T=np.matmul(T,(np.matmul(j*Y1,delta_i0)+n_delta_i0))
    R=np.matmul(f,T)-delta_i0
    T=inv_a_XL@T

    #求解衍射效率
    DE_ri=np.real(R*np.conj(R))*np.real(kz1/(k0*n1*np.cos(thetai)))
    DE_ti=np.real(T*np.conj(T))*np.real(kz2/(k0*n1*np.cos(thetai)))
    for i in range(2*m+1):
        T_effi[count,i]=DE_ti[i,i]#DE_ti只有对角线上有数
        R_effi[count,i]=DE_ri[i,i]
    T_sum[count,0]=sum(T_effi[count,:])
    R_sum[count,0]=sum(R_effi[count,:])
    # T_effi[count,0]=sum(sum(DE_ti))
    # R_effi[count,0]=sum(sum(DE_ri))
    count+=1

plt.figure()
# plt.plot(wavelengh,T_sum,label='Transmission')
# plt.plot(wavelengh,R_sum,label='Reflection')
# plt.plot(wavelengh,R_effi+T_effi,label='Sum')
VirtualLab=[0.00416,0.00125,0.0112,0.0424,0.0241,0.18207,0.37863,0.18207,0.0241,0.0424,0.0112,0.00125,0.00416]
VirtualLab_Sum=sum(VirtualLab)
# plt.plot(h,T_effi.T,label='Transmission')
plt.plot(h,R_effi.T,label='Reflection',marker='v')
plt.plot(h,VirtualLab,label="VirtualLab",marker='^')
plt.axhline(R_sum,label="Sum",color="red",marker="*")
plt.axhline(VirtualLab_Sum,label="VirtualLab_Sum",marker='h')
plt.xlabel("diffraction order")
plt.ylabel("diffraction efficiency")
plt.title("4微米周期 占空比0.5 槽深2微米 材料Al")
plt.legend()
plt.show()