import numpy as np
import cmath
from scipy import linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import quad
#对光栅区域的傅里叶谐波展开，使用Rytov条件。只需注意的是n_gr=n1
#本脚本求解的为TE偏振态
#本脚本仅求解反射波

class Rectangular:
    '''
    T:光栅周期
    Amplitude:光栅齿的高度
    fill_factor:光栅脊部的占空比
    '''
    T=0
    Amplitude=0
    fill_factor=0
    def __init__(self,T,Amplitude,fill_factor):
        self.T=T
        self.Amplitude=Amplitude
        self.fill_factor=fill_factor
    
class Triangular:
    T=0
    base_angle=0
    fill_factor=0
    def __init__(self,T,base_angle,fill_factor):
        self.T=T
        self.base_angle=base_angle
        self.fill_factor=fill_factor

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

def iteration(n_rd,n_gr,d,k0,kx,m,f,g):
    #首先构造周期结构介电常数傅里叶变换矩阵E
    #Moharam论文中使用的是介质光栅公式，即基底和光栅区域使用的介电常数不同，因此对于普通光栅情况（非介质光栅）情况，应使用傅里叶级数展开方式
    #介电常数函数的变化就是光栅区域几何外形的变化情况
    #先试验矩形光栅的情况
    E=np.zeros((2*m+1,2*m+1),dtype=np.float64)
    E=E.astype('complex')
    for i in range(2*m+1):
        for j in range(2*m+1):
            if i==j:
                E[i,j]=n_rd**2*fill_factor+n_gr**2*(1-fill_factor)
            else:
                k=i-j
                E[i,j]=(n_rd**2-n_gr**2)*np.sin(pi*k*fill_factor)/(pi*k)
    
    A=kx**2-E
    Q,W=LA.eig(A)#Q是A矩阵的特征值，W是A矩阵的特征向量
    Q=np.sqrt(Q)#应该注意的是，Q只取正值
    # Q=Q.real#能保证Q是正值吗？
    Q=np.diag(Q)
    V=W@Q
    X=np.diag(np.exp(-k0*d*np.diag(Q)))
    #通过f_n+1,g_n+1求解a_n,b_n
    # f_g=np.concatenate((f,g),axis=0)
    temp=np.block([[-W,f],[V,g]])
    temp2=np.concatenate((W@X,V@X),axis=0)
    ab=np.linalg.inv(temp)@temp2
    a=ab[:2*m+1,:]
    b=ab[2*m+1:,:]
    #矩阵一大，求逆就变得极为困难，可以采用其他方法
    #手动求逆
    # Wi=np.linalg.inv(W)
    # Vi=np.linalg.inv(V)
    # Oi=0.5*np.block([[Wi,Vi],[Wi,-Vi]])
    # a=0.5*(Wi*f+Vi*g)
    # b=0.5*(Wi*f-Vi*g)
    '''
    #LU分解法
    # P,L,U=lu(temp)
    # B=f_g
    # Pb=np.dot(P,B)
    # y=np.zeros_like(B,dtype=np.float64)
    # n=len(B)
    # for i in range(n):
    #     y[i]=Pb[i]-sum(L[i,j]*y[j] for j in range(i))
    # x=np.zeros_like(B,dtype=np.float64)
    # # x=x.astype('complex')
    # for i in range(n-1,-1,-1):
    #     x[i]=(y[i]-sum(U[i,j]*x[j] for j in range(i+1,n)))/U[i,i]
    # a_b=x
    '''

    # a=a_b[:2*m+1,:2*m+1]
    # b=a_b[2*m+1:,:2*m+1]
    f_L=W@(np.eye(2*m+1)+X@a)
    g_L=V@(np.eye(2*m+1)-X@a)
    # f_L=W@(np.eye(2*m+1)+X@a)
    # g_L=V@(np.eye(2*m+1)-X@a)
    return f_L,g_L,b

def slicer(thickness,k):
    '''
    k:切片数
    用于将光栅区域切片,切片厚度取为波长的1/12~1/20较为适宜
    '''
    single_thickness=thickness/k
    # num=int(np.ceil(thickness/single_thickness))#应该向上取整
    d=np.zeros((k,1))
    for i in range(k):
        d[i,0]=single_thickness
    # d[num-1,0]=thickness-(num-1)*single_thickness
    return d

def stable_inv_tikhonov(A,lambda_reg=1e-5):
    n=A.shape[0]
    A_reg=A+lambda_reg*np.eye(n)
    return np.linalg.inv(A_reg)

#下列部分为光栅的系数
m=10#傅里叶谐波数，指的是1~m次,这一项影响的是计算精度
# m_fourier=np.linspace(-m,m,2*m+1)
j=cmath.sqrt(-1)
n1=1.00027#入射区域的折射率
n2=1.00027#出射区域的折射率
thetai=0#入射角
pi=np.pi
period=4*1e-6#光栅周期
#对于硅基底的光栅，需要考虑其复折射率的情况
n_rd=3.8827+j*0.019626#光栅脊部折射率
n_gr=n1#光栅槽部折射率
thickness=20*1e-6#光栅区域2的厚度
fill_factor=0.5#光栅的占空比
d=slicer(thickness,256)
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
    
    # Nonhomo=[1,np.zeros((m-1,1)),j*np.cos(theta)/n1,np.zeros((m-1,1))]
    #构造z方向的波矢分量
    k0=2*pi/lam


    #考虑对k_xi使用补全，使其长度等于耦合波长度。实际上就是对h的长度进行补全,也就没必要计算衍射级次了
    k_xi=k0*(n1*np.sin(thetai)-h*(lam/period))#(2m+1,1)大小
    KX=np.diag(k_xi/k0)

    kz1=(k0**2*n1**2-k_xi**2)
    kz1=kz1.astype('complex')
    kz1=np.sqrt(kz1)
    kz2=(k0**2*n2**2-k_xi**2)
    kz2=kz2.astype('complex')
    kz2=np.sqrt(kz2)
    kz1=np.diag(kz1)#(2m+1,2m+1)
    kz2=np.diag(kz2)
    # Z1=kz1/(n1**2*k0)
    # Z2=kz2/(n2**2*k0)
    Y1=kz1/k0
    Y2=kz2/k0
    delta_i0=np.zeros((len(k_xi),1))
    delta_i0[m]=1
    n_delta_i0=delta_i0*j*np.cos(thetai)*n1

    # R_Coeffi=np.block([[np.eye(2*m+1)],[-j*kz1]])#(2m+1,4m+2)

    #构造T矩阵系数f,g
    f=np.eye(2*m+1)
    g=j*Y2
    inv_a_XL=np.eye(2*m+1)
    #求解R的系数
    for i in range(len(d)):
        [f,g,b]=iteration(n_rd,n_gr,d[i,0],k0,KX,m,f,g)
        inv_a_XL=inv_a_XL@b
        #对a矩阵使用SVD分解看看，迭代解是否稳定。a是病态矩阵
        # if np.linalg.det(a_reconstruct)==0:
        #     raise ValueError("矩阵不可逆")
        # U,S,VT=np.linalg.svd(a_reconstruct)
        # threshold=1e-5
        # S_truncated=np.where(S>threshold,S,0)
        # a_reconstruct=U@S_truncated@VT
        # S_pseudo_inv=np.zeros_like(S_truncated)
        # S_pseudo_inv[S_truncated>0]=1/S_truncated[S_truncated>0]
        # S_pseudo_inv=np.diag(S_pseudo_inv)
        # a_inv=VT.T@S_pseudo_inv@U.T
        # a_inv=np.linalg.inv(a)
        # inv_a_XL=inv_a_XL@a_inv@X
        #最终得到系数f1,g1

    '''逆矩阵法
    Nonhomo=np.zeros((4*m+2,1),dtype=np.float64)
    Nonhomo=Nonhomo.astype('complex')
    Nonhomo[0,0]=1
    Nonhomo[2*m+1,0]=j*np.cos(thetai)/n1
    f_g=np.concatenate((f,g),axis=0)
    temp=np.concatenate((f_g,-R_Coeffi),axis=1)
    Result=LA.inv(temp)*Nonhomo
    Result=[[T1],[R]]
    T=inv_a_XL@Result[:2*m+1,0]#求解T
    R=Result[2*m+1:,0]
    '''

    #试试高斯消去法
    T=np.linalg.inv(j*np.matmul(Y1,f)+g)
    T=np.matmul(T,(np.matmul(j*Y1,delta_i0)+n_delta_i0))
    R=np.matmul(f,T)-delta_i0
    T=inv_a_XL@T
    # T=inv_a_XL@T

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
plt.plot(h,T_effi.T,label='Transmission')
plt.plot(h,R_effi.T,label='Reflection')
plt.legend()
plt.show()