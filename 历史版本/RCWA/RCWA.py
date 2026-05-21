import numpy as np
import cmath
from scipy import linalg as LA
#首先求解最简单的矩形光栅模型吧:)
#注意到平常所用的光栅，其介电常数是恒定的，不需要进行Rytov条件展开，可以简化计算

#接下来进行介电常数的傅里叶展开
def grating_fft(epsilon):
    #本函数用于介电常数均匀的光栅的介电常数傅里叶展开
    '''
    epsilon:光栅的介电常数的离散数组
    '''
    fourier_coefficient=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
    return fourier_coefficient

def grating_fourier_harmonics(order,fill_factor,n_ridge,n_groove,epsilon):
    f=fill_factor
    if n_ridge==n_groove:
        #非Rytov条件
        grating_fft(epsilon=epsilon)
    elif n_ridge!=n_groove:
        #使用Rytov条件
        if order==0:
            return f*n_ridge**2+(1-f)*n_groove**2
        elif order!=0:
            return (n_ridge**2-n_groove**2)*np.sin(order*np.pi*f)/(order*np.pi)

def slicer(Nx):
    '''
    本函数用于将光栅切片
    Nx:光栅沿x方向的切片数
    '''

#接下来构造E矩阵
num_ord=50
Nx=100
fourier_array=grating_fft(None)
E=np.zeros((2*num_ord+1,2*num_ord+1))
E=E.astype('complex')
p_index=np.range(-num_ord,num_ord+1)
for prow in range(2*num_ord+1):
    for pcol in range(2*num_ord+1):
        pfft=p_index[prow]-p_index[pcol]
        E[prow,pcol]=fourier_array[Nx+pfft]

#接下来，构造A矩阵
j=cmath.sqrt(-1)
n1=1#折射率
n2=1
theta=0*np.pi/180
lam0=632.8*1e-9#初始波长
d=1#光栅区域厚度
indices=np.range(-num_ord,num_ord+1)
p=10*1e-6#光栅周期
k0=2*np.pi/lam0
k_xi=k0*(n1*np.sin(theta)-indices*(lam0/p))
KX=np.diag(k_xi/k0)
A=KX@KX-E

eigenvals,W=LA.eig(A)
eigenvals=eigenvals.astype('complex')
Q=np.diag(np.sqrt(eigenvals))
V=W@Q
X=np.diag(np.exp(-k0*Q*d))
K_I=k0**2*(n1**2-(k_xi/k0)**2)
K_II=k0**2*(n2**2-(k_xi/k0)**2)
K_I=K_I.astype('complex')
K_I=np.sqrt(K_I)
K_II=K_II.astype('complex')
K_II=np.sqrt(K_II)
Y_I=np.diag(K_I/k0)
Y_II=np.diag(K_II/k0)
delta_i0=np.zeros((len(k_xi),1))
delta_i0[0]=1
#求解C矩阵
C_prew=np.zeros((len(Y_I)*2),len(Y_I)*2)
#填充C矩阵
temp=Y_I@W*j-V
for i in range(len(Y_I)):
    for j in range(len(Y_I)):
        C_prew[i,j]=temp[i,j]

temp=(j*Y_I@W+V)@X
for i in range(len(Y_I)):
    for j in range(len(Y_I)):
        C_prew[i,j+len(Y_I)]=temp[i,j]

temp=(j*Y_II@W-V)@X
for i in range(len(Y_I)):
    for j in range(len(Y_I)):
        C_prew[i+len(Y_I),j]=temp[i,j]

temp=j*Y_II@W+V
for i in range(len(Y_I)):
    for j in range(len(Y_I)):
        C_prew[i+len(Y_I),j+len(Y_I)]=temp[i,j]

#求解C矩阵
temp=np.zeros((2*num_ord,1))
for i in range(num_ord):
    temp[i,1]=j*K_I/k0*delta_i0[i]+j*n1*np.cos(theta)*delta_i0[i]
C=np.linalg.inv(C_prew)@temp

#求解Ri，Ti矩阵
diffrac_order=5#衍射光的总级次
R=np.zeros((5,1))
T=np.zeros((5,1))
for i in range(diffrac_order):
    for m in num_ord:
        R[i,1]=R[i,1]+W[i,m]*(C[m,1]+C[m+num_ord,1]*np.exp(-k0*Q[m,1]*d))-delta_i0[i,1]
        T[i,1]=T[i,1]+W[i,m]*(C[m,1]*np.exp(-k0*Q[m,1]*d)+C[m+num_ord,1])

#求解完毕