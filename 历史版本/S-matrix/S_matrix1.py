import numpy as np
import matplotlib.pyplot as plt

#本脚本使用RCWA和散射矩阵法(S-Matrix)，分析TE波入射时的衍射效率,参考Rumpf论文
#暂时只考虑一维光栅情况
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
pi=np.pi
j=np.emath.sqrt(-1)
lam=632.8*1e-9
thetai=0*pi/180#xz平面内的入射角
m=41#光栅沿x方向展开的傅里叶谐波数
Nx=1024
a=np.zeros(2*m+1)
a[m]=1
O=np.zeros(2*m+1)
inc=np.concatenate((a,O))
'''
待办:
改进阈值功能，小于阈值的直接设为零
可以尝试自适应截断法
考虑使用Li规则
'''

class Material:
    '''
    n:材料折射率，一般取复数
    '''
    n=0+0j
    material={
        "Al":1.4482+7.5367j,
        "Si":3.8827+0.019626j,
        "SiO2":1.4570+1e-4j,
        "Air":1
    }

    def __init__(self,name):
        '''
        name:使用的材料,需要是字符串,即: "..."
        '''
        self.n=self.material[name]

class Rectangular:
    T=0
    amplitude=0
    fill_factor=0
    name="Rectangular"
    n1=0+0j
    n_rd=0+0j
    n_gr=0+0j
    n2=0+0j
    substrate_thickness=0*1e-6
    def __init__(self,T,amplitude,fill_factor,material_incident,material,matetial_substrate,substrate_thick):
        '''
        T:光栅周期
        amplitude:栅齿高度
        fill_factor:占空比
        material_incident:入射区域的材料,传入字符串
        material:光栅区域使用的材料,传入字符串
        matetial_substrate:光栅基底使用的材料,传入字符串
        '''
        grating_incident=Material(material_incident)
        grating_material=Material(material)
        grating_substrate_material=Material(matetial_substrate)

        self.n1=grating_incident.n
        self.n_rd=grating_material.n
        self.n_gr=self.n1
        self.n2=grating_substrate_material.n
        self.T=T
        self.amplitude=amplitude
        self.fill_factor=fill_factor
        self.substrate_thickness=substrate_thick

class Triangular:
    T=0
    base_angle=0
    fill_factor=0
    amplitude=0
    name="Triangular"
    n1=0
    n_rd=0+0j
    n_gr=0+0j
    n2=0+0j
    substrate_thickness=0
    def __init__(self,T,base_angle,fill_factor,material_incident,material,material_substrate,substrate_thick):
        '''
        T:光栅周期
        base_angle:三角形光栅的底角
        fill_factor:占空比
        material_incident:入射区域的材料，一般是空气
        material:光栅栅齿的材料
        material_substrate:光栅基底的材料
        '''
        self.T=T
        self.base_angle=base_angle/180*pi
        self.fill_factor=fill_factor
        self.amplitude=T*fill_factor/2*np.tan(base_angle)
        grating_incident=Material(material_incident)
        grating=Material(material)
        grating_substrate=Material(material_substrate)

        self.n1=grating_incident.n
        self.n_rd=grating.n
        self.n_gr=self.n1
        self.n2=grating_substrate.n
        self.substrate_thickness=substrate_thick

def grating_model(grating,E,kx,k0,d):
    '''
    本函数实现的功能为求解各层光栅的模态,即光栅矩阵的特征值和特征向量
    m:傅里叶谐波数/倏逝波的级次
    fill_factor:光栅(切片)的占空比
    n_rd:光栅脊部的折射率(可以是复数)
    n_gr:光栅槽部的折射率(一般为空气)
    kx:电磁场的横向波矢
    k0:波矢
    d:光栅(切片)的厚度
    '''
    kz=np.zeros((2*m+1,2*m+1),dtype=complex)
    for i in range(2*m+1):
        if grating.n_rd**2>kx[i,i]**2:
            kz[i,i]=np.sqrt(grating.n_rd**2-kx[i,i]**2)
        else:
            kz[i,i]=-j*np.sqrt(kx[i,i]**2-grating.n_rd**2)
    Q=kx**2-E
    LAM2,W=np.linalg.eig(Q)
    LAM=np.diag(np.sqrt(LAM2))
    omiga=j*kz
    V=Q@np.linalg.inv(omiga)
    X=np.exp(-LAM*k0*d)
    return W,V,X

def construct_s(W,V,W0,V0,X):
    '''
    W:总的W矩阵
    V:总的V矩阵
    X:总的X矩阵
    p:第p层
    '''
    A=np.linalg.inv(W)@W0+np.linalg.inv(V)@V0
    B=np.linalg.inv(W)@W0-np.linalg.inv(V)@V0
    S11=S22=np.linalg.inv(A-X@B@np.linalg.inv(A)@X@B)@(X@B@np.linalg.inv(A)@X@A-B)
    S12=S21=np.linalg.inv(A-X@B@np.linalg.inv(A)@X@B)@X@(A-B@np.linalg.inv(A)@B)
    S=np.block([[S11,S12],[S21,S22]])
    return S

def star(S1,S2):
    '''
    该函数是仅适用于散射矩阵法的特殊的星乘(*)
    '''
    S11=S1[:2*m+1,:2*m+1]+S1[:2*m+1,2*m+1:]@np.linalg.inv(np.eye(2*m+1)-S2[:2*m+1,:2*m+1]@S1[2*m+1:,2*m+1:])@S2[:2*m+1,:2*m+1]@S1[2*m+1:,:2*m+1]
    S12=S1[:2*m+1,2*m+1:]@np.linalg.inv(np.eye(2*m+1)-S2[:2*m+1,:2*m+1]@S1[2*m+1:,2*m+1:])@S2[:2*m+1,2*m+1:]
    S21=S2[2*m+1:,:2*m+1]@np.linalg.inv(np.eye(2*m+1)-S1[2*m+1:,2*m+1:]@S2[:2*m+1,:2*m+1])@S1[2*m+1:,:2*m+1]
    S22=S2[2*m+1:,2*m+1:]+S2[2*m+1:,:2*m+1]@np.linalg.inv(np.eye(2*m+1)-S1[2*m+1:,2*m+1:]@S2[:2*m+1,:2*m+1])@S1[2*m+1:,2*m+1:]@S2[:2*m+1,2*m+1:]
    S=np.block([[S11,S12],[S21,S22]])
    return S

def calcFreeSpace(kx,m):
    kz=np.zeros((2*m+1,2*m+1),dtype=complex)
    for i in range(2*m+1):
        if 1>kx[i,i]**2:
            kz[i,i]=np.sqrt(1-kx[i,i]**2)
        else:
            kz[i,i]=-j*np.sqrt(kx[i,i]**2-1)
    Q=kx**2-np.eye(2*m+1)
    LAM2,W0=np.linalg.eig(Q)
    LAM=np.diag(np.sqrt(LAM2))
    omiga=j*kz
    V0=Q@np.linalg.inv(omiga)
    return W0,V0,kz

def calcRefle():
    pass

def calcTrans(grating,KX,W0,V0):
    E=np.eye(2*m+1,dtype=complex)*grating.n2**2
    kz=np.zeros((2*m+1,2*m+1),dtype=complex)
    for i in range(2*m+1):
        if grating.n2**2>KX[i,i]**2:
            kz[i,i]=np.sqrt(grating.n2**2-KX[i,i]**2)
        else:
            kz[i,i]=-j*np.sqrt(KX[i,i]**2-grating.n2**2)
    Q=KX**2-E
    LAM2,W=np.linalg.eig(Q)
    LAM=np.diag(np.sqrt(LAM2))
    omiga=j*kz
    V=Q@np.linalg.inv(omiga)
    A=np.linalg.inv(W0)@W+np.linalg.inv(V0)@V
    B=np.linalg.inv(W0)@W-np.linalg.inv(V0)@V
    S11=B@np.linalg.inv(A)
    S12=0.5*(A-B@np.linalg.inv(A)@B)
    S21=2*np.linalg.inv(A)
    S22=-np.linalg.inv(A)@B
    S=np.block([[S11,S12],[S21,S22]])
    return S

def slicer(grating,n):
    thickness=grating.amplitude
    fill_factor=np.zeros(n)
    single_thick=thickness/n
    d=np.ones(n)*single_thick
    if grating.name=="Rectangular":
        fill_factor=np.ones(n)*grating.fill_factor
    elif grating.name=="Triangular":
        for i in range(n):
            fill_factor[i]=grating.fill_factor*(2*i+1)/(2*n)
    return d,fill_factor

def seek_order(thetai,grating,lam0):
    '''
    本函数用于计算有效的衍射级次，改善画图功能
    thetai:入射角
    grating:光栅对象
    '''
    thetai=thetai*pi/180
    m_max=0
    m_min=0
    period=grating.T
    #寻找最大衍射级
    m=0
    while (m*lam0-period*np.sin(thetai))/period<=1:
        m+=1
    m_max=m-1
    m=0
    while (m*lam0-period*np.sin(thetai))/period>=-1:
        m-=1
    m_min=m+1
    order=np.linspace(m_min,m_max,m_max-m_min+1)
    return order

def substrate_slicer(grating,n):
    thickness=grating.substrate_thickness
    single_thick=thickness/n
    fill_factor=1
    d=np.ones(n)*single_thick
    return d,fill_factor

def calckz2(grating,KX):
    '''
    计算波矢kz2
    '''
    j=np.emath.sqrt(-1)
    kz2=np.zeros((2*m+1,2*m+1),dtype=complex)
    for i in range(2*m+1):
        if grating.n2**2>KX[i,i]**2:
            kz2[i,i]=np.sqrt(grating.n2**2-KX[i,i]**2)
        else:
            kz2[i,i]=-j*np.sqrt(KX[i,i]**2-grating.n2**2)
    return kz2

def grating_fft(epsilon):
    fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
    # x=np.linspace(1,Nx,Nx)
    # plt.plot(fourier_coeffi,x)
    # plt.show()
    E=np.zeros((2*m+1,2*m+1),dtype=complex)
    p0=int(Nx/2)
    for i in range(2*m+1):
        for j in range(2*m+1):
            k=i-j
            E[i,j]=fourier_coeffi[p0+k]
    return E

# grating=Rectangular(4*1e-6,2*1e-6,0.5,"Air","Al","SiO2",10*1e-6)
grating=Triangular(4*1e-6,31.6,1,"Air","Al","SiO2",10*1e-6)
# phi=0*pi/180#yz平面内的入射角，锥形入射情况
k0=2*pi/lam
kx=grating.n1*np.sin(thetai)
# ky=grating.n1*np.sin(thetai)
h=np.linspace(-m,m,2*m+1)
order=seek_order(thetai,grating,lam)
kxi=k0*(kx-h*lam/grating.T)#波矢沿x方向展开
# kyi=k0*(ky+n*lam/grating.T)#波矢沿y方向展开！！！应该注意，如果真的是二维光栅，其y方向的光栅周期应进行修正
KX=np.diag(kxi/k0)
W0,V0,kz1=calcFreeSpace(KX,m)
kz1=np.diag(kz1)#kz1是入射区域z方向的波矢,或者说代表空气中的波矢
S_global=np.block(
    [[np.zeros((2*m+1,2*m+1),dtype=complex),np.eye(2*m+1,dtype=complex)],[np.eye(2*m+1,dtype=complex),np.zeros((2*m+1,2*m+1),dtype=complex)]])
k=100#切片数
d,fill_factor=slicer(grating,k)
#计算device区域的散射矩阵
for i in range(k):
    #采用fft方法构造光栅介电常数矩阵
    epsilon=np.zeros(Nx,dtype=complex)
    epsilon[:int(fill_factor[i]*Nx)]=grating.n_rd**2
    epsilon[int(fill_factor[i]*Nx):]=grating.n_gr**2
    # epsilon=1/epsilon
    E=grating_fft(epsilon)
    # E=np.linalg.inv(E)
    W,V,X=grating_model(grating,E,KX,k0,d[i])
    S=construct_s(W,V,W0,V0,X)
    S_global=star(S_global,S)
    #检测代码，检查中间层的RT参数
    result=S_global@inc
    R=result[:2*m+1]
    T=result[2*m+1:]
    R_effi=np.real(R*np.conj(R))*np.real(kz1/grating.n1)
    kz2=calckz2(grating,KX)
    kz2=np.diag(kz2)
    T_effi=np.real(T*np.conj(T))*np.real(kz2/grating.n1)

#计算透射区域的矩阵
S=calcTrans(grating,KX,W0,V0)
S_global=star(S_global,S)
kz2=calckz2(grating,KX)
kz2=np.diag(kz2)
#全局散射矩阵计算完成
result=S_global@inc
R=result[:2*m+1]
T=result[2*m+1:]
R_effi=np.real(R*np.conj(R))*np.real(kz1/grating.n1)
T_effi=np.real(T*np.conj(T))*np.real(kz2/grating.n1)

#绘制衍射效率曲线
plt.figure()
plt.plot(order,T_effi[int(order[0]+m):int(order[len(order)-1]+m+1)],label='Transmission')
plt.plot(order,R_effi[int(order[0]+m):int(order[len(order)-1]+m+1)],label='Reflection')
plt.xlabel('Diffraction Order')
plt.ylabel('Diffraction Efficiency')
plt.title('散射矩阵法4微米周期三角光栅')
plt.legend()
plt.show()