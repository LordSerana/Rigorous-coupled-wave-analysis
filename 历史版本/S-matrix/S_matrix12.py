import numpy as np
import matplotlib.pyplot as plt

#矩形光栅结果完全一致,接下来进行代码结构的改进
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
pi=np.pi
lam=632.8*1e-9
thetai=0*pi/180#xz平面内的入射角
pTE=1
pTM=0
aTE=[0,1,0]
aTM=[np.cos(thetai),0,np.sin(thetai)]
P=pTE*aTE+pTM*aTM
m=20#光栅沿x方向展开的傅里叶谐波数
delta0=np.zeros(2*m+1)
delta0[m]=1
esrc=np.concatenate((P[0]*delta0,P[1]*delta0))
Nx=1024
'''
待办:
改进阈值功能，小于阈值的直接设为零
可以尝试自适应截断法
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

def block_matrix(arrays,axis1=1,axis2=0):
    return np.concatenate([
        np.concatenate(sub_array,axis=axis1)
        for sub_array in arrays
        ],axis=axis2)

def grating_model(E,E_recip,k0,d):
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
    E_inv=np.linalg.inv(E)
    E_recip_inv=np.linalg.inv(E_recip)
    P=np.block([[KX@E_inv@KY,np.eye(2*m+1)-KX@E_inv@KX],[KY@E_inv@KY-np.eye(2*m+1),-KY@E_inv@KX]])
    Q=np.block([[KX@KY,E-KX@KX],[KY@KY-E_recip_inv,-KY@KX]])
    omiga2=P@Q
    LAM2,W=np.linalg.eig(omiga2)
    LAM=np.sqrt(LAM2,dtype=complex)
    LAM = -np.abs(LAM.real) + 1j*np.abs(LAM.imag)
    V=Q@W@np.diag(1/LAM)
    X=np.exp(LAM*k0*d)
    X=np.diag(X)
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
    assert S1.shape==S2.shape
    half=int(S1.shape[0]/2)
    S11=S1[:half,:half]+S1[:half,half:]@np.linalg.inv(np.eye(half)-S2[:half,:half]@S1[half:,half:])@S2[:half,:half]@S1[half:,:half]
    S12=S1[:half,half:]@np.linalg.inv(np.eye(half)-S2[:half,:half]@S1[half:,half:])@S2[:half,half:]
    S21=S2[half:,:half]@np.linalg.inv(np.eye(half)-S1[half:,half:]@S2[:half,:half])@S1[half:,:half]
    S22=S2[half:,half:]+S2[half:,:half]@np.linalg.inv(np.eye(half)-S1[half:,half:]@S2[:half,:half])@S1[half:,half:]@S2[:half,half:]
    S=np.block([[S11,S12],[S21,S22]])
    return S

def homogeneous_isotropic_matrix(er,ur,kx,ky):
    n=er*ur#不需要平方，已经是平方后的结果
    W=np.eye(kx.shape[0]*2)
    kz=np.sqrt((n-kx**2-ky**2).astype('complex'))
    LAM=np.concatenate([1j*kz,1j*kz],axis=0)
    LAM=np.concatenate([LAM,-LAM],axis=0)#取负LAM的原因是V22=-V11，自然特征值也应取反

    V11=np.diag(kx*ky/kz)
    V12=np.diag((n-kx**2)/kz)
    V21=np.diag((ky**2-n)/kz)
    V22=-V11

    V=-1j/ur*block_matrix([
        [V11,V12],
        [V21,V22]
        ])
    W=block_matrix([
        [W,W],
        [V,-V]
        ])
    return LAM,W

def build_scatter_from_AB(A,B):
    half=A.shape[0]//2
    a11=A[:half,:half]
    a12=A[:half,half:]
    a21=A[half:,:half]
    a22=A[half:,half:]

    b11=B[:half,:half]
    b12=B[:half,half:]
    b21=B[half:,:half]
    b22=B[half:,half:]

    S=np.linalg.solve(
        block_matrix([
            [-a12,b11],
            [-a22,b21]
        ]),
        block_matrix([
            [a11,-b12],
            [a21,-b22]
        ])
    )

    s11=S[:half,:half]
    s12=S[:half,half:]
    s21=S[half:,:half]
    s22=S[half:,half:]
    return (s11,s12,s21,s22)

def build_scatter_side(er,ur,kx,ky,W0,transmission_side=False):
    LAM,W=homogeneous_isotropic_matrix(er,ur,kx,ky)
    if transmission_side:
        A=W0
        B=W
    else:
        A=W
        B=W0
    return build_scatter_from_AB(A,B),W,LAM

def slicer(grating,n):
    '''
    改进下切片算法，直接返回介电常数数组
    '''
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

def grating_fft(grating,fill_factor,case):
    '''
    fill_factor:光栅切片占空比;case:介电常数矩阵计算规则
    '''
    if case=='FFT':
        epsilon=np.ones(Nx,dtype=complex)*grating.n_gr**2
        temp=int(fill_factor*Nx/2)
        q0=int(Nx/2)
        epsilon[q0-temp:q0+temp+1]=grating.n_rd**2
        epsilon_recip=1/epsilon            
        fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
        fourier_coeffi_recip=np.fft.fftshift(np.fft.fft(epsilon_recip,axis=0)/epsilon.shape[0])
        E=Toeplitz(fourier_coeffi,2*m+1)
        E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    else:
        fourier_coeffi=np.zeros(4*m+1,dtype=complex)
        fourier_coeffi_recip=np.zeros(4*m+1,dtype=complex)
        temp=grating.n_rd**2-grating.n_gr**2
        temp2=(1/grating.n_rd)**2-(1/grating.n_gr)**2
        for i in range(-2*m,2*m+1,1):
            if i!=0:
                fourier_coeffi[i+2*m]=temp*np.sin(pi*i*fill_factor)/pi/i
                fourier_coeffi_recip[i+2*m]=temp2*np.sin(pi*i*fill_factor)/pi/i
            else:
                fourier_coeffi[i+2*m]=grating.n_rd**2*fill_factor+grating.n_gr**2*(1-fill_factor)
                fourier_coeffi_recip[i+2*m]=(1/grating.n_rd)**2*fill_factor+(1/grating.n_gr)**2*(1-fill_factor)
        E=Toeplitz(fourier_coeffi,2*m+1)
        E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
        #需要构造E_recip
    return E,E_recip

def calck():
    k0=2*pi/lam
    kinc=[np.sin(thetai)*np.cos(phi),np.sin(thetai)*np.sin(phi),np.cos(thetai)]
    kinc=[temp*grating.n1 for temp in kinc]
    KX=np.diag(kinc[0]-2*pi*h/k0/grating.T)
    KY=np.zeros((2*m+1,2*m+1))
    kzref=np.zeros((2*m+1,2*m+1),dtype=complex)
    for i in range(2*m+1):
        kzref[i,i]=np.sqrt(grating.n1**2-KX[i,i]**2,dtype=complex)
    return k0,kinc,KX,KY,-kzref

def calcEffi():
    Wref=Wtrn=np.eye(4*m+2)
    csrc=np.linalg.inv(Wref)@esrc
    cref=S_global[:4*m+2,:4*m+2]@csrc
    ctrn=S_global[4*m+2:,:4*m+2]@csrc
    eref=Wref@cref
    rx=eref[:2*m+1]
    ry=eref[2*m+1:]
    rz=-np.linalg.inv(kzref)@(KX@rx+KY@ry)
    etrn=Wtrn@ctrn
    tx=etrn[:2*m+1]
    ty=etrn[2*m+1:]
    tz=-np.linalg.inv(kzref)@(KX@tx+KY@ty)
    r2=abs(rx)**2+abs(ry)**2+abs(rz)**2
    R=np.real(-kzref)/np.real(kinc[2])*r2
    R_effi=np.sum(R,axis=1)
    t2=abs(tx)**2+abs(ty)**2+abs(tz)**2
    T=np.real(-kzref)/np.real(kinc[2])*t2
    T_effi=np.sum(T,axis=1)
    return R_effi,T_effi

def calc_diffraction_angle(order):
    '''
    根据传入的衍射级次的有效级次，计算对应的衍射角
    '''
    thetam=np.zeros_like(order)
    j=0
    for i in order:
        temp=np.asin(i*lam/grating.T-np.sin(thetai))
        thetam[j]=temp
        j+=1
    return thetam

def Toeplitz(fourier_coeffi,nDim):
    '''
    构造Toeplitz矩阵
    '''
    A=np.zeros((nDim,nDim),dtype=complex)
    p0=int(fourier_coeffi.shape[0]/2)
    for i in range(nDim):
        for j in range(nDim):
            k=i-j
            A[i,j]=fourier_coeffi[p0+k]
    return A

def diffraction_order(order):
    thetam=calc_diffraction_angle(order)
    thetam=thetam/np.pi*180
    print(thetam)

grating=Rectangular(4*1e-6,2*1e-6,0.5,"Air","Al","Al",1*1e-6)
# grating=Triangular(4*1e-6,31.6,1,"Air","Al","Al",5*1e-6)
phi=0*pi/180#yz平面内的入射角，锥形入射情况
h=np.linspace(-m,m,2*m+1)
k0,kinc,KX,KY,kzref=calck()
order=seek_order(thetai,grating,lam)
LAM,W1=homogeneous_isotropic_matrix(1,1,np.diag(KX),np.diag(KY))
size=int(W1.shape[0]/2)
W0=W1[:size,:size]
V0=W1[size:,:size]
S_global=np.block(
    [[np.zeros((4*m+2,4*m+2),dtype=complex),np.eye(4*m+2,dtype=complex)],[np.eye(4*m+2,dtype=complex),np.zeros((4*m+2,4*m+2),dtype=complex)]])
k=20#切片数,不要超过5
d,fill_factor=slicer(grating,k)
#计算device区域的散射矩阵
for i in range(k):
    E,E_recip=grating_fft(grating,fill_factor[i],'formula')
    W,V,X=grating_model(E,E_recip,k0,d[i])
    S=construct_s(W,V,W0,V0,X)
    S_global=star(S_global,S)
    #检测代码，检查中间层的RT参数

#计算透射区域的矩阵
Ssub,Wsub,LAMsub=build_scatter_side(grating.n2**2,1,np.diag(KX),np.diag(KY),W1,transmission_side=True)
Ssub=np.block([[Ssub[0],Ssub[1]],[Ssub[2],Ssub[3]]])
Sref,Wref,LAMref=build_scatter_side(1,1,np.diag(KX),np.diag(KY),W1)
Sref=np.block([[Sref[0],Sref[1]],[Sref[2],Sref[3]]])
S_global=star(S_global,Ssub)
S_global=star(Sref,S_global)
#全局散射矩阵计算完成
R_effi,T_effi=calcEffi()
#绘制衍射效率曲线
plt.figure()
# plt.plot(order,T_effi[int(order[0]+m):int(order[len(order)-1]+m+1)],label='Transmission')
plt.plot(order,R_effi[int(order[0]+m):int(order[len(order)-1]+m+1)],label='Reflection')
# total_effi=0
# for i in range(int(order[0]+m),int(order[len(order)-1]+m+1)):
#     total_effi=total_effi+T_effi[i]+R_effi[i]
# plt.axhline(y=total_effi,color='green',label='Total_Efficiency')
# VirtualLab_R=[0.00416,0.00125,0.0112,0.0424,0.0241,0.18207,0.37863,0.18207,0.0241,0.0424,0.0112,0.00125,0.00416]#Al,TE,Rectangular
VirtualLab_R=[0.0079361,0.0257,0.001586,0.067849,0.019138,0.23866,0.081028,0.23866,0.019138,0.067849,0.001586,0.0257,0.0079361]#Al,TM,Rectangular
# VirtualLab_R=[0.16591,0.18795,0.06182,0.0113,0.00782,0.015,0.0008,0.015,0.00782,0.0113,0.06182,0.18795,0.16591]#Al,TE,Triangular
plt.plot(order,VirtualLab_R,label='VLab_R')
plt.xlabel('Diffraction Order')
plt.ylabel('Diffraction Efficiency')
plt.title('4微米周期矩形光栅')
plt.legend()
plt.show()