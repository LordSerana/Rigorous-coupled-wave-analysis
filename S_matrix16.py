import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Set_polarization import Set_Polarization
from C_Method.Grating import Triangular
from C_Method.Toeplitze import Toeplitz
from S_matrix.Layer import Layer
from S_matrix.Star import Star
from S_matrix.CalcEffi import calcEffi
from S_matrix.Plot_Effi import Plot_Effi
import matplotlib.pyplot as plt
from S_matrix.F_series_gen import F_series_gen
from S_matrix.Build_scatter_side import build_scatter_side
from S_matrix.Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
###验证4*4矩阵形式的正确性

def Build_M_RCWA(kx, ky, E, E_recip_inv):
    """
    标准全矢量 RCWA 4×4 本征矩阵
    对应 Moharam / Li 体系
    状态向量: [Ex, Ey, Hx, Hy]^T
    """

    nDim = kx.shape[0]
    O = np.zeros((nDim, nDim), dtype=complex)
    I = np.eye(nDim, dtype=complex)

    # --- P block（来自 curl H） ---
    P11 = ky @ E_recip_inv @ ky - I
    P12 = -ky @ E_recip_inv @ kx
    P21 = -kx @ E_recip_inv @ ky
    P22 = kx @ E_recip_inv @ kx - I

    # --- 4×4 block matrix ---
    M = np.block([
        [ O,   O,   O,   I ],
        [ O,   O,  -I,   O ],
        [ P11, P12, O,   O ],
        [ P21, P22, O,   O ]
    ])

    return M

def Build_S_layer(Eigenvector, Eigenvalue, d):
    """
    把 RCWA 周期层的模态传播算符
    转换为等效两端口 S 矩阵
    """

    # 模态传播（注意 i）
    P = np.diag(np.exp(1j * Eigenvalue * d))

    Z = np.zeros_like(P)

    # S 矩阵：尺寸 = 4n × 4n
    S_layer = np.block([
        [Z, P],
        [P, Z]
    ])

    return S_layer

def layer_mode(layer,Constant):
    #计算介电常数卷积矩阵
    Nx=Constant['Nx']
    m=Constant['n_Tr']//2
    epsilon=np.ones(Nx,dtype=complex)
    temp=int(layer.fill_factor*Nx/2)
    q0=int(Nx/2)
    epsilon[q0-temp:q0+temp+1]=layer.n**2
    epsilon_recip=1/epsilon            
    fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
    fourier_coeffi_recip=np.fft.fftshift(np.fft.fft(epsilon_recip,axis=0)/epsilon.shape[0])
    E=Toeplitz(fourier_coeffi,2*m+1)
    E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    E_recip_inv=np.linalg.inv(E_recip)
    return E,E_recip_inv

def Calculate_Poynting(Eigenvector,Eigenvalue):
    '''
    本函数用于计算本征模态的坡印廷矢量,根据计算结果的正负进行排序
    '''
    num=np.shape(Eigenvector)[1]
    Sz=np.zeros(num,dtype=float)
    block_size=int(num/4)
    for i in range(num):
        temp=Eigenvector[:,i]
        Ex=temp[:block_size]
        Ey=temp[block_size:2*block_size]
        Hx=temp[2*block_size:3*block_size]
        Hy=temp[3*block_size:]
        Sz[i]=np.real(np.vdot(Ex,Hy)-np.vdot(Ey,Hx))
    #首先对Poynting向量初步排序，正的排在前半，负的排在后半
    tol=1e-12
    forward=[]
    backward=[]
    for i in range(num):
        if Sz[i]>tol:
            forward.append(i)
        elif Sz[i]<-tol:
            backward.append(i)
        else:
            if np.imag(Eigenvalue[i])>0:
                forward.append(i)
            else:
                backward.append(i)
    new_ind=np.array(forward+backward,dtype=int)
    Eigenvalue=Eigenvalue[new_ind]
    Eigenvector=Eigenvector[:,new_ind]
    half=num//2
    if len(forward)!=half or len(backward)!=half:
        raise ValueError(
            f"Poynting分类错误:forward={len(forward)},backward={len(backward)},应各为{half}"
        )
    return Eigenvector,Eigenvalue

# def Calculate_Gap(kx,ky,Constant,layer):
    E,E_recip_inv=layer_mode(layer,Constant)
    M=Build_M_RCWA(kx,ky,E,E_recip_inv)
    LAM,W=np.linalg.eig(M)
    Eigenvector,Eigenvalue=Calculate_Poynting(W,LAM)
    half=np.shape(Eigenvector[0,:])[0]//2
    V_g_E_P=Eigenvector[:half,:half]
    V_g_E_N=Eigenvector[:half,half:]
    V_g_H_P=Eigenvector[half:,:half]
    V_g_H_N=Eigenvector[half:,half:]
    Constant['V_g_E_P']=V_g_E_P
    Constant['V_g_E_N']=V_g_E_N
    Constant['V_g_H_P']=V_g_H_P
    Constant['V_g_H_N']=V_g_H_N
    return Constant

def Calculate_Ref(Constant):
    Sref,Wref,LAMref=build_scatter_side(Constant['n1'],1,Constant['kx'],Constant['ky'],Constant['W'])
    Sref=np.block([[Sref[0],Sref[1]],[Sref[2],Sref[3]]])
    return Sref

def Calculate_trn(Constant):
    Strn,Wtrn,LAMtrn=build_scatter_side(Constant['n2'],1,Constant['kx'],Constant['ky'],Constant['W'],transmission_side=True)
    Strn=np.block([[Strn[0],Strn[1]],[Strn[2],Strn[3]]])
    return Strn

def Compute(Constant,layers,plot=False):
    kinc=Constant['kinc']
    kx=np.diag(kinc[0]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])#已经归一化
    Constant['kx']=kx
    temp=Constant['n_Tr']//2
    if Constant['dimension']==1:
        ky=np.zeros_like(kx)
        ky[temp,temp]=kinc[1]
    else:
        ky=np.diag(kinc[1]-2*np.pi*Constant['my']/Constant['k0']/Constant['period'])#已经归一化
    Constant['ky']=ky
    kzref=np.zeros((2*temp+1,2*temp+1),dtype=complex)
    for i in range(2*temp+1):
        val=Constant['n1']**2-kx[i,i]**2-ky[i,i]**2
        kz=np.sqrt(val+0j)
        if np.imag(kz)<0:
            kz=-kz
        kzref[i,i]=kz
    Constant['kzref']=-kzref
    nDim=Constant['n_Tr']
    ###构造M矩阵
    LAM,W=homogeneous_isotropic_matrix(1,1,kx,ky)
    temp=int(W.shape[0]/2)
    W0=W[:temp,:temp]
    Constant['W0']=W0
    Constant['W']=W
    V_g_E_P=W[:temp,:temp]
    V_g_E_N=W[:temp,temp:]
    V_g_H_P=W[temp:,:temp]
    V_g_H_N=W[temp:,temp:]
    temp1=np.block([[V_g_E_P,V_g_E_N],[V_g_H_P,V_g_H_N]])#Gap_medium的散射矩阵
    zero=np.zeros((2*nDim,2*nDim))
    S_global=np.block([[zero,np.eye(2*nDim)],[np.eye(2*nDim),zero]])
    S_ref=Calculate_Ref(Constant)
    S_global=Star(S_global,S_ref)
    for i in layers[1:-1]:
        E,E_recip_inv=layer_mode(i,Constant)
        M=Build_M_RCWA(kx,ky,E,E_recip_inv)
        #########计算EigenVector和Eigenvalue，并进行排序
        LAM,W=np.linalg.eig(M)
        Eigenvector,Eigenvalue=Calculate_Poynting(W,LAM)
        # Eigenvector=W
        # Eigenvalue=LAM
        #########构造S矩阵
        Prop=Build_S_layer(Eigenvector,Eigenvalue,Constant['depth'])
        Z=np.zeros_like(Prop)
        I=np.eye(Prop.shape[0])
        S_layer=np.block([[Z,Prop],[Prop,Z]])
        S_global=Star(S_global,S_layer)
    S_trn=Calculate_trn(Constant)
    S_global=Star(S_global,S_trn)
    R_effi,T_effi=calcEffi(Constant['p'],Constant,S_global)
    Plot_Effi(R_effi,T_effi,Constant)

###########################设定仿真常数################################
thetai=np.radians(0)#入射角thetai
phi=np.radians(0)#入射角phi
wavelength=632.8*1e-9
n1=1
n2=1.4482+7.5367j
pTM=0
pTE=1
Constant=Set_Polarization(thetai,phi,wavelength,n1,pTM,pTE)
m=15
Constant['n_Tr']=2*m+1
Constant['mx']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
Constant['my']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
Constant['period']=4*1e-6
Constant['Nx']=2**10
Constant['n2']=n2
Constant['e1']=Constant['n1']**2
Constant['e2']=Constant['n2']**2
Constant['c']=299792458
Constant['omiga']=2*np.pi*Constant['c']/Constant['wavelength']
Constant['accuracy']=1e-9
Constant['error']=0.001#相对误差
# R_effi=[]
Abs_error=[]
Rela_error=[]
#####################设定光栅参数#####################################
grating=Triangular(4*1e-6,30,1)
a,a_diff=grating.profile()
Constant['dimension']=1#光栅是一维光栅
Constant['a']=a
Constant['diff_a']=a_diff
Constant['depth']=Constant['period']/2*np.tan(np.radians(30))
#####################################################################
layers=[
    Layer(n=1,t=1*1e-6),
    Layer(n=1.4482+7.5367j,t=1.8*1e-6,fill_factor=1),
    Layer(n=1.4482+7.5367j,t=4*1e-6)
    ]
Compute(Constant,layers)