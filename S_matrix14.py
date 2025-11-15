import numpy as np
import matplotlib.pyplot as plt

#改进13py的代码结构
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#解决plt画图中文乱码问题
#材料参数：
#"Al":1.4482+7.5367j,"Si":3.8827+0.019626j,"SiO2":1.4570+1e-4j,

class Layer:
    def __init__(self,**kwargs):
        self.n=kwargs.get('n',1)
        self.er=kwargs.get('er',self.n**2)
        self.ur=kwargs.get('ur',1)
        self.t=kwargs.get('t',0)
        self.n=np.sqrt(self.er*self.ur)
        self.fill_factor=kwargs.get('fill_factor',1)

def block_matrix(arrays,axis1=1,axis2=0):
    return np.concatenate([
        np.concatenate(sub_array,axis=axis1)
        for sub_array in arrays
        ],axis=axis2)

def build_scatter_side(er,ur,kx,ky,W0,transmission_side=False):
    LAM,W=homogeneous_isotropic_matrix(er,ur,kx,ky)
    if transmission_side:
        A=W0
        B=W
    else:
        A=W
        B=W0
    return build_scatter_from_AB(A,B),W,LAM

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

def Set_Polarization(thetai,phi,wavelength,n1,pTM,pTE):
    Constant={'thetai':thetai,'phi':phi,'wavelength':wavelength,'n1':n1,
              'pTM':pTM,'pTE':pTE}
    k0=2*np.pi/wavelength
    Constant['k0']=k0
    n=[0,0,1]
    kinc=n1*np.array([np.sin(thetai)*np.cos(phi),np.sin(thetai)*np.sin(phi),np.cos(thetai)])
    Constant['kinc']=kinc
    if thetai==0:
        aTE=[0,1,0]
    else:
        aTE=np.cross(n,kinc)/np.linalg.norm(np.cross(n,kinc))
    aTM=np.cross(kinc,aTE)/np.linalg.norm(np.cross(kinc,aTE))
    p=np.dot(pTE,aTE)+np.dot(pTM,aTM)
    p=p/np.linalg.norm(p)
    Constant['p']=p
    return Constant

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

def layer_mode(layer,Constant,case='FFT'):
    #####从介电常数序列计算Toeplitze矩阵
    Nx=Constant['Nx']
    m=Constant['n_Tr']//2
    if case=='FFT':
        epsilon=np.ones(Nx,dtype=complex)*layer.n**2
        temp=int(layer.fill_factor*Nx/2)
        q0=int(Nx/2)
        epsilon[q0-temp:q0+temp+1]=Constant['n1']**2
        epsilon_recip=1/epsilon            
        fourier_coeffi=np.fft.fftshift(np.fft.fft(epsilon,axis=0)/epsilon.shape[0])
        fourier_coeffi_recip=np.fft.fftshift(np.fft.fft(epsilon_recip,axis=0)/epsilon.shape[0])
        E=Toeplitz(fourier_coeffi,2*m+1)
        E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    else:
        fourier_coeffi=np.zeros(4*m+1,dtype=complex)
        fourier_coeffi_recip=np.zeros(4*m+1,dtype=complex)
        temp=Constant['n1']**2-layer.n**2
        temp2=(1/Constant['n1'])**2-(1/layer.n)**2
        for i in range(-2*m,2*m+1,1):
            if i!=0:
                fourier_coeffi[i+2*m]=temp*np.sin(np.pi*i*layer.fill_factor)/np.pi/i
                fourier_coeffi_recip[i+2*m]=temp2*np.sin(np.pi*i*layer.fill_factor)/np.pi/i
            else:
                fourier_coeffi[i+2*m]=Constant['n1']**2*layer.fill_factor+layer.n**2*(1-layer.fill_factor)
                fourier_coeffi_recip[i+2*m]=(1/Constant['n1'])**2*layer.fill_factor+(1/layer.n)**2*(1-layer.fill_factor)
        E=Toeplitz(fourier_coeffi,2*m+1)
        E_recip=Toeplitz(fourier_coeffi_recip,2*m+1)
    #####计算本层的模态
    E_inv=np.linalg.inv(E)
    E_recip_inv=np.linalg.inv(E_recip)
    kx=Constant['kx']
    ky=Constant['ky']
    P=np.block([[kx@E_inv@ky,np.eye(2*m+1)-kx@E_inv@kx],[ky@E_inv@ky-np.eye(2*m+1),-ky@E_inv@kx]])
    Q=np.block([[kx@ky,E-kx@kx],[ky@ky-E_recip_inv,-ky@kx]])
    omiga2=P@Q
    LAM2,W=np.linalg.eig(omiga2)
    LAM=np.sqrt(LAM2,dtype=complex)
    LAM = -np.abs(LAM.real) + 1j*np.abs(LAM.imag)
    V=Q@W@np.diag(1/LAM)
    X=np.exp(LAM*Constant['k0']*layer.t)
    X=np.diag(X)
    #####构建散射矩阵S
    W0=Constant['W0']
    V0=Constant['V0']
    A=np.linalg.inv(W)@W0+np.linalg.inv(V)@V0
    B=np.linalg.inv(W)@W0-np.linalg.inv(V)@V0
    S11=S22=np.linalg.inv(A-X@B@np.linalg.inv(A)@X@B)@(X@B@np.linalg.inv(A)@X@A-B)
    S12=S21=np.linalg.inv(A-X@B@np.linalg.inv(A)@X@B)@X@(A-B@np.linalg.inv(A)@B)
    S=np.block([[S11,S12],[S21,S22]])
    return S

def Star(S1,S2):
    assert S1.shape==S2.shape
    half=int(S1.shape[0]/2)
    S11=S1[:half,:half]+S1[:half,half:]@np.linalg.inv(np.eye(half)-S2[:half,:half]@S1[half:,half:])@S2[:half,:half]@S1[half:,:half]
    S12=S1[:half,half:]@np.linalg.inv(np.eye(half)-S2[:half,:half]@S1[half:,half:])@S2[:half,half:]
    S21=S2[half:,:half]@np.linalg.inv(np.eye(half)-S1[half:,half:]@S2[:half,:half])@S1[half:,:half]
    S22=S2[half:,half:]+S2[half:,:half]@np.linalg.inv(np.eye(half)-S1[half:,half:]@S2[:half,:half])@S1[half:,half:]@S2[:half,half:]
    S=np.block([[S11,S12],[S21,S22]])
    return S

def calcEffi(p,Constant,S_global):
    m=Constant['n_Tr']//2
    kzref=Constant['kzref']
    kx=Constant['kx']
    ky=Constant['ky']
    kinc=Constant['kinc']
    delta0=np.zeros(2*m+1)
    delta0[m]=1
    esrc=np.concatenate((p[0]*delta0,p[1]*delta0))
    Wref=Wtrn=np.eye(4*m+2)
    csrc=np.linalg.inv(Wref)@esrc
    cref=S_global[:4*m+2,:4*m+2]@csrc
    ctrn=S_global[4*m+2:,:4*m+2]@csrc
    eref=Wref@cref
    rx=eref[:2*m+1]
    ry=eref[2*m+1:]
    rz=-np.linalg.inv(kzref)@(kx@rx+ky@ry)
    etrn=Wtrn@ctrn
    tx=etrn[:2*m+1]
    ty=etrn[2*m+1:]
    tz=-np.linalg.inv(kzref)@(kx@tx+ky@ty)
    r2=abs(rx)**2+abs(ry)**2+abs(rz)**2
    R=np.real(-kzref)/np.real(kinc[2])*r2
    R_effi=np.sum(R,axis=1)
    t2=abs(tx)**2+abs(ty)**2+abs(tz)**2
    T=np.real(-kzref)/np.real(kinc[2])*t2
    T_effi=np.sum(T,axis=1)
    return R_effi,T_effi

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
    
def Error(array1,array2):
    abs_error=abs(array2-array1)
    rela_error=abs(abs_error/array1)
    return max(abs_error),max(rela_error)

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
Constant['accuracy']=1e-9
Constant['error']=0.001#相对误差
R_effi=[]
Abs_error=[]
Rela_error=[]
############################设定仿真设备层#################################
layers=[
    Layer(n=1,t=1*1e-6),
    Layer(n=1.4482+7.5367j,t=1.8*1e-6,fill_factor=0.5),
    Layer(n=1.4482+7.5367j,t=4*1e-6)
    ]

n=0
for m in range(15,40):
    Constant['n_Tr']=2*m+1
    Constant['mx']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
    Constant['my']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
    Constant=Compute(Constant,layers,plot=False)
    R_effi.append(Constant['R_effi'])
    if n!=0:
        abs_error,rela_error=Error(R_effi[-2],R_effi[-1])
        Abs_error.append(abs_error)
        Rela_error.append(rela_error)
    n+=1
print("最大绝对误差:"+str(abs_error))#绝对误差
print("最大相对误差:"+str(rela_error))#相对误差