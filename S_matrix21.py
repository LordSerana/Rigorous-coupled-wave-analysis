import numpy as np
import sys
sys.path.append("E:/Project/Python")
from S_matrix.Layer import Layer
from S_matrix.Grating import Sinusoidal,Triangular,Blazed
from S_matrix.Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix
from S_matrix.Layer_mode import layer_mode
from S_matrix.Star import Star
from S_matrix.Build_scatter_side import build_scatter_side
from S_matrix.CalcEffi import calcEffi
from S_matrix.Plot_Effi import Plot_Effi
from openpyxl import load_workbook
from S_matrix.CheckConvergence import CheckConvergence
import time
from matplotlib import pyplot as plt

#暂时固定该版本

def Set_Polarization(thetai,phi,wavelength,pTE,pTM,m,Nx,accuracy,grating,n,layers,Rough=False):
    '''
    thetai:x方向入射角
    phi:y方向入射角
    n1:入射介质折射率
    n2:基底介质折射率
    wavelength:波长
    pTE:TE分量
    pTM:TM分量
    m:截断数
    n:切片数
    '''
    Constant={}
    Constant['thetai']=np.radians(thetai)
    Constant['phi']=np.radians(phi)
    n1=layers[0].n
    n2=layers[-1].n
    Constant['n1']=n1
    Constant['n2']=n2
    Constant['wavelength']=wavelength
    Constant['pTE']=pTE
    Constant['pTM']=pTM
    Constant['k0']=2*np.pi/wavelength
    Constant['n']=n
    n=[0,0,1]
    kinc=Constant['n1']*np.array([np.sin(Constant['thetai'])*np.cos(Constant['phi']),
                                  np.sin(Constant['thetai'])*np.sin(Constant['phi']),np.cos(Constant['thetai'])])
    Constant['kinc']=kinc
    if thetai==0:
        aTE=[0,1,0]
    else:
        aTE=np.cross(n,Constant['kinc'])/np.linalg.norm(np.cross(n,Constant['kinc']))
    aTM=np.cross(Constant['kinc'],aTE)/np.linalg.norm(np.cross(Constant['kinc'],aTE))
    p=np.dot(pTE,aTE)+np.dot(pTM,aTM)
    p=p/np.linalg.norm(p)
    Constant['p']=p
    Constant['n_Tr']=2*m+1#Truncation number
    Constant['period']=grating.T
    Constant['depth_max']=grating.depth
    Constant['fill_factor']=grating.fill_factor
    Constant['mx']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
    Constant['my']=np.arange(-(Constant['n_Tr']//2),Constant['n_Tr']//2+1)
    Constant['Nx']=Nx#x方向上的傅里叶快速变换采样数
    Constant['accuracy']=accuracy
    kx=np.diag(kinc[0]-2*np.pi*Constant['mx']/Constant['k0']/Constant['period'])
    kx=kx.astype('complex')
    Constant['kx']=kx
    ky=np.zeros((Constant['n_Tr'],Constant['n_Tr']))
    ky=ky.astype('complex')
    Constant['ky']=ky
    #====================计算反射侧========================
    kzref=np.zeros((Constant['n_Tr'],Constant['n_Tr']),dtype=complex)
    temp=Constant['n1']**2
    for i in range(Constant['n_Tr']):
        kzref[i,i]=np.sqrt(temp-kx[i,i]**2-ky[i,i]**2,dtype=complex)
    Constant['kzref']=kzref
    Ref_mask=np.abs(np.imag(np.diag(Constant['kzref'])))<Constant['accuracy']
    Ref_set=Constant['mx'][Ref_mask]
    Ref_set_ind=np.where(Ref_mask)
    Constant['Ref_set']=Ref_set
    Constant['Ref_set_ind']=Ref_set_ind
    #==================计算透射侧=============================
    kztrn=np.zeros((Constant['n_Tr'],Constant['n_Tr']),dtype=complex)
    temp=Constant['n2']**2
    for i in range(Constant['n_Tr']):
        kztrn[i,i]=np.sqrt(temp-kx[i,i]**2-ky[i,i]**2,dtype=complex)
    Constant['kztrn']=kztrn
    Trn_mask=np.abs(np.imag(np.diag(Constant['kztrn'])))<Constant['accuracy']
    Trn_set=Constant['mx'][Trn_mask]
    Trn_set_ind=np.where(Trn_mask)
    Constant['Trn_set']=Trn_set
    Constant['Trn_set_ind']=Trn_set_ind
    #==========================================================
    LAM,W=homogeneous_isotropic_matrix(1,1,kx,ky)
    temp=int(W.shape[0]/2)
    W0=W[:temp,:temp]
    V0=W[temp:,:temp]
    Constant['W0']=W0
    Constant['V0']=V0
    Constant['W']=W
    return Constant

def Compute(Constant,layers):
    kx=Constant['kx']
    ky=Constant['ky']
    temp=Constant['n_Tr']//2
    Ref_set_ind=Constant['Ref_set_ind']
    Trn_set_ind=Constant['Trn_set_ind']
    S_global=np.block(
        [[np.zeros((4*temp+2,4*temp+2),dtype=complex),np.eye(4*temp+2,dtype=complex)],
         [np.eye(4*temp+2,dtype=complex),np.zeros((4*temp+2,4*temp+2),dtype=complex)]])
    for i in layers[1:-1]:
        S=layer_mode(i,Constant,'FFT')
        S_global=Star(S_global,S)
    Ssub,Wsub,LAMsub=build_scatter_side(Constant['n2']**2,1,np.diag(kx),np.diag(ky),Constant['W'],transmission_side=True)
    Ssub=np.block([[Ssub[0],Ssub[1]],[Ssub[2],Ssub[3]]])
    Sref,Wref,LAMref=build_scatter_side(1,1,np.diag(kx),np.diag(ky),Constant['W'])
    Sref=np.block([[Sref[0],Sref[1]],[Sref[2],Sref[3]]])
    S_global=Star(S_global,Ssub)
    S_global=Star(Sref,S_global)
    R_effi,T_effi=calcEffi(Constant['p'],Constant,S_global)
    Constant['R_effi']=R_effi[Ref_set_ind]
    Constant['T_effi']=T_effi[Trn_set_ind]
    return Constant

def Slice(layers,grating,Constant):
    '''
    n:切片数
    layers:传进仿真层,对中间层进行切片,并返回新的layers函数
    '''
    Constant['n_extra']=0
    n=Constant['n']
    if grating.name!="Rectangular":
        origin_FillFactor=Constant['fill_factor']
        offset=layers[1].offset
        depth=Constant['depth_max']/Constant['n']#切片层的平均厚度
        layer0=layers[0]
        layer_new=[]
        layer_new.append(layer0)
        Refrac_idx=layers[1].n
        if grating.name=="Blazed":
            #=============具体来说,实现效果为将堆叠结构重整为左边对齐的闪耀光栅结构
            for i in range(n):
                #等体积切片法
                fill_factor=(2*i+1)/2/n*origin_FillFactor
                #等厚切片法
                # fill_factor=i/n*origin_FillFactor
                offset=fill_factor/2-origin_FillFactor/2
                layer=Layer(n=Refrac_idx,t=depth,fill_factor=fill_factor,offset=offset)
                layer_new.append(layer)
            for i in layers[2:]:
                layer_new.append(i)
        elif grating.name=="Triangular":
            for i in range(n):
                #等厚切片
                # fill_factor=i/n*origin_FillFactor
                #等体积切片法
                fill_factor=(2*i+1)/2/n*origin_FillFactor
                offset=0#取0/-0.5都行,即翻转结构
                layer=Layer(n=Refrac_idx,t=depth,fill_factor=fill_factor,offset=offset)
                layer_new.append(layer)
            for i in layers[2:]:
                layer_new.append(i)
        elif grating.name=="Sinusoidal":
            for i in range(n):
                #等体积切片
                z1=i*depth
                z2=(i+1)*depth
                V=grating.Volume(z1,z2)
                avg_fillfactor=V/depth/grating.T
                #等厚切片=======================================================
                # z=(i+1)*depth
                # if z>grating.depth:
                #     z=grating.depth
                # width=grating.Width_at_z(z)
                # avg_fillfactor=width/grating.T
                #===============================================================
                layer=Layer(n=Refrac_idx,t=depth,fill_factor=avg_fillfactor,offset=0)
                layer_new.append(layer)
            for i in layers[2:]:
                layer_new.append(i)
    else:
        return layers
    return layer_new

def Generate_height_array(grating, T, Nx):
    """
    T:光栅周期
    返回两个数组：
    x:一个周期内的离散x坐标(长度为Nx)
    s:每个x处材料-空气界面的z坐标(从光栅顶部算起，向下为正)
    """
    x=np.linspace(0, T, Nx, endpoint=False)  # 周期内均匀采样
    
    if grating.name == "Triangular":
        # 以三角槽为例，材料存在于 s(x) 以下
        # 假设三角形顶点在中心，底边占满整个周期
        # 则边界高度 s(x) = depth * 2 * |x - T/2| / T  （形成倒V形槽）
        half=T/2
        s=grating.depth*2*np.abs(x-half)/T
    elif grating.name == "Blazed":
        # 闪耀光栅：锯齿形轮廓，左低右高（或左高右低）
        s=grating.depth*(x/T)#0到depth线性变化，形成锯齿
    elif grating.name == "Sinusoidal":
        s=grating.depth*0.5*(1+np.sin(2*np.pi*x/T-np.pi/2))#调整相位使顶部/底部合理
    elif grating.name == "Rectangular":
        # 矩形光栅，直接按占空比生成0和depth的阶跃
        s = np.where(np.mod(x,T)<grating.fill_factor*T,0.0,grating.depth)
    else:
        raise ValueError("Unknown grating type")
    return x, s

def Add_roughness(s, T, Nx, rms, corr_length, seed=None):
    '''
    RMS=1.25*Ra,具有单位
    corr_length=0.05~0.3*T
    '''
    rng = np.random.default_rng(seed)
    # 生成带有一定空间相关性的随机扰动
    if corr_length is None or corr_length <= 0:
        rough = rng.normal(0, rms, Nx)
    else:
        # 使用傅里叶滤波生成高斯相关的粗糙度
        dx = T / Nx
        k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi   # 空间角频率
        S = np.exp(-(k * corr_length / 2.0) ** 2)  # 高斯功率谱
        S[0] = 0  # 去掉均值，保证平均高度不变
        amp = np.sqrt(S)
        phases = rng.uniform(0, 2*np.pi, Nx)
        z = amp * np.exp(1j * phases)
        rough = np.fft.ifft(z).real
        # 归一化到指定RMS
        rough = rough / np.std(rough) * rms
    return s + rough

def slice_from_height_array(x, s, T, z_i, z_i1):
    """
    根据高度数组 s 计算第 i 层的 fill_factor 和 offset。
    约定：材料存在于 z > s(x) 的区域。
    """
    # 材料在本层内的法向厚度（沿z方向）：
    # 若 s(x) < z_i   -> 整个切片都是材料，长度 = dz
    # 若 z_i < s(x) < z_i1 -> 材料从 s(x) 到 z_i1，长度 = z_i1 - s(x)
    # 若 s(x) >= z_i1 -> 本层无材料
    length = np.maximum(0, z_i1 - np.maximum(s, z_i))
    
    fill_factor = np.mean(length) / (z_i1 - z_i)
    
    if fill_factor > 0:
        # 计算材料区域的质心，用于确定offset
        x_norm = x / T          # 归一化到 [0,1)
        weights = length
        center = np.sum(x_norm * weights) / np.sum(weights)
        # 通常RCWA中offset定义为相对单元中心的偏移
        offset = center - 0.5
    else:
        offset = 0.0
    
    return fill_factor, offset

#============仿真设备层==============================
layers=[
    Layer(n=1,t=1*1e-6),
    Layer(n=1.4482+7.5367j),
    Layer(n=1.4482+7.5367j,t=4*1e-6),
    ]
grating=Sinusoidal(4*1e-6,1,2*1e-6)
# grating=Triangular(4*1e-6,36,1)
# grating=Blazed(T=4*1e-6,angle=30,fill_factor=1,n=1)
#====================================================
#从左侧入射定义为-，衍射光在0级光左侧为负，右侧为正
#=======切片收敛性校验==========================
# Result_last=0
# start_time=time.time()
# for n in range(5,150,1):
#     layers=[
#         Layer(n=1,t=1*1e-6),
#         Layer(n=1.4482+7.5367j),
#         Layer(n=1.4482+7.5367j,t=4*1e-6),
#         ]
#     Constant=Set_Polarization(thetai=0,phi=0,wavelength=632.8*1e-9,pTE=1,pTM=0,m=50,Nx=2**10,accuracy=1e-9,
#                             grating=grating,n=n,layers=layers,Rough=False)
#     layers=Slice(layers,grating,Constant)
#     Constant=Compute(Constant,layers)
#     Result_new=Constant['R_effi']
#     if n==5:
#         continue
#     else:
#         Error=CheckConvergence(Result_last=Result_last,Result_new=Result_new)
#         print(f"n={n},Error={Error}")
#         if Error <0.03:
#             end_time=time.time()
#             print(f"计算所花时间为:{end_time-start_time}")
#             break
#     Result_last=Result_new
#===============平常情况
Constant=Set_Polarization(thetai=0,phi=0,wavelength=632.8*1e-9,pTE=1,pTM=0,m=40,Nx=2**10,accuracy=1e-9,
                        grating=grating,n=50,layers=layers,Rough=False)
layers=Slice(layers,grating,Constant)
# x,s_ideal=Generate_height_array(grating=grating,T=grating.T,Nx=Constant['Nx'])
# s_rough=Add_roughness(s=s_ideal,T=grating.T,Nx=Constant['Nx'],rms=1*1e-9,corr_length=0.2*grating.T,seed=648)
# plt.plot(x,s_rough)
# plt.show()
Constant=Compute(Constant,layers)
#=========打印输出
start_order=Constant['Ref_set'][0]
for order,efficiency in enumerate(Constant['R_effi']):
    print(f"{order+start_order}:{efficiency:.4f}")
# print(Constant['R_effi'])
print(f"Sum:{sum(Constant['R_effi'])}")
# Plot_Effi(Constant,[],[])
#========================================================
# file_path='C:/Users/123/Desktop/三角光栅2微米扫描数据.xlsx'
# wb=load_workbook(file_path)
# ws=wb.active
# start_col=2
# start_row=2#行
# counter=0
# save_interval=5
# for fill_factor in np.arange(start=0.7,stop=1,step=0.1):
#     for angle in np.arange(start=5,stop=45.5,step=0.5):
#         layers=[
#             Layer(n=1,t=1*1e-6),
#             Layer(n=1.4482+7.5367j,t=2*1e-6,fill_factor=fill_factor),
#             Layer(n=1.4482+7.5367j,t=4*1e-6)
#             ]
#         # grating=Blazed(4*1e-6,angle=angle,fill_factor=fill_factor,n=1)
#         grating=Triangular(T=2*1e-6,base_angle=angle,fill_factor=fill_factor)
#         Constant=Set_Polarization(thetai=0,phi=0,n1=1,n2=1.4482+7.5367j,wavelength=632.8*1e-9,
#         pTE=1,pTM=0,m=20,Nx=2**10,accuracy=1e-9,grating=grating,n=50)

#         layers=Slice(layers,grating,Constant)
#         Constant=Compute(Constant,layers)
#         R3=Constant['R_effi'][0]
#         ws.cell(row=start_row,column=start_col,value=R3)
#         counter+=1
#         start_row+=1
#         if counter%save_interval==0:
#             wb.save(file_path)
#             print(f"已保存{counter}个数据")
#     start_col+=1
#     start_row=2
# wb.save(file_path)