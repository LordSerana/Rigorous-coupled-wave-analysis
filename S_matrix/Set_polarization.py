import numpy as np
from Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix

def Set_Polarization(thetai,phi,n1,n2,wavelength,pTE,pTM,m,Nx,accuracy,grating,n,Rough=False):
    '''
    thetai:x方向入射角
    phi:y方向入射角
    n1:入射介质折射率
    n2:基底介质折射率
    wavelength:波长
    pTE:TE分量
    pTM:TM分量
    m:截断数
    '''
    Constant={}
    Constant['thetai']=np.radians(thetai)
    Constant['phi']=np.radians(phi)
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
    if Rough==True:
        h=Roughness(0.32*1e-6,Nx,115)
        Constant['Rough']=h
    return Constant

def Roughness(Ra,Nx,seed=None):
    rng=np.random.default_rng(seed)
    h=rng.normal(size=Nx)
    h=h/np.mean(np.abs(h))*Ra
    return h