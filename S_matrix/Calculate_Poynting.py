import numpy as np

def Calculate_Poynting(Eigenvector,Eigenvalue,kx,Constant):
    '''
    本函数用于计算本征模态的坡印廷矢量,根据计算结果的正负进行排序
    '''
    num=np.shape(Eigenvector)[1]
    Sz=np.zeros(num,dtype=complex)
    Sx=np.zeros_like(Sz)
    Sy=np.zeros_like(Sz)
    block_size=int(num/4)
    for i in range(num):
        temp=Eigenvector[:,i]
        Ex=temp[:block_size]
        Ey=temp[block_size:2*block_size]
        Hx=temp[2*block_size:3*block_size]
        Hy=temp[3*block_size:]
        Ez=-Ex-Ey-np.diag(kx)/Constant['e1']*Hy
        Hz=np.diag(kx)*Ey
        Sz[i]=np.dot(Ex,Hy)-np.dot(Ey,Hx)
        Sx[i]=np.dot(Ey,Hz)-np.dot(Ez,Hy)
        Sy[i]=np.dot(Ez,Hx)-np.dot(Ex,Hz)
    Poynting_Result=np.where(Sz.imag==0,Sz.real,Sz.imag)
    #首先对Poynting向量初步排序，正的排在前半，负的排在后半
    zero=np.where(Poynting_Result==0)
    zero_p=np.where(Eigenvalue[zero].real>0)
    p=zero[0][zero_p]
    Poynting_P_ind=np.concatenate([np.where(Poynting_Result>0)[0],p])
    Poynting_P_ind=np.sort(Poynting_P_ind)#坡印廷矢量为正，按升序排列
    ##############坡印廷矢量中负的项的排序
    Poynting_N_ind=np.where(Poynting_Result<0)
    zero_n=np.where(Eigenvalue[zero].real<0)
    n=zero[0][zero_n]
    Poynting_N_ind=np.concatenate([np.where(Poynting_Result<0)[0],n])
    Poynting_N_ind=np.sort(Poynting_N_ind)#坡印廷矢量为负，按升序排列
    new_ind=np.concatenate([Poynting_P_ind,Poynting_N_ind])
    Eigenvalue=Eigenvalue[new_ind]
    Eigenvector=Eigenvector[:,new_ind]
    #需要进一步对Eigenvector和Eigenvalue排序，按照Eigenvalue虚部降序排序
    # temp=Eigenvalue.real**2+Eigenvalue.imag**2
    ####先采用虚部的绝对值排序，虚部为0则用实部的值
    # temp=np.where(Eigenvalue.imag==0,Eigenvalue.real,Eigenvalue.imag)
    # half=temp.shape[0]//2
    # P_ind=np.argsort(-temp[:half])
    # N_ind=np.argsort(-temp[half:])
    # new_ind=np.concatenate([P_ind,N_ind+half])
    # Eigenvalue=Eigenvalue[new_ind]
    # Eigenvector=Eigenvector[:,new_ind]
    #########计算C的值
    C=np.zeros_like(Sz)
    for i in range(num):
        C[i]=np.dot(np.abs(Sx),np.abs(Sx))/(np.dot(np.abs(Sx),np.abs(Sx))+np.dot(np.abs(Sy),np.abs(Sy)))
        # temp=Eigenvector[:,i]
        # Ex=temp[:block_size]
        # Ey=temp[block_size:2*block_size]
        # C[i]=np.dot(np.abs(Ex),np.abs(Ex))/(np.dot(np.abs(Ex),np.abs(Ex))+np.dot(np.abs(Ey),np.abs(Ey)))
    half=C.shape[0]//2
    P_ind=np.argsort(-C[:half])
    N_ind=np.argsort(-C[half:])
    new_ind=np.concatenate([P_ind,N_ind+half])
    Eigenvalue=Eigenvalue[new_ind]
    Eigenvector=Eigenvector[:,new_ind]
    return Eigenvector,Eigenvalue