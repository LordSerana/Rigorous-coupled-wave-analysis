import numpy as np

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
        Sz[i]=np.real(np.dot(1j*Ex,np.conj(Hy))-np.dot(1j*Ey,np.conj(Hx)))
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
            if np.real(Eigenvalue[i])<0:
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