import numpy as np

def SortEigenvalue(eig,vect,threshold):
    real_parts=np.real(eig)
    imag_parts=np.imag(eig)
    abs_real=np.abs(real_parts)
    abs_imag=np.abs(imag_parts)
    #=======判断特征值的主导性
    n=len(eig)
    is_real_dominant=(abs_real>threshold*abs_imag)
    is_imag_dominant=(abs_imag>threshold*abs_real)
    is_mixed=(~is_real_dominant)&(~is_imag_dominant)
    idxRealAndPositive=np.zeros(n,dtype=bool)
    idxRealAndNegative=np.zeros(n,dtype=bool)
    idxImagAndPositive=np.zeros(n,dtype=bool)
    idxImagAndNegative=np.zeros(n,dtype=bool)
    #===============实数主导,按实数正负分类
    real_dominant_indices=np.where(is_real_dominant)[0]
    for idx in real_dominant_indices:
        if real_parts[idx]>0:
            idxRealAndPositive[idx]=True
        else:
            idxRealAndNegative[idx]=True
    #================虚数主导
    imag_dominant_indices=np.where(is_imag_dominant)[0]
    for idx in imag_dominant_indices:
        if imag_parts[idx]>0:
            idxImagAndPositive[idx]=True
        else:
            idxImagAndNegative[idx]=True
    #================混合特征值,复数
    mixed_indices=np.where(is_mixed)[0]
    for idx in mixed_indices:
        if imag_parts[idx]>0:
            idxImagAndPositive[idx]=True
        else:
            idxImagAndNegative[idx]=True

    eig_real_p=eig[idxRealAndPositive]
    vect_real_p=vect[:,idxRealAndPositive]
    eig_real_n=eig[idxRealAndNegative]
    vect_real_n=vect[:,idxRealAndNegative]
    eig_imag_p=eig[idxImagAndPositive]
    vect_imag_p=vect[:,idxImagAndPositive]
    eig_imag_n=eig[idxImagAndNegative]
    vect_imag_n=vect[:,idxImagAndNegative]

    if len(eig_real_p)!=0:
        ind=np.argsort(eig_real_p.real)[::-1]
        eig_real_p=eig_real_p[ind]
        vect_real_p=vect_real_p[:,ind]
    if len(eig_imag_p)!=0:
        ind=np.argsort(eig_imag_p.imag)
        eig_imag_p=eig_imag_p[ind]
        vect_imag_p=vect_imag_p[:,ind]
    if len(eig_real_n)!=0:
        ind=np.argsort(eig_real_n.real)
        eig_real_n=eig_real_n[ind]
        vect_real_n=vect_real_n[:,ind]
    if len(eig_imag_n)!=0:
        ind=np.argsort(eig_imag_n.imag)[::-1]
        eig_imag_n=eig_imag_n[ind]
        vect_imag_n=vect_imag_n[:,ind]
    
    eig_p=np.block([eig_real_p,eig_imag_p])
    vect_p=np.block([vect_real_p,vect_imag_p])
    eig_n=np.block([eig_real_n,eig_imag_n])
    vect_n=np.block([vect_real_n,vect_imag_n])
    return eig_p,vect_p,eig_n,vect_n