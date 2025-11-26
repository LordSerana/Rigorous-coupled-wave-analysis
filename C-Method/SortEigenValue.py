import numpy as np

def SortEigenvalueChand(V1,rho1,V2,rho2,tol,nDim):
    #分离实特征值，入射介质
    real_eig1p_ind=(abs(np.imag(rho1))<tol)&(np.real(rho1)>tol)
    real_eig1p=rho1[real_eig1p_ind]
    sort_idx1=np.argsort(-real_eig1p)
    real_eig1p=real_eig1p[sort_idx1]
    #分离实特征值，透射介质
    real_eig2n_ind=(abs(np.imag(rho2))<tol)&(np.real(rho2)<-tol)
    real_eig2n=rho2[real_eig2n_ind]
    sort_idx2=np.argsort(-real_eig2n)
    real_eig2n=real_eig2n[sort_idx2]
    #分离虚特征值，入射介质
    imag_eig1p_ind=(np.imag(rho1)>tol)
    imag_eig1p=rho1[imag_eig1p_ind]
    sort_idx3=np.argsort(np.abs(np.imag(imag_eig1p)))
    imag_eig1p=imag_eig1p[sort_idx3]
    #分离虚特征值，透射介质
    imag_eig2n_ind=(np.imag(rho2)<-tol)
    imag_eig2n=rho2[imag_eig2n_ind]
    sort_idx4=np.argsort(np.abs(np.imag(imag_eig2n)))
    imag_eig2n=imag_eig2n[sort_idx4]
    #提取并排序对应的特征向量
    #入射介质虚部特征向量
    s_imag_Vec1p=V1[:nDim,:][:,imag_eig1p_ind]
    imag_Vec1p=s_imag_Vec1p[:,sort_idx3]
    #透射介质虚部特征向量
    s_imag_Vec2n=V2[:nDim,:][:,imag_eig2n_ind]
    imag_Vec2n=s_imag_Vec2n[:,sort_idx4]
    return real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n