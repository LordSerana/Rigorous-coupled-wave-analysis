import numpy as np

def Eigen(A,B1,B2,a_mat,nDim):
    '''
    求解特征值和特征向量
    '''
    IB1=np.linalg.solve(np.diag(B1),np.eye(nDim))
    IB2=np.linalg.solve(np.diag(B2),np.eye(nDim))
    AUX_mat=np.eye(nDim)+a_mat@a_mat
    #eigenvalue matrix from (12),incident medium
    ChandM1=np.block([[-IB1@(np.diag(A)@a_mat+a_mat@np.diag(A)),IB1@AUX_mat],[np.eye(nDim),np.zeros((nDim,nDim))]])
    #eigenvalue matrix from (12),transmission medium
    ChandM2=np.block([[-IB2@(np.diag(A)@a_mat+a_mat@np.diag(A)),IB2@AUX_mat],[np.eye(nDim),np.zeros((nDim,nDim))]])
    rho1,V1=np.linalg.eig(ChandM1)
    rho2,V2=np.linalg.eig(ChandM2)
    rho1=1/rho1#eigenvalue of ChandM1,计算的是原本的ρ值
    rho2=1/rho2#eigenvalue of ChandM2,计算的是原本的ρ值
    return V1,rho1,V2,rho2