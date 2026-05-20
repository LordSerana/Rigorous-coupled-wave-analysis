import numpy as np

def Eigen(Constant):
    a_mat=Constant['a_mat']
    alpha_m=Constant['alpha_m']
    beta1_m=Constant['beta1_m']
    beta2_m=Constant['beta2_m']
    IB1=np.linalg.solve(np.diag(beta1_m**2),np.eye(Constant['n_Tr'],dtype=np.complex128))
    IB2=np.linalg.solve(np.diag(beta2_m**2),np.eye(Constant['n_Tr'],dtype=np.complex128))
    AUX_mat=np.eye(Constant['n_Tr'],dtype=np.complex128)+a_mat@a_mat
    Matrix1=np.block([[-IB1@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m)),IB1@AUX_mat],
    [np.eye(Constant['n_Tr'],dtype=np.complex128),np.zeros((Constant['n_Tr'],Constant['n_Tr']),dtype=np.complex128)]])
    Matrix2=np.block([[-IB2@(np.diag(alpha_m)@a_mat+a_mat@np.diag(alpha_m)),IB2@AUX_mat],
    [np.eye(Constant['n_Tr'],dtype=np.complex128),np.zeros((Constant['n_Tr'],Constant['n_Tr']),dtype=np.complex128)]])
    eig1,vec1=np.linalg.eig(Matrix1)
    eig2,vec2=np.linalg.eig(Matrix2)
    eig1=1/eig1
    eig2=1/eig2
    return eig1,vec1,eig2,vec2