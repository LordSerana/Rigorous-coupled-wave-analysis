import numpy as np

def Toeplitz(fourier_coeffi,nDim):
    '''
    构造Toeplitz矩阵
    '''
    A=np.zeros((nDim,nDim),dtype=complex)
    q0=len(fourier_coeffi)//2
    for i in range(nDim):
        for j in range(nDim):
            A[i,j]=fourier_coeffi[q0+i-j]
    return A