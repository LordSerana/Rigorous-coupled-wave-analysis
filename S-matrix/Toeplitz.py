import numpy as np

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