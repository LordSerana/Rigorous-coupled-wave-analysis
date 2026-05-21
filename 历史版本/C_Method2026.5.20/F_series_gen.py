import numpy as np

def F_series_gen(a_fun,m,T,nDim,cut_small=True):
    '''
    profile:光栅轮廓函数,m:FFT级次,T:周期长度,nDim:模态数
    该函数返回-N/2~N/2级谐波
    '''
    N=2**m
    tol=1e-9
    x=np.linspace(0,T,N+1)
    f=a_fun(x)
    fhat=f
    fhat[0]=(f[0]+f[-1])/2
    fhat=fhat[:-1]
    fourier_coeffi=np.fft.fftshift(np.fft.fft(fhat)/fhat.shape[0])
    #减去系数中的小量
    if cut_small==True:
        ind_small_real=(np.abs(np.real(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_real]=1j*np.imag(fourier_coeffi[ind_small_real])
        ind_small_imag=(np.abs(np.imag(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_imag]=np.real(fourier_coeffi[ind_small_imag])
    q0=len(fourier_coeffi)//2
    return fourier_coeffi[q0-nDim:q0+nDim+1]