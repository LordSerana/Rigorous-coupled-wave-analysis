import numpy as np

def F_series_gen(array,nDim,cut_small=True):
    fourier_coeffi=np.fft.fftshift(np.fft.fft(array)/array.shape[0])
    # fourier_coeffi=np.fft.fft(array)
    tol=1e-9
    if cut_small==True:
        ind_small_real=(np.abs(np.real(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_real]=1j*np.imag(fourier_coeffi[ind_small_real])
        ind_small_imag=(np.abs(np.imag(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_imag]=np.real(fourier_coeffi[ind_small_imag])
    q0=len(fourier_coeffi)//2
    return fourier_coeffi[q0-nDim:q0+nDim+1]