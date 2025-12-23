import numpy as np

def F_series_gen(array,nDim,cut_small=True):
    fourier_coeffi=np.fft.fftshift(np.fft.fft(array)/array.shape[0])
    N=len(array)
    # fourier_coeffi=np.fft.fft(array)
    tol=1e-9
    if cut_small==True:
        ind_small_real=(np.abs(np.real(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_real]=1j*np.imag(fourier_coeffi[ind_small_real])
        ind_small_imag=(np.abs(np.imag(fourier_coeffi))<tol)
        fourier_coeffi[ind_small_imag]=np.real(fourier_coeffi[ind_small_imag])
    q0=N//2
    start_idx=q0-nDim
    end_idx=q0+nDim+1
    # 边界检查
    if start_idx < 0 or end_idx > N:
        available_nDim = min(q0, N - q0 - 1)
        raise ValueError(
            f"nDim={nDim} 超出范围。对于长度{N}的数组，"
            f"最大可用nDim为{available_nDim}"
        )
    return fourier_coeffi[start_idx:end_idx]