import sys
sys.path.append("E:/Project/Python")
from C_Method.Toeplitz import Toeplitz
from C_Method.Eigen import Eigen
from C_Method.F_series_gen import F_series_gen
from C_Method.SortEigenvalue import SortEigenvalue
from C_Method.GenerateFField import GenerateFField
from C_Method.GenerateGField import GenerateGFieldsChand
import numpy as np
from C_Method.Perturbation import Perturbation

def Compute(Constant):
    a_diff=Constant['a_diff']
    nDim=Constant['n_Tr']
    a_diff_vec=F_series_gen(a_diff,nDim)
    ##观察傅里叶频谱分量
    # temp_x=range(-nDim,nDim+1)
    # plt.plot(temp_x,a_diff_vec)
    # plt.show()
    #####
    a_mat=Toeplitz(a_diff_vec,nDim)
    Constant['a_mat']=a_mat
    rho1,V1,rho2,V2=Eigen(Constant)
    eig1_p,vect1_p,_,_=SortEigenvalue(rho1,V1,100)
    _,_,eig2_n,vect2_n=SortEigenvalue(rho2,V2,100)
    Constant['eig1_p']=eig1_p
    Constant['eig2_n']=eig2_n
    Constant['vect1_p']=vect1_p
    Constant['vect2_n']=vect2_n
    #Assemble F-matrices
    Fmn,Fmk,Fm0,Fmq,Fmr=GenerateFField(Constant)
    #Assemble G-matrices
    Gmn,Gmk,Gm0,Gmq,Gmr=GenerateGFieldsChand(Constant)
    #Assemble matrix for matching boundary conditions and solve the linear system
    Gm0=Gm0.T
    Gm0=Gm0.flatten()
    MatBC=np.block([[Fmn,Fmq,-Fmk,-Fmr],[Gmn,Gmq,-Gmk,-Gmr]])
    MatBC=Perturbation(MatBC)
    VecBC=np.concatenate((-Fm0,-Gm0),axis=0)
    RVec=np.linalg.solve(MatBC,VecBC)
    # RVec=np.linalg.inv(MatBC)@VecBC
    #绘制衍射效率曲线
    return RVec