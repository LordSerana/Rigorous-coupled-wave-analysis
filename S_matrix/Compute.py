import numpy as np
import sys
sys.path.append('E:/Project/python')
from S_matrix.Layer_mode import layer_mode
from S_matrix.Build_scatter_side import build_scatter_side
from S_matrix.Star import Star
from S_matrix.CalcEffi import calcEffi
import matplotlib.pyplot as plt
from S_matrix.Calculate_Poynting import Calculate_Poynting

def Compute(Constant,layers):
    kx=Constant['kx']
    ky=Constant['ky']
    temp=Constant['n_Tr']//2
    Ref_set_ind=Constant['Ref_set_ind']
    Trn_set_ind=Constant['Trn_set_ind']
    S_global=np.block(
        [[np.zeros((4*temp+2,4*temp+2),dtype=complex),np.eye(4*temp+2,dtype=complex)],
         [np.eye(4*temp+2,dtype=complex),np.zeros((4*temp+2,4*temp+2),dtype=complex)]])
    for i in layers[1:-1]:
        S=layer_mode(i,Constant,'FFT')
        S_global=Star(S_global,S)
    Ssub,Wsub,LAMsub=build_scatter_side(Constant['n2']**2,1,np.diag(kx),np.diag(ky),Constant['W'],transmission_side=True)
    Ssub=np.block([[Ssub[0],Ssub[1]],[Ssub[2],Ssub[3]]])
    Sref,Wref,LAMref=build_scatter_side(1,1,np.diag(kx),np.diag(ky),Constant['W'])
    Sref=np.block([[Sref[0],Sref[1]],[Sref[2],Sref[3]]])
    S_global=Star(S_global,Ssub)
    S_global=Star(Sref,S_global)
    R_effi,T_effi=calcEffi(Constant['p'],Constant,S_global)
    Constant['R_effi']=R_effi[Ref_set_ind]
    Constant['T_effi']=T_effi[Trn_set_ind]
    return Constant