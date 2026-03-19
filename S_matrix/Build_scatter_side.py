import sys
sys.path.append("E:/Project/python")
from S_matrix.Build_scatter_from_AB import build_scatter_from_AB
from S_matrix.Homogeneous_isotropic_matrix import homogeneous_isotropic_matrix
from S_matrix.Calculate_Poynting import Calculate_Poynting

def build_scatter_side(er,ur,kx,ky,W0,transmission_side=False):
    LAM,W=homogeneous_isotropic_matrix(er,ur,kx,ky)
    W,LAM=Calculate_Poynting(W,LAM)
    if transmission_side:
        A=W0
        B=W
    else:
        A=W
        B=W0
    return build_scatter_from_AB(A,B),W,LAM