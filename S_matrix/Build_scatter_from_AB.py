import numpy as np
from Block_matrix import block_matrix

def build_scatter_from_AB(A,B):
    half=A.shape[0]//2
    a11=A[:half,:half]
    a12=A[:half,half:]
    a21=A[half:,:half]
    a22=A[half:,half:]

    b11=B[:half,:half]
    b12=B[:half,half:]
    b21=B[half:,:half]
    b22=B[half:,half:]

    S=np.linalg.solve(
        block_matrix([
            [-a12,b11],
            [-a22,b21]
        ]),
        block_matrix([
            [a11,-b12],
            [a21,-b22]
        ])
    )

    s11=S[:half,:half]
    s12=S[:half,half:]
    s21=S[half:,:half]
    s22=S[half:,half:]
    return (s11,s12,s21,s22)