import numpy as np

def Star(S1,S2):
    assert S1.shape==S2.shape
    half=int(S1.shape[0]/2)
    S11=S1[:half,:half]+S1[:half,half:]@np.linalg.inv(np.eye(half)-S2[:half,:half]@S1[half:,half:])@S2[:half,:half]@S1[half:,:half]
    S12=S1[:half,half:]@np.linalg.inv(np.eye(half)-S2[:half,:half]@S1[half:,half:])@S2[:half,half:]
    S21=S2[half:,:half]@np.linalg.inv(np.eye(half)-S1[half:,half:]@S2[:half,:half])@S1[half:,:half]
    S22=S2[half:,half:]+S2[half:,:half]@np.linalg.inv(np.eye(half)-S1[half:,half:]@S2[:half,:half])@S1[half:,half:]@S2[:half,half:]
    S=np.block([[S11,S12],[S21,S22]])
    return S