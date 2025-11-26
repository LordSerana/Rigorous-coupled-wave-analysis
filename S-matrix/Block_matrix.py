import numpy as np

def block_matrix(arrays,axis1=1,axis2=0):
    return np.concatenate([
        np.concatenate(sub_array,axis=axis1)
        for sub_array in arrays
        ],axis=axis2)