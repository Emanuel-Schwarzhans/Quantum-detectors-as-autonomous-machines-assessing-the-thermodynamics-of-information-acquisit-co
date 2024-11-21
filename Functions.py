import numpy as np
import scipy
from functools import reduce
from qutip import *






def sig_p(i,L):
    s_list=[]
    for ii in range(0, L,1):
        s_list.append(qeye(2))
    if i is not None and i < L:
        s_list[i] = sigmap()
    else:
        print("L out of bounds")

    return tensor(s_list)


def sig_m(i,L):
    s_list=[]
    for ii in range(0, L,1):
        s_list.append(qeye(2))
    if i is not None and i < L:
        s_list[i] = sigmam()
    else:
        print("L out of bounds")

    return tensor(s_list)

def sig_x(i,L):
    s_list=[]
    for ii in range(0, L,1):
        s_list.append(qeye(2))
    if i is not None and i < L:
        s_list[i] = sigmax()
    else:
        print("L out of bounds")

    return tensor(s_list)

def sig_y(i,L):
    s_list=[]
    for ii in range(0, L,1):
        s_list.append(qeye(2))
    if i is not None and i < L:
        s_list[i] = sigmay()
    else:
        print("L out of bounds")

    return tensor(s_list)

def sig_z(i,L):
    s_list=[]
    for ii in range(0, L,1):
        s_list.append(qeye(2))
    if i is not None and i < L:
        s_list[i] = sigmaz()
    else:
        print("L out of bounds")

    return tensor(s_list)

def Vectorize_Mat (M):
    return np.concatenate([M[:, i:i+1] for i in range(M.shape[1])], axis=0)

def matrizise_vector(vec):
    n = int(np.sqrt(len(vec)))
    return np.array([[vec[i * n + j] for j in range(n)] for i in range(n)])

#chop does not yet work perfectly
def chop(matrix, threshold):
    # Set values below the threshold to zero
    result_matrix = np.where(matrix < threshold, 0, matrix)

    return result_matrix

def nth_level_destroy(n,d):
    # Create the basis state corresponding to the n-th level
    k = ket([1 if i == n else 0 for i in range(0, d)])
    b = bra([1 if i == n-1 else 0 for i in range(0, d)])

    # Create the projector onto the n-th level
    projector = k*b

    return projector

def matrix_element(m,n,d): # dxd matrix with all zero but the m,n element, which is 1
    out=Qobj(scipy.sparse.csr_matrix(([1],([m],[n])),shape=(d,d)).toarray())
    return(out.to(core.data.CSR))
