import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
from functools import reduce
from qutip import *
import qutip 
from matplotlib import cm
import statistics

import numpy as np
import re, ast


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

#----------------------------------------------

# def H_0(h,ii,L):
#     return(h/2*sig_z(ii,L))

# def H_int(J,gh,D,i,j,L):
#     return(J*((gh+1)/2 *sig_x(i,L)*sig_x(j,L)+(1-gh)/2*sig_y(i,L)*sig_y(j,L))+D *sig_z(i,L)*sig_z(j,L))

# def H_drive(O,i,L):
#     return(O*sig_x(i,L))

# def H_tot(J,h,gh,D,O,L):
#     H=0
#     for ii in range(0, L-1,1):
#         H+=H_int(J,gh,D,ii,ii+1,L) 
#     for jj in range(0, L,1):
#         H+=H_0(h,jj,L)

#     H+= H_drive(O,0,L)
#     return(H)

# def H_eff_sparse(J,h,gh,D,O,g,N,L):
#     H=Qobj(H_tot(J,h,gh,D,O,L).data.astype(np.complex128)) - 1j/2 *np.dot(L_plus(g,N,L).dag(), L_plus(g,N,L))-1j/2 *np.dot(L_minus(g,N,L).dag(),L_minus(g,N,L))
#     return(csr_matrix(H))


# def L_plus(g,N,L):
#     return(np.sqrt(g*N)*sig_p(L-1,L))

# def L_minus(g,N,L):
#     return(np.sqrt(g*(N+1))*sig_m(L-1,L))

def JJ(LList, NuList):
    result = np.zeros_like(np.kron(LList[0],LList[0]))
    for i in range(len(NuList)):
        result += NuList[i] * np.kron(LList[i].conj(), LList[i])
    return result

def Current(sst,JJj):
    N_sst=sst.shape[0]
    id=Vectorize_Mat(np.identity(int(N_sst)))
    return(np.real(np.dot(np.dot(id.reshape(1,-1),JJj),Vectorize_Mat(sst))[0][0]))



def Noise(Lbb, steadys, JJ2, JJ):
    N_steadys = steadys.shape[0]
    N_Lbb = Lbb.shape[0]
    N_JJ2 = JJ2.shape[0]
    N_JJ = JJ.shape[0]

    Id_sqrt_steadys = np.identity(int(N_steadys))
    Id_sqrt_Lbb = np.identity(int(np.sqrt(N_Lbb)))
    
    # Vectorize matrices
    vectorized_Id_sqrt_steadys = Vectorize_Mat(Id_sqrt_steadys)
    vectorized_Id_sqrt_Lbb = Vectorize_Mat(Id_sqrt_Lbb)
    vectorized_steadys = Vectorize_Mat(steadys)

    # Calculate the expression using NumPy operations
    expr1 = np.dot(np.dot(vectorized_Id_sqrt_steadys.reshape(1,-1), JJ2), vectorized_steadys)
    expr3 = vectorized_steadys * np.dot(np.dot(vectorized_Id_sqrt_steadys.reshape(1,-1), JJ), vectorized_steadys)

    # Construct matrices for linear solving
    A = np.append(Lbb, vectorized_Id_sqrt_steadys.reshape(1,-1), axis=0)
    B = np.append(JJ.dot(vectorized_steadys) - expr3, 0)

    # Solve the linear equation
    x = np.linalg.lstsq(A, B, rcond=None)[0]

    result = (expr1-2*np.dot(np.dot(vectorized_Id_sqrt_steadys.reshape(1,-1), JJ),x))[0][0].real

    return result

# def Entropy_Production(steady,JJj,NN,h):
#     if NN==0: 
#         print("diverging entropy production due to zero temperature bath")
#         return(-1)
#     return(np.abs(Current(steady,JJ([L_plus(g,N,L),L_minus(g,N,L)],[h,-h])))*np.log(1/NN+1))

def initial_state(L):
    return(basis(2**L,0))


# def make_trajectory_plottable(data,t_f): #also outputs the reshaped trajectories such that the span the whole x axis (stay constant if no jump occurs)
#    traj=[]
#    for i in range(0,len(data.col_which)):
#       #color=cmap(i)
#       col_which_0=np.concatenate([[0],np.cumsum(data.col_which[i]*2-1)])
#       col_which_0=np.concatenate([col_which_0,[col_which_0[-1]]])
#       col_times_0=np.concatenate([[0],data.col_times[i],[t_f]])
#       traj.append([col_times_0,col_which_0])
#       #plt.step(col_times_0,col_which_0,color=color)
#    return(traj)

def make_trajectory_plottable(data, t_f):
    traj = []
    for i in range(len(data.col_which)):
        # Calculate cumulative sums
        col_which_0 = [0]  # Initialize with 0
        for value in data.col_which[i]:
            col_which_0.append(col_which_0[-1] + (2 * value - 1))

        # Append the last element to ensure same length as col_times_0
        col_which_0.append(col_which_0[-1])

        # Convert col_times[i] to a list
        col_times_i = list(data.col_times[i])

        # Add times to span the whole x-axis
        col_times_0 = [0] + col_times_i + [t_f]

        traj.append([col_times_0, col_which_0])

    return traj

def montecarlo_by_parameter(J,h,gh,D,O,g,NN,L,t_f,steps,ntraj_in,target_tol=0.1,psi0=[],print_state=False):
   times = np.linspace(0.0, t_f, steps) # initialize time step array
   if psi0 == []: 
       psi0 = tensor([qutip.rand_ket(2) for _ in range(L)]) # intialize initial state randomly
   H = H_tot(J,h,gh,D,O,L) # intialize hamiltonian
   c_ops_in=[L_plus(g,NN,L), L_minus(g,NN,L)] #define jump operators
   data = mcsolve(H, psi0, times,e_ops=[sig_p(L-1,L).dag()*sig_p(L-1,L)],c_ops=c_ops_in, ntraj=ntraj_in,target_tol=target_tol,options={"store_final_state": print_state,"keep_runs_results": False,"progress_bar" :False,"map": "parallel","improved_sampling": True}) # run montecarlo method
   return(data)


def data_omega_curve(J,h,gh,D,O,g,NN,L,t_f=100,steps=1000,ntraj_in=500,target_tol=0.1,psi0=[],print_state=False):
    Data_Omega=montecarlo_by_parameter(J,h,gh,D,O,g,NN,L,t_f,steps,ntraj_in,target_tol=target_tol,psi0=psi0,print_state=print_state)
    # Data_Omega_all.append(Data_Omega)
    traj=make_trajectory_plottable(Data_Omega,t_f)
    vari=statistics.variance([sublist[1][-1] for sublist in traj])/t_f #the sublist thing selects only the last elements of the cumulative jump count, ie at the last timestep
    curr= statistics.mean([np.sum(sublist)/t_f for sublist in Data_Omega.col_which])
    dyn_act= statistics.mean([np.sum(sublist)/t_f for sublist in Data_Omega.col_which])
    # Data_statistics_all.append([O,curr,vari,dyn_act])
    av_exp=Data_Omega.average_expect
    st_exp=Data_Omega.std_exp
    return([O,curr,vari,dyn_act,av_exp,st_exp])
    # return(curr)

def nth_level_projector(n,d):
    # Create the basis state corresponding to the n-th level
    nth_level_state = basis(d, n)
    
    # Create the projector onto the n-th level
    projector = ket2dm(nth_level_state)
    
    return projector

def nth_level_create(n,d):
    # Create the basis state corresponding to the n-th level
    nth_level_state = basis(d, n)
    
    # Create the projector onto the n-th level
    projector = ket2dm(nth_level_state)
    
    return projector

def matrix_element(m,n,d):
    return(Qobj(scipy.sparse.csr_matrix(([1],([m],[n])),shape=(d,d)).toarray())) 

#--------------------Defining system----------------

def H_0(E_S,E_L,ladder_dim):
    H=0
    ladder=0
    for i in range(0,ladder_dim-2): # ladder
        ladder += i*E_L*nth_level_projector(i,ladder_dim)
    ladder += (ladder_dim-2)*E_L*nth_level_projector(ladder_dim-1,ladder_dim) 
    ladder += ((ladder_dim-2)*E_L-E_S)*nth_level_projector(ladder_dim-2,ladder_dim)
    H += tensor(qeye(2),ladder,qeye(2))
    H += tensor(E_S*nth_level_projector(1,2),qeye(ladder_dim),qeye(2)) # system

    H += tensor(qeye(2),qeye(ladder_dim),E_L*nth_level_projector(1,2)) # environment
    return(H)

def H_int(ladder_dim):
    H_SL=tensor(sigmam(),matrix_element(ladder_dim-1,ladder_dim-2,ladder_dim),qeye(2))+tensor(sigmap(),matrix_element(ladder_dim-2,ladder_dim-1,ladder_dim),qeye(2))
    H_LB=
print(H_0(1,2,3))
print(tensor(qeye(2),qeye(2)).ptrace(0))
