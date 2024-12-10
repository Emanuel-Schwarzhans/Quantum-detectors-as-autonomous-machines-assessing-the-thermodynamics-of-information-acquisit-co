import numpy as np
from functools import reduce
from qutip import *
import numpy as np
from Functions import *

## Defining the system


def H0_full(e_s,e_l,e_c,e_h,d_l):
    H_l= tensor(identity(2),identity(2),Qobj(np.sum([(e_l*(n-1)+e_s)*matrix_element(n,n,d_l).full() for n in range(2,d_l)],axis=0)),identity(2)) #levels until d_l
    H_l+= tensor(identity(2),identity(2),e_s*matrix_element(1,1,d_l),identity(2))  # off resonant gap
    H_h= tensor(identity(2),(e_c+e_l)*matrix_element(1,1,2),identity(d_l),identity(2)) # system
    H_c= tensor(e_c*matrix_element(1,1,2),identity(2),identity(d_l),identity(2)) # system
    H_in= tensor(identity(2),identity(2),identity(d_l),e_s*matrix_element(1,1,2)) # system
    # H_out= tensor(identity(2),identity(2),identity(d_l),identity(2),(e_s+d_l*e_l)*matrix_element(1,1,2)) # system
    H0=H_l+H_h+H_c+H_in
    return(H0.to(core.data.CSR))

def H0_k_nosys(e_s,e_l,e_c,e_h,d_l,k):
    H_l= tensor(identity(2),identity(2),Qobj(np.sum([n*e_l*matrix_element(n,n,d_l).full() for n in range(0,k+1)],axis=0))) #levels until k
    H_l+= tensor(identity(2),identity(2),Qobj(np.sum([(e_l*(n-1)+e_s)*matrix_element(n,n,d_l).full() for n in range(k+1,d_l)],axis=0))) #levels until d_l-1
    H_h= tensor(identity(2),(e_c+e_l)*matrix_element(1,1,2),identity(d_l)) # hot bit
    H_c= tensor(e_c*matrix_element(1,1,2),identity(2),identity(d_l)) # cold bit
    H0=H_l+H_h+H_c
    return(H0.to(core.data.CSR))

def HI_full(g_sl,g_ml,d_l):
    H_al= g_sl*tensor(identity(2),identity(2),matrix_element(0,1,d_l),matrix_element(1,0,2)) # system-ladder interaction
    H_ml= g_ml*tensor(matrix_element(0,1,2),matrix_element(1,0,2),Qobj(np.sum([matrix_element(n-1,n,d_l).full() for n in range(2,d_l)],axis=0)),identity(2)) # ladder-bath interaction starting from first level below smaller gap, ie d_l-3
    # H_lb= g_ml*tensor(identity(2),identity(2),matrix_element(d_l-1,0,d_l),identity(2),matrix_element(0,1,2))
    HI_0=H_al+H_ml
    HI_0=HI_0+HI_0.dag()
    return(HI_0.to(core.data.CSR))

def HI_k_nosys(g_ml,d_l,k):
    H_ml= g_ml*tensor(matrix_element(0,1,2),matrix_element(1,0,2),
                      Qobj(np.sum([matrix_element(n,n+1,d_l).full() for n in range(0,k)],axis=0))) # ladder-bath interaction below k level
    H_ml+= g_ml*tensor(matrix_element(0,1,2),matrix_element(1,0,2),
                       Qobj(np.sum([matrix_element(n,n+1,d_l).full() for n in range(k+1,d_l-1)],axis=0))) # ladder-bath interaction above k level
    HI_0=HI_0+HI_0.dag()
    return(HI_0.to(core.data.CSR))

def c_ops_ladder_list(rate,Tb,e_s,e_l,d_l): #returns a list of all jump operators in the ladder plus [0], minus [1] and the energy level list [2]
    e_levels=np.append(np.array([0,e_s]),[e_l*(l)+e_s for l in range(1,d_l-1)]) # Make list of energy levels
    Rb_pl=[np.sqrt(rate*np.exp(-(e_levels[l+1]-e_levels[l])/Tb)) for l in range(d_l-1)] # Make up rates between gaps from energy levels
    Rb_mi=[np.sqrt(rate) for l in range(d_l-1)] # Down rates are given by detailed balance, setting the base-rate to rate
    c_ops_ladder_pl_list=[Rb_pl[l]*tensor(identity(2),identity(2),matrix_element(l+1,l,d_l),identity(2)) for l in range(d_l-1)]
    c_ops_ladder_mi_list=[Rb_mi[l]*tensor(identity(2),identity(2),matrix_element(l,l+1,d_l),identity(2)) for l in range(d_l-1)]
    return([c_ops_ladder_pl_list,c_ops_ladder_mi_list,e_levels])

def c_ops_ladder_list_k_nosys(rate,Tb,e_s,e_l,d_l,k): #returns a list of all jump operators in the ladder plus [0], minus [1] and the energy level list [2]
    e_levels=np.append(np.array([n*e_l for n in range(0,k+1)]),np.array([(e_l*(n-1)+e_s) for n in range(k+1,d_l)]))# Make list of energy levels
    Rb_pl=[np.sqrt(rate)*np.exp(-(e_levels[l+1]-e_levels[l])/Tb) for l in range(0,d_l-1)] # Make up rates between gaps from energy levels
    Rb_mi=[np.sqrt(rate) for l in range(0,d_l-1)] # Down rates are given by detailed balance, setting the base-rate to rate
    c_ops_ladder_pl_list=[Rb_pl[l]*tensor(identity(2),identity(2),matrix_element(l+1,l,d_l)) for l in range(0,d_l-1)]
    c_ops_ladder_mi_list=[Rb_mi[l]*tensor(identity(2),identity(2),matrix_element(l,l+1,d_l)) for l in range(0,d_l-1)]
    return([c_ops_ladder_pl_list,c_ops_ladder_mi_list,e_levels])

def c_ops_full(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_c,e_l,d_l): # d_l is the total dimension of the ladder (including the photon part)
    RH_pl=np.sqrt(rateM*np.exp(-1/TH*(e_c+e_l))/(1+np.exp(-1/TH*(e_c+e_l))))
    RH_mi=np.sqrt(rateM*1/(1+np.exp(-1/TH*(e_c+e_l))))
    RC_pl=np.sqrt(rateM*np.exp(-1/TC*e_c)/(1+np.exp(-1/TC*e_c)))
    RC_mi=np.sqrt(rateM*1/(1+np.exp(-1/TC*e_c)))
    c_c_p=RC_pl*tensor(matrix_element(1,0,2),identity(2),identity(d_l),identity(2))
    c_c_m=RC_mi*tensor(matrix_element(0,1,2),identity(2),identity(d_l),identity(2))
    c_h_p=RH_pl*tensor(identity(2),matrix_element(1,0,2),identity(d_l),identity(2))
    c_h_m=RH_mi*tensor(identity(2),matrix_element(0,1,2),identity(d_l),identity(2))
    c_b_p=Qobj(np.sum(c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[0]))
    c_b_m=Qobj(np.sum(c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[1]))
    c_out_curr_p=rateD*np.exp(-(e_l*(d_l-2)+e_s)/Td)*tensor(identity(2),identity(2),matrix_element(d_l-1,0,d_l),identity(2))
    c_out_curr_m=rateD*tensor(identity(2),identity(2),matrix_element(0,d_l-1,d_l),identity(2))
    # return([c_c_p,c_c_m,c_h_p,c_h_m,c_out_curr_p,c_out_curr_m])
    return([c_c_p,c_c_m,c_h_p,c_h_m,c_b_p,c_b_m,c_out_curr_p,c_out_curr_m])

def c_ops_k_nosys(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_c,e_l,d_l,k): # d_l is the total dimension of the ladder (including the photon part)
    RH_pl=np.sqrt(rateM*np.exp(-1/TH*(e_c+e_l))/(1+np.exp(-1/TH*(e_c+e_l))))
    RH_mi=np.sqrt(rateM*1/(1+np.exp(-1/TH*(e_c+e_l))))
    RC_pl=np.sqrt(rateM*np.exp(-1/TC*e_c)/(1+np.exp(-1/TC*e_c)))
    RC_mi=np.sqrt(rateM*1/(1+np.exp(-1/TC*e_c)))
    c_c_p=RC_pl*tensor(matrix_element(1,0,2),identity(2),identity(d_l))
    c_c_m=RC_mi*tensor(matrix_element(0,1,2),identity(2),identity(d_l))
    c_h_p=RH_pl*tensor(identity(2),matrix_element(1,0,2),identity(d_l))
    c_h_m=RH_mi*tensor(identity(2),matrix_element(0,1,2),identity(d_l))
    c_b_p=Qobj(np.sum(c_ops_ladder_list_k_nosys(rateB,Tb,e_s,e_l,d_l,k)[0]))
    c_b_m=Qobj(np.sum(c_ops_ladder_list_k_nosys(rateB,Tb,e_s,e_l,d_l,k)[1]))
    c_out_curr_p=rateD*np.exp(-(e_l*(d_l-2)+e_s)/Td)*tensor(identity(2),identity(2),matrix_element(d_l-1,0,d_l))
    c_out_curr_m=rateD*tensor(identity(2),identity(2),matrix_element(0,d_l-1,d_l))
    return([c_c_p,c_c_m,c_h_p,c_h_m,c_b_p,c_b_m,c_out_curr_p,c_out_curr_m])
