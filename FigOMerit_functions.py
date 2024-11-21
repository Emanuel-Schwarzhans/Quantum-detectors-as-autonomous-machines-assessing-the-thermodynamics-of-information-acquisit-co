import numpy as np
from functools import reduce
from qutip import *
from Functions import *
from Hamiltonians import *


def get_dynamics(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,g_sl,g_ml,d_l,t_f,t_steps):
    # Total Hamiltonian
    e_H=e_s+e_C
    H=H0_full(e_s,e_l,e_C,e_H,d_l)+HI_full(g_sl,g_ml,d_l)

    # Jump operators
    c_ops=c_ops_full(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,d_l)

    # Observable operators
    e_current_waste=Qobj(np.sum([-(c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[2][l+1]-c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[2][l])*c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[0][l].dag()*c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[0][l]+(c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[2][l+1]-c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[2][l])*c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[1][l].dag()*c_ops_ladder_list(rateB,Tb,e_s,e_l,d_l)[1][l] for l in range(d_l-1)]))
    # e_current_top=c_ops[7]-c_ops[6]
    e_current=(e_l*(d_l-2)+e_s)*(c_ops[7].dag()*c_ops[7]-c_ops[6].dag()*c_ops[6])
    e_machine_heat_dissip =e_C*(c_ops[1].dag()*c_ops[1]-c_ops[0].dag()*c_ops[0])
    e_machine_heat_draw =e_H*(-c_ops[3].dag()*c_ops[3]+c_ops[2].dag()*c_ops[2])
    e_ready=tensor(identity(2),identity(2),matrix_element(0,0,d_l),identity(2))
    e_ops=[e_current,e_ready,e_machine_heat_dissip,e_machine_heat_draw,e_current_waste]

    # [tensor(identity(2),identity(d_l),matrix_element(1,1,2)),tensor(identity(2),matrix_element(d_l-2,d_l-2,d_l),identity(2)),tensor(identity(2),Qobj(np.sum([matrix_element(n,n,d_l).full() for n in range(0,d_l-2)],axis=0)),identity(2)),tensor(matrix_element(1,1,2),identity(d_l),identity(2)),c_bath_mi.dag()*c_bath_mi]
    steady_prev_run=steadystate(H,c_ops)
    psi0 = tensor(ptrace(steady_prev_run,0),ptrace(steady_prev_run,1),ptrace(steady_prev_run,2),matrix_element(1,1,2))
    times = np.linspace(0., t_f, t_steps)
    return([mesolve(H,psi0, tlist=times,c_ops=c_ops, e_ops=e_ops),steady_prev_run,e_ops,c_ops,t_f,t_steps,rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,g_sl,g_ml,d_l])

def efficiency(result):
    steady_prev_run=result[1]
    t_f=result[4]
    t_steps=result[5]
    result_data=result[0]
    noise_floor_curr=expect(result[2][0],steady_prev_run)
    noise_floor_waste=expect(result[2][-1],steady_prev_run)
    return(np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)/(np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)+np.sum((result_data.expect[-1]-noise_floor_waste)*t_f/t_steps)))

def dark_counts(result):
    return(expect(result[2][0],result[1]))

def energy_out(result,e_l,e_s,d_l):
    steady_prev_run=result[1]
    t_f=result[4]
    t_steps=result[5]
    result_data=result[0]
    noise_floor_curr=expect(result[2][0],steady_prev_run)
    return(((d_l-2)*e_l+e_s)*np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps))


def entropy_production(result,TC):
    t_f=result[4]
    t_steps=result[5]
    return((np.sum((result[0].expect[2]-expect(result[2][2],result[1])+result[0].expect[4]-expect(result[2][4],result[1]))*t_f/t_steps))/TC)

def jitter(result): # Keep in mind that this is the variance (often FWHM is used, which I might want to consider as well, though its related)
    # result_data=result[0]
    # t_f=result[4]
    # t_steps=result[5]
    steady_prev_run=result[1]
    noise_floor_curr=expect(result[2][0],steady_prev_run)
    # normalize=np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)
    # return(np.array((result[0].expect[0]-noise_floor_curr)/normalize).var())
    return(np.array((result[0].expect[0]-noise_floor_curr)).var())

def get_parameterlist(result):
    outstring=["mesolve(H,psi0, tlist=times,c_ops=c_ops, e_ops=e_ops)","steady_prev_run","e_ops","c_ops","t_f","t_steps","rateM","rateB","rateD","TH","TC","Tb","Td","e_s","e_C","e_l","d_l"]
    return([[outstring[x],result[x]] for x in range(4,len(result))])

def get_all_figures_of_merit(result): #returns list of [efficiency, dark count rate, jitter, entropy production]
    TC=result[10]

    effic=efficiency(result)
    darc=dark_counts(result)
    jitt=jitter(result)
    ent=entropy_production(result,result[10])

    return([effic,darc,jitt,ent,result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18]])

def get_virtual_temp(result):
    E_C=result[14]
    E_H=result[13]+result[14]
    T_C=result[10]
    T_H=result[9]
    return((E_H-E_C)/(E_H/T_H-E_C/T_C))
