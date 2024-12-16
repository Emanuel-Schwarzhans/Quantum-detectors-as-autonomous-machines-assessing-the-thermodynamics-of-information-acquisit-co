import numpy as np
from functools import reduce
from qutip import *
from Functions import *
from Hamiltonians import *
from scipy.stats.qmc import *
import os


def get_dynamics(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,g_sl,g_ml,d_l,t_f,t_steps):
    # Total Hamiltonian
    e_H=e_s+e_C
    H=H0_full(e_s,e_l,e_c,d_l)+HI_full(g_sl,g_ml,d_l)

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

def get_dynamics_k(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,g_sl,g_ml,d_l,t_f,t_steps,k):
    # Total Hamiltonian
    e_H=e_s+e_C
    H=H0_k(e_s,e_l,e_C,d_l,k)+HI_k(g_sl,g_ml,d_l,k)

    # Jump operators
    c_ops=c_ops_k(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,d_l,k)

    # Observable operators
    e_current_waste=Qobj(np.sum([-(c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[2][l+1]
                                   -c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[2][l])*c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[0][l].dag()*c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[0][l]
                                   +(c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[2][l+1]
                                     -c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[2][l])*c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[1][l].dag()*c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[1][l]
                                     for l in range(d_l-1)]))
    # e_current_top=c_ops[7]-c_ops[6]
    e_current=(e_l*(d_l-2)+e_s)*(c_ops[7].dag()*c_ops[7]-c_ops[6].dag()*c_ops[6])
    e_machine_heat_dissip =e_C*(c_ops[1].dag()*c_ops[1]-c_ops[0].dag()*c_ops[0])
    e_machine_heat_draw =e_H*(-c_ops[3].dag()*c_ops[3]+c_ops[2].dag()*c_ops[2])
    e_ready=tensor(identity(2),identity(2),matrix_element(0,0,d_l),identity(2))
    e_ops=[e_current,e_ready,e_machine_heat_dissip,e_machine_heat_draw,e_current_waste]

    steady_prev_run=steadystate(H,c_ops)
    psi0 = tensor(ptrace(steady_prev_run,0),ptrace(steady_prev_run,1),ptrace(steady_prev_run,2),matrix_element(1,1,2))
    times = np.linspace(0., t_f, t_steps)
    return([
        mesolve(H,psi0, tlist=times,c_ops=c_ops, e_ops=e_ops),
        steady_prev_run,
        e_ops,
        c_ops,
        t_f,
        t_steps,
        rateM,
        rateB,
        rateD,
        TH,
        TC,
        Tb,
        Td,
        e_s,
        e_C,
        e_l,
        g_sl,
        g_ml,
        d_l])


#######################
# In the following result should be given as the output of get_dynamics or get_dynamics_k
#######################
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

def get_all_figures_of_merit(result): #returns list of [efficiency, dark count rate, jitter, entropy production,...] input is
    TC=result[10]
    TH=result[9]
    e_C=result[14]
    e_l=result[15]
    TV=e_l/((e_l-e_C)/TH-e_C/TC)
    effic=efficiency(result)
    darc=dark_counts(result)
    jitt=jitter(result)
    ent=entropy_production(result,result[10])

    return([effic,darc,jitt,ent,result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],TV])

def out_index(el_str): #Outputs a dictionary relating entries of get_all_figures_of_merit to its indices
    list=["efficiency",
         "dark count rate",
         "jitter",
         "entropy production",
         "t_f",
         "t_steps",
         "rateM",
         "rateB",
         "rateD",
         "TH",
         "TC",
         "Tb",
         "Td",
         "e_s",
         "e_C",
         "e_l",
         "g_sl",
         "g_ml",
         "d_l",
         "TV"
    ]
    return(list.index(el_str))


def TH_from_TV_TC(TV,TC,e_C,e_L):
    return((e_L-e_C)/(e_L/TV+e_C/TC))

def TV_from_TH_TC(TH,TC,e_C,e_L):
    e_H=e_L+e_C
    return(e_L/(e_H/TH-e_C/TC))

#Input is the output of get_all_figures_of_merit
#output is a list with the same structure but without the elements that have non-negative virtual temperature
def neg_virt_temp_filter(datalist):
    filtered_list = [[inner for inner in outer
                      if get_virtual_temp(inner[out_index("TH")],inner[out_index("TC")],inner[out_index("e_s")],inner[out_index("e_C")])<0]
                      for outer in datalist]
    return(filtered_list)


def generate_sample_set_d_TC_TV_RB(d_range,R_B_range,T_C_range,T_V_range,e_s,e_C_factor,e_max,sample_size):
    #Generate samples of variables

    filtered_samples=[]

    while len(filtered_samples) < sample_size:
        sampler=LatinHypercube(4)
        samples=np.array(sampler.random(n=50))
        #scale d and make d integers
        d_scaled_samples=np.floor(scale(samples,[d_range[0]],[d_range[1]+1])).astype(int)[:,0]

        #scale other variables
        RB_TH_TC_scaled_samples = scale(samples[:,1:], [R_B_range[0],T_C_range[0],T_V_range[0]],[R_B_range[1],T_C_range[1],T_V_range[1]])

        #re-combine dimension samples with other variable samples
        scaled_samples=np.column_stack((d_scaled_samples,RB_TH_TC_scaled_samples))

        #implementing sampling constraint for temperature
        T_C = scaled_samples[:, 2]
        T_V = scaled_samples[:, 3]
        d_l=scaled_samples[:,0]
        e_l=(e_s+e_max)/(d_l-2)
        e_C=e_C_factor*e_l

        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)
        valid_samples = scaled_samples[T_H >= 0]

        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)
        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)

def generate_sample_set_d_TC_TV(d_range,T_C_range,T_V_range,e_s,e_C_factor,e_max,sample_size):
    #Generate samples of variables
    sampler=LatinHypercube(3)
    filtered_samples=[]
    samples=np.array(sampler.random(n=50))

    while len(filtered_samples) < sample_size:
        sampler=LatinHypercube(3)
        samples=np.array(sampler.random(n=50))

        #scale d and make d integers
        d_scaled_samples=np.floor(scale(samples,[d_range[0]],[d_range[1]+1])).astype(int)[:,0]

        #scale other variables
        RB_TH_TC_scaled_samples = scale(samples[:,1:], [T_C_range[0],T_V_range[0]],[T_C_range[1],T_V_range[1]])

        #re-combine dimension samples with other variable samples
        scaled_samples=np.column_stack((d_scaled_samples,RB_TH_TC_scaled_samples))

        #implementing sampling constraint for temperature
        T_C = scaled_samples[:, 1]
        T_V = scaled_samples[:, 2]
        d_l=scaled_samples[:,0]
        e_l=(e_s+e_max)/(d_l-2)
        e_C=e_C_factor*e_l

        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)
        valid_samples = scaled_samples[T_H >= 0]

        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)
        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)


def safe_qload(filename):
    if os.path.exists(filename + '.qu'):
        return qload(filename)
    else:
        return []

def generate_dataset_d_TC_TV(sample_set,filename_save_load,rateM,rateB,rateD,e_s,e_C_factor,e_max,g_sl,g_ml,t_f,t_steps,k):
    FOM_LHC_sampling=safe_qload(filename_save_load)
    for sample in sample_set:
        d_l,TC,TV = sample  # Unpack parameters
        Tb=TC
        Td=TC
        d_l=int(d_l)
        e_l=(e_max+e_s)/(d_l-2)
        TH=TH_from_TV_TC(TV,TC,e_l*e_C_factor,e_l)
        FOM_vals = get_all_figures_of_merit(get_dynamics_k(
            rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C_factor*e_l,e_l,g_sl,g_ml,d_l,t_f,t_steps,k))  # Evaluate function
        FOM_LHC_sampling.append(FOM_vals)
    qsave(FOM_LHC_sampling,filename_save_load)
    return(FOM_LHC_sampling)


def generate_dataset_d_TC_TV_RB(sample_set,filename_save_load,rateM,rateD,e_s,e_C_factor,e_max,g_sl,g_ml,t_f,t_steps,k):
    FOM_LHC_sampling=safe_qload(filename_save_load)

    for sample in sample_set:
        d_l,rateB,TC,TV = sample  # Unpack parameters
        Tb=TC
        Td=TC
        d_l=int(d_l)
        e_l=(e_max+e_s)/(d_l-2)
        TH=TH_from_TV_TC(TV,TC,e_l*e_C_factor,e_l)
        FOM_vals = get_all_figures_of_merit(get_dynamics_k(
            rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C_factor*e_l,e_l,g_sl,g_ml,d_l,t_f,t_steps,k))  # Evaluate function
        FOM_LHC_sampling.append(FOM_vals)
    qsave(FOM_LHC_sampling,filename_save_load)
    return(FOM_LHC_sampling)
