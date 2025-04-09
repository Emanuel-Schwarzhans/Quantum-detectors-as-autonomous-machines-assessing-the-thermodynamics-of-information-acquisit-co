import numpy as np
from functools import reduce
from qutip import *
from Functions import *
from Hamiltonians import *
from scipy.stats.qmc import *
import pandas as pd
import os


def get_dynamics_k_new(parameters):
    # Total Hamiltonian
    t_f = parameters["t_f"]
    t_steps = parameters["t_steps"]

    parameters["TH"]=TH_from_TV_TC(parameters["TV"],
                                        parameters["TC"],
                                        parameters["e_l"]*parameters["e_C_factor"],
                                        parameters["e_l"])


    H=H_k_params(parameters)
    # H=H0_k(e_s,e_l,e_C,d_l,k)+HI_k(g_sl,g_ml,d_l,k)

    # Jump operators
    # c_ops=c_ops_k(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,d_l,k)
    c_ops=get_c_ops(parameters)
    ### --- e_ops:######################################
    # index     0                   1                   2               3               4                   5
    # operator  detect current      ladder-bath-curr    cold-m curr     hot-m curr      ready-state pop     energy-curr ladder-bath
    e_ops=get_e_ops(parameters)
    ####################################################


    steady_prev_run=steadystate(H,c_ops)
    psi0 = tensor(ptrace(steady_prev_run,[0,1,2]),matrix_element(1,1,2))
    times = np.linspace(0., t_f, t_steps)

    return([
        mesolve(H, psi0,
                tlist=times,
                c_ops=c_ops,
                e_ops=e_ops),       #0
        steady_prev_run,            #1
        e_ops,                      #2
        c_ops,                      #3
        parameters["t_f"],          #4
        parameters["t_steps"],      #5
        parameters["rateM"],        #6
        parameters["rateB"],        #7
        parameters["rateD"],        #8
        parameters["TH"],           #9
        parameters["TC"],           #10
        parameters["Tb"],           #11
        parameters["Td"],           #12
        parameters["e_s"],          #13
        parameters["e_C"],          #14
        parameters["e_l"],          #15
        parameters["g_sl"],         #16
        parameters["g_ml"],         #17
        parameters["d_l"],          #18
        parameters["k"],            #19
        H])                         #20


def get_dynamics_k(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,g_sl,g_ml,d_l,t_f,t_steps,k):
    # Total Hamiltonian
    H=H0_k(e_s,e_l,e_C,d_l,k)+HI_k(g_sl,g_ml,d_l,k)

    # Jump operators
    c_ops=c_ops_k(rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C,e_l,d_l,k)


    ### --- e_ops:######################################
    # index     0                   1                   2               3               4                   5
    # operator  detect current      ladder-bath-curr    cold-m curr     hot-m curr      ready-state pop     energy-curr ladder-bath
    e_ready=tensor(identity(2),identity(2),matrix_element(k,k,d_l),identity(2))
    e_energy_curr_ladder_bath=-Qobj(np.sum([c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[2][i]*
                               (c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[0][i].dag()*c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[0][i]
                                -c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[1][i].dag()*c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[1][i])
                                for i in range(len(c_ops_ladder_list_k(rateB,Tb,e_s,e_l,d_l,k)[0]))]))

    e_ops=[Qobj(c_ops[2*i+1].dag()*c_ops[2*i+1]-c_ops[2*i].dag()*c_ops[2*i])  for i in range(int(len(c_ops)/2))] # all the currents
    e_ops.append(e_ready)
    e_ops.append(e_energy_curr_ladder_bath)
    ####################################################


    steady_prev_run=steadystate(H,c_ops)
    psi0 = tensor(ptrace(steady_prev_run,[0,1,2]),matrix_element(1,1,2))
    times = np.linspace(0., t_f, t_steps)

    return([
        mesolve(H,psi0,
                tlist=times,
                c_ops=c_ops,
                e_ops=e_ops),       #0
        steady_prev_run,            #1
        e_ops,                      #2
        c_ops,                      #3
        t_f,                        #4
        t_steps,                    #5
        rateM,                      #6
        rateB,                      #7
        rateD,                      #8
        TH,                         #9
        TC,                         #10
        Tb,                         #11
        Td,                         #12
        e_s,                        #13
        e_C,                        #14
        e_l,                        #15
        g_sl,                       #16
        g_ml,                       #17
        d_l,                        #18
        k,                          #19
        H])                         #20


#######################
# In the following result should be given as the output of get_dynamics or get_dynamics_k
#######################
# def efficiency(result):
#     steady_prev_run=result[1]
#     t_f=result[4]
#     t_steps=result[5]
#     result_data=result[0]
#     noise_floor_curr=expect(result[2][0],steady_prev_run)
#     # noise_floor_waste=expect(result[2][-1],steady_prev_run)
#     # return(np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)/(np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)+np.sum((result_data.expect[-1]-noise_floor_waste)*t_f/t_steps)))
#     return(np.trapz(np.array(result_data.expect[0]),dx=t_f/t_steps)-t_f*noise_floor_curr) # this is equalt to the total number of excess jumps in the detector channel


####get_e_ops returns exp operators####
def get_e_ops(parameters):
    c_ops=get_c_ops(parameters)
    # c_ops=c_ops_k(parameters["rateM"],
    #         parameters["rateB"],
    #         parameters["rateD"],
    #         parameters["TH"],
    #         parameters["TC"],
    #         parameters["Tb"],
    #         parameters["Td"],
    #         parameters["e_s"],
    #         parameters["e_C"],
    #         parameters["e_l"],
    #         parameters["d_l"],
    #         parameters["k"])
    ### --- e_ops:######################################
    # index     0                   1                   2               3               4                   5
    # operator  detect current      ladder-bath-curr    cold-m curr     hot-m curr      ready-state pop     energy-curr ladder-bath
    e_ready=Qobj(tensor(identity(2),identity(2),matrix_element(parameters["k"],parameters["k"],parameters["d_l"]),identity(2)))
    e_energy_curr_ladder_bath=-Qobj(np.sum([c_ops_ladder_list_k(parameters["rateB"],
                                                                parameters["Tb"],
                                                                parameters["e_s"],
                                                                parameters["e_l"],
                                                                parameters["d_l"],
                                                                parameters["k"])[2][i]*
                               (c_ops_ladder_list_k(parameters["rateB"],
                                                    parameters["Tb"],
                                                    parameters["e_s"],
                                                    parameters["e_l"],
                                                    parameters["d_l"],
                                                    parameters["k"])[0][i].dag()
                                *c_ops_ladder_list_k(parameters["rateB"],
                                                     parameters["Tb"],
                                                     parameters["e_s"],
                                                     parameters["e_l"],
                                                     parameters["d_l"],
                                                     parameters["k"])[0][i]
                                -c_ops_ladder_list_k(parameters["rateB"],
                                                     parameters["Tb"],
                                                     parameters["e_s"],
                                                     parameters["e_l"],
                                                     parameters["d_l"],
                                                     parameters["k"])[1][i].dag()
                                *c_ops_ladder_list_k(parameters["rateB"],
                                                     parameters["Tb"],
                                                     parameters["e_s"],
                                                     parameters["e_l"],
                                                     parameters["d_l"],
                                                     parameters["k"])[1][i])
                                for i in range(len(c_ops_ladder_list_k(parameters["rateB"],
                                                                       parameters["Tb"],
                                                                       parameters["e_s"],
                                                                       parameters["e_l"],
                                                                       parameters["d_l"],
                                                                       parameters["k"])[0]))]))

    e_ops=[Qobj(c_ops[2*i+1].dag()*c_ops[2*i+1]-c_ops[2*i].dag()*c_ops[2*i])  for i in range(int(len(c_ops)/2))] # all the currents
    e_ops.append(e_ready)
    e_ops.append(e_energy_curr_ladder_bath)
    return(e_ops)
    ####################################################


def L_efficiency(DrazInv,init_state,e_ops,parameters): #takes Lindbladian L,
    # Superoperator to be converted to supermatrix representation. If q_oper is type="oper",
    # then it is taken to act by conjugation, such that to_super(A) == sprepost(A, A.dag()).
    # init_state=operator_to_vector(tensor(ptrace(steady_prev_run,[0,1,2]),matrix_element(1,1,2))) # initialising in the conditional "photon is there" state
    e_ops_current=e_ops[0]
    # Lrho=vector_to_operator(pseudo_inverse(liouvillian(H,c_ops),method="spsolve")*init_state)
    Lrho=vector_to_operator(DrazInv@init_state)
    effic=-(e_ops_current@Lrho).tr()
    return(np.real(effic))


def L_entropy_production(DrazInv,init_state,e_ops,parameters):
    TC=parameters["TC"]
    if TC==0:
        return(np.inf)
    else:
        EC=parameters["e_l"]*parameters["e_C_factor"]
        Lrho=vector_to_operator(DrazInv*init_state)
        return(np.real(-1/TC*((e_ops[5]+EC*e_ops[2])*Lrho).tr()))

def L_entropy_steady_rate(DrazInv,init_state,e_ops,parameters):
    TC=parameters["TC"]
    if TC==0:
        return(np.inf)
    else:
        e_C=parameters["e_l"]*parameters["e_C_factor"]
        steady_prev_run=init_state
        return(expect(e_ops[2]*e_C+e_ops[5],steady_prev_run)/TC)

def L_jitter(DrazInv,init_state,e_ops,parameters):
    e_ops_current=e_ops[0]
    L3rho=vector_to_operator(DrazInv@DrazInv@DrazInv@init_state)
    L2rho=vector_to_operator(DrazInv@DrazInv@init_state)
    Lrho=vector_to_operator(DrazInv@init_state)
    normalize=(e_ops_current@Lrho).tr()

    expt2=(2*e_ops_current/normalize*L3rho).tr()
    expt=(e_ops_current/normalize*L2rho).tr()
    jitter=np.sqrt((expt2-expt**2))

    return(np.real(jitter))


def L_dark_counts(DrazInv,steady_state,e_ops,parameters):
    return(expect(steady_state,e_ops[0]))


def machine_efficiency(parameters):
    TC=parameters["TC"]
    TH=parameters["TH"]
    TV=parameters["TV"]
    f=parameters["e_C_factor"]
    if TH==np.inf and TC==0:
        return(1-1/(f+1)) # this is the efficiency of the machine in the limit of infite TH and 0 TC
    elif TH==np.inf:
        return(1+1/(TV/TC-1))
    elif TC==0:
        return((1-TC/TH)*(1-1/(f+1)))
    else:
        effic=(1-TC/TH)*(1+1/(TV/TC-1))
        return(effic)

def L_first_gap(L):
    eigenen=np.array(L.eigenenergies())
    return(np.real(eigenen[-1]-eigenen[-2]))



def efficiency(result):
    # Superoperator to be converted to supermatrix representation. If q_oper is type="oper", then it is taken to act by conjugation, such that to_super(A) == sprepost(A, A.dag()).
    steady_prev_run=result[1]
    init_state=operator_to_vector(tensor(ptrace(steady_prev_run,[0,1,2]),matrix_element(1,1,2))) # initialising in the conditional "photon is there" state
    result_data=result[0]
    H=result[20]
    c_ops=result[3]
    l=liouvillian(H,c_ops)
    e_ops_current=result[2][0]
    Lrho=vector_to_operator(pseudo_inverse(liouvillian(H,c_ops),method="spsolve")*init_state)
    effic=-(e_ops_current*Lrho).tr()
    return(effic)



def dark_counts(result):
    return(expect(result[2][0],result[1]))


def entropy_production(result):
    TC=result[10]
    e_C=result[14]
    t_f=result[4]
    t_steps=result[5]
    entropy_dynamics=(result[0].expect[5]+result[0].expect[2]*e_C)/TC
    entropyrate=entropy_steady_rate(result)
    return(np.trapz(np.array(entropy_dynamics),dx=t_f/t_steps)-t_f*entropyrate) # this is equalt to the total number of excess jumps in the detector channel

def entropy_steady_rate(result):
    TC=result[10]
    e_C=result[14]
    steady_prev_run=result[1]
    return(expect(result[2][2]*e_C+result[2][5],steady_prev_run)/TC)


def jitter(result): # Keep in mind that this is the variance (often FWHM is used, which I might want to consider as well, though its related)
    result_data=result[0]
    t_f=result[4]
    t_steps=result[5]
    steady_prev_run=result[1]
    noise_floor_curr=expect(result[2][0],steady_prev_run)
    normalize=np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)
    t=np.linspace(0,t_f,t_steps)
    expt=np.trapz(t*np.array((result[0].expect[0]-noise_floor_curr)/normalize),t)
    expt2=np.trapz(t**2*np.array((result[0].expect[0]-noise_floor_curr)/normalize),t)
    return(np.sqrt(expt2-expt**2))

def get_parameterlist(result):
    outstring=["mesolve(H,psi0, tlist=times,c_ops=c_ops, e_ops=e_ops)","steady_prev_run","e_ops","c_ops","t_f","t_steps","rateM","rateB","rateD","TH","TC","Tb","Td","e_s","e_C","e_l","d_l","TV","k"]
    return([[outstring[x],result[x]] for x in range(4,len(result))])

def get_all_figures_of_merit(result): #returns list of [efficiency, dark count rate, jitter, entropy production,...] input is
    TC=result[10]
    TH=result[9]
    e_C=result[14]
    e_l=result[15]
    TV=e_l/((e_l+e_C)/TH-e_C/TC)
    effic=efficiency(result)
    darc=dark_counts(result)
    jitt=jitter(result)
    ent=entropy_production(result)
    ent_rate=entropy_steady_rate(result)
    return([effic,darc,jitt,ent,ent_rate,result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],TV,result[19]])


def L_get_all_figures_of_merit(DrazInv,init_state,steady_state,e_ops,parameters,L=None): #returns list of [efficiency, dark count rate, jitter, entropy production,...] input is
    effic=L_efficiency(DrazInv,init_state,e_ops,parameters)
    darc=L_dark_counts(DrazInv,steady_state,e_ops,parameters)
    jitt=L_jitter(DrazInv,init_state,e_ops,parameters)
    ent=L_entropy_production(DrazInv,init_state,e_ops,parameters)
    ent_rate=L_entropy_steady_rate(DrazInv,steady_state,e_ops,parameters)
    if L != None:
        first_gap=L_first_gap(L)
    else:
        first_gap=None
    output={"efficiency":effic,
            "dark count rate":darc,
            "jitter":jitt,
            "entropy production":ent,
            "entropy rate":ent_rate,
            "machine_efficiency":machine_efficiency(parameters),
            "L first gap":first_gap
    }
    output.update(parameters)
    output_df=pd.DataFrame([output])
    # return([effic,darc,jitt,ent,ent_rate,result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],TV,result[19]])
    return(output_df)

def out_index(el_str): #Outputs a dictionary relating entries of get_all_figures_of_merit to its indices
    list=["efficiency",
         "dark count rate",
         "jitter",
         "entropy production",
         "entropy rate",
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
         "TV",
         "k"
    ]
    return(list.index(el_str))


def TH_from_TV_TC(TV,TC,e_C,e_L):
        # Handle division by zero using numpy where
    result = np.where(TC == 0, np.inf, (e_L + e_C) / (e_L / TV + e_C / TC))
    return result


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
        e_l=(-e_s+e_max)/(d_l-2)
        e_C=e_C_factor*e_l

        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)
        valid_samples = scaled_samples[T_H >= 0]

        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)
        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)

def generate_sample_set_d_TC_TV(d_range,T_C_range,T_V_range,e_s,e_C_factor,e_max,d_l,sample_size):
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
        e_l=(-e_s+e_max)/(d_l-2)
        e_C=e_C_factor*e_l

        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)
        valid_samples = scaled_samples[T_H >= 0]

        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)
        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)


def generate_sample_set_TC_MaxTV_g_ml_rateD(T_C_range,g_ml_range,rateD_range,parameters,sample_size):
    #Generate samples of variables TC, TV, g_ml, g_ld

    filtered_samples=[]

    while len(filtered_samples) < sample_size:
        sampler=LatinHypercube(3)
        samples=np.array(sampler.random(n=50))# 50 is just a number big enough to have a good set to filter from while not taking too long an overshooting too much


        #scale other variables
        scaled_samples = scale(samples, [T_C_range[0],g_ml_range[0],rateD_range[0]],
                               [T_C_range[1],g_ml_range[1],rateD_range[1]])

        #implementing sampling constraint for temperature
        T_C = scaled_samples[:, 0]
        T_V=-T_C/parameters["e_C_factor"]*(1+parameters["epsilon"])
        d_l=parameters["d_l"]
        e_l=(parameters["e_max"]-parameters["e_s"])/(d_l-2)
        e_C=parameters["e_C_factor"]*e_l

        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)
        valid_samples = scaled_samples[T_H >= T_C*(e_l+e_C)/e_C]
        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)

        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)


def generate_sample_set_TC_MaxTV_g_ml_g_sl_rateD_eCfactor_rateM(T_C_range,g_ml_range,g_sl_range,rateD_range,e_C_factor_range,rateM_range,parameters,sample_size):

    filtered_samples=[]

    while len(filtered_samples) < sample_size:
        sampler=LatinHypercube(6)
        samples=np.array(sampler.random(n=50))# 50 is just a number big enough to have a good set to filter from while not taking too long an overshooting too much


        #scale other variables
        scaled_samples = scale(samples, [T_C_range[0],g_ml_range[0],g_sl_range[0],rateD_range[0],e_C_factor_range[0],rateM_range[0]],
                               [T_C_range[1],g_ml_range[1],g_sl_range[1],rateD_range[1],e_C_factor_range[1],rateM_range[1]])

        #implementing sampling constraint for temperature
        T_C = scaled_samples[:, 0]
        e_C_factor=scaled_samples[:,4]
        T_V=-T_C/e_C_factor*(1+parameters["epsilon"])
        d_l=parameters["d_l"]
        e_l=(parameters["e_max"]-parameters["e_s"])/(d_l-2)
        e_C=e_C_factor*e_l
        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)

        valid_samples = scaled_samples[T_H >= T_C*(e_l+e_C)/e_C]
        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)

        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)

def generate_sample_set_TC_TV_g_ml_rateD(T_C_range,T_V_range,g_ml_range,rateD_range,parameters,sample_size):
    #Generate samples of variables TC, TV, g_ml, g_ld

    filtered_samples=[]

    while len(filtered_samples) < sample_size:
        sampler=LatinHypercube(4)
        samples=np.array(sampler.random(n=50))# 50 is just a number big enough to have a good set to filter from while not taking too long an overshooting too much


        #scale other variables
        scaled_samples = scale(samples, [T_C_range[0],T_V_range[0],g_ml_range[0],rateD_range[0]],
                               [T_C_range[1],T_V_range[1],g_ml_range[1],rateD_range[1]])

        #implementing sampling constraint for temperature
        T_C = scaled_samples[:, 0]
        T_V = scaled_samples[:, 1]
        d_l=parameters["d_l"]
        e_l=(parameters["e_max"]-parameters["e_s"])/(d_l-2)
        e_C=parameters["e_C_factor"]*e_l

        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)
        valid_samples = scaled_samples[T_H >= T_C*(e_l+e_C)/e_C]

        # Add valid samples to the filtered list
        filtered_samples.extend(valid_samples)
        if len(filtered_samples) > sample_size:
            filtered_samples = filtered_samples[:sample_size]
    return(filtered_samples)


def generate_sample_set_TC_TV_g_ml_g_sl_rateD_eCfactor_rateM(T_C_range,T_V_range,g_ml_range,g_sl_range,rateD_range,e_C_factor_range,rateM_range,parameters,sample_size):
    #Generate samples of variables TC, TV, g_ml, g_ld

    filtered_samples=[]

    while len(filtered_samples) < sample_size:
        sampler=LatinHypercube(7)
        samples=np.array(sampler.random(n=50))# 50 is just a number big enough to have a good set to filter from while not taking too long an overshooting too much

        #scale other variables
        scaled_samples = scale(samples, [T_C_range[0],T_V_range[0],g_ml_range[0],g_sl_range[0],rateD_range[0],e_C_factor_range[0],rateM_range[0]],
                               [T_C_range[1],T_V_range[1],g_ml_range[1],g_sl_range[1],rateD_range[1],e_C_factor_range[1],rateM_range[1]])

        #implementing sampling constraint for temperature
        T_C = scaled_samples[:, 0]
        T_V = scaled_samples[:, 1]
        e_C_factor=scaled_samples[:,4]
        d_l=parameters["d_l"]
        e_l=(parameters["e_max"]-parameters["e_s"])/(d_l-2)
        e_C=e_C_factor*e_l
        T_H=TH_from_TV_TC(T_V,T_C,e_C,e_l)

        valid_samples = scaled_samples[T_H >= T_C*(e_l+e_C)/e_C]

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

def generate_dataset_TC_TV_g_ml_g_sl_rateD_eCfactor_rateM(sample_set,filename_save_load,parameters):
    try:
        FOM_LHC_sampling = pd.read_csv(filename_save_load)  # Load existing dataset
    except FileNotFoundError:
        print("File does not exist, starting new one with tile: ", filename_save_load)
        FOM_LHC_sampling = pd.DataFrame()  # Start fresh if file doesn't exist

    for sample in sample_set:
        parameters["TC"],parameters["TV"],parameters["g_ml"],parameters["g_sl"],parameters["rateD"], parameters["e_C_factor"],parameters["rateM"] = sample  # Unpack parameters

        if parameters["Tb_indep_flag"]==False:
            parameters["Tb"]=parameters["TC"]

        if parameters["Td_indep_flag"]==False:
            parameters["Td"]=parameters["TC"]

        parameters["e_l"] = (parameters["e_max"] - parameters["e_s"]) / (parameters["d_l"] - 2)
        parameters["e_C"] = parameters["e_l"] * parameters["e_C_factor"]
        parameters["TH"] = TH_from_TV_TC(parameters["TV"], parameters["TC"], parameters["e_C"], parameters["e_l"])

        H=(
            H0_k(   parameters["e_s"],
                    parameters["e_l"],
                    parameters["e_C"],
                    parameters["d_l"],
                    parameters["k"])
            +HI_k(parameters["g_sl"],
                    parameters["g_ml"],
                    parameters["d_l"],
                    parameters["k"]))

        e_ops=get_e_ops(parameters)
        # Jump operators
        c_ops=c_ops_k(parameters["rateM"], parameters["rateB"], parameters["rateD"], parameters["TH"],
                parameters["TC"], parameters["Tb"], parameters["Td"], parameters["e_s"],
                parameters["e_C"], parameters["e_l"], parameters["d_l"], parameters["k"])

        l=liouvillian(H,c_ops)

        drinv=pseudo_inverse(l,method="spsolve")

        steady=steadystate(H,c_ops)
        init=operator_to_vector(tensor(ptrace(steady,[0,1,2]),matrix_element(1,1,2)))

        if parameters["first_gap_flag"]==True:
            FOM_vals = L_get_all_figures_of_merit(drinv,init,steady,e_ops,parameters,L=l)  # Evaluate function
        else:
            FOM_vals = L_get_all_figures_of_merit(drinv,init,steady,e_ops,parameters)  # Evaluate function
        FOM_LHC_sampling=pd.concat([FOM_LHC_sampling, FOM_vals])

    FOM_LHC_sampling.reset_index(drop=True, inplace=True)
    FOM_LHC_sampling.to_csv(filename_save_load,index=False)
    return(FOM_LHC_sampling)



def generate_dataset_TC_MaxTV_g_ml_rateD(sample_set,filename_save_load,parameters):
    try:
        FOM_LHC_sampling = pd.read_csv(filename_save_load)  # Load existing dataset
    except FileNotFoundError:
        print("File does not exist, starting new one with tile: ", filename_save_load)
        FOM_LHC_sampling = pd.DataFrame()  # Start fresh if file doesn't exist

    for sample in sample_set:
        parameters["TC"],parameters["g_ml"],parameters["rateD"] = sample  # Unpack parameters

        if parameters["Tb_indep_flag"]==False:
            parameters["Tb"]=parameters["TC"]

        if parameters["Td_indep_flag"]==False:
            parameters["Td"]=parameters["TC"]

        # Set TV to max given the flag is for it is raised
        if parameters["Max_TV_flag"]== True:
            parameters["TV"]=-parameters["TC"]/parameters["e_C_factor"]*(1+parameters["epsilon"])
        else:
            raise(ValueError("Max_TV_flag=False, but needs to be set True for this parameter sampling function!"))


        parameters["e_l"] = (parameters["e_max"] - parameters["e_s"]) / (parameters["d_l"] - 2)
        parameters["e_C"] = parameters["e_l"] * parameters["e_C_factor"]
        parameters["TH"] = TH_from_TV_TC(parameters["TV"], parameters["TC"], parameters["e_C"], parameters["e_l"])

        H=(
            H0_k(   parameters["e_s"],
                    parameters["e_l"],
                    parameters["e_C"],
                    parameters["d_l"],
                    parameters["k"])
            +HI_k(parameters["g_sl"],
                    parameters["g_ml"],
                    parameters["d_l"],
                    parameters["k"]))

        e_ops=get_e_ops(parameters)
        # Jump operators
        c_ops=c_ops_k(parameters["rateM"], parameters["rateB"], parameters["rateD"], parameters["TH"],
                parameters["TC"], parameters["Tb"], parameters["Td"], parameters["e_s"],
                parameters["e_C"], parameters["e_l"], parameters["d_l"], parameters["k"])
        l=liouvillian(H,c_ops)
        drinv=pseudo_inverse(l,method="spsolve")
        steady=steadystate(H,c_ops)
        init=operator_to_vector(tensor(ptrace(steady,[0,1,2]),matrix_element(1,1,2)))

        FOM_vals = L_get_all_figures_of_merit(drinv,init,steady,e_ops,parameters)  # Evaluate function
        FOM_LHC_sampling=pd.concat([FOM_LHC_sampling, FOM_vals])

    FOM_LHC_sampling.reset_index(drop=True, inplace=True)
    FOM_LHC_sampling.to_csv(filename_save_load,index=False)
    return(FOM_LHC_sampling)


def generate_dataset_TC_MaxTV_g_ml_g_sl_rateD_eCfactor_rateM(sample_set,filename_save_load,parameters):
    try:
        FOM_LHC_sampling = pd.read_csv(filename_save_load)  # Load existing dataset
    except FileNotFoundError:
        print("File does not exist, starting new one with tile: ", filename_save_load)
        FOM_LHC_sampling = pd.DataFrame()  # Start fresh if file doesn't exist

    for sample in sample_set:
        parameters["TC"],parameters["g_ml"],parameters["g_sl"],parameters["rateD"], parameters["e_C_factor"],parameters["rateM"] = sample  # Unpack parameters

        if parameters["Tb_indep_flag"]==False:
            parameters["Tb"]=parameters["TC"]

        if parameters["Td_indep_flag"]==False:
            parameters["Td"]=parameters["TC"]

        # Set TV to max given the flag is for it is raised
        if parameters["Max_TV_flag"]== True:
            parameters["TV"]=-parameters["TC"]/parameters["e_C_factor"]*(1+parameters["epsilon"])
        else:
            raise(ValueError("Max_TV_flag=False, but needs to be set True for this parameter sampling function!"))


        parameters["e_l"] = (parameters["e_max"] - parameters["e_s"]) / (parameters["d_l"] - 2)
        parameters["e_C"] = parameters["e_l"] * parameters["e_C_factor"]
        parameters["TH"] = TH_from_TV_TC(parameters["TV"], parameters["TC"], parameters["e_C"], parameters["e_l"])

        H=(
            H0_k(   parameters["e_s"],
                    parameters["e_l"],
                    parameters["e_C"],
                    parameters["d_l"],
                    parameters["k"])
            +HI_k(parameters["g_sl"],
                    parameters["g_ml"],
                    parameters["d_l"],
                    parameters["k"]))

        e_ops=get_e_ops(parameters)
        # Jump operators
        c_ops=c_ops_k(parameters["rateM"], parameters["rateB"], parameters["rateD"], parameters["TH"],
                parameters["TC"], parameters["Tb"], parameters["Td"], parameters["e_s"],
                parameters["e_C"], parameters["e_l"], parameters["d_l"], parameters["k"])
        l=liouvillian(H,c_ops)
        drinv=pseudo_inverse(l,method="spsolve")
        steady=steadystate(H,c_ops)
        init=operator_to_vector(tensor(ptrace(steady,[0,1,2]),matrix_element(1,1,2)))

        FOM_vals = L_get_all_figures_of_merit(drinv,init,steady,e_ops,parameters)  # Evaluate function
        FOM_LHC_sampling=pd.concat([FOM_LHC_sampling, FOM_vals])

    FOM_LHC_sampling.reset_index(drop=True, inplace=True)
    FOM_LHC_sampling.to_csv(filename_save_load,index=False)
    return(FOM_LHC_sampling)




def generate_dataset_d_TC_TV(sample_set,filename_save_load,rateM,rateB,rateD,e_s,e_C_factor,e_max,g_sl,g_ml,t_f,t_steps,k):
    FOM_LHC_sampling=safe_qload(filename_save_load)
    for sample in sample_set:
        d_l,TC,TV = sample  # Unpack parameters
        Tb=TC
        Td=TC
        d_l=int(d_l)
        e_l=(e_max-e_s)/(d_l-2)
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
        e_l=(e_max-e_s)/(d_l-2)
        TH=TH_from_TV_TC(TV,TC,e_l*e_C_factor,e_l)
        FOM_vals = get_all_figures_of_merit(get_dynamics_k(
            rateM,rateB,rateD,TH,TC,Tb,Td,e_s,e_C_factor*e_l,e_l,g_sl,g_ml,d_l,t_f,t_steps,k))  # Evaluate function
        FOM_LHC_sampling.append(FOM_vals)
    qsave(FOM_LHC_sampling,filename_save_load)
    return(FOM_LHC_sampling)




def get_FoM_vari_one_parameter(param_dict,param_str,param_range): # exception: e_l can not be used as parameter for change
    data=[]
    parameters=param_dict.copy()
    for param in param_range:
        parameters[param_str]=param
        if param_str=="e_s" or param_str=="e_max":
            parameters["e_l"]=(-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2)
        if param_str=="TV":
            parameters["TH"]=TH_from_TV_TC(parameters["TV"],parameters["TC"],parameters["e_C"],
                                        (-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2))
        if param_str=="TH" or param_str=="TC":
            parameters["TV"]=TV_from_TH_TC(parameters["TH"],parameters["TC"],parameters["e_l"]*parameters["e_C_factor"],parameters["e_l"])
        parameters[param_str]=param
        if (parameters["TH"] >=parameters["TC"]*(parameters["e_l"]+parameters["e_C_factor"]*parameters["e_l"])/(parameters["e_C_factor"]*parameters["e_l"]) and parameters["TV"]<=0):

            data.append(get_all_figures_of_merit(
            get_dynamics_k(
                parameters["rateM"],
                parameters["rateB"],
                parameters["rateD"],
                parameters["TH"],
                parameters["TC"],
                parameters["Tb"],
                parameters["Td"],
                parameters["e_s"],
                parameters["e_l"]*parameters["e_C_factor"],
                (-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2),
                parameters["g_sl"],
                parameters["g_ml"],
                parameters["d_l"],
                parameters["t_f"],
                parameters["t_steps"],
                parameters["k"]
                )
                ))
    return(data)


def Parameter_plot(param_dict, param_str, param_range):
    # Check if the parameter to be varied is a basic parameter
    if param_str not in ["TC", "TV", "g_ml", "rateD", "e_C_factor", "e_max", "e_s","Tb","Td"]:
        raise ValueError("param_str not basic parameter")

    data = pd.DataFrame()
    parameters = param_dict.copy()




    # Iterate over the range of the parameter to be varied
    for param in param_range:
        if parameters["Td_eq_Tb_flag"]== True and (parameters["Tb_indep_flag"]==False or parameters["Td_indep_flag"]==False):
            raise ValueError("Td=Tb flag raised but Tb or Td are not independent")
        elif parameters["Td_eq_Tb_flag"]== True:
            if param_str == "Tb":
                parameters["Td"] = param
            elif param_str == "Td":
                parameters["Tb"] = param
            elif parameters["Tb"]!=parameters["Td"]:
                raise ValueError("Td=Tb flag raised but non of them is the parameter and they are not equal, set them equal before!")
        else:
            # Ensure Tb and Td are set to TC if the independent flag is False
            if parameters["Tb_indep_flag"] == False:
                parameters["Tb"] = parameters["TC"]
            if parameters["Td_indep_flag"] == False:
                parameters["Td"] = parameters["TC"]

        # Set Tb and Td equal given that flag is raised, error if they are not the same and non is param_str



        parameters[param_str] = param

        # Set TV to max given the flag is for it is raised
        if (parameters["Max_TV_flag"]==True):
            parameters["TV"]=-parameters["TC"]/parameters["e_C_factor"]*(1+parameters["epsilon"])



        # Calculate e_l and e_C based on the current parameter values
        parameters["e_l"] = (parameters["e_max"] - parameters["e_s"]) / (parameters["d_l"] - 2)
        parameters["e_C"] = parameters["e_l"] * parameters["e_C_factor"]
        # Calculate TH based on the current parameter values
        parameters["TH"] = TH_from_TV_TC(parameters["TV"], parameters["TC"], parameters["e_C"], parameters["e_l"])
        # Get the Liouvillian, Drazin inverse, initial state, and steady state
        L, DrazInv, init_state, steady_state = get_L_Draz_Init_Steady(parameters)
        # Get the expectation operators
        e_ops = get_e_ops(parameters)
        # Check if the current parameter values are valid
        if (parameters["TH"] >= 0 and parameters["TV"] <= 0):
            # Concatenate the figures of merit for the current parameter values to the data DataFrame
            data = pd.concat([data, L_get_all_figures_of_merit(DrazInv, init_state, steady_state, e_ops, parameters)])

    return data


def L_get_FoM_vari_one_parameter(param_dict,param_str,param_range): # exception: e_l can not be used as parameter for change
    data=pd.DataFrame()
    parameters=param_dict.copy()

    parameters["e_l"]=(parameters["e_max"]-parameters["e_s"])/(parameters["d_l"]-2)
    parameters["e_C"]=parameters["e_l"]*parameters["e_C_factor"]
    if param_str != ("e_l" and "TV" and "TH" and "e_C_factor"):
       parameters["TH"]=TH_from_TV_TC(parameters["TV"],parameters["TC"],parameters["e_C"],(-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2))

    for param in param_range:
        parameters[param_str]=param
        if param_str=="e_C_factor":
            parameters["e_C"]=parameters["e_l"]*parameters["e_C_factor"]
            parameters["TH"]=TH_from_TV_TC(parameters["TV"],
                                           parameters["TC"],
                                           parameters["e_C"],
                                           (-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2))
        if param_str=="e_s" or param_str=="e_max":
            parameters["e_l"]=(-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2)

        if param_str=="TV" or param_str=="TC":
            parameters["TH"]=TH_from_TV_TC(parameters["TV"],parameters["TC"],parameters["e_C"],
                                        (-parameters["e_s"] + parameters["e_max"]) / (parameters["d_l"] - 2))

        if param_str=="TH":
            parameters["TV"]=TV_from_TH_TC(parameters["TH"],parameters["TC"],parameters["e_l"]*parameters["e_C_factor"],parameters["e_l"])

        parameters[param_str]=param
        if parameters["Tb_indep_flag"]==False:
            parameters["Tb"]=parameters["TC"]
        if parameters["Td_indep_flag"]==False:
            parameters["Td"]=parameters["TC"]

        if (parameters["TH"] >=0 and parameters["TV"]<=0 and parameters["TH"]>= parameters["TC"]):

            L,DrazInv,init_state,steady_state=get_L_Draz_Init_Steady(parameters)
            e_ops=get_e_ops(parameters)
            data=pd.concat([data, L_get_all_figures_of_merit(DrazInv,init_state,steady_state,e_ops,parameters)])
    return(data)


def get_L_Draz_Init_Steady(parameters):
# returns vectorized Liouvillian, Drazin inverse and initial state, and density matrix steady state (not vectorised),
# takes the dict of parameters
    parameters_copy=parameters.copy()
    if parameters_copy["Tb_indep_flag"]==False:
        parameters_copy["Tb"]=parameters_copy["TC"]
    if parameters_copy["Td_indep_flag"]==False:
        parameters_copy["Td"]=parameters_copy["TC"]


    parameters_copy["TH"]=TH_from_TV_TC(parameters_copy["TV"],
                                        parameters_copy["TC"],
                                        parameters_copy["e_l"]*parameters_copy["e_C_factor"],
                                        parameters_copy["e_l"])

    # H=(
    #     H0_k(   parameters_copy["e_s"],
    #             parameters_copy["e_l"],
    #             parameters_copy["e_C"],
    #             parameters_copy["d_l"],
    #             parameters_copy["k"])
    #     +HI_k(parameters_copy["g_sl"],
    #             parameters_copy["g_ml"],
    #             parameters_copy["d_l"],
    #             parameters_copy["k"]))

    H=H_k_params(parameters_copy)
    # Jump operators
    c_ops=get_c_ops(parameters_copy)

    l=liouvillian(H,c_ops)
    steady=steadystate(H,c_ops)
    drinv=pseudo_inverse(l,rhoss=steady,method="spsolve")
    init=operator_to_vector(tensor(ptrace(steady,[0,1,2]),matrix_element(1,1,2)))

    return([l,drinv,init,steady])


def get_parameter_range_of_dataset(dataset):
    rel_par_range={}
    for par in dataset.columns:
        maxpar=dataset[par].values.max()
        minpar=dataset[par].values.min()
        rel_par_range[par]=[minpar,maxpar]
    return(pd.DataFrame([rel_par_range]))

from qutip import Qobj
import numpy as np

def chop(matrix, threshold=1e-10):
    """
    Sets all elements of a matrix below a given threshold to 0.
    Works with both numpy arrays and Qobj.

    Parameters:
        matrix (numpy.ndarray or Qobj): The input matrix.
        threshold (float): The threshold below which elements are set to 0.

    Returns:
        numpy.ndarray or Qobj: The modified matrix with small elements set to 0.
    """
    if isinstance(matrix, Qobj):
        # If it's a Qobj, apply chop to its real and imaginary parts separately
        chopped_data = np.array(matrix.full())
        real_part = np.real(chopped_data)
        imag_part = np.imag(chopped_data)
        real_part[np.abs(real_part) < threshold] = 0
        imag_part[np.abs(imag_part) < threshold] = 0
        chopped_data = real_part + 1j * imag_part
        return Qobj(chopped_data, dims=matrix.dims)
    elif np.isscalar(matrix):
        # If it's a scalar (real or complex), return it as is
        if matrix < threshold:
            return 0
        else:
            return matrix
    elif np.iscomplexobj(matrix):
        # If it's a complex object, apply chop to both real and imaginary parts
        real_part = np.real(matrix)
        imag_part = np.imag(matrix)
        real_part[np.abs(real_part) < threshold] = 0
        imag_part[np.abs(imag_part) < threshold] = 0
        return real_part + 1j * imag_part


def find_largest_overlap_qutip(list1, list2):
    """
    Find the index of the vector in list2 that has the largest overlap with each vector in list1.

    Args:
        list1 (list of Qobj): First list of Qobj vectors.
        list2 (list of Qobj): Second list of Qobj vectors.

    Returns:
        list of tuples: Each tuple contains the index of the vector in list1 and the index of the vector in list2
                        with the largest overlap.
    """
    result = []
    for i, vec1 in enumerate(list1):
        max_overlap = -np.inf
        max_index = -1
        for j, vec2 in enumerate(list2):
            overlap = np.abs((vec1.trans() @ vec2))  # Compute the absolute value of the inner product
            if overlap > max_overlap:
                max_overlap = overlap
                max_index = j
        result.append((i,max_index, max_overlap))  # Append the indices of the vectors with the largest overlap
    return result

def eigensystem_LR(L):
    eigsys_R=np.array(L.eigenstates())
    eigsys_L=np.array(L.trans().eigenstates())
    eigensystem_LR_out=[]
# returns the eigenvalues and the left and right eigenvectors of the Liouvillian L and the eigenvalues
    overlap_list=find_largest_overlap_qutip(eigsys_L[1],eigsys_R[1])
    for ovl in overlap_list:
        eigst_L=Qobj(eigsys_L[1,ovl[0]])
        eigst_R=Qobj(eigsys_R[1,ovl[1]])
        # eigst_L=eigst_L/(eigst_L.trans()@eigst_R)
        eigst_R=eigst_R/(eigst_L.trans()@eigst_R)
        # print(eigst_L.trans()@eigst_R)
        eigval=eigsys_L[0][ovl[0]]
        eigensystem_LR_out.append([eigval,eigst_L,eigst_R])
    return eigensystem_LR_out


def get_current_super_op(parameters):
    cops=get_c_ops(parameters)
    JD_super= qutip.sprepost(cops[1],cops[1].dag())-qutip.sprepost(cops[0],cops[0].dag())

    return(JD_super)



def overlaps_in_efficiency(parameters):
    L, DI, init, steady = get_L_Draz_Init_Steady(parameters)
    cops=get_c_ops(parameters)
    JD_super =  get_current_super_op(parameters)
    LRes=eigensystem_LR(L)
    vec_id=operator_to_vector(identity([2,2,3,2]))
    overlaps=[]
    for i in range(len(LRes)):
        overlap=vec_id.trans()@JD_super@LRes[i][2] * LRes[i][1].trans()@init
        overlaps.append(overlap)
    return(overlaps)

def overlaps_eigvals_in_efficiency(parameters):
    L, DI, init, steady = get_L_Draz_Init_Steady(parameters)
    cops=get_c_ops(parameters)
    JD_super =  get_current_super_op(parameters)
    LRes=eigensystem_LR(L)
    vec_id=operator_to_vector(identity([2,2,3,2]))
    overlaps=[]
    for i in range(len(LRes)):
        overlap=vec_id.trans()@JD_super@LRes[i][2] * LRes[i][1].trans()@init
        overlaps.append([overlap,LRes[i][0]])
    return(overlaps)

def reduce_list_to_real_values(list, epsilon=1e-10):
    list_copy= list.copy()
    real_values = []
    for el in list_copy:
        if np.abs(np.imag(el)) < epsilon:
            real_values.append(el)
        else:
            index = next((j for j, element in enumerate(list_copy) if np.abs(el - np.conj(element)) < epsilon), None)
            if index is None:
                raise ValueError("This list cannot be cast into a list of real values")
            real_values.append(el + list_copy[index])
            list_copy.pop(index)  # remove that element to not count twice
    return real_values
