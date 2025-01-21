import numpy as np
from functools import reduce
from qutip import *
from Functions import *
from Hamiltonians import *
from scipy.stats.qmc import *
import os



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

    e_ops=[Qobj(c_ops[2*i+1].dag()*c_ops[2*i+1]-c_ops[2*i].dag()*c_ops[2*i])  for i in range(int(len(c_ops)/2))]
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
        k])                         #19


#######################
# In the following result should be given as the output of get_dynamics or get_dynamics_k
#######################
def efficiency(result):
    steady_prev_run=result[1]
    t_f=result[4]
    t_steps=result[5]
    result_data=result[0]
    noise_floor_curr=expect(result[2][0],steady_prev_run)
    # noise_floor_waste=expect(result[2][-1],steady_prev_run)
    # return(np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)/(np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)+np.sum((result_data.expect[-1]-noise_floor_waste)*t_f/t_steps)))
    return(np.trapz(np.array(result_data.expect[0]),dx=t_f/t_steps)-t_f*noise_floor_curr) # this is equalt to the total number of excess jumps in the detector channel

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
    # result_data=result[0]
    # t_f=result[4]
    # t_steps=result[5]
    steady_prev_run=result[1]
    noise_floor_curr=expect(result[2][0],steady_prev_run)
    # normalize=np.sum((result_data.expect[0]-noise_floor_curr)*t_f/t_steps)
    # return(np.array((result[0].expect[0]-noise_floor_curr)/normalize).var())
    return(np.array((result[0].expect[0]-noise_floor_curr)).var())

def get_parameterlist(result):
    outstring=["mesolve(H,psi0, tlist=times,c_ops=c_ops, e_ops=e_ops)","steady_prev_run","e_ops","c_ops","t_f","t_steps","rateM","rateB","rateD","TH","TC","Tb","Td","e_s","e_C","e_l","d_l","TV","k"]
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
    ent=entropy_production(result)
    ent_rate=entropy_steady_rate(result)
    return([effic,darc,jitt,ent,ent_rate,result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18],TV,result[19]])

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
    return((e_L-e_C)/(e_L/TV+e_C/TC))

def TV_from_TH_TC(TH,TC,e_C,e_L):
    e_H=e_L-e_C
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
        e_l=(-e_s+e_max)/(d_l-2)
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
        if (parameters["TH"] >=0 and parameters["TV"]<=0):
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
