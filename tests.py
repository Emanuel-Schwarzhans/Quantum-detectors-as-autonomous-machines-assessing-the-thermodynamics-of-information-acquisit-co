import pytest
import numpy as np
from functools import reduce
from qutip import *
from Functions import *
from Hamiltonians import *
from scipy.stats.qmc import *
import pandas as pd
from FigOMerit_functions import *

@pytest.fixture
def default_parameters():
    parameters = {
        # General parameters
        "d_l": 3,         # Ladder dimension
        "e_max": 1,       # Energy of most excited level of ladder (threshold energy)
        "e_s": 0.1,       # System (photon) energy
        "TH": 10,         # Hot bath temperature
        "TC": 0.1,        # Cold bath temperature

        # Interaction strengths
        "g_sl": 1,        # System-ladder interaction strength
        "g_ml": 1,        # Machine-ladder interaction strength

        # Energies
        "e_C": 1,       # Cold bath gap
        "e_C_factor": 0.3, # Factor for cold bath gap calculation
        "t_f": 3000,       # Final time
        "t_steps": 3000,   # Number of timesteps

        # Jump rates
        "rateM": 1,       # Dissipator rate machine-baths
        "rateD": 1,       # Dissipation rate detection channel
        "rateB": 1,     # Thermalizing rate ladder-bath
        "Tb": 0.1,        # Ladder bath temperature
        "Td": 0.1,        # Detection channel temperature

        # Constant k
        "k": 1,

        "Tb_indep_flag": False,  # Flag for independent ladder bath temperature,
        "Td_indep_flag": False,  # Flag for independent detection channel temperature
        "Td_eq_Tb_flag": False,  # Flag for equal ladder bath and detection channel temperature
        "Max_TV_flag": False,    # Flag for maximum setting TV to the maximum value given by TV>-TC e_L/e_C= -TC/e_C_factor
        "epsilon": 10**(-6),        # Small number for deviation from maximum TV value, only used if Max_TV_flag is True
        }
    parameters["e_l"]=(parameters["e_max"]-parameters["e_s"])/(parameters["d_l"]-2)
    parameters["e_C"]=parameters["e_l"]*parameters["e_C_factor"]
    parameters["TV"]=TV_from_TH_TC(parameters["TH"],parameters["TC"],parameters["e_C"],parameters["e_l"])
    return(parameters)

def test_get_current_super_op(default_parameters):
    # Randomly assign values to the parameters in the dictionary
    parameters=default_parameters
    L, DI, init, steady = get_L_Draz_Init_Steady(parameters)
    cops=get_c_ops(parameters)
    JD = -cops[0].dag()@cops[0]+cops[1].dag()@cops[1]
    JD_super= get_current_super_op(parameters)

    ident=qutip.identity([2,2,3,2])
    vec_ident=operator_to_vector(ident)
    assert(vec_ident.trans()@JD_super@operator_to_vector(steady)==(JD@steady).tr())

def test_overlaps_in_efficiency(default_parameters):
    parameters=default_parameters
    L, DI, init, steady = get_L_Draz_Init_Steady(parameters)
    LRes=eigensystem_LR(L)
    overlaps=overlaps_in_efficiency(parameters)
    overlaps_effic=-np.sum([overlaps[i]/(LRes[i][0]) for i in range(len(LRes)-1)])
    L_effic=L_efficiency(DI,init,get_e_ops(parameters),parameters)
    assert((np.real(overlaps_effic)-np.real(L_effic))<1e-8)


# def test_L_first_gap():
#     L=liouvillian(Qobj([[0,0],[0,1]]),c_ops=Qobj([[0,1],[0,0]]))
#     L_eig_FG=L.eigenenergies()[-2]-L.eigenenergies()[-2]
#     FG=L_first_gap
#     assert(np.abs(L_eig_FG-FG)<1e-8)


def test_reduce_list_to_real_values():
    list1=[1+0j, 2+3j, 2-3j, 0.5+0j, 0.5-1j,0.5+1j, 0.2-0j]
    list2=[1+0j, 2+3j, 2-2j]
    assert(reduce_list_to_real_values(list1) == [(1+0j), (4+0j),(0.5+0j), (1+0j), (0.2+0j)])
    with pytest.raises(ValueError):
        reduce_list_to_real_values(list2)
