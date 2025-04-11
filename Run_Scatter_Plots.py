import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from Functions import *
from Hamiltonians import *
from FigOMerit_functions import *
from scipy.stats.qmc import *
import os


def main():
    rateD_range=[0.1,2]
    g_ml_range=[0.1,2]
    g_sl_range=[0.1,2]
    T_C_range=[0.05,2]
    T_V_range=[-10,-0]
    e_C_factor_range=[0.01,2]
    rateM_range=[0.1,2]
    parameters["first_gap_flag"]=True


    filename="FOM_Dinv_exact_TC_TV_g_sl_rateD_eCfactor_rateM_WithFirstGap_v1"

    parameters["Max_TV_flag"]=False
    parameters["first_gap_flag"]=True

    loop_dataset_generation(rateD_range,
                            g_ml_range,
                            g_sl_range,
                            T_C_range,
                            T_V_range,
                            e_C_factor_range,rateM_range,parameters,
                            "testfile.csv",
                            N_loops=10,
                            loop_size=10,
                            error_cap=100)
