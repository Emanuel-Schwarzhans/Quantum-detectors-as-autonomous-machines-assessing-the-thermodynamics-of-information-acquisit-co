from Functions import *
from Hamiltonians import *
from FigOMerit_functions import *
from scipy.stats.qmc import *
import sys


def main():
    try:
        e_max = float(sys.argv[1])
    except ValueError:
        print("That's not a valid real number.")
    ## Define Parameters
    parameters = {
        # General parameters
        "d_l": 3,         # Ladder dimension
        "e_max": e_max,       # Energy of most excited level of ladder (threshold energy)
        "e_s": 0.1,       # System (photon) energy

        # Jump rates
        "rateB": 0.7,     # Thermalizing rate ladder-bath

        # protected gap position
        "k": 1,

        # Flags
        "Tb_indep_flag": False,  # Flag for independent ladder bath temperature,
        "Td_indep_flag": False,  # Flag for independent detection channel temperature
        "Td_eq_Tb_flag": False,  # Flag for equal ladder bath and detection channel temperature
        "Max_TV_flag": False,    # Flag for maximum setting TV to the maximum value given by TV>-TC e_L/e_C= -TC/e_C_factor
        "first_gap_flag": True,  # Flag for sampling the first gap (consumes more computational resource)
        "epsilon": 10**(-6),        # Small number for deviation from maximum TV value, only used if Max_TV_flag is True
        "noise_flag": False,  # Flag for calculating steady-sate noise with L_get_all_figures_of_merit function
    }



    ## Define ranges for parameters

    rateD_range=[0.1,1]
    g_ml_range=[0.1,1]
    g_sl_range=[0.1,1]
    T_C_range=[0.2,1]
    T_V_range=[-30,-0]
    e_C_factor_range=[0.1,1]
    rateM_range=[0.1,1]


    # filename="./Output/Dante_Scatter_plots_v2.csv"
    filename="test3.csv"

    parameters["Max_TV_flag"]=False
    parameters["first_gap_flag"]=True
    parameters["noise_flag"]=True

    parameters["scew_flag_TC"]=True
    parameters["scew_factor_TC"]=2

    ## Generate dataset

    loop_dataset_generation(rateD_range,
                            g_ml_range,
                            g_sl_range,
                            T_C_range,
                            T_V_range,
                            e_C_factor_range,rateM_range,
                            parameters,
                            filename,
                            N_loops=10,
                            loop_size=10,
                            error_cap=100)

if __name__ == "__main__":
    main()
