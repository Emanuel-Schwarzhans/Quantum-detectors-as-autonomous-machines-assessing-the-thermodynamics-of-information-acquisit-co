from Functions import *
from Hamiltonians import *
from FigOMerit_functions import *
from scipy.stats.qmc import *



def main():
    ## Define Parameters
    parameters = {
        # General parameters
        "d_l": 3,         # Ladder dimension
        "e_max": 1,       # Energy of most excited level of ladder (threshold energy)
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
        "first_gap_flag": False,  # Flag for sampling the first gap (consumes more computational resource)
        "epsilon": 10**(-6),        # Small number for deviation from maximum TV value, only used if Max_TV_flag is True
    }



    ## Define ranges for parameters

    rateD_range=[0.1,2]
    g_ml_range=[0.1,2]
    g_sl_range=[0.1,2]
    T_C_range=[0.05,2]
    T_V_range=[-20,-0]
    e_C_factor_range=[0.1,2]
    rateM_range=[0.1,2]
    parameters["first_gap_flag"]=True


    filename="Dante_Scatter_plots_v1.csv"

    parameters["Max_TV_flag"]=False
    parameters["first_gap_flag"]=True

    ## Generate dataset

    loop_dataset_generation(rateD_range,
                            g_ml_range,
                            g_sl_range,
                            T_C_range,
                            T_V_range,
                            e_C_factor_range,rateM_range,
                            parameters,
                            filename,
                            N_loops=1,
                            loop_size=10,
                            error_cap=100)

if __name__ == "__main__":
    main()
